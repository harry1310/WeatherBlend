using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using WeatherBlend.Commands;
using WeatherBlend.Config;
using WeatherBlend.Site;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Regression for the recurring "no tiles on the overview after a Sunday
/// retrain" bug — most recently Membury on 2026-06-14, and the same class on
/// 2026-05-26.
///
/// The shape: a retrain mints a new champion version, but the predict step
/// that produces its forecasts hasn't run yet (Membury retrained AFTER the
/// 16:20 predict cycle that day). So the only temperature predictions on disk
/// belong to the PREVIOUS champion — same phase (2b), different version. The
/// overview's <see cref="ChampionMatcher"/> matches rows to the champion by
/// PHASE precisely so the still-valid previous-champion rows keep rendering
/// across this gap. That fallback needs BOTH the row's phase AND the champion's
/// phase to be in the phase map. <see cref="ModelMetadataRepository.GetPhaseByVersion"/>
/// only resolves the versions it's handed, and the renderer used to hand it
/// only versions that had predictions — so the brand-new champion (zero
/// predictions) was absent, its phase unknown, the fallback silently failed,
/// and the whole tile grid blanked.
///
/// <see cref="RenderSiteCommand.WithChampionVersions"/> closes the gap by
/// feeding the current champion versions into the phase map regardless of
/// whether they have predictions yet; their phase resolves off the on-disk
/// bundle (synced even with no predictions). These tests pin that end to end.
/// </summary>
public sealed class PhaseMapChampionFallbackTests : IDisposable
{
    private readonly string _root;
    private readonly string _models;
    private readonly ModelMetadataRepository _repo;

    // The 2026-06-14 Membury versions: previous champion has predictions,
    // today's freshly-minted champion does not. Both phase 2b (suffixless).
    private const string Station = "membury_devon";
    private const string PrevChampion = "v2026-06-07_143856";   // has predictions
    private const string NewChampion = "v2026-06-14_151908";    // minted today, no predictions yet

    public PhaseMapChampionFallbackTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-phasemap-tests-" + Guid.NewGuid().ToString("N"));
        _models = Path.Combine(_root, "models");
        Directory.CreateDirectory(_models);

        // Both bundles on disk with phase 2b — the new champion's bundle is
        // synced by the retrain even though it has no predictions yet.
        AddTemperatureBundle(PrevChampion, "2b");
        AddTemperatureBundle(NewChampion, "2b");

        _repo = new ModelMetadataRepository(
            NullLogger<ModelMetadataRepository>.Instance,
            new AppConfig { Storage = new StorageConfig { ModelsPath = _models } });
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private void AddTemperatureBundle(string version, string phase)
    {
        var dir = Path.Combine(_models, "temperature", Station, version);
        Directory.CreateDirectory(dir);
        ModelArtifact.SaveTrainingMetadata(dir, new ModelArtifact.TrainingMetadata
        {
            Version = version,
            Target = "temperature",
            Phase = phase,
            LocationName = Station,
        });
    }

    [Fact]
    public void WithChampionVersions_includes_a_champion_that_has_no_predictions()
    {
        // Predictions reference only the PREVIOUS champion.
        var predictionKeys = new[] { (Station, PrevChampion) };
        var champByStation = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [Station] = NewChampion,
        };

        var keys = RenderSiteCommand
            .WithChampionVersions(predictionKeys, champByStation)
            .ToList();

        keys.Should().Contain((Station, PrevChampion));
        keys.Should().Contain((Station, NewChampion),
            "the current champion must enter the phase map even before it has any predictions");
    }

    [Fact]
    public void GetPhaseByVersion_resolves_the_predictions_only_champion_phase_via_the_union()
    {
        var predictionKeys = new[] { (Station, PrevChampion) };
        var champByStation = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [Station] = NewChampion,
        };

        var tempKeys = RenderSiteCommand.WithChampionVersions(predictionKeys, champByStation);
        var phaseByVersion = _repo.GetPhaseByVersion(
            tempKeys,
            Enumerable.Empty<(string Station, string Version)>(),
            Enumerable.Empty<(string Station, int WindowHours, string Version)>(),
            Enumerable.Empty<(string Station, string Version)>());

        phaseByVersion.Should().ContainKey(PrevChampion).WhoseValue.Should().Be("2b");
        phaseByVersion.Should().ContainKey(NewChampion).WhoseValue.Should().Be("2b",
            "the champion's phase is read off its on-disk bundle even with zero predictions");
    }

    [Fact]
    public void OverviewMatcher_renders_previous_champion_rows_when_new_champion_has_no_predictions()
    {
        var predictionKeys = new[] { (Station, PrevChampion) };
        var champByStation = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [Station] = NewChampion,
        };

        var tempKeys = RenderSiteCommand.WithChampionVersions(predictionKeys, champByStation);
        var phaseByVersion = _repo.GetPhaseByVersion(
            tempKeys,
            Enumerable.Empty<(string Station, string Version)>(),
            Enumerable.Empty<(string Station, int WindowHours, string Version)>(),
            Enumerable.Empty<(string Station, string Version)>());

        // The overview filter: champion = the new version.
        var matcher = new ChampionMatcher(NewChampion, phaseByVersion);

        matcher.MatchesChampionPhase(PrevChampion)
            .Should().BeTrue("the previous champion shares phase 2b with the new one, so its still-valid rows belong on the tile grid");
    }

    [Fact]
    public void OverviewMatcher_blanks_without_the_union_documenting_the_bug()
    {
        // Pre-fix behaviour: phase map built from PREDICTIONS ONLY, so the
        // brand-new champion (no predictions) never enters it.
        var predictionsOnly = new[] { (Station, PrevChampion) };
        var phaseByVersion = _repo.GetPhaseByVersion(
            predictionsOnly,
            Enumerable.Empty<(string Station, string Version)>(),
            Enumerable.Empty<(string Station, int WindowHours, string Version)>(),
            Enumerable.Empty<(string Station, string Version)>());

        phaseByVersion.Should().NotContainKey(NewChampion);

        var matcher = new ChampionMatcher(NewChampion, phaseByVersion);

        // The champion's phase is unknown, so the phase fallback can't fire and
        // the previous champion's rows are dropped — this is the blank overview.
        matcher.MatchesChampionPhase(PrevChampion)
            .Should().BeFalse("without the champion in the phase map the fallback starves — the regression the union fixes");
    }
}
