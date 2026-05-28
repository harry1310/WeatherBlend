using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train.Element;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// End-to-end smoke for the three temperature phases served by
/// <see cref="TrainCommand"/>/<see cref="TempPredictCommand"/>:
/// 2b (lean champion), 2c (rich challenger) and 2d (exact-runtime,
/// Bonehill-only). 2d uses a different forecast tree (raw S3 exact
/// cycles, RunTimeSource='exact') and is covered in a separate test.
/// </summary>
public class TempPredictSmokeTests
{
    private readonly ITestOutputHelper _output;

    public TempPredictSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private async Task RunTempPhaseSmoke(
        string phaseTag,
        string featureSet,
        string locationName,
        string stationFriendly,
        string bundleSuffix)
    {
        var trainStart = new DateTime(2026, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        const int trainDays = 30;
        var predictAnchor = trainStart.AddDays(trainDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ($"smoke-{locationName}", stationFriendly) });

        // Temperature trains against ERA5 (gapless truth); no rainfall
        // needed unless the feature set joins precip persistence.
        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, trainStart, trainDays,
            runTimeSource: "offset_day");
        await SmokeFixtures.WriteEra5TruthAsync(
            scope.Era5Path, locationName, trainStart, trainDays);
        // Predict-side: 'reported' rows past the anchor.
        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor.AddHours(1), nDays: 6,
            runTimeSource: "reported");

        // TrainCommand depends on DryWindow + Element + Precip
        // commands (it's the cross-target dispatcher). For target=
        // temperature the temp dispatch arm never touches the others,
        // but the constructor is non-nullable, so wire them up with
        // their minimal deps anyway.
        var metadata = new ModelMetadataRepository(
            new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
        var precipConformal = new PrecipConformalFitCommand(
            new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
        var precip = new PrecipTrainCommand(
            new XunitLogger<PrecipTrainCommand>(_output), scope.Config, precipConformal);
        var dryWindowConformal = new DryWindowConformalFitCommand(
            new XunitLogger<DryWindowConformalFitCommand>(_output), scope.Config, metadata);
        var dryWindow = new DryWindowTrainCommand(
            new XunitLogger<DryWindowTrainCommand>(_output), scope.Config, metadata, dryWindowConformal);
        var element = new ElementTrainCommand(
            new XunitLogger<ElementTrainCommand>(_output),
            Array.Empty<IElementBlender>());

        var trainCmd = new TrainCommand(
            new XunitLogger<TrainCommand>(_output),
            scope.Config, dryWindow, element, precip);

        // TrainCommand dispatches on target + feature-set; lead "all"
        // runs every default lead.
        var trainRc = await trainCmd.RunAsync(
            target: "temperature",
            lead: "all",
            station: null,
            window: null,
            featureSet: featureSet,
            ct: default);
        trainRc.Should().Be(0, $"Phase {phaseTag} train should succeed on the smoke fixture");

        // Temperature bundle dir: data/models/temperature/{location}/{version}/
        var locationDir = Path.Combine(scope.ModelsPath, "temperature", locationName);
        Directory.Exists(locationDir).Should().BeTrue();
        var bundleName = Directory.EnumerateDirectories(locationDir)
            .Select(d => Path.GetFileName(d)!)
            .Where(n => bundleSuffix.Length == 0 ? !n.Contains("_phase") : n.EndsWith(bundleSuffix))
            .OrderBy(v => v, StringComparer.Ordinal)
            .LastOrDefault();
        bundleName.Should().NotBeNull($"Phase {phaseTag} should mint a bundle with suffix '{bundleSuffix}'");
        var bundleDir = Path.Combine(locationDir, bundleName!);
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty(
            $"Phase {phaseTag} bundle must contain LightGBM lead_*.zip artefacts");
        File.Exists(Path.Combine(bundleDir, "training_metadata.json")).Should().BeTrue();

        var predictCmd = new TempPredictCommand(
            new XunitLogger<TempPredictCommand>(_output), scope.Config);
        var predictRc = await predictCmd.RunAsync(
            target: "temperature",
            modelVersion: "current",
            forDate: DateOnly.FromDateTime(predictAnchor),
            locationOverride: null,
            ct: default);
        predictRc.Should().Be(0, $"Phase {phaseTag} predict should succeed against the freshly-minted bundle");

        var predParquet = Path.Combine(
            scope.PredictionsPath, "temperature", locationName,
            $"model_version={bundleName}",
            $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
        File.Exists(predParquet).Should().BeTrue($"predict_{phaseTag} should emit {predParquet}");

        var rows = await ParquetSerializer.DeserializeAsync<TempPredictionRow>(predParquet);
        rows.Should().NotBeEmpty();
        rows.Should().AllSatisfy(r =>
        {
            r.LocationName.Should().Be(locationName);
            r.ModelVersion.Should().Be(bundleName);
        });
    }

    // -----------------------------------------------------------------
    // Phase 2b — lean LightGBM temperature blender (champion)
    // -----------------------------------------------------------------

    [Fact]
    public Task Phase2b_train_then_predict_writes_non_empty_predictions_parquet()
        => RunTempPhaseSmoke(
            phaseTag: "2b",
            featureSet: "lean",
            locationName: "bonehill_rocks",
            stationFriendly: "Bellever Dartmoor",
            bundleSuffix: "");

    // -----------------------------------------------------------------
    // Phase 2c — rich LightGBM temperature blender (challenger)
    // -----------------------------------------------------------------

    [Fact]
    public Task Phase2c_train_then_predict_writes_non_empty_predictions_parquet()
        => RunTempPhaseSmoke(
            phaseTag: "2c",
            featureSet: "rich",
            locationName: "bonehill_rocks",
            stationFriendly: "Bellever Dartmoor",
            bundleSuffix: "_phase2c");

    // -----------------------------------------------------------------
    // Phase 2d — exact-runtime temperature, Bonehill-only
    // -----------------------------------------------------------------

    /// <summary>
    /// Exact-runtime smoke. Phase 2d reads <c>RunTimeSource='exact'</c>
    /// rows from raw S3 archives — different model ids (gfs_ncep,
    /// ecmwf_ifs_oper, ecmwf_aifs_oper, met_office_global, gefs_ncep_mean)
    /// and a different ValidTime grid (hour ∈ {0,6,12,18}). T2 tier
    /// default also runs lead {12, 24}.
    /// </summary>
    [Fact]
    public async Task Phase2d_exact_runtime_train_then_predict_writes_non_empty_predictions_parquet()
    {
        const string locationName = "bonehill_rocks";
        const string stationFriendly = "Bellever Dartmoor";
        var trainStart = new DateTime(2025, 8, 1, 0, 0, 0, DateTimeKind.Utc);
        // T2 start date is 2024-02-29; the smoke fixture spans well past it.
        // RunPhase2dAsync needs ≥500 rows per lead and the exact-runtime
        // valid grid is {0, 6, 12, 18} → 4 stamps/day. 150 days gives 600
        // rows per lead with comfortable headroom over the threshold.
        const int trainDays = 150;
        var predictAnchor = trainStart.AddDays(trainDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ($"smoke-{locationName}", stationFriendly) });

        // Train-side exact-runtime tree.
        await SmokeFixtures.WriteExactRuntimeForecastTreeAsync(
            scope.ForecastsPath, locationName, trainStart, trainDays);
        await SmokeFixtures.WriteEra5TruthAsync(
            scope.Era5Path, locationName, trainStart, trainDays);

        // Predict-side: exact-runtime live forecasts past anchor for the
        // valid grid the predict command reads (lead 12/24, valid hours
        // {0,6,12,18}).
        await SmokeFixtures.WriteExactRuntimeForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor, nDays: 5);

        var metadata = new ModelMetadataRepository(
            new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
        var precipConformal = new PrecipConformalFitCommand(
            new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
        var precip = new PrecipTrainCommand(
            new XunitLogger<PrecipTrainCommand>(_output), scope.Config, precipConformal);
        var dryWindowConformal = new DryWindowConformalFitCommand(
            new XunitLogger<DryWindowConformalFitCommand>(_output), scope.Config, metadata);
        var dryWindow = new DryWindowTrainCommand(
            new XunitLogger<DryWindowTrainCommand>(_output), scope.Config, metadata, dryWindowConformal);
        var element = new ElementTrainCommand(
            new XunitLogger<ElementTrainCommand>(_output),
            Array.Empty<IElementBlender>());

        var trainCmd = new TrainCommand(
            new XunitLogger<TrainCommand>(_output),
            scope.Config, dryWindow, element, precip);

        // featureSet="exact" + lead "all" runs every default exact lead
        // ({12, 24} under the T2 tier).
        var trainRc = await trainCmd.RunAsync(
            target: "temperature",
            lead: "all",
            station: null,
            window: null,
            featureSet: "exact",
            tier: "T2",
            includeUkv: false,
            exactLeads: new[] { 12, 24 },
            cycles: null,
            locationOverride: null,
            ct: default);
        trainRc.Should().Be(0, "Phase 2d train should succeed on the exact-runtime smoke fixture");

        var locationDir = Path.Combine(scope.ModelsPath, "temperature", locationName);
        Directory.Exists(locationDir).Should().BeTrue();
        var bundleName = Directory.EnumerateDirectories(locationDir)
            .Select(d => Path.GetFileName(d)!)
            .Where(n => n.EndsWith("_phase2d"))
            .OrderBy(v => v, StringComparer.Ordinal)
            .LastOrDefault();
        bundleName.Should().NotBeNull("Phase 2d should mint a bundle with the _phase2d suffix");
        var bundleDir = Path.Combine(locationDir, bundleName!);
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty();
        File.Exists(Path.Combine(bundleDir, "training_metadata.json")).Should().BeTrue();

        var predictCmd = new TempPredictCommand(
            new XunitLogger<TempPredictCommand>(_output), scope.Config);
        var predictRc = await predictCmd.RunAsync(
            target: "temperature",
            modelVersion: "current",
            forDate: DateOnly.FromDateTime(predictAnchor),
            locationOverride: null,
            ct: default);
        predictRc.Should().Be(0);

        var predParquet = Path.Combine(
            scope.PredictionsPath, "temperature", locationName,
            $"model_version={bundleName}",
            $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
        File.Exists(predParquet).Should().BeTrue($"predict_2d should emit {predParquet}");
    }
}
