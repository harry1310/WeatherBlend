using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Storage;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Wind;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// End-to-end smoke for the existing ERA5-truth <see cref="WindBlender"/>
/// (PhaseTag <c>wind</c>). Confirms the Phase 2-era element pipeline still
/// trains end-to-end against the synthetic fixtures — a coverage gap noted
/// 2026-05-28 while shipping the Phase 3 <c>wind_speed_lgb</c> sibling.
///
/// Sequence:
///   1. Write a 30-day OM forecast tree (RunTimeSource='offset_day')
///   2. Write a matching 30-day ERA5 truth tree
///   3. Drive <see cref="TrainCommand"/> with target='wind' through
///      the element dispatch arm — ElementTrainCommand → WindBlender →
///      ElementTrainerHarness
///   4. Assert the bundle dir contains per-lead lead_*.zip + sidecars
///   5. Assert MANIFEST.Stations.{location}.Active carries the new
///      <c>_wind</c>-tagged version
///
/// Predict-side coverage is left for now — the Phase 2 wind blender's
/// WindPredictPipeline already round-trips via the existing Temp / 4b
/// pipelines transitively in <c>FullPipelineSmokeTests</c>; this test
/// targets the train path that has no other end-to-end coverage.
/// </summary>
[Trait("Category", "Smoke")]
public class ElementWindSmokeTests
{
    private readonly ITestOutputHelper _output;

    public ElementWindSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task Wind_train_writes_bundle_and_promotes_manifest()
    {
        const string locationName = "bonehill_rocks";
        // Train window must start on/after TrainingWindow.UkmoCleanWindowStart
        // (2024-09-01). 2025-02-01 + 30 days lands fully inside that window
        // so UKMO rows aren't all-NaN.
        var trainStart = new DateTime(2025, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        const int trainDays = 30;

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            // Wind blender doesn't read rainfall but SmokeScope requires
            // a station list to satisfy AppConfig binding.
            rainfallStations: new[] { ("smoke-bonehill", "Bellever Dartmoor") });

        // Write fixtures to a fake-R2 dir + invoke scripts/sync_train_data.sh
        // (same script the production workflow uses) to copy them into the
        // trainer's read paths. Missing pull declaration in the script's
        // per-phase needs table = smoke fails at PR time.
        var fakeR2 = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteForecastTreeAsync(
            Path.Combine(fakeR2, "data", "forecasts"),
            locationName, trainStart, trainDays,
            runTimeSource: "offset_day");
        await SmokeFixtures.WriteEra5TruthAsync(
            Path.Combine(fakeR2, "data", "truth", "era5"),
            locationName, trainStart, trainDays);
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "wind",
            r2Source: fakeR2, localRoot: scope.Root);

        // DI wiring — ElementTrainCommand needs the wind blender registered
        // for the target='wind' dispatch to find it. Other blenders aren't
        // needed for this test but pass them anyway so the smoke also
        // covers DI-resolution shape if the harness ever ranges over
        // the full set.
        var element = new ElementTrainCommand(
            new XunitLogger<ElementTrainCommand>(_output),
            new IElementBlender[]
            {
                new WindBlender(new XunitLogger<WindBlender>(_output), scope.Config),
            });

        // The cross-target dispatch path (target=wind hits the elementTarget
        // arm in TrainCommand.RunAsync). Precip/DryWindow legs are stubbed
        // because the constructor needs non-null instances — they aren't
        // invoked for target=wind.
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
        var train = new TrainCommand(
            new XunitLogger<TrainCommand>(_output), scope.Config,
            dryWindow, element, precip);

        var rc = await train.RunAsync(
            target: "wind", lead: "all",
            station: null, window: null, featureSet: "lean",
            ct: default);
        rc.Should().Be(0, "element wind train should succeed on the synthetic fixture");

        // Per-location bundle dir: data/models/wind/{location}/v{ts}/.
        // WindBlender's PhaseTag == ModelDirName == "wind" so the harness
        // emits an UNSUFFIXED dir name (v{ts} only). Sibling phases like
        // wind_speed_lgb suffix their dirs (v{ts}_wind_speed_lgb) so the
        // two coexist; this test trains only the existing wind phase.
        var locationDir = Path.Combine(scope.ModelsPath, "wind", locationName);
        Directory.Exists(locationDir).Should().BeTrue();
        var dirs = Directory.EnumerateDirectories(locationDir)
            .Select(d => Path.GetFileName(d)!)
            .ToList();
        dirs.Should().HaveCount(1, "wind train should mint exactly one bundle dir");
        var bundleName = dirs[0];
        bundleName.Should().StartWith("v", "version dir convention is v{yyyy-MM-dd_HHmmss}");
        bundleName.Should().NotContain("_wind_speed_lgb",
            "unsuffixed wind dir must not be confused with a sibling-phase suffix");
        var bundleDir = Path.Combine(locationDir, bundleName);

        // Per-lead LightGBM artefacts + the four canonical sidecars the
        // harness writes alongside (training_metadata, feature_schema,
        // feature_importance, training_summary).
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty(
            "wind bundle must contain LightGBM lead_*.zip artefacts");
        foreach (var sidecar in new[]
        {
            "training_metadata.json", "feature_schema.json",
            "feature_importance.json", "training_summary.json",
        })
        {
            File.Exists(Path.Combine(bundleDir, sidecar)).Should().BeTrue(
                $"wind bundle should contain {sidecar}");
        }

        // Manifest promotion — Stations-keyed. Active contains the new
        // version name (the phase isn't part of the version name itself;
        // it's stamped in training_metadata.json and inferred at resolve
        // time). All this test asserts is that promotion happened.
        var manifestPath = Path.Combine(scope.ModelsPath, "wind", "MANIFEST.json");
        File.Exists(manifestPath).Should().BeTrue("wind MANIFEST.json must be promoted");
        var manifestJson = await File.ReadAllTextAsync(manifestPath);
        manifestJson.Should().Contain(bundleName, "manifest Active list should carry the trained version")
            .And.Contain(locationName, "manifest should be keyed under the trained location");
    }
}
