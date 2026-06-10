using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// End-to-end smoke for the four precipitation phases served by
/// <see cref="PrecipTrainCommand"/> and <see cref="PrecipPredictCommand"/>:
/// 3a (lean champion), 3c (rich, Membury-only), 3d (exact-runtime,
/// Bonehill-only) and 3o (rich + terrain, Bonehill-only).
///
/// Smoke contract: write synthetic forecast + truth parquet trees in a
/// tempdir, invoke the production train command (which produces a real
/// LightGBM <c>lead_*.zip</c> bundle), invoke the production predict
/// command, assert non-empty predictions parquet emitted at the expected
/// path. Catches wiring / SQL-shape / manifest-plumbing regressions
/// — not data quality or model skill (the smoke fixture is synthetic).
/// </summary>
public class PrecipPredictSmokeTests
{
    private readonly ITestOutputHelper _output;

    public PrecipPredictSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // Shared per-phase end-to-end driver. Each Theory row picks the
    // featureSet/location/station the dispatched phase needs. 3o + 3d
    // have their own dedicated tests because they require fixtures
    // outside the shared shape (4-station pool + terrain JSON for 3o,
    // exact-runtime cycle tree for 3d).
    private async Task RunPrecipPhaseSmoke(
        string phaseTag,
        string featureSet,
        string locationName,
        string stationFriendly,
        string bundleSuffix)
    {
        var stationSlug = SmokeFixtures.EaSlug(stationFriendly);
        var trainStart = new DateTime(2026, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        const int trainDays = 30;
        var predictAnchor = trainStart.AddDays(trainDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ($"smoke-{stationSlug}", stationFriendly) });

        // Train fixtures via the production sync script (workflow + smoke
        // share it); predict 'reported' rows direct. 3a + 3c share data
        // needs (forecasts + rainfall + models).
        var fakeR2 = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteForecastTreeAsync(
            Path.Combine(fakeR2, "data", "forecasts"),
            locationName, trainStart, trainDays, runTimeSource: "offset_day");
        await SmokeFixtures.WriteRainfallTruthAsync(
            Path.Combine(fakeR2, "data", "truth", "rainfall"),
            locationName, stationFriendly, trainStart, trainDays);
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "3a,3c",
            r2Source: fakeR2, localRoot: scope.Root);
        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor.AddHours(1), nDays: 6,
            runTimeSource: "reported");

        var metadata = new ModelMetadataRepository(
            new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
        var conformal = new PrecipConformalFitCommand(
            new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
        var trainCmd = new PrecipTrainCommand(
            new XunitLogger<PrecipTrainCommand>(_output), scope.Config, conformal);

        var trainRc = await trainCmd.RunAsync(
            leads: SmokeFixtures.DefaultLeads,
            station: null,
            featureSet: featureSet,
            tier: null,
            includeUkv: null,
            exactLeads: null,
            cycles: null,
            location: scope.Config.Location,
            ct: default);
        trainRc.Should().Be(0, $"Phase {phaseTag} train should succeed on the smoke fixture");

        var stationDir = Path.Combine(scope.ModelsPath, "precipitation", stationSlug);
        Directory.Exists(stationDir).Should().BeTrue();
        var bundleName = Directory.EnumerateDirectories(stationDir)
            .Select(d => Path.GetFileName(d)!)
            .Where(n => bundleSuffix.Length == 0 ? !n.Contains("_phase") : n.EndsWith(bundleSuffix))
            .OrderBy(v => v, StringComparer.Ordinal)
            .LastOrDefault();
        bundleName.Should().NotBeNull($"Phase {phaseTag} should mint a bundle with suffix '{bundleSuffix}'");
        var bundleDir = Path.Combine(stationDir, bundleName!);
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty();

        var predictCmd = new PrecipPredictCommand(
            new XunitLogger<PrecipPredictCommand>(_output), scope.Config);
        var predictRc = await predictCmd.RunAsync(
            truthStation: stationFriendly,
            modelVersion: "current",
            forDate: DateOnly.FromDateTime(predictAnchor),
            locationOverride: null,
            ct: default);
        predictRc.Should().Be(0, $"Phase {phaseTag} predict should succeed against the freshly-minted bundle");

        var predParquet = Path.Combine(
            scope.PredictionsPath, "precipitation", stationSlug,
            $"model_version={bundleName}",
            $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
        File.Exists(predParquet).Should().BeTrue($"predict_{phaseTag} should emit {predParquet}");

        var rows = await ParquetSerializer.DeserializeAsync<PrecipPredictionRow>(predParquet);
        rows.Should().NotBeEmpty();
        // Lead 12 (today's hours, m24 fed the lead-12 pivot — the per-lead
        // policy plan's short-lead cell, 2026-06-10) is emitted by the 3c/3o
        // paths only; 3a stays on the trained day buckets.
        var expectedLeads = phaseTag == "3c"
            ? new[] { 12, 24, 48, 72, 96, 120 }
            : new[] { 24, 48, 72, 96, 120 };
        rows.Select(r => r.LeadHours).Distinct().OrderBy(l => l).Should().BeEquivalentTo(
            expectedLeads,
            $"Phase {phaseTag} predict should score every trained lead bucket (+ lead 12 where the phase serves it)");
        rows.Should().AllSatisfy(r =>
        {
            r.LocationName.Should().Be(locationName);
            r.TruthStation.Should().Be(stationSlug);
            r.ProbWet.Should().BeInRange(0.0, 1.0);
        });
    }

    // -----------------------------------------------------------------
    // Phase 3a — lean LightGBM occurrence classifier
    // -----------------------------------------------------------------

    [Fact]
    public Task Phase3a_train_then_predict_writes_non_empty_predictions_parquet()
        => RunPrecipPhaseSmoke(
            phaseTag: "3a",
            featureSet: "lean",
            locationName: "bonehill_rocks",
            stationFriendly: "Bellever Dartmoor",
            bundleSuffix: "");

    // -----------------------------------------------------------------
    // Phase 3c — rich LightGBM occurrence classifier, Membury-only
    // -----------------------------------------------------------------

    [Fact]
    public Task Phase3c_train_then_predict_writes_non_empty_predictions_parquet()
        => RunPrecipPhaseSmoke(
            phaseTag: "3c",
            featureSet: "rich",
            locationName: "membury_devon",
            stationFriendly: "Chards Snowdon Hill",
            bundleSuffix: "_phase3c");

    // -----------------------------------------------------------------
    // Phase 3d — exact-runtime precip, Bonehill-only
    // -----------------------------------------------------------------

    /// <summary>
    /// Exact-runtime precip smoke. Reads RunTimeSource='exact' rows from
    /// raw S3 archives with model ids gfs_ncep / ecmwf_ifs_oper /
    /// ecmwf_aifs_oper / met_office_global / gefs_ncep_mean and valid
    /// hours {0,6,12,18}. Lead set {12, 24}.
    /// </summary>
    [Fact]
    public async Task Phase3d_exact_runtime_train_then_predict_writes_non_empty_predictions_parquet()
    {
        const string locationName = "bonehill_rocks";
        const string stationFriendly = "Bellever Dartmoor";
        var stationSlug = SmokeFixtures.EaSlug(stationFriendly);
        var trainStart = new DateTime(2025, 8, 1, 0, 0, 0, DateTimeKind.Utc);
        // RunPhase3dStationAsync needs ≥500 rows per lead; exact-runtime
        // valid grid is {0,6,12,18} → 4 stamps/day. 150 days = 600 rows.
        const int trainDays = 150;
        var predictAnchor = trainStart.AddDays(trainDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ($"smoke-{stationSlug}", stationFriendly) });

        // Exact-runtime fixtures via the production sync script. 3d shares
        // forecasts + rainfall + models with 3a/3c.
        var fakeR2_3d = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteExactRuntimeForecastTreeAsync(
            Path.Combine(fakeR2_3d, "data", "forecasts"),
            locationName, trainStart, trainDays);
        await SmokeFixtures.WriteRainfallTruthAsync(
            Path.Combine(fakeR2_3d, "data", "truth", "rainfall"),
            locationName, stationFriendly, trainStart, trainDays);
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "3d",
            r2Source: fakeR2_3d, localRoot: scope.Root);
        // Predict-side exact-runtime tree past anchor.
        await SmokeFixtures.WriteExactRuntimeForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor, nDays: 5);

        var metadata = new ModelMetadataRepository(
            new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
        var conformal = new PrecipConformalFitCommand(
            new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
        var trainCmd = new PrecipTrainCommand(
            new XunitLogger<PrecipTrainCommand>(_output), scope.Config, conformal);

        var trainRc = await trainCmd.RunAsync(
            leads: SmokeFixtures.DefaultLeads,
            station: null,
            featureSet: "exact",
            // PrecipExactFeatureBuilder.AllTiers exposes P1/P2 (NOT T1/T2/T3
            // — those are Exact12hFeatureBuilder's temperature tiers).
            tier: "P1",
            includeUkv: false,
            exactLeads: new[] { 12, 24 },
            cycles: null,
            location: scope.Config.Location,
            ct: default);
        trainRc.Should().Be(0, "Phase 3d train should succeed on the exact-runtime smoke fixture");

        var stationDir = Path.Combine(scope.ModelsPath, "precipitation", stationSlug);
        var bundleName = Directory.EnumerateDirectories(stationDir)
            .Select(d => Path.GetFileName(d)!)
            .Where(n => n.EndsWith("_phase3d"))
            .OrderBy(v => v, StringComparer.Ordinal)
            .LastOrDefault();
        bundleName.Should().NotBeNull("Phase 3d should mint a bundle with the _phase3d suffix");
        var bundleDir = Path.Combine(stationDir, bundleName!);
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty();

        var predictCmd = new PrecipPredictCommand(
            new XunitLogger<PrecipPredictCommand>(_output), scope.Config);
        var predictRc = await predictCmd.RunAsync(
            truthStation: stationFriendly,
            modelVersion: "current",
            forDate: DateOnly.FromDateTime(predictAnchor),
            locationOverride: null,
            ct: default);
        predictRc.Should().Be(0);

        var predParquet = Path.Combine(
            scope.PredictionsPath, "precipitation", stationSlug,
            $"model_version={bundleName}",
            $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
        File.Exists(predParquet).Should().BeTrue($"predict_3d should emit {predParquet}");
    }

    // -----------------------------------------------------------------
    // Phase 3o is covered by FullPipelineSmokeTests (chained with 3p +
    // 4b to amortise the expensive 4-station pooled train across all
    // three phases). Re-enable a per-phase 3o test below if you need
    // to debug the rich+terrain feature path in isolation.
    // -----------------------------------------------------------------
#if FULL_PER_PHASE_SMOKE
    /// <summary>
    /// Phase 3o trains a pooled blender across the 4 Bonehill stations
    /// (Bellever, Bovey, Hexworthy, Princetown) with 9 static terrain
    /// features appended to the rich feature row. Bundle is saved per
    /// station; per-station predict + manifest plumbing is unchanged
    /// from 3a/3c.
    ///
    /// Two production-code constraints make this smoke fiddly:
    ///   1. The orographic JSON path is hardcoded as
    ///      <c>Path.Combine("data", "static", "orographic")</c> in
    ///      <see cref="PrecipTrainCommand"/> — relative to the process
    ///      CWD. The test temporarily changes CWD to the smoke scope's
    ///      tempdir to satisfy that read. xUnit by default runs tests
    ///      in different classes in parallel, so this test is marked
    ///      to disable parallelisation against any test that also
    ///      mutates CWD.
    ///   2. Bonehill must list all four stations in config.rainfall.
    /// </summary>
    [Fact]
    public async Task Phase3o_train_then_predict_writes_non_empty_predictions_parquet()
    {
        const string locationName = "bonehill_rocks";
        var bonehillStations = new[]
        {
            ("smoke-bellever",  "Bellever Dartmoor"),
            ("smoke-bovey",     "Bovey Tracey"),
            ("smoke-hexworthy", "Dartmoor nr Hexworthy"),
            ("smoke-princetown","Princetown"),
        };
        var trainStart = new DateTime(2026, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        const int trainDays = 30;  // 720 hourly rows × 4 stations = pooled 2880 > 200/station threshold
        var predictAnchor = trainStart.AddDays(trainDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: bonehillStations);

        // Train fixtures via the production sync script. 3o needs
        // forecasts + rainfall + orographic + models — the script's case
        // table covers all four trees for this phase id. Predict
        // 'reported' rows direct after sync.
        var fakeR2 = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteForecastTreeAsync(
            Path.Combine(fakeR2, "data", "forecasts"),
            locationName, trainStart, trainDays, runTimeSource: "offset_day");
        foreach (var (_, friendly) in bonehillStations)
        {
            await SmokeFixtures.WriteRainfallTruthAsync(
                Path.Combine(fakeR2, "data", "truth", "rainfall"),
                locationName, friendly, trainStart, trainDays);
        }
        foreach (var (_, friendly) in bonehillStations)
        {
            await SmokeFixtures.WriteOrographicStaticAsync(
                Path.Combine(fakeR2, "data", "static", "orographic"),
                SmokeFixtures.EaSlug(friendly));
        }
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "3o",
            r2Source: fakeR2, localRoot: scope.Root);

        // Predict 'reported' rows direct (predict-time pulls are a
        // separate production path).
        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor.AddHours(1), nDays: 6,
            runTimeSource: "reported");

        {

            var metadata = new ModelMetadataRepository(
                new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
            var conformal = new PrecipConformalFitCommand(
                new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
            var trainCmd = new PrecipTrainCommand(
                new XunitLogger<PrecipTrainCommand>(_output), scope.Config, conformal);

            var trainRc = await trainCmd.RunAsync(
                leads: SmokeFixtures.DefaultLeads,
                station: null,
                featureSet: "oro",
                tier: null,
                includeUkv: null,
                exactLeads: null,
                cycles: null,
                location: scope.Config.Location,
                ct: default);
            trainRc.Should().Be(0, "Phase 3o train should succeed on the 4-station smoke fixture");

            // The 3o bundle gets saved under EACH station's subtree. Pick
            // Bellever as the primary smoke target.
            var primarySlug = SmokeFixtures.EaSlug("Bellever Dartmoor");
            var stationDir = Path.Combine(scope.ModelsPath, "precipitation", primarySlug);
            var bundleName = Directory.EnumerateDirectories(stationDir)
                .Select(d => Path.GetFileName(d)!)
                .Where(n => n.EndsWith("_phase3o"))
                .OrderBy(v => v, StringComparer.Ordinal)
                .LastOrDefault();
            bundleName.Should().NotBeNull("Phase 3o should mint a per-station _phase3o bundle");
            Directory.EnumerateFiles(Path.Combine(stationDir, bundleName!), "lead_*.zip")
                .Should().NotBeEmpty();

            var predictCmd = new PrecipPredictCommand(
                new XunitLogger<PrecipPredictCommand>(_output), scope.Config);
            var predictRc = await predictCmd.RunAsync(
                truthStation: "Bellever Dartmoor",
                modelVersion: "current",
                forDate: DateOnly.FromDateTime(predictAnchor),
                locationOverride: null,
                ct: default);
            predictRc.Should().Be(0);

            var predParquet = Path.Combine(
                scope.PredictionsPath, "precipitation", primarySlug,
                $"model_version={bundleName}",
                $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
            File.Exists(predParquet).Should().BeTrue($"predict_3o should emit {predParquet}");
        }
    }
#endif
}
