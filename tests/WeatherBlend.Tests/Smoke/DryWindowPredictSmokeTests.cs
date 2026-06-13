using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// End-to-end smoke for the two dry-window phases served by
/// <see cref="DryWindowTrainCommand"/>/<see cref="DryWindowPredictCommand"/>:
/// 3b (LightGBM-per-(station, window, lead) champion) and 3p
/// (Gaussian copula MC over the bound 3o hourly P(wet) marginals).
///
/// 3b is the regular train + predict cycle. 3p doesn't train a per-window
/// LightGBM — its bundle is correlation.json + training_metadata.json
/// (no <c>lead_*.zip</c>), and predict reads a previously-emitted 3o
/// hourly predictions parquet to draw correlated samples.
/// </summary>
[Trait("Category", "Smoke")]
public class DryWindowPredictSmokeTests
{
    private readonly ITestOutputHelper _output;

    public DryWindowPredictSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // -----------------------------------------------------------------
    // Phase 3b — LightGBM per (station, window, lead)
    // -----------------------------------------------------------------

    [Fact]
    public async Task Phase3b_train_then_predict_writes_non_empty_dry_window_parquet()
    {
        const string locationName = "bonehill_rocks";
        const string stationFriendly = "Bellever Dartmoor";
        var stationSlug = SmokeFixtures.EaSlug(stationFriendly);
        var trainStart = new DateTime(2025, 11, 1, 0, 0, 0, DateTimeKind.Utc);
        // 3b operates on per-day rows; the train guard rejects <100 days.
        const int trainDays = 130;
        var predictAnchor = trainStart.AddDays(trainDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ("smoke-bellever", stationFriendly) });

        // 3b uses Leads.Short = {24, 48, 72} only; restrict the fixture
        // to those three so the parquet count is 3/5 of the 3a fixture.
        // Train fixtures go via the production sync script (workflow + smoke
        // share scripts/sync_train_data.sh); predict 'reported' rows direct.
        var leads = new[] { 24, 48, 72 };
        var fakeR2 = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteForecastTreeAsync(
            Path.Combine(fakeR2, "data", "forecasts"),
            locationName, trainStart, trainDays,
            runTimeSource: "offset_day", leads: leads);
        await SmokeFixtures.WriteRainfallTruthAsync(
            Path.Combine(fakeR2, "data", "truth", "rainfall"),
            locationName, stationFriendly, trainStart, trainDays);
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "3b",
            r2Source: fakeR2, localRoot: scope.Root);

        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor.AddHours(1), nDays: 4,
            runTimeSource: "reported", leads: leads);

        var metadata = new ModelMetadataRepository(
            new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
        var conformal = new DryWindowConformalFitCommand(
            new XunitLogger<DryWindowConformalFitCommand>(_output), scope.Config, metadata);
        var trainCmd = new DryWindowTrainCommand(
            new XunitLogger<DryWindowTrainCommand>(_output), scope.Config, metadata, conformal);

        // 3b trains per (station, window, lead). Default windows are
        // {3, 4, 6}; default leads are Short = {24, 48, 72}. Restrict
        // to one window (6h) + 3 leads for the smoke so we don't run 9
        // LightGBM fits when 3 are enough to prove the wiring.
        var trainRc = await trainCmd.RunAsync(
            stationArg: stationFriendly,
            windowArg: "6",
            leads: new[] { 24, 48, 72 },
            location: scope.Config.Location,
            ct: default);
        trainRc.Should().Be(0, "Phase 3b train should succeed on the smoke fixture");

        // Bundle path: data/models/dry_window/{slug}/window_6h/{version}/lead_*.zip
        var compositeDir = Path.Combine(scope.ModelsPath, "dry_window", stationSlug, "window_6h");
        Directory.Exists(compositeDir).Should().BeTrue($"train should mint a composite dir at {compositeDir}");
        var versions = Directory.EnumerateDirectories(compositeDir)
            .Select(d => Path.GetFileName(d)!)
            .ToList();
        versions.Should().NotBeEmpty();
        var bundleName = versions.OrderBy(v => v, StringComparer.Ordinal).Last();
        var bundleDir = Path.Combine(compositeDir, bundleName);

        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty(
            "3b bundle must contain LightGBM lead_*.zip artefacts");
        File.Exists(Path.Combine(bundleDir, "training_metadata.json")).Should().BeTrue();
        File.Exists(Path.Combine(bundleDir, "dry_window_climatology.json")).Should().BeTrue(
            "3b bundle must have dry_window_climatology.json (predict reads it as a fallback)");

        var predictCmd = new DryWindowPredictCommand(
            NullLogger<DryWindowPredictCommand>.Instance, scope.Config);
        var predictRc = await predictCmd.RunAsync(
            stationArg: stationFriendly,
            windowArg: "6",
            modelVersion: "current",
            forDate: DateOnly.FromDateTime(predictAnchor),
            locationOverride: null,
            ct: default);
        predictRc.Should().Be(0, "Phase 3b predict should succeed against the freshly-minted bundle");

        // Predictions parquet under
        // data/predictions/dry_window/{slug}/window_6h/model_version={v}/date={d}/predictions.parquet
        var predDir = Path.Combine(
            scope.PredictionsPath, "dry_window", stationSlug, "window_6h",
            $"model_version={bundleName}",
            $"date={predictAnchor:yyyy-MM-dd}");
        var predParquet = Path.Combine(predDir, "predictions.parquet");
        File.Exists(predParquet).Should().BeTrue($"predict_3b should emit {predParquet}");

        var rows = await ParquetSerializer.DeserializeAsync<DryWindowPredictionRow>(predParquet);
        rows.Should().NotBeEmpty("predict parquet must hold at least one scored row");
        rows.Should().AllSatisfy(r =>
        {
            r.LocationName.Should().Be(locationName);
            r.TruthStation.Should().Be(stationSlug);
            r.WindowHours.Should().Be(6);
            r.ModelVersion.Should().Be(bundleName);
            r.ProbHasDryWindow.Should().BeInRange(0.0, 1.0);
        });
        rows.Select(r => r.LeadHours).Distinct().OrderBy(l => l).Should().BeEquivalentTo(
            new[] { 24, 48, 72 },
            "3b should score every trained lead");
    }

    // -----------------------------------------------------------------
    // Phase 3p is covered by FullPipelineSmokeTests (chained with 3o +
    // 4b to amortise the expensive 4-station 3o train across all three
    // phases). Re-enable the per-phase 3p test below if you need to
    // debug the Σ fit / copula MC path in isolation.
    // -----------------------------------------------------------------
#if FULL_PER_PHASE_SMOKE
    /// <summary>
    /// 3p doesn't fit a LightGBM model. The "train" step fits a Σ
    /// correlation matrix on the train-slice EA truth (one Σ per
    /// station, used across all windows × leads), and binds to a 3o
    /// champion in the manifest. Predict reads 3o's live predictions
    /// and runs copula MC over them.
    ///
    /// Prerequisites: a real 3o bundle in the manifest, and a 3o
    /// hourly predictions parquet at the predict anchor. This smoke
    /// trains 3o inline (4-station Bonehill pool with synthetic
    /// terrain JSONs) then trains + predicts 3p.
    ///
    /// 3p needs ≥200 daytime-complete days of EA truth to fit Σ — the
    /// fixture supplies 220 days (the bake-off threshold + headroom).
    /// </summary>
    [Fact]
    public async Task Phase3p_train_then_predict_writes_non_empty_dry_window_parquet()
    {
        const string locationName = "bonehill_rocks";
        var bonehillStations = new[]
        {
            ("smoke-bellever",   "Bellever Dartmoor"),
            ("smoke-bovey",      "Bovey Tracey"),
            ("smoke-hexworthy",  "Dartmoor nr Hexworthy"),
            ("smoke-princetown", "Princetown"),
        };
        // 3p Σ fit reads EA truth only (no forecast join), so the truth
        // tree needs ≥200 days but the forecast tree (used by the 3o
        // prerequisite train) only needs ~30 days for >200 rows per
        // (station, lead). Truncating the forecast fixture keeps the
        // total parquet count + DuckDB glob cost manageable.
        var truthStart = new DateTime(2025, 7, 1, 0, 0, 0, DateTimeKind.Utc);
        const int truthDays = 220;
        const int forecastDays = 30;
        // 3o builds rows by JOINING forecast × truth on ValidTimeUtc, so
        // the forecast window has to OVERLAP the truth. Align both at the
        // same start.
        var trainStart = truthStart;
        var predictAnchor = truthStart.AddDays(truthDays);

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: bonehillStations);

        // 3p chains off 3o, so train fixtures cover both: forecasts +
        // rainfall + orographic JSONs (3o needs orography for the rich
        // feature set). All written to fake-R2 then synced via the
        // production script.
        var fakeR2 = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteForecastTreeAsync(
            Path.Combine(fakeR2, "data", "forecasts"),
            locationName, trainStart, forecastDays,
            runTimeSource: "offset_day",
            leads: new[] { 24, 48, 72 });
        foreach (var (_, friendly) in bonehillStations)
        {
            await SmokeFixtures.WriteRainfallTruthAsync(
                Path.Combine(fakeR2, "data", "truth", "rainfall"),
                locationName, friendly, truthStart, truthDays);
        }
        foreach (var (_, friendly) in bonehillStations)
        {
            await SmokeFixtures.WriteOrographicStaticAsync(
                Path.Combine(fakeR2, "data", "static", "orographic"),
                SmokeFixtures.EaSlug(friendly));
        }
        // 3p needs forecasts + rainfall + orographic + models — script's
        // case table picks up all three trees for the 3p phase id.
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "3o,3p",
            r2Source: fakeR2, localRoot: scope.Root);

        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor.AddHours(1), nDays: 4,
            runTimeSource: "reported",
            leads: new[] { 24, 48, 72 });

        {

            var metadata = new ModelMetadataRepository(
                new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
            var precipConformal = new PrecipConformalFitCommand(
                new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
            var precipTrain = new PrecipTrainCommand(
                new XunitLogger<PrecipTrainCommand>(_output), scope.Config, precipConformal);

            // Pre-req: train Phase 3o so the 4-station pool has a
            // champion in the manifest. 3p resolves the latest 3o
            // version per station via ModelArtifact.ResolveStationPhaseVersion.
            var oroRc = await precipTrain.RunAsync(
                leads: new[] { 24, 48, 72 },
                station: null,
                featureSet: "oro",
                tier: null, includeUkv: null, exactLeads: null, cycles: null,
                location: scope.Config.Location,
                ct: default);
            oroRc.Should().Be(0, "Phase 3o train (3p prerequisite) should succeed");

            // 3p predict reads 3o's live predictions parquet from disk —
            // run 3o predict for each station so those parquets exist.
            var precipPredict = new PrecipPredictCommand(
                new XunitLogger<PrecipPredictCommand>(_output), scope.Config);
            foreach (var (_, friendly) in bonehillStations)
            {
                var rcPredict3o = await precipPredict.RunAsync(
                    truthStation: friendly,
                    modelVersion: "current",
                    forDate: DateOnly.FromDateTime(predictAnchor),
                    locationOverride: null,
                    ct: default);
                rcPredict3o.Should().Be(0, $"3o predict for {friendly} (3p prerequisite) should succeed");
            }

            var dryWindowConformal = new DryWindowConformalFitCommand(
                new XunitLogger<DryWindowConformalFitCommand>(_output), scope.Config, metadata);
            var dryWindowTrain = new DryWindowTrainCommand(
                new XunitLogger<DryWindowTrainCommand>(_output), scope.Config, metadata, dryWindowConformal);

            var trainRc = await dryWindowTrain.RunPhase3pAsync(scope.Config.Location, default);
            trainRc.Should().Be(0, "Phase 3p train (Σ fit + bind to 3o) should succeed");

            // 3p bundle layout: data/models/dry_window/{slug}/window_Nh/{v}_phase3p/
            // with correlation.json + training_metadata.json, NO lead_*.zip.
            var primarySlug = SmokeFixtures.EaSlug("Bellever Dartmoor");
            var dryWindowDir = Path.Combine(scope.ModelsPath, "dry_window", primarySlug, "window_6h");
            Directory.Exists(dryWindowDir).Should().BeTrue();
            var bundleName = Directory.EnumerateDirectories(dryWindowDir)
                .Select(d => Path.GetFileName(d)!)
                .Where(n => n.EndsWith("_phase3p"))
                .OrderBy(v => v, StringComparer.Ordinal)
                .LastOrDefault();
            bundleName.Should().NotBeNull("Phase 3p should mint a _phase3p bundle per (station, window)");
            var bundleDir = Path.Combine(dryWindowDir, bundleName!);
            File.Exists(Path.Combine(bundleDir, "correlation.json")).Should().BeTrue(
                "3p bundle must carry correlation.json (the fitted Σ)");
            File.Exists(Path.Combine(bundleDir, "training_metadata.json")).Should().BeTrue();
            Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().BeEmpty(
                "3p has no LightGBM model — bundle should NOT have lead_*.zip");

            var predictCmd = new DryWindowPredictCommand(
                new XunitLogger<DryWindowPredictCommand>(_output), scope.Config);
            var predictRc = await predictCmd.RunAsync(
                stationArg: "Bellever Dartmoor",
                windowArg: "6",
                modelVersion: "current",
                forDate: DateOnly.FromDateTime(predictAnchor),
                locationOverride: null,
                ct: default);
            predictRc.Should().Be(0, "Phase 3p predict should succeed");

            var predParquet = Path.Combine(
                scope.PredictionsPath, "dry_window", primarySlug, "window_6h",
                $"model_version={bundleName}",
                $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
            File.Exists(predParquet).Should().BeTrue($"predict_3p should emit {predParquet}");
        }
    }
#endif
}
