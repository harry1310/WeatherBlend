using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// Single chained smoke for the three Bonehill phases that share an
/// expensive Phase 3o train as a prerequisite: 3o, 3p (copula MC over
/// 3o), and 4b (composition of 4a + 3o). Running them as separate
/// per-phase tests pays the 4-station 3o train cost three times
/// (≈15 min wasted); this test pays it once.
///
/// Sequence:
///   1. Train 3o (4-station Bonehill pool, terrain features)
///   2. Run 3o predict per station (writes the parquets 3p + 4b will read)
///   3. Train 3p (Σ fit over EA truth; binds to 3o champion in manifest)
///   4. Run 3p predict (copula MC over the just-written 3o parquets)
///   5. Fake a Phase 4a bundle per station + promote in manifest
///   6. Fake a Phase 4a hourly predictions parquet per station
///   7. Run Phase4bMintCommand (joins 4a + 3o test_predictions per station)
///   8. Run Phase4bPredictCommand (live composition from today's 4a + 3o)
///
/// 3a is NOT trained here — Phase4bMintCommand now pulls its location
/// pin + climatology from 3o (the actual stage-2 source) rather than
/// 3a. 3a is covered end-to-end by PrecipPredictSmokeTests in isolation.
///
/// Per-phase assertions live inline; if any phase fails the rest still
/// runs (FluentAssertions soft-fails) so you see every breakage in one
/// log. CWD is swapped to the smoke scope root because Phase 3o's
/// terrain-JSON loader uses a hardcoded relative path
/// (<c>Path.Combine("data", "static", "orographic")</c>).
/// </summary>
[Trait("Category", "Smoke")]
public class FullPipelineSmokeTests
{
    private readonly ITestOutputHelper _output;

    public FullPipelineSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task Bonehill_3o_3p_4b_chain_writes_non_empty_predictions()
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
        // tree needs ≥200 days but the forecast tree (used by 3o train)
        // only needs ~30 days for >200 rows per (station, lead).
        var truthStart = new DateTime(2025, 7, 1, 0, 0, 0, DateTimeKind.Utc);
        // 3p Σ fit's hardcoded threshold is 200 daytime-complete days;
        // 205 gives comfortable headroom without paying for the extra
        // ~15 days of parquets the previous 220-day fixture wrote.
        const int truthDays = 205;
        const int forecastDays = 30;
        var predictAnchor = truthStart.AddDays(truthDays);
        var leads = new[] { 24, 48, 72 };

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName, rainfallStations: bonehillStations);

        // ---- Shared fixtures ----
        // Train data (forecasts + rainfall + orographic) via the production
        // sync script. Predict 'reported' forecast rows written direct
        // (predict-time pulls are a separate production path).
        var fakeR2 = Path.Combine(scope.Root, "fake-r2");
        await SmokeFixtures.WriteForecastTreeAsync(
            Path.Combine(fakeR2, "data", "forecasts"),
            locationName, truthStart, forecastDays,
            runTimeSource: "offset_day", leads: leads);
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
        // Full chain trains 3o + 3p + 4b. Script's case table covers
        // each phase's tree set; passing all three resolves the union.
        SmokeFixtures.RunSyncTrainData(
            location: locationName, phases: "3o,3p,4b",
            r2Source: fakeR2, localRoot: scope.Root);

        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, predictAnchor.AddHours(1), nDays: 4,
            runTimeSource: "reported", leads: leads);

        {

            // ---- Command graph ----
            var metadata = new ModelMetadataRepository(
                new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
            var precipConformal = new PrecipConformalFitCommand(
                new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
            var precipTrain = new PrecipTrainCommand(
                new XunitLogger<PrecipTrainCommand>(_output), scope.Config, precipConformal);
            var precipPredict = new PrecipPredictCommand(
                new XunitLogger<PrecipPredictCommand>(_output), scope.Config);
            var dryWindowConformal = new DryWindowConformalFitCommand(
                new XunitLogger<DryWindowConformalFitCommand>(_output), scope.Config, metadata);
            var dryWindowTrain = new DryWindowTrainCommand(
                new XunitLogger<DryWindowTrainCommand>(_output), scope.Config, metadata, dryWindowConformal);
            var dryWindowPredict = new DryWindowPredictCommand(
                new XunitLogger<DryWindowPredictCommand>(_output), scope.Config);
            var phase4bMint = new Phase4bMintCommand(
                new XunitLogger<Phase4bMintCommand>(_output), scope.Config);
            var phase4bPredict = new Phase4bPredictCommand(
                new XunitLogger<Phase4bPredictCommand>(_output), scope.Config);

            // ---- 1. Phase 3o train (4-station pool with terrain) ----
            // No 3a in this chain — 4b mint pulls location + climatology
            // from the 3o bundle directly (rewired 2026-05-26). 3a's own
            // train+predict path is covered by PrecipPredictSmokeTests.
            var rc3o = await precipTrain.RunAsync(
                leads: leads, station: null, featureSet: "oro",
                tier: null, includeUkv: null, exactLeads: null, cycles: null,
                location: scope.Config.Location, ct: default);
            rc3o.Should().Be(0, "Phase 3o train should succeed");
            var primarySlug = SmokeFixtures.EaSlug("Bellever Dartmoor");
            var stationDir = Path.Combine(scope.ModelsPath, "precipitation", primarySlug);
            var bundle3o = Directory.EnumerateDirectories(stationDir)
                .Select(d => Path.GetFileName(d)!)
                .Where(n => n.EndsWith("_phase3o"))
                .OrderBy(v => v, StringComparer.Ordinal)
                .LastOrDefault();
            bundle3o.Should().NotBeNull("3o should mint a _phase3o bundle per station");
            Directory.EnumerateFiles(Path.Combine(stationDir, bundle3o!), "lead_*.zip")
                .Should().NotBeEmpty();

            // ---- 3. Phase 3o predict per station (3p + 4b read these) ----
            foreach (var (_, friendly) in bonehillStations)
            {
                var rcPredict3o = await precipPredict.RunAsync(
                    truthStation: friendly,
                    modelVersion: "current",
                    forDate: DateOnly.FromDateTime(predictAnchor),
                    locationOverride: null, ct: default);
                rcPredict3o.Should().Be(0, $"3o predict for {friendly} should succeed");
            }
            var pred3o = Path.Combine(
                scope.PredictionsPath, "precipitation", primarySlug,
                $"model_version={bundle3o}",
                $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
            File.Exists(pred3o).Should().BeTrue("3o predictions parquet must exist");

            // ---- 4. Phase 3p train (Σ fit on truth, binds to 3o) ----
            var rc3pTrain = await dryWindowTrain.RunPhase3pAsync(scope.Config.Location, default);
            rc3pTrain.Should().Be(0, "Phase 3p train (Σ fit + bind to 3o) should succeed");
            var dryWindowDir = Path.Combine(
                scope.ModelsPath, "dry_window", primarySlug, "window_6h");
            var bundle3p = Directory.EnumerateDirectories(dryWindowDir)
                .Select(d => Path.GetFileName(d)!)
                .Where(n => n.EndsWith("_phase3p"))
                .OrderBy(v => v, StringComparer.Ordinal)
                .LastOrDefault();
            bundle3p.Should().NotBeNull("3p should mint a _phase3p bundle per (station, window)");
            var bundle3pDir = Path.Combine(dryWindowDir, bundle3p!);
            File.Exists(Path.Combine(bundle3pDir, "correlation.json")).Should().BeTrue(
                "3p bundle must carry correlation.json (fitted Σ)");
            Directory.EnumerateFiles(bundle3pDir, "lead_*.zip").Should().BeEmpty(
                "3p has no LightGBM model — bundle should NOT have lead_*.zip");

            // ---- 5. Phase 3p predict (copula MC over the 3o parquets) ----
            var rc3pPredict = await dryWindowPredict.RunAsync(
                stationArg: "Bellever Dartmoor",
                windowArg: "6",
                modelVersion: "current",
                forDate: DateOnly.FromDateTime(predictAnchor),
                locationOverride: null, ct: default);
            rc3pPredict.Should().Be(0, "Phase 3p predict should succeed");
            var pred3p = Path.Combine(
                scope.PredictionsPath, "dry_window", primarySlug, "window_6h",
                $"model_version={bundle3p}",
                $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
            File.Exists(pred3p).Should().BeTrue("3p predictions parquet must exist");

            // ---- 5b. Stale-pin fallback (regression, 2026-06-11) ----
            // A mid-week 3o re-promotion leaves 3p's bound precip_3o_version
            // pointing at a version that no longer writes predictions; the
            // first anchor with no old-version partition starved all 20 3p
            // cells and tripped the coverage guard (run 27321393054, the
            // night after the 3c/3o policy retrain). Predict must fall back
            // to the station's CURRENT Active 3o. Simulate by re-pinning the
            // bundle to a version that never predicted, then re-running.
            var meta3pPath = Path.Combine(bundle3pDir, "training_metadata.json");
            var meta3pJson = System.Text.Json.Nodes.JsonNode.Parse(
                await File.ReadAllTextAsync(meta3pPath))!;
            meta3pJson["Hyperparameters"]![DryWindow3pPredictor.Precip3oVersionKey] =
                "v1900-01-01_000000_phase3o";
            await File.WriteAllTextAsync(meta3pPath, meta3pJson.ToJsonString(
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            File.Delete(pred3p);

            var rc3pFallback = await dryWindowPredict.RunAsync(
                stationArg: "Bellever Dartmoor",
                windowArg: "6",
                modelVersion: "current",
                forDate: DateOnly.FromDateTime(predictAnchor),
                locationOverride: null, ct: default);
            rc3pFallback.Should().Be(0,
                "3p predict must survive a stale 3o pin by falling back to the current Active 3o");
            File.Exists(pred3p).Should().BeTrue(
                "3p predictions parquet must be re-emitted via the Active-3o fallback");

            // ---- 6. Fake Phase 4a per station + promote in manifest ----
            // 4b mint joins 4a.test_predictions × 3o.test_predictions on
            // (valid_time, lead). The fake 4a's test rows must overlap
            // with 3o's test slice — which is the last ~15% of the 30-day
            // forecast window. Compute that slice explicitly so the join
            // produces non-empty rows.
            var oroTestStart = truthStart.AddDays((int)Math.Floor(forecastDays * 0.85));
            var oroTestDays = forecastDays - (int)Math.Floor(forecastDays * 0.85);
            foreach (var (_, friendly) in bonehillStations)
            {
                var slug = SmokeFixtures.EaSlug(friendly);
                var v4a = await SmokeFixtures.WriteFakePhase4aBundleAsync(
                    scope.ModelsPath, slug, locationName, predictAnchor,
                    leads: leads,
                    testSliceStart: oroTestStart,
                    testSliceDays: oroTestDays);
                ModelArtifact.PromoteStationVersion(
                    scope.ModelsPath, "precipitation", slug, v4a, "4a");
                await SmokeFixtures.WriteFakePhase4aPredictionsAsync(
                    scope.PredictionsPath, slug, locationName, v4a, predictAnchor, leads: leads);

                // 4b became a 3-way mean(4a, 3o, 3c) on 2026-06-02, so the mint +
                // live predict now also require a 3c member. The chain trains 3o
                // for real but not 3c, so hand-build a 3c stand-in (test_predictions
                // for the mint join + live predictions for the live composition),
                // overlapping the same test slice / anchor as 4a so the (V,L)
                // joins are non-empty. Found by suffix glob — no manifest promote.
                var v3c = await SmokeFixtures.WriteFakePhase4aBundleAsync(
                    scope.ModelsPath, slug, locationName, predictAnchor,
                    leads: leads, testSliceStart: oroTestStart, testSliceDays: oroTestDays,
                    rngSeed: 75, phaseSuffix: "_phase3c", phaseTag: "3c");
                await SmokeFixtures.WriteFakePhase4aPredictionsAsync(
                    scope.PredictionsPath, slug, locationName, v3c, predictAnchor, leads: leads, rngSeed: 76);
            }

            // ---- 7. Phase 4b mint (joins 4a + 3o test_predictions) ----
            var rc4bMint = await phase4bMint.RunAsync(default);
            rc4bMint.Should().Be(0, "Phase 4b mint should succeed");
            var bundle4b = Directory.EnumerateDirectories(stationDir)
                .Select(d => Path.GetFileName(d)!)
                .Where(n => n.EndsWith("_phase4b"))
                .OrderBy(v => v, StringComparer.Ordinal)
                .LastOrDefault();
            bundle4b.Should().NotBeNull("4b mint should emit a _phase4b bundle per station");
            var bundle4bDir = Path.Combine(stationDir, bundle4b!);
            File.Exists(Path.Combine(bundle4bDir, "test_predictions.parquet")).Should().BeTrue();

            // ---- 8. Phase 4b predict (live composition) ----
            var rc4bPredict = await phase4bPredict.RunAsync(
                forDate: DateOnly.FromDateTime(predictAnchor), ct: default);
            rc4bPredict.Should().Be(0, "Phase 4b predict should emit per-station parquets");
            var pred4b = Path.Combine(
                scope.PredictionsPath, "precipitation", primarySlug,
                $"model_version={bundle4b}",
                $"date={predictAnchor:yyyy-MM-dd}", "predictions.parquet");
            File.Exists(pred4b).Should().BeTrue("4b predictions parquet must exist");
            var rows4b = await ParquetSerializer.DeserializeAsync<PrecipPredictionRow>(pred4b);
            rows4b.Should().NotBeEmpty();
            rows4b.Should().AllSatisfy(r =>
            {
                r.LocationName.Should().Be(locationName);
                r.TruthStation.Should().Be(primarySlug);
                r.ProbWet.Should().BeInRange(0.0, 1.0);
            });
        }
    }
}
