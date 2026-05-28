using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Storage;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Wind;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// End-to-end smoke for Phase 3 of WIND_BLENDER_PLAN — the LightGBM
/// <see cref="WindSpeedLgbBlender"/> trained on Dunkeswell SYNOP truth
/// (MIDAS Open 01383) with the 29-feature lean+ORO+spread set.
///
/// Sequence:
///   1. Write a 60-day OM forecast tree (RunTimeSource='offset_day')
///      starting 2024-09-01 — the intersection of MIDAS coverage
///      ({2022,2023,2024}) and TrainingWindow.UkmoCleanWindowStart.
///   2. Write Dunkeswell SYNOP BADC-CSV truth for 2024 spanning the
///      same window via <see cref="SmokeFixtures.WriteDunkeswellMidasAsync"/>.
///   3. Write the static orographic JSON for bonehill_rocks at
///      <c>{cwd}/data/static/orographic/bonehill_rocks.json</c>
///      (CWD-relative path the blender uses).
///   4. Drive <see cref="TrainCommand"/> with target='wind-speed-lgb'
///      through the element dispatch arm — TrainCommand →
///      ElementTrainCommand → WindSpeedLgbBlender → ElementTrainerHarness.
///   5. Assert the bundle dir is suffixed <c>v{ts}_wind_speed_lgb</c>
///      (so it coexists with any sibling <c>wind</c> phase bundle in
///      the same data/models/wind/{location}/ tree) and contains
///      per-lead lead_*.zip + sidecars.
///   6. Assert MANIFEST.Stations.{location}.Active carries the new
///      <c>_wind_speed_lgb</c>-tagged version.
///
/// Slice 3.C extension: when <c>WindBlendMintCommand</c> ships, this
/// test will be extended to run the mint against the wind_speed_lgb
/// output (plus a hand-built wind_mvn parquet stand-in) and assert the
/// blended wind_speed parquet output. The fixture is shaped that way
/// (Bonehill SmokeScope, full forecast tree) for the future composition.
/// </summary>
public class WindSpeedLgbSmokeTests
{
    private readonly ITestOutputHelper _output;

    public WindSpeedLgbSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task WindSpeedLgb_train_writes_suffixed_bundle_and_promotes_manifest()
    {
        const string locationName = "bonehill_rocks";
        // 2024-09-01: TrainingWindow.UkmoCleanWindowStart floor + inside
        // the MIDAS DunkeswellYears {2022, 2023, 2024} window. 60 days
        // gives ~1440 hourly stamps per lead, well above the harness's
        // 500-row floor after the inner-join against MIDAS.
        var trainStart = new DateTime(2024, 9, 1, 0, 0, 0, DateTimeKind.Utc);
        const int trainDays = 60;

        using var env = new EnvScope();
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ("smoke-bonehill", "Bellever Dartmoor") });

        // Forecast tree (offset_day rows). The SmokeFixtures default
        // LeanModelIds set covers every NWP wind_speed_lgb's BlendersConfig
        // entry references (5 required + UKMO optional).
        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, trainStart, trainDays,
            runTimeSource: "offset_day");

        // Dunkeswell SYNOP truth (BADC-CSV). The loader parses files
        // matching midas-open*_01383_*_{year}.csv under
        // {scope.Root}/truth/midas/raw — Path.GetDirectoryName of
        // ForecastsPath yields scope.Root, so the blender resolves
        // midasRoot relative to that.
        var midasRoot = Path.Combine(scope.Root, "truth", "midas", "raw");
        await SmokeFixtures.WriteDunkeswellMidasAsync(
            midasRoot, startUtc: trainStart, nDays: trainDays);

        // Orographic JSON — WindSpeedLgbBlender resolves the path
        // relative to ForecastsPath's parent (config-driven, no CWD
        // mutation). For SmokeScope's storage layout that means
        // {scope.Root}/static/orographic/{location}.json.
        var oroRoot = Path.Combine(
            Path.GetDirectoryName(scope.Config.Storage.ForecastsPath)!,
            "static", "orographic");
        Directory.CreateDirectory(oroRoot);
        await SmokeFixtures.WriteOrographicStaticAsync(oroRoot, locationName);

        // No CurrentDirectory swap needed — both orography + MIDAS paths
        // resolve from cfg.Storage.ForecastsPath. This test runs in
        // parallel with FullPipelineSmokeTests + PrecipPredictSmokeTests
        // + DryWindowPredictSmokeTests (all of which still swap CWD for
        // 3o's CWD-relative orography) without contention.
        var element = new ElementTrainCommand(
            new XunitLogger<ElementTrainCommand>(_output),
            new IElementBlender[]
            {
                new WindSpeedLgbBlender(
                    new XunitLogger<WindSpeedLgbBlender>(_output), scope.Config),
            });

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
            target: "wind-speed-lgb", lead: "all",
            station: null, window: null, featureSet: "lean",
            ct: default);
        rc.Should().Be(0, "wind-speed-lgb train should succeed on the synthetic fixture");

        // Bundle layout: data/models/wind/{location}/v{ts}_wind_speed_lgb/.
        // The suffix is automatic in ElementTrainerHarness when PhaseTag
        // differs from ModelDirName (sibling-phase disambiguation).
        var locationDir = Path.Combine(scope.ModelsPath, "wind", locationName);
        Directory.Exists(locationDir).Should().BeTrue();
        var dirs = Directory.EnumerateDirectories(locationDir)
            .Select(d => Path.GetFileName(d)!)
            .ToList();
        dirs.Should().HaveCount(1, "this test trains exactly one phase");
        var bundleName = dirs[0];
        bundleName.Should().EndWith("_wind_speed_lgb",
            "sibling-phase dirs must carry the wind_speed_lgb suffix so they don't collide with the existing wind phase");
        var bundleDir = Path.Combine(locationDir, bundleName);

        // Per-lead LightGBM artefacts + the four canonical sidecars.
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty(
            "wind_speed_lgb bundle must contain LightGBM lead_*.zip artefacts");
        foreach (var sidecar in new[]
        {
            "training_metadata.json", "feature_schema.json",
            "feature_importance.json", "training_summary.json",
        })
        {
            File.Exists(Path.Combine(bundleDir, sidecar)).Should().BeTrue(
                $"wind_speed_lgb bundle should contain {sidecar}");
        }

        // training_metadata.json Phase = "wind_speed_lgb" (the PhaseTag).
        var metadataJson = await File.ReadAllTextAsync(
            Path.Combine(bundleDir, "training_metadata.json"));
        metadataJson.Should().Contain("\"Phase\": \"wind_speed_lgb\"",
            "training_metadata.json must stamp Phase = wind_speed_lgb so manifest resolution can disambiguate from the existing wind phase");
        metadataJson.Should().Contain("\"LocationName\": \"bonehill_rocks\"");

        // feature_schema.json carries the 29-feature lean+ORO+spread spec.
        // 6 wsp + 6 wdir + 5 ORO_LEAN + 12 SPREAD = 29; one of the SPREAD
        // names is a stable witness.
        var schemaJson = await File.ReadAllTextAsync(
            Path.Combine(bundleDir, "feature_schema.json"));
        schemaJson.Should().Contain("oro_upwind_gain",
            "feature_schema must carry the ORO_LEAN block (witness: oro_upwind_gain)");
        schemaJson.Should().Contain("wsp_spd_mean",
            "feature_schema must carry the SPREAD block (witness: wsp_spd_mean)");

        // Manifest promotion — Stations-keyed under the location.
        var manifestPath = Path.Combine(scope.ModelsPath, "wind", "MANIFEST.json");
        File.Exists(manifestPath).Should().BeTrue("wind MANIFEST.json must be promoted");
        var manifestJson = await File.ReadAllTextAsync(manifestPath);
        manifestJson.Should().Contain(bundleName,
            "manifest Active list should carry the freshly-trained version");
        manifestJson.Should().Contain(locationName,
            "manifest should be keyed under the trained location");

        // ---- Slice 3.B: predict-side smoke ----
        //
        // Drive ElementPredictCommand against the freshly-minted bundle.
        // Two surfaces under test:
        //   1. The phase-based dispatch routes wind_speed_lgb versions to
        //      WindSpeedLgbPredictPipeline (not the existing WindPredictPipeline
        //      which only pivots 2 cols).
        //   2. The output parquet lands at the suffixed model_version
        //      subtree, ready for the Slice 3.C wind_blend mint to consume.
        //
        // Predict window: PredictAnchor.Compute uses DateTime.UtcNow when
        // forDate is null, anchoring at today's date. The predict pipeline
        // then targets each lead L's day starting at 00:00 UTC of
        // (today + L/24 days). The forecast-tree fixture writes per-day
        // parquets with valid-times = dayStart.AddHours(0..23), so liveStart
        // MUST be aligned to 00:00 UTC of tomorrow for the per-day starts
        // to begin at 00:00 (not at whatever offset DateTime.UtcNow lands on).
        // 4 days covers the {24, 48, 72}h leads inclusive.
        var liveStart = DateTime.UtcNow.Date.AddDays(1);
        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, liveStart, nDays: 4,
            runTimeSource: "reported");

        var predictCmd = new ElementPredictCommand(
            new XunitLogger<ElementPredictCommand>(_output), scope.Config);
        var predictRc = await predictCmd.RunAsync(
            ElementTargets.WindSpeedLgb,
            modelVersion: "current",
            forDate: null,
            locationOverride: null,
            ct: default);
        predictRc.Should().Be(0, "wind_speed_lgb predict should succeed against the freshly-trained bundle");

        // Predictions land under data/predictions/wind/{loc}/model_version={bundleName}/date={today}/.
        // bundleName carries the _wind_speed_lgb suffix, so the model_version
        // subtree is distinct from any sibling wind ERA5 champion that would
        // share the same data/predictions/wind/{loc}/ root.
        var todayUtc = DateTime.UtcNow.Date;
        var predDir = Path.Combine(
            scope.PredictionsPath, "wind",
            $"model_version={bundleName}",
            $"date={todayUtc:yyyy-MM-dd}");
        Directory.Exists(predDir).Should().BeTrue(
            $"predict should land at {predDir}");
        var predParquet = Path.Combine(predDir, "predictions.parquet");
        File.Exists(predParquet).Should().BeTrue(
            $"predict should emit predictions.parquet at {predParquet}");

        var predRows = await Parquet.Serialization.ParquetSerializer
            .DeserializeAsync<WeatherBlend.Models.ElementPredictionRow>(predParquet);
        predRows.Should().NotBeEmpty("predict should write at least one row per (lead, valid)");
        predRows.Should().AllSatisfy(r =>
        {
            r.Element.Should().Be("wind",
                "wind_speed_lgb is a sibling phase under the wind physical target");
            r.LocationName.Should().Be(locationName);
            r.ModelVersion.Should().Be(bundleName,
                "ModelVersion must carry the _wind_speed_lgb suffix so the Slice 3.C blend mint can join on it");
            r.LeadHours.Should().BeOneOf(24, 48, 72);
            r.BlendValue.Should().BeGreaterOrEqualTo(0.0,
                "wind speed magnitude is non-negative");
        });

        // ---- Slice 3.C: wind_blend live composition ----
        //
        // Fabricate a fake wind_mvn predictions parquet (in production this
        // lands via WeatherProbabilistic's predict_wind_mvn.py). Path
        // matches WP's output layout: predictions/wind_direction/{loc}/
        // model_version={v}_wind_mvn/date={d}/predictions.parquet.
        // Shape matches WindDirectionPredictionRow (the C#-side consumer
        // schema). Cells join wind_speed_lgb's by (ValidTimeUtc, LeadHours).
        var mvnVersion = "v_smoke_wind_mvn";
        var mvnDir = Path.Combine(
            scope.PredictionsPath, "wind_direction", locationName,
            $"model_version={mvnVersion}", $"date={todayUtc:yyyy-MM-dd}");
        Directory.CreateDirectory(mvnDir);
        var mvnPath = Path.Combine(mvnDir, "predictions.parquet");
        var mvnRows = predRows.Select(lgb => new WeatherBlend.Models.WindDirectionPredictionRow
        {
            LocationName = locationName,
            ModelVersion = mvnVersion,
            PredictionMadeAtUtc = lgb.PredictionMadeAtUtc,
            ValidTimeUtc = lgb.ValidTimeUtc,
            LeadHours = lgb.LeadHours,
            // Bivariate-normal params: u = +1, v = -2 → speed = sqrt(5) ≈ 2.24,
            // direction = atan2(-1, 2)·180/π ≈ 333° (NNW). Constant cells
            // suffice for the smoke — the blend math just needs a finite
            // BlendSpeedMagnitude per matched (V, L).
            MuU = 1.0, MuV = -2.0,
            SigmaU = 0.5, SigmaV = 0.7, Rho = 0.1,
            BlendDirection = 333.4,
            BlendDirectionCi95Lo = 310.0, BlendDirectionCi95Hi = 357.0,
            BlendSpeedMagnitude = Math.Sqrt(1.0 * 1.0 + 2.0 * 2.0),
            BlendSpeedCi95Lo = 1.5, BlendSpeedCi95Hi = 3.2,
        }).ToList();
        await Parquet.Serialization.ParquetSerializer.SerializeAsync(mvnRows, mvnPath);
        File.Exists(mvnPath).Should().BeTrue("fake wind_mvn parquet should land for the blend to read");

        // Run the live blend command.
        var blendCmd = new WindBlendPredictCommand(
            new XunitLogger<WindBlendPredictCommand>(_output), scope.Config);
        var blendRc = await blendCmd.RunAsync(forDate: null, ct: default);
        blendRc.Should().Be(0,
            "wind_blend predict should succeed when both wind_speed_lgb and wind_mvn parquets are present");

        // Output lands at predictions/wind/model_version=v_wind_blend_live/date={today}/.
        var blendDir = Path.Combine(
            scope.PredictionsPath, "wind",
            $"model_version={WindBlendPredictCommand.VersionTag}",
            $"date={todayUtc:yyyy-MM-dd}");
        Directory.Exists(blendDir).Should().BeTrue($"blend should land at {blendDir}");
        var blendParquet = Path.Combine(blendDir, "predictions.parquet");
        File.Exists(blendParquet).Should().BeTrue($"blend should emit predictions.parquet at {blendParquet}");

        var blendRows = await Parquet.Serialization.ParquetSerializer
            .DeserializeAsync<WeatherBlend.Models.ElementPredictionRow>(blendParquet);
        blendRows.Should().NotBeEmpty("blend should write at least one row per matched (lead, valid)");
        blendRows.Count.Should().Be(predRows.Count,
            "blend should match every wind_speed_lgb cell (mvn fake covers them all 1:1)");
        blendRows.Should().AllSatisfy(r =>
        {
            r.Element.Should().Be("wind", "blend is under the wind physical target");
            r.LocationName.Should().Be(locationName);
            r.ModelVersion.Should().Be(WindBlendPredictCommand.VersionTag,
                "blend output stamps the fixed v_wind_blend_live tag (no MANIFEST mint yet)");
            r.LeadHours.Should().BeOneOf(24, 48, 72);
            r.BlendValue.Should().BeGreaterOrEqualTo(0.0, "wind speed magnitude is non-negative");
            r.BlendValue.Should().BeLessThan(100.0, "wind speed in m/s must be physically plausible");
        });

        // Verify the sigmoid actually mixed the two — at any matched cell the
        // blended speed is bounded between min(lgb, mvn) and max(lgb, mvn).
        var lgbByKey = predRows.ToDictionary(r => (r.ValidTimeUtc, r.LeadHours), r => r.BlendValue);
        const double mvnSpd = 2.2360679775;  // sqrt(5), matches the synthetic fake above
        blendRows.Should().AllSatisfy(r =>
        {
            var lgbSpd = lgbByKey[(r.ValidTimeUtc, r.LeadHours)];
            var lo = Math.Min(lgbSpd, mvnSpd);
            var hi = Math.Max(lgbSpd, mvnSpd);
            r.BlendValue.Should().BeInRange(lo - 1e-6, hi + 1e-6,
                "convex combination of two values must land between them");
        });
    }
}
