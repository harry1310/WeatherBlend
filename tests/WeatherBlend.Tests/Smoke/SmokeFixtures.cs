using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using NetEscapades.Configuration.Yaml;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// Shared helpers for the WB-side end-to-end smoke harness.
///
/// Contract: every smoke test owns a <see cref="SmokeScope"/> (tempdir +
/// AppConfig pointing at it), writes synthetic parquet trees that match
/// the production schemas the train + predict commands read, invokes
/// the production train command (which produces a real ML.NET lead_*.zip
/// + sidecars), then invokes the production predict command and asserts
/// the output parquet exists and has rows.
///
/// Why use the real train command instead of fabricating bundles by
/// hand: each phase has 6+ sidecar files (conformal_calibrator_NNh.json,
/// dry_window_climatology.json, feature_importance.json, …) that the
/// predict path reads. Recreating them by hand would lock the smoke
/// against a frozen-in-time bundle shape; invoking the train command
/// keeps the smoke faithful to whatever the current trainer writes.
///
/// All synthetic data uses fixed-seed RNG so a smoke failure is
/// deterministic — same seed every run.
/// </summary>
internal static class SmokeFixtures
{
    // -------------------------------------------------------------------
    // The 7-NWP lean set the precipitation feature builder expects.
    // Order matches PrecipFeatureBuilder.BuildSpec lean output (gfs,
    // ecmwf, icon, mf, gem, aifs, jma).
    // -------------------------------------------------------------------
    public static readonly string[] LeanModelIds = new[]
    {
        "gfs_seamless",
        "ecmwf_ifs025",
        "icon_seamless",
        "meteofrance_seamless",
        "ukmo_seamless",
        "gem_seamless",
        "ecmwf_aifs025_single",
        "jma_seamless",
    };

    // Default lead set the production train/predict commands cycle through.
    public static readonly int[] DefaultLeads = new[] { 24, 48, 72, 96, 120 };

    // -------------------------------------------------------------------
    // AppConfig builder — minimal but production-shaped.
    // -------------------------------------------------------------------

    /// <summary>
    /// Build an AppConfig with one location + its rainfall stations,
    /// Storage paths rooted at <paramref name="root"/>, and Blenders
    /// configured for the leads we use. The shipped config.yaml is
    /// authoritative for some structural defaults (Blenders.Precip etc.)
    /// — we load it then override Storage + Locations.
    /// </summary>
    public static AppConfig BuildAppConfig(
        string root,
        string locationName,
        IReadOnlyList<(string Id, string Name)> rainfallStations,
        double latitude = 50.5831,
        double longitude = -3.7931,
        double elevationMeters = 393.0)
    {
        // Load shipped config to inherit Blenders / DryWindow / Variables defaults.
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new ConfigurationBuilder().AddYamlFile(configPath, optional: false).Build();
        var bound = new AppConfig();
        cfg.Bind(bound);

        // Override Locations: only the smoke's target location, with its
        // rainfall stations. The Locations list drives both train and
        // predict; everything else (Variables, Models, Blenders) stays
        // from the shipped config.
        bound.Locations = new List<LocationConfig>
        {
            new LocationConfig
            {
                Name = locationName,
                DisplayName = locationName,
                Latitude = latitude,
                Longitude = longitude,
                ElevationMeters = elevationMeters,
                Rainfall = new RainfallConfig
                {
                    Stations = rainfallStations
                        .Select(s => new RainfallStationConfig { Id = s.Id, Name = s.Name })
                        .ToList(),
                },
                Tabs = new List<string> { "rain", "temperature", "dry_window" },
            },
        };

        // Override Storage to point at the tempdir. All eight subtrees
        // train + predict read/write.
        bound.Storage = new StorageConfig
        {
            ForecastsPath    = Path.Combine(root, "forecasts"),
            ObservationsPath = Path.Combine(root, "obs"),
            Era5Path         = Path.Combine(root, "era5"),
            RainfallPath     = Path.Combine(root, "rainfall"),
            PredictionsPath  = Path.Combine(root, "predictions"),
            ReportsPath      = Path.Combine(root, "reports"),
            MetOfficeObsPath = Path.Combine(root, "mo_obs"),
            ModelsPath       = Path.Combine(root, "models"),
        };

        return bound;
    }

    // -------------------------------------------------------------------
    // Forecast tree writer
    // -------------------------------------------------------------------

    /// <summary>
    /// Write a synthetic Open-Meteo-previous-runs-shaped forecast tree
    /// under <c>{ForecastsPath}/location={loc}/model={m}/date={d}/...parquet</c>.
    /// One parquet per (model, valid-date), 24 hourly valid times per
    /// day, every lead in <paramref name="leads"/>. Storm-clustered
    /// precip + diurnal temperature so LightGBM/BART has a learnable
    /// signal at any per-NWP feature.
    /// </summary>
    /// <param name="runTimeSource">
    /// <c>"offset_day"</c> for train-time previous-runs rows, or
    /// <c>"reported"</c> for live predict rows. Each call writes one
    /// source; the file name encodes the source so two calls into the
    /// same date directory don't overwrite each other.
    /// </param>
    public static async Task WriteForecastTreeAsync(
        string forecastsPath,
        string locationName,
        DateTime startUtc,
        int nDays,
        string runTimeSource,
        IReadOnlyList<string>? models = null,
        IReadOnlyList<int>? leads = null,
        int rngSeed = 42)
    {
        models ??= LeanModelIds;
        leads ??= DefaultLeads;
        var rng = new Random(rngSeed);

        var totalHours = nDays * 24;
        var basePrecip = BuildBasePrecip(totalHours, rng);
        var baseTemp = BuildBaseTemp(totalHours);

        foreach (var model in models)
        {
            // Per-model bias so the precip + spread features have spread.
            var precipBias = 0.7 + 0.6 * rng.NextDouble();
            var tempBias = -1.5 + 3.0 * rng.NextDouble();

            for (int d = 0; d < nDays; d++)
            {
                var dayStart = startUtc.AddDays(d);
                var rows = new List<ForecastRow>(leads.Count * 24);
                foreach (var lead in leads)
                {
                    for (int h = 0; h < 24; h++)
                    {
                        var idx = d * 24 + h;
                        var valid = dayStart.AddHours(h);
                        var runTime = valid.AddHours(-lead);
                        var p = Math.Max(0.0, basePrecip[idx] * precipBias + Gauss(rng, 0.0, 0.15));
                        var t = baseTemp[idx] + tempBias + Gauss(rng, 0.0, 0.6);
                        var rh = Clamp(80.0 - 5.0 * t + 20.0 * (p > 0.1 ? 1 : 0) + Gauss(rng, 0, 8.0), 20.0, 100.0);
                        var dp = t - (100.0 - rh) / 5.0;
                        rows.Add(new ForecastRow
                        {
                            LocationName = locationName,
                            Model = model,
                            RunTimeUtc = runTime,
                            ValidTimeUtc = valid,
                            LeadHours = lead,
                            RunTimeSource = runTimeSource,
                            Precipitation = p,
                            RelativeHumidity2m = rh,
                            Temperature2m = t,
                            DewPoint2m = dp,
                            CloudCoverLow  = Clamp(50 + 40 * (p > 0.1 ? 1 : 0) + Gauss(rng, 0, 10), 0, 100),
                            CloudCoverMid  = Clamp(40 + 30 * (p > 0.1 ? 1 : 0) + Gauss(rng, 0, 10), 0, 100),
                            CloudCoverHigh = Clamp(30 + Gauss(rng, 0, 15), 0, 100),
                            CloudCover     = Clamp(50 + 30 * (p > 0.1 ? 1 : 0) + Gauss(rng, 0, 12), 0, 100),
                            Cape = Math.Max(0.0, Gauss(rng, 50, 80)),
                            WindSpeed10m = Math.Max(0.0, 8.0 + Gauss(rng, 0, 3)),
                            WindGusts10m = Math.Max(0.0, 12.0 + Gauss(rng, 0, 4)),
                            WindDirection10m = 360.0 * rng.NextDouble(),
                            SurfacePressure = 1013.0 + Gauss(rng, 0, 8),
                            Visibility = Math.Max(100.0, 20000.0 - 5000.0 * (p > 0.1 ? 1 : 0) + Gauss(rng, 0, 1500)),
                            ShortwaveRadiation = Math.Max(0.0, 300.0 * Math.Sin(2 * Math.PI * h / 24.0 - Math.PI / 2.0) + Gauss(rng, 0, 30)),
                            DirectRadiation = Math.Max(0.0, 200.0 * Math.Sin(2 * Math.PI * h / 24.0 - Math.PI / 2.0) + Gauss(rng, 0, 25)),
                            DiffuseRadiation = Math.Max(0.0, 80.0 + Gauss(rng, 0, 15)),
                        });
                    }
                }

                var dir = Path.Combine(forecastsPath, $"location={locationName}", $"model={model}", $"date={dayStart:yyyy-MM-dd}");
                Directory.CreateDirectory(dir);
                // File name encodes source so train (offset_day) and
                // predict (reported) fixtures coexist without overwrite.
                var fileName = $"run=00_{runTimeSource}.parquet";
                await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, fileName));
            }
        }
    }

    // -------------------------------------------------------------------
    // Rainfall truth (EA 15-min) writer
    // -------------------------------------------------------------------

    /// <summary>
    /// Write a synthetic EA rainfall truth tree under
    /// <c>{RainfallPath}/location={loc}/station={slug}/date={d}/rainfall.parquet</c>.
    /// Four 15-min readings per hour (matches the strict-4-of-4 hourly
    /// rule the train SQL applies). Wet/dry pattern correlated with the
    /// forecast tree's base signal via shared seed so the trained
    /// blender has a learnable target.
    /// </summary>
    public static async Task WriteRainfallTruthAsync(
        string rainfallPath,
        string locationName,
        string stationFriendly,
        DateTime startUtc,
        int nDays,
        int rngSeed = 49)
    {
        var rng = new Random(rngSeed);
        var fcRng = new Random(42);  // match WriteForecastTree's default seed for correlation
        var totalHours = nDays * 24;
        var basePrecip = BuildBasePrecip(totalHours, fcRng);

        var slug = EaSlug(stationFriendly);
        var stationDir = Path.Combine(rainfallPath, $"location={locationName}", $"station={slug}");

        for (int d = 0; d < nDays; d++)
        {
            var dayStart = startUtc.AddDays(d);
            var rows = new List<RainfallRow>(24 * 4);
            for (int h = 0; h < 24; h++)
            {
                var idx = d * 24 + h;
                // Observation noise around the forecast signal.
                var hourly = Math.Max(0.0, basePrecip[idx] + Gauss(rng, 0.0, 0.05));
                var perQuarter = hourly / 4.0;
                var baseT = dayStart.AddHours(h);
                for (int q = 0; q < 4; q++)
                {
                    rows.Add(new RainfallRow
                    {
                        LocationName = locationName,
                        StationId = $"smoke-{slug}",
                        StationName = stationFriendly,
                        ObservedTimeUtc = baseT.AddMinutes(15 * q),
                        Value15MinMm = perQuarter,
                        Quality = "Good",
                        Completeness = "Complete",
                    });
                }
            }
            var dir = Path.Combine(stationDir, $"date={dayStart:yyyy-MM-dd}");
            Directory.CreateDirectory(dir);
            await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "rainfall.parquet"));
        }
    }

    // -------------------------------------------------------------------
    // Exact-runtime forecast tree (Phase 2d / 3d)
    // -------------------------------------------------------------------

    /// <summary>
    /// Exact-runtime model ids, in canonical order (matches
    /// Exact12hFeatureBuilder.CanonicalModelOrder).
    /// </summary>
    public static readonly string[] ExactRuntimeModelIds = new[]
    {
        "gfs_ncep",
        "ecmwf_ifs_oper",
        "ecmwf_aifs_oper",
        "met_office_global",
        "gefs_ncep_mean",
    };

    /// <summary>
    /// Write a synthetic exact-runtime forecast tree under the same path
    /// shape as <see cref="WriteForecastTreeAsync"/>, but emits rows with
    /// <c>RunTimeSource='exact'</c>, the exact-runtime model ids, and
    /// ValidTimes restricted to the synoptic 6-hour grid {0, 6, 12, 18}.
    /// Phase 2d (T2 tier) defaults to lead 12; Phase 3d uses {12, 24}.
    /// </summary>
    public static async Task WriteExactRuntimeForecastTreeAsync(
        string forecastsPath,
        string locationName,
        DateTime startUtc,
        int nDays,
        IReadOnlyList<string>? models = null,
        IReadOnlyList<int>? leads = null,
        int rngSeed = 31)
    {
        models ??= ExactRuntimeModelIds;
        leads ??= new[] { 12, 24 };
        var validHours = new[] { 0, 6, 12, 18 };
        var rng = new Random(rngSeed);
        var totalValidStamps = nDays * validHours.Length;
        var baseTemp = new double[totalValidStamps];
        var basePrecip = new double[totalValidStamps];
        for (int i = 0; i < totalValidStamps; i++)
        {
            baseTemp[i] = 12.0 + 6.0 * Math.Sin(2 * Math.PI * i / (validHours.Length * 365.0))
                                + 4.0 * Math.Sin(2 * Math.PI * i / validHours.Length - Math.PI / 2.0);
            basePrecip[i] = Math.Max(0.0, Gauss(rng, 0.2, 0.4));
        }

        foreach (var model in models)
        {
            var modelBiasT = Gauss(rng, 0, 1.0);
            var modelBiasP = 0.7 + 0.6 * rng.NextDouble();
            for (int d = 0; d < nDays; d++)
            {
                var dayStart = startUtc.AddDays(d);
                var rows = new List<ForecastRow>();
                foreach (var lead in leads)
                {
                    foreach (var h in validHours)
                    {
                        var stampIdx = d * validHours.Length + Array.IndexOf(validHours, h);
                        var valid = dayStart.AddHours(h);
                        var runTime = valid.AddHours(-lead);
                        rows.Add(new ForecastRow
                        {
                            LocationName = locationName,
                            Model = model,
                            RunTimeUtc = runTime,
                            ValidTimeUtc = valid,
                            LeadHours = lead,
                            RunTimeSource = "exact",
                            Temperature2m = baseTemp[stampIdx] + modelBiasT + Gauss(rng, 0, 0.5),
                            DewPoint2m = baseTemp[stampIdx] - 3.0 + Gauss(rng, 0, 0.6),
                            RelativeHumidity2m = Clamp(80.0 + Gauss(rng, 0, 8), 20.0, 100.0),
                            Precipitation = basePrecip[stampIdx] * modelBiasP + Math.Max(0.0, Gauss(rng, 0, 0.1)),
                            CloudCoverLow  = Clamp(50 + Gauss(rng, 0, 15), 0, 100),
                            CloudCoverMid  = Clamp(40 + Gauss(rng, 0, 15), 0, 100),
                            CloudCoverHigh = Clamp(30 + Gauss(rng, 0, 15), 0, 100),
                            Cape = Math.Max(0.0, Gauss(rng, 50, 80)),
                            WindSpeed10m = Math.Max(0.0, 8.0 + Gauss(rng, 0, 3)),
                            WindDirection10m = 360.0 * rng.NextDouble(),
                            SurfacePressure = 1013.0 + Gauss(rng, 0, 8),
                        });
                    }
                }
                var dir = Path.Combine(forecastsPath, $"location={locationName}", $"model={model}", $"date={dayStart:yyyy-MM-dd}");
                Directory.CreateDirectory(dir);
                await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "run=00_exact.parquet"));
            }
        }
    }

    // -------------------------------------------------------------------
    // Orographic static JSON (Phase 3o)
    // -------------------------------------------------------------------

    /// <summary>
    /// Write a synthetic v1 orographic JSON at
    /// <c>data/static/orographic/{slug}.json</c> relative to the supplied
    /// <paramref name="staticRoot"/>. Phase 3o's train command reads this
    /// path via a HARDCODED <c>Path.Combine("data", "static", "orographic")</c>
    /// — callers must arrange for the process CWD or staticRoot to land
    /// on the right disk location. Values are dummy but satisfy the
    /// <see cref="OroStaticFeatures"/> loader's v1 required-field schema.
    /// </summary>
    public static async Task WriteOrographicStaticAsync(
        string staticRoot,
        string slug,
        double elevationVsCellM = 30.0,
        double relief5kmM = 200.0,
        double terrainRuggedness5kmM = 40.0)
    {
        Directory.CreateDirectory(staticRoot);
        var upwind = new Dictionary<string, double>
        {
            ["N"] = 10.0, ["NE"] = 15.0, ["E"] = 5.0, ["SE"] = -5.0,
            ["S"] = -10.0, ["SW"] = -15.0, ["W"] = -8.0, ["NW"] = 2.0,
        };
        var json = new
        {
            slug = slug,
            elevation_vs_cell_m = elevationVsCellM,
            relief_5km_m = relief5kmM,
            terrain_ruggedness_5km_m = terrainRuggedness5kmM,
            upwind_gain_5km = upwind,
            terrain_gradient_dx = 0.005,
            terrain_gradient_dy = -0.003,
        };
        var path = Path.Combine(staticRoot, $"{slug}.json");
        await File.WriteAllTextAsync(path,
            System.Text.Json.JsonSerializer.Serialize(json, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    }

    // -------------------------------------------------------------------
    // Fake Phase 4a bundle (python-trained in production; WB never trains
    // it, so the 4b mint smoke needs a hand-built stand-in).
    // -------------------------------------------------------------------

    /// <summary>
    /// Write a minimal Phase 4a bundle under
    /// <c>{modelsRoot}/precipitation/{stationSlug}/{version}_phase4a/</c>:
    /// training_metadata.json + test_predictions.parquet, plus a
    /// climatology.json so PrecipPredictCommand.RunStationAsync's
    /// climatology gate passes (production 4a writes one too). Used as a
    /// stand-in for the python-trained 4a artefact that Phase4bMintCommand
    /// joins with 3o.
    /// </summary>
    public static async Task<string> WriteFakePhase4aBundleAsync(
        string modelsRoot,
        string stationSlug,
        string locationName,
        DateTime anchor,
        IReadOnlyList<int>? leads = null,
        DateTime? testSliceStart = null,
        int testSliceDays = 14,
        int rngSeed = 71)
    {
        leads ??= DefaultLeads;
        var rng = new Random(rngSeed);
        var version = anchor.ToString("'v'yyyy-MM-dd_HHmmss") + "_phase4a";
        var bundleDir = Path.Combine(modelsRoot, "precipitation", stationSlug, version);
        Directory.CreateDirectory(bundleDir);

        // test_predictions.parquet — Phase 4b mint inner-joins
        // 4a.test_predictions × 3o.test_predictions on (valid_time, lead),
        // so the fake 4a's test rows MUST overlap with 3o's test slice
        // (which is the last 15% of the forecast train window). Default
        // testSliceStart is `anchor − testSliceDays days`, which works
        // when anchor == trainStart + forecastDays AND testSliceDays
        // fits inside the train window; pass an explicit testSliceStart
        // when those assumptions don't hold (e.g. the FullPipeline smoke
        // where truth spans 220d but forecasts only 30d).
        var testStart = testSliceStart ?? anchor.AddDays(-testSliceDays);
        var testRows = new List<WeatherBlend.Train.Common.TestPredictionRow>();
        for (int d = 0; d < testSliceDays; d++)
        {
            var day = testStart.AddDays(d);
            foreach (var lead in leads)
            {
                for (int h = 0; h < 24; h++)
                {
                    testRows.Add(new WeatherBlend.Train.Common.TestPredictionRow
                    {
                        valid_time = day.AddHours(h),
                        station = stationSlug,
                        lead = lead,
                        p_wet = Math.Clamp(rng.NextDouble() * 0.6 + 0.1, 0.0, 1.0),
                        observed_wet = (byte)(rng.NextDouble() < 0.3 ? 1 : 0),
                    });
                }
            }
        }
        await ParquetSerializer.SerializeAsync(testRows,
            Path.Combine(bundleDir, "test_predictions.parquet"));

        // training_metadata.json — minimal but with the fields downstream
        // reads (Phase, LocationName, PerLead).
        var perLead = leads.ToDictionary(
            l => l.ToString(),
            l => new
            {
                LeadHours = l,
                TrainRows = 1000,
                ValRows = 200,
                TestRows = 14 * 24,
                BestSingle = "(per-cell BART)",
                BlendTestMae = 0.15,
                BlendTestRmse = 0.0,
                BlendTestBias = 0.0,
                BestSingleValMae = (double?)null,
                BestSingleTestMae = (double?)null,
            });
        var metadata = new
        {
            Version = version,
            Target = "precipitation",
            Phase = "4a",
            LocationName = locationName,
            DataSource = "smoke-fake",
            TrainedAtUtc = DateTime.UtcNow,
            Hyperparameters = new Dictionary<string, object> { ["library"] = "smoke-fake" },
            DeviationsFromBrief = new[] { "Hand-built 4a stand-in for the 4b smoke." },
            PerLead = perLead,
        };
        await File.WriteAllTextAsync(
            Path.Combine(bundleDir, "training_metadata.json"),
            System.Text.Json.JsonSerializer.Serialize(metadata,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

        // climatology.json — required by PrecipPredictCommand even for
        // Python-trained phases. Minimal monthly P(wet) constant.
        var climJson = "{ \"MonthlyProbWet\": { " + string.Join(", ",
            Enumerable.Range(1, 12).Select(m => $"\"{m}\": 0.25")) + " } }";
        await File.WriteAllTextAsync(Path.Combine(bundleDir, "climatology.json"), climJson);

        return version;
    }

    /// <summary>
    /// Write a minimal Phase 4a hourly predictions parquet at the path
    /// Phase4bPredictCommand reads from:
    /// <c>{predictionsRoot}/precipitation/{slug}/model_version={v}/date={anchor}/predictions.parquet</c>.
    /// One row per (lead, hour-of-target-day) — matches the layout
    /// predict_4a.py emits per cycle.
    /// </summary>
    public static async Task WriteFakePhase4aPredictionsAsync(
        string predictionsRoot,
        string stationSlug,
        string locationName,
        string version,
        DateTime anchor,
        IReadOnlyList<int>? leads = null,
        int rngSeed = 73)
    {
        leads ??= DefaultLeads;
        var rng = new Random(rngSeed);
        var dir = Path.Combine(predictionsRoot, "precipitation", stationSlug,
            $"model_version={version}", $"date={anchor:yyyy-MM-dd}");
        Directory.CreateDirectory(dir);
        var rows = new List<PrecipPredictionRow>();
        foreach (var lead in leads)
        {
            var targetDay = anchor.AddHours(lead).Date;
            for (int h = 0; h < 24; h++)
            {
                rows.Add(new PrecipPredictionRow
                {
                    LocationName = locationName,
                    TruthStation = stationSlug,
                    ModelVersion = version,
                    PredictionMadeAtUtc = anchor,
                    ValidTimeUtc = targetDay.AddHours(h),
                    LeadHours = lead,
                    ProbWet = Math.Clamp(rng.NextDouble() * 0.5 + 0.1, 0.0, 1.0),
                    ClimatologyPWet = 0.25,
                    FeatureVectorHash = "smoke-fake",
                });
            }
        }
        await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "predictions.parquet"));
    }

    // -------------------------------------------------------------------
    // ERA5 truth writer — for temperature train (2b/2c)
    // -------------------------------------------------------------------

    /// <summary>
    /// Write a synthetic ERA5 truth tree under
    /// <c>{Era5Path}/location={loc}/date={d}/data.parquet</c>. One
    /// hourly row per valid time; correlated with the forecast tree's
    /// base temperature so the temperature blender has a learnable
    /// target.
    /// </summary>
    public static async Task WriteEra5TruthAsync(
        string era5Path,
        string locationName,
        DateTime startUtc,
        int nDays,
        int rngSeed = 67)
    {
        var rng = new Random(rngSeed);
        var totalHours = nDays * 24;
        var baseTemp = BuildBaseTemp(totalHours);
        for (int d = 0; d < nDays; d++)
        {
            var dayStart = startUtc.AddDays(d);
            var rows = new List<Era5Row>(24);
            for (int h = 0; h < 24; h++)
            {
                var idx = d * 24 + h;
                rows.Add(new Era5Row
                {
                    LocationName = locationName,
                    ValidTimeUtc = dayStart.AddHours(h),
                    Temperature2m = baseTemp[idx] + Gauss(rng, 0.0, 0.4),
                    DewPoint2m = baseTemp[idx] - 3.0 + Gauss(rng, 0.0, 0.6),
                    RelativeHumidity2m = Clamp(75.0 + Gauss(rng, 0, 8), 20.0, 100.0),
                    Precipitation = Math.Max(0.0, Gauss(rng, 0.1, 0.3)),
                    CloudCover = Clamp(50.0 + Gauss(rng, 0, 25), 0.0, 100.0),
                    WindSpeed10m = Math.Max(0.0, 8.0 + Gauss(rng, 0, 3)),
                    SurfacePressure = 1013.0 + Gauss(rng, 0, 6),
                });
            }
            var dir = Path.Combine(era5Path, $"location={locationName}", $"date={dayStart:yyyy-MM-dd}");
            Directory.CreateDirectory(dir);
            await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "data.parquet"));
        }
    }

    // -------------------------------------------------------------------
    // Signal generators — shared seed so forecast + truth correlate
    // -------------------------------------------------------------------

    private static double[] BuildBasePrecip(int totalHours, Random rng)
    {
        var diurnal = new double[totalHours];
        for (int i = 0; i < totalHours; i++)
        {
            diurnal[i] = 0.5 + 0.4 * Math.Sin(2 * Math.PI * i / 24.0 - Math.PI / 2.0);
        }
        var storms = new double[totalHours];
        var nDays = Math.Max(1, totalHours / 24);
        var stormStarts = new HashSet<int>();
        for (int s = 0; s < 2 * nDays; s++)
        {
            stormStarts.Add(rng.Next(totalHours));
        }
        foreach (var s in stormStarts)
        {
            var width = rng.Next(2, 10);
            for (int k = 0; k < width && s + k < totalHours; k++)
            {
                storms[s + k] += -Math.Log(1.0 - rng.NextDouble()) * 2.5;  // Exponential(2.5)
            }
        }
        var result = new double[totalHours];
        for (int i = 0; i < totalHours; i++) result[i] = Math.Max(0.0, storms[i] * diurnal[i]);
        return result;
    }

    private static double[] BuildBaseTemp(int totalHours)
    {
        var t = new double[totalHours];
        for (int i = 0; i < totalHours; i++)
        {
            t[i] = 12.0
                + 6.0 * Math.Sin(2 * Math.PI * i / (24.0 * 365.0))
                + 4.0 * Math.Sin(2 * Math.PI * i / 24.0 - Math.PI / 2.0);
        }
        return t;
    }

    private static double Gauss(Random rng, double mean, double sigma)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + sigma * z;
    }

    private static double Clamp(double v, double lo, double hi)
        => v < lo ? lo : v > hi ? hi : v;

    public static string EaSlug(string friendlyName)
        => "ea_" + string.Join("_", friendlyName.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries));
}

// -------------------------------------------------------------------
// XUnit-friendly logger — routes train/predict logs to test output so
// failures show why instead of "guard rejected" with no context.
// -------------------------------------------------------------------

internal sealed class XunitLogger<T> : ILogger<T>
{
    private readonly ITestOutputHelper _out;
    public XunitLogger(ITestOutputHelper @out) { _out = @out; }

    public IDisposable BeginScope<TState>(TState state) where TState : notnull => NullScope.Instance;
    public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Information;

    public void Log<TState>(LogLevel level, EventId id, TState state, Exception? ex, Func<TState, Exception?, string> formatter)
    {
        if (!IsEnabled(level)) return;
        try { _out.WriteLine($"[{level}] {formatter(state, ex)}"); } catch { /* test already done */ }
        if (ex is not null) try { _out.WriteLine(ex.ToString()); } catch { }
    }

    private sealed class NullScope : IDisposable
    {
        public static readonly NullScope Instance = new();
        public void Dispose() { }
    }
}

// -------------------------------------------------------------------
// Env var scope — set/restore process env vars within a using block.
// -------------------------------------------------------------------

/// <summary>
/// Sets one or more env vars on construction, restores their previous
/// values on Dispose. Used by the smoke tests to flip the existing
/// <c>WB_SKIP_CONFORMAL</c> escape hatch on so each phase's smoke
/// doesn't pay the ~3 min/lead conformal-fit cost.
/// </summary>
internal sealed class EnvScope : IDisposable
{
    private readonly Dictionary<string, string?> _prev = new();

    /// <summary>The standard smoke environment: skip the post-train conformal
    /// fit AND cap LightGBM hyperparams to ~10× faster than production. All
    /// of these are production escape hatches read by trainers with safe
    /// fallbacks, so an unset env equals the production defaults.</summary>
    public static readonly (string Name, string Value)[] StandardSmokeVars = new[]
    {
        ("WB_SKIP_CONFORMAL", "1"),
        ("WB_SMOKE_ITER",    "60"),
        ("WB_SMOKE_ESR",     "20"),
        ("WB_SMOKE_LEAVES",  "15"),
    };

    /// <summary>Default: applies <see cref="StandardSmokeVars"/>. Use this
    /// in every smoke unless you need a custom env override.</summary>
    public EnvScope() : this(StandardSmokeVars) { }

    public EnvScope(params (string Name, string Value)[] vars)
    {
        foreach (var (name, value) in vars)
        {
            _prev[name] = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, value);
        }
    }

    public void Dispose()
    {
        foreach (var (name, prev) in _prev)
        {
            Environment.SetEnvironmentVariable(name, prev);
        }
    }
}

// -------------------------------------------------------------------
// Scope: tempdir + AppConfig, cleaned up on Dispose.
// -------------------------------------------------------------------

/// <summary>
/// Tempdir lifecycle for a single smoke test. Holds the AppConfig used
/// by the test and exposes the storage roots. Use via <c>using var
/// scope = new SmokeScope(locationName, rainfallStations)</c>.
/// </summary>
internal sealed class SmokeScope : IDisposable
{
    public string Root { get; }
    public AppConfig Config { get; }

    public string ForecastsPath => Config.Storage.ForecastsPath;
    public string RainfallPath  => Config.Storage.RainfallPath;
    public string Era5Path      => Config.Storage.Era5Path;
    public string ModelsPath    => Config.Storage.ModelsPath;
    public string PredictionsPath => Config.Storage.PredictionsPath;

    public SmokeScope(
        string locationName,
        IReadOnlyList<(string Id, string Name)> rainfallStations,
        double latitude = 50.5831,
        double longitude = -3.7931,
        double elevationMeters = 393.0)
    {
        Root = Path.Combine(Path.GetTempPath(), "wb-smoke-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Root);
        Config = SmokeFixtures.BuildAppConfig(
            Root, locationName, rainfallStations,
            latitude, longitude, elevationMeters);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(Root)) Directory.Delete(Root, recursive: true);
        }
        catch
        {
            // Tempdir cleanup is best-effort; a leaked R-side handle or
            // a held file lock shouldn't fail the test.
        }
    }
}
