using FluentAssertions;
using WeatherBlend.Commands;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// Smoke for the wind_blend live mint (<see cref="WindBlendPredictCommand"/>)
/// against the POST-CUTOVER member shapes (2026-06-10: wind_speed_lgb moved
/// to Python — quantile-LGB + cross-conformal CQR in WeatherProbabilistic's
/// predict_wind_speed_pi.py). Replaces the retired WindSpeedLgbSmokeTests,
/// whose .NET train/predict surfaces no longer exist.
///
/// Surfaces under test:
///   1. Champion `wind` version resolution via the wind MANIFEST + the
///      phases.yaml lineup (champion phase = "wind", read from each Active
///      version dir's training_metadata.json — NOT the *_wind_speed_lgb
///      sibling).
///   2. The lgb member is read from a PYTHON-written parquet that carries
///      the CQR band sidecar columns (BandLoMs / BandHiMs / ConformalQ) on
///      top of the ElementPredictionRow schema — union_by_name must let the
///      .NET reader ignore the extras (the load-bearing cutover contract).
///   3. The mint is the fixed 50/50 mean of champion + lgb speeds, stamped
///      v_wind_blend_live.
/// </summary>
public class WindBlendSmokeTests
{
    private readonly ITestOutputHelper _output;

    public WindBlendSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// The Python predictor's output row: every ElementPredictionRow column
    /// (predict_wind_speed_pi.py emits the exact .NET schema so wind_blend /
    /// verify keep working) PLUS the CQR band sidecar. Mirrored here as a
    /// standalone type because ElementPredictionRow is sealed.
    /// </summary>
    private sealed class PythonWindSpeedLgbRow
    {
        public string LocationName { get; set; } = "";
        public string Element { get; set; } = "";
        public string ModelVersion { get; set; } = "";
        public DateTime PredictionMadeAtUtc { get; set; }
        public DateTime ValidTimeUtc { get; set; }
        public int LeadHours { get; set; }
        public double BlendValue { get; set; }
        public double? ModelGfs { get; set; }
        public double? ModelEcmwf { get; set; }
        public double? ModelIcon { get; set; }
        public double? ModelMf { get; set; }
        public double? ModelUkmo { get; set; }
        public double? ModelGem { get; set; }
        public double? ModelAifs { get; set; }
        public DateTime? RunTimeGfs { get; set; }
        public DateTime? RunTimeEcmwf { get; set; }
        public DateTime? RunTimeIcon { get; set; }
        public DateTime? RunTimeMf { get; set; }
        public DateTime? RunTimeUkmo { get; set; }
        public DateTime? RunTimeGem { get; set; }
        public DateTime? RunTimeAifs { get; set; }
        public double? Mean { get; set; }
        public double? Std { get; set; }
        public double? Range { get; set; }
        public string FeatureVectorHash { get; set; } = "";
        // CQR band sidecar — the Python-only extras union_by_name must ignore.
        public double BandLoMs { get; set; }
        public double BandHiMs { get; set; }
        public double ConformalQ { get; set; }
    }

    [Fact]
    public async Task WindBlend_mints_5050_mean_of_champion_and_python_lgb_parquet()
    {
        const string locationName = "bonehill_rocks";
        using var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ("smoke-bonehill", "Bellever Dartmoor") });

        var todayUtc = DateTime.UtcNow.Date;
        var dateStr = todayUtc.ToString("yyyy-MM-dd");
        var madeAt = DateTime.UtcNow;

        // --- Champion `wind` bundle + manifest -----------------------------
        // ResolveStationChampionVersion walks the phases.yaml lineup and, per
        // Active version, reads training_metadata.json's Phase from the
        // models tree — so the champion bundle dir + metadata must exist.
        const string champVer = "v2026-06-01_000000";
        var champBundle = Path.Combine(scope.ModelsPath, "wind", locationName, champVer);
        Directory.CreateDirectory(champBundle);
        await File.WriteAllTextAsync(
            Path.Combine(champBundle, "training_metadata.json"),
            $$"""
            {"Version":"{{champVer}}","Target":"wind","Phase":"wind","LocationName":"{{locationName}}"}
            """);

        // The Python-promoted lgb sibling also sits in Active (promoted as
        // challenger by train_wind_speed_pi.py) — present here to prove the
        // champion resolution does NOT pick it.
        const string lgbVer = "v2026-06-09_213001_wind_speed_lgb";
        var lgbBundle = Path.Combine(scope.ModelsPath, "wind", locationName, lgbVer);
        Directory.CreateDirectory(lgbBundle);
        await File.WriteAllTextAsync(
            Path.Combine(lgbBundle, "training_metadata.json"),
            $$"""
            {"Version":"{{lgbVer}}","Target":"wind-speed-lgb","Phase":"wind_speed_lgb","LocationName":"{{locationName}}"}
            """);

        await File.WriteAllTextAsync(
            Path.Combine(scope.ModelsPath, "wind", "MANIFEST.json"),
            System.Text.Json.JsonSerializer.Serialize(new
            {
                Target = "wind",
                Stations = new Dictionary<string, object>
                {
                    [locationName] = new
                    {
                        Versions = new[] { champVer, lgbVer },
                        Active = new[] { champVer, lgbVer },
                        Location = locationName,
                    },
                },
            }));

        // --- Member predictions --------------------------------------------
        var leads = new[] { 24, 48, 72 };
        var valids = leads.SelectMany(lead =>
            Enumerable.Range(0, 4).Select(h =>
                (Lead: lead, Valid: todayUtc.AddHours(lead + h)))).ToList();

        // Champion wind rows — constant 4.0 m/s.
        var champRows = valids.Select(c => new WeatherBlend.Models.ElementPredictionRow
        {
            LocationName = locationName,
            Element = "wind",
            ModelVersion = champVer,
            PredictionMadeAtUtc = madeAt,
            ValidTimeUtc = c.Valid,
            LeadHours = c.Lead,
            BlendValue = 4.0,
            FeatureVectorHash = "champ",
        }).ToList();
        var champDir = Path.Combine(scope.PredictionsPath, "wind",
            $"model_version={champVer}", $"date={dateStr}");
        Directory.CreateDirectory(champDir);
        await Parquet.Serialization.ParquetSerializer.SerializeAsync(
            champRows, Path.Combine(champDir, "predictions.parquet"));

        // Python lgb rows — constant 6.0 m/s, with the CQR band sidecar.
        var lgbRows = valids.Select(c => new PythonWindSpeedLgbRow
        {
            LocationName = locationName,
            Element = "wind",
            ModelVersion = lgbVer,
            PredictionMadeAtUtc = madeAt,
            ValidTimeUtc = c.Valid,
            LeadHours = c.Lead,
            BlendValue = 6.0,
            ModelGfs = 5.5, ModelEcmwf = 6.5, ModelIcon = 6.0,
            Mean = 6.0, Std = 0.4, Range = 1.0,
            FeatureVectorHash = "py:abc123",
            BandLoMs = 3.8, BandHiMs = 8.4, ConformalQ = 0.42,
        }).ToList();
        var lgbDir = Path.Combine(scope.PredictionsPath, "wind",
            $"model_version={lgbVer}", $"date={dateStr}");
        Directory.CreateDirectory(lgbDir);
        await Parquet.Serialization.ParquetSerializer.SerializeAsync(
            lgbRows, Path.Combine(lgbDir, "predictions.parquet"));

        // --- Mint ------------------------------------------------------------
        var blendCmd = new WindBlendPredictCommand(
            new XunitLogger<WindBlendPredictCommand>(_output), scope.Config);
        var rc = await blendCmd.RunAsync(forDate: null, ct: default);
        rc.Should().Be(0,
            "wind_blend mint should succeed when the champion parquet + the Python lgb parquet (with band sidecar) are both present");

        var blendParquet = Path.Combine(scope.PredictionsPath, "wind",
            $"model_version={WindBlendPredictCommand.VersionTag}",
            $"date={dateStr}", "predictions.parquet");
        File.Exists(blendParquet).Should().BeTrue(
            $"blend should emit predictions.parquet at {blendParquet}");

        var blendRows = await Parquet.Serialization.ParquetSerializer
            .DeserializeAsync<WeatherBlend.Models.ElementPredictionRow>(blendParquet);
        blendRows.Should().HaveCount(valids.Count,
            "every (valid, lead) cell has both members, so every cell should mint");
        blendRows.Should().AllSatisfy(r =>
        {
            r.Element.Should().Be("wind");
            r.LocationName.Should().Be(locationName);
            r.ModelVersion.Should().Be(WindBlendPredictCommand.VersionTag);
            // 0.5·4.0 + 0.5·6.0 — the fixed 50/50 mean (2026-06-09 cross-truth
            // bake-off: beats either single on real wind; fitted weight overfit).
            r.BlendValue.Should().BeApproximately(5.0, 1e-9);
            // Provenance slots are copied from the lgb member.
            r.ModelGfs.Should().Be(5.5);
            r.FeatureVectorHash.Should().StartWith("wind_blend:");
        });
    }
}
