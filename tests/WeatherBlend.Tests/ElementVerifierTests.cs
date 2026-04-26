using FluentAssertions;
using WeatherBlend.Evaluate.Element;
using WeatherBlend.Models;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pure-logic tests for <see cref="ElementVerifier"/>. No DuckDB or parquet — caller
/// supplies in-memory predictions, truth dict, and metadata. Verifies the join,
/// drift detection, and per-model best-single picking.
/// </summary>
public class ElementVerifierTests
{
    private static ElementPredictionRow Pred(
        string version, int lead, DateTime valid, double blend,
        double? gfs = null, double? ecmwf = null) =>
        new()
        {
            LocationName = "bonehill_rocks",
            Element = "wind",
            ModelVersion = version,
            PredictionMadeAtUtc = valid.AddHours(-lead),
            ValidTimeUtc = valid,
            LeadHours = lead,
            BlendValue = blend,
            ModelGfs = gfs,
            ModelEcmwf = ecmwf,
            FeatureVectorHash = "x",
        };

    [Fact]
    public void Compute_groups_by_version_and_lead_and_drops_rows_without_truth()
    {
        var t0 = new DateTime(2026, 4, 10, 0, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Pred("v1", 24, t0,                blend: 5.0, gfs: 4.5, ecmwf: 5.5),
            Pred("v1", 24, t0.AddHours(1),    blend: 6.0, gfs: 5.5, ecmwf: 6.5),
            Pred("v1", 24, t0.AddHours(2),    blend: 7.0, gfs: 6.5, ecmwf: 7.5),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [t0]                = 5.0,
            [t0.AddHours(1)]    = 6.0,
            // t0+2 has no truth — that row must be dropped.
        };

        var rows = ElementVerifier.Compute(new ElementVerifier.Inputs
        {
            Predictions = preds,
            TruthByTime = truth,
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata>(),
            AsOfUtc = t0.AddDays(7),         // window covers t0..t0+2
            WindowDays = 14,
            Era5LatencyDays = 0,
            DriftThreshold = 1.5,
        });

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.ModelVersion.Should().Be("v1");
        r.LeadHours.Should().Be(24);
        r.N.Should().Be(2);                  // third row dropped (no truth)
        r.BlendMae.Should().BeApproximately(0, 1e-9);  // blend matches truth exactly
    }

    [Fact]
    public void Drift_flag_fires_when_blend_MAE_exceeds_threshold_x_training_MAE()
    {
        var t0 = new DateTime(2026, 4, 10, 0, 0, 0, DateTimeKind.Utc);
        // Blend predicts 10, truth is 0 → MAE = 10. Training MAE was 1; threshold 1.5 → 1.5 < 10 → drift.
        var preds = new[]
        {
            Pred("vX", 24, t0,             blend: 10.0),
            Pred("vX", 24, t0.AddHours(1), blend: 10.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [t0]              = 0,
            [t0.AddHours(1)]  = 0,
        };
        var md = new ModelArtifact.TrainingMetadata
        {
            Version = "vX",
            PerLead = new()
            {
                ["24"] = new ModelArtifact.PerLeadStats { LeadHours = 24, BlendTestMae = 1.0 },
            },
        };

        var rows = ElementVerifier.Compute(new ElementVerifier.Inputs
        {
            Predictions = preds,
            TruthByTime = truth,
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata> { ["vX"] = md },
            AsOfUtc = t0.AddDays(7),
            WindowDays = 14,
            Era5LatencyDays = 0,
            DriftThreshold = 1.5,
        });

        rows.Should().HaveCount(1);
        rows[0].DriftFlag.Should().BeTrue();
        rows[0].ReferenceTestMae.Should().Be(1.0);
        rows[0].BlendMae.Should().BeApproximately(10, 1e-9);
    }

    [Fact]
    public void BestSingle_picks_lowest_MAE_per_model_in_window()
    {
        var t0 = new DateTime(2026, 4, 10, 0, 0, 0, DateTimeKind.Utc);
        // GFS is closer to truth than ECMWF here.
        var preds = new[]
        {
            Pred("v", 24, t0,             blend: 5.0, gfs: 5.0, ecmwf: 0.0),
            Pred("v", 24, t0.AddHours(1), blend: 5.0, gfs: 5.0, ecmwf: 0.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [t0]              = 5,
            [t0.AddHours(1)]  = 5,
        };

        var rows = ElementVerifier.Compute(new ElementVerifier.Inputs
        {
            Predictions = preds,
            TruthByTime = truth,
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata>(),
            AsOfUtc = t0.AddDays(7),
            WindowDays = 14,
            Era5LatencyDays = 0,
            DriftThreshold = 1.5,
        });

        rows.Should().HaveCount(1);
        rows[0].BestSingleName.Should().Be("gfs");
        rows[0].BestSingleMae.Should().BeApproximately(0, 1e-9);
    }
}
