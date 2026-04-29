using FluentAssertions;
using WeatherBlend.Evaluate.Temp;
using WeatherBlend.Models;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class VerifierTests
{
    private static readonly DateTime AsOf = new(2026, 4, 22, 0, 0, 0, DateTimeKind.Utc);
    private const string V1 = "v2026-04-20_100000";

    [Fact]
    public void Compute_excludes_predictions_outside_rolling_window()
    {
        // Inside window: ValidTime well within [AsOf - 14d, AsOf - 5d].
        var inside = MakePrediction(V1, leadHours: 24, valid: AsOf.AddDays(-7), blend: 10.2);
        // Before window (16d ago): should be excluded.
        var tooOld = MakePrediction(V1, leadHours: 24, valid: AsOf.AddDays(-16), blend: 10.5);
        // After window (2d ago → within ERA5 latency): should be excluded.
        var tooNew = MakePrediction(V1, leadHours: 24, valid: AsOf.AddDays(-2), blend: 10.6);

        var truth = new Dictionary<DateTime, double>
        {
            [inside.ValidTimeUtc] = 10.0,
            [tooOld.ValidTimeUtc] = 10.0,
            [tooNew.ValidTimeUtc] = 10.0,
        };

        var rows = TempVerifier.Compute(BaseInputs(new[] { inside, tooOld, tooNew }, truth));

        rows.Should().ContainSingle();
        rows[0].N.Should().Be(1, "only the in-window row should be scored");
    }

    [Fact]
    public void Compute_drops_rows_without_matching_era5_truth()
    {
        var withTruth    = MakePrediction(V1, 24, AsOf.AddDays(-7), blend: 12.0);
        var withoutTruth = MakePrediction(V1, 24, AsOf.AddDays(-8), blend: 12.5);

        var truth = new Dictionary<DateTime, double>
        {
            [withTruth.ValidTimeUtc] = 12.3,
            // withoutTruth intentionally absent
        };

        var rows = TempVerifier.Compute(BaseInputs(new[] { withTruth, withoutTruth }, truth));

        rows.Should().ContainSingle();
        rows[0].N.Should().Be(1);
    }

    [Fact]
    public void Compute_stratifies_by_version_and_lead()
    {
        // Distinct valid times so the truth map has one entry per (version, lead) row.
        var preds = new[]
        {
            MakePrediction(V1,          24, AsOf.AddDays(-7), blend: 10.0),
            MakePrediction(V1,          48, AsOf.AddDays(-8), blend: 11.0),
            MakePrediction("v-newer",   24, AsOf.AddDays(-9), blend: 10.2),
        };
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0);

        var rows = TempVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().HaveCount(3);
        // Sort order is ordinal on ModelVersion, then numeric on LeadHours.
        // '-' (0x2D) < '2' (0x32) so "v-newer" sorts before "v2026-…".
        rows.Select(r => (r.ModelVersion, r.LeadHours))
            .Should().Equal(
                ("v-newer", 24),
                (V1, 24),
                (V1, 48));
    }

    [Fact]
    public void Compute_flags_drift_when_blend_mae_exceeds_threshold_times_training_mae()
    {
        // Training blend test MAE (via metadata) = 1.0°C. Rolling blend MAE will be 2.0°C.
        // Threshold = 1.5 → 2.0 > 1.5 × 1.0 → drift.
        var preds = Enumerable.Range(0, 5)
            .Select(i => MakePrediction(V1, 24, AsOf.AddDays(-7 - i), blend: 12.0))
            .ToArray();
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0); // error = 2.0 everywhere

        var metadata = new Dictionary<string, ModelArtifact.TrainingMetadata>
        {
            [V1] = MetadataWithLead(24, blendTestMae: 1.0),
        };

        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = preds,
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().ContainSingle();
        rows[0].BlendMae.Should().BeApproximately(2.0, 1e-9);
        rows[0].ReferenceTestMae.Should().BeApproximately(1.0, 1e-9);
        rows[0].DriftFlag.Should().BeTrue();
    }

    [Fact]
    public void Compute_does_not_flag_drift_when_within_threshold()
    {
        // Rolling MAE 1.4 < 1.5 × 1.0 = 1.5 → no drift.
        var preds = Enumerable.Range(0, 5)
            .Select(i => MakePrediction(V1, 24, AsOf.AddDays(-7 - i), blend: 11.4))
            .ToArray();
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0); // error = 1.4

        var metadata = new Dictionary<string, ModelArtifact.TrainingMetadata>
        {
            [V1] = MetadataWithLead(24, blendTestMae: 1.0),
        };

        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = preds,
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().ContainSingle();
        rows[0].DriftFlag.Should().BeFalse();
    }

    [Fact]
    public void Compute_leaves_reference_blank_and_drift_false_when_metadata_missing()
    {
        var preds = new[] { MakePrediction(V1, 24, AsOf.AddDays(-7), blend: 10.0) };
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0);

        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = preds,
            TruthByTime = truth,
            // empty metadata dict
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata>(),
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().ContainSingle();
        rows[0].ReferenceTestMae.Should().BeNull();
        rows[0].DriftFlag.Should().BeFalse();
    }

    [Fact]
    public void Compute_picks_best_single_by_window_mae()
    {
        // Truth = 10.0. Make ECMWF perfect (10.0), GFS off by 2, others off by 1.
        // Best-single should be temp_ecmwf with MAE 0.
        var preds = Enumerable.Range(0, 3).Select(i => new TempPredictionRow
        {
            LocationName = "Bonehill Rocks",
            ModelVersion = V1,
            PredictionMadeAtUtc = AsOf.AddDays(-7 - i).AddHours(-24),
            ValidTimeUtc = AsOf.AddDays(-7 - i),
            LeadHours = 24,
            BlendTemperature = 10.1,
            TempGfs   = 12.0,
            TempEcmwf = 10.0,
            TempIcon  = 11.0,
            TempMf    = 11.0,
            TempUkmo  = 11.0,
            TempGem   = 11.0,
            TempMean  = 11.0,
            FeatureVectorHash = "abc",
        }).ToArray();
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0);

        var rows = TempVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].BestSingleName.Should().Be("temp_ecmwf");
        rows[0].BestSingleMae.Should().BeApproximately(0.0, 1e-9);
    }

    [Fact]
    public void Compute_persistence_uses_truth_at_t_minus_lead_and_counts_drops()
    {
        // Two predictions 24h apart. Provide truth at t − 24h only for one of them.
        var valid1 = AsOf.AddDays(-7);
        var valid2 = AsOf.AddDays(-8);
        var preds = new[]
        {
            MakePrediction(V1, 24, valid1, blend: 10.0),
            MakePrediction(V1, 24, valid2, blend: 10.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [valid1] = 10.0,
            [valid2] = 10.0,
            // Persistence lookup: valid1 - 24h
            [valid1.AddHours(-24)] = 8.0,
            // valid2 - 24h intentionally missing → dropped
        };

        var rows = TempVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].PersistenceDropped.Should().Be(1);
        rows[0].PersistenceMae.Should().BeApproximately(2.0, 1e-9, "|8.0 - 10.0| = 2.0 for the one resolved pair");
    }

    [Fact]
    public void Compute_mean_of_models_prefers_precomputed_TempMean()
    {
        var valid = AsOf.AddDays(-7);
        var preds = new[]
        {
            new TempPredictionRow
            {
                LocationName = "Bonehill Rocks",
                ModelVersion = V1,
                PredictionMadeAtUtc = valid.AddHours(-24),
                ValidTimeUtc = valid,
                LeadHours = 24,
                BlendTemperature = 10.0,
                // Per-model inputs deliberately wrong — we want to prove TempMean is used verbatim.
                TempGfs = 0.0, TempEcmwf = 0.0, TempIcon = 0.0, TempMf = 0.0, TempUkmo = 0.0, TempGem = 0.0,
                TempMean = 9.5,
                FeatureVectorHash = "hash",
            },
        };
        var truth = new Dictionary<DateTime, double> { [valid] = 10.0 };

        var rows = TempVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].MeanMae.Should().BeApproximately(0.5, 1e-9, "|9.5 - 10.0|, using the precomputed TempMean");
    }

    [Fact]
    public void Compute_returns_empty_when_no_predictions_survive_filters()
    {
        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata>(),
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().BeEmpty();
    }

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------

    private static TempVerifier.Inputs BaseInputs(
        IEnumerable<TempPredictionRow> preds,
        IReadOnlyDictionary<DateTime, double> truth)
        => new()
        {
            Predictions = preds.ToList(),
            TruthByTime = truth,
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata>(),
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
        };

    private static TempPredictionRow MakePrediction(
        string version, int leadHours, DateTime valid, double blend)
        => new()
        {
            LocationName = "Bonehill Rocks",
            ModelVersion = version,
            PredictionMadeAtUtc = valid.AddHours(-leadHours),
            ValidTimeUtc = valid,
            LeadHours = leadHours,
            BlendTemperature = blend,
            TempGfs   = blend + 0.2,
            TempEcmwf = blend + 0.1,
            TempIcon  = blend + 0.3,
            TempMf    = blend + 0.4,
            TempUkmo  = blend + 0.5,
            TempGem   = blend + 0.6,
            TempMean  = blend + 0.35,
            TempStd   = 0.2,
            TempRange = 0.4,
            FeatureVectorHash = "fakehash",
        };

    private static ModelArtifact.TrainingMetadata MetadataWithLead(int leadHours, double blendTestMae)
        => new()
        {
            Version = V1,
            Target = "temperature",
            Phase = "2b",
            DataSource = "previous_runs_api",
            TrainedAtUtc = AsOf.AddDays(-2),
            PerLead = new Dictionary<string, ModelArtifact.PerLeadStats>
            {
                [leadHours.ToString()] = new()
                {
                    LeadHours = leadHours,
                    BlendTestMae = blendTestMae,
                    BestSingle = "temp_ecmwf",
                },
            },
        };
}
