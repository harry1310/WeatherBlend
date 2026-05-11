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
    public void Compute_does_not_flag_drift_when_n_below_min_drift_n()
    {
        // Same shape as the drift-fires test (rolling MAE = 2.0 > 1.5 × 1.0)
        // but only ONE prediction in the window — the MinDriftN gate should
        // suppress the flag. Reference MAE is still surfaced so reviewers
        // can see why the cell didn't alert.
        var preds = new[] { MakePrediction(V1, 24, AsOf.AddDays(-7), blend: 12.0) };
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0);
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
            MinDriftN = 5,        // n=1 < 5 → flag suppressed
        });

        rows.Should().ContainSingle();
        rows[0].N.Should().Be(1);
        rows[0].BlendMae.Should().BeApproximately(2.0, 1e-9);
        rows[0].ReferenceTestMae.Should().BeApproximately(1.0, 1e-9);
        rows[0].DriftFlag.Should().BeFalse();   // gated by MinDriftN
    }

    [Fact]
    public void Compute_flags_drift_when_n_meets_min_drift_n()
    {
        // Boundary: exactly MinDriftN predictions, MAE still > threshold → flag fires.
        var preds = Enumerable.Range(0, 5)
            .Select(i => MakePrediction(V1, 24, AsOf.AddDays(-7 - i), blend: 12.0))
            .ToArray();
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, _ => 10.0);
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
            MinDriftN = 5,        // n=5 == 5 → flag fires
        });

        rows.Should().ContainSingle();
        rows[0].N.Should().Be(5);
        rows[0].DriftFlag.Should().BeTrue();
    }

    [Fact]
    public void Compute_per_lead_threshold_slope_zero_keeps_flat_threshold()
    {
        // Regression: with slope=0 the threshold is identical at every lead.
        // Two cells at the same MAE ratio (1.7× reference), leads 24 and 120 —
        // both should fire under a flat 1.5× threshold.
        var p24 = Enumerable.Range(0, 10).Select(i =>
            MakePrediction(V1, 24, AsOf.AddDays(-7).AddHours(-i), blend: 11.7)).ToArray();
        var p120 = Enumerable.Range(0, 10).Select(i =>
            MakePrediction(V1, 120, AsOf.AddDays(-7).AddHours(-i - 100), blend: 11.7)).ToArray();
        var truth = p24.Concat(p120).ToDictionary(p => p.ValidTimeUtc, _ => 10.0);

        var metadata = new Dictionary<string, ModelArtifact.TrainingMetadata>
        {
            [V1] = new()
            {
                Phase = "2b",
                PerLead =
                {
                    ["24"]  = new() { LeadHours = 24,  BlendTestMae = 1.0 },
                    ["120"] = new() { LeadHours = 120, BlendTestMae = 1.0 },
                }
            }
        };

        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = p24.Concat(p120).ToArray(),
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
            DriftThresholdSlopePer24h = 0.0,    // flat — pre-2026-05-11 behaviour
            MinDriftN = 1,
        });

        rows.Should().HaveCount(2);
        rows.Single(r => r.LeadHours == 24).DriftFlag.Should().BeTrue();
        rows.Single(r => r.LeadHours == 120).DriftFlag.Should().BeTrue();
    }

    [Fact]
    public void Compute_per_lead_threshold_slope_relaxes_drift_at_long_leads()
    {
        // Slope = 0.1 → effective threshold at lead 120 = 1.5 + 0.1 × 4 = 1.9.
        // MAE ratio 1.7 fires at lead 24 (threshold stays 1.5) but is now
        // BELOW threshold at lead 120 (1.7 < 1.9). Tests that the per-lead
        // relaxation works on the long-lead end without affecting lead 24.
        var p24 = Enumerable.Range(0, 10).Select(i =>
            MakePrediction(V1, 24, AsOf.AddDays(-7).AddHours(-i), blend: 11.7)).ToArray();
        var p120 = Enumerable.Range(0, 10).Select(i =>
            MakePrediction(V1, 120, AsOf.AddDays(-7).AddHours(-i - 100), blend: 11.7)).ToArray();
        var truth = p24.Concat(p120).ToDictionary(p => p.ValidTimeUtc, _ => 10.0);

        var metadata = new Dictionary<string, ModelArtifact.TrainingMetadata>
        {
            [V1] = new()
            {
                Phase = "2b",
                PerLead =
                {
                    ["24"]  = new() { LeadHours = 24,  BlendTestMae = 1.0 },
                    ["120"] = new() { LeadHours = 120, BlendTestMae = 1.0 },
                }
            }
        };

        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = p24.Concat(p120).ToArray(),
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
            DriftThresholdSlopePer24h = 0.1,   // production default
            MinDriftN = 1,
        });

        rows.Single(r => r.LeadHours == 24).DriftFlag.Should().BeTrue();    // 1.7 > 1.5
        rows.Single(r => r.LeadHours == 120).DriftFlag.Should().BeFalse();  // 1.7 < 1.9 (relaxed)
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

    // -----------------------------------------------------------------------
    // ComputeActualLeadBuckets — the 6h actual-lead bucket view added 2026-05-05
    // alongside the trained-lead view. Same input shape, different grouping
    // (by ValidTime − PredictionMadeAt, floored to nearest 6h bucket).
    // -----------------------------------------------------------------------

    [Fact]
    public void ComputeActualLeadBuckets_groups_by_actual_lead_in_6h_buckets()
    {
        // Two predictions at the same trained lead (24) but different
        // PredictionMadeAt times → different actual leads → different buckets.
        // Pred A made 24h before V → actual lead 24h → bucket [24, 30).
        // Pred B made 30h before V → actual lead 30h → bucket [30, 36).
        var v1 = AsOf.AddDays(-7).AddHours(0); // inside window
        var predA = MakePrediction(V1, leadHours: 24, valid: v1, blend: 10.0);
        var predB = new TempPredictionRow
        {
            LocationName = "Bonehill Rocks",
            ModelVersion = V1,
            PredictionMadeAtUtc = v1.AddHours(-30), // 30h before valid
            ValidTimeUtc = v1,
            LeadHours = 24, // trained-lead bucket label, not actual lead
            BlendTemperature = 10.5,
            FeatureVectorHash = "fakehash",
        };
        var truth = new Dictionary<DateTime, double> { [v1] = 10.0 };

        var rows = TempVerifier.ComputeActualLeadBuckets(BaseInputs(new[] { predA, predB }, truth));

        // Two distinct buckets: lead 24 (pred A) and lead 30 (pred B).
        rows.Should().HaveCount(2);
        rows.Select(r => r.ActualLeadBucketLowH).Should().BeEquivalentTo(new int?[] { 24, 30 });
    }

    [Fact]
    public void ComputeActualLeadBuckets_each_row_carries_bucket_low_in_LeadHours()
    {
        // Bucket rows reuse LeadHours field to carry the bucket lower bound
        // (so consumers keying on LeadHours don't need a special-case for
        // bucket rows). Sanity-check that.
        var v = AsOf.AddDays(-7);
        var pred = MakePrediction(V1, leadHours: 24, valid: v, blend: 10.0);
        var truth = new Dictionary<DateTime, double> { [v] = 10.1 };

        var rows = TempVerifier.ComputeActualLeadBuckets(BaseInputs(new[] { pred }, truth));

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.LeadHours.Should().Be(r.ActualLeadBucketLowH);
        r.LeadHours.Should().Be(24); // 24h actual lead → bucket [24, 30)
    }

    [Fact]
    public void ComputeActualLeadBuckets_no_persistence_no_drift_on_bucket_rows()
    {
        // Bucket rows have no per-actual-lead training-time baseline to
        // compare against, so PersistenceMae and ReferenceTestMae should be
        // null and DriftFlag should be false.
        var v = AsOf.AddDays(-7);
        var pred = MakePrediction(V1, leadHours: 24, valid: v, blend: 10.0);
        var truth = new Dictionary<DateTime, double> { [v] = 10.1 };

        var rows = TempVerifier.ComputeActualLeadBuckets(BaseInputs(new[] { pred }, truth));

        rows[0].PersistenceMae.Should().BeNull();
        rows[0].ReferenceTestMae.Should().BeNull();
        rows[0].DriftFlag.Should().BeFalse();
    }

    [Fact]
    public void ComputeActualLeadBuckets_preserves_blend_mae_relative_to_trained_view()
    {
        // For a single prediction at strict actual lead 24h, the bucket-view
        // row's BlendMae should equal what the trained-view row reports
        // (since both score the same prediction the same way).
        var v = AsOf.AddDays(-7);
        var pred = MakePrediction(V1, leadHours: 24, valid: v, blend: 10.5);
        var truth = new Dictionary<DateTime, double> { [v] = 10.0 };

        var trainedRows = TempVerifier.Compute(BaseInputs(new[] { pred }, truth));
        var bucketRows  = TempVerifier.ComputeActualLeadBuckets(BaseInputs(new[] { pred }, truth));

        trainedRows.Should().HaveCount(1);
        bucketRows.Should().HaveCount(1);
        bucketRows[0].BlendMae.Should().BeApproximately(trainedRows[0].BlendMae, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Phase 2d (exact-runtime) — best-single picker dispatches by phase.
    // 2b/2c rows score from temp_gfs / temp_ecmwf / ... slots; 2d rows
    // score from the new TempXxxExact slots. The verifier's MetadataByVersion
    // is the source of truth for "what phase produced this row".
    // -----------------------------------------------------------------------

    [Fact]
    public void Compute_uses_exact_per_model_columns_for_phase_2d()
    {
        const string V2d = "v2026-05-05_120000_phase2d";
        var valid = AsOf.AddDays(-7);
        // 2d row: offset_day TempGfs / TempEcmwf / ... null; *Exact slots populated.
        // Truth = 10.0; AIFS-exact closest at 10.05; that's the best-single name.
        var pred = new TempPredictionRow
        {
            LocationName = "Bonehill Rocks",
            ModelVersion = V2d,
            PredictionMadeAtUtc = valid.AddHours(-12),
            ValidTimeUtc = valid,
            LeadHours = 12,
            BlendTemperature = 10.1,
            TempGfsExact      = 11.0,
            TempIfsOperExact  = 10.5,
            TempAifsOperExact = 10.05,
            TempMoGlobalExact = 11.5,
            TempUkvExact      = 10.3,
            FeatureVectorHash = "fakehash",
        };
        var truth = new Dictionary<DateTime, double> { [valid] = 10.0 };

        var inputs = new TempVerifier.Inputs
        {
            Predictions = new[] { pred },
            TruthByTime = truth,
            MetadataByVersion = new Dictionary<string, ModelArtifact.TrainingMetadata>
            {
                [V2d] = new() { Version = V2d, Target = "temperature", Phase = "2d" },
            },
            AsOfUtc = AsOf,
            WindowDays = 14,
            Era5LatencyDays = 5,
            DriftThreshold = 1.5,
        };

        var rows = TempVerifier.Compute(inputs);

        rows.Should().ContainSingle();
        rows[0].BestSingleName.Should().Be("temp_aifs_oper_exact",
            "phase 2d should pick best-single from the exact-runtime per-model slots");
    }

    [Fact]
    public void ModelNamesForPhase_returns_distinct_lists_for_2b_vs_2d()
    {
        TempVerifier.ModelNamesForPhase("2b").Should().BeEquivalentTo(
            "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem", "temp_aifs");
        TempVerifier.ModelNamesForPhase("2c").Should().BeEquivalentTo(
            TempVerifier.ModelNamesForPhase("2b"),
            "rich (2c) shares the offset_day per-model slot list with lean (2b)");
        TempVerifier.ModelNamesForPhase("2d").Should().BeEquivalentTo(
            "temp_gfs_exact", "temp_ifs_oper_exact", "temp_aifs_oper_exact",
            "temp_moglobal_exact", "temp_ukv_exact");
        // Unknown / null phase falls back to offset_day list — matches every
        // shipped phase pre-2d so legacy artefacts keep verifying correctly.
        TempVerifier.ModelNamesForPhase(null).Should().BeEquivalentTo(
            TempVerifier.ModelNamesForPhase("2b"));
        TempVerifier.ModelNamesForPhase("nonsense").Should().BeEquivalentTo(
            TempVerifier.ModelNamesForPhase("2b"));
    }
}
