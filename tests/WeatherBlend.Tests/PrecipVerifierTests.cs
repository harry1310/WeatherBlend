using FluentAssertions;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipVerifierTests
{
    private static readonly DateTime AsOf = new(2026, 4, 22, 0, 0, 0, DateTimeKind.Utc);
    private const string Station = "ea_bellever_dartmoor";
    private const string V1 = "v2026-04-20_100000";

    [Fact]
    public void Compute_excludes_predictions_outside_rolling_window_and_without_truth()
    {
        var inside  = MakePrediction(valid: AsOf.AddDays(-10), probWet: 0.6, clim: 0.3);
        var tooOld  = MakePrediction(valid: AsOf.AddDays(-40), probWet: 0.6, clim: 0.3);
        var tooNew  = MakePrediction(valid: AsOf.AddDays(-2),  probWet: 0.6, clim: 0.3);
        var inWinButNoTruth = MakePrediction(valid: AsOf.AddDays(-11), probWet: 0.6, clim: 0.3);

        var truth = Truth(new Dictionary<DateTime, double>
        {
            [inside.ValidTimeUtc] = 0.5,
            [tooOld.ValidTimeUtc] = 0.5,
            [tooNew.ValidTimeUtc] = 0.5,
            // inWinButNoTruth intentionally absent from truth map
        });

        var rows = PrecipVerifier.Compute(BaseInputs(
            new[] { inside, tooOld, tooNew, inWinButNoTruth }, truth));

        rows.Should().ContainSingle();
        rows[0].N.Should().Be(1);
    }

    [Fact]
    public void Compute_brier_matches_hand_calculation()
    {
        // Four predictions: [0.8, 0.3, 0.9, 0.1] vs truth-mm [0.5, 0, 0, 0.2]
        // → binary truth [1, 0, 0, 1]
        // Squared errors: (0.8-1)^2=0.04, (0.3-0)^2=0.09, (0.9-0)^2=0.81, (0.1-1)^2=0.81
        // Brier = (0.04 + 0.09 + 0.81 + 0.81) / 4 = 0.4375
        var preds = new[]
        {
            MakePrediction(AsOf.AddDays(-7),  probWet: 0.8, clim: 0.25),
            MakePrediction(AsOf.AddDays(-8),  probWet: 0.3, clim: 0.25),
            MakePrediction(AsOf.AddDays(-9),  probWet: 0.9, clim: 0.25),
            MakePrediction(AsOf.AddDays(-10), probWet: 0.1, clim: 0.25),
        };
        var truth = Truth(new Dictionary<DateTime, double>
        {
            [preds[0].ValidTimeUtc] = 0.5,
            [preds[1].ValidTimeUtc] = 0.0,
            [preds[2].ValidTimeUtc] = 0.0,
            [preds[3].ValidTimeUtc] = 0.2,
        });

        var rows = PrecipVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].BlendBrier.Should().BeApproximately(0.4375, 1e-9);
        rows[0].WetN.Should().Be(2);
        rows[0].WetRate.Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void Compute_bss_uses_climatology_column_as_reference()
    {
        // Pre-compute expected values: blend Brier vs clim Brier = 1 − blend/clim.
        var preds = new[]
        {
            MakePrediction(AsOf.AddDays(-7), probWet: 0.9, clim: 0.25),
            MakePrediction(AsOf.AddDays(-8), probWet: 0.1, clim: 0.25),
        };
        var truth = Truth(new Dictionary<DateTime, double>
        {
            [preds[0].ValidTimeUtc] = 0.5, // wet → binary 1
            [preds[1].ValidTimeUtc] = 0.0, // dry → binary 0
        });

        var rows = PrecipVerifier.Compute(BaseInputs(preds, truth));

        // Blend: ((0.9-1)^2 + (0.1-0)^2)/2 = (0.01 + 0.01)/2 = 0.01
        // Clim:  ((0.25-1)^2 + (0.25-0)^2)/2 = (0.5625 + 0.0625)/2 = 0.3125
        // BSS = 1 − 0.01/0.3125 = 0.968
        rows[0].BlendBrier.Should().BeApproximately(0.01, 1e-9);
        rows[0].ClimBrier.Should().BeApproximately(0.3125, 1e-9);
        rows[0].Bss.Should().BeApproximately(1.0 - 0.01 / 0.3125, 1e-9);
    }

    [Fact]
    public void Compute_stratifies_by_station_and_version_and_lead()
    {
        var preds = new[]
        {
            MakePrediction(AsOf.AddDays(-7), probWet: 0.5, clim: 0.25, station: "ea_bellever_dartmoor", version: V1, lead: 24),
            MakePrediction(AsOf.AddDays(-7), probWet: 0.5, clim: 0.25, station: "ea_princetown",        version: V1, lead: 24),
            MakePrediction(AsOf.AddDays(-8), probWet: 0.5, clim: 0.25, station: "ea_bellever_dartmoor", version: V1, lead: 48),
        };
        var truth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>
        {
            ["ea_bellever_dartmoor"] = new Dictionary<DateTime, double>
            {
                [preds[0].ValidTimeUtc] = 0.5,
                [preds[2].ValidTimeUtc] = 0.0,
            },
            ["ea_princetown"] = new Dictionary<DateTime, double>
            {
                [preds[1].ValidTimeUtc] = 0.0,
            },
        };

        var rows = PrecipVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().HaveCount(3);
        rows.Select(r => (r.TruthStation, r.ModelVersion, r.LeadHours))
            .Should().Equal(
                ("ea_bellever_dartmoor", V1, 24),
                ("ea_bellever_dartmoor", V1, 48),
                ("ea_princetown",        V1, 24));
    }

    [Fact]
    public void Compute_flags_drift_when_blend_brier_exceeds_threshold_times_training_brier()
    {
        // Blend always p=0.5, truth all wet → Brier = (0.5-1)^2 = 0.25.
        // Reference Brier = 0.10, drift threshold 1.5 → 0.25 > 0.15 → drift.
        var preds = Enumerable.Range(0, 5)
            .Select(i => MakePrediction(AsOf.AddDays(-7 - i), probWet: 0.5, clim: 0.25))
            .ToArray();
        var truth = Truth(preds.ToDictionary(p => p.ValidTimeUtc, _ => 0.5));

        var metadata = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>
        {
            [(Station, V1)] = MetadataWithLead(24, blendTestBrier: 0.10),
        };

        var rows = PrecipVerifier.Compute(new PrecipVerifier.Inputs
        {
            Predictions = preds,
            TruthByStationTime = truth,
            MetadataByKey = metadata,
            AsOfUtc = AsOf,
            WindowDays = 30,
            LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().ContainSingle();
        rows[0].BlendBrier.Should().BeApproximately(0.25, 1e-9);
        rows[0].ReferenceTestBrier.Should().BeApproximately(0.10, 1e-9);
        rows[0].DriftFlag.Should().BeTrue();
    }

    [Fact]
    public void Compute_leaves_reference_blank_and_drift_false_when_metadata_missing()
    {
        var preds = new[] { MakePrediction(AsOf.AddDays(-7), probWet: 0.5, clim: 0.25) };
        var truth = Truth(preds.ToDictionary(p => p.ValidTimeUtc, _ => 0.5));

        var rows = PrecipVerifier.Compute(new PrecipVerifier.Inputs
        {
            Predictions = preds,
            TruthByStationTime = truth,
            MetadataByKey = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>(),
            AsOfUtc = AsOf,
            WindowDays = 30,
            LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().ContainSingle();
        rows[0].ReferenceTestBrier.Should().BeNull();
        rows[0].DriftFlag.Should().BeFalse();
    }

    [Fact]
    public void Compute_picks_best_single_by_window_brier()
    {
        // Truth: wet at t1, dry at t2 → binary [1, 0].
        // GFS always predicts wet (1,1) → Brier = 0.5 (one miss)
        // ECMWF perfectly aligned (1,0) → Brier = 0.0
        // Others tied at 0.5.
        var preds = new[]
        {
            WithPerModelIndicators(MakePrediction(AsOf.AddDays(-7), probWet: 0.5, clim: 0.25),
                gfs: 1.0, ecmwf: 1.0, icon: 0.0, mf: 0.0, ukmo: 0.0, gem: 0.0),
            WithPerModelIndicators(MakePrediction(AsOf.AddDays(-8), probWet: 0.5, clim: 0.25),
                gfs: 1.0, ecmwf: 0.0, icon: 1.0, mf: 1.0, ukmo: 1.0, gem: 1.0),
        };
        var truth = Truth(new Dictionary<DateTime, double>
        {
            [preds[0].ValidTimeUtc] = 0.5,
            [preds[1].ValidTimeUtc] = 0.0,
        });

        var rows = PrecipVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].BestSingleName.Should().Be("ecmwf");
        rows[0].BestSingleBrier.Should().BeApproximately(0.0, 1e-9);
    }

    [Fact]
    public void Compute_persistence_uses_truth_at_t_minus_lead_and_counts_drops()
    {
        var valid1 = AsOf.AddDays(-7);
        var valid2 = AsOf.AddDays(-8);
        var preds = new[]
        {
            MakePrediction(valid1, probWet: 0.5, clim: 0.25),
            MakePrediction(valid2, probWet: 0.5, clim: 0.25),
        };
        var truth = Truth(new Dictionary<DateTime, double>
        {
            [valid1] = 0.5, // binary 1
            [valid2] = 0.0, // binary 0
            [valid1.AddHours(-24)] = 0.5, // persistence says 1 → matches truth 1 → err 0
            // valid2 - 24h intentionally missing → dropped
        });

        var rows = PrecipVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].PersistenceDropped.Should().Be(1);
        rows[0].PersistenceBrier.Should().BeApproximately(0.0, 1e-9);
    }

    [Fact]
    public void Compute_frequency_bias_at_05_threshold()
    {
        // 4 preds → 3 forecast-wet at p>=0.5, truth wet in 2 → bias = 3/2 = 1.5
        var preds = new[]
        {
            MakePrediction(AsOf.AddDays(-7),  probWet: 0.9, clim: 0.25),
            MakePrediction(AsOf.AddDays(-8),  probWet: 0.6, clim: 0.25),
            MakePrediction(AsOf.AddDays(-9),  probWet: 0.7, clim: 0.25),
            MakePrediction(AsOf.AddDays(-10), probWet: 0.1, clim: 0.25),
        };
        var truth = Truth(new Dictionary<DateTime, double>
        {
            [preds[0].ValidTimeUtc] = 0.5,
            [preds[1].ValidTimeUtc] = 0.0,
            [preds[2].ValidTimeUtc] = 0.5,
            [preds[3].ValidTimeUtc] = 0.0,
        });

        var rows = PrecipVerifier.Compute(BaseInputs(preds, truth));

        rows.Should().ContainSingle();
        rows[0].FrequencyBiasAt05.Should().BeApproximately(1.5, 1e-9);
    }

    [Fact]
    public void Compute_returns_empty_when_no_predictions_survive_filters()
    {
        var rows = PrecipVerifier.Compute(new PrecipVerifier.Inputs
        {
            Predictions = Array.Empty<PrecipPredictionRow>(),
            TruthByStationTime = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
            MetadataByKey = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>(),
            AsOfUtc = AsOf,
            WindowDays = 30,
            LatencyDays = 5,
            DriftThreshold = 1.5,
        });

        rows.Should().BeEmpty();
    }

    // ---------- helpers ----------

    private static PrecipVerifier.Inputs BaseInputs(
        IEnumerable<PrecipPredictionRow> preds,
        IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> truth)
        => new()
        {
            Predictions = preds.ToList(),
            TruthByStationTime = truth,
            MetadataByKey = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>(),
            AsOfUtc = AsOf,
            WindowDays = 30,
            LatencyDays = 5,
            DriftThreshold = 1.5,
        };

    private static IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> Truth(
        IReadOnlyDictionary<DateTime, double> byTime)
        => new Dictionary<string, IReadOnlyDictionary<DateTime, double>>
        {
            [Station] = byTime,
        };

    private static PrecipPredictionRow MakePrediction(
        DateTime valid, double probWet, double clim,
        string station = Station, string version = V1, int lead = 24)
        => new()
        {
            LocationName = "Bonehill Rocks",
            TruthStation = station,
            ModelVersion = version,
            PredictionMadeAtUtc = valid.AddHours(-lead),
            ValidTimeUtc = valid,
            LeadHours = lead,
            ProbWet = probWet,
            ClimatologyPWet = clim,
            PrecipAgreementWet01 = 0.5,
            FeatureVectorHash = "fakehash",
        };

    private static PrecipPredictionRow WithPerModelIndicators(
        PrecipPredictionRow row,
        double gfs, double ecmwf, double icon, double mf, double ukmo, double gem)
        => new()
        {
            LocationName = row.LocationName,
            TruthStation = row.TruthStation,
            ModelVersion = row.ModelVersion,
            PredictionMadeAtUtc = row.PredictionMadeAtUtc,
            ValidTimeUtc = row.ValidTimeUtc,
            LeadHours = row.LeadHours,
            ProbWet = row.ProbWet,
            ClimatologyPWet = row.ClimatologyPWet,
            PrecipGfs   = gfs   >= 0.5 ? 1.0 : 0.0,
            PrecipEcmwf = ecmwf >= 0.5 ? 1.0 : 0.0,
            PrecipIcon  = icon  >= 0.5 ? 1.0 : 0.0,
            PrecipMf    = mf    >= 0.5 ? 1.0 : 0.0,
            PrecipUkmo  = ukmo  >= 0.5 ? 1.0 : 0.0,
            PrecipGem   = gem   >= 0.5 ? 1.0 : 0.0,
            PrecipAgreementWet01 = row.PrecipAgreementWet01,
            FeatureVectorHash = row.FeatureVectorHash,
        };

    private static ModelArtifact.TrainingMetadata MetadataWithLead(int leadHours, double blendTestBrier)
        => new()
        {
            Version = V1,
            Target = "precipitation",
            Phase = "3a",
            DataSource = "previous_runs_api",
            TrainedAtUtc = AsOf.AddDays(-2),
            PerLead = new Dictionary<string, ModelArtifact.PerLeadStats>
            {
                [leadHours.ToString()] = new()
                {
                    LeadHours = leadHours,
                    BlendTestMae = blendTestBrier,
                    BestSingle = "ecmwf",
                },
            },
        };

    // ---- Phase 3d (exact-runtime) per-phase ModelNames tests -----------------

    [Fact]
    public void ModelNamesForPhase_returns_distinct_lists_for_3a_vs_3d()
    {
        PrecipVerifier.ModelNamesForPhase("3a")
            .Should().BeEquivalentTo("gfs", "ecmwf", "icon", "mf", "ukmo", "gem", "aifs", "jma");
        PrecipVerifier.ModelNamesForPhase("3c")
            .Should().BeEquivalentTo(PrecipVerifier.ModelNamesForPhase("3a"),
                "rich (3c) shares the offset_day per-model slot list with lean (3a)");
        PrecipVerifier.ModelNamesForPhase("3d")
            .Should().BeEquivalentTo(
                "gfs_exact", "ifs_oper_exact", "aifs_oper_exact", "moglobal_exact", "ukv_exact");
        // Unknown / null phase falls back to offset_day list — preserves
        // behaviour for legacy artefacts predating per-phase ModelNames.
        PrecipVerifier.ModelNamesForPhase(null)
            .Should().BeEquivalentTo(PrecipVerifier.ModelNamesForPhase("3a"));
        PrecipVerifier.ModelNamesForPhase("nonsense")
            .Should().BeEquivalentTo(PrecipVerifier.ModelNamesForPhase("3a"));
    }
}
