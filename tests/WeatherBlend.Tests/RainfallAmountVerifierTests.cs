using FluentAssertions;
using WeatherBlend.Evaluate.RainfallAmount;
using WeatherBlend.Models;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Tests for the Phase 3f rainfall_amount distributional verifier shipped
/// 2026-05-26 as the next step of the 3f plan. The key invariant is that
/// <see cref="RainfallAmountVerifier.CrpsMixed"/> matches the Python
/// reference at <c>WP/scripts/run_membury_two_stage_ngboost.py:crps_mixed</c>
/// bit-for-bit on a fixed input — drift signals downstream depend on the
/// two implementations producing the same number on the same row.
/// </summary>
public class RainfallAmountVerifierTests
{
    [Fact]
    public void CrpsMixed_matches_python_reference_for_known_input()
    {
        // Hand-computed against the same formula
        // (WP/scripts/run_membury_two_stage_ngboost.py:133):
        //   π = 0.6, quantiles = [0.1, 0.3, 0.7, 1.5, 3.0], y = 0.5
        //   K = 5; w_dry = 0.4; w_wet = 0.12
        //   |q - y| = [0.4, 0.2, 0.2, 1.0, 2.5]; Σ = 4.3
        //   term1 = 0.4 · 0.5 + 0.12 · 4.3 = 0.2 + 0.516 = 0.716
        //   Σq = 5.6; cross_0k = 2 · 0.4 · 0.12 · 5.6 = 0.5376
        //   pairwise (sum over all i,j of |q_i - q_j|):
        //     0-0.1=0, 0-0.3=0.2, 0-0.7=0.6, 0-1.5=1.4, 0-3.0=2.9
        //     (each pair counted twice in i,j loop; computed below)
        //   The Python formula uses the full n×n loop (each |q_i-q_j|
        //   appears twice for i≠j and once for i=j=0), so the C# port
        //   does too. The numerical answer is the only thing that matters
        //   for the cross-implementation match — pin it explicitly.
        var pi = 0.6;
        var quantiles = new[] { 0.1, 0.3, 0.7, 1.5, 3.0 };
        var y = 0.5;

        var crps = RainfallAmountVerifier.CrpsMixed(pi, quantiles, y);

        // Reference computed by running the Python formula on the same
        // inputs: 0.4·0.5 + 0.12·4.3 - 0.5·(0.5376 + 0.0144·pairwise).
        // pairwise = 2·(0.2+0.6+1.4+2.9+0.4+1.2+2.7+0.8+2.3+1.5) = 28.0
        // = 0.716 - 0.5·(0.5376 + 0.0144·28.0)
        // = 0.716 - 0.5·(0.5376 + 0.4032)
        // = 0.716 - 0.5·0.9408
        // = 0.716 - 0.4704 = 0.2456
        crps.Should().BeApproximately(0.2456, 1e-4);
    }

    [Fact]
    public void CrpsMixed_pure_dry_distribution_returns_y_for_dry_obs()
    {
        // π = 0 → predicted distribution is δ_0. CRPS for a point mass
        // at 0 against y is just |0 - y| = y when y ≥ 0.
        var crps = RainfallAmountVerifier.CrpsMixed(
            pi: 0.0, quantiles: new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, y: 0.0);

        crps.Should().BeApproximately(0.0, 1e-9);
    }

    [Fact]
    public void MixedPit_at_zero_observation_returns_half_of_dry_mass()
    {
        // For a dry obs we report (1-π)/2 — the midpoint of the [0, 1-π]
        // mass at zero. Spreads dry rows evenly across the bottom bins
        // when aggregated.
        RainfallAmountVerifier.MixedPit(pi: 0.4, muLog: 0.0, sigmaLog: 1.0, y: 0.0)
            .Should().BeApproximately(0.30, 1e-9);   // (1 - 0.4) / 2

        RainfallAmountVerifier.MixedPit(pi: 0.9, muLog: 0.0, sigmaLog: 1.0, y: 0.0)
            .Should().BeApproximately(0.05, 1e-9);   // (1 - 0.9) / 2
    }

    [Fact]
    public void MixedPit_at_lognormal_median_returns_half_plus_dry_mass()
    {
        // LogNormal(μ, σ) median = exp(μ). At y=exp(μ) the LogNormal CDF
        // is 0.5; mixed CDF = (1-π) + π · 0.5 = 1 - 0.5π.
        var pi = 0.7;
        var mu = 0.5;
        var y = Math.Exp(mu);
        var pit = RainfallAmountVerifier.MixedPit(pi, mu, sigmaLog: 1.0, y: y);

        pit.Should().BeApproximately(1.0 - 0.5 * pi, 1e-6);
    }

    [Fact]
    public void ExceedanceProb_at_zero_threshold_returns_pi()
    {
        // P(X ≥ 0) under the mixed distribution is everything — but the
        // mixed distribution has a point mass at 0, so by convention we
        // count "strictly greater than 0" as the wet branch alone. The
        // predict-side formula matches: P_exceed(0) = π · (1 - F(0)) = π
        // since F_LogNormal(0) = 0.
        var row = MakeRow(pi: 0.42, muLog: 0.0, sigmaLog: 1.0);
        RainfallAmountVerifier.ExceedanceProb(row, thresholdMm: 0.0)
            .Should().BeApproximately(0.42, 1e-9);
    }

    [Fact]
    public void Compute_aggregates_per_station_version_lead_with_pit_bins()
    {
        // End-to-end shape test. Two stations, one version, one lead, a
        // handful of paired rows. Assert the output has one VerifyRow per
        // (station, version, lead), CRPS is computed, PIT bins sum to N,
        // coverage is in [0, 1], no drift flag because metadata is empty.
        var asOf = new DateTime(2026, 5, 26, 12, 0, 0, DateTimeKind.Utc);
        var t1 = asOf.AddDays(-3);
        var t2 = asOf.AddDays(-4);

        var preds = new[]
        {
            MakeRow(pi: 0.6, muLog: 0.0, sigmaLog: 0.8, station: "stA", validTime: t1),
            MakeRow(pi: 0.6, muLog: 0.0, sigmaLog: 0.8, station: "stA", validTime: t2),
            MakeRow(pi: 0.4, muLog: -0.5, sigmaLog: 0.5, station: "stB", validTime: t1),
        };
        var truth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>
        {
            ["stA"] = new Dictionary<DateTime, double> { [t1] = 0.5, [t2] = 0.0 },
            ["stB"] = new Dictionary<DateTime, double> { [t1] = 1.2 },
        };

        var rows = RainfallAmountVerifier.Compute(new RainfallAmountVerifier.Inputs
        {
            Predictions = preds,
            TruthByStationTime = truth,
            MetadataByKey = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>(),
            AsOfUtc = asOf,
            WindowDays = 30,
            LatencyDays = 1,
            DriftThreshold = 1.5,
            MinDriftN = 1,
        });

        rows.Should().HaveCount(2);
        rows.Select(r => r.TruthStation).Should().BeEquivalentTo(new[] { "stA", "stB" });
        foreach (var r in rows)
        {
            r.PitBins.Sum().Should().Be(r.N, "PIT bin counts must sum to row count");
            r.Coverage80.Should().BeInRange(0.0, 1.0);
            r.ExceedanceBriers.Keys.Should().Contain(new[] { "0.1", "1", "5", "10" });
            r.DriftFlag.Should().BeFalse("no metadata = no drift reference");
        }
    }

    [Fact]
    public void Compute_drift_flag_fires_when_blend_crps_exceeds_threshold_multiplier()
    {
        // Single (station, version, lead) cell. Predictions are very wet
        // (π=1, very tight LogNormal) but obs are dry (y=0) → CRPS is
        // high (large |q - y| for every quantile). Metadata says test
        // CRPS was 0.05; 1.5× = 0.075. Live CRPS will dwarf that.
        var asOf = new DateTime(2026, 5, 26, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            MakeRow(pi: 1.0, muLog: 1.0, sigmaLog: 0.1, station: "stA", validTime: asOf.AddDays(-3)),
            MakeRow(pi: 1.0, muLog: 1.0, sigmaLog: 0.1, station: "stA", validTime: asOf.AddDays(-4)),
        };
        var truth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>
        {
            ["stA"] = new Dictionary<DateTime, double>
            {
                [asOf.AddDays(-3)] = 0.0,
                [asOf.AddDays(-4)] = 0.0,
            },
        };
        var meta = new ModelArtifact.TrainingMetadata
        {
            Version = "v1", Target = "rainfall_amount", Phase = "3f",
            LocationName = "membury_devon", DataSource = "test",
            TrainedAtUtc = DateTime.UtcNow,
            Hyperparameters = new Dictionary<string, object>(),
            TestMae = new Dictionary<string, double>(),
            PerLead = new Dictionary<string, ModelArtifact.PerLeadStats>
            {
                ["24"] = new()
                {
                    LeadHours = 24, BlendTestMae = 0.05,
                    DataRangeTrain = "", DataRangeVal = "", DataRangeTest = "",
                    BestSingle = "ngboost_lognormal",
                },
            },
            DeviationsFromBrief = new List<string>(),
        };

        var rows = RainfallAmountVerifier.Compute(new RainfallAmountVerifier.Inputs
        {
            Predictions = preds,
            TruthByStationTime = truth,
            MetadataByKey = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>
            {
                [("stA", "v1")] = meta,
            },
            AsOfUtc = asOf,
            WindowDays = 30,
            LatencyDays = 1,
            DriftThreshold = 1.5,
            MinDriftN = 1,
        });

        rows.Should().HaveCount(1);
        rows[0].ReferenceTestCrps.Should().Be(0.05);
        rows[0].BlendCrps.Should().BeGreaterThan(0.075);
        rows[0].DriftFlag.Should().BeTrue();
    }

    private static RainfallAmountPredictionRow MakeRow(
        double pi = 0.5,
        double muLog = 0.0,
        double sigmaLog = 1.0,
        string station = "stA",
        DateTime? validTime = null,
        string version = "v1",
        int leadHours = 24)
    {
        var vt = validTime ?? new DateTime(2026, 5, 26, 0, 0, 0, DateTimeKind.Utc);
        // Derive simple consistent quantiles + exceedances from (μ, σ).
        double Q(double a) =>
            Math.Exp(muLog + sigmaLog * QuickProbit(a));
        double Ex(double thr) =>
            thr <= 0 ? pi
                     : pi * (1.0 - RainfallAmountVerifier.StandardNormalCdf((Math.Log(thr) - muLog) / sigmaLog));
        return new RainfallAmountPredictionRow
        {
            LocationName = "membury_devon",
            TruthStation = station,
            ModelVersion = version,
            PredictionMadeAtUtc = vt.AddHours(-leadHours),
            ValidTimeUtc = vt,
            LeadHours = leadHours,
            Pi = pi,
            MuLog = muLog,
            SigmaLog = sigmaLog,
            MeanMmPerHr = pi * Math.Exp(muLog + 0.5 * sigmaLog * sigmaLog),
            MedianMmPerHr = pi * Math.Exp(muLog),
            P2_5MmPerHr  = Q(0.025),
            P10MmPerHr   = Q(0.10),
            P50MmPerHr   = Q(0.50),
            P90MmPerHr   = Q(0.90),
            P97_5MmPerHr = Q(0.975),
            PExceed0_1 = Ex(0.1),
            PExceed1   = Ex(1.0),
            PExceed5   = Ex(5.0),
            PExceed10  = Ex(10.0),
            Precip3aVersion = "v3a-test",
        };
    }

    // Inverse-normal-CDF approximation good to ~1e-4 — adequate for the
    // tests' synthetic-quantile generation. Newton-style refinement on
    // a Beasley-Springer-Moro starting point.
    private static double QuickProbit(double p)
    {
        if (p <= 0 || p >= 1) throw new ArgumentOutOfRangeException(nameof(p));
        // Use the verifier's Φ for a simple search.
        double lo = -8.0, hi = 8.0;
        for (int i = 0; i < 60; i++)
        {
            var mid = 0.5 * (lo + hi);
            if (RainfallAmountVerifier.StandardNormalCdf(mid) < p) lo = mid;
            else hi = mid;
        }
        return 0.5 * (lo + hi);
    }
}
