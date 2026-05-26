using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Evaluate.RainfallAmount;

/// <summary>
/// Pure verification for Phase 3f (rainfall_amount). 3f emits a mixed
/// predictive distribution
///
///   F(x) = (1 − π) · δ_0(x) + π · LogNormal(μ_log, σ_log)(x)
///
/// per (valid_time, lead). Verify scores it against EA hourly rainfall
/// truth and emits the following metrics per (truth station, model
/// version, lead):
///
///   * <b>CRPS</b> (primary) — mean continuous ranked probability score
///     using the quantile-based estimator from
///     <c>WP/scripts/run_membury_two_stage_ngboost.py:crps_mixed</c>.
///     Lower is better. Same direction + scale as Brier.
///   * <b>MAE_wet</b> — mean |median − observed| restricted to wet
///     observations. Robust point-skill check.
///   * <b>Coverage80</b> — fraction of obs in [P10, P90]. Calibration
///     target is 0.80; departures flag under/over-spread.
///   * <b>PIT</b> mean + 10-bin histogram. Per-row PIT for a mixed
///     distribution is
///       y = 0:   uniform on [0, 1-π]    (dry mass spans the bottom)
///       y &gt; 0:   (1-π) + π · Φ((ln y - μ)/σ)
///     A flat histogram = well-calibrated; bumps at 0/1 = under/over-spread.
///   * <b>Exceedance Brier</b> per threshold (0.1 / 1 / 5 / 10 mm/h) —
///     calibration of the binary exceedance forecasts the
///     rainfall_amount card surfaces.
///
/// Drift flag fires when rolling blend CRPS exceeds <c>driftThreshold ×
/// training_metadata.PerLeadStats[lead].BlendTestMae</c> — same shape
/// as the precip/temp verify. <c>BlendTestMae</c> is repurposed to
/// hold test-set CRPS for 3f (train_3f.py stamps it post-training);
/// keeps the verify-history schema single-pattern at the cost of a
/// slightly misleading field name on the bundle.
///
/// Inputs are pre-joined in-memory; no DuckDB here so every metric path is
/// unit-testable. The Python <c>crps_mixed</c> formula is reproduced
/// verbatim in <see cref="CrpsMixed"/> and pinned by a unit test that
/// compares against a hand-computed reference.
/// </summary>
public static class RainfallAmountVerifier
{
    /// <summary>EA hourly rainfall reading at or above this threshold counts
    /// the hour as wet for MAE_wet stratification. Matches the precip
    /// blender's wet definition so verify cross-checks line up.</summary>
    public const double WetThresholdMm = 0.1;

    /// <summary>Number of PIT histogram bins. 10 equal-width bins on [0, 1]
    /// is the canonical default for human-readable calibration plots.</summary>
    public const int PitBinCount = 10;

    /// <summary>Exceedance thresholds (mm/h) the row schema carries fields for.
    /// The verifier surfaces a Brier per threshold; the order here matches the
    /// row's P_Exceed_* columns so reporters can iterate in one shot.</summary>
    public static readonly double[] ExceedanceThresholdsMm = { 0.1, 1.0, 5.0, 10.0 };

    public sealed class Inputs
    {
        public required IReadOnlyList<RainfallAmountPredictionRow> Predictions { get; init; }

        /// <summary>Hourly truth mm/hour keyed by (TruthStation slug, ValidTimeUtc).
        /// Same shape PrecipVerifier consumes — reuse the same truth loader.</summary>
        public required IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> TruthByStationTime { get; init; }

        /// <summary>One entry per distinct (TruthStation, ModelVersion) that appears
        /// in Predictions. <c>BlendTestMae</c> on each metadata is interpreted as
        /// the bundle's test-set CRPS (stamped by train_3f.py).</summary>
        public required IReadOnlyDictionary<(string Station, string Version), ModelArtifact.TrainingMetadata> MetadataByKey { get; init; }

        public required DateTime AsOfUtc { get; init; }
        public required int WindowDays { get; init; }

        /// <summary>EA readings settle as they're validated — skip predictions
        /// inside this latency cutoff to avoid scoring against provisional data.</summary>
        public required int LatencyDays { get; init; }

        public required double DriftThreshold { get; init; }

        /// <summary>Minimum N for a drift flag to fire. Same defaults as
        /// PrecipVerifier — tests use 1, production sets to 10.</summary>
        public int MinDriftN { get; init; } = 1;
    }

    public sealed record VerifyRow(
        string TruthStation,
        string ModelVersion,
        int LeadHours,
        int N,
        int WetN,
        double WetRate,
        double BlendCrps,
        double MaeWet,
        double Coverage80,
        double PitMean,
        IReadOnlyList<int> PitBins,
        IReadOnlyDictionary<string, double> ExceedanceBriers,
        double? ReferenceTestCrps,
        bool DriftFlag);

    public static IReadOnlyList<VerifyRow> Compute(Inputs input)
    {
        var windowStart = input.AsOfUtc.AddDays(-input.WindowDays);
        var windowEnd   = input.AsOfUtc.AddDays(-input.LatencyDays);

        // Group on the same (station, version, lead) cell shape the rest of
        // the verify family uses so the JSON rows downstream slot into the
        // existing Models-page lookup without a new cell-key.
        var groups = input.Predictions
            .Where(r => r.ValidTimeUtc >= windowStart && r.ValidTimeUtc <= windowEnd)
            .GroupBy(r => (r.TruthStation, r.ModelVersion, r.LeadHours))
            .OrderBy(g => g.Key.TruthStation, StringComparer.Ordinal)
            .ThenBy(g => g.Key.ModelVersion,  StringComparer.Ordinal)
            .ThenBy(g => g.Key.LeadHours);

        var output = new List<VerifyRow>();
        foreach (var g in groups)
        {
            if (!input.TruthByStationTime.TryGetValue(g.Key.TruthStation, out var truthByTime))
                continue;

            // Pair predictions with truth on valid_time. Drop rows where the
            // truth is missing (EA gauge gap) — the metric is honest about
            // what's actually been scored, not padded with NaN.
            var paired = new List<(RainfallAmountPredictionRow Row, double Y)>();
            foreach (var r in g)
            {
                if (truthByTime.TryGetValue(r.ValidTimeUtc, out var y))
                    paired.Add((r, y));
            }
            if (paired.Count == 0) continue;

            var crpsValues = new double[paired.Count];
            var wetMaeAccum = 0.0;
            int wetN = 0;
            int inCi80 = 0;
            var pitValues = new double[paired.Count];
            var pitBins = new int[PitBinCount];
            var exceedanceBriers = new Dictionary<string, double>(ExceedanceThresholdsMm.Length, StringComparer.Ordinal);
            var exceedanceAccum = new double[ExceedanceThresholdsMm.Length];

            // Each row's mixed quantile vector for the CRPS estimator. Match
            // the configured plan alphas [0.025, 0.1, 0.5, 0.9, 0.975] —
            // these are the quantile fields persisted on the prediction row.
            for (int i = 0; i < paired.Count; i++)
            {
                var (row, y) = paired[i];
                var qs = new[]
                {
                    row.P2_5MmPerHr, row.P10MmPerHr, row.P50MmPerHr, row.P90MmPerHr, row.P97_5MmPerHr,
                };
                crpsValues[i] = CrpsMixed(row.Pi, qs, y);

                if (y >= WetThresholdMm)
                {
                    wetMaeAccum += Math.Abs(row.MedianMmPerHr - y);
                    wetN++;
                }

                if (y >= row.P10MmPerHr && y <= row.P90MmPerHr)
                    inCi80++;

                pitValues[i] = MixedPit(row.Pi, row.MuLog, row.SigmaLog, y);
                var binIdx = (int)Math.Min(Math.Floor(pitValues[i] * PitBinCount), PitBinCount - 1);
                pitBins[binIdx]++;

                for (int t = 0; t < ExceedanceThresholdsMm.Length; t++)
                {
                    var thr = ExceedanceThresholdsMm[t];
                    var pHat = ExceedanceProb(row, thr);
                    var obs = y >= thr ? 1.0 : 0.0;
                    exceedanceAccum[t] += (pHat - obs) * (pHat - obs);
                }
            }

            for (int t = 0; t < ExceedanceThresholdsMm.Length; t++)
                exceedanceBriers[FormatThresholdKey(ExceedanceThresholdsMm[t])] = exceedanceAccum[t] / paired.Count;

            var meanCrps = crpsValues.Average();
            var maeWet = wetN > 0 ? wetMaeAccum / wetN : double.NaN;
            var coverage = (double)inCi80 / paired.Count;
            var pitMean = pitValues.Average();

            // Drift reference: training-time test CRPS stamped on the bundle as
            // BlendTestMae (verify_3f.py is gone — train_3f.py fills the field
            // post-training using the same crps_mixed formula on the held-out
            // test slice; see docs/RAINFALL_AMOUNT_3F_PLAN.md §4 / the
            // train_3f.py change shipped in this commit's sibling).
            double? referenceCrps = null;
            bool driftFlag = false;
            if (input.MetadataByKey.TryGetValue((g.Key.TruthStation, g.Key.ModelVersion), out var meta)
                && meta.PerLead.TryGetValue(g.Key.LeadHours.ToString(), out var perLead)
                && perLead.BlendTestMae > 0)
            {
                referenceCrps = perLead.BlendTestMae;
                driftFlag = paired.Count >= input.MinDriftN
                            && meanCrps > input.DriftThreshold * referenceCrps.Value;
            }

            output.Add(new VerifyRow(
                TruthStation: g.Key.TruthStation,
                ModelVersion: g.Key.ModelVersion,
                LeadHours:    g.Key.LeadHours,
                N:            paired.Count,
                WetN:         wetN,
                WetRate:      (double)wetN / paired.Count,
                BlendCrps:    meanCrps,
                MaeWet:       maeWet,
                Coverage80:   coverage,
                PitMean:      pitMean,
                PitBins:      pitBins,
                ExceedanceBriers: exceedanceBriers,
                ReferenceTestCrps: referenceCrps,
                DriftFlag:    driftFlag));
        }

        return output;
    }

    /// <summary>
    /// Mixed-distribution CRPS estimator from a quantile representation.
    /// Direct port of <c>crps_mixed</c> in
    /// <c>WP/scripts/run_membury_two_stage_ngboost.py:133</c>:
    ///
    /// <code>
    /// K = quantiles.length
    /// w_dry = 1 - π
    /// w_wet = π / K
    /// term1 = w_dry · y + w_wet · Σ_k |q_k - y|
    /// cross_0k = 2 · w_dry · w_wet · Σ_k q_k
    /// pairwise = Σ_{i,j} |q_i - q_j|
    /// cross_kl = w_wet² · pairwise
    /// CRPS = term1 - 0.5 · (cross_0k + cross_kl)
    /// </code>
    ///
    /// Pinned bit-for-bit by <c>CrpsMixed_matches_python_reference</c> in
    /// the test suite.
    /// </summary>
    public static double CrpsMixed(double pi, IReadOnlyList<double> quantiles, double y)
    {
        var k = quantiles.Count;
        var wDry = 1.0 - pi;
        var wWet = pi / k;

        double sumAbs = 0.0, sumQ = 0.0;
        for (int i = 0; i < k; i++)
        {
            sumAbs += Math.Abs(quantiles[i] - y);
            sumQ   += quantiles[i];
        }

        double pairwise = 0.0;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                pairwise += Math.Abs(quantiles[i] - quantiles[j]);

        var term1    = wDry * y + wWet * sumAbs;
        var cross0k  = 2.0 * wDry * wWet * sumQ;
        var crossKl  = wWet * wWet * pairwise;
        return term1 - 0.5 * (cross0k + crossKl);
    }

    /// <summary>
    /// PIT value for the mixed distribution. The mixed CDF is
    ///   F(y) = (1-π) · 1[y ≥ 0] + π · LogNormalCDF(y; μ, σ)
    /// For y &gt; 0 this returns F(y) directly. For y = 0 (a dry obs) the
    /// distribution has a point mass of size (1-π) there — by the
    /// randomised-PIT convention we sample uniformly on [0, 1-π]; the
    /// midpoint (1-π)/2 keeps the per-row PIT deterministic + the
    /// histogram interpretation honest at the population level.
    /// </summary>
    public static double MixedPit(double pi, double muLog, double sigmaLog, double y)
    {
        if (y <= 0.0) return (1.0 - pi) * 0.5;
        var z = (Math.Log(y) - muLog) / sigmaLog;
        var lognormCdf = StandardNormalCdf(z);
        return (1.0 - pi) + pi * lognormCdf;
    }

    /// <summary>
    /// Mixed P(mm/h ≥ threshold). For threshold &gt; 0:
    ///   P = π · (1 - LogNormalCDF(threshold; μ, σ))
    /// matching the predict-side derivation. Re-derived here rather than
    /// reading the row's PExceed_* fields so the verifier can compute
    /// Brier on any threshold, not just the four the prediction row carries.
    /// </summary>
    internal static double ExceedanceProb(RainfallAmountPredictionRow row, double thresholdMm)
    {
        if (thresholdMm <= 0.0) return row.Pi;
        var z = (Math.Log(thresholdMm) - row.MuLog) / row.SigmaLog;
        var lognormCdf = StandardNormalCdf(z);
        return row.Pi * (1.0 - lognormCdf);
    }

    /// <summary>
    /// Standard normal CDF Φ(z) via Abramowitz-Stegun 7.1.26
    /// (max abs error ≈ 1.5e-7). Same approximation
    /// <see cref="WeatherBlend.Train.DryWindow.DryWindow3pPredictor.StandardNormalCdf"/>
    /// uses — duplicating it here so the Evaluate tier doesn't pull on
    /// the Train namespace. (Cheap; sub-microsecond.)
    /// </summary>
    public static double StandardNormalCdf(double z)
    {
        const double a1 =  0.254829592;
        const double a2 = -0.284496736;
        const double a3 =  1.421413741;
        const double a4 = -1.453152027;
        const double a5 =  1.061405429;
        const double p  =  0.3275911;

        double x = z / Math.Sqrt(2.0);
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + p * x);
        double erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * erf);
    }

    /// <summary>Threshold key shape used in <see cref="VerifyHistoryRow.ExceedanceBriers"/>
    /// and the markdown report. Round-trip-friendly: <c>"0.1" / "1" / "5" / "10"</c>.</summary>
    public static string FormatThresholdKey(double thresholdMm) =>
        thresholdMm == Math.Floor(thresholdMm)
            ? ((int)thresholdMm).ToString(System.Globalization.CultureInfo.InvariantCulture)
            : thresholdMm.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
}
