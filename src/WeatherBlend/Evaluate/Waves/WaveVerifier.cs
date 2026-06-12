using WeatherBlend.Evaluate.Temp;
using WeatherBlend.Models;

namespace WeatherBlend.Evaluate.Waves;

/// <summary>
/// Pure verification logic for the wave_height blend vs wave-buoy truth.
/// Mirrors <c>ElementVerifier</c>'s shape (join → stratify → drift flags)
/// with two wave-specific deviations:
///
///   1. <b>Lead bucketing.</b> wave_height_lgb is ANY-LEAD v1 — each predict
///      cycle emits one continuous hourly forecast (leads 0..~146), so the
///      framework's per-trained-lead stratification has no trained buckets to
///      use. Per-hourly-lead rows would be 146 cells of N≈windowDays each
///      (noise), and pooling everything would hide the real skill-vs-lead
///      decay. Compromise: 24h-wide actual-lead buckets labelled by their
///      UPPER edge (24 = leads [0,24), 48 = [24,48), …) so the labels line up
///      with the canonical Leads.Full columns everywhere downstream.
///
///   2. <b>Coverage-only drift.</b> The drift flag fires when the empirical
///      coverage of the announced 90% band falls well below nominal (default
///      floor 0.75). There is deliberately NO MAE-threshold drift: the only
///      training-time MAE reference the bundle has (<c>calibration.json
///      cal_mae</c>, 0.107 m) is scored vs era5_ocean, while the buoy
///      (Sevenstones, 28 km W of the forecast cell) carries a ~0.3 m
///      location-mismatch MAE on its own (2026-06-11 truth-cell check) — a
///      1.5× cal_mae threshold would alarm on every run forever. Coverage is
///      the honest signal, and it still catches gross model failure (a
///      broken forecast pushes truth out of the band). cal_mae is carried on
///      each row as CONTEXT for the report/sidecar, never as a trigger.
/// </summary>
public static class WaveVerifier
{
    /// <summary>Per-version cal-slice context parsed from the bundle's
    /// <c>calibration.json</c> — surfaces on the report/sidecar next to the
    /// buoy numbers, never used as a drift trigger. All fields nullable — a
    /// missing/partial file degrades the context columns rather than
    /// failing the run.</summary>
    public sealed record CalibrationReference(
        double? CalMae,
        double? CalBandCoverage,
        double? NominalCoverage,
        string? LeadMode,
        string? Note);

    public sealed class Inputs
    {
        public required IReadOnlyList<WavePredictionRow> Predictions { get; init; }

        /// <summary>Buoy significant wave height (m) keyed by valid hour.</summary>
        public required IReadOnlyDictionary<DateTime, double> TruthByTime { get; init; }

        /// <summary>Buoy wave period Tz (s) keyed by valid hour. Optional —
        /// empty dict skips the period columns. Scores the UNBLENDED
        /// best_match period pass-through; the buoy-Tz-vs-model-mean-period
        /// definitional offset (~1-2 s) lands in PeriodBias, which is why
        /// period never trips drift.</summary>
        public IReadOnlyDictionary<DateTime, double> PeriodTruthByTime { get; init; }
            = new Dictionary<DateTime, double>();

        public required IReadOnlyDictionary<string, CalibrationReference> ReferenceByVersion { get; init; }

        public required DateTime AsOfUtc { get; init; }
        public required int WindowDays { get; init; }

        /// <summary>Days excluded from the recent end of the window. The buoy
        /// realtime feed lands within ~90 min (no ERA5-style release lag), so
        /// production default is 0 — later QC'd archive re-pulls are absorbed
        /// by the next rolling run.</summary>
        public required int TruthLatencyDays { get; init; }

        /// <summary>Coverage below this fires the drift flag ("well below
        /// the nominal 0.90"). Default 0.75. The ONLY drift trigger — see the
        /// class summary for why there is no MAE threshold.</summary>
        public double CoverageDriftFloor { get; init; } = 0.75;

        /// <summary>Minimum pair count for either drift flag to fire — same
        /// noise gate as the other verifiers (production passes 10).</summary>
        public int MinDriftN { get; init; } = 1;
    }

    public sealed record VerifyRow(
        string ModelVersion,
        /// <summary>Upper edge of the 24h-wide actual-lead bucket: 24 covers
        /// leads [0,24), 48 covers [24,48), … — aligned with Leads.Full labels.</summary>
        int LeadBucketHours,
        int N,
        double BlendMae,
        double BlendRmse,
        double BlendBias,
        /// <summary>Pairs where both band bounds were present (coverage denominator).</summary>
        int NBand,
        /// <summary>Fraction of banded pairs with buoy Hs ∈ [BandLoM, BandHiM].
        /// NaN when <see cref="NBand"/> is 0.</summary>
        double BandCoverage,
        /// <summary>The bundle's calibration-slice MAE vs era5_ocean —
        /// CONTEXT only, never a drift trigger (see class summary).</summary>
        double? ReferenceCalMae,
        /// <summary>Coverage-based drift: band coverage well below nominal
        /// (below <see cref="Inputs.CoverageDriftFloor"/>) on ≥ MinDriftN
        /// banded pairs.</summary>
        bool DriftFlag,
        /// <summary>Pairs where both the pass-through period and buoy Tz
        /// were present.</summary>
        int NPeriod = 0,
        /// <summary>Pass-through period MAE vs buoy Tz, s. NaN when
        /// <see cref="NPeriod"/> is 0. Carries the definitional offset —
        /// see <see cref="Inputs.PeriodTruthByTime"/>.</summary>
        double PeriodMae = double.NaN,
        /// <summary>Mean (forecast − buoy Tz), s. The stable definitional
        /// component lives here; a CHANGE in this bias is the signal.</summary>
        double PeriodBias = double.NaN);

    /// <summary>24h-wide bucket labelled by upper edge: lead 0..23 → 24,
    /// 24..47 → 48, etc. Negative leads (live-collector artefacts) clamp
    /// into the first bucket.</summary>
    public static int BucketLead(int leadHours) => Math.Max(0, leadHours) / 24 * 24 + 24;

    public static IReadOnlyList<VerifyRow> Compute(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.TruthLatencyDays);

        var kept = inputs.Predictions
            .Where(p => p.ValidTimeUtc >= windowStart && p.ValidTimeUtc <= windowEnd)
            .Where(p => inputs.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();

        var groups = kept
            .GroupBy(p => (p.ModelVersion, Bucket: BucketLead(p.LeadHours)))
            .OrderBy(g => g.Key.ModelVersion, StringComparer.Ordinal)
            .ThenBy(g => g.Key.Bucket);

        var result = new List<VerifyRow>();
        foreach (var g in groups)
            result.Add(BuildRow(g.Key.ModelVersion, g.Key.Bucket, g.ToList(), inputs));
        return result;
    }

    private static VerifyRow BuildRow(
        string version, int bucket, IReadOnlyList<WavePredictionRow> preds, Inputs inputs)
    {
        var actual = preds.Select(p => inputs.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend  = preds.Select(p => p.BlendValue).ToArray();
        var stats  = TempMetrics.Compute(blend, actual);

        int nBand = 0, nCovered = 0;
        for (int i = 0; i < preds.Count; i++)
        {
            var p = preds[i];
            if (!p.BandLoM.HasValue || !p.BandHiM.HasValue) continue;
            if (double.IsNaN(actual[i])) continue;
            nBand++;
            if (actual[i] >= p.BandLoM.Value && actual[i] <= p.BandHiM.Value) nCovered++;
        }
        var coverage = nBand == 0 ? double.NaN : (double)nCovered / nBand;

        inputs.ReferenceByVersion.TryGetValue(version, out var reference);

        var coverageDrift = nBand >= inputs.MinDriftN
            && !double.IsNaN(coverage)
            && coverage < inputs.CoverageDriftFloor;

        // Period pass-through pairs — independent join (period truth can be
        // present/absent independently of Hs at a given hour).
        int nPeriod = 0; double periodAbsSum = 0, periodErrSum = 0;
        foreach (var p in preds)
        {
            if (!p.WavePeriodS.HasValue) continue;
            if (!inputs.PeriodTruthByTime.TryGetValue(p.ValidTimeUtc, out var tz)) continue;
            var err = p.WavePeriodS.Value - tz;
            nPeriod++;
            periodAbsSum += Math.Abs(err);
            periodErrSum += err;
        }

        return new VerifyRow(
            ModelVersion:    version,
            LeadBucketHours: bucket,
            N:               stats.N,
            BlendMae:        stats.Mae,
            BlendRmse:       stats.Rmse,
            BlendBias:       stats.Bias,
            NBand:           nBand,
            BandCoverage:    coverage,
            ReferenceCalMae: reference?.CalMae,
            DriftFlag:       coverageDrift,
            NPeriod:         nPeriod,
            PeriodMae:       nPeriod == 0 ? double.NaN : periodAbsSum / nPeriod,
            PeriodBias:      nPeriod == 0 ? double.NaN : periodErrSum / nPeriod);
    }
}
