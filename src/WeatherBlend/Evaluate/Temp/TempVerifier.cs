using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Evaluate.Temp;

/// <summary>
/// Pure verification domain logic. Joins production predictions against ERA5 truth,
/// stratifies rolling metrics by (model version, lead) — never a single cross-version
/// number — and flags drift when rolling blend MAE exceeds a threshold × the training-
/// time test MAE for that lead.
///
/// No I/O: callers supply pre-loaded predictions, a truth map keyed by ValidTime, and
/// the training metadata for each version that appears in the prediction set. Keeping
/// this class pure means every branch (window exclusion, ERA5 latency cutoff, drift
/// threshold, persistence lookup, best-single picking) is testable without DuckDB.
/// </summary>
public static class TempVerifier
{
    // Same ordering as TempFeatureBuilder.ModelColumns — kept local to avoid a circular dep
    // and because verify's per-model ordering is verification-specific anyway.
    internal static readonly string[] ModelNames =
        { "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem", "temp_aifs" };

    public sealed class Inputs
    {
        public required IReadOnlyList<TempPredictionRow> Predictions { get; init; }

        /// <summary>ERA5 temperature by ValidTimeUtc. Must cover both the window and
        /// the persistence-lookback range (ValidTime − LeadHours for each prediction).</summary>
        public required IReadOnlyDictionary<DateTime, double> TruthByTime { get; init; }

        /// <summary>One entry per distinct ModelVersion that appears in Predictions.
        /// Missing versions still produce rows — their drift column is blank.</summary>
        public required IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata> MetadataByVersion { get; init; }

        public required DateTime AsOfUtc { get; init; }

        /// <summary>Rolling window size. Brief locks this at 14 days.</summary>
        public required int WindowDays { get; init; }

        /// <summary>Skip predictions whose ValidTimeUtc is within this many days of AsOf
        /// — ERA5 hasn't been released yet, so there's no truth to compare against.
        /// Brief locks this at 5 days.</summary>
        public required int Era5LatencyDays { get; init; }

        /// <summary>Drift flag fires when rolling blend MAE &gt; threshold × training-test MAE.
        /// Brief locks this at 1.5.</summary>
        public required double DriftThreshold { get; init; }
    }

    public sealed record VerifyRow(
        string ModelVersion,
        int LeadHours,
        int N,
        double BlendMae,
        double BlendRmse,
        double BlendBias,
        double MeanMae,
        double MeanBias,
        string BestSingleName,
        double BestSingleMae,
        double? PersistenceMae,
        int PersistenceDropped,
        double? ReferenceTestMae,
        bool DriftFlag,
        // 2026-05-05 addition: bucketing-by-actual-lead view. Null on the
        // existing trained-lead rows; set to the 6-hour bucket lower bound
        // (0, 6, 12, 18, 24, ...) on rows produced by ComputeActualLeadBuckets.
        // No persistence + no drift flag on bucket rows — there's no per-
        // actual-lead training-time baseline to compare to.
        int? ActualLeadBucketLowH = null);

    /// <summary>The 6h actual-lead bucket size used for the bucketed view.
    /// Verify groups predictions whose actual lead falls in <c>[low, low+6)</c>
    /// where <c>actual lead = ValidTimeUtc - PredictionMadeAtUtc (hours)</c>.
    /// Same bucket size used in 2026-05-05 ad-hoc bucketing analysis; see
    /// memory/project_temp_actual_lead_bucketing.md for the rationale.</summary>
    public const int ActualLeadBucketHours = 6;

    public static IReadOnlyList<VerifyRow> Compute(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.Era5LatencyDays);

        // Keep rows (a) inside the window and (b) with matching truth. Without truth
        // there's nothing to score; dropping upstream keeps every downstream metric
        // honest (no silent NaN padding).
        var kept = inputs.Predictions
            .Where(p => p.ValidTimeUtc >= windowStart && p.ValidTimeUtc <= windowEnd)
            .Where(p => inputs.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();

        var groups = kept
            .GroupBy(p => (p.ModelVersion, p.LeadHours))
            .OrderBy(g => g.Key.ModelVersion, StringComparer.Ordinal)
            .ThenBy(g => g.Key.LeadHours);

        var result = new List<VerifyRow>();
        foreach (var g in groups)
            result.Add(BuildRow(g.Key.ModelVersion, g.Key.LeadHours, g.ToList(), inputs));
        return result;
    }

    /// <summary>
    /// Parallel view that groups predictions by their ACTUAL lead at
    /// prediction time (<c>ValidTimeUtc − PredictionMadeAtUtc</c>) in 6-hour
    /// buckets, instead of by the trained-lead bucket label. Becomes
    /// meaningful once predict emits 24 hourly rows per cycle (the
    /// 2026-05-04 hourly-temp-predict change), where each trained-lead
    /// bucket spreads predictions across actual leads <c>L .. L+23h</c>.
    /// Per-bucket: blend MAE / mean-of-models MAE / best single. No
    /// drift flag — there's no per-actual-lead training-time baseline.
    /// </summary>
    public static IReadOnlyList<VerifyRow> ComputeActualLeadBuckets(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.Era5LatencyDays);

        var kept = inputs.Predictions
            .Where(p => p.ValidTimeUtc >= windowStart && p.ValidTimeUtc <= windowEnd)
            .Where(p => inputs.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();

        // Bucket by floor((ValidTime - PredictionMadeAt).TotalHours / 6) * 6.
        // Negative actual leads (test artifacts where predict ran AFTER
        // valid time) are kept honestly — they bucket to negative bucket
        // labels and the renderer can decide whether to surface them.
        var groups = kept
            .Select(p => new
            {
                Pred = p,
                BucketLow = (int)Math.Floor((p.ValidTimeUtc - p.PredictionMadeAtUtc).TotalHours / ActualLeadBucketHours) * ActualLeadBucketHours,
            })
            .GroupBy(x => (x.Pred.ModelVersion, x.BucketLow))
            .OrderBy(g => g.Key.ModelVersion, StringComparer.Ordinal)
            .ThenBy(g => g.Key.BucketLow);

        var result = new List<VerifyRow>();
        foreach (var g in groups)
        {
            var preds = g.Select(x => x.Pred).ToList();
            // BuildBucketRow mirrors BuildRow but skips persistence (the
            // (ValidTime − bucketLow) lookback isn't a meaningful "yesterday"
            // anymore at sub-24h buckets) and skips the drift baseline lookup.
            result.Add(BuildBucketRow(g.Key.ModelVersion, g.Key.BucketLow, preds, inputs));
        }
        return result;
    }

    private static VerifyRow BuildBucketRow(
        string version,
        int bucketLowH,
        IReadOnlyList<TempPredictionRow> preds,
        Inputs inputs)
    {
        var actual = preds.Select(p => inputs.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend  = preds.Select(p => p.BlendTemperature).ToArray();
        var blendStats = TempMetrics.Compute(blend, actual);

        var meanPred = preds.Select(RowMean).ToArray();
        var meanStats = TempMetrics.Compute(meanPred, actual);

        var bestName = "";
        var bestMae  = double.PositiveInfinity;
        foreach (var name in ModelNames)
        {
            var p = preds.Select(row => TempFor(row, name) ?? double.NaN).ToArray();
            var s = TempMetrics.Compute(p, actual);
            if (s.N > 0 && s.Mae < bestMae)
            {
                bestMae = s.Mae;
                bestName = name;
            }
        }

        // LeadHours field on the row carries the bucket low for the bucket
        // view (so consumers that key on LeadHours don't have to special-case
        // bucket rows). The discriminator is ActualLeadBucketLowH being non-null.
        return new VerifyRow(
            ModelVersion:       version,
            LeadHours:          bucketLowH,
            N:                  blendStats.N,
            BlendMae:           blendStats.Mae,
            BlendRmse:          blendStats.Rmse,
            BlendBias:          blendStats.Bias,
            MeanMae:            meanStats.Mae,
            MeanBias:           meanStats.Bias,
            BestSingleName:     bestName,
            BestSingleMae:      double.IsPositiveInfinity(bestMae) ? double.NaN : bestMae,
            PersistenceMae:     null,
            PersistenceDropped: 0,
            ReferenceTestMae:   null,
            DriftFlag:          false,
            ActualLeadBucketLowH: bucketLowH);
    }

    private static VerifyRow BuildRow(
        string version,
        int leadHours,
        IReadOnlyList<TempPredictionRow> preds,
        Inputs inputs)
    {
        var actual = preds.Select(p => inputs.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend  = preds.Select(p => p.BlendTemperature).ToArray();
        var blendStats = TempMetrics.Compute(blend, actual);

        // Mean-of-models: if TempMean is populated (predict writes it), trust it;
        // otherwise recompute from non-null per-model temps on the row. A row where
        // every model is null becomes NaN and TempMetrics.Compute drops it.
        var meanPred = preds.Select(RowMean).ToArray();
        var meanStats = TempMetrics.Compute(meanPred, actual);

        // Best single: pick the per-model column with the lowest MAE in this window.
        // Window-local, not val-set-local — the brief calls for baselines "recomputed
        // from per-model inputs" so this is an observed-in-production ranking.
        var bestName = "";
        var bestMae  = double.PositiveInfinity;
        foreach (var name in ModelNames)
        {
            var p = preds.Select(row => TempFor(row, name) ?? double.NaN).ToArray();
            var s = TempMetrics.Compute(p, actual);
            if (s.N > 0 && s.Mae < bestMae)
            {
                bestMae = s.Mae;
                bestName = name;
            }
        }

        // Persistence: truth at (ValidTime − LeadHours). Dropped rows (no truth at lag)
        // don't count toward N but are reported so a sudden spike in drops is visible.
        var persPred = new double[preds.Count];
        var persDropped = 0;
        for (int i = 0; i < preds.Count; i++)
        {
            var lagT = preds[i].ValidTimeUtc.AddHours(-leadHours);
            if (inputs.TruthByTime.TryGetValue(lagT, out var v))
                persPred[i] = v;
            else
            {
                persPred[i] = double.NaN;
                persDropped++;
            }
        }
        var persStats = TempMetrics.Compute(persPred, actual);

        // Reference test MAE from training metadata + drift flag. Missing metadata is
        // not an error — older versions may have been pruned or the prediction may
        // pre-date today's manifest. Row still renders, drift column just blank.
        double? refMae = null;
        var drift = false;
        if (inputs.MetadataByVersion.TryGetValue(version, out var md)
            && md.PerLead.TryGetValue(leadHours.ToString(), out var ls)
            && ls.BlendTestMae > 0)
        {
            refMae = ls.BlendTestMae;
            drift = blendStats.N > 0 && blendStats.Mae > inputs.DriftThreshold * ls.BlendTestMae;
        }

        return new VerifyRow(
            ModelVersion:       version,
            LeadHours:          leadHours,
            N:                  blendStats.N,
            BlendMae:           blendStats.Mae,
            BlendRmse:          blendStats.Rmse,
            BlendBias:          blendStats.Bias,
            MeanMae:            meanStats.Mae,
            MeanBias:           meanStats.Bias,
            BestSingleName:     bestName,
            BestSingleMae:      double.IsPositiveInfinity(bestMae) ? double.NaN : bestMae,
            PersistenceMae:     persStats.N > 0 ? persStats.Mae : (double?)null,
            PersistenceDropped: persDropped,
            ReferenceTestMae:   refMae,
            DriftFlag:          drift);
    }

    private static double RowMean(TempPredictionRow p)
    {
        if (p.TempMean.HasValue) return p.TempMean.Value;
        double sum = 0; int n = 0;
        foreach (var name in ModelNames)
        {
            var v = TempFor(p, name);
            if (v.HasValue) { sum += v.Value; n++; }
        }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double? TempFor(TempPredictionRow p, string name) => name switch
    {
        "temp_gfs"   => p.TempGfs,
        "temp_ecmwf" => p.TempEcmwf,
        "temp_icon"  => p.TempIcon,
        "temp_mf"    => p.TempMf,
        "temp_ukmo"  => p.TempUkmo,
        "temp_gem"   => p.TempGem,
        "temp_aifs"  => p.TempAifs,
        _ => null,
    };
}
