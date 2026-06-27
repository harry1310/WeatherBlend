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
    // Per-phase per-model column lists. 2b/2c (offset_day) read from the
    // canonical 7-NWP slots; 2d (exact-runtime) reads from the 5 *Exact slots
    // since its model identities don't match (raw IFS oper vs Open-Meteo
    // ecmwf_ifs025, raw MO Global vs ukmo_seamless, plus UKV which has no
    // offset_day twin). Kept local to avoid a circular dep; verify's per-model
    // ordering is verification-specific anyway.
    internal static readonly string[] ModelNamesOffsetDay =
        { "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem", "temp_aifs" };

    /// <summary>Phase 2d per-model column list — distinct from offset_day's
    /// because the model identities don't fully overlap. Order matches the
    /// canonical Exact12hFeatureBuilder column order plus UKV at the end.</summary>
    internal static readonly string[] ModelNamesExact =
        { "temp_gfs_exact", "temp_ifs_oper_exact", "temp_aifs_oper_exact",
          "temp_moglobal_exact", "temp_ukv_exact" };

    /// <summary>Resolve the per-NWP column list for a given training phase.
    /// Defaults to the offset_day list — preserves behaviour for any version
    /// whose metadata is missing (legacy artefacts) or has an unknown phase.</summary>
    internal static string[] ModelNamesForPhase(string? phase) =>
        string.Equals(phase, "2d", StringComparison.Ordinal) ? ModelNamesExact : ModelNamesOffsetDay;

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

        /// <summary>Base drift threshold at lead 24h. Effective threshold at
        /// other leads is <c>DriftThreshold + DriftThresholdSlopePer24h × max(0, (L-24)/24)</c>.
        /// Brief originally locked this at 1.5; slope was 0 (flat) until 2026-05-11.</summary>
        public required double DriftThreshold { get; init; }

        /// <summary>
        /// Per-additional-24h relaxation of the drift threshold. Long-lead
        /// forecasts naturally degrade — a 120h prediction missing 1.5× its
        /// training MAE is much less alarming than a 24h one doing the same.
        /// With the default 0.0, behaviour is the flat-threshold pre-2026-05-11
        /// model (good for tests). Production sets this to 0.1 (or whatever
        /// the user tunes), giving e.g. 24h:1.5×, 48h:1.6×, 72h:1.7×, 120h:1.9×.
        /// </summary>
        public double DriftThresholdSlopePer24h { get; init; } = 0.0;

        /// <summary>
        /// Minimum N for a drift flag to fire. Cells with fewer rows than this
        /// still appear in the per-version table but their DriftFlag stays false
        /// — a single unlucky prediction will routinely cross 1.5× the training
        /// MAE without indicating real model regression. With MinDriftN=10,
        /// smoke-test versions (n=1) and long-retired versions (n→0 in the
        /// rolling window) self-suppress; recently-retired and currently-deployed
        /// champions accumulate enough to be evaluated. No separate "active
        /// versions" filter is needed — predict workflows stop emitting rows
        /// for retired versions, so their counts naturally fade out of the
        /// 14-day window within ~2 weeks. Default 1 preserves pre-2026-05-11
        /// behaviour for tests; production sets to 10.
        /// </summary>
        public int MinDriftN { get; init; } = 1;
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
        // (24, 30, 36, ...) on rows produced by ComputeActualLeadBuckets.
        // No persistence + no drift flag on bucket rows — there's no per-
        // actual-lead training-time baseline to compare to.
        int? ActualLeadBucketLowH = null);

    /// <summary>The 6h actual-lead bucket size used for the bucketed view.
    /// Verify groups predictions whose actual NWP forecast lead falls in
    /// <c>[low, low+6)</c> where <c>actual lead = ValidTimeUtc − freshest
    /// contributing NWP RunTime (hours)</c> — see <see cref="FreshestNwpRunTime"/>
    /// for why the NWP cycle, not PredictionMadeAtUtc, is the reference instant.
    /// Same bucket size used in 2026-05-05 ad-hoc bucketing analysis; see
    /// memory/project_temp_actual_lead_bucketing.md for the rationale.</summary>
    public const int ActualLeadBucketHours = 6;

    public static IReadOnlyList<VerifyRow> Compute(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.Era5LatencyDays);

        // Keep rows (a) inside the window and (b) with matching truth. No
        // version filter — MinDriftN on the drift gate handles noise from
        // smoke tests + long-retired versions; predict naturally stops emitting
        // rows for retired versions so they fade from the rolling window.
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
    /// Parallel view that groups predictions by their ACTUAL NWP forecast lead
    /// (<c>ValidTimeUtc − freshest contributing NWP RunTime</c>) in 6-hour
    /// buckets, instead of by the trained-lead bucket label. Becomes
    /// meaningful once predict emits 24 hourly rows per cycle (the
    /// 2026-05-04 hourly-temp-predict change), where each trained-lead
    /// bucket spreads predictions across a *calendar day* of valid times.
    /// Per-bucket: blend MAE / mean-of-models MAE / best single. No
    /// drift flag — there's no per-actual-lead training-time baseline.
    ///
    /// The reference instant is the freshest NWP cycle that fed the blend, NOT
    /// PredictionMadeAtUtc (the wall-clock moment the cron fired). Those differ
    /// sharply: the predict job runs mid-day but emits the whole target calendar
    /// day (00:00–23:00 of <c>anchorDay + L/24</c>), so <c>ValidTime −
    /// PredictionMadeAt</c> for the early hours of that day is &lt; L even
    /// though the blend was fed a ≥L-lead NWP forecast (predict floors every
    /// NWP cycle at <c>RunTime ≤ ValidTime − L</c>). Bucketing off the NWP
    /// cycle reports the genuine forecast lead, so a min-lead-24 model never
    /// shows sub-24h buckets. (2026-06-02 — Harry: the old PredictionMadeAt
    /// math made 2b/2c look like they were making &lt;24h forecasts.)
    /// </summary>
    public static IReadOnlyList<VerifyRow> ComputeActualLeadBuckets(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.Era5LatencyDays);

        var kept = inputs.Predictions
            .Where(p => p.ValidTimeUtc >= windowStart && p.ValidTimeUtc <= windowEnd)
            .Where(p => inputs.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();

        // Bucket by floor((ValidTime − freshestNwpRunTime).TotalHours / 6) * 6.
        // Rows with no NWP cycle time at all (legacy parquets pre-dating
        // run-time capture) can't be placed on an NWP-lead axis honestly, so
        // they're dropped from this view rather than guessed at. Current
        // production rows always carry RunTimes, and the 14-day rolling window
        // ages out anything older.
        var groups = kept
            .Select(p => new { Pred = p, Run = FreshestNwpRunTime(p) })
            .Where(x => x.Run.HasValue)
            .Select(x => new
            {
                x.Pred,
                BucketLow = (int)Math.Floor((x.Pred.ValidTimeUtc - x.Run!.Value).TotalHours / ActualLeadBucketHours) * ActualLeadBucketHours,
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

    /// <summary>
    /// The freshest (max) NWP cycle time among every per-model RunTime
    /// populated on the row — offset_day (2b/2c) slots and exact-runtime (2d)
    /// slots alike. This is the reference instant for the bucketed actual-lead
    /// view: a blended prediction's effective forecast lead is how far ahead
    /// its most recent contributing cycle was, i.e. <c>ValidTime − freshest
    /// RunTime</c>. Because predict floors every contributing cycle at
    /// <c>RunTime ≤ ValidTime − L</c>, the freshest cycle is still ≥ L hours
    /// out, so the bucket can never read below the trained lead.
    ///
    /// Returns null only when the row carries no cycle times at all (legacy
    /// parquets pre-dating run-time capture); the caller drops such rows.
    /// </summary>
    private static DateTime? FreshestNwpRunTime(TempPredictionRow p)
    {
        DateTime? best = null;
        void Consider(DateTime? t)
        {
            if (t.HasValue && (best is null || t.Value > best.Value)) best = t.Value;
        }
        // offset_day (2b/2c)
        Consider(p.RunTimeGfs);   Consider(p.RunTimeEcmwf); Consider(p.RunTimeIcon);
        Consider(p.RunTimeMf);    Consider(p.RunTimeUkmo);  Consider(p.RunTimeGem);
        Consider(p.RunTimeAifs);
        // exact-runtime (2d)
        Consider(p.RunTimeGfsExact);      Consider(p.RunTimeIfsOperExact);
        Consider(p.RunTimeAifsOperExact); Consider(p.RunTimeMoGlobalExact);
        Consider(p.RunTimeUkvExact);
        return best;
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

        var modelNames = ModelNamesForPhase(PhaseFor(version, inputs));
        var meanPred = preds.Select(p => RowMean(p, modelNames)).ToArray();
        var meanStats = TempMetrics.Compute(meanPred, actual);

        var bestName = "";
        var bestMae  = double.PositiveInfinity;
        foreach (var name in modelNames)
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
        var modelNames = ModelNamesForPhase(PhaseFor(version, inputs));
        var meanPred = preds.Select(p => RowMean(p, modelNames)).ToArray();
        var meanStats = TempMetrics.Compute(meanPred, actual);

        // Best single: pick the per-model column with the lowest MAE in this window.
        // Window-local, not val-set-local — the brief calls for baselines "recomputed
        // from per-model inputs" so this is an observed-in-production ranking.
        var bestName = "";
        var bestMae  = double.PositiveInfinity;
        foreach (var name in modelNames)
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

        // Reference test MAE from training metadata + drift flag. Two gates:
        //   (1) N >= MinDriftN              — suppress single-prediction noise
        //   (2) MAE > effectiveThreshold(L) — actual breach, lead-aware
        // ActiveVersions filtering happens upstream in Compute(), so by this
        // point we only see currently-deployed versions. Reference MAE is
        // surfaced even when N < MinDriftN so reviewers can see thin-sample
        // cells in the per-version table.
        double? refMae = null;
        var drift = false;
        if (inputs.MetadataByVersion.TryGetValue(version, out var md)
            && md.PerLead.TryGetValue(leadHours.ToString(), out var ls)
            && ls.BlendTestMae > 0)
        {
            refMae = ls.BlendTestMae;
            var effectiveThreshold = inputs.DriftThreshold
                + inputs.DriftThresholdSlopePer24h * Math.Max(0.0, (leadHours - 24) / 24.0);
            drift = blendStats.N >= inputs.MinDriftN
                 && blendStats.Mae > effectiveThreshold * ls.BlendTestMae;
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

    private static double RowMean(TempPredictionRow p, IReadOnlyList<string> modelNames)
    {
        if (p.TempMean.HasValue) return p.TempMean.Value;
        double sum = 0; int n = 0;
        foreach (var name in modelNames)
        {
            var v = TempFor(p, name);
            if (v.HasValue) { sum += v.Value; n++; }
        }
        return n == 0 ? double.NaN : sum / n;
    }

    /// <summary>Resolve the training phase for a version by looking it up
    /// in <see cref="Inputs.MetadataByVersion"/>. Versions whose metadata
    /// isn't loaded (e.g. mid-rotation, file truncated) get null → falls back
    /// to the offset_day column list, which matches every shipped phase
    /// pre-2d. Cheap to call per row but typically there's only a handful of
    /// distinct versions per Compute call so the dict lookup is fine.</summary>
    private static string? PhaseFor(string version, Inputs inputs)
    {
        if (inputs.MetadataByVersion.TryGetValue(version, out var md))
            return md.Phase;
        return null;
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
        // Phase 2d (exact-runtime) per-model slots.
        "temp_gfs_exact"      => p.TempGfsExact,
        "temp_ifs_oper_exact" => p.TempIfsOperExact,
        "temp_aifs_oper_exact"=> p.TempAifsOperExact,
        "temp_moglobal_exact" => p.TempMoGlobalExact,
        "temp_ukv_exact"      => p.TempUkvExact,
        _ => null,
    };
}
