using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// Shared "one row per valid time" collapse for site rendering.
///
/// Every chart / table / tile lookup that wants a single value per valid
/// time has to dedup first: the predictions tree accumulates several rows
/// per valid hour (multiple lead buckets covering the same hour, repeated
/// predict cycles, and — twice now — a second ModelVersion of the same
/// phase straddling the Sunday retrain). Hand-rolled
/// GroupBy/OrderBy/ThenByDescending/First copies of this collapse were the
/// repo's most-shipped bug class (wind speed + direction charts broke twice
/// on the second-ModelVersion case; a raw ToDictionary killed the whole
/// 2026-05-31 site render), so the two pick rules live here once:
///
///   * <see cref="LatestPerValid{T,TKey}(IEnumerable{T},Func{T,TKey},Func{T,int},Func{T,DateTime})"/>
///     — smallest LeadHours first (the most recent NWP cycle that covers
///     the hour), freshest made-at on lead ties.
///   * <see cref="FreshestPerValid{T,TKey}"/> — freshest made-at only, for
///     call sites already pinned to a single lead (per-lead forecast
///     pages) or that genuinely want "newest write wins".
///
/// The grouping key is generic so composite keys collapse with the same
/// rule — (TargetDate, Lead), (WindowHours, StartHour), … — not just a
/// bare ValidTimeUtc. Output order: groups appear in first-seen source
/// order (LINQ GroupBy contract); callers re-sort (usually by valid time)
/// or feed a ToDictionary exactly as before.
/// </summary>
internal static class SeriesDedup
{
    /// <summary>
    /// One row per <paramref name="validKey"/> group: smallest
    /// <paramref name="leadHours"/> wins (most recent cycle); ties broken
    /// by freshest <paramref name="madeAt"/>.
    /// </summary>
    public static IEnumerable<T> LatestPerValid<T, TKey>(
        this IEnumerable<T> rows,
        Func<T, TKey> validKey,
        Func<T, int> leadHours,
        Func<T, DateTime> madeAt)
        => rows
            .GroupBy(validKey)
            .Select(g => g.OrderBy(leadHours).ThenByDescending(madeAt).First());

    /// <summary>
    /// One row per <paramref name="validKey"/> group: smallest
    /// <paramref name="leadHours"/> wins, source order on ties (stable
    /// sort). For sources with no meaningful made-at column (e.g. the Met
    /// Office Spot capture) — kept as a separate overload rather than
    /// inventing a tiebreak the original call sites didn't have.
    /// </summary>
    public static IEnumerable<T> LatestPerValid<T, TKey>(
        this IEnumerable<T> rows,
        Func<T, TKey> validKey,
        Func<T, int> leadHours)
        => rows
            .GroupBy(validKey)
            .Select(g => g.OrderBy(leadHours).First());

    /// <summary>
    /// One row per <paramref name="validKey"/> group: freshest
    /// <paramref name="madeAt"/> wins, source order on ties (stable sort).
    /// </summary>
    public static IEnumerable<T> FreshestPerValid<T, TKey>(
        this IEnumerable<T> rows,
        Func<T, TKey> validKey,
        Func<T, DateTime> madeAt)
        => rows
            .GroupBy(validKey)
            .Select(g => g.OrderByDescending(madeAt).First());

    // -----------------------------------------------------------------
    // Verify-history version pooling — the rolling skill charts' analogue
    // of the per-valid collapse above.
    // -----------------------------------------------------------------

    /// <summary>
    /// Collapse same-key verify rows to ONE pooled row per key before a
    /// rolling skill chart plots them. During a model-version transition,
    /// verify scores BOTH the outgoing and incoming bundles of the same
    /// phase for the same as-of week (each has predictions inside the
    /// scoring window), so a sidecar carries two rows per (asOf, phase,
    /// lead). Plotting both gives the rolling line two Y-values per X —
    /// vertical cliffs, made huge because the incoming version's first
    /// weeks score on a few young rows (n≈10-50) against the outgoing
    /// version's full settled sample (n≈140). The verify metrics are
    /// MEANS over scored rows, so the n-weighted average across versions
    /// IS the true pooled "how did this phase do this week" — pool rather
    /// than pick, and the version-transition weeks stay on the chart
    /// honestly. (Fix first shipped on the 3f CRPS charts 2026-06-11;
    /// lifted here so every rolling verify chart shares it.)
    ///
    /// The key is caller-supplied so each chart pools at its own
    /// granularity — (AsOfUtc, LeadHours) on the single-target 3f charts,
    /// (Target, AsOfUtc, Station, Phase, LeadHours) on the multi-target
    /// wind chart. Groups with a single row pass through untouched.
    /// </summary>
    public static List<(VerifyHistoryFile File, VerifyHistoryRow Row)> PoolVerifyRowsPerKey<TKey>(
        this IEnumerable<(VerifyHistoryFile File, VerifyHistoryRow Row)> rows,
        Func<(VerifyHistoryFile File, VerifyHistoryRow Row), TKey> key)
        => rows
            .GroupBy(key)
            .Select(g => g.Count() == 1
                ? g.First()
                : (g.First().File, PoolVerifyRows(g.Select(t => t.Row).ToList())))
            .ToList();

    /// <summary>
    /// n-weighted pool of same-(asOf, station, phase, lead) verify rows —
    /// one per model VERSION during a bundle transition (see
    /// <see cref="PoolVerifyRowsPerKey"/> for why pooling is the honest
    /// move). Means (MAE / Brier / CRPS / coverage / PIT-mean /
    /// per-threshold Briers) pool as Σ(nᵢ·mᵢ)/Σnᵢ; counts (N, PitBins)
    /// sum; DriftFlag = any. The pooled row carries the largest-n
    /// version's ModelVersion with a "+k more" suffix so the Models-page
    /// provenance stays honest.
    /// </summary>
    public static VerifyHistoryRow PoolVerifyRows(IReadOnlyList<VerifyHistoryRow> rows)
    {
        var totalN = rows.Sum(r => r.N);
        double? WMean(Func<VerifyHistoryRow, double?> f)
        {
            double sum = 0; var n = 0;
            foreach (var r in rows)
            {
                var v = f(r);
                if (v is null || !double.IsFinite(v.Value) || r.N == 0) continue;
                sum += v.Value * r.N; n += r.N;
            }
            return n == 0 ? null : sum / n;
        }
        List<int>? pitBins = null;
        foreach (var r in rows)
        {
            if (r.PitBins is null || r.PitBins.Count != 10) continue;
            pitBins ??= new List<int>(new int[10]);
            for (int i = 0; i < 10; i++) pitBins[i] += r.PitBins[i];
        }
        Dictionary<string, double>? briers = null;
        var brierKeys = rows.Where(r => r.ExceedanceBriers is not null)
            .SelectMany(r => r.ExceedanceBriers!.Keys).Distinct().ToList();
        if (brierKeys.Count > 0)
        {
            briers = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var k in brierKeys)
            {
                var v = WMean(r => r.ExceedanceBriers is not null && r.ExceedanceBriers.TryGetValue(k, out var b) ? b : null);
                if (v is not null) briers[k] = v.Value;
            }
        }
        var primary = rows.OrderByDescending(r => r.N).First();
        return new VerifyHistoryRow
        {
            Station = primary.Station,
            ModelVersion = rows.Count > 1
                ? $"{primary.ModelVersion} (+{rows.Count - 1} more pooled)"
                : primary.ModelVersion,
            Phase = primary.Phase,
            LeadHours = primary.LeadHours,
            WindowHours = primary.WindowHours,
            ViewKind = primary.ViewKind,
            ActualLeadBucketLowH = primary.ActualLeadBucketLowH,
            N = totalN,
            BlendMetric = WMean(r => r.BlendMetric) ?? primary.BlendMetric,
            // Clim / mean-of-models are per-row means over the same scored
            // rows, so they pool with the same n-weighting as the blend
            // metric. Best-single + reference metrics are version-scoped
            // (different bundles trained against different baselines) —
            // carry the largest-n version's values like ModelVersion.
            ClimMetric = WMean(r => r.ClimMetric),
            MeanOfModelsMetric = WMean(r => r.MeanOfModelsMetric),
            BestSingleName = primary.BestSingleName,
            BestSingleMetric = primary.BestSingleMetric,
            ReferenceTrainingMetric = primary.ReferenceTrainingMetric,
            DriftFlag = rows.Any(r => r.DriftFlag),
            MaeWet = WMean(r => r.MaeWet),
            Coverage80 = WMean(r => r.Coverage80),
            PitMean = WMean(r => r.PitMean),
            PitBins = pitBins,
            ExceedanceBriers = briers,
        };
    }
}
