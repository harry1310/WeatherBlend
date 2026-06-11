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
}
