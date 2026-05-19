using Parquet.Serialization;

namespace WeatherBlend.Predict;

/// <summary>
/// Shared merge-on-write for the per-cycle predictions parquet that every
/// predict command produces. Replaces the open / deserialize-existing /
/// concat / GroupBy-dedup / order / serialize block that was hand-rolled in
/// six commands (and half-extracted once as <c>ElementPredictWriter</c>).
///
/// The dedup key is a REQUIRED caller-supplied selector — there is no default
/// on purpose. Every predict writer that has shipped a wrong key silently
/// collapsed sibling rows: keying on <c>(PredictionMadeAtUtc, LeadHours)</c>
/// alone drops all but one of the hourly rows a single run emits per lead
/// (the 2026-05-07 feels-like / element bug). The key must capture the row's
/// full identity — usually <c>(PredictionMadeAtUtc, LeadHours, ValidTimeUtc)</c>,
/// but start-hour adds <c>StartHourUtc</c> and dry-window is day-granular
/// (<c>(PredictionMadeAtUtc, LeadHours)</c> only).
/// </summary>
public static class PredictionParquetWriter
{
    /// <summary>
    /// Pure merge: concat <paramref name="existing"/> + <paramref name="incoming"/>,
    /// deduplicate keeping the row with the latest <paramref name="freshness"/>
    /// per <paramref name="dedupKey"/>, and return ordered by
    /// <paramref name="orderBy"/>. Side-effect free — unit-testable in isolation.
    /// </summary>
    public static List<TRow> Merge<TRow, TKey>(
        IEnumerable<TRow> existing,
        IEnumerable<TRow> incoming,
        Func<TRow, TKey> dedupKey,
        Func<TRow, DateTime> freshness,
        Func<IEnumerable<TRow>, IOrderedEnumerable<TRow>> orderBy)
        => orderBy(
               existing.Concat(incoming)
                   .GroupBy(dedupKey)
                   .Select(g => g.MaxBy(freshness)!))
           .ToList();

    /// <summary>
    /// Merge-on-write IO. Creates the parent directory, deserialises any
    /// existing parquet at <paramref name="outPath"/>, applies
    /// <paramref name="merge"/> against the incoming rows, and re-serialises.
    /// Returns the merged row count. Callers log their own per-target message.
    /// </summary>
    public static async Task<int> WriteAsync<TRow>(
        string outPath,
        IReadOnlyList<TRow> incoming,
        Func<IEnumerable<TRow>, IEnumerable<TRow>, List<TRow>> merge,
        CancellationToken ct)
        where TRow : new()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

        List<TRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<TRow>(outPath, cancellationToken: ct)).ToList()
            : new List<TRow>();

        var merged = merge(existing, incoming);

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        return merged.Count;
    }

    /// <summary>
    /// Convenience overload for callers that don't keep a named merge function:
    /// builds the <see cref="Merge{TRow,TKey}"/> closure from the three
    /// selectors and forwards to <see cref="WriteAsync{TRow}"/>.
    /// </summary>
    public static Task<int> WriteAsync<TRow, TKey>(
        string outPath,
        IReadOnlyList<TRow> incoming,
        Func<TRow, TKey> dedupKey,
        Func<TRow, DateTime> freshness,
        Func<IEnumerable<TRow>, IOrderedEnumerable<TRow>> orderBy,
        CancellationToken ct)
        where TRow : new()
        => WriteAsync(
            outPath,
            incoming,
            (existing, inc) => Merge(existing, inc, dedupKey, freshness, orderBy),
            ct);
}
