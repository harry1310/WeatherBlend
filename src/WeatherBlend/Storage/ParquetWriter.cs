using Parquet;
using Parquet.Schema;
using Parquet.Serialization;
using WeatherBlend.Models;

namespace WeatherBlend.Storage;

/// <summary>
/// Writes forecast/observation rows to Parquet, partitioned on disk by
/// location/model/date/hour. DuckDB can read this hive-style partitioning natively
/// via read_parquet('path/**/*.parquet', hive_partitioning = true).
/// </summary>
public static class ParquetWriter
{
    public static async Task WriteForecastsAsync(
        string basePath,
        IReadOnlyList<ForecastRow> rows,
        CancellationToken ct = default)
    {
        if (rows.Count == 0) return;

        // All rows here share location, model, run_time - caller groups appropriately.
        var first = rows[0];
        var runDate = first.RunTimeUtc.ToString("yyyy-MM-dd");
        var runHour = first.RunTimeUtc.ToString("HH");

        var dir = Path.Combine(
            basePath,
            $"location={first.LocationName}",
            $"model={first.Model}",
            $"date={runDate}");
        Directory.CreateDirectory(dir);

        var file = Path.Combine(dir, $"run={runHour}.parquet");
        await ParquetSerializer.SerializeAsync(rows, file, cancellationToken: ct);
    }

    /// <summary>
    /// Historical-forecast writer. Groups hourly rows by valid-date and writes one
    /// file per day — mirrors ERA5's layout. Drops the run=HH key from the path since
    /// the historical-forecast API doesn't expose a true issue time (per-row RunTime
    /// is synthesized as midnight of valid date; see OpenMeteoClient.Parse).
    /// Path: location=.../model=.../date=&lt;valid_date&gt;/data.parquet
    /// </summary>
    public static async Task WriteHistoricalForecastsAsync(
        string basePath,
        IReadOnlyList<ForecastRow> rows,
        CancellationToken ct = default)
    {
        if (rows.Count == 0) return;

        foreach (var group in rows.GroupBy(r => new { r.LocationName, r.Model, Date = r.ValidTimeUtc.Date }))
        {
            var dateStr = group.Key.Date.ToString("yyyy-MM-dd");
            var dir = Path.Combine(
                basePath,
                $"location={group.Key.LocationName}",
                $"model={group.Key.Model}",
                $"date={dateStr}");
            Directory.CreateDirectory(dir);

            var file = Path.Combine(dir, "data.parquet");
            await ParquetSerializer.SerializeAsync(
                group.OrderBy(r => r.ValidTimeUtc).ToList(), file, cancellationToken: ct);
        }
    }

    public static async Task WriteEra5Async(
        string basePath,
        IReadOnlyList<Era5Row> rows,
        CancellationToken ct = default)
    {
        if (rows.Count == 0) return;

        // Partition by valid date so each file holds at most 24 hourly rows.
        foreach (var group in rows.GroupBy(r => r.ValidTimeUtc.Date))
        {
            var first = group.First();
            var dateStr = group.Key.ToString("yyyy-MM-dd");
            var dir = Path.Combine(basePath, $"location={first.LocationName}", $"date={dateStr}");
            Directory.CreateDirectory(dir);
            var file = Path.Combine(dir, "data.parquet");
            await ParquetSerializer.SerializeAsync(group.OrderBy(r => r.ValidTimeUtc).ToList(), file, cancellationToken: ct);
        }
    }

    public static async Task WriteObservationsAsync(
        string basePath,
        IReadOnlyList<ObservationRow> rows,
        CancellationToken ct = default)
    {
        if (rows.Count == 0) return;

        // Group by observation date so partitions stay tidy.
        foreach (var group in rows.GroupBy(r => r.ObservedTimeUtc.Date))
        {
            var first = group.First();
            var dateStr = group.Key.ToString("yyyy-MM-dd");
            var dir = Path.Combine(
                basePath,
                $"location={first.LocationName}",
                $"station={first.Station}",
                $"date={dateStr}");
            Directory.CreateDirectory(dir);

            // Append to existing file if present by reading + merging + rewriting.
            // For PoC volumes (a few hundred rows/day) this is fine.
            var file = Path.Combine(dir, "observations.parquet");
            var existing = File.Exists(file)
                ? (await ParquetSerializer.DeserializeAsync<ObservationRow>(file, cancellationToken: ct)).ToList()
                : new List<ObservationRow>();

            var merged = existing
                .Concat(group)
                .GroupBy(r => r.ObservedTimeUtc)        // dedupe by observation time
                .Select(g => g.Last())
                .OrderBy(r => r.ObservedTimeUtc)
                .ToList();

            await ParquetSerializer.SerializeAsync(merged, file, cancellationToken: ct);
        }
    }
}
