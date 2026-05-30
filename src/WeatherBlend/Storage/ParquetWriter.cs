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
    /// Previous Runs writer. Groups rows by valid-date and writes one file per day.
    /// Path: location=.../model=.../date=&lt;valid_date&gt;/previous_runs.parquet
    /// </summary>
    public static async Task WritePreviousRunsAsync(
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

            var file = Path.Combine(dir, "previous_runs.parquet");
            await ParquetSerializer.SerializeAsync(
                group.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours).ToList(),
                file, cancellationToken: ct);
        }
    }

    /// <summary>
    /// Historical-forecast pressure writer. Same partition scheme as the Previous
    /// Runs writer, but a SEPARATE file (hist_forecast.parquet) so it can never
    /// overwrite previous_runs.parquet — the two coexist in one date partition
    /// exactly as reported (run=NN.parquet) and offset_day (previous_runs.parquet)
    /// already do. Rows carry RunTimeSource=hist_forecast (lead-unlabelled pressure).
    /// Path: location=.../model=.../date=&lt;valid_date&gt;/hist_forecast.parquet
    /// </summary>
    public static async Task WriteHistoricalForecastAsync(
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

            var file = Path.Combine(dir, "hist_forecast.parquet");
            await ParquetSerializer.SerializeAsync(
                group.OrderBy(r => r.ValidTimeUtc).ToList(),
                file, cancellationToken: ct);
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
            // Skip dates Open-Meteo hasn't published yet — its archive endpoint
            // returns a `time` array of timestamps even for unpublished days,
            // with every variable null. Persisting those as 24-row "scaffold"
            // parquets clutters R2 and forces a write-then-overwrite cycle once
            // real data arrives. Temperature2m is the canary: if it's null for
            // every hour, the day isn't out yet. Mirrors the WHERE-clause
            // filter every downstream query already applies.
            if (group.All(r => r.Temperature2m is null)) continue;

            var first = group.First();
            var dateStr = group.Key.ToString("yyyy-MM-dd");
            var dir = Path.Combine(basePath, $"location={first.LocationName}", $"date={dateStr}");
            Directory.CreateDirectory(dir);
            var file = Path.Combine(dir, "data.parquet");
            await ParquetSerializer.SerializeAsync(group.OrderBy(r => r.ValidTimeUtc).ToList(), file, cancellationToken: ct);
        }
    }

    /// <summary>
    /// EA rainfall writer. Partitioned by location/station/date so each file holds
    /// at most 96 rows (one day of 15-minute totals). Merges with any existing
    /// partition file, deduping on ObservedTimeUtc with last-write-wins — matches
    /// the METAR observation pattern so re-runs and backfills are idempotent.
    /// Path: location=.../station=&lt;name&gt;/date=&lt;yyyy-MM-dd&gt;/rainfall.parquet
    /// </summary>
    public static async Task WriteRainfallAsync(
        string basePath,
        IReadOnlyList<RainfallRow> rows,
        CancellationToken ct = default)
    {
        if (rows.Count == 0) return;

        foreach (var group in rows.GroupBy(r => new { r.LocationName, r.StationName, Date = r.ObservedTimeUtc.Date }))
        {
            var dateStr = group.Key.Date.ToString("yyyy-MM-dd");
            var dir = Path.Combine(
                basePath,
                $"location={group.Key.LocationName}",
                $"station={group.Key.StationName}",
                $"date={dateStr}");
            Directory.CreateDirectory(dir);

            var file = Path.Combine(dir, "rainfall.parquet");
            var existing = File.Exists(file)
                ? (await ParquetSerializer.DeserializeAsync<RainfallRow>(file, cancellationToken: ct)).ToList()
                : new List<RainfallRow>();

            var merged = existing
                .Concat(group)
                .GroupBy(r => r.ObservedTimeUtc)
                .Select(g => g.Last())
                .OrderBy(r => r.ObservedTimeUtc)
                .ToList();

            await ParquetSerializer.SerializeAsync(merged, file, cancellationToken: ct);
        }
    }

    /// <summary>
    /// Met Office Land Observations writer. Partitioned by location/geohash/date so
    /// each file holds at most 24 hourly rows. Merges with any existing partition
    /// file, deduping on ObservedTimeUtc with last-write-wins — rolling 48h fetches
    /// overlap heavily so repeated runs must be idempotent.
    /// Path: location=.../geohash=&lt;hash&gt;/date=&lt;yyyy-MM-dd&gt;/observations.parquet
    /// </summary>
    public static async Task WriteMetOfficeObservationsAsync(
        string basePath,
        IReadOnlyList<MetOfficeObservationRow> rows,
        CancellationToken ct = default)
    {
        if (rows.Count == 0) return;

        foreach (var group in rows.GroupBy(r => new { r.LocationName, r.Geohash, Date = r.ObservedTimeUtc.Date }))
        {
            var dateStr = group.Key.Date.ToString("yyyy-MM-dd");
            var dir = Path.Combine(
                basePath,
                $"location={group.Key.LocationName}",
                $"geohash={group.Key.Geohash}",
                $"date={dateStr}");
            Directory.CreateDirectory(dir);

            var file = Path.Combine(dir, "observations.parquet");
            var existing = File.Exists(file)
                ? (await ParquetSerializer.DeserializeAsync<MetOfficeObservationRow>(file, cancellationToken: ct)).ToList()
                : new List<MetOfficeObservationRow>();

            var merged = existing
                .Concat(group)
                .GroupBy(r => r.ObservedTimeUtc)
                .Select(g => g.Last())
                .OrderBy(r => r.ObservedTimeUtc)
                .ToList();

            await ParquetSerializer.SerializeAsync(merged, file, cancellationToken: ct);
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
