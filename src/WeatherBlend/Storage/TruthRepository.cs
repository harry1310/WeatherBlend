using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Storage;

/// <summary>
/// Single source of truth (literally) for reads from the three "truth" parquet
/// trees: ERA5 reanalysis (training + temperature/element verify), EA rainfall
/// gauges (precip + dry-window verify + render-site rainfall charts), and
/// METAR (render-site comparison line). Sits one layer above
/// <see cref="ParquetReader"/>.
///
/// What lifts in here:
///   * The 4-of-4 quarter-hour completeness gate for EA rainfall — previously
///     duplicated character-for-character in <c>RenderSiteCommand</c>,
///     <c>PrecipVerifyCommand</c>, and <c>DryWindowVerifyCommand</c>. A future
///     change to the gate (different data source, finer resolution) lands in
///     one place.
///   * The EA station slug → <c>StationName</c> mapping (
///     <c>"ea_<Slugify(StationConfig.Name)>"</c>) — also previously duplicated.
///   * The location filter, the "ValidTimeUtc/ObservedTimeUtc range", the
///     <c>hive_partitioning=false</c> + <c>union_by_name=true</c> incantation,
///     and the canonical row-mapping lambdas.
///
/// What does NOT lift in here: anything that joins truth to predictions or
/// applies derived logic (rolling MAE, Brier, drift). Those stay at the
/// command/render layer where they belong.
/// </summary>
public sealed class TruthRepository
{
    private readonly ILogger<TruthRepository> _log;
    private readonly AppConfig _cfg;

    public TruthRepository(ILogger<TruthRepository> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    /// <summary>
    /// Hourly ERA5 reading at the configured location, keyed by
    /// <c>ValidTimeUtc</c>. Default column is <c>Temperature2m</c>; element
    /// verify passes the per-target column name (e.g. <c>RelativeHumidity2m</c>,
    /// <c>WindSpeed10m</c>, <c>ShortwaveRadiation</c>, <c>CloudCover</c>).
    /// Empty dict on missing tree (degraded with a warning).
    /// </summary>
    public IReadOnlyDictionary<DateTime, double> GetEra5Hourly(
        DateTime start, DateTime end, CancellationToken ct, string column = "Temperature2m")
    {
        ct.ThrowIfCancellationRequested();
        ValidateColumnName(column);

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet"));
        var sql = $@"
SELECT ValidTimeUtc, {column}
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND {column} IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'";

        var rows = ParquetReader.Query(sql,
            r => (Time: r.GetDateTime(0), Val: r.GetDouble(1)),
            _log, $"ERA5 tree empty — {column} truth dict will be empty.", ct);
        return rows.ToDictionary(x => x.Time, x => x.Val);
    }

    /// <summary>
    /// EA rainfall in mm/h at the named station, aggregated from 15-min
    /// readings with the <strong>4-of-4 quarter-hour completeness gate</strong>:
    /// only hours with all four readings present pass through, so a partial
    /// hour can never flip a wet/dry indicator. Same rule the precip blender
    /// was trained on. <paramref name="stationName"/> is the
    /// <c>StationName</c> column value (e.g. <c>"Bellever"</c>), not the slug.
    /// </summary>
    public IReadOnlyDictionary<DateTime, double> GetEaHourlyRainfall(
        string stationName, DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet"));

        // end + 1h so the last hour's four 15-min readings are all in range.
        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND StationName  = '{stationName.Replace("'", "''")}'
  AND Value15MinMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end.AddHours(1):yyyy-MM-dd HH:mm:ss}'
GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1";

        var rows = ParquetReader.Query(sql,
            r => (Hour: r.GetDateTime(0), Mm: r.GetDouble(1)),
            _log, $"Rainfall tree empty for station {stationName}.", ct);
        return rows.ToDictionary(x => x.Hour, x => x.Mm);
    }

    /// <summary>
    /// Bulk fetch of EA hourly rainfall for every configured station, keyed
    /// by <c>StationName</c> (the friendly form, e.g. <c>"Bellever Dartmoor"</c>).
    /// Single SQL query rather than the per-station fan-out
    /// <see cref="GetEaHourlyRainfallByStation"/> uses — pays one read for
    /// many stations, useful when the caller wants every station's data at
    /// once (start-hour verify groups by station internally). Same 4-of-4
    /// quarter-hour completeness gate.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>>
        GetEaHourlyRainfallAllStations(DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet"));

        var sql = $@"
SELECT StationName,
       date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND Value15MinMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end.AddHours(1):yyyy-MM-dd HH:mm:ss}'
GROUP BY StationName, valid_time
HAVING COUNT(*) = 4
ORDER BY StationName, valid_time";

        var rows = ParquetReader.Query(sql,
            r => (Station: r.GetString(0), Hour: r.GetDateTime(1), Mm: r.GetDouble(2)),
            _log, "Rainfall tree empty — bulk-by-station read returns no rows.", ct);

        var result = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.Ordinal);
        foreach (var grp in rows.GroupBy(r => r.Station, StringComparer.Ordinal))
        {
            result[grp.Key] = grp.ToDictionary(r => r.Hour, r => r.Mm);
        }
        return result;
    }

    /// <summary>
    /// Multi-station fan-out of <see cref="GetEaHourlyRainfall"/>, taking
    /// <c>ea_*</c> slugs (the format predictions use in their
    /// <c>TruthStation</c> column). Slug → <c>StationName</c> mapping is
    /// resolved against <c>AppConfig.Location.Rainfall.Stations</c>; slugs
    /// without a config match get an empty dict and a warning, mirroring the
    /// pre-repo behaviour. Returned dict is keyed by the original slug so
    /// callers can keep their slug-based join logic intact.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>>
        GetEaHourlyRainfallByStation(
            IReadOnlyList<string> slugs, DateTime start, DateTime end, CancellationToken ct)
    {
        if (slugs.Count == 0)
            return new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase);

        var stationNamesBySlug = _cfg.Location.Rainfall.Stations.ToDictionary(
            s => StationSlug.WithEaPrefix(s.Name),
            s => s.Name,
            StringComparer.OrdinalIgnoreCase);

        var result = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase);
        foreach (var slug in slugs)
        {
            if (!stationNamesBySlug.TryGetValue(slug, out var stationName))
            {
                _log.LogWarning("Rainfall truth: no config station for slug {Slug}.", slug);
                result[slug] = new Dictionary<DateTime, double>();
                continue;
            }
            result[slug] = GetEaHourlyRainfall(stationName, start, end, ct);
        }
        return result;
    }

    /// <summary>
    /// METAR observations from the named ICAO station. Returns chronological
    /// (ObservedTimeUtc, Temperature2m) tuples — list rather than dict because
    /// chart code wants the time-sorted sequence. Empty list when
    /// <paramref name="station"/> is blank or the tree is empty.
    /// </summary>
    public IReadOnlyList<(DateTime ObservedTimeUtc, double Temperature2m)> GetMetarTemperature(
        string station, DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (string.IsNullOrWhiteSpace(station))
        {
            _log.LogWarning("No METAR station provided — returning empty observation list.");
            return Array.Empty<(DateTime, double)>();
        }

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.ObservationsPath, "**", "*.parquet"));
        var sql = $@"
SELECT ObservedTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND Station = '{station.Replace("'", "''")}'
  AND Temperature2m IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ObservedTimeUtc";

        return ParquetReader.Query(sql,
            r => (r.GetDateTime(0), r.GetDouble(1)),
            _log, $"METAR tree empty for station {station} — observation list will be empty.", ct);
    }

    /// <summary>
    /// Met Office DataHub Land Observations temperature for the configured
    /// geohash cell. Geohash gcj0z3 (~22 km NNW of Bonehill at Cocktree
    /// Throat / Taw Green, north Devon, ~120-150m elevation) is closer and
    /// less elevation-biased vs Bonehill's 393m than the EGTE METAR (~30 km
    /// E, 31m), so it ships on the temp skill chart as a second cross-check
    /// alongside METAR. List rather than dict — chart code wants the
    /// time-sorted sequence, same shape as <see cref="GetMetarTemperature"/>.
    /// Empty list when the obs tree hasn't landed for this location yet.
    /// </summary>
    public IReadOnlyList<(DateTime ObservedTimeUtc, double Temperature2m)> GetMetOfficeObsTemperature(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.MetOfficeObsPath, "**", "*.parquet"));
        var sql = $@"
SELECT ObservedTimeUtc, AirTemperature
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND AirTemperature IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ObservedTimeUtc";

        return ParquetReader.Query(sql,
            r => (r.GetDateTime(0), r.GetDouble(1)),
            _log, "Met Office obs tree empty — observation list will be empty.", ct);
    }

    /// <summary>
    /// Defensive guard against SQL injection via <c>column</c> — the only repo
    /// input that's interpolated as a SQL identifier rather than a quoted
    /// string. Element verify is the only caller that passes anything other
    /// than the default; the column names there come from a closed enum
    /// (<c>Temperature2m</c>, <c>RelativeHumidity2m</c>, <c>WindSpeed10m</c>,
    /// <c>ShortwaveRadiation</c>, <c>CloudCover</c>) but if a future caller
    /// passes a user-controlled string, this stops the obvious shape of
    /// attack.
    /// </summary>
    private static void ValidateColumnName(string column)
    {
        if (string.IsNullOrEmpty(column) || column.Length > 64)
            throw new ArgumentException("Column name must be 1..64 chars.", nameof(column));
        for (int i = 0; i < column.Length; i++)
        {
            var c = column[i];
            if (!char.IsLetterOrDigit(c) && c != '_')
                throw new ArgumentException(
                    $"Column name '{column}' contains a non-identifier character.", nameof(column));
        }
    }

}
