using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Single source of truth for the hourly-rainfall WET LABEL used by every
/// precipitation occurrence builder:
/// <list type="bullet">
///   <item>lean 3a (<see cref="PrecipFeatureBuilder"/>),</item>
///   <item>rich 3c (<see cref="PrecipRichFeatureBuilder"/>),</item>
///   <item>rich+oro 3o (<see cref="Oro.PrecipRichOroFeatureBuilder"/>, which
///         builds on the rich rows).</item>
/// </list>
///
/// Both truth sources resolve here so the EA-vs-WeatherLink switch lives in ONE
/// place rather than being duplicated per builder:
/// <list type="bullet">
///   <item>EA Hydrology 15-min gauges — hourly SUM(Value15MinMm) with the
///         4-of-4 completeness rule (HAVING COUNT(*) = 4); never fabricate a
///         partial hour.</item>
///   <item>WeatherLink hourly gauges (e.g. the Lands End cove gauge for Sennen)
///         — already hourly, so SUM(RainfallMm) with no completeness rule, read
///         from their own <c>location=</c> partition (no StationName).</item>
/// </list>
/// </summary>
public static class PrecipTruthLoader
{
    /// <summary>
    /// Config-driven overload — resolves the EA-vs-WeatherLink source from the station's
    /// <see cref="RainfallStationConfig.Source"/>, so callers pass the station and never
    /// branch on the source themselves.
    /// </summary>
    public static Dictionary<DateTime, double> LoadHourlyRain(
        RainfallStationConfig station, StorageConfig storage, string locationName,
        DateTime? minValidTime, CancellationToken ct)
    {
        var (wlPath, wlKey) = station.WeatherLinkTruth(storage);
        return LoadHourlyRain(storage.RainfallPath, locationName, station.Name, minValidTime, ct, wlPath, wlKey);
    }

    /// <summary>
    /// Hourly rainfall (mm) keyed by hour-of-observation UTC. Pass
    /// <paramref name="weatherLinkTruthPath"/> (+ <paramref name="weatherLinkTruthLocation"/>)
    /// to label a station against a WeatherLink gauge instead of the EA tree;
    /// leave them null for the default EA Hydrology path.
    /// </summary>
    public static Dictionary<DateTime, double> LoadHourlyRain(
        string rainfallPath,
        string locationName,
        string stationName,
        DateTime? minValidTime,
        CancellationToken ct,
        string? weatherLinkTruthPath = null,
        string? weatherLinkTruthLocation = null)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var minObservedFilter = minValidTime.HasValue
            ? $"  AND ObservedTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'\n"
            : string.Empty;

        string sql, sourcePath;
        if (!string.IsNullOrEmpty(weatherLinkTruthPath))
        {
            // WeatherLink truth (e.g. the Lands End cove gauge for Sennen): the
            // collector already aggregates to HOURLY RainfallMm — one row per hour —
            // so SUM-by-hour with NO 4-of-4 completeness rule, and the rain lives
            // under its own location= partition (no StationName).
            sourcePath = weatherLinkTruthPath;
            var wlGlob = SqlGlob.Escape(Path.Combine(weatherLinkTruthPath, "**", "*.parquet"));
            var escWlLoc = (weatherLinkTruthLocation ?? string.Empty).Replace("'", "''");
            sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(RainfallMm) AS mm
FROM read_parquet('{wlGlob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escWlLoc}'
  AND RainfallMm IS NOT NULL
{minObservedFilter}GROUP BY 1
ORDER BY 1;
";
        }
        else
        {
            sourcePath = rainfallPath;
            var rnGlob = SqlGlob.Escape(Path.Combine(rainfallPath, "**", "*.parquet"));
            var escLocation = locationName.Replace("'", "''");
            var escStation  = stationName.Replace("'", "''");
            sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm
FROM read_parquet('{rnGlob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLocation}'
  AND StationName  = '{escStation}'
  AND Value15MinMm IS NOT NULL
{minObservedFilter}GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1;
";
        }

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var dict = new Dictionary<DateTime, double>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                dict[r.GetDateTime(0)] = r.GetDouble(1);
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            // DuckDB's raw "No files found" crash is nearly un-diagnosable inside a
            // CI log — the actual mistake is upstream (workflow didn't sync the
            // rainfall tree). Translate to a clear InvalidOperationException with the
            // path that was checked so operators can go straight to the fix.
            throw new InvalidOperationException(
                $"Rainfall truth tree is empty at '{sourcePath}'. " +
                $"The precipitation occurrence blenders (3a/3c/3o) need the hourly " +
                $"rain observations for the wet label + persistence features anchored " +
                $"at run_time = valid_time - leadHours; sync the truth tree " +
                $"(data/truth/rainfall, or data/truth/weatherlink for a WeatherLink-" +
                $"sourced station) from R2 before invoking the command.",
                ex);
        }
        return dict;
    }
}
