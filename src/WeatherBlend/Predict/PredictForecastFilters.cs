namespace WeatherBlend.Predict;

/// <summary>
/// Small SQL WHERE-fragment builders shared by the temperature + precipitation predict paths.
/// Keeps the "live-cycle, as-of" filter identical across targets so they stay in sync if the
/// semantics ever change. Not a general-purpose query builder — just enough glue to avoid
/// copy-pasted literal strings.
/// </summary>
public static class PredictForecastFilters
{
    /// <summary>
    /// WHERE-fragment selecting forecast rows that belong to the live cycle tree for
    /// <paramref name="locationName"/>, issued at or before <paramref name="asOfRunTime"/>,
    /// with valid-time in [earliestValid, latestValid]. Historical-forecast rows
    /// (RunTimeSource='offset_day') are excluded. Single-quote literals in the location
    /// name are escaped; callers interpolate the result directly.
    /// </summary>
    public static string LiveCycleAsOf(
        string locationName,
        DateTime asOfRunTime,
        DateTime earliestValid,
        DateTime latestValid)
    {
        var loc = locationName.Replace("'", "''");
        return $@"LocationName = '{loc}'
      AND (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
      AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'";
    }
}
