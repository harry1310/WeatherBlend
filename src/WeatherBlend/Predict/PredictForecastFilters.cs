using WeatherBlend.Train;

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
    ///
    /// <para>When <paramref name="leadHoursLowerBound"/> is set, the filter additionally
    /// enforces <c>RunTimeUtc &lt;= ValidTimeUtc − leadHoursLowerBound</c>, which is the
    /// "no younger than lead-N" constraint the lead-bucketed blenders need to stay on
    /// their training distribution. Concretely: a lead-24 model was trained on Open-Meteo
    /// <c>previous_day_1</c> rows (forecasts that existed ≥ 24h before the valid time);
    /// at predict time we want the freshest cycle still satisfying that bound, not the
    /// freshest cycle period — which can be 0–6h fresher than training and produces
    /// out-of-distribution inputs once we predict for valid times within 24h of anchor.
    /// Defaults to <c>null</c> = no lead constraint, preserving the original behaviour
    /// for callers that already shoot at <c>anchor + L</c> exactly (where the constraint
    /// is automatically satisfied by the latest cycle).</para>
    /// </summary>
    public static string LiveCycleAsOf(
        string locationName,
        DateTime asOfRunTime,
        DateTime earliestValid,
        DateTime latestValid,
        int? leadHoursLowerBound = null)
    {
        var loc = locationName.Replace("'", "''");
        // Defensive Model IN (...) filter against the canonical model list — without
        // it, DuckDB happily reads any model partition under data/forecasts/ even when
        // it has weird schema/dictionary encoding. Interpolated *_hourly variants had
        // all-NULL RunTimeUtc dictionary blocks that crashed the union-by-name read in
        // CI 2026-04-28 even after the partitions were thought to be deleted.
        // Restricting to the canonical model list means stray experimental partitions
        // can sit on disk without breaking predict.
        var modelList = string.Join(", ",
            TempFeatureBuilder.CanonicalModelOrder.Select(m => $"'{m.Replace("'", "''")}'"));
        var leadClause = leadHoursLowerBound is int L
            ? $"\n      AND RunTimeUtc <= ValidTimeUtc - INTERVAL {L} HOUR"
            : "";
        return $@"LocationName = '{loc}'
      AND Model IN ({modelList})
      AND (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
      AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'{leadClause}";
    }
}
