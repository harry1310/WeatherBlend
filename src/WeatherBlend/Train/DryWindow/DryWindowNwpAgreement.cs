using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Per-day NWP-ensemble consensus on hourly wet/dry classification.
/// Used by Phase 3n (regime-conditioned copula MC) to pick which Σ to apply
/// at predict time: high consensus → settled regime Σ; low consensus →
/// unsettled regime Σ.
///
/// Same agreement function is used at train time (offset_day forecasts)
/// and predict time (live reported forecasts) — the regime label of a day
/// must be consistent across both regimes so predict-time bucketing matches
/// the train-time slice the bucket's Σ was fit on.
/// </summary>
public static class DryWindowNwpAgreement
{
    /// <summary>Wet threshold matching 3a's hourly labeling: ≥0.1 mm/h = wet.</summary>
    public const double WetThresholdMmH = 0.1;

    /// <summary>Minimum models required per hour for that hour to count.
    /// Below this we drop the hour rather than let a 1-or-2-model "consensus"
    /// poison the day average.</summary>
    public const int MinModelsPerHour = 3;

    /// <summary>
    /// Compute the day's NWP consensus score from a (model × hour) precip
    /// matrix. NaN entries are treated as "model didn't run for this hour".
    /// Returns NaN if too few hours have ≥<see cref="MinModelsPerHour"/>
    /// models — the day can't be reliably regime-classified.
    /// </summary>
    /// <param name="precipMmH">[n_models, n_daytime_hours] precip values in mm/h. NaN = missing.</param>
    /// <returns>1.0 = unanimous every hour; 0.5 = max disagreement every hour. NaN = unclassifiable.</returns>
    public static double ComputePerDay(double[,] precipMmH)
    {
        int nModels = precipMmH.GetLength(0);
        int nHours = precipMmH.GetLength(1);

        double sumConsensus = 0.0;
        int validHours = 0;
        for (int h = 0; h < nHours; h++)
        {
            int wet = 0, total = 0;
            for (int m = 0; m < nModels; m++)
            {
                var v = precipMmH[m, h];
                if (double.IsNaN(v)) continue;
                total++;
                if (v >= WetThresholdMmH) wet++;
            }
            if (total < MinModelsPerHour) continue;
            double pWet = (double)wet / total;
            sumConsensus += Math.Max(pWet, 1.0 - pWet);
            validHours++;
        }
        if (validHours == 0) return double.NaN;
        return sumConsensus / validHours;
    }

    /// <summary>
    /// Load per-day (model × hour) precip matrices for the train-slice
    /// daytime windows from offset_day forecasts. Returns a dictionary
    /// keyed by target_date with the matrix dimensioned
    /// (canonical model order count, daytime_hours).
    ///
    /// Missing (model, hour) cells appear as NaN — agreement computation
    /// tolerates them and skips under-covered hours. Days where no hour
    /// makes the cut are simply absent from the dictionary.
    /// </summary>
    public static Dictionary<DateTime, double[,]> LoadOffsetDayPerNwpDaytime(
        string forecastsRoot, string locationName, int leadHours,
        IReadOnlyList<string> models,
        Func<DateOnly, (int Start, int EndExclusive)> daytimeHoursFor)
    {
        var glob = Path.Combine(
            forecastsRoot, $"location={locationName}", "model=*", "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");
        var sql = $@"
SELECT Model, ValidTimeUtc, CAST(Precipitation AS DOUBLE) AS precip
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LeadHours = {leadHours}
  AND (RunTimeSource IS NULL OR RunTimeSource = 'offset_day')
  AND Precipitation IS NOT NULL";
        return RunAgreementQuery(sql, models, daytimeHoursFor);
    }

    /// <summary>
    /// Load per-(target_date, lead) (model × hour) precip matrices for live
    /// predict-time agreement. Reads RunTimeSource='reported' rows for the
    /// canonical NWP set across the requested ValidTime window.
    /// </summary>
    public static Dictionary<DateTime, double[,]> LoadLivePerNwpDaytime(
        string forecastsRoot, string locationName, int leadHours,
        IReadOnlyList<string> models,
        Func<DateOnly, (int Start, int EndExclusive)> daytimeHoursFor,
        DateTime windowStart, DateTime windowEnd)
    {
        var glob = Path.Combine(
            forecastsRoot, $"location={locationName}", "model=*", "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");
        // No RunTimeSource filter — live tree may be 'reported' or a mix; we
        // want whatever the cycle wrote. ValidTime window narrows volume.
        var sql = $@"
SELECT Model, ValidTimeUtc, CAST(Precipitation AS DOUBLE) AS precip
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LeadHours = {leadHours}
  AND Precipitation IS NOT NULL
  AND ValidTimeUtc >= timestamp '{windowStart:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <  timestamp '{windowEnd:yyyy-MM-dd HH:mm:ss}'";
        return RunAgreementQuery(sql, models, daytimeHoursFor);
    }

    private static Dictionary<DateTime, double[,]> RunAgreementQuery(
        string sql, IReadOnlyList<string> models,
        Func<DateOnly, (int Start, int EndExclusive)> daytimeHoursFor)
    {
        var modelIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < models.Count; i++) modelIndex[models[i]] = i;

        // First pass: collect rows.
        var rowsByDate = new Dictionary<DateTime, List<(string Model, DateTime Vt, double Precip)>>();
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = sql;
            using var rdr = cmd.ExecuteReader();
            while (rdr.Read())
            {
                var model = rdr.GetString(0);
                if (!modelIndex.ContainsKey(model)) continue;  // not in canonical set
                var vt = DateTime.SpecifyKind(rdr.GetDateTime(1), DateTimeKind.Utc);
                var precip = rdr.GetDouble(2);
                var date = vt.Date;
                if (!rowsByDate.TryGetValue(date, out var list))
                {
                    list = new List<(string, DateTime, double)>();
                    rowsByDate[date] = list;
                }
                list.Add((model, vt, precip));
            }
        }

        // Second pass: build per-date matrix sliced to daytime.
        var result = new Dictionary<DateTime, double[,]>();
        foreach (var (date, rows) in rowsByDate)
        {
            var (startH, endH) = daytimeHoursFor(DateOnly.FromDateTime(date));
            int nHours = endH - startH;
            if (nHours <= 0) continue;

            var mat = new double[models.Count, nHours];
            for (int m = 0; m < models.Count; m++)
                for (int h = 0; h < nHours; h++)
                    mat[m, h] = double.NaN;

            foreach (var (model, vt, precip) in rows)
            {
                int hour = vt.Hour;
                if (hour < startH || hour >= endH) continue;
                int mi = modelIndex[model];
                int hi = hour - startH;
                // If the same (model, hour) appears more than once (multiple
                // RunTimes feeding the same ValidTime), the last write wins —
                // doesn't matter for the binary wet/dry call.
                mat[mi, hi] = precip;
            }

            result[date] = mat;
        }
        return result;
    }
}
