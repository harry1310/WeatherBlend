using DuckDB.NET.Data;

namespace WeatherBlend.Train;

/// <summary>
/// Builds the 13-feature training dataset for the temperature blender, one lead at a time.
///
/// Data flow for <see cref="BuildForLead"/>:
///   1. Read all forecast parquet across 6 models via DuckDB (hive_partitioning=false,
///      in-file Model/LocationName columns are authoritative — see CLAUDE.md gotchas).
///   2. Filter to Previous Runs rows (<c>RunTimeSource = 'offset_day'</c>) at the exact
///      requested lead so the blender sees real issue-time vs. valid-time pairs.
///   3. Dedupe to the latest forecast per (ValidTime, Model) — `collect` runs every 3h
///      and stacks multiple forecasts per valid time; we keep the most recent RunTime.
///   4. Pivot so each valid_time has one row with 6 per-model temperature columns.
///   5. Inner-join to ERA5 on ValidTimeUtc.
///   6. Drop any row where any of the 6 model forecasts is null (blender shouldn't
///      learn "model X missing => ..." patterns).
///   7. Compute spread + cyclical calendar features in .NET.
/// </summary>
public static class FeatureBuilder
{
    /// <summary>Feature column order — load-bearing. Persisted in feature_schema.json.</summary>
    public static readonly IReadOnlyList<string> FeatureNames = new[]
    {
        "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem",
        "temp_mean", "temp_std", "temp_range",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    // Map Open-Meteo model ids → feature column suffix. Ordering matches FeatureNames[0..5].
    public static readonly IReadOnlyList<(string ModelId, string Col)> ModelColumns = new[]
    {
        ("gfs_seamless",         "temp_gfs"),
        ("ecmwf_ifs025",         "temp_ecmwf"),
        ("icon_seamless",        "temp_icon"),
        ("meteofrance_seamless", "temp_mf"),
        ("ukmo_seamless",        "temp_ukmo"),
        ("gem_seamless",         "temp_gem"),
    };

    /// <summary>
    /// Produces one dataset per lead ∈ {24, 48, 72}. Filters on
    /// RunTimeSource = 'offset_day' AND LeadHours = lead — each training row comes
    /// from the Open-Meteo Previous Runs API with an exact lead horizon. Any lead
    /// value is technically accepted but only 24/48/72 have backfill behind them.
    /// </summary>
    public static List<TrainingRow> BuildForLead(
        string forecastsPath,
        string era5Path,
        string locationName,
        int leadHours,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = NormaliseGlob(Path.Combine(era5Path, "**", "*.parquet"));

        // RunTimeSource filter is load-bearing: historical-forecast rows with
        // RunTime synthesised to midnight can land on LeadHours=24/48/72 by
        // coincidence and would contaminate the lead-specific training set.
        // Without the filter, "24h forecast" would include 0h analyses issued
        // at midnight.
        var fcWhere = $"LocationName = '{locationName}' AND RunTimeSource = 'offset_day' AND LeadHours = {leadHours} AND Temperature2m IS NOT NULL";

        // One row per (valid_time, model) — latest RunTime wins when live-collected
        // data has stacked multiple forecasts for the same valid time.
        // Wind direction is pulled too (mean across models) so evaluation can slice
        // by wind quadrant without rebuilding the whole pipeline.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, Temperature2m, WindDirection10m,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN Temperature2m END) AS temp_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN Temperature2m END) AS temp_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN Temperature2m END) AS temp_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN Temperature2m END) AS temp_mf,
        MAX(CASE WHEN Model = 'ukmo_seamless'        THEN Temperature2m END) AS temp_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN Temperature2m END) AS temp_gem,
        AVG(WindDirection10m) AS wind_dir_mean
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, Temperature2m AS era5_temp
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND Temperature2m IS NOT NULL
)
SELECT
    p.ValidTimeUtc,
    p.temp_gfs, p.temp_ecmwf, p.temp_icon, p.temp_mf, p.temp_ukmo, p.temp_gem,
    p.wind_dir_mean,
    e.era5_temp
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
WHERE p.temp_gfs   IS NOT NULL
  AND p.temp_ecmwf IS NOT NULL
  AND p.temp_icon  IS NOT NULL
  AND p.temp_mf    IS NOT NULL
  AND p.temp_ukmo  IS NOT NULL
  AND p.temp_gem   IS NOT NULL
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<TrainingRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();

            var valid = r.GetDateTime(0);
            var t = new double[6];
            for (int i = 0; i < 6; i++) t[i] = r.GetDouble(1 + i);

            var windDirMean = r.IsDBNull(7) ? double.NaN : r.GetDouble(7);
            var era5Temp = r.GetDouble(8);

            rows.Add(ComposeRow(valid, t, windDirMean, era5Temp));
        }

        return rows;
    }

    /// <summary>
    /// Given raw per-model temperatures + valid_time, compute the 13 features.
    /// Factored out for unit testing without DuckDB in the loop.
    /// </summary>
    public static TrainingRow ComposeRow(
        DateTime validTimeUtc,
        double[] perModelTemps,
        double windDirMeanDeg,
        double era5Temp)
    {
        if (perModelTemps.Length != 6)
            throw new ArgumentException("Expected 6 model temperatures", nameof(perModelTemps));

        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < 6; i++)
        {
            var x = perModelTemps[i];
            sum += x;
            sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
        }
        var mean = sum / 6.0;
        // Population std (N, not N-1) — consistent with numpy default, fine for a feature.
        var variance = Math.Max(0.0, (sumSq / 6.0) - (mean * mean));
        var std = Math.Sqrt(variance);
        var range = max - min;

        // Ensure DateTime kind is UTC so later comparisons don't silently shift.
        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);

        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doy = v.DayOfYear;
        var doyAngle = 2.0 * Math.PI * (doy - 1) / 365.0;

        return new TrainingRow
        {
            ValidTimeUtc = v,
            TempGfs   = (float)perModelTemps[0],
            TempEcmwf = (float)perModelTemps[1],
            TempIcon  = (float)perModelTemps[2],
            TempMf    = (float)perModelTemps[3],
            TempUkmo  = (float)perModelTemps[4],
            TempGem   = (float)perModelTemps[5],
            TempMean  = (float)mean,
            TempStd   = (float)std,
            TempRange = (float)range,
            HourSin   = (float)Math.Sin(hourAngle),
            HourCos   = (float)Math.Cos(hourAngle),
            DoySin    = (float)Math.Sin(doyAngle),
            DoyCos    = (float)Math.Cos(doyAngle),
            WindDirMean = (float)windDirMeanDeg,
            Era5Temp  = (float)era5Temp,
        };
    }

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");
}
