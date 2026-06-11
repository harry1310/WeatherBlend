using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// SQL pivot + row composition for the wind blender — spec-driven.
/// Pulls per-model <c>wind_speed_10m</c> and <c>wind_direction_10m</c> from the
/// Previous Runs forecast tree and joins to ERA5 <c>wind_speed_10m</c> as truth.
///
/// Model membership comes from <c>blenders.wind</c> in config.yaml. MétéoFrance
/// is excluded entirely there (Open-Meteo Previous Runs ships no MF wind data —
/// audit 2026-04-25). UKMO sits in <c>optionalModels</c> — its rows are kept
/// when UKMO is silent (NaN) but contribute when present.
///
/// Feature layout per N active models:
///   N per-model wind_spd
///   2N per-model wind_dir (sin block, then cos block)
///   3 ensemble spread (mean, std, range — over wind_spd)
///   4 cyclical calendar
/// Total = 3N + 7. For N=5 → 22 features (matches legacy shape).
/// </summary>
public static class WindFeatureBuilder
{
    public const string SpecTarget = "wind";
    public const string SpecFeatureSet = "default";

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build;
        // only the wind layout below is builder-specific.
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
            var names = new List<string>();
            foreach (var m in orderedModels) names.Add($"wind_spd_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"wind_dir_sin_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"wind_dir_cos_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[] { "wind_spd_mean", "wind_spd_std", "wind_spd_range" });
            names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });
            return names;
        });

    public static List<RegressionTrainingRow> BuildForLead(
        string forecastsPath,
        string era5Path,
        string locationName,
        BlenderSpec spec,
        CancellationToken ct = default,
        string? floorOverride = null)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = SqlGlob.Escape(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = SqlGlob.Escape(Path.Combine(era5Path, "**", "*.parquet"));
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var n = spec.Models.Count;

        // No training-window floor by default: the floor bake-off (2026-06-09, common
        // OOS holdout) showed dropping the old UkmoCleanWindowStart=2024-09 floor IMPROVES
        // wind at every lead (−0.8..−2.5%) — the extra ~Feb–Aug 2024 rows help and UKMO
        // (optional) is NaN-tolerated by LightGBM. floorOverride re-imposes a floor if ever needed.
        var floorClause = floorOverride is null ? "" : $"AND ValidTimeUtc >= TIMESTAMP '{floorOverride}'";

        var pivotSpd = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN WindSpeed10m END) AS spd_{TempFeatureBuilder.ShortName(m)}"));
        var pivotDir = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN WindDirection10m END) AS dir_{TempFeatureBuilder.ShortName(m)}"));
        var selectSpd = string.Join(", ", spec.Models.Select(m => $"p.spd_{TempFeatureBuilder.ShortName(m)}"));
        var selectDir = string.Join(", ", spec.Models.Select(m => $"p.dir_{TempFeatureBuilder.ShortName(m)}"));

        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m =>
                $"p.spd_{TempFeatureBuilder.ShortName(m)} IS NOT NULL AND p.dir_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.spd_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, WindSpeed10m, WindDirection10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND WindSpeed10m IS NOT NULL
      AND WindDirection10m IS NOT NULL
      {floorClause}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotSpd},
        {pivotDir}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, WindSpeed10m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND WindSpeed10m IS NOT NULL
      {floorClause}
)
SELECT p.ValidTimeUtc, {selectSpd}, {selectDir}, e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var spd = new double[n];
        var dir = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++) spd[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++) dir[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            var truth = r.GetDouble(1 + 2 * n);
            rows.Add(ComposeRow(spec, valid, spd, dir, truth));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> speeds,
        IReadOnlyList<double> directionsDeg,
        double era5WindSpeed)
    {
        var n = spec.Models.Count;
        if (speeds.Count != n) throw new ArgumentException($"Expected {n} speeds", nameof(speeds));
        if (directionsDeg.Count != n) throw new ArgumentException($"Expected {n} directions", nameof(directionsDeg));

        // Spread over wind_spd (NaN-safe so optional-model NaN slots are skipped).
        var spread = InterModelSpread.From(speeds);

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var cal = CalendarFeatures.From(v);

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < n; i++) features[i] = (float)speeds[i];
        // Wind-direction unit vectors: NaN propagates via Math.Sin/Cos.
        for (int i = 0; i < n; i++)
        {
            var rad = directionsDeg[i] * Math.PI / 180.0;
            features[n + i] = (float)Math.Sin(rad);
        }
        for (int i = 0; i < n; i++)
        {
            var rad = directionsDeg[i] * Math.PI / 180.0;
            features[2 * n + i] = (float)Math.Cos(rad);
        }
        features[3 * n + 0] = (float)spread.Mean;
        features[3 * n + 1] = (float)spread.Std;
        features[3 * n + 2] = (float)spread.Range;
        features[3 * n + 3] = cal.HourSin;
        features[3 * n + 4] = cal.HourCos;
        features[3 * n + 5] = cal.DoySin;
        features[3 * n + 6] = cal.DoyCos;

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5WindSpeed,
        };
    }
}
