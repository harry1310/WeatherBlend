using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Humidity;

/// <summary>
/// Spec-driven SQL pivot + row composition for the humidity blender.
/// Pulls per-model <c>relative_humidity_2m</c> + <c>dew_point_2m</c>; joins to ERA5
/// <c>relative_humidity_2m</c> as truth.
///
/// Model membership comes from <c>blenders.humidity</c>: 5 models at lead 24
/// (no UKMO), 4 at lead 48/72 (no UKMO + no MF). Best-single is picked over the
/// per-model rh slots (the spec's first N FeatureNames).
///
/// Layout per N active models: N rh + N dp + 3 spread (rh) + 4 calendar = 2N + 7.
/// For N=5 → 17 features. For N=4 → 15.
/// </summary>
public static class HumidityFeatureBuilder
{
    public const string SpecTarget = "humidity";
    public const string SpecFeatureSet = "default";

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var blender = blendersCfg.Get(SpecTarget, SpecFeatureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var orderedRequired = TempFeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = TempFeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = TempFeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h.");

        var names = new List<string>();
        foreach (var m in orderedModels) names.Add($"rh_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"dp_{TempFeatureBuilder.ShortName(m)}");
        names.AddRange(new[] { "rh_mean", "rh_std", "rh_range" });
        names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });

        return new BlenderSpec
        {
            Target = SpecTarget,
            FeatureSet = SpecFeatureSet,
            LeadHours = leadHours,
            RequiredModels = orderedRequired,
            OptionalModels = orderedOptional,
            Models = orderedModels,
            FeatureNames = names,
            DataSource = BlenderDataSource.OpenMeteoPreviousRuns,
            Tier = SpecFeatureSet,
            UkvStrategy = null,
        };
    }

    public static List<RegressionTrainingRow> BuildForLead(
        string forecastsPath, string era5Path, string locationName, BlenderSpec spec, CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob  = Norm(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = Norm(Path.Combine(era5Path, "**", "*.parquet"));
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var n = spec.Models.Count;

        var pivotRh = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN RelativeHumidity2m END) AS rh_{TempFeatureBuilder.ShortName(m)}"));
        var pivotDp = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN DewPoint2m END) AS dp_{TempFeatureBuilder.ShortName(m)}"));
        var selectRh = string.Join(", ", spec.Models.Select(m => $"p.rh_{TempFeatureBuilder.ShortName(m)}"));
        var selectDp = string.Join(", ", spec.Models.Select(m => $"p.dp_{TempFeatureBuilder.ShortName(m)}"));
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m =>
                $"p.rh_{TempFeatureBuilder.ShortName(m)} IS NOT NULL AND p.dp_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.rh_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RelativeHumidity2m, DewPoint2m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND RelativeHumidity2m IS NOT NULL
      AND DewPoint2m IS NOT NULL
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotRh},
        {pivotDp}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, RelativeHumidity2m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND RelativeHumidity2m IS NOT NULL
)
SELECT p.ValidTimeUtc, {selectRh}, {selectDp}, e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var rh = new double[n];
        var dp = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++) rh[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++) dp[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            var truth = r.GetDouble(1 + 2 * n);
            rows.Add(ComposeRow(spec, valid, rh, dp, truth));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec, DateTime validTimeUtc,
        IReadOnlyList<double> rh, IReadOnlyList<double> dp, double era5Rh)
    {
        var n = spec.Models.Count;
        if (rh.Count != n) throw new ArgumentException($"Expected {n} RH values", nameof(rh));
        if (dp.Count != n) throw new ArgumentException($"Expected {n} dewpoint values", nameof(dp));

        // Spread over RH (NaN-safe).
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = rh[i];
            if (double.IsNaN(x)) continue;
            sum += x; sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            present++;
        }
        var mean  = present == 0 ? double.NaN : sum / present;
        var var0  = present == 0 ? double.NaN : Math.Max(0.0, (sumSq / present) - (mean * mean));
        var std   = double.IsNaN(var0) ? double.NaN : Math.Sqrt(var0);
        var range = present == 0 ? double.NaN : max - min;

        var v = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle  = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < n; i++) features[i] = (float)rh[i];
        for (int i = 0; i < n; i++) features[n + i] = (float)dp[i];
        features[2 * n + 0] = (float)mean;
        features[2 * n + 1] = (float)std;
        features[2 * n + 2] = (float)range;
        features[2 * n + 3] = (float)Math.Sin(hourAngle);
        features[2 * n + 4] = (float)Math.Cos(hourAngle);
        features[2 * n + 5] = (float)Math.Sin(doyAngle);
        features[2 * n + 6] = (float)Math.Cos(doyAngle);

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Rh,
        };
    }

    private static string Norm(string p) => p.Replace('\\', '/').Replace("'", "''");
}
