using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Common;

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
/// Layout per N active models: N rh + N dp + 3 spread (rh) + 4 calendar
/// + 6 cross-variable ensemble means = 2N + 13. For N=5 → 23 features.
///
/// The 6 aux means (temp / dew-point depression / cloud low/mid/high / wind)
/// were added 2026-06-10 after the humidity-aifs-bakeoff: the 17-feature
/// blend LOST to raw AIFS at every lead (the live bundle's own test slices
/// agreed); +aux beats AIFS at 24/48h (−1.3/−2.6%) and closes 72h to +0.6%.
/// Same pattern the radiation blender productionised 2026-06-04. They sit
/// at the END of the feature vector so every earlier offset (spread at 2N,
/// calendar at 2N+3) is unchanged for pre-change bundles, and ComposeRow is
/// schema-gated on "temp_mean" so predict keeps working against old
/// 17-feature bundles until the next retrain mints the new layout.
/// </summary>
public static class HumidityFeatureBuilder
{
    public const string SpecTarget = "humidity";
    public const string SpecFeatureSet = "default";

    /// <summary>Cross-variable NWP-ensemble means appended to the feature
    /// vector (see class doc). Order is load-bearing — ComposeRow fills the
    /// tail slots positionally.</summary>
    public static readonly string[] RichAuxNames =
        { "temp_mean", "dewdep_mean", "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean", "wind_mean" };

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build;
        // only the humidity layout below is builder-specific.
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
            var names = new List<string>();
            foreach (var m in orderedModels) names.Add($"rh_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"dp_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[] { "rh_mean", "rh_std", "rh_range" });
            names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });
            names.AddRange(RichAuxNames);
            return names;
        });

    public static List<RegressionTrainingRow> BuildForLead(
        string forecastsPath, string era5Path, string locationName, BlenderSpec spec, CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob  = SqlGlob.Escape(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = SqlGlob.Escape(Path.Combine(era5Path, "**", "*.parquet"));
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
    SELECT ValidTimeUtc, Model, RelativeHumidity2m, DewPoint2m, Temperature2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh, WindSpeed10m,
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
        {pivotDp},
        AVG(Temperature2m)              AS temp_mean,
        AVG(Temperature2m - DewPoint2m) AS dewdep_mean,
        AVG(CloudCoverLow)              AS cloud_low_mean,
        AVG(CloudCoverMid)              AS cloud_mid_mean,
        AVG(CloudCoverHigh)             AS cloud_high_mean,
        AVG(WindSpeed10m)               AS wind_mean
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, RelativeHumidity2m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND RelativeHumidity2m IS NOT NULL
)
SELECT p.ValidTimeUtc, {selectRh}, {selectDp},
       p.temp_mean, p.dewdep_mean, p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean, p.wind_mean,
       e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        int nAux = RichAuxNames.Length;
        var rows = new List<RegressionTrainingRow>();
        var rh = new double[n];
        var dp = new double[n];
        var aux = new double[nAux];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++) rh[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++) dp[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            for (int i = 0; i < nAux; i++) aux[i] = r.IsDBNull(1 + 2 * n + i) ? double.NaN : r.GetDouble(1 + 2 * n + i);
            var truth = r.GetDouble(1 + 2 * n + nAux);
            rows.Add(ComposeRow(spec, valid, rh, dp, truth, aux));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec, DateTime validTimeUtc,
        IReadOnlyList<double> rh, IReadOnlyList<double> dp, double era5Rh,
        IReadOnlyList<double>? aux = null)
    {
        var n = spec.Models.Count;
        if (rh.Count != n) throw new ArgumentException($"Expected {n} RH values", nameof(rh));
        if (dp.Count != n) throw new ArgumentException($"Expected {n} dewpoint values", nameof(dp));
        // Schema-gated aux block: pre-2026-06-10 bundles have no temp_mean in
        // their saved spec and keep the legacy 2N+7 layout; new bundles
        // require the 6 aux means.
        var hasAux = spec.FeatureNames.Contains("temp_mean");
        if (hasAux && (aux is null || aux.Count != RichAuxNames.Length))
            throw new ArgumentException(
                $"Spec carries the aux block but {(aux is null ? "no" : aux.Count.ToString())} aux values were supplied.",
                nameof(aux));

        // Spread over RH (NaN-safe).
        var spread = InterModelSpread.From(rh);

        var v = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var cal = CalendarFeatures.From(v);

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < n; i++) features[i] = (float)rh[i];
        for (int i = 0; i < n; i++) features[n + i] = (float)dp[i];
        features[2 * n + 0] = (float)spread.Mean;
        features[2 * n + 1] = (float)spread.Std;
        features[2 * n + 2] = (float)spread.Range;
        features[2 * n + 3] = cal.HourSin;
        features[2 * n + 4] = cal.HourCos;
        features[2 * n + 5] = cal.DoySin;
        features[2 * n + 6] = cal.DoyCos;
        if (hasAux)
            for (int i = 0; i < RichAuxNames.Length; i++)
                features[2 * n + 7 + i] = (float)aux![i];

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Rh,
        };
    }
}
