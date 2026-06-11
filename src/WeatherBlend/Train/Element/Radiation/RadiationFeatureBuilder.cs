using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Radiation;

/// <summary>
/// Spec-driven SQL pivot + row composition for the shortwave-radiation blender.
///
/// Model membership comes from <c>blenders.radiation</c>: 5 models at lead 24
/// (gfs/ecmwf/icon/mf/gem; UKMO permanently excluded — its SW field is
/// essentially never populated), 4 at lead 48/72 (no MF — live API horizon ~36h).
///
/// Layout per N active models: N sw + N direct + N diffuse + 3 spread (sw)
/// + 4 calendar = 3N + 7. For N=5 → 22 features. For N=4 → 19.
/// </summary>
public static class RadiationFeatureBuilder
{
    public const string SpecTarget = "radiation";
    public const string SpecFeatureSet = "default";

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build;
        // only the radiation layout below is builder-specific.
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
            var names = new List<string>();
            foreach (var m in orderedModels) names.Add($"sw_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"direct_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"diffuse_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[] { "sw_mean", "sw_std", "sw_range" });
            names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });
            // Cross-variable RICH features (productionised 2026-06-04 after the
            // radiation-rich-bakeoff: rich-mean beats lean at every lead, −0.9/−0.5/
            // −2.5% at 24/48/72h). Cloud cover is the physical inverse-driver of SW;
            // RH adds the haze/humidity attenuation signal. Ensemble means (AVG over
            // present models), NaN-safe. ComposeRow writes these only when the loaded
            // spec carries them, so older lean bundles keep predicting unchanged.
            names.AddRange(new[] { "cloud_mean", "rh_mean" });
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

        var pivotSw = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN ShortwaveRadiation END) AS sw_{TempFeatureBuilder.ShortName(m)}"));
        var pivotDr = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN DirectRadiation END) AS dr_{TempFeatureBuilder.ShortName(m)}"));
        var pivotDf = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN DiffuseRadiation END) AS df_{TempFeatureBuilder.ShortName(m)}"));
        var selectSw = string.Join(", ", spec.Models.Select(m => $"p.sw_{TempFeatureBuilder.ShortName(m)}"));
        var selectDr = string.Join(", ", spec.Models.Select(m => $"p.dr_{TempFeatureBuilder.ShortName(m)}"));
        var selectDf = string.Join(", ", spec.Models.Select(m => $"p.df_{TempFeatureBuilder.ShortName(m)}"));
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.sw_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.sw_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, ShortwaveRadiation, DirectRadiation, DiffuseRadiation,
           CloudCover, RelativeHumidity2m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotSw},
        {pivotDr},
        {pivotDf},
        AVG(CloudCover)        AS cloud_mean,
        AVG(RelativeHumidity2m) AS rh_mean
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, ShortwaveRadiation AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND ShortwaveRadiation IS NOT NULL
)
SELECT p.ValidTimeUtc, {selectSw}, {selectDr}, {selectDf}, p.cloud_mean, p.rh_mean, e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var sw = new double[n];
        var dr = new double[n];
        var df = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++) sw[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++) dr[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            for (int i = 0; i < n; i++) df[i] = r.IsDBNull(1 + 2 * n + i) ? double.NaN : r.GetDouble(1 + 2 * n + i);
            var cloudMean = r.IsDBNull(1 + 3 * n)     ? double.NaN : r.GetDouble(1 + 3 * n);
            var rhMean    = r.IsDBNull(1 + 3 * n + 1) ? double.NaN : r.GetDouble(1 + 3 * n + 1);
            var truth = r.GetDouble(1 + 3 * n + 2);
            rows.Add(ComposeRow(spec, valid, sw, dr, df, truth, cloudMean, rhMean));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec, DateTime validTimeUtc,
        IReadOnlyList<double> sw, IReadOnlyList<double> direct, IReadOnlyList<double> diffuse,
        double era5Sw, double cloudMean = double.NaN, double rhMean = double.NaN)
    {
        var n = spec.Models.Count;
        if (sw.Count      != n) throw new ArgumentException($"Expected {n} SW values", nameof(sw));
        if (direct.Count  != n) throw new ArgumentException($"Expected {n} direct values", nameof(direct));
        if (diffuse.Count != n) throw new ArgumentException($"Expected {n} diffuse values", nameof(diffuse));

        // Spread over SW (NaN-safe).
        var spread = InterModelSpread.From(sw);

        var v = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var cal = CalendarFeatures.From(v);

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < n; i++) features[i] = (float)sw[i];
        for (int i = 0; i < n; i++) features[n + i] = (float)direct[i];
        for (int i = 0; i < n; i++) features[2 * n + i] = (float)diffuse[i];
        features[3 * n + 0] = (float)spread.Mean;
        features[3 * n + 1] = (float)spread.Std;
        features[3 * n + 2] = (float)spread.Range;
        features[3 * n + 3] = cal.HourSin;
        features[3 * n + 4] = cal.HourCos;
        features[3 * n + 5] = cal.DoySin;
        features[3 * n + 6] = cal.DoyCos;
        // RICH cross-variable means — written only when the (loaded) spec carries
        // them, so lean bundles (FeatureCount == 3N+7) keep predicting unchanged
        // and rich bundles (3N+9) get cloud_mean/rh_mean. Schema-gated, mirrors
        // the upper-air predict pattern.
        if (spec.FeatureCount > 3 * n + 7)
        {
            features[3 * n + 7] = (float)cloudMean;
            features[3 * n + 8] = (float)rhMean;
        }

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Sw,
        };
    }
}
