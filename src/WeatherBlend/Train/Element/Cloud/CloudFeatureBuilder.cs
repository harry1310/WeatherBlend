using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Cloud;

/// <summary>
/// Spec-driven SQL pivot + row composition for the cloud-cover blender.
/// Pulls per-model total <c>cloud_cover</c> only — layered cloud is unavailable
/// in Open-Meteo Previous Runs at Bonehill.
///
/// Model membership comes from <c>blenders.cloud</c> in config.yaml. UKMO is
/// in <c>requiredModels</c> at every lead (2026-04-26 bake-off restored it);
/// MF is required at lead 24h and excluded at 48/72h (live API horizon ~36h).
///
/// Layout per N active models: N per-model cc + 3 spread + 4 calendar = N + 7.
/// For N=6 → 13 features (matches legacy shape). For N=5 → 12.
/// </summary>
public static class CloudFeatureBuilder
{
    public const string SpecTarget = "cloud";
    public const string SpecFeatureSet = "default";

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build;
        // only the cloud layout below is builder-specific.
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
            var names = new List<string>();
            foreach (var m in orderedModels) names.Add($"cc_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[] { "cc_mean", "cc_std", "cc_range" });
            names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });
            // Ensemble-mean CAPE at the 24h lead ONLY (productionised 2026-06-04 from the
            // cloud-cape-bakeoff: cape_mean −1.7% MAE @24h, mean-only; flat/negative at
            // 48/72h once the CAPE forecast decays). Lead-gated here; ComposeRow writes it
            // only when the (loaded) spec carries it, so 48/72h + older bundles stay lean.
            if (leadHours == 24) names.Add("cape_mean");
            return names;
        });

    public static List<RegressionTrainingRow> BuildForLead(
        string forecastsPath, string era5Path, string locationName, BlenderSpec spec,
        CancellationToken ct = default, string? floorOverride = null)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob  = SqlGlob.Escape(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = SqlGlob.Escape(Path.Combine(era5Path, "**", "*.parquet"));
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var n = spec.Models.Count;
        // No training-window floor by default: with UKMO demoted to optional (config,
        // 2026-06-09) + no floor, cloud gains ~Feb–Aug 2024 and beats the floored
        // UKMO-required production cloud by −1.6..−2.6% (bake-off 2026-06-09).
        // floorOverride re-imposes a floor if ever needed.
        var floorClause = floorOverride is null ? "" : $"AND ValidTimeUtc >= TIMESTAMP '{floorOverride}'";

        var pivotCc = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN CloudCover END) AS cc_{TempFeatureBuilder.ShortName(m)}"));
        var selectCc = string.Join(", ", spec.Models.Select(m => $"p.cc_{TempFeatureBuilder.ShortName(m)}"));
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.cc_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.cc_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, CloudCover, Cape,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND CloudCover IS NOT NULL
      {floorClause}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotCc},
        AVG(Cape) AS cape_mean
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, CloudCover AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND CloudCover IS NOT NULL
      {floorClause}
)
SELECT p.ValidTimeUtc, {selectCc}, p.cape_mean, e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var cc = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++) cc[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            var capeMean = r.IsDBNull(1 + n) ? double.NaN : r.GetDouble(1 + n);
            var truth = r.GetDouble(1 + n + 1);
            rows.Add(ComposeRow(spec, valid, cc, truth, capeMean));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec, DateTime validTimeUtc, IReadOnlyList<double> cc, double era5Cc,
        double capeMean = double.NaN)
    {
        var n = spec.Models.Count;
        if (cc.Count != n) throw new ArgumentException($"Expected {n} cloud values", nameof(cc));

        // Spread over cloud cover (NaN-safe).
        var spread = InterModelSpread.From(cc);

        var v = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var cal = CalendarFeatures.From(v);

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < n; i++) features[i] = (float)cc[i];
        features[n + 0] = (float)spread.Mean;
        features[n + 1] = (float)spread.Std;
        features[n + 2] = (float)spread.Range;
        features[n + 3] = cal.HourSin;
        features[n + 4] = cal.HourCos;
        features[n + 5] = cal.DoySin;
        features[n + 6] = cal.DoyCos;
        // CAPE mean — only when the (loaded) spec carries it (24h lead). Lean/48h/72h
        // bundles (FeatureCount == n+7) keep predicting unchanged. Schema-gated.
        if (spec.FeatureCount > n + 7)
            features[n + 7] = (float)capeMean;

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Cc,
        };
    }
}
