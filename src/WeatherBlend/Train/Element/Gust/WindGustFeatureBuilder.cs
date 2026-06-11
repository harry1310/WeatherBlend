using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Gust;

/// <summary>
/// SQL pivot + row composition for the wind-gust blender — spec-driven.
/// Pulls per-model <c>wind_gusts_10m</c> AND <c>wind_speed_10m</c> from the
/// Previous Runs forecast tree (both needed — wsp drives the per-NWP gust
/// ratio) and joins to ERA5 <c>wind_gusts_10m</c> as truth.
///
/// Model membership comes from <c>blenders.wind_gust</c> in config.yaml.
/// Only the 4 production NWPs with gust forecasts on Open-Meteo
/// (GFS / ICON / GEM / UKMO) participate; ECMWF / MF / JMA / AIFS publish
/// no gust on Open-Meteo Previous Runs (audited 2026-05-27 — see
/// project_gust_production_scope_2026-05-27.md). UKMO is optional and
/// restricted to the UKMO-clean window (post-2024-09).
///
/// Feature layout per N active models:
///   N per-model gust
///   N per-model gust ratio (gust / max(wsp, 0.5), clipped [0.5, 4.0])
///   2 ratio spread (mean, std — over the N ratios, NaN-safe)
/// Total = 2N + 2. For N=4 → 10 features (production-scope shape).
///
/// Bake-off on 4-NWP scope 2026-05-27: MAE 1.0418 vs NWP-mean 1.2764
/// (-18.4%) on 2024 Bonehill, 1,251 test rows. Adding 4 archive NWPs
/// (gfs_ncep + MO Global + MO UKV + IFS oper) gets to 0.9738 but is
/// parked as a broader cross-blender question.
/// </summary>
public static class WindGustFeatureBuilder
{
    public const string SpecTarget = "wind_gust";
    public const string SpecFeatureSet = "default";

    /// <summary>Lower / upper clip bounds for the per-NWP gust ratio.
    /// 0.5 floor prevents degenerate divide-by-zero artefacts in calm winds;
    /// 4.0 ceiling clamps NWP-error spikes that pop ratios into double-digit
    /// territory. Matched in <see cref="ComposeRow"/> and the bake-off.</summary>
    public const float RatioMin = 0.5f;
    public const float RatioMax = 4.0f;

    /// <summary>Wind-speed floor used in the ratio denominator. Below this,
    /// the model would see arbitrarily large ratios that are meaningless
    /// in low-wind regimes. 0.5 m/s matches the bake-off clip.</summary>
    public const double WindSpeedFloor = 0.5;

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build;
        // only the gust layout below is builder-specific.
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
            var names = new List<string>();
            foreach (var m in orderedModels) names.Add($"gust_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"gust_ratio_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[] { "gust_ratio_mean", "gust_ratio_std" });
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

        // No training-window floor by default: floor bake-off (2026-06-09) showed dropping
        // the old 2024-09 floor IMPROVES gust at every lead (up to −13% @48h). UKMO is
        // optional + NaN-tolerated. floorOverride re-imposes a floor if ever needed.
        var floorClause = floorOverride is null ? "" : $"AND ValidTimeUtc >= TIMESTAMP '{floorOverride}'";

        var pivotGust = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN WindGusts10m END) AS gust_{TempFeatureBuilder.ShortName(m)}"));
        var pivotWsp = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN WindSpeed10m END) AS wsp_{TempFeatureBuilder.ShortName(m)}"));
        var selectGust = string.Join(", ", spec.Models.Select(m => $"p.gust_{TempFeatureBuilder.ShortName(m)}"));
        var selectWsp = string.Join(", ", spec.Models.Select(m => $"p.wsp_{TempFeatureBuilder.ShortName(m)}"));

        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m =>
                $"p.gust_{TempFeatureBuilder.ShortName(m)} IS NOT NULL AND p.wsp_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m =>
            $"p.gust_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, WindGusts10m, WindSpeed10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND WindGusts10m IS NOT NULL
      AND WindSpeed10m IS NOT NULL
      {floorClause}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotGust},
        {pivotWsp}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, WindGusts10m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND WindGusts10m IS NOT NULL
      {floorClause}
)
SELECT p.ValidTimeUtc, {selectGust}, {selectWsp}, e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var gust = new double[n];
        var wsp = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++) gust[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++) wsp[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            var truth = r.GetDouble(1 + 2 * n);
            rows.Add(ComposeRow(spec, valid, gust, wsp, truth));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> gustsMs,
        IReadOnlyList<double> speedsMs,
        double era5GustMs)
    {
        var n = spec.Models.Count;
        if (gustsMs.Count != n) throw new ArgumentException($"Expected {n} gusts", nameof(gustsMs));
        if (speedsMs.Count != n) throw new ArgumentException($"Expected {n} speeds", nameof(speedsMs));

        // Per-NWP ratio: gust / max(wsp, 0.5), clipped to [0.5, 4.0]. NaN if either
        // input is NaN (LightGBM handles NaN natively via missing-value branches).
        var ratios = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(gustsMs[i]) || double.IsNaN(speedsMs[i]))
            {
                ratios[i] = float.NaN;
                continue;
            }
            var denom = Math.Max(speedsMs[i], WindSpeedFloor);
            var raw = (float)(gustsMs[i] / denom);
            ratios[i] = Math.Clamp(raw, RatioMin, RatioMax);
        }

        // NaN-safe spread over present ratios. Deliberately NOT
        // InterModelSpread: the accumulation here is over FLOAT ratios
        // (x*x is a float multiply, the variance uses the float mean) and
        // there's no min/max — funnelling through the double-based helper
        // would change the trained feature bits.
        double sum = 0, sumSq = 0;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = ratios[i];
            if (float.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            present++;
        }
        var ratioMean = present == 0 ? float.NaN : (float)(sum / present);
        var ratioStd = present == 0 ? float.NaN
            : (float)Math.Sqrt(Math.Max(0.0, (sumSq / present) - (ratioMean * ratioMean)));

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < n; i++) features[i] = (float)gustsMs[i];
        for (int i = 0; i < n; i++) features[n + i] = ratios[i];
        features[2 * n + 0] = ratioMean;
        features[2 * n + 1] = ratioStd;

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5GustMs,
        };
    }
}
