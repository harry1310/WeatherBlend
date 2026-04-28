using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Builds the lean temperature blender's training dataset, one lead at a time.
///
/// **New canonical path (Phase 1+ unify-model-membership refactor):**
/// <see cref="BuildSpec"/> resolves a runtime <see cref="BlenderSpec"/> from
/// <c>config.yaml</c>'s <c>blenders</c> section. <see cref="BuildForLead(string,string,string,BlenderSpec,CancellationToken)"/>
/// then pivots only the spec's active models — no <c>CAST(NULL AS DOUBLE)</c>
/// stamps for excluded slots — and packs features into a
/// <see cref="RegressionTrainingRow.Features"/> vector in the order
/// <see cref="BlenderSpec.FeatureNames"/> declares.
///
/// **Legacy path (still used by RichFeatureBuilder, ScoreHistoricalCommand,
/// ElementBakeoffCommand until their phases land):** the older static
/// <see cref="FeatureNames"/> / <see cref="ModelColumns"/> / fixed-13-feature
/// <c>BuildForLead(int leadHours, ...)</c> overload returning <see cref="TrainingRow"/>
/// is preserved verbatim. Marked obsolete so the warnings track migration progress.
/// Removed once Phases 3 + 5 land.
/// </summary>
public static class FeatureBuilder
{
    public const string Target = "temperature";
    public const string FeatureSet = "lean";

    /// <summary>Canonical model ordering (matches the project's config.yaml models list).
    /// AIFS sits at the end so adding it doesn't shift existing per-model feature indexes.
    /// JMA / KNMI HARMONIE / DMI HARMONIE / raw Met Office UKV+Global partitions are in
    /// the models registry (collected daily) but deliberately NOT in this list — the
    /// bake-offs 2026-04-28 found their net effect on the blender ranges from zero
    /// (raw MO, HARMONIE) to mixed (JMA: precip-only win pending Phase 3 rollout).
    /// Adding to this list is only valid once a model is wired into a blender spec
    /// AND has a per-model output field on the relevant PredictionRow type.
    /// See memory/project_met_office_raw_negative_result.md +
    /// project_jma_harmonie_bakeoff_2026-04-28.md.</summary>
    public static readonly IReadOnlyList<string> CanonicalModelOrder = new[]
    {
        "gfs_seamless",
        "ecmwf_ifs025",
        "icon_seamless",
        "meteofrance_seamless",
        "ukmo_seamless",
        "gem_seamless",
        "ecmwf_aifs025_single",
        "jma_seamless",
    };

    /// <summary>Stable short suffix used in feature column names (temp_gfs, temp_ecmwf, ...).</summary>
    public static string ShortName(string modelId) => modelId switch
    {
        "gfs_seamless" => "gfs",
        "ecmwf_ifs025" => "ecmwf",
        "icon_seamless" => "icon",
        "meteofrance_seamless" => "mf",
        "ukmo_seamless" => "ukmo",
        "gem_seamless" => "gem",
        "ecmwf_aifs025_single" => "aifs",
        "jma_seamless" => "jma",
        _ => throw new ArgumentException($"Unknown modelId '{modelId}'", nameof(modelId)),
    };

    // -----------------------------------------------------------------------
    // New canonical API (post unify-model-membership refactor)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Resolve the runtime <see cref="BlenderSpec"/> for a given lead. The
    /// active required + optional model lists come from config; the feature
    /// schema (per-model temps in canonical order, then mean/std/range, then
    /// the four cyclical calendar features) is computed here.
    /// </summary>
    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var blender = blendersCfg.Get(Target, FeatureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var overlap = requiredSet.Intersect(optionalSet, StringComparer.OrdinalIgnoreCase).ToArray();
        if (overlap.Length > 0)
            throw new InvalidOperationException(
                $"BlenderConfig {Target}/{FeatureSet} at lead {leadHours}h has model(s) listed as both " +
                $"required and optional: [{string.Join(",", overlap)}].");

        var orderedRequired = CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = CanonicalModelOrder.Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {Target}/{FeatureSet} at lead {leadHours}h.");

        var featureNames = new List<string>(orderedModels.Count + 7);
        foreach (var m in orderedModels) featureNames.Add($"temp_{ShortName(m)}");
        featureNames.Add("temp_mean");
        featureNames.Add("temp_std");
        featureNames.Add("temp_range");
        featureNames.Add("hour_sin");
        featureNames.Add("hour_cos");
        featureNames.Add("doy_sin");
        featureNames.Add("doy_cos");

        return new BlenderSpec
        {
            Target = Target,
            FeatureSet = FeatureSet,
            LeadHours = leadHours,
            RequiredModels = orderedRequired,
            OptionalModels = orderedOptional,
            Models = orderedModels,
            FeatureNames = featureNames,
        };
    }

    /// <summary>
    /// Builds the training rows for one (target, featureSet, lead) blender.
    /// SQL pivot includes only spec.Models — excluded models never appear in
    /// the row vector at all.
    /// </summary>
    public static List<RegressionTrainingRow> BuildForLead(
        string forecastsPath,
        string era5Path,
        string locationName,
        BlenderSpec spec,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = NormaliseGlob(Path.Combine(era5Path, "**", "*.parquet"));
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";

        var pivotCols = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN Temperature2m END) AS temp_{ShortName(m)}"));
        var selectCols = string.Join(", ", spec.Models.Select(m => $"p.temp_{ShortName(m)}"));

        // Post-pivot WHERE = (every required NOT NULL) AND (at least one of all NOT NULL).
        // The second clause is a universal safety check: even an all-optional spec drops
        // rows where every model is silent (the row would have an all-NaN feature vector).
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.temp_{ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.temp_{ShortName(m)} IS NOT NULL")) + ")";
        var whereClause = $"({requiredNotNull})\n  AND {anyNotNull}";

        var fcWhere =
            $"LocationName = '{locationName}' AND RunTimeSource = 'offset_day' " +
            $"AND LeadHours = {spec.LeadHours} AND Temperature2m IS NOT NULL " +
            $"AND Model IN {modelInClause}";

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
        {pivotCols},
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
    {selectCols},
    p.wind_dir_mean,
    e.era5_temp
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
WHERE {whereClause}
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var modelCount = spec.Models.Count;
        var temps = new double[modelCount];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < modelCount; i++)
                temps[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            var windDirMean = r.IsDBNull(1 + modelCount) ? double.NaN : r.GetDouble(1 + modelCount);
            var era5Temp = r.GetDouble(2 + modelCount);
            rows.Add(ComposeRow(spec, valid, temps, windDirMean, era5Temp));
        }
        return rows;
    }

    /// <summary>
    /// Pack per-model temps + spread + calendar features into a
    /// <see cref="RegressionTrainingRow"/> with <c>Features.Length == spec.FeatureCount</c>.
    /// Spread features (mean/std/range) skip NaN entries — for strict-policy
    /// blenders all values are guaranteed present, but staying NaN-safe means
    /// the same code path serves the COALESCE-any (precip-style) blenders too.
    /// </summary>
    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> perModelTemps,
        double windDirMeanDeg,
        double era5Temp)
    {
        if (perModelTemps.Count != spec.Models.Count)
            throw new ArgumentException(
                $"Expected {spec.Models.Count} model temperatures (one per spec.Models), got {perModelTemps.Count}.",
                nameof(perModelTemps));

        // Population std (N, not N-1) — matches numpy default; fine for a feature.
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int n = 0;
        for (int i = 0; i < perModelTemps.Count; i++)
        {
            var x = perModelTemps[i];
            if (double.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            n++;
        }
        var mean = n == 0 ? double.NaN : sum / n;
        var variance = n == 0 ? double.NaN : Math.Max(0.0, (sumSq / n) - (mean * mean));
        var std = double.IsNaN(variance) ? double.NaN : Math.Sqrt(variance);
        var range = n == 0 ? double.NaN : max - min;

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);

        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

        var features = new float[spec.FeatureCount];
        for (int i = 0; i < perModelTemps.Count; i++)
            features[i] = (float)perModelTemps[i];
        var spreadStart = perModelTemps.Count;
        features[spreadStart + 0] = (float)mean;
        features[spreadStart + 1] = (float)std;
        features[spreadStart + 2] = (float)range;
        features[spreadStart + 3] = (float)Math.Sin(hourAngle);
        features[spreadStart + 4] = (float)Math.Cos(hourAngle);
        features[spreadStart + 5] = (float)Math.Sin(doyAngle);
        features[spreadStart + 6] = (float)Math.Cos(doyAngle);

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Temp,
            WindDirMean = (float)windDirMeanDeg,
        };
    }

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");

    // -----------------------------------------------------------------------
    // Legacy API — kept verbatim for non-yet-migrated callers (RichFeatureBuilder,
    // ScoreHistoricalCommand, ElementBakeoffCommand). Removed after Phases 3+5.
    // -----------------------------------------------------------------------

    /// <summary>Legacy fixed feature-name list for the 13-feature 2b blender.</summary>
    [Obsolete("Use BlenderSpec.FeatureNames from BuildSpec(...) instead.")]
    public static readonly IReadOnlyList<string> FeatureNames = new[]
    {
        "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem",
        "temp_mean", "temp_std", "temp_range",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    /// <summary>Legacy per-model column tuple list. Replaced by <see cref="CanonicalModelOrder"/> + <see cref="ShortName"/>.</summary>
    [Obsolete("Iterate CanonicalModelOrder + ShortName(modelId) instead.")]
    public static readonly IReadOnlyList<(string ModelId, string Col)> ModelColumns = new[]
    {
        ("gfs_seamless",         "temp_gfs"),
        ("ecmwf_ifs025",         "temp_ecmwf"),
        ("icon_seamless",        "temp_icon"),
        ("meteofrance_seamless", "temp_mf"),
        ("ukmo_seamless",        "temp_ukmo"),
        ("gem_seamless",         "temp_gem"),
    };

    /// <summary>Legacy fixed-shape build path (returns <see cref="TrainingRow"/>). Removed after Phase 3.</summary>
    [Obsolete("Use BuildForLead(BlenderSpec spec) — returns RegressionTrainingRow.")]
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
        var modelInClause = LegacyModelInClauseForLead(leadHours);
        var fcWhere = $"LocationName = '{locationName}' AND RunTimeSource = 'offset_day' AND LeadHours = {leadHours} AND Temperature2m IS NOT NULL AND Model IN {modelInClause}";

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
        CAST(NULL AS DOUBLE)                                                  AS temp_ukmo,
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
  AND {(leadHours >= 96 ? "TRUE" : "p.temp_mf    IS NOT NULL")}
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
            for (int i = 0; i < 6; i++) t[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            var windDirMean = r.IsDBNull(7) ? double.NaN : r.GetDouble(7);
            var era5Temp = r.GetDouble(8);
            rows.Add(ComposeRow(valid, t, windDirMean, era5Temp));
        }
        return rows;
    }

    /// <summary>Legacy fixed-13-feature row composition. Removed after Phase 3.</summary>
    [Obsolete("Use ComposeRow(BlenderSpec spec, ...) — returns RegressionTrainingRow.")]
    public static TrainingRow ComposeRow(
        DateTime validTimeUtc,
        double[] perModelTemps,
        double windDirMeanDeg,
        double era5Temp)
    {
        if (perModelTemps.Length != 6)
            throw new ArgumentException("Expected 6 model temperatures", nameof(perModelTemps));

        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int n = 0;
        for (int i = 0; i < 6; i++)
        {
            var x = perModelTemps[i];
            if (double.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            n++;
        }
        var mean = n == 0 ? double.NaN : sum / n;
        var variance = n == 0 ? double.NaN : Math.Max(0.0, (sumSq / n) - (mean * mean));
        var std = double.IsNaN(variance) ? double.NaN : Math.Sqrt(variance);
        var range = n == 0 ? double.NaN : max - min;

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

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

    /// <summary>Legacy: per-lead model membership for the temperature blender.</summary>
    [Obsolete("Use BlenderConfig.ModelsForLead via BlendersConfig.")]
    public static IReadOnlyList<string> ModelsRequiredForLead(int leadHours)
        => leadHours >= 96
            ? new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless" }
            : new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless" };

    private static string LegacyModelInClauseForLead(int leadHours)
#pragma warning disable CS0618
        => "(" + string.Join(",", ModelsRequiredForLead(leadHours).Select(m => $"'{m}'")) + ")";
#pragma warning restore CS0618
}
