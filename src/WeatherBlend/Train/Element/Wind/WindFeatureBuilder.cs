using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

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
        foreach (var m in orderedModels) names.Add($"wind_spd_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"wind_dir_sin_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"wind_dir_cos_{TempFeatureBuilder.ShortName(m)}");
        names.AddRange(new[] { "wind_spd_mean", "wind_spd_std", "wind_spd_range" });
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
        var n = spec.Models.Count;

        // UKMO data starts 2024-09; restrict the training window so rows from before that
        // are dropped (they'd have UKMO=NaN at every row). Even with UKMO optional, the
        // post-pivot at-least-one safety check would survive but the blender wouldn't see
        // UKMO contribution at all in the early window — better to be uniform.
        var minTs = $"TIMESTAMP '{TrainingWindow.UkmoCleanWindowStart}'";

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
      AND ValidTimeUtc >= {minTs}
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
      AND ValidTimeUtc >= {minTs}
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
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = speeds[i];
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

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle  = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

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
        features[3 * n + 0] = (float)mean;
        features[3 * n + 1] = (float)std;
        features[3 * n + 2] = (float)range;
        features[3 * n + 3] = (float)Math.Sin(hourAngle);
        features[3 * n + 4] = (float)Math.Cos(hourAngle);
        features[3 * n + 5] = (float)Math.Sin(doyAngle);
        features[3 * n + 6] = (float)Math.Cos(doyAngle);

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5WindSpeed,
        };
    }

    private static string NormaliseGlob(string path) => path.Replace('\\', '/').Replace("'", "''");
}
