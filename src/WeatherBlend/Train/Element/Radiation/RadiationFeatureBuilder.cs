using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

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
    {
        var blender = blendersCfg.Get(SpecTarget, SpecFeatureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var orderedRequired = FeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = FeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = FeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h.");

        var names = new List<string>();
        foreach (var m in orderedModels) names.Add($"sw_{FeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"direct_{FeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"diffuse_{FeatureBuilder.ShortName(m)}");
        names.AddRange(new[] { "sw_mean", "sw_std", "sw_range" });
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

        var pivotSw = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN ShortwaveRadiation END) AS sw_{FeatureBuilder.ShortName(m)}"));
        var pivotDr = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN DirectRadiation END) AS dr_{FeatureBuilder.ShortName(m)}"));
        var pivotDf = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN DiffuseRadiation END) AS df_{FeatureBuilder.ShortName(m)}"));
        var selectSw = string.Join(", ", spec.Models.Select(m => $"p.sw_{FeatureBuilder.ShortName(m)}"));
        var selectDr = string.Join(", ", spec.Models.Select(m => $"p.dr_{FeatureBuilder.ShortName(m)}"));
        var selectDf = string.Join(", ", spec.Models.Select(m => $"p.df_{FeatureBuilder.ShortName(m)}"));
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.sw_{FeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.sw_{FeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, ShortwaveRadiation, DirectRadiation, DiffuseRadiation,
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
        {pivotDf}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, ShortwaveRadiation AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND ShortwaveRadiation IS NOT NULL
)
SELECT p.ValidTimeUtc, {selectSw}, {selectDr}, {selectDf}, e.truth
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
            var truth = r.GetDouble(1 + 3 * n);
            rows.Add(ComposeRow(spec, valid, sw, dr, df, truth));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec, DateTime validTimeUtc,
        IReadOnlyList<double> sw, IReadOnlyList<double> direct, IReadOnlyList<double> diffuse,
        double era5Sw)
    {
        var n = spec.Models.Count;
        if (sw.Count      != n) throw new ArgumentException($"Expected {n} SW values", nameof(sw));
        if (direct.Count  != n) throw new ArgumentException($"Expected {n} direct values", nameof(direct));
        if (diffuse.Count != n) throw new ArgumentException($"Expected {n} diffuse values", nameof(diffuse));

        // Spread over SW (NaN-safe).
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = sw[i];
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
        for (int i = 0; i < n; i++) features[i] = (float)sw[i];
        for (int i = 0; i < n; i++) features[n + i] = (float)direct[i];
        for (int i = 0; i < n; i++) features[2 * n + i] = (float)diffuse[i];
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
            Label = (float)era5Sw,
        };
    }

    private static string Norm(string p) => p.Replace('\\', '/').Replace("'", "''");
}
