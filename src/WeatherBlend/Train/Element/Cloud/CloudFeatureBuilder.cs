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
        foreach (var m in orderedModels) names.Add($"cc_{TempFeatureBuilder.ShortName(m)}");
        names.AddRange(new[] { "cc_mean", "cc_std", "cc_range" });
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
        var minTs = $"TIMESTAMP '{TrainingWindow.UkmoCleanWindowStart}'";

        var pivotCc = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN CloudCover END) AS cc_{TempFeatureBuilder.ShortName(m)}"));
        var selectCc = string.Join(", ", spec.Models.Select(m => $"p.cc_{TempFeatureBuilder.ShortName(m)}"));
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.cc_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.cc_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, CloudCover,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND CloudCover IS NOT NULL
      AND ValidTimeUtc >= {minTs}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotCc}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, CloudCover AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND CloudCover IS NOT NULL
      AND ValidTimeUtc >= {minTs}
)
SELECT p.ValidTimeUtc, {selectCc}, e.truth
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
            var truth = r.GetDouble(1 + n);
            rows.Add(ComposeRow(spec, valid, cc, truth));
        }
        return rows;
    }

    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec, DateTime validTimeUtc, IReadOnlyList<double> cc, double era5Cc)
    {
        var n = spec.Models.Count;
        if (cc.Count != n) throw new ArgumentException($"Expected {n} cloud values", nameof(cc));

        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = cc[i];
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
        for (int i = 0; i < n; i++) features[i] = (float)cc[i];
        features[n + 0] = (float)mean;
        features[n + 1] = (float)std;
        features[n + 2] = (float)range;
        features[n + 3] = (float)Math.Sin(hourAngle);
        features[n + 4] = (float)Math.Cos(hourAngle);
        features[n + 5] = (float)Math.Sin(doyAngle);
        features[n + 6] = (float)Math.Cos(doyAngle);

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Cc,
        };
    }

    private static string Norm(string p) => p.Replace('\\', '/').Replace("'", "''");
}
