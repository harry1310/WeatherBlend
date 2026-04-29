using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Builds one precipitation-blender dataset per lead.
///
/// Structure mirrors <see cref="TempFeatureBuilder"/>:
///   1. Aggregate EA 15-min rainfall readings up to hourly totals for the
///      primary rainfall station. Hours with fewer than 4 readings are
///      dropped — a partial-hour total would be a label bug.
///   2. Read forecasts filtered to RunTimeSource='offset_day' and LeadHours=lead.
///      Dedupe to the latest forecast per (ValidTime, Model), pivot to one row
///      per ValidTime with per-model precip / probability / covariate columns.
///   3. Inner-join to hourly truth on ValidTimeUtc.
///   4. Compose cyclical calendar features and ensemble-spread features in .NET.
///
/// Unlike the temperature pipeline we do NOT drop rows where a model is missing
/// precipitation — LightGBM handles NaN natively and some models only publish
/// probability at certain leads. We only require the ensemble mean to be
/// computable (at least one non-null precip forecast).
/// </summary>
public static class PrecipFeatureBuilder
{
    public const double WetThresholdMm = 0.1;

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");

    public const string SpecTarget = "precipitation";
    public const string SpecFeatureSet = "lean";

    /// <summary>
    /// Resolve the runtime <see cref="BlenderSpec"/> for lean precipitation at a
    /// given lead. Feature ordering: per-model precip values, then per-model
    /// probability values, then 4 ensemble-spread features
    /// (precip_mean / precip_std / precip_max / precip_agreement_wet_01),
    /// then 7 covariates (rh_mean / dew_depression_mean / cloud_low_mean /
    /// cloud_mid_mean / cloud_high_mean / cape_mean / wind_speed_mean), then
    /// 4 cyclical calendar features.
    /// </summary>
    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var blender = blendersCfg.Get(SpecTarget, SpecFeatureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var overlap = requiredSet.Intersect(optionalSet, StringComparer.OrdinalIgnoreCase).ToArray();
        if (overlap.Length > 0)
            throw new InvalidOperationException(
                $"BlenderConfig {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h has model(s) " +
                $"listed as both required and optional: [{string.Join(",", overlap)}].");

        var orderedRequired = TempFeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = TempFeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = TempFeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h.");

        // prob_* features removed 2026-04-28 — every prob_<model> had 0.000 gain at every
        // lead in lean + rich precip (Open-Meteo's precipitation_probability adds nothing
        // the trees can't infer from the precipitation rate). Old artefacts with prob_*
        // in their feature_schema can't predict under this code — retrain required.
        var n = orderedModels.Count;
        var featureNames = new List<string>(n + 4 + 7 + 4);
        foreach (var m in orderedModels) featureNames.Add($"precip_{TempFeatureBuilder.ShortName(m)}");
        featureNames.AddRange(new[] { "precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01" });
        featureNames.AddRange(new[]
        {
            "rh_mean", "dew_depression_mean",
            "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
            "cape_mean", "wind_speed_mean",
        });
        featureNames.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });

        return new BlenderSpec
        {
            Target = SpecTarget,
            FeatureSet = SpecFeatureSet,
            LeadHours = leadHours,
            RequiredModels = orderedRequired,
            OptionalModels = orderedOptional,
            Models = orderedModels,
            FeatureNames = featureNames,
        };
    }

    /// <summary>
    /// Builds the binary-classification training rows for one (station, lead).
    /// SQL pivot includes only spec.Models — excluded models never appear in the
    /// row vector at all. Post-pivot WHERE: every required NOT NULL AND at least
    /// one of all NOT NULL.
    /// </summary>
    public static List<BinaryTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        BlenderSpec spec,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var rnGlob = NormaliseGlob(Path.Combine(rainfallPath, "**", "*.parquet"));
        var escStation = stationName.Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";

        var precipPivot = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN Precipitation END) AS precip_{TempFeatureBuilder.ShortName(m)}"));
        var precipSelect = string.Join(", ", spec.Models.Select(m => $"p.precip_{TempFeatureBuilder.ShortName(m)}"));

        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.precip_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.precip_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH hourly_truth AS (
    SELECT
        date_trunc('hour', ObservedTimeUtc) AS valid_time,
        SUM(Value15MinMm) AS precip_mm_hour
    FROM read_parquet('{rnGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{escLocation}'
      AND StationName  = '{escStation}'
      AND Value15MinMm IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) = 4
),
latest AS (
    SELECT
        ValidTimeUtc, Model,
        Precipitation,
        RelativeHumidity2m, Temperature2m, DewPoint2m,
        CloudCoverLow, CloudCoverMid, CloudCoverHigh,
        Cape, WindSpeed10m,
        ROW_NUMBER() OVER (
            PARTITION BY ValidTimeUtc, Model
            ORDER BY RunTimeUtc DESC
        ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{escLocation}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        {precipPivot},
        AVG(RelativeHumidity2m) AS rh_mean,
        AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
        AVG(CloudCoverLow)  AS cloud_low_mean,
        AVG(CloudCoverMid)  AS cloud_mid_mean,
        AVG(CloudCoverHigh) AS cloud_high_mean,
        AVG(Cape)           AS cape_mean,
        AVG(WindSpeed10m)   AS wind_speed_mean
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
)
SELECT
    p.ValidTimeUtc,
    {precipSelect},
    p.rh_mean, p.dew_depression_mean,
    p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean,
    p.cape_mean, p.wind_speed_mean,
    t.precip_mm_hour
FROM pivoted p
JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
WHERE ({requiredNotNull})
  AND {anyNotNull}
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<BinaryTrainingRow>();
        var n = spec.Models.Count;
        var precip = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++)
                precip[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            var baseIdx = 1 + n;
            var rh     = r.IsDBNull(baseIdx + 0) ? double.NaN : r.GetDouble(baseIdx + 0);
            var dewDep = r.IsDBNull(baseIdx + 1) ? double.NaN : r.GetDouble(baseIdx + 1);
            var cL     = r.IsDBNull(baseIdx + 2) ? double.NaN : r.GetDouble(baseIdx + 2);
            var cM     = r.IsDBNull(baseIdx + 3) ? double.NaN : r.GetDouble(baseIdx + 3);
            var cH     = r.IsDBNull(baseIdx + 4) ? double.NaN : r.GetDouble(baseIdx + 4);
            var cape   = r.IsDBNull(baseIdx + 5) ? double.NaN : r.GetDouble(baseIdx + 5);
            var wind   = r.IsDBNull(baseIdx + 6) ? double.NaN : r.GetDouble(baseIdx + 6);
            var truth  = r.GetDouble(baseIdx + 7);

            rows.Add(ComposeRow(spec, valid, precip, rh, dewDep, cL, cM, cH, cape, wind, truth));
        }
        return rows;
    }

    /// <summary>
    /// Pack per-model precip + spread + covariates + calendar into a
    /// <see cref="BinaryTrainingRow"/> with <c>Features.Length == spec.FeatureCount</c>.
    /// Spread aggregates skip NaN entries (NaN-safe — important because
    /// precipitation defaults to the lenient COALESCE-any policy).
    /// </summary>
    public static BinaryTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> perModelPrecip,
        double rhMean,
        double dewDepressionMean,
        double cloudLowMean,
        double cloudMidMean,
        double cloudHighMean,
        double capeMean,
        double windSpeedMean,
        double truthMmHour)
    {
        var n = spec.Models.Count;
        if (perModelPrecip.Count != n)
            throw new ArgumentException(
                $"Expected {n} model precip values (one per spec.Models), got {perModelPrecip.Count}.",
                nameof(perModelPrecip));

        // Ensemble spread features — NaN-safe.
        double sum = 0, sumSq = 0, max = double.NegativeInfinity;
        int wetCount = 0, presentCount = 0;
        for (int i = 0; i < n; i++)
        {
            var x = perModelPrecip[i];
            if (double.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            if (x > max) max = x;
            if (x >= WetThresholdMm) wetCount++;
            presentCount++;
        }
        double mean = presentCount == 0 ? double.NaN : sum / presentCount;
        double std;
        if (presentCount <= 1) std = 0.0;
        else
        {
            var variance = Math.Max(0.0, (sumSq / presentCount) - (mean * mean));
            std = Math.Sqrt(variance);
        }
        if (presentCount == 0) max = double.NaN;
        var agreement = presentCount == 0 ? double.NaN : (double)wetCount / presentCount;

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle  = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

        var features = new float[spec.FeatureCount];
        int idx = 0;
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelPrecip[i];
        features[idx++] = (float)mean;
        features[idx++] = (float)std;
        features[idx++] = (float)max;
        features[idx++] = (float)agreement;
        features[idx++] = (float)rhMean;
        features[idx++] = (float)dewDepressionMean;
        features[idx++] = (float)cloudLowMean;
        features[idx++] = (float)cloudMidMean;
        features[idx++] = (float)cloudHighMean;
        features[idx++] = (float)capeMean;
        features[idx++] = (float)windSpeedMean;
        features[idx++] = (float)Math.Sin(hourAngle);
        features[idx++] = (float)Math.Cos(hourAngle);
        features[idx++] = (float)Math.Sin(doyAngle);
        features[idx++] = (float)Math.Cos(doyAngle);
        if (idx != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Feature pack mismatch: wrote {idx}, expected {spec.FeatureCount}");

        return new BinaryTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = truthMmHour >= WetThresholdMm,
            TruthMmHour = (float)truthMmHour,
        };
    }
}
