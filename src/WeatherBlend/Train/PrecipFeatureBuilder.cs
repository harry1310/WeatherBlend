using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Builds one precipitation-blender dataset per lead.
///
/// Structure mirrors <see cref="FeatureBuilder"/>:
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
    /// <summary>
    /// Feature column order for the occurrence classifier.
    /// Persisted in feature_schema.json so inference uses the same order.
    /// </summary>
    public static readonly IReadOnlyList<string> OccurrenceFeatureNames = new[]
    {
        // Per-model quantitative precip (6).
        "precip_gfs", "precip_ecmwf", "precip_icon", "precip_mf", "precip_ukmo", "precip_gem",
        // Per-model declared precip probability (6).
        "prob_gfs", "prob_ecmwf", "prob_icon", "prob_mf", "prob_ukmo", "prob_gem",
        // Ensemble spread features (4).
        "precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01",
        // Meteo covariates (7).
        "rh_mean", "dew_depression_mean",
        "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
        "cape_mean", "wind_speed_mean",
        // Cyclical calendar (4).
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    public static readonly IReadOnlyList<(string ModelId, string PrecipCol, string ProbCol)> ModelColumns = new[]
    {
        ("gfs_seamless",         "precip_gfs",   "prob_gfs"),
        ("ecmwf_ifs025",         "precip_ecmwf", "prob_ecmwf"),
        ("icon_seamless",        "precip_icon",  "prob_icon"),
        ("meteofrance_seamless", "precip_mf",    "prob_mf"),
        ("ukmo_seamless",        "precip_ukmo",  "prob_ukmo"),
        ("gem_seamless",         "precip_gem",   "prob_gem"),
    };

    public const double WetThresholdMm = 0.1;

    public static List<PrecipTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        int leadHours,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var rnGlob = NormaliseGlob(Path.Combine(rainfallPath, "**", "*.parquet"));

        var escStation = stationName.Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");

        // RunTimeSource filter matches FeatureBuilder: exclude historical-forecast
        // rows whose RunTime was synthesised to midnight and would contaminate lead
        // 24/48/72 buckets.
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
        Precipitation, PrecipitationProbability,
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
      AND LeadHours = {leadHours}
      -- UKMO excluded entirely (Pattern 1 final, 2026-04-26 four-way bake-off).
      -- The 6-model + restricted-window variant looked good in an earlier same-window
      -- bake-off but lost decisively (1-9 pct Brier across all 9 station/lead cells)
      -- once tested apples-to-apples on the same fixed test rows: the ~30 pct
      -- training-data loss from the date filter outweighs UKMO's signal. Bagging is
      -- a separate +0.5-2 pct win on top of any data-window choice (kept; see
      -- PrecipOccurrenceTrainer.Hyperparameters).
      -- MF excluded at lead >= 96h: Open-Meteo Previous Runs caps MF at ~72h, so
      -- including it at the 5-day horizon yields zero rows. The 4-model variant
      -- trains on the data that does exist.
      AND Model IN {ModelInClauseForLead(leadHours)}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN Precipitation END) AS precip_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN Precipitation END) AS precip_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN Precipitation END) AS precip_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN Precipitation END) AS precip_mf,
        CAST(NULL AS DOUBLE)                                                  AS precip_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN Precipitation END) AS precip_gem,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN PrecipitationProbability END) AS prob_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN PrecipitationProbability END) AS prob_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN PrecipitationProbability END) AS prob_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN PrecipitationProbability END) AS prob_mf,
        CAST(NULL AS DOUBLE)                                                              AS prob_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN PrecipitationProbability END) AS prob_gem,
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
    p.precip_gfs, p.precip_ecmwf, p.precip_icon, p.precip_mf, p.precip_ukmo, p.precip_gem,
    p.prob_gfs,   p.prob_ecmwf,   p.prob_icon,   p.prob_mf,   p.prob_ukmo,   p.prob_gem,
    p.rh_mean, p.dew_depression_mean,
    p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean,
    p.cape_mean, p.wind_speed_mean,
    t.precip_mm_hour
FROM pivoted p
JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
WHERE COALESCE(p.precip_gfs, p.precip_ecmwf, p.precip_icon, p.precip_mf, p.precip_ukmo, p.precip_gem) IS NOT NULL
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<PrecipTrainingRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();

            var valid = r.GetDateTime(0);

            var precip = new double[6];
            for (int i = 0; i < 6; i++)
                precip[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);

            var probs = new double[6];
            for (int i = 0; i < 6; i++)
                probs[i] = r.IsDBNull(7 + i) ? double.NaN : r.GetDouble(7 + i);

            var rh     = r.IsDBNull(13) ? double.NaN : r.GetDouble(13);
            var dewDep = r.IsDBNull(14) ? double.NaN : r.GetDouble(14);
            var cL     = r.IsDBNull(15) ? double.NaN : r.GetDouble(15);
            var cM     = r.IsDBNull(16) ? double.NaN : r.GetDouble(16);
            var cH     = r.IsDBNull(17) ? double.NaN : r.GetDouble(17);
            var cape   = r.IsDBNull(18) ? double.NaN : r.GetDouble(18);
            var wind   = r.IsDBNull(19) ? double.NaN : r.GetDouble(19);
            var truth  = r.GetDouble(20);

            rows.Add(ComposeRow(valid, precip, probs, rh, dewDep, cL, cM, cH, cape, wind, truth));
        }

        return rows;
    }

    public static PrecipTrainingRow ComposeRow(
        DateTime validTimeUtc,
        double[] perModelPrecip,
        double[] perModelProb,
        double rhMean,
        double dewDepressionMean,
        double cloudLowMean,
        double cloudMidMean,
        double cloudHighMean,
        double capeMean,
        double windSpeedMean,
        double truthMmHour)
    {
        if (perModelPrecip.Length != 6)
            throw new ArgumentException("Expected 6 model precip values", nameof(perModelPrecip));
        if (perModelProb.Length != 6)
            throw new ArgumentException("Expected 6 model probability values", nameof(perModelProb));

        // Ensemble spread features — NaN-safe. LightGBM understands NaN, so we
        // keep the per-model inputs as-is but make sure the aggregated summary
        // reflects only the non-null members (otherwise mean is biased when a
        // model drops out).
        double sum = 0, sumSq = 0, max = double.NegativeInfinity;
        int wetCount = 0, presentCount = 0;
        for (int i = 0; i < 6; i++)
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
        if (presentCount <= 1)
        {
            std = 0.0;
        }
        else
        {
            var variance = Math.Max(0.0, (sumSq / presentCount) - (mean * mean));
            std = Math.Sqrt(variance);
        }
        if (presentCount == 0) max = double.NaN;

        var agreement = presentCount == 0
            ? double.NaN
            : (double)wetCount / presentCount;

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);

        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle  = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

        return new PrecipTrainingRow
        {
            ValidTimeUtc = v,
            PrecipGfs   = (float)perModelPrecip[0],
            PrecipEcmwf = (float)perModelPrecip[1],
            PrecipIcon  = (float)perModelPrecip[2],
            PrecipMf    = (float)perModelPrecip[3],
            PrecipUkmo  = (float)perModelPrecip[4],
            PrecipGem   = (float)perModelPrecip[5],
            ProbGfs   = (float)perModelProb[0],
            ProbEcmwf = (float)perModelProb[1],
            ProbIcon  = (float)perModelProb[2],
            ProbMf    = (float)perModelProb[3],
            ProbUkmo  = (float)perModelProb[4],
            ProbGem   = (float)perModelProb[5],
            PrecipMean = (float)mean,
            PrecipStd  = (float)std,
            PrecipMax  = (float)max,
            PrecipAgreementWet01 = (float)agreement,
            RhMean = (float)rhMean,
            DewDepressionMean = (float)dewDepressionMean,
            CloudLowMean = (float)cloudLowMean,
            CloudMidMean = (float)cloudMidMean,
            CloudHighMean = (float)cloudHighMean,
            CapeMean = (float)capeMean,
            WindSpeedMean = (float)windSpeedMean,
            HourSin = (float)Math.Sin(hourAngle),
            HourCos = (float)Math.Cos(hourAngle),
            DoySin  = (float)Math.Sin(doyAngle),
            DoyCos  = (float)Math.Cos(doyAngle),
            WetBinary = truthMmHour >= WetThresholdMm,
            PrecipMmHour = (float)truthMmHour,
        };
    }

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");

    /// <summary>
    /// Per-lead model membership for the precipitation blender. Mirrors
    /// <see cref="FeatureBuilder.ModelsRequiredForLead"/>: lead &lt; 96h drops only
    /// UKMO; lead &gt;= 96h drops UKMO + MF (Open-Meteo Previous Runs caps MF at ~72h).
    /// </summary>
    public static IReadOnlyList<string> ModelsRequiredForLead(int leadHours)
        => leadHours >= 96
            ? new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless" }
            : new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless" };

    private static string ModelInClauseForLead(int leadHours)
        => "(" + string.Join(",", ModelsRequiredForLead(leadHours).Select(m => $"'{m}'")) + ")";

    // -----------------------------------------------------------------------
    // New canonical API (Phase 2 of unify-model-membership refactor).
    // Spec-driven, dynamic-shape vector via BinaryTrainingRow.Features.
    // -----------------------------------------------------------------------

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

        var orderedRequired = FeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = FeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = FeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h.");

        var n = orderedModels.Count;
        var featureNames = new List<string>(2 * n + 4 + 7 + 4);
        foreach (var m in orderedModels) featureNames.Add($"precip_{FeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) featureNames.Add($"prob_{FeatureBuilder.ShortName(m)}");
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
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN Precipitation END) AS precip_{FeatureBuilder.ShortName(m)}"));
        var probPivot = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN PrecipitationProbability END) AS prob_{FeatureBuilder.ShortName(m)}"));
        var precipSelect = string.Join(", ", spec.Models.Select(m => $"p.precip_{FeatureBuilder.ShortName(m)}"));
        var probSelect = string.Join(", ", spec.Models.Select(m => $"p.prob_{FeatureBuilder.ShortName(m)}"));

        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.precip_{FeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.precip_{FeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

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
        Precipitation, PrecipitationProbability,
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
        {probPivot},
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
    {probSelect},
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
        var probs = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < n; i++)
                precip[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++)
                probs[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            var baseIdx = 1 + 2 * n;
            var rh     = r.IsDBNull(baseIdx + 0) ? double.NaN : r.GetDouble(baseIdx + 0);
            var dewDep = r.IsDBNull(baseIdx + 1) ? double.NaN : r.GetDouble(baseIdx + 1);
            var cL     = r.IsDBNull(baseIdx + 2) ? double.NaN : r.GetDouble(baseIdx + 2);
            var cM     = r.IsDBNull(baseIdx + 3) ? double.NaN : r.GetDouble(baseIdx + 3);
            var cH     = r.IsDBNull(baseIdx + 4) ? double.NaN : r.GetDouble(baseIdx + 4);
            var cape   = r.IsDBNull(baseIdx + 5) ? double.NaN : r.GetDouble(baseIdx + 5);
            var wind   = r.IsDBNull(baseIdx + 6) ? double.NaN : r.GetDouble(baseIdx + 6);
            var truth  = r.GetDouble(baseIdx + 7);

            rows.Add(ComposeRow(spec, valid, precip, probs, rh, dewDep, cL, cM, cH, cape, wind, truth));
        }
        return rows;
    }

    /// <summary>
    /// Pack per-model precip + prob + spread + covariates + calendar into a
    /// <see cref="BinaryTrainingRow"/> with <c>Features.Length == spec.FeatureCount</c>.
    /// Spread aggregates skip NaN entries (NaN-safe — important because
    /// precipitation defaults to the lenient COALESCE-any policy).
    /// </summary>
    public static BinaryTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> perModelPrecip,
        IReadOnlyList<double> perModelProb,
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
        if (perModelProb.Count != n)
            throw new ArgumentException(
                $"Expected {n} model probability values, got {perModelProb.Count}.",
                nameof(perModelProb));

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
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelProb[i];
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
