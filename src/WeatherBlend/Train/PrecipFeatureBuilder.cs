using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Common;

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
    /// Exact-runtime models carrying backfilled multi-level pressure (model id
    /// → short feature suffix). met_office_global is excluded (no pressure
    /// backfill). Used by the experimental --upper-air path only. Internal so
    /// the rich (3c) UA A/B in <see cref="PrecipRichFeatureBuilder"/> reuses the
    /// exact same model set + column definitions — one source of truth.
    internal static readonly (string Model, string Short)[] UpperAirModels =
    {
        ("gfs_ncep", "gfs"),
        ("ecmwf_ifs_oper", "ifs"),
        ("ecmwf_aifs_oper", "aifs"),
        ("gefs_ncep_mean", "gefs"),
    };

    /// Full backfilled pressure set (ForecastRow column → feature short). The
    /// curated 3 (t850/gh850/gh500) undersold precip — RH850 (mid-level
    /// moisture) + t700/t500 instability + winds are the precip-relevant
    /// fields. Wind direction raw (deg) for this pass. Mirrors
    /// PrecipExactFeatureBuilder.UaPressureCols. Internal — shared with 3c (see
    /// <see cref="UpperAirModels"/>).
    internal static readonly (string Col, string Short)[] UaPressureCols =
    {
        ("Temperature850hPa", "t850"), ("Temperature700hPa", "t700"), ("Temperature500hPa", "t500"),
        ("GeopotentialHeight850hPa", "gh850"), ("GeopotentialHeight500hPa", "gh500"),
        ("WindSpeed850hPa", "ws850"), ("WindSpeed500hPa", "ws500"),
        ("WindDirection850hPa", "wd850"), ("WindDirection500hPa", "wd500"),
        ("RelativeHumidity850hPa", "rh850"),
    };

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours, bool withUpperAir = false)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build.
        // The UA variant stamps FeatureSet "lean-ua" (Tier stays "lean").
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
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

            // Experimental upper-air block (2026-06-01; FULL set) — APPENDED last so
            // baseline feature indices are untouched (ComposeRow's exact-count guard
            // catches any miscount). Per exact model: the FULL UaPressureCols
            // (RH850 + t700/t500 + winds, not just the temp-oriented curated 3),
            // lead-matched and joined to offset_day rows by a leak-free backward
            // ASOF (see BuildForLead) + ensemble t850_mean + rh850_mean. Full set
            // because the curated 3 undersold precip (the 3d-full re-run jumped to
            // ~−18% Brier once RH850/instability were added).
            if (withUpperAir)
            {
                foreach (var (_, s) in UpperAirModels)
                    foreach (var (_, sh) in UaPressureCols)
                        featureNames.Add($"{sh}_{s}");
                featureNames.Add("t850_mean");
                featureNames.Add("rh850_mean");
            }
            return featureNames;
        }, featureSetLabel: withUpperAir ? SpecFeatureSet + "-ua" : null);

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
        DateTime? minValidTime = null,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = SqlGlob.Escape(Path.Combine(forecastsPath, "**", "*.parquet"));
        var rnGlob = SqlGlob.Escape(Path.Combine(rainfallPath, "**", "*.parquet"));
        var escStation = stationName.Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";

        // Optional per-phase training-data cutoff (2026-05-26 — see
        // PhaseRegistry.ParseMinValidTime + project_3a_3b_data_drift memory).
        // Applied to BOTH SQL CTEs that select on ValidTimeUtc so truth and
        // forecast rows are clipped on the same boundary — otherwise the
        // inner-join between hourly_truth and pivoted would silently drop
        // any row past the cutoff. ISO-format here keeps the DuckDB literal
        // unambiguous regardless of DuckDB's default date locale.
        var minValidTimeFilter = minValidTime.HasValue
            ? $"      AND ValidTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'\n"
            : string.Empty;
        var minObservedTimeFilter = minValidTime.HasValue
            ? $"      AND ObservedTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'\n"
            : string.Empty;

        var precipPivot = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN Precipitation END) AS precip_{TempFeatureBuilder.ShortName(m)}"));
        var precipSelect = string.Join(", ", spec.Models.Select(m => $"p.precip_{TempFeatureBuilder.ShortName(m)}"));

        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.precip_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.precip_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        // Experimental upper-air join (derived from the spec, so Build stays in
        // sync with BuildSpec). For each offset_day valid-time V, attach the
        // freshest exact lead-{LeadHours} pressure forecast with valid_time <= V
        // (backward ASOF). Because exact lead-L valid_time = cycle + L, that row
        // was issued <= V - L, i.e. >= L hours before V — genuinely in hand when
        // the L-hour-lead prediction for V was made. The match is up to ~6h
        // below V (exact cycles are 6-hourly): "stale but real", forward-filled,
        // LEFT so pre-pressure rows just get NULL UA (LightGBM handles gaps).
        var withUpperAir = spec.FeatureNames.Contains("t850_mean");
        string uaCte = "", uaSelectSql = "", uaJoin = "";
        if (withUpperAir)
        {
            var uaModelIn = "(" + string.Join(",", UpperAirModels.Select(x => $"'{x.Model}'")) + ")";
            var uaPivots = string.Join(",\n           ", UpperAirModels.SelectMany(x =>
                UaPressureCols.Select(c => $"MAX(CASE WHEN Model = '{x.Model}' THEN {c.Col} END) AS {c.Short}_{x.Short}")));
            var uaInnerCols = string.Join(", ", UaPressureCols.Select(c => c.Col));
            uaCte = $@",
exact_ua AS (
    SELECT valid_time_ua,
           {uaPivots}
    FROM (
        SELECT ValidTimeUtc AS valid_time_ua, Model,
               {uaInnerCols},
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{escLocation}'
          AND RunTimeSource = 'exact'
          AND LeadHours = {spec.LeadHours}
          AND Model IN {uaModelIn}
    )
    WHERE rn = 1
    GROUP BY valid_time_ua
)";
            uaSelectSql = ",\n    " + string.Join(", ", UpperAirModels.SelectMany(x =>
                UaPressureCols.Select(c => $"x.{c.Short}_{x.Short}")));
            uaJoin = "ASOF LEFT JOIN exact_ua x ON p.ValidTimeUtc >= x.valid_time_ua";
        }

        var sql = $@"
WITH hourly_truth AS (
    SELECT
        date_trunc('hour', ObservedTimeUtc) AS valid_time,
        SUM(Value15MinMm) AS precip_mm_hour
    FROM read_parquet('{rnGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{escLocation}'
      AND StationName  = '{escStation}'
      AND Value15MinMm IS NOT NULL
{minObservedTimeFilter}    GROUP BY 1
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
{minValidTimeFilter}),
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
){uaCte}
SELECT
    p.ValidTimeUtc,
    {precipSelect},
    p.rh_mean, p.dew_depression_mean,
    p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean,
    p.cape_mean, p.wind_speed_mean,
    t.precip_mm_hour{uaSelectSql}
FROM pivoted p
JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
{uaJoin}
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

            // Trailing upper-air columns (after truth), UpperAirModels ×
            // UaPressureCols order, then ensemble t850_mean + rh850_mean.
            double[]? uaValues = null;
            if (withUpperAir)
            {
                int uaStart = baseIdx + 8;
                int mc = UpperAirModels.Length;
                int pc = UaPressureCols.Length;
                int t850Off = Array.FindIndex(UaPressureCols, c => c.Short == "t850");
                int rh850Off = Array.FindIndex(UaPressureCols, c => c.Short == "rh850");
                uaValues = new double[pc * mc + 2];
                double t850Sum = 0, rh850Sum = 0; int t850N = 0, rh850N = 0;
                for (int k = 0; k < mc; k++)
                    for (int j = 0; j < pc; j++)
                    {
                        var v = r.IsDBNull(uaStart + pc * k + j) ? double.NaN : r.GetDouble(uaStart + pc * k + j);
                        uaValues[pc * k + j] = v;
                        if (j == t850Off && !double.IsNaN(v)) { t850Sum += v; t850N++; }
                        if (j == rh850Off && !double.IsNaN(v)) { rh850Sum += v; rh850N++; }
                    }
                uaValues[pc * mc]     = t850N == 0 ? double.NaN : t850Sum / t850N;
                uaValues[pc * mc + 1] = rh850N == 0 ? double.NaN : rh850Sum / rh850N;
            }

            rows.Add(ComposeRow(spec, valid, precip, rh, dewDep, cL, cM, cH, cape, wind, truth, uaValues));
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
        double truthMmHour,
        IReadOnlyList<double>? upperAir = null)
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
        var cal = CalendarFeatures.From(v);

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
        features[idx++] = cal.HourSin;
        features[idx++] = cal.HourCos;
        features[idx++] = cal.DoySin;
        features[idx++] = cal.DoyCos;
        if (upperAir is not null)
            for (int i = 0; i < upperAir.Count; i++)
                features[idx++] = (float)upperAir[i];
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

    // -----------------------------------------------------------------------
    // Predict-time upper-air (live-tree analogue of the exact_ua CTE in
    // BuildForLead). Lets PrecipPredictCommand compose the same UA block the
    // trainer did, via the same leak-free backward ASOF (freshest exact
    // lead-L pressure ≤ target valid). Shared by the 3c and 3o predict paths.
    // -----------------------------------------------------------------------

    /// <summary>
    /// Pulls the freshest <c>RunTimeSource='exact'</c> lead-L pressure row per
    /// valid_time, pivoted model-major (<see cref="UpperAirModels"/> ×
    /// <see cref="UaPressureCols"/>), within [earliestValid, latestValid].
    /// Returns a list sorted ascending by valid_time; pair with
    /// <see cref="UpperAirValuesFor"/> for the backward ASOF at row-assembly
    /// time. Empty when the live tree carries no exact pressure yet (e.g.
    /// before the collector change has accumulated cycles).
    /// </summary>
    public static List<(DateTime ValidTimeUa, double[] PerModelCol)> LoadUpperAirLive(
        string forecastsPath, string locationName, int leadHours,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var escLoc = locationName.Replace("'", "''");
        var modelIn = "(" + string.Join(",", UpperAirModels.Select(x => $"'{x.Model}'")) + ")";
        var innerCols = string.Join(", ", UaPressureCols.Select(c => c.Col));
        var pivots = string.Join(",\n           ", UpperAirModels.SelectMany(x =>
            UaPressureCols.Select(c => $"MAX(CASE WHEN Model = '{x.Model}' THEN {c.Col} END) AS {c.Short}_{x.Short}")));
        var sql = $@"
WITH ex AS (
    SELECT ValidTimeUtc AS valid_time_ua, Model, {innerCols},
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{escLoc}'
      AND RunTimeSource = 'exact'
      AND LeadHours = {leadHours}
      AND Model IN {modelIn}
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                            AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
)
SELECT valid_time_ua,
       {pivots}
FROM ex WHERE rn = 1 GROUP BY valid_time_ua ORDER BY valid_time_ua;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        int width = UpperAirModels.Length * UaPressureCols.Length;
        var result = new List<(DateTime, double[])>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var t = r.GetDateTime(0);
            var vals = new double[width];
            for (int i = 0; i < width; i++)
                vals[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            result.Add((t, vals));
        }
        return result;
    }

    /// <summary>
    /// Backward-ASOF lookup over <see cref="LoadUpperAirLive"/>'s output: the UA
    /// feature vector for the freshest entry with valid_time ≤ <paramref
    /// name="validTimeUtc"/> — model-major per-model×col, then ensemble
    /// t850_mean + rh850_mean (same length + order as the trainer's uaValues).
    /// All-NaN when nothing qualifies (ComposeRow then emits NaN UA slots,
    /// which LightGBM treats as missing).
    /// </summary>
    public static double[] UpperAirValuesFor(
        IReadOnlyList<(DateTime ValidTimeUa, double[] PerModelCol)> asof, DateTime validTimeUtc)
    {
        int mc = UpperAirModels.Length, pc = UaPressureCols.Length;
        var outv = new double[pc * mc + 2];
        int found = -1;
        for (int i = 0; i < asof.Count; i++)
        {
            if (asof[i].ValidTimeUa <= validTimeUtc) found = i; else break;
        }
        if (found < 0) { Array.Fill(outv, double.NaN); return outv; }

        var src = asof[found].PerModelCol;
        Array.Copy(src, outv, pc * mc);
        int t850Off = Array.FindIndex(UaPressureCols, c => c.Short == "t850");
        int rh850Off = Array.FindIndex(UaPressureCols, c => c.Short == "rh850");
        double t850Sum = 0, rh850Sum = 0; int t850N = 0, rh850N = 0;
        for (int k = 0; k < mc; k++)
        {
            var tv = src[pc * k + t850Off]; if (!double.IsNaN(tv)) { t850Sum += tv; t850N++; }
            var rv = src[pc * k + rh850Off]; if (!double.IsNaN(rv)) { rh850Sum += rv; rh850N++; }
        }
        outv[pc * mc] = t850N == 0 ? double.NaN : t850Sum / t850N;
        outv[pc * mc + 1] = rh850N == 0 ? double.NaN : rh850Sum / rh850N;
        return outv;
    }
}
