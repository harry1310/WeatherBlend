using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Builds one dry-window training dataset per (station, lead, window-length).
///
/// Pipeline:
///   1. Load hourly rainfall truth for the station with 4-of-4 quality gate.
///   2. Build (date → has-dry-window-of-length-N) labels via
///      <see cref="DryWindowLabelBuilder"/> — dropping days with any missing
///      truth hour.
///   3. Load 24-hour forecast windows: RunTimeSource='offset_day',
///      LeadHours ∈ {L..L+23}, dedup to the latest forecast per
///      (ValidTime, Model). Valid-time buckets into the UTC target-day
///      D = midnight(valid) - L hours, modulo the hour-of-day being covered.
///      Concretely: hour h of day D is covered by LeadHours = L + h, so
///      forecasts with LeadHours ∈ {L..L+23} run from midnight Monday (lead
///      L + 0) through 23:00 Monday (lead L + 23) if the anchor sits L hours
///      earlier.
///   4. For each labeled date, compose per-model features (nan if the model
///      is missing any hour), ensemble summaries, and covariates from the
///      ensemble-mean time series.
/// </summary>
public static class DryWindowFeatureBuilder
{
    /// <summary>
    /// Fixed feature-column order for the Phase 3b base set (53 columns).
    /// Persisted in <c>feature_schema.json</c> for any 3b artefact and inherited
    /// unchanged by 3d-calibrated (which just re-maps 3b's output through PAV).
    /// Phase 3d-shape extends this list with <see cref="ShapeFeatureNames"/> —
    /// callers must use <see cref="FeatureNamesForPhase"/> to get the right set.
    /// </summary>
    public static readonly IReadOnlyList<string> FeatureNames = new[]
    {
        // Per-model day totals.
        "precip_sum_gfs", "precip_sum_ecmwf", "precip_sum_icon", "precip_sum_mf", "precip_sum_ukmo", "precip_sum_gem",
        // Per-model max hour.
        "precip_max_hour_gfs", "precip_max_hour_ecmwf", "precip_max_hour_icon", "precip_max_hour_mf", "precip_max_hour_ukmo", "precip_max_hour_gem",
        // Per-model wet-hour count.
        "wet_hour_count_gfs", "wet_hour_count_ecmwf", "wet_hour_count_icon", "wet_hour_count_mf", "wet_hour_count_ukmo", "wet_hour_count_gem",
        // Per-model longest dry-run hours.
        "longest_dry_run_gfs", "longest_dry_run_ecmwf", "longest_dry_run_icon", "longest_dry_run_mf", "longest_dry_run_ukmo", "longest_dry_run_gem",
        // Per-model self-prediction: dry window of length N exists.
        "has_dry_window_gfs", "has_dry_window_ecmwf", "has_dry_window_icon", "has_dry_window_mf", "has_dry_window_ukmo", "has_dry_window_gem",
        // Per-model max probability.
        "prob_max_gfs", "prob_max_ecmwf", "prob_max_icon", "prob_max_mf", "prob_max_ukmo", "prob_max_gem",
        // Ensemble summaries.
        "precip_sum_mean", "precip_sum_std", "precip_sum_max",
        "agreement_has_dry_window", "longest_dry_run_mean", "wet_hour_count_mean",
        // Covariates.
        "rh_mean", "rh_min", "dew_depression_max",
        "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
        "cape_max", "wind_mean", "wind_max",
        // Calendar.
        "doy_sin", "doy_cos",
    };

    /// <summary>
    /// Phase 3d-shape adds 7 within-day rain-structure features derived from the
    /// ensemble-mean hourly precip vector. Layered on top of <see cref="FeatureNames"/>
    /// — never used standalone.
    /// </summary>
    public static readonly IReadOnlyList<string> ShapeFeatureNames = new[]
    {
        "first_wet_hour", "last_wet_hour",
        "longest_forecast_dry_run_hours", "longest_forecast_wet_run_hours",
        "n_rain_events",
        "morning_precip_sum", "afternoon_precip_sum",
    };

    /// <summary>Phase identifier strings persisted in <c>training_metadata.Phase</c>.</summary>
    public const string Phase3b = "3b";
    public const string Phase3dShape = "3d-shape";
    public const string Phase3dCalibrated = "3d-calibrated";

    /// <summary>
    /// Resolve the feature-column ordering for a given training phase. 3b and
    /// 3d-calibrated reuse the base 53 columns (calibrated re-maps 3b's output);
    /// 3d-shape appends the 7 shape columns. Unknown phases default to base.
    /// </summary>
    public static IReadOnlyList<string> FeatureNamesForPhase(string phase)
        => phase == Phase3dShape
            ? FeatureNames.Concat(ShapeFeatureNames).ToArray()
            : FeatureNames;

    /// <summary>Model identifier order aligned with per-model feature suffixes.</summary>
    public static readonly IReadOnlyList<string> ModelIds = new[]
    {
        "gfs_seamless", "ecmwf_ifs025", "icon_seamless",
        "meteofrance_seamless", "ukmo_seamless", "gem_seamless",
    };

    public const double WetThresholdMm = PrecipFeatureBuilder.WetThresholdMm;

    public static List<DryWindowTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        int leadHours,
        int windowHours,
        CancellationToken ct = default)
    {
        if (windowHours < 1 || windowHours > 24)
            throw new ArgumentOutOfRangeException(nameof(windowHours));

        // --- Step 1: labels via the shared builder (truth side) -----------------
        var truth = LoadHourlyTruth(rainfallPath, locationName, stationName, ct);
        var labels = DryWindowLabelBuilder.Build(truth, new[] { windowHours });
        if (labels.Labels.Count == 0) return new List<DryWindowTrainingRow>();

        var labelByDate = labels.Labels
            .Where(l => l.WindowHours == windowHours)
            .ToDictionary(l => l.Date, l => l.HasDryWindow);

        // Daily truth totals for diagnostics (not a feature).
        var truthMmByDate = truth
            .GroupBy(h => DateOnly.FromDateTime(h.HourUtc))
            .ToDictionary(g => g.Key, g => g.Sum(h => h.PrecipMmHour));

        // --- Step 2: forecasts for the lead band --------------------------------
        var forecast = LoadForecasts(
            forecastsPath, locationName, leadHours, leadHours + 23, ct);

        // Bucket forecast rows by (target-date, model, hour-of-day).
        // Target-date for forecast hour (ValidTimeUtc) is simply DateOnly.FromDateTime(valid).
        // hour-of-day ∈ [0, 23] tells us which slot this row fills for the per-model 24-vector.
        var perModelDay = new Dictionary<(DateOnly Date, string Model), ForecastDay>();
        foreach (var f in forecast)
        {
            ct.ThrowIfCancellationRequested();
            var d = DateOnly.FromDateTime(f.ValidTimeUtc);
            var key = (d, f.Model);
            if (!perModelDay.TryGetValue(key, out var day))
            {
                day = new ForecastDay();
                perModelDay[key] = day;
            }
            day.SetHour(f.ValidTimeUtc.Hour, f);
        }

        // --- Step 3: compose one training row per labeled date ------------------
        var rows = new List<DryWindowTrainingRow>(labelByDate.Count);
        foreach (var (date, label) in labelByDate.OrderBy(kv => kv.Key))
        {
            ct.ThrowIfCancellationRequested();

            var modelDays = new List<ForecastDay?>(ModelIds.Count);
            foreach (var modelId in ModelIds)
                modelDays.Add(perModelDay.TryGetValue((date, modelId), out var d) ? d : null);

            // If no model supplied any hour for this day, drop it — we have no features.
            if (!modelDays.Any(d => d is { AnyPresent: true })) continue;

            var row = ComposeRow(date, windowHours, modelDays, label,
                truthMmByDate.TryGetValue(date, out var mmDay) ? mmDay : 0.0);
            rows.Add(row);
        }
        return rows;
    }

    internal static DryWindowTrainingRow ComposeRow(
        DateOnly targetDate,
        int windowHours,
        IReadOnlyList<ForecastDay?> modelDays,
        bool label,
        double truthMmDay)
    {
        // Per-model features. NaN for models that don't have a complete 24-hour day.
        var perModelSum       = new double[6];
        var perModelMaxHour   = new double[6];
        var perModelWetCount  = new double[6];
        var perModelLongestDry = new double[6];
        var perModelHasDry    = new double[6];
        var perModelProbMax   = new double[6];

        for (int i = 0; i < 6; i++)
        {
            var day = modelDays[i];
            if (day is null || !day.IsComplete)
            {
                perModelSum[i] = double.NaN;
                perModelMaxHour[i] = double.NaN;
                perModelWetCount[i] = double.NaN;
                perModelLongestDry[i] = double.NaN;
                perModelHasDry[i] = double.NaN;
                perModelProbMax[i] = double.NaN;
                continue;
            }

            double sum = 0, maxHr = 0, probMax = double.NaN;
            int wetHours = 0, run = 0, longest = 0;
            bool hasAnyProb = false;
            for (int h = 0; h < 24; h++)
            {
                var p = day.Precip[h] ?? 0.0;
                sum += p;
                if (p > maxHr) maxHr = p;
                if (p >= WetThresholdMm) { wetHours++; run = 0; }
                else                     { run++; if (run > longest) longest = run; }

                var pr = day.Prob[h];
                if (pr.HasValue)
                {
                    if (!hasAnyProb || pr.Value > probMax) probMax = pr.Value;
                    hasAnyProb = true;
                }
            }

            perModelSum[i]       = sum;
            perModelMaxHour[i]   = maxHr;
            perModelWetCount[i]  = wetHours;
            perModelLongestDry[i] = longest;
            perModelHasDry[i]    = longest >= windowHours ? 1.0 : 0.0;
            perModelProbMax[i]   = hasAnyProb ? probMax : double.NaN;
        }

        // Ensemble summaries (NaN-skip across the six per-model slots).
        (double mean, double std, double max) = MeanStdMax(perModelSum);
        var longestMean  = NaNMean(perModelLongestDry);
        var wetHoursMean = NaNMean(perModelWetCount);
        var agreement    = NaNMean(perModelHasDry); // fraction-of-models-saying-yes

        // Covariates from the ensemble-mean time series across the 24 hours.
        var env = EnvelopeCovariates(modelDays);

        // Shape features (Phase 3d) — derived from the ensemble-mean hourly precip vector.
        var shape = ShapeFeatures(modelDays);

        var doyAngle = 2.0 * Math.PI * (targetDate.DayOfYear - 1) / 365.0;

        return new DryWindowTrainingRow
        {
            TargetDateUtc = targetDate.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc),
            WindowHours = windowHours,

            PrecipSumGfs   = (float)perModelSum[0],
            PrecipSumEcmwf = (float)perModelSum[1],
            PrecipSumIcon  = (float)perModelSum[2],
            PrecipSumMf    = (float)perModelSum[3],
            PrecipSumUkmo  = (float)perModelSum[4],
            PrecipSumGem   = (float)perModelSum[5],

            PrecipMaxHourGfs   = (float)perModelMaxHour[0],
            PrecipMaxHourEcmwf = (float)perModelMaxHour[1],
            PrecipMaxHourIcon  = (float)perModelMaxHour[2],
            PrecipMaxHourMf    = (float)perModelMaxHour[3],
            PrecipMaxHourUkmo  = (float)perModelMaxHour[4],
            PrecipMaxHourGem   = (float)perModelMaxHour[5],

            WetHourCountGfs   = (float)perModelWetCount[0],
            WetHourCountEcmwf = (float)perModelWetCount[1],
            WetHourCountIcon  = (float)perModelWetCount[2],
            WetHourCountMf    = (float)perModelWetCount[3],
            WetHourCountUkmo  = (float)perModelWetCount[4],
            WetHourCountGem   = (float)perModelWetCount[5],

            LongestDryRunGfs   = (float)perModelLongestDry[0],
            LongestDryRunEcmwf = (float)perModelLongestDry[1],
            LongestDryRunIcon  = (float)perModelLongestDry[2],
            LongestDryRunMf    = (float)perModelLongestDry[3],
            LongestDryRunUkmo  = (float)perModelLongestDry[4],
            LongestDryRunGem   = (float)perModelLongestDry[5],

            HasDryWindowGfs   = (float)perModelHasDry[0],
            HasDryWindowEcmwf = (float)perModelHasDry[1],
            HasDryWindowIcon  = (float)perModelHasDry[2],
            HasDryWindowMf    = (float)perModelHasDry[3],
            HasDryWindowUkmo  = (float)perModelHasDry[4],
            HasDryWindowGem   = (float)perModelHasDry[5],

            ProbMaxGfs   = (float)perModelProbMax[0],
            ProbMaxEcmwf = (float)perModelProbMax[1],
            ProbMaxIcon  = (float)perModelProbMax[2],
            ProbMaxMf    = (float)perModelProbMax[3],
            ProbMaxUkmo  = (float)perModelProbMax[4],
            ProbMaxGem   = (float)perModelProbMax[5],

            PrecipSumMean = (float)mean,
            PrecipSumStd  = (float)std,
            PrecipSumMax  = (float)max,
            AgreementHasDryWindow = (float)agreement,
            LongestDryRunMean = (float)longestMean,
            WetHourCountMean  = (float)wetHoursMean,

            RhMean = (float)env.RhMean,
            RhMin  = (float)env.RhMin,
            DewDepressionMax = (float)env.DewDepressionMax,
            CloudLowMean  = (float)env.CloudLowMean,
            CloudMidMean  = (float)env.CloudMidMean,
            CloudHighMean = (float)env.CloudHighMean,
            CapeMax  = (float)env.CapeMax,
            WindMean = (float)env.WindMean,
            WindMax  = (float)env.WindMax,

            DoySin = (float)Math.Sin(doyAngle),
            DoyCos = (float)Math.Cos(doyAngle),

            FirstWetHour = (float)shape.FirstWetHour,
            LastWetHour  = (float)shape.LastWetHour,
            LongestForecastDryRunHours = (float)shape.LongestDryRun,
            LongestForecastWetRunHours = (float)shape.LongestWetRun,
            NRainEvents       = (float)shape.NRainEvents,
            MorningPrecipSum   = (float)shape.MorningPrecipSum,
            AfternoonPrecipSum = (float)shape.AfternoonPrecipSum,

            HasDryWindow = label,
            PrecipMmDay = (float)truthMmDay,
        };
    }

    // ------------------------------------------------------------------------
    // SQL loaders
    // ------------------------------------------------------------------------

    private static List<DryWindowLabelBuilder.HourlyTruth> LoadHourlyTruth(
        string rainfallPath, string locationName, string stationName, CancellationToken ct)
    {
        var glob = NormaliseGlob(Path.Combine(rainfallPath, "**", "*.parquet"));
        var escStation  = stationName.Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");

        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLocation}'
  AND StationName  = '{escStation}'
  AND Value15MinMm IS NOT NULL
GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var rows = new List<DryWindowLabelBuilder.HourlyTruth>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var hour = DateTime.SpecifyKind(r.GetDateTime(0), DateTimeKind.Utc);
            rows.Add(new DryWindowLabelBuilder.HourlyTruth(hour, r.GetDouble(1)));
        }
        return rows;
    }

    internal sealed record ForecastRow(
        DateTime ValidTimeUtc, string Model,
        double? Precip, double? Prob,
        double? Rh, double? T, double? Td,
        double? CloudLow, double? CloudMid, double? CloudHigh,
        double? Cape, double? Wind);

    private static List<ForecastRow> LoadForecasts(
        string forecastsPath, string locationName, int leadLo, int leadHi, CancellationToken ct)
    {
        var glob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var escLocation = locationName.Replace("'", "''");

        var sql = $@"
WITH latest AS (
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
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{escLocation}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours BETWEEN {leadLo} AND {leadHi}
)
SELECT ValidTimeUtc, Model,
       Precipitation, PrecipitationProbability,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       Cape, WindSpeed10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var rows = new List<ForecastRow>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            rows.Add(new ForecastRow(
                DateTime.SpecifyKind(r.GetDateTime(0), DateTimeKind.Utc),
                r.GetString(1),
                r.IsDBNull(2) ? null : r.GetDouble(2),
                r.IsDBNull(3) ? null : r.GetDouble(3),
                r.IsDBNull(4) ? null : r.GetDouble(4),
                r.IsDBNull(5) ? null : r.GetDouble(5),
                r.IsDBNull(6) ? null : r.GetDouble(6),
                r.IsDBNull(7) ? null : r.GetDouble(7),
                r.IsDBNull(8) ? null : r.GetDouble(8),
                r.IsDBNull(9) ? null : r.GetDouble(9),
                r.IsDBNull(10) ? null : r.GetDouble(10),
                r.IsDBNull(11) ? null : r.GetDouble(11)));
        }
        return rows;
    }

    // ------------------------------------------------------------------------
    // Per-model day holder
    // ------------------------------------------------------------------------

    internal sealed class ForecastDay
    {
        public double?[] Precip { get; } = new double?[24];
        public double?[] Prob   { get; } = new double?[24];
        public double?[] Rh     { get; } = new double?[24];
        public double?[] T      { get; } = new double?[24];
        public double?[] Td     { get; } = new double?[24];
        public double?[] CloudLow  { get; } = new double?[24];
        public double?[] CloudMid  { get; } = new double?[24];
        public double?[] CloudHigh { get; } = new double?[24];
        public double?[] Cape { get; } = new double?[24];
        public double?[] Wind { get; } = new double?[24];

        public bool AnyPresent { get; private set; }

        public bool IsComplete
        {
            get
            {
                for (int h = 0; h < 24; h++)
                    if (!Precip[h].HasValue) return false;
                return true;
            }
        }

        public void SetHour(int hour, ForecastRow row)
        {
            Precip[hour] = row.Precip;
            Prob[hour]   = row.Prob;
            Rh[hour]     = row.Rh;
            T[hour]      = row.T;
            Td[hour]     = row.Td;
            CloudLow[hour]  = row.CloudLow;
            CloudMid[hour]  = row.CloudMid;
            CloudHigh[hour] = row.CloudHigh;
            Cape[hour] = row.Cape;
            Wind[hour] = row.Wind;
            AnyPresent = true;
        }
    }

    // ------------------------------------------------------------------------
    // Aggregation helpers
    // ------------------------------------------------------------------------

    private static (double Mean, double Std, double Max) MeanStdMax(double[] xs)
    {
        double sum = 0, sumSq = 0, max = double.NegativeInfinity;
        int n = 0;
        foreach (var x in xs)
        {
            if (double.IsNaN(x)) continue;
            sum += x; sumSq += x * x;
            if (x > max) max = x;
            n++;
        }
        if (n == 0) return (double.NaN, double.NaN, double.NaN);
        var mean = sum / n;
        var std = n <= 1 ? 0.0 : Math.Sqrt(Math.Max(0, (sumSq / n) - mean * mean));
        return (mean, std, max);
    }

    private static double NaNMean(double[] xs)
    {
        double sum = 0; int n = 0;
        foreach (var x in xs) { if (double.IsNaN(x)) continue; sum += x; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    private readonly record struct EnvelopeResult(
        double RhMean, double RhMin, double DewDepressionMax,
        double CloudLowMean, double CloudMidMean, double CloudHighMean,
        double CapeMax, double WindMean, double WindMax);

    /// <summary>
    /// Build an "ensemble-mean" time series from the available models, then
    /// reduce each variable to day-level aggregates. NaN at an hour means no
    /// model supplied that variable — the aggregate skips it.
    /// </summary>
    private static EnvelopeResult EnvelopeCovariates(IReadOnlyList<ForecastDay?> modelDays)
    {
        double rhSum = 0, rhCount = 0, rhMin = double.PositiveInfinity;
        double ddMax = double.NegativeInfinity;
        double clSum = 0, cmSum = 0, chSum = 0;
        int clCount = 0, cmCount = 0, chCount = 0;
        double capeMax = double.NegativeInfinity;
        double windSum = 0, windCount = 0, windMax = double.NegativeInfinity;

        for (int h = 0; h < 24; h++)
        {
            double rhH = 0, tH = 0, tdH = 0, clH = 0, cmH = 0, chH = 0, capeH = 0, windH = 0;
            int rhN = 0, tN = 0, tdN = 0, clN = 0, cmN = 0, chN = 0, capeN = 0, windN = 0;
            foreach (var day in modelDays)
            {
                if (day is null) continue;
                if (day.Rh[h].HasValue)     { rhH   += day.Rh[h]!.Value;    rhN++; }
                if (day.T[h].HasValue)      { tH    += day.T[h]!.Value;     tN++; }
                if (day.Td[h].HasValue)     { tdH   += day.Td[h]!.Value;    tdN++; }
                if (day.CloudLow[h].HasValue)  { clH += day.CloudLow[h]!.Value;  clN++; }
                if (day.CloudMid[h].HasValue)  { cmH += day.CloudMid[h]!.Value;  cmN++; }
                if (day.CloudHigh[h].HasValue) { chH += day.CloudHigh[h]!.Value; chN++; }
                if (day.Cape[h].HasValue)      { capeH += day.Cape[h]!.Value;    capeN++; }
                if (day.Wind[h].HasValue)      { windH += day.Wind[h]!.Value;    windN++; }
            }

            if (rhN > 0)
            {
                var rh = rhH / rhN;
                rhSum += rh; rhCount++;
                if (rh < rhMin) rhMin = rh;
            }
            if (tN > 0 && tdN > 0)
            {
                var dd = (tH / tN) - (tdH / tdN);
                if (dd > ddMax) ddMax = dd;
            }
            if (clN > 0) { clSum += clH / clN; clCount++; }
            if (cmN > 0) { cmSum += cmH / cmN; cmCount++; }
            if (chN > 0) { chSum += chH / chN; chCount++; }
            if (capeN > 0)
            {
                var cape = capeH / capeN;
                if (cape > capeMax) capeMax = cape;
            }
            if (windN > 0)
            {
                var w = windH / windN;
                windSum += w; windCount++;
                if (w > windMax) windMax = w;
            }
        }

        return new EnvelopeResult(
            RhMean: rhCount == 0 ? double.NaN : rhSum / rhCount,
            RhMin:  double.IsPositiveInfinity(rhMin) ? double.NaN : rhMin,
            DewDepressionMax: double.IsNegativeInfinity(ddMax) ? double.NaN : ddMax,
            CloudLowMean:  clCount == 0 ? double.NaN : clSum / clCount,
            CloudMidMean:  cmCount == 0 ? double.NaN : cmSum / cmCount,
            CloudHighMean: chCount == 0 ? double.NaN : chSum / chCount,
            CapeMax:  double.IsNegativeInfinity(capeMax) ? double.NaN : capeMax,
            WindMean: windCount == 0 ? double.NaN : windSum / windCount,
            WindMax:  double.IsNegativeInfinity(windMax) ? double.NaN : windMax);
    }

    internal readonly record struct ShapeResult(
        double FirstWetHour, double LastWetHour,
        double LongestDryRun, double LongestWetRun,
        double NRainEvents,
        double MorningPrecipSum, double AfternoonPrecipSum);

    /// <summary>
    /// Phase 3d shape features. Builds an ensemble-mean hourly precip vector
    /// (NaN-skip across models per hour, treating "no model supplied this hour"
    /// as zero — same convention as per-model walks in <see cref="ComposeRow"/>),
    /// then derives within-day rain structure summaries:
    ///   first/last wet hour (sentinels 24/-1 when fully dry),
    ///   longest contiguous dry/wet runs in hours,
    ///   number of rain events (maximal wet runs separated by ≥1 dry hour),
    ///   morning (06–11) and afternoon (12–17) precip totals.
    /// Returns NaN for every feature if no model contributed any hour.
    /// </summary>
    internal static ShapeResult ShapeFeatures(IReadOnlyList<ForecastDay?> modelDays)
    {
        var meanPrecip = new double[24];
        var present = new bool[24];
        bool any = false;
        for (int h = 0; h < 24; h++)
        {
            double sum = 0;
            int n = 0;
            foreach (var day in modelDays)
            {
                if (day is null) continue;
                if (day.Precip[h].HasValue) { sum += day.Precip[h]!.Value; n++; }
            }
            if (n > 0)
            {
                meanPrecip[h] = sum / n;
                present[h] = true;
                any = true;
            }
            else
            {
                meanPrecip[h] = 0.0; // matches per-model "p ?? 0.0" convention
            }
        }

        if (!any)
        {
            var nan = double.NaN;
            return new ShapeResult(nan, nan, nan, nan, nan, nan, nan);
        }

        int firstWet = -1, lastWet = -1;
        int dryRun = 0, wetRun = 0, longestDry = 0, longestWet = 0;
        int nEvents = 0;
        bool inWet = false;
        for (int h = 0; h < 24; h++)
        {
            bool wet = meanPrecip[h] >= WetThresholdMm;
            if (wet)
            {
                if (firstWet < 0) firstWet = h;
                lastWet = h;
                wetRun++;
                if (wetRun > longestWet) longestWet = wetRun;
                dryRun = 0;
                if (!inWet) { nEvents++; inWet = true; }
            }
            else
            {
                dryRun++;
                if (dryRun > longestDry) longestDry = dryRun;
                wetRun = 0;
                inWet = false;
            }
        }

        double morning = 0, afternoon = 0;
        for (int h = 6; h <= 11; h++) morning   += meanPrecip[h];
        for (int h = 12; h <= 17; h++) afternoon += meanPrecip[h];

        return new ShapeResult(
            FirstWetHour: firstWet < 0 ? 24.0 : firstWet,
            LastWetHour:  lastWet,
            LongestDryRun: longestDry,
            LongestWetRun: longestWet,
            NRainEvents: nEvents,
            MorningPrecipSum: morning,
            AfternoonPrecipSum: afternoon);
    }

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");
}
