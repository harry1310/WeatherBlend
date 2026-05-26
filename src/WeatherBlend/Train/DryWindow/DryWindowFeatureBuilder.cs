using System.Text;
using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

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
    /// <summary>Phase identifier strings persisted in <c>training_metadata.Phase</c>.</summary>
    public const string Phase3b = "3b";
    // Phase3dShape / Phase3dCalibrated / Phase3e (the old cascade) retired
    // 2026-05-04. The dry-window-3f MLP local bake-off path was retired
    // 2026-05-25 in model-cleanup Phase 1 (its sibling 3g/3j/3n/3s MC
    // phases went with it). The "3f" identifier is now reserved for the
    // Membury precip-volume model that ships in cleanup Phase 4.

    public const double WetThresholdMm = PrecipFeatureBuilder.WetThresholdMm;

    // ------------------------------------------------------------------------
    // SQL loaders
    // ------------------------------------------------------------------------

    private static List<DryWindowLabelBuilder.HourlyTruth> LoadHourlyTruth(
        string rainfallPath, string locationName, string stationName,
        DateTime? minValidTime, CancellationToken ct)
    {
        var glob = NormaliseGlob(Path.Combine(rainfallPath, "**", "*.parquet"));
        var escStation  = stationName.Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");
        var minObservedFilter = minValidTime.HasValue
            ? $"  AND ObservedTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'\n"
            : string.Empty;

        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLocation}'
  AND StationName  = '{escStation}'
  AND Value15MinMm IS NOT NULL
{minObservedFilter}GROUP BY 1
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

        /// <summary>
        /// True iff every hour inside <c>[startHour, endHour)</c> has a precip
        /// value. Replaces the old "all 24 hours" check — under the daytime
        /// regime, missing overnight hours are fine because we never look
        /// at them, and demanding them would needlessly drop days for
        /// models that only ship a daytime band.
        /// </summary>
        public bool IsCompleteWithin(int startHour, int endHour)
        {
            for (int h = startHour; h < endHour; h++)
                if (!Precip[h].HasValue) return false;
            return true;
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
    /// Build an "ensemble-mean" time series from the available models within
    /// the daytime window <c>[startHour, endHour)</c>, then reduce each
    /// variable to day-level aggregates. NaN at an hour means no model
    /// supplied that variable — the aggregate skips it.
    /// </summary>
    private static EnvelopeResult EnvelopeCovariates(
        IReadOnlyList<ForecastDay?> modelDays, int startHour, int endHour)
    {
        double rhSum = 0, rhCount = 0, rhMin = double.PositiveInfinity;
        double ddMax = double.NegativeInfinity;
        double clSum = 0, cmSum = 0, chSum = 0;
        int clCount = 0, cmCount = 0, chCount = 0;
        double capeMax = double.NegativeInfinity;
        double windSum = 0, windCount = 0, windMax = double.NegativeInfinity;

        for (int h = startHour; h < endHour; h++)
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

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");

    // -----------------------------------------------------------------------
    // New canonical API (Phase 4 of unify-model-membership refactor).
    // Spec-driven, dynamic-shape vector via Common.DryWindowTrainingRow.Features.
    // -----------------------------------------------------------------------

    public const string SpecTarget = "dry_window";

    /// <summary>
    /// Resolve the runtime <see cref="BlenderSpec"/> for dry-window at a given lead.
    /// Phase argument is retained for API compatibility but only the 53-feature
    /// base layout (<see cref="Phase3b"/>, <c>featureSet="base"</c>) is supported
    /// after the 2026-05-04 retirement of 3d-shape / 3e / 3f.
    ///
    /// Layout per active model count N:
    ///   6 per-model variables × N (sum, max_hour, wet_count, longest_dry, has_dry, prob_max)
    ///   6 ensemble summaries
    ///   9 covariates
    ///   2 calendar (doy_sin, doy_cos — no hour features at day granularity)
    /// → 6N + 17.
    /// </summary>
    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours, string phase)
    {
        var featureSet = "base";
        var blender = blendersCfg.Get(SpecTarget, featureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var overlap = requiredSet.Intersect(optionalSet, StringComparer.OrdinalIgnoreCase).ToArray();
        if (overlap.Length > 0)
            throw new InvalidOperationException(
                $"BlenderConfig {SpecTarget}/{featureSet} at lead {leadHours}h has model(s) " +
                $"listed as both required and optional: [{string.Join(",", overlap)}].");

        var orderedRequired = TempFeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = TempFeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = TempFeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{featureSet} at lead {leadHours}h.");

        var names = new List<string>();
        foreach (var m in orderedModels) names.Add($"precip_sum_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"precip_max_hour_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"wet_hour_count_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"longest_dry_run_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"has_dry_window_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"prob_max_{TempFeatureBuilder.ShortName(m)}");
        names.AddRange(new[]
        {
            "precip_sum_mean", "precip_sum_std", "precip_sum_max",
            "agreement_has_dry_window", "longest_dry_run_mean", "wet_hour_count_mean",
        });
        names.AddRange(new[]
        {
            "rh_mean", "rh_min", "dew_depression_max",
            "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
            "cape_max", "wind_mean", "wind_max",
        });
        names.AddRange(new[] { "doy_sin", "doy_cos" });

        return new BlenderSpec
        {
            Target = SpecTarget,
            FeatureSet = featureSet,
            LeadHours = leadHours,
            RequiredModels = orderedRequired,
            OptionalModels = orderedOptional,
            Models = orderedModels,
            FeatureNames = names,
            DataSource = BlenderDataSource.OpenMeteoPreviousRuns,
            Tier = featureSet,
            UkvStrategy = null,
        };
    }

    /// <summary>
    /// Build day-anchored training rows for one (station, lead, window) blender.
    /// SQL pivot pulls only spec.Models. Per-model day features are NaN if a model
    /// doesn't supply every hour inside the configured daytime window.
    /// Required-model gate isn't enforced at the day level — dry-window's
    /// policy is "include the day if any model supplied any hour" (matches
    /// legacy AnyPresent check).
    /// </summary>
    public static List<CommonRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        BlenderSpec spec,
        int windowHours,
        DaytimeWindow daytime,
        DateTime? minValidTime = null,
        CancellationToken ct = default)
    {
        if (windowHours < 1 || windowHours > daytime.DurationHours)
            throw new ArgumentOutOfRangeException(nameof(windowHours),
                $"Window length {windowHours} exceeds the daytime range ({daytime.DurationHours}h).");

        // Step 1: hourly truth + per-day labels.
        var truth = LoadHourlyTruth(rainfallPath, locationName, stationName, minValidTime, ct);
        var labels = DryWindowLabelBuilder.Build(truth, new[] { windowHours }, daytime);
        if (labels.Labels.Count == 0) return new List<CommonRow>();

        var labelByDate = labels.Labels
            .Where(l => l.WindowHours == windowHours)
            .ToDictionary(l => l.Date, l => l.HasDryWindow);

        var truthMmByDate = truth
            .GroupBy(h => DateOnly.FromDateTime(h.HourUtc))
            .ToDictionary(g => g.Key, g => g.Sum(h => h.PrecipMmHour));

        // Step 2: forecasts for the lead band, restricted to spec.Models.
        var forecast = LoadForecastsForSpec(
            forecastsPath, locationName, spec, minValidTime, ct);

        // Bucket per (target-date, model-id) → ForecastDay (24-hour holder).
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

        // Step 3: compose one row per labeled date — features ordered per spec.Models.
        var rows = new List<CommonRow>(labelByDate.Count);
        foreach (var (date, label) in labelByDate.OrderBy(kv => kv.Key))
        {
            ct.ThrowIfCancellationRequested();

            var modelDays = new List<ForecastDay?>(spec.Models.Count);
            foreach (var modelId in spec.Models)
                modelDays.Add(perModelDay.TryGetValue((date, modelId), out var d) ? d : null);

            var (startHour, endHour) = daytime.UtcHourRangeFor(date);

            // Drop the day unless at least one model supplied a COMPLETE precip
            // window across the daytime hours. A bare AnyPresent check (any
            // forecast row exists at all) is too loose: the offset_day archive
            // carries temperature-only rows for 2022-2023 (precip is backfilled
            // only from 2024-01), and those would otherwise enter as all-NaN-
            // precip training rows — doubling the row count and tripping the
            // RetrainGuard (the 2026-05-17 dry-window 3b retrain failure).
            // IsCompleteWithin requires a real precip value at every daytime
            // hour, so a precip-less day is correctly excluded.
            if (!modelDays.Any(d => d is not null && d.IsCompleteWithin(startHour, endHour)))
                continue;

            var row = ComposeRow(spec, date, windowHours, modelDays, label,
                truthMmByDate.TryGetValue(date, out var mmDay) ? mmDay : 0.0,
                startHour, endHour);
            rows.Add(row);
        }
        return rows;
    }

    /// <summary>
    /// Pack the dry-window feature vector. Layout matches <see cref="BuildSpec"/>'s
    /// <c>FeatureNames</c>: 6 per-model groups × N + 6 ensemble summaries + 9 covariates
    /// + 2 calendar. All per-model and ensemble aggregates are restricted to the
    /// daytime hour range <c>[startHour, endHour)</c>.
    /// </summary>
    internal static CommonRow ComposeRow(
        BlenderSpec spec,
        DateOnly targetDate,
        int windowHours,
        IReadOnlyList<ForecastDay?> modelDays,
        bool label,
        double truthMmDay,
        int startHour,
        int endHour)
    {
        var n = spec.Models.Count;
        if (modelDays.Count != n)
            throw new ArgumentException($"Expected {n} model days, got {modelDays.Count}", nameof(modelDays));

        // Per-model features. NaN when the model doesn't supply every daytime hour.
        var perModelSum        = new double[n];
        var perModelMaxHour    = new double[n];
        var perModelWetCount   = new double[n];
        var perModelLongestDry = new double[n];
        var perModelHasDry     = new double[n];
        var perModelProbMax    = new double[n];

        for (int i = 0; i < n; i++)
        {
            var day = modelDays[i];
            if (day is null || !day.IsCompleteWithin(startHour, endHour))
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
            for (int h = startHour; h < endHour; h++)
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
            perModelSum[i] = sum;
            perModelMaxHour[i] = maxHr;
            perModelWetCount[i] = wetHours;
            perModelLongestDry[i] = longest;
            perModelHasDry[i] = longest >= windowHours ? 1.0 : 0.0;
            perModelProbMax[i] = hasAnyProb ? probMax : double.NaN;
        }

        var (sumMean, sumStd, sumMax) = MeanStdMax(perModelSum);
        var longestMean  = NaNMean(perModelLongestDry);
        var wetHoursMean = NaNMean(perModelWetCount);
        var agreement    = NaNMean(perModelHasDry);

        var env = EnvelopeCovariates(modelDays, startHour, endHour);

        var doyAngle = 2.0 * Math.PI * (targetDate.DayOfYear - 1) / 365.0;

        var features = new float[spec.FeatureCount];
        int idx = 0;
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelSum[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelMaxHour[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelWetCount[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelLongestDry[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelHasDry[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelProbMax[i];
        features[idx++] = (float)sumMean;
        features[idx++] = (float)sumStd;
        features[idx++] = (float)sumMax;
        features[idx++] = (float)agreement;
        features[idx++] = (float)longestMean;
        features[idx++] = (float)wetHoursMean;
        features[idx++] = (float)env.RhMean;
        features[idx++] = (float)env.RhMin;
        features[idx++] = (float)env.DewDepressionMax;
        features[idx++] = (float)env.CloudLowMean;
        features[idx++] = (float)env.CloudMidMean;
        features[idx++] = (float)env.CloudHighMean;
        features[idx++] = (float)env.CapeMax;
        features[idx++] = (float)env.WindMean;
        features[idx++] = (float)env.WindMax;
        features[idx++] = (float)Math.Sin(doyAngle);
        features[idx++] = (float)Math.Cos(doyAngle);
        if (idx != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Dry-window feature pack mismatch: wrote {idx}, expected {spec.FeatureCount}");

        return new CommonRow
        {
            TargetDateUtc = targetDate.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc),
            WindowHours = windowHours,
            Features = features,
            Label = label,
            PrecipMmDay = (float)truthMmDay,
        };
    }

    private static List<ForecastRow> LoadForecastsForSpec(
        string forecastsPath, string locationName, BlenderSpec spec,
        DateTime? minValidTime, CancellationToken ct)
    {
        var glob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var escLocation = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var minValidTimeFilter = minValidTime.HasValue
            ? $"      AND ValidTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'\n"
            : string.Empty;

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
      AND LeadHours BETWEEN {spec.LeadHours} AND {spec.LeadHours + 23}
      AND Model IN {modelInClause}
{minValidTimeFilter})
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
}
