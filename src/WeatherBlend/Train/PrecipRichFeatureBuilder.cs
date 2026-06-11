using System.Text;
using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Builds the 55-feature Phase 3c occurrence-blender dataset for a single lead.
///
/// Structure mirrors <see cref="PrecipFeatureBuilder"/> — same RunTimeSource filter,
/// same per-model pivot, same hourly-rainfall label — but pulls per-model humidity
/// (dew, rh, temperature-minus-dewpoint) and surface pressure in addition to the
/// lean covariate aggregates, and derives four EA-observation persistence features
/// anchored at <c>run_time = valid_time - leadHours</c>.
///
/// Kept parallel to <see cref="PrecipFeatureBuilder"/> rather than extending it: the
/// SQL shape and row-composition logic are separate concerns and there's no value in
/// fusing them. Phase 3a stays bit-for-bit unchanged.
///
/// Features that would need trailing-lead cells from the same run (same-run precip
/// persistence, pressure tendency) are NOT included — Phase 1 training parquet only
/// persists leads {24, 48, 72} per <c>offset_day</c> run, so the H-1/H-2/H-3 cells
/// those features need don't exist. Live-cycle data has them; training data doesn't.
/// Switching training to live cycles would break the "same split as 3a" guarantee,
/// so the tier is out of scope for 3c.
/// </summary>
public static class PrecipRichFeatureBuilder
{
    /// <summary>
    /// Trailing-rainfall persistence anchored at <paramref name="runTimeUtc"/>. Any
    /// feature whose supporting hours aren't fully present in <paramref name="hourly"/>
    /// returns <see cref="double.NaN"/> — same "don't fabricate partial data" rule the
    /// hourly label uses (HAVING COUNT(*) = 4).
    /// </summary>
    public static Persistence ComputePersistence(
        IReadOnlyDictionary<DateTime, double> hourly,
        DateTime runTimeUtc)
    {
        // Window: (runTime - N, runTime]. runTime itself is included — it's an
        // observation the cycle would have had in hand at issue time.
        double sum24 = 0, sum72 = 0;
        int wet24 = 0;
        bool cover24 = true, cover72 = true;
        for (int h = 0; h < 72; h++)
        {
            var t = runTimeUtc.AddHours(-h);
            if (!hourly.TryGetValue(t, out var mm))
            {
                if (h < 24) cover24 = false;
                cover72 = false;
                continue;
            }
            sum72 += mm;
            if (h < 24)
            {
                sum24 += mm;
                if (mm >= PrecipFeatureBuilder.WetThresholdMm) wet24++;
            }
        }

        // Trailing dry run: consecutive dry hours ending at runTime, walking
        // backwards until a wet hour or a missing reading. Missing reading stops
        // the count — we can't claim "dry since X" if X doesn't exist.
        int dryRun = 0;
        for (int h = 0; h < 72; h++)
        {
            var t = runTimeUtc.AddHours(-h);
            if (!hourly.TryGetValue(t, out var mm)) break;
            if (mm > PrecipFeatureBuilder.WetThresholdMm) break;
            dryRun++;
        }

        return new Persistence(
            Prev24hMm: cover24 ? sum24 : double.NaN,
            Prev72hMm: cover72 ? sum72 : double.NaN,
            WetHoursLast24h: cover24 ? wet24 : double.NaN,
            DryHoursTrailing: dryRun);
    }

    public readonly record struct Persistence(
        double Prev24hMm,
        double Prev72hMm,
        double WetHoursLast24h,
        double DryHoursTrailing);

    /// <summary>
    /// Hourly rainfall (mm) keyed by hour-of-observation UTC. Public for predict-time
    /// reuse — the Phase 3c inference path needs the same 4-of-4 aggregation to
    /// compute persistence features at run_time anchors.
    /// </summary>
    public static Dictionary<DateTime, double> LoadHourlyRain(
        string rainfallPath,
        string locationName,
        string stationName,
        DateTime? minValidTime,
        CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var rnGlob = SqlGlob.Escape(Path.Combine(rainfallPath, "**", "*.parquet"));
        var escLocation = locationName.Replace("'", "''");
        var escStation  = stationName.Replace("'", "''");
        var minObservedFilter = minValidTime.HasValue
            ? $"  AND ObservedTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'\n"
            : string.Empty;

        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm
FROM read_parquet('{rnGlob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLocation}'
  AND StationName  = '{escStation}'
  AND Value15MinMm IS NOT NULL
{minObservedFilter}GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var dict = new Dictionary<DateTime, double>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                dict[r.GetDateTime(0)] = r.GetDouble(1);
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            // DuckDB's raw "No files found" crash is nearly un-diagnosable inside a
            // CI log — the actual mistake is upstream (workflow didn't sync the
            // rainfall tree). Translate to a clear InvalidOperationException with the
            // path that was checked so operators can go straight to the fix.
            throw new InvalidOperationException(
                $"Rainfall truth tree is empty at '{rainfallPath}'. " +
                $"Phase 3c precip requires the hourly EA rainfall observations for " +
                $"persistence features anchored at run_time = valid_time - leadHours; " +
                $"sync 'data/truth/rainfall' from R2 before invoking the command.",
                ex);
        }
        return dict;
    }

    // -----------------------------------------------------------------------
    // New canonical API (Phase 3 of unify-model-membership refactor).
    // Spec-driven, dynamic-shape vector via BinaryTrainingRow.Features.
    // -----------------------------------------------------------------------

    public const string SpecTarget = "precipitation";
    public const string SpecFeatureSet = "rich";

    /// <summary>Resolve the runtime <see cref="BlenderSpec"/> for rich precipitation at a given lead.</summary>
    /// <param name="withUpperAir">Experimental (2026-06-01): append the FULL
    /// multi-level pressure block (per exact model × <see cref="PrecipFeatureBuilder.UaPressureCols"/>
    /// + ensemble t850_mean/rh850_mean) to test whether upper-air adds value ON
    /// TOP of the rich surface features (humidity / dew-depression / EA
    /// persistence) — i.e. is the lean 3a-UA win orthogonal or already captured
    /// here. Appended LAST so the baseline rich indices are untouched.</param>
    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours, bool withUpperAir = false)
        // Shared membership/guards/spec-field boilerplate in BlenderSpec.Build.
        // The UA variant stamps FeatureSet "rich-ua" (Tier stays "rich").
        => BlenderSpec.Build(blendersCfg, SpecTarget, SpecFeatureSet, leadHours, orderedModels =>
        {
            // Layout (N = orderedModels.Count) — prob_* removed 2026-04-28 (zero-gain
            // features, see PrecipFeatureBuilder.BuildSpec for full reasoning):
            //   N precip + 4 spread + 7 covariates + 4 calendar  (= N+15, the lean precip block)
            //   N dew + N rh + N dew_depression + N pressure     (= 4N rich secondaries)
            //   4 EA persistence
            // Total = 5N + 19. N=8 (with JMA) → 59 features; N=7 → 54.
            var names = new List<string>();
            foreach (var m in orderedModels) names.Add($"precip_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[] { "precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01" });
            names.AddRange(new[]
            {
                "rh_mean", "dew_depression_mean",
                "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
                "cape_mean", "wind_speed_mean",
            });
            names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });
            foreach (var m in orderedModels) names.Add($"dew_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"rh_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"dew_depression_{TempFeatureBuilder.ShortName(m)}");
            foreach (var m in orderedModels) names.Add($"pressure_{TempFeatureBuilder.ShortName(m)}");
            names.AddRange(new[]
            {
                "ea_rain_prev_24h_mm",
                "ea_rain_prev_72h_mm",
                "ea_wet_hours_last_24h",
                "ea_dry_hours_trailing",
            });

            // Experimental upper-air block — APPENDED last (after EA persistence) so
            // every baseline rich index is untouched; ComposeRow's exact-count guard
            // catches a miscount. Same FULL set + lead-matched backward-ASOF join the
            // lean 3a-UA path uses (see PrecipFeatureBuilder), so the only variable
            // between this and the 3a-UA A/B is the surface-feature richness.
            if (withUpperAir)
            {
                foreach (var (_, s) in PrecipFeatureBuilder.UpperAirModels)
                    foreach (var (_, sh) in PrecipFeatureBuilder.UaPressureCols)
                        names.Add($"{sh}_{s}");
                names.Add("t850_mean");
                names.Add("rh850_mean");
            }
            return names;
        }, featureSetLabel: withUpperAir ? SpecFeatureSet + "-ua" : null);

    /// <summary>
    /// Build training rows for one (target=precipitation, feature-set=rich, station, lead).
    /// SQL pivot includes only spec.Models columns. Post-pivot WHERE: every required
    /// per-model precip NOT NULL AND at least one of all NOT NULL. EA persistence
    /// features are derived from the same hourly rainfall index used by 3a.
    /// </summary>
    public static List<BinaryTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        BlenderSpec spec,
        DateTime? minValidTime = null,
        CancellationToken ct = default,
        DateTime? maxValidTime = null)
    {
        var hourlyRain = LoadHourlyRain(rainfallPath, locationName, stationName, minValidTime, ct);
        if (hourlyRain.Count == 0) return new List<BinaryTrainingRow>();

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = SqlGlob.Escape(Path.Combine(forecastsPath, "**", "*.parquet"));
        var escLocation = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var n = spec.Models.Count;

        // Experimental upper-air block — identical mechanism to the lean 3a-UA
        // path (PrecipFeatureBuilder.BuildForLead): for each offset_day valid V,
        // a leak-free backward ASOF attaches the freshest exact lead-{LeadHours}
        // multi-level pressure forecast with valid_time <= V (issued >= L hours
        // before V, genuinely in hand). LEFT join so pre-pressure rows get NULL
        // UA (LightGBM handles gaps). Detected off the spec so Build stays in
        // sync with BuildSpec.
        var withUpperAir = spec.FeatureNames.Contains("t850_mean");
        var uaModels = PrecipFeatureBuilder.UpperAirModels;
        var uaCols = PrecipFeatureBuilder.UaPressureCols;
        string uaCte = "", uaSelectSql = "", uaJoin = "";
        if (withUpperAir)
        {
            var uaModelIn = "(" + string.Join(",", uaModels.Select(x => $"'{x.Model}'")) + ")";
            var uaPivots = string.Join(",\n           ", uaModels.SelectMany(x =>
                uaCols.Select(c => $"MAX(CASE WHEN Model = '{x.Model}' THEN {c.Col} END) AS {c.Short}_{x.Short}")));
            var uaInnerCols = string.Join(", ", uaCols.Select(c => c.Col));
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
            uaSelectSql = ",\n    " + string.Join(", ", uaModels.SelectMany(x =>
                uaCols.Select(c => $"x.{c.Short}_{x.Short}")));
            uaJoin = "ASOF LEFT JOIN exact_ua x ON pivoted.ValidTimeUtc >= x.valid_time_ua";
        }

        var sb = new StringBuilder();
        sb.AppendLine("WITH latest AS (");
        sb.AppendLine("    SELECT ValidTimeUtc, Model,");
        sb.AppendLine("           Precipitation, PrecipitationProbability,");
        sb.AppendLine("           RelativeHumidity2m, Temperature2m, DewPoint2m,");
        sb.AppendLine("           CloudCoverLow, CloudCoverMid, CloudCoverHigh,");
        sb.AppendLine("           Cape, WindSpeed10m, SurfacePressure,");
        sb.AppendLine("           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn");
        sb.AppendLine($"    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)");
        sb.AppendLine($"    WHERE LocationName = '{escLocation}'");
        sb.AppendLine("      AND RunTimeSource = 'offset_day'");
        sb.AppendLine($"      AND LeadHours = {spec.LeadHours}");
        sb.AppendLine($"      AND Model IN {modelInClause}");
        if (minValidTime.HasValue)
            sb.AppendLine($"      AND ValidTimeUtc >= TIMESTAMP '{minValidTime.Value:yyyy-MM-dd HH:mm:ss}'");
        if (maxValidTime.HasValue)
            sb.AppendLine($"      AND ValidTimeUtc <= TIMESTAMP '{maxValidTime.Value:yyyy-MM-dd HH:mm:ss}'");
        sb.AppendLine("),");
        sb.AppendLine("pivoted AS (");
        sb.AppendLine("    SELECT ValidTimeUtc,");
        for (int i = 0; i < n; i++)
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN Precipitation END) AS precip_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        for (int i = 0; i < n; i++)
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN DewPoint2m END) AS dew_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        for (int i = 0; i < n; i++)
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN RelativeHumidity2m END) AS rh_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        for (int i = 0; i < n; i++)
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN Temperature2m - DewPoint2m END) AS dewdep_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        for (int i = 0; i < n; i++)
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN SurfacePressure END) AS pressure_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine("        AVG(RelativeHumidity2m)         AS rh_mean,");
        sb.AppendLine("        AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,");
        sb.AppendLine("        AVG(CloudCoverLow)  AS cloud_low_mean,");
        sb.AppendLine("        AVG(CloudCoverMid)  AS cloud_mid_mean,");
        sb.AppendLine("        AVG(CloudCoverHigh) AS cloud_high_mean,");
        sb.AppendLine("        AVG(Cape)           AS cape_mean,");
        sb.AppendLine("        AVG(WindSpeed10m)   AS wind_speed_mean");
        sb.AppendLine("    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc");
        sb.AppendLine(")");
        sb.Append(uaCte);
        if (withUpperAir) sb.AppendLine();
        sb.AppendLine("SELECT ValidTimeUtc,");
        for (int i = 0; i < n; i++) sb.Append($"    precip_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        for (int i = 0; i < n; i++) sb.Append($"    dew_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        for (int i = 0; i < n; i++) sb.Append($"    rh_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        for (int i = 0; i < n; i++) sb.Append($"    dewdep_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        for (int i = 0; i < n; i++) sb.Append($"    pressure_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        sb.AppendLine("    rh_mean, dew_depression_mean,");
        sb.AppendLine("    cloud_low_mean, cloud_mid_mean, cloud_high_mean,");
        sb.Append("    cape_mean, wind_speed_mean");
        sb.AppendLine(uaSelectSql);
        sb.AppendLine("FROM pivoted");
        if (withUpperAir) sb.AppendLine(uaJoin);
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"precip_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"precip_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";
        sb.AppendLine($"WHERE ({requiredNotNull})");
        sb.AppendLine($"  AND {anyNotNull}");
        sb.AppendLine("ORDER BY ValidTimeUtc;");

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sb.ToString();
        using var r = cmd.ExecuteReader();

        var rows = new List<BinaryTrainingRow>();
        var precip = new double[n];
        var dew    = new double[n];
        var rh     = new double[n];
        var dewdep = new double[n];
        var pres   = new double[n];

        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            if (!hourlyRain.TryGetValue(valid, out var truth))
                continue;  // No label — drop row, matches lean builder's join semantics.

            int col = 1;
            for (int i = 0; i < n; i++) { precip[i] = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            for (int i = 0; i < n; i++) { dew[i]    = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            for (int i = 0; i < n; i++) { rh[i]     = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            for (int i = 0; i < n; i++) { dewdep[i] = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            for (int i = 0; i < n; i++) { pres[i]   = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            var rhMean   = r.IsDBNull(col + 0) ? double.NaN : r.GetDouble(col + 0);
            var dewDepMn = r.IsDBNull(col + 1) ? double.NaN : r.GetDouble(col + 1);
            var cL       = r.IsDBNull(col + 2) ? double.NaN : r.GetDouble(col + 2);
            var cM       = r.IsDBNull(col + 3) ? double.NaN : r.GetDouble(col + 3);
            var cH       = r.IsDBNull(col + 4) ? double.NaN : r.GetDouble(col + 4);
            var cape     = r.IsDBNull(col + 5) ? double.NaN : r.GetDouble(col + 5);
            var wind     = r.IsDBNull(col + 6) ? double.NaN : r.GetDouble(col + 6);

            // Trailing upper-air columns (after wind_speed_mean), in
            // UpperAirModels × UaPressureCols order, then ensemble t850_mean +
            // rh850_mean computed in .NET (mirrors PrecipFeatureBuilder).
            double[]? uaValues = null;
            if (withUpperAir)
            {
                int uaStart = col + 7;  // col == 1+5N here; covariates occupy col..col+6
                int mc = uaModels.Length;
                int pc = uaCols.Length;
                int t850Off = Array.FindIndex(uaCols, c => c.Short == "t850");
                int rh850Off = Array.FindIndex(uaCols, c => c.Short == "rh850");
                uaValues = new double[pc * mc + 2];
                double t850Sum = 0, rh850Sum = 0; int t850N = 0, rh850N = 0;
                for (int k = 0; k < mc; k++)
                    for (int j = 0; j < pc; j++)
                    {
                        var ucol = uaStart + pc * k + j;
                        var uv = r.IsDBNull(ucol) ? double.NaN : r.GetDouble(ucol);
                        uaValues[pc * k + j] = uv;
                        if (j == t850Off && !double.IsNaN(uv)) { t850Sum += uv; t850N++; }
                        if (j == rh850Off && !double.IsNaN(uv)) { rh850Sum += uv; rh850N++; }
                    }
                uaValues[pc * mc]     = t850N == 0 ? double.NaN : t850Sum / t850N;
                uaValues[pc * mc + 1] = rh850N == 0 ? double.NaN : rh850Sum / rh850N;
            }

            var runTime = valid.AddHours(-spec.LeadHours);
            var persistence = ComputePersistence(hourlyRain, runTime);

            rows.Add(ComposeRow(spec, valid, precip, dew, rh, dewdep, pres,
                rhMean, dewDepMn, cL, cM, cH, cape, wind,
                persistence.Prev24hMm, persistence.Prev72hMm,
                persistence.WetHoursLast24h, persistence.DryHoursTrailing,
                truth, uaValues));
        }
        return rows;
    }

    /// <summary>
    /// Pack the rich precipitation feature vector. Layout matches
    /// <see cref="BuildSpec"/>'s <c>FeatureNames</c>: 2N (precip + prob) + 4 spread
    /// + 7 covariates + 4 calendar + 4N (dew/rh/dewdep/pressure) + 4 EA persistence.
    /// </summary>
    public static BinaryTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> perModelPrecip,
        IReadOnlyList<double> perModelDew,
        IReadOnlyList<double> perModelRh,
        IReadOnlyList<double> perModelDewDepression,
        IReadOnlyList<double> perModelPressure,
        double rhMean,
        double dewDepressionMean,
        double cloudLowMean,
        double cloudMidMean,
        double cloudHighMean,
        double capeMean,
        double windSpeedMean,
        double eaRainPrev24hMm,
        double eaRainPrev72hMm,
        double eaWetHoursLast24h,
        double eaDryHoursTrailing,
        double truthMmHour,
        IReadOnlyList<double>? upperAir = null)
    {
        var n = spec.Models.Count;
        if (perModelPrecip.Count != n) throw new ArgumentException($"Expected {n} model precip values", nameof(perModelPrecip));
        if (perModelDew.Count    != n) throw new ArgumentException($"Expected {n} model dew values",    nameof(perModelDew));
        if (perModelRh.Count     != n) throw new ArgumentException($"Expected {n} model rh values",     nameof(perModelRh));
        if (perModelDewDepression.Count != n) throw new ArgumentException($"Expected {n} model depression values", nameof(perModelDewDepression));
        if (perModelPressure.Count != n) throw new ArgumentException($"Expected {n} model pressure values", nameof(perModelPressure));

        // First N+15 features: per-model precip + 4 spread + 7 covariates + 4 calendar.
        // (prob_* removed 2026-04-28 — see PrecipFeatureBuilder.BuildSpec.)
        // Inlined here (rather than delegated to PrecipFeatureBuilder.ComposeRow) because the
        // lean composer asserts it filled the WHOLE vector — feeding it a rich spec would trip
        // the assertion. Same NaN-safe spread logic, just packed directly into the rich vector.
        var features = new float[spec.FeatureCount];

        double sum = 0, sumSq = 0, max = double.NegativeInfinity;
        int wetCount = 0, presentCount = 0;
        for (int i = 0; i < n; i++)
        {
            var x = perModelPrecip[i];
            if (double.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            if (x > max) max = x;
            if (x >= PrecipFeatureBuilder.WetThresholdMm) wetCount++;
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
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelDew[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelRh[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelDewDepression[i];
        for (int i = 0; i < n; i++) features[idx++] = (float)perModelPressure[i];
        features[idx++] = (float)eaRainPrev24hMm;
        features[idx++] = (float)eaRainPrev72hMm;
        features[idx++] = (float)eaWetHoursLast24h;
        features[idx++] = (float)eaDryHoursTrailing;
        if (upperAir is not null)
            for (int i = 0; i < upperAir.Count; i++)
                features[idx++] = (float)upperAir[i];

        if (idx != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Rich precip feature pack mismatch: wrote {idx}, expected {spec.FeatureCount}");

        return new BinaryTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = truthMmHour >= PrecipFeatureBuilder.WetThresholdMm,
            TruthMmHour = (float)truthMmHour,
        };
    }
}
