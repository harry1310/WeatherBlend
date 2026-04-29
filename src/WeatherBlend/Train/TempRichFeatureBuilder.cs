using System.Text;
using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Builds the rich (Phase 2c) temperature blender's training dataset, one lead at a
/// time. Same chronological pipeline as <see cref="TempFeatureBuilder"/> but pulls 10
/// secondary variables per active model (dew, rh, cloud {total/low/mid/high}, wind
/// speed/dir/gusts, pressure) and computes a small bank of cross-model aggregates
/// on top of the lean block.
///
/// Spec-driven: which models contribute and how (required vs optional vs excluded)
/// comes from <c>blenders.temperature.rich</c> in <c>config.yaml</c>. Layout per N
/// active models: N temps + 3 spread + 4 calendar + N×11 per-model secondaries
/// (wind_dir as sin+cos) + 9 aggregates = 12N + 16. For N=6 → 88 features.
/// </summary>
public static class TempRichFeatureBuilder
{
    public const string SpecTarget = "temperature";
    public const string SpecFeatureSet = "rich";

    /// <summary>
    /// Per-model secondary variables, in stable order. <c>WindDirection10m</c> is
    /// special — it expands to sin+cos (2 columns per model). All other vars are
    /// one column per (var, model).
    /// </summary>
    public static readonly IReadOnlyList<(string SrcColumn, string FeaturePrefix)> SecondaryVars = new[]
    {
        ("DewPoint2m",         "dew"),
        ("RelativeHumidity2m", "rh"),
        ("CloudCover",         "cloud"),
        ("CloudCoverLow",      "cloud_low"),
        ("CloudCoverMid",      "cloud_mid"),
        ("CloudCoverHigh",     "cloud_high"),
        ("WindSpeed10m",       "wind_speed"),
        ("WindDirection10m",   "wind_dir"),     // → sin+cos
        ("WindGusts10m",       "wind_gusts"),
        ("SurfacePressure",    "pressure"),
    };

    /// <summary>Resolve the runtime <see cref="BlenderSpec"/> for rich temperature at a given lead.</summary>
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

        // Layout (N = orderedModels.Count):
        //   N per-model temps, 3 spread, 4 calendar  (= N+7, the lean block)
        //   per-model secondaries (var-major × model): each non-wind-dir var = N cols,
        //     wind_dir = 2N cols (sin+cos)
        //   9 aggregates
        var names = new List<string>();
        foreach (var m in orderedModels) names.Add($"temp_{TempFeatureBuilder.ShortName(m)}");
        names.AddRange(new[] { "temp_mean", "temp_std", "temp_range" });
        names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });

        foreach (var (_, prefix) in SecondaryVars)
        {
            if (prefix == "wind_dir")
            {
                foreach (var m in orderedModels) names.Add($"wind_dir_sin_{TempFeatureBuilder.ShortName(m)}");
                foreach (var m in orderedModels) names.Add($"wind_dir_cos_{TempFeatureBuilder.ShortName(m)}");
            }
            else
            {
                foreach (var m in orderedModels) names.Add($"{prefix}_{TempFeatureBuilder.ShortName(m)}");
            }
        }
        names.AddRange(new[]
        {
            "dew_mean", "dew_std",
            "rh_mean", "rh_std",
            "cloud_mean",
            "wind_speed_mean", "wind_speed_std",
            "pressure_mean", "pressure_std",
        });

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

    /// <summary>
    /// Build training rows for one (target=temperature, feature-set=rich, lead) blender.
    /// SQL pivot includes only spec.Models columns. Post-pivot WHERE: every required
    /// per-model temp NOT NULL AND at least one of all temps NOT NULL.
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
        var n = spec.Models.Count;

        // Column order in the SELECT must match what we read in the loop.
        // Build dynamically from spec.Models so the layout stays in lockstep.
        var sb = new StringBuilder();
        sb.AppendLine("WITH latest AS (");
        sb.AppendLine("    SELECT ValidTimeUtc, Model,");
        sb.AppendLine("           Temperature2m, DewPoint2m, RelativeHumidity2m,");
        sb.AppendLine("           CloudCover, CloudCoverLow, CloudCoverMid, CloudCoverHigh,");
        sb.AppendLine("           WindSpeed10m, WindDirection10m, WindGusts10m, SurfacePressure,");
        sb.AppendLine("           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn");
        sb.AppendLine($"    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)");
        sb.AppendLine($"    WHERE LocationName = '{locationName}'");
        sb.AppendLine("      AND RunTimeSource = 'offset_day'");
        sb.AppendLine($"      AND LeadHours = {spec.LeadHours}");
        sb.AppendLine("      AND Temperature2m IS NOT NULL");
        sb.AppendLine($"      AND Model IN {modelInClause}");
        sb.AppendLine("),");
        sb.AppendLine("pivoted AS (");
        sb.AppendLine("    SELECT ValidTimeUtc,");
        // N temps
        for (int i = 0; i < n; i++)
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN Temperature2m END) AS temp_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        // 10 secondary vars × N models
        for (int s = 0; s < SecondaryVars.Count; s++)
        {
            for (int i = 0; i < n; i++)
                sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN {SecondaryVars[s].SrcColumn} END) AS sec_{s}_{i},");
        }
        sb.AppendLine("        AVG(WindDirection10m) AS wind_dir_mean");
        sb.AppendLine("    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc");
        sb.AppendLine("),");
        sb.AppendLine("era5 AS (");
        sb.AppendLine("    SELECT ValidTimeUtc, Temperature2m AS era5_temp");
        sb.AppendLine($"    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)");
        sb.AppendLine($"    WHERE LocationName = '{locationName}' AND Temperature2m IS NOT NULL");
        sb.AppendLine(")");
        sb.AppendLine("SELECT p.ValidTimeUtc,");
        for (int i = 0; i < n; i++) sb.Append($"    p.temp_{TempFeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        for (int s = 0; s < SecondaryVars.Count; s++)
            for (int i = 0; i < n; i++)
                sb.Append($"    p.sec_{s}_{i},");
        sb.AppendLine();
        sb.AppendLine("    p.wind_dir_mean, e.era5_temp");
        sb.AppendLine("FROM pivoted p JOIN era5 e USING (ValidTimeUtc)");
        // Post-pivot WHERE: every required NOT NULL AND at least one of all NOT NULL.
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.temp_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.temp_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";
        sb.AppendLine($"WHERE ({requiredNotNull})");
        sb.AppendLine($"  AND {anyNotNull}");
        sb.AppendLine("ORDER BY p.ValidTimeUtc;");

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sb.ToString();
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var temps = new double[n];
        var secs = new double[SecondaryVars.Count][];
        for (int s = 0; s < SecondaryVars.Count; s++) secs[s] = new double[n];

        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            int col = 1;
            for (int i = 0; i < n; i++) { temps[i] = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            for (int s = 0; s < SecondaryVars.Count; s++)
                for (int i = 0; i < n; i++) { secs[s][i] = r.IsDBNull(col) ? double.NaN : r.GetDouble(col); col++; }
            var windDirMean = r.IsDBNull(col) ? double.NaN : r.GetDouble(col);
            var era5Temp = r.GetDouble(col + 1);
            rows.Add(ComposeRow(spec, valid, temps,
                dewPoints: secs[0], rhs: secs[1], clouds: secs[2],
                cloudLows: secs[3], cloudMids: secs[4], cloudHighs: secs[5],
                windSpeeds: secs[6], windDirsDeg: secs[7], windGusts: secs[8], pressures: secs[9],
                windDirMeanDeg: windDirMean, era5Temp: era5Temp));
        }
        return rows;
    }

    /// <summary>
    /// Pack the rich temperature feature vector. Layout matches
    /// <see cref="BuildSpec"/>'s <c>FeatureNames</c>: N temps + 3 spread + 4 calendar
    /// + N×11 per-model secondaries (wind_dir as sin+cos) + 9 aggregates.
    /// </summary>
    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> temps,
        IReadOnlyList<double> dewPoints,
        IReadOnlyList<double> rhs,
        IReadOnlyList<double> clouds,
        IReadOnlyList<double> cloudLows,
        IReadOnlyList<double> cloudMids,
        IReadOnlyList<double> cloudHighs,
        IReadOnlyList<double> windSpeeds,
        IReadOnlyList<double> windDirsDeg,
        IReadOnlyList<double> windGusts,
        IReadOnlyList<double> pressures,
        double windDirMeanDeg,
        double era5Temp)
    {
        var n = spec.Models.Count;
        if (temps.Count != n) throw new ArgumentException($"Expected {n} temps", nameof(temps));
        if (dewPoints.Count != n) throw new ArgumentException($"Expected {n} dew points", nameof(dewPoints));

        var features = new float[spec.FeatureCount];

        // Lean block: N temps + 3 spread + 4 calendar.
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = temps[i];
            if (double.IsNaN(x)) continue;
            sum += x; sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            present++;
        }
        var meanT  = present == 0 ? double.NaN : sum / present;
        var var0   = present == 0 ? double.NaN : Math.Max(0.0, (sumSq / present) - (meanT * meanT));
        var stdT   = double.IsNaN(var0) ? double.NaN : Math.Sqrt(var0);
        var rangeT = present == 0 ? double.NaN : max - min;

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle  = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;

        for (int i = 0; i < n; i++) features[i] = (float)temps[i];
        features[n + 0] = (float)meanT;
        features[n + 1] = (float)stdT;
        features[n + 2] = (float)rangeT;
        features[n + 3] = (float)Math.Sin(hourAngle);
        features[n + 4] = (float)Math.Cos(hourAngle);
        features[n + 5] = (float)Math.Sin(doyAngle);
        features[n + 6] = (float)Math.Cos(doyAngle);

        // Per-model secondaries (var-major × model, in SecondaryVars order;
        // wind_dir expanded to sin+cos blocks).
        int idx = n + 7;
        var secLists = new IReadOnlyList<double>[]
        {
            dewPoints, rhs, clouds, cloudLows, cloudMids, cloudHighs,
            windSpeeds, /* wind_dir handled separately */ null!, windGusts, pressures,
        };
        for (int s = 0; s < SecondaryVars.Count; s++)
        {
            if (SecondaryVars[s].FeaturePrefix == "wind_dir")
            {
                for (int i = 0; i < n; i++)
                {
                    var rad = windDirsDeg[i] * Math.PI / 180.0;
                    features[idx++] = (float)(double.IsNaN(windDirsDeg[i]) ? double.NaN : Math.Sin(rad));
                }
                for (int i = 0; i < n; i++)
                {
                    var rad = windDirsDeg[i] * Math.PI / 180.0;
                    features[idx++] = (float)(double.IsNaN(windDirsDeg[i]) ? double.NaN : Math.Cos(rad));
                }
            }
            else
            {
                var src = secLists[s];
                if (src.Count != n) throw new ArgumentException($"Expected {n} values for {SecondaryVars[s].FeaturePrefix}");
                for (int i = 0; i < n; i++) features[idx++] = (float)src[i];
            }
        }

        // 9 cross-model aggregates (NaN-skipping).
        features[idx++] = (float)NanMean(dewPoints);
        features[idx++] = (float)NanStd(dewPoints);
        features[idx++] = (float)NanMean(rhs);
        features[idx++] = (float)NanStd(rhs);
        features[idx++] = (float)NanMean(clouds);
        features[idx++] = (float)NanMean(windSpeeds);
        features[idx++] = (float)NanStd(windSpeeds);
        features[idx++] = (float)NanMean(pressures);
        features[idx++] = (float)NanStd(pressures);

        if (idx != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Rich temp feature pack mismatch: wrote {idx}, expected {spec.FeatureCount}");

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Temp,
            WindDirMean = (float)windDirMeanDeg,
        };
    }

    private static double NanMean(IReadOnlyList<double> values)
    {
        double sum = 0; int n = 0;
        for (int i = 0; i < values.Count; i++)
            if (!double.IsNaN(values[i])) { sum += values[i]; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double NanStd(IReadOnlyList<double> values)
    {
        var mean = NanMean(values);
        if (double.IsNaN(mean)) return double.NaN;
        double sumSq = 0; int n = 0;
        for (int i = 0; i < values.Count; i++)
            if (!double.IsNaN(values[i])) { sumSq += (values[i] - mean) * (values[i] - mean); n++; }
        return n == 0 ? double.NaN : Math.Sqrt(sumSq / n);
    }

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");
}
