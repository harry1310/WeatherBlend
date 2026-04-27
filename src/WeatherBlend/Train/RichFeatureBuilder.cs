using System.Text;
using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// Builds the 88-feature Phase 2c training dataset. Same chronological pipeline as
/// <see cref="FeatureBuilder"/> — same SQL filters, same join, same row-drop policy
/// based on per-model temperature presence — but pulls 11 secondary variables per
/// model (dew, rh, cloud {total/low/mid/high}, wind speed/dir/gusts, pressure) and
/// computes a small bank of cross-model aggregates.
///
/// Kept parallel to <see cref="FeatureBuilder"/> rather than extending it: the SQL
/// shape, the per-row composition logic, and the FeatureNames list are all separate
/// concerns and there's no value in fusing them. Phase 2b stays bit-for-bit unchanged.
///
/// Missing-value policy: secondaries can be NULL upstream (a model may not expose a
/// variable, or Open-Meteo may have dropped it). Missing → <c>float.NaN</c>; LightGBM
/// handles NaN as missing natively. Aggregates skip NaN inputs; if every model is
/// missing a variable, the aggregate is also NaN.
/// </summary>
public static class RichFeatureBuilder
{
    /// <summary>Per-model variables we pull from forecast parquet, in stable order.
    /// Each (var, model) pair becomes one feature column suffixed with the model id.</summary>
    public static readonly IReadOnlyList<(string SrcColumn, string FeaturePrefix)> Secondaries = new[]
    {
        ("DewPoint2m",         "dew"),
        ("RelativeHumidity2m", "rh"),
        ("CloudCover",         "cloud"),
        ("CloudCoverLow",      "cloud_low"),
        ("CloudCoverMid",      "cloud_mid"),
        ("CloudCoverHigh",     "cloud_high"),
        ("WindSpeed10m",       "wind_speed"),
        ("WindDirection10m",   "wind_dir"),     // expanded to sin/cos in ComposeRow
        ("WindGusts10m",       "wind_gusts"),
        ("SurfacePressure",    "pressure"),
    };

    // Suffix per model id, matching FeatureBuilder.ModelColumns ordering.
    private static readonly string[] ModelSuffixes = { "gfs", "ecmwf", "icon", "mf", "ukmo", "gem" };

    /// <summary>
    /// Feature column order — load-bearing. Persisted in feature_schema.json.
    /// Layout: 13 lean (matches FeatureBuilder.FeatureNames) → 66 per-model secondaries
    /// (var-major, then model: dew_gfs..dew_gem, rh_gfs..rh_gem, …, wind_dir_sin_gfs..,
    /// wind_dir_cos_gfs..) → 9 aggregates.
    /// </summary>
    public static readonly IReadOnlyList<string> FeatureNames = BuildFeatureNames();

    private static List<string> BuildFeatureNames()
    {
        var names = new List<string>();
        // 13 lean features — ordering matches FeatureBuilder.FeatureNames exactly.
        names.AddRange(FeatureBuilder.FeatureNames);

        // 66 per-model secondaries. Wind direction expands to sin + cos so each is its
        // own column; everything else is one column per (var, model).
        foreach (var (_, prefix) in Secondaries)
        {
            if (prefix == "wind_dir")
            {
                foreach (var m in ModelSuffixes) names.Add($"wind_dir_sin_{m}");
                foreach (var m in ModelSuffixes) names.Add($"wind_dir_cos_{m}");
            }
            else
            {
                foreach (var m in ModelSuffixes) names.Add($"{prefix}_{m}");
            }
        }

        // 9 aggregates.
        names.AddRange(new[]
        {
            "dew_mean", "dew_std",
            "rh_mean", "rh_std",
            "cloud_mean",
            "wind_speed_mean", "wind_speed_std",
            "pressure_mean", "pressure_std",
        });
        return names;
    }

    public static List<RichTrainingRow> BuildForLead(
        string forecastsPath,
        string era5Path,
        string locationName,
        int leadHours,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = NormaliseGlob(Path.Combine(era5Path, "**", "*.parquet"));

        var fcWhere = $"LocationName = '{locationName}' AND RunTimeSource = 'offset_day' AND LeadHours = {leadHours} AND Temperature2m IS NOT NULL";

        var sql = BuildPivotSql(fcGlob, eraGlob, locationName, fcWhere);

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RichTrainingRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            rows.Add(ReadRow(r));
        }
        return rows;
    }

    /// <summary>
    /// Compose a RichTrainingRow from raw per-model values + ERA5 truth. Missing
    /// secondaries should be passed as NaN. Factored out so unit tests can hit the
    /// composition logic without DuckDB.
    /// </summary>
    public static RichTrainingRow ComposeRow(
        DateTime validTimeUtc,
        double[] temps,            // 6, all required
        double[] dewPoints,        // 6, NaN allowed
        double[] rhs,
        double[] clouds,
        double[] cloudLows,
        double[] cloudMids,
        double[] cloudHighs,
        double[] windSpeeds,
        double[] windDirsDeg,
        double[] windGusts,
        double[] pressures,
        double era5Temp)
    {
        if (temps.Length != 6) throw new ArgumentException("Expected 6 temps", nameof(temps));

        // Reuse the lean ComposeRow to populate the 13 base features + Label + WindDirMean.
        var leanRow = FeatureBuilder.ComposeRow(validTimeUtc, temps,
            windDirMeanDeg: NanIgnoringMean(windDirsDeg),
            era5Temp: era5Temp);

        var rich = new RichTrainingRow
        {
            ValidTimeUtc = leanRow.ValidTimeUtc,
            TempGfs   = leanRow.TempGfs,
            TempEcmwf = leanRow.TempEcmwf,
            TempIcon  = leanRow.TempIcon,
            TempMf    = leanRow.TempMf,
            TempUkmo  = leanRow.TempUkmo,
            TempGem   = leanRow.TempGem,
            TempMean  = leanRow.TempMean,
            TempStd   = leanRow.TempStd,
            TempRange = leanRow.TempRange,
            HourSin   = leanRow.HourSin,
            HourCos   = leanRow.HourCos,
            DoySin    = leanRow.DoySin,
            DoyCos    = leanRow.DoyCos,
            WindDirMean = leanRow.WindDirMean,
            Era5Temp  = leanRow.Era5Temp,
        };

        Assign(rich, dewPoints,
            (r, v) => r.DewGfs   = v, (r, v) => r.DewEcmwf = v, (r, v) => r.DewIcon = v,
            (r, v) => r.DewMf    = v, (r, v) => r.DewUkmo  = v, (r, v) => r.DewGem  = v);
        Assign(rich, rhs,
            (r, v) => r.RhGfs   = v, (r, v) => r.RhEcmwf = v, (r, v) => r.RhIcon = v,
            (r, v) => r.RhMf    = v, (r, v) => r.RhUkmo  = v, (r, v) => r.RhGem  = v);
        Assign(rich, clouds,
            (r, v) => r.CloudGfs   = v, (r, v) => r.CloudEcmwf = v, (r, v) => r.CloudIcon = v,
            (r, v) => r.CloudMf    = v, (r, v) => r.CloudUkmo  = v, (r, v) => r.CloudGem  = v);
        Assign(rich, cloudLows,
            (r, v) => r.CloudLowGfs   = v, (r, v) => r.CloudLowEcmwf = v, (r, v) => r.CloudLowIcon = v,
            (r, v) => r.CloudLowMf    = v, (r, v) => r.CloudLowUkmo  = v, (r, v) => r.CloudLowGem  = v);
        Assign(rich, cloudMids,
            (r, v) => r.CloudMidGfs   = v, (r, v) => r.CloudMidEcmwf = v, (r, v) => r.CloudMidIcon = v,
            (r, v) => r.CloudMidMf    = v, (r, v) => r.CloudMidUkmo  = v, (r, v) => r.CloudMidGem  = v);
        Assign(rich, cloudHighs,
            (r, v) => r.CloudHighGfs   = v, (r, v) => r.CloudHighEcmwf = v, (r, v) => r.CloudHighIcon = v,
            (r, v) => r.CloudHighMf    = v, (r, v) => r.CloudHighUkmo  = v, (r, v) => r.CloudHighGem  = v);
        Assign(rich, windSpeeds,
            (r, v) => r.WindSpeedGfs   = v, (r, v) => r.WindSpeedEcmwf = v, (r, v) => r.WindSpeedIcon = v,
            (r, v) => r.WindSpeedMf    = v, (r, v) => r.WindSpeedUkmo  = v, (r, v) => r.WindSpeedGem  = v);
        Assign(rich, windGusts,
            (r, v) => r.WindGustsGfs   = v, (r, v) => r.WindGustsEcmwf = v, (r, v) => r.WindGustsIcon = v,
            (r, v) => r.WindGustsMf    = v, (r, v) => r.WindGustsUkmo  = v, (r, v) => r.WindGustsGem  = v);
        Assign(rich, pressures,
            (r, v) => r.PressureGfs   = v, (r, v) => r.PressureEcmwf = v, (r, v) => r.PressureIcon = v,
            (r, v) => r.PressureMf    = v, (r, v) => r.PressureUkmo  = v, (r, v) => r.PressureGem  = v);

        // Wind direction → sin/cos per model. NaN propagates: sin(NaN)=cos(NaN)=NaN.
        var sins = new double[6];
        var coss = new double[6];
        for (int i = 0; i < 6; i++)
        {
            var rad = windDirsDeg[i] * Math.PI / 180.0;
            sins[i] = double.IsNaN(windDirsDeg[i]) ? double.NaN : Math.Sin(rad);
            coss[i] = double.IsNaN(windDirsDeg[i]) ? double.NaN : Math.Cos(rad);
        }
        Assign(rich, sins,
            (r, v) => r.WindDirSinGfs   = v, (r, v) => r.WindDirSinEcmwf = v, (r, v) => r.WindDirSinIcon = v,
            (r, v) => r.WindDirSinMf    = v, (r, v) => r.WindDirSinUkmo  = v, (r, v) => r.WindDirSinGem  = v);
        Assign(rich, coss,
            (r, v) => r.WindDirCosGfs   = v, (r, v) => r.WindDirCosEcmwf = v, (r, v) => r.WindDirCosIcon = v,
            (r, v) => r.WindDirCosMf    = v, (r, v) => r.WindDirCosUkmo  = v, (r, v) => r.WindDirCosGem  = v);

        // Cross-model aggregates over the non-NaN subset.
        rich.DewMean         = (float)NanIgnoringMean(dewPoints);
        rich.DewStd          = (float)NanIgnoringStd(dewPoints);
        rich.RhMean          = (float)NanIgnoringMean(rhs);
        rich.RhStd           = (float)NanIgnoringStd(rhs);
        rich.CloudMean       = (float)NanIgnoringMean(clouds);
        rich.WindSpeedMean   = (float)NanIgnoringMean(windSpeeds);
        rich.WindSpeedStd    = (float)NanIgnoringStd(windSpeeds);
        rich.PressureMean    = (float)NanIgnoringMean(pressures);
        rich.PressureStd     = (float)NanIgnoringStd(pressures);

        return rich;
    }

    private static void Assign(
        RichTrainingRow row,
        double[] values,
        Action<RichTrainingRow, float> setGfs,
        Action<RichTrainingRow, float> setEcmwf,
        Action<RichTrainingRow, float> setIcon,
        Action<RichTrainingRow, float> setMf,
        Action<RichTrainingRow, float> setUkmo,
        Action<RichTrainingRow, float> setGem)
    {
        if (values.Length != 6) throw new ArgumentException("Expected 6 per-model values", nameof(values));
        setGfs(row, (float)values[0]);
        setEcmwf(row, (float)values[1]);
        setIcon(row, (float)values[2]);
        setMf(row, (float)values[3]);
        setUkmo(row, (float)values[4]);
        setGem(row, (float)values[5]);
    }

    private static double NanIgnoringMean(double[] values)
    {
        double sum = 0; int n = 0;
        foreach (var v in values)
            if (!double.IsNaN(v)) { sum += v; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double NanIgnoringStd(double[] values)
    {
        var mean = NanIgnoringMean(values);
        if (double.IsNaN(mean)) return double.NaN;
        double sumSq = 0; int n = 0;
        foreach (var v in values)
            if (!double.IsNaN(v)) { sumSq += (v - mean) * (v - mean); n++; }
        return n == 0 ? double.NaN : Math.Sqrt(sumSq / n);
    }

    private static RichTrainingRow ReadRow(DuckDB.NET.Data.DuckDBDataReader r)
    {
        var valid = r.GetDateTime(0);
        var temps = new double[6];
        for (int i = 0; i < 6; i++) temps[i] = r.GetDouble(1 + i);

        // Secondaries: 10 vars × 6 models = 60 columns starting at index 7.
        // Order matches Secondaries × ModelSuffixes (var-major, then model).
        const int secStart = 7;
        var secs = new double[Secondaries.Count][];
        int col = secStart;
        for (int s = 0; s < Secondaries.Count; s++)
        {
            secs[s] = new double[6];
            for (int m = 0; m < 6; m++)
            {
                secs[s][m] = r.IsDBNull(col) ? double.NaN : r.GetDouble(col);
                col++;
            }
        }

        var windDirCol = col;        // ensemble-mean wind dir (diagnostic)
        var era5Col    = col + 1;
        var windDirMean = r.IsDBNull(windDirCol) ? double.NaN : r.GetDouble(windDirCol);
        var era5Temp = r.GetDouble(era5Col);

        // Index lookup matches the SECONDARIES order above.
        // dew=0 rh=1 cloud=2 cloud_low=3 cloud_mid=4 cloud_high=5
        // wind_speed=6 wind_dir=7 wind_gusts=8 pressure=9
        return ComposeRow(
            validTimeUtc: valid,
            temps:        temps,
            dewPoints:    secs[0],
            rhs:          secs[1],
            clouds:       secs[2],
            cloudLows:    secs[3],
            cloudMids:    secs[4],
            cloudHighs:   secs[5],
            windSpeeds:   secs[6],
            windDirsDeg:  secs[7],
            windGusts:    secs[8],
            pressures:    secs[9],
            era5Temp:     era5Temp);
        // windDirMean is recomputed inside ComposeRow from the per-model dirs;
        // the SQL column is kept for parity with FeatureBuilder but unused.
    }

    private static string BuildPivotSql(string fcGlob, string eraGlob, string locationName, string fcWhere)
    {
        // Build the long pivot column list programmatically — 6 temps + 60 secondaries
        // + 1 wind-dir mean + ERA5 join.
        var sb = new StringBuilder();
        sb.AppendLine("WITH latest AS (");
        sb.AppendLine("    SELECT ValidTimeUtc, Model,");
        sb.AppendLine("           Temperature2m, DewPoint2m, RelativeHumidity2m,");
        sb.AppendLine("           CloudCover, CloudCoverLow, CloudCoverMid, CloudCoverHigh,");
        sb.AppendLine("           WindSpeed10m, WindDirection10m, WindGusts10m, SurfacePressure,");
        sb.AppendLine("           ROW_NUMBER() OVER (");
        sb.AppendLine("               PARTITION BY ValidTimeUtc, Model");
        sb.AppendLine("               ORDER BY RunTimeUtc DESC");
        sb.AppendLine("           ) AS rn");
        sb.AppendLine($"    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)");
        sb.AppendLine($"    WHERE {fcWhere}");
        sb.AppendLine("),");
        sb.AppendLine("pivoted AS (");
        sb.AppendLine("    SELECT");
        sb.AppendLine("        ValidTimeUtc,");
        // 6 temperature columns (mandatory, used for the not-null filter at the bottom).
        for (int m = 0; m < 6; m++)
        {
            var (modelId, _) = FeatureBuilder.ModelColumns[m];
            sb.AppendLine($"        MAX(CASE WHEN Model = '{modelId}' THEN Temperature2m END) AS temp_{ModelSuffixes[m]},");
        }
        // 10 secondary vars × 6 models.
        for (int s = 0; s < Secondaries.Count; s++)
        {
            var (src, _) = Secondaries[s];
            for (int m = 0; m < 6; m++)
            {
                var (modelId, _) = FeatureBuilder.ModelColumns[m];
                sb.AppendLine($"        MAX(CASE WHEN Model = '{modelId}' THEN {src} END) AS sec_{s}_{m},");
            }
        }
        sb.AppendLine("        AVG(WindDirection10m) AS wind_dir_mean");
        sb.AppendLine("    FROM latest");
        sb.AppendLine("    WHERE rn = 1");
        sb.AppendLine("    GROUP BY ValidTimeUtc");
        sb.AppendLine("),");
        sb.AppendLine("era5 AS (");
        sb.AppendLine("    SELECT ValidTimeUtc, Temperature2m AS era5_temp");
        sb.AppendLine($"    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)");
        sb.AppendLine($"    WHERE LocationName = '{locationName}'");
        sb.AppendLine("      AND Temperature2m IS NOT NULL");
        sb.AppendLine(")");
        sb.AppendLine("SELECT");
        sb.Append("    p.ValidTimeUtc, ");
        sb.AppendLine("p.temp_gfs, p.temp_ecmwf, p.temp_icon, p.temp_mf, p.temp_ukmo, p.temp_gem,");
        for (int s = 0; s < Secondaries.Count; s++)
        {
            sb.Append("    ");
            for (int m = 0; m < 6; m++)
            {
                sb.Append($"p.sec_{s}_{m}");
                sb.Append(',');
                if (m < 5) sb.Append(' ');
            }
            sb.AppendLine();
        }
        sb.AppendLine("    p.wind_dir_mean,");
        sb.AppendLine("    e.era5_temp");
        sb.AppendLine("FROM pivoted p");
        sb.AppendLine("JOIN era5 e USING (ValidTimeUtc)");
        // Same row-drop policy as 2b: every model must have a temperature.
        sb.AppendLine("WHERE p.temp_gfs   IS NOT NULL");
        sb.AppendLine("  AND p.temp_ecmwf IS NOT NULL");
        sb.AppendLine("  AND p.temp_icon  IS NOT NULL");
        sb.AppendLine("  AND p.temp_mf    IS NOT NULL");
        sb.AppendLine("  AND p.temp_ukmo  IS NOT NULL");
        sb.AppendLine("  AND p.temp_gem   IS NOT NULL");
        sb.AppendLine("ORDER BY p.ValidTimeUtc;");
        return sb.ToString();
    }

    private static string NormaliseGlob(string path)
        => path.Replace('\\', '/').Replace("'", "''");

    // -----------------------------------------------------------------------
    // New canonical API (Phase 3 of unify-model-membership refactor).
    // Spec-driven, dynamic-shape vector via RegressionTrainingRow.Features.
    // -----------------------------------------------------------------------

    public const string SpecTarget = "temperature";
    public const string SpecFeatureSet = "rich";

    /// <summary>
    /// Per-model secondary variables for the rich temperature blender, in stable order.
    /// <c>WindDirection10m</c> is special — it expands to sin+cos (2 columns per model).
    /// All other vars are one column per (var, model).
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

        var orderedRequired = FeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = FeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = FeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h.");

        // Layout (N = orderedModels.Count):
        //   N per-model temps, 3 spread, 4 calendar  (= N+7, the lean block)
        //   per-model secondaries (var-major × model): each non-wind-dir var = N cols,
        //     wind_dir = 2N cols (sin+cos)
        //   9 aggregates
        var names = new List<string>();
        foreach (var m in orderedModels) names.Add($"temp_{FeatureBuilder.ShortName(m)}");
        names.AddRange(new[] { "temp_mean", "temp_std", "temp_range" });
        names.AddRange(new[] { "hour_sin", "hour_cos", "doy_sin", "doy_cos" });

        foreach (var (_, prefix) in SecondaryVars)
        {
            if (prefix == "wind_dir")
            {
                foreach (var m in orderedModels) names.Add($"wind_dir_sin_{FeatureBuilder.ShortName(m)}");
                foreach (var m in orderedModels) names.Add($"wind_dir_cos_{FeatureBuilder.ShortName(m)}");
            }
            else
            {
                foreach (var m in orderedModels) names.Add($"{prefix}_{FeatureBuilder.ShortName(m)}");
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
            sb.AppendLine($"        MAX(CASE WHEN Model = '{spec.Models[i]}' THEN Temperature2m END) AS temp_{FeatureBuilder.ShortName(spec.Models[i])},");
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
        for (int i = 0; i < n; i++) sb.Append($"    p.temp_{FeatureBuilder.ShortName(spec.Models[i])},");
        sb.AppendLine();
        for (int s = 0; s < SecondaryVars.Count; s++)
            for (int i = 0; i < n; i++)
                sb.Append($"    p.sec_{s}_{i},");
        sb.AppendLine();
        sb.AppendLine("    p.wind_dir_mean, e.era5_temp");
        sb.AppendLine("FROM pivoted p JOIN era5 e USING (ValidTimeUtc)");
        // Post-pivot WHERE: every required NOT NULL AND at least one of all NOT NULL.
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.temp_{FeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyNotNull = "(" + string.Join(" OR ", spec.Models.Select(m => $"p.temp_{FeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";
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

        // First N+7 features: per-model temps + spread + calendar — same as lean.
        // Inlined here (rather than delegated to FeatureBuilder.ComposeRow) because the
        // lean composer asserts it filled the WHOLE vector — feeding it a rich spec would
        // trip the assertion. Same NaN-safe spread logic, packed directly.
        var features = new float[spec.FeatureCount];

        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int present = 0;
        for (int i = 0; i < n; i++)
        {
            var x = temps[i];
            if (double.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
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
}
