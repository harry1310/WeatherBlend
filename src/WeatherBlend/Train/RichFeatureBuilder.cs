using System.Text;
using DuckDB.NET.Data;

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
}
