using System.Text;
using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Oro;

/// <summary>
/// Phase 3c-oro feature builder: rich (55 / 59) feature precip vector with an
/// 8-feature terrain block appended, for the pooled-multi-station bake-off
/// (docs/OROGRAPHIC_BONEHILL_3C_PLAN.md).
///
/// Wraps <see cref="PrecipRichFeatureBuilder.BuildForLead"/> to get the rich
/// per-row vector unchanged, then re-runs a small auxiliary SQL pass to pull
/// NWP-mean wind + temperature + dewpoint + pressure per valid_time. From
/// those it computes the 8 dynamic + static terrain features and appends
/// them to each row's feature vector.
///
/// Two SQL passes (rich pivot + this aux) rather than one fused query because
/// the rich builder's pivot is already complex and the cost of an extra pass
/// over per-station forecasts (~10k rows × ~10 cols × ~7 NWPs) is in the
/// hundreds of ms — negligible against training time, and keeps the rich
/// builder's contract bit-for-bit.
/// </summary>
public static class PrecipRichOroFeatureBuilder
{
    public const string SpecFeatureSet = "rich-oro";

    /// <summary>Names of the 9 terrain features, appended after the rich vector.
    /// The trailing <c>oro_station_id</c> is an integer 0..N-1 (ordered by
    /// caller convention) that lets LightGBM learn per-station offsets
    /// directly, so the other terrain features are evaluated on what they
    /// uniquely add beyond station identity.</summary>
    public static readonly string[] TerrainFeatureNames =
    {
        "oro_elevation_vs_cell_m",
        "oro_relief_5km_m",
        "oro_ruggedness_5km_m",
        "oro_wind_sin",                   // mean of per-model sin(wind_dir_from)
        "oro_wind_cos",                   // mean of per-model cos(wind_dir_from)
        "oro_upwind_gain_per_wind_5km_m", // 8-sector lookup at NWP-mean wind dir
        "oro_uplift_m_per_s",             // max(0, u·dx + v·dy)
        "oro_uplift_x_q_g_per_kg",        // uplift × specific humidity (g/kg)
        "oro_station_id",                 // integer 0..N-1; coarse cell/station axis
    };

    public const int TerrainFeatureCount = 9;

    /// <summary>Resolve the runtime <see cref="BlenderSpec"/> for rich-oro precipitation at a given lead.</summary>
    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        // Reuse rich's model membership + names — same precipitation blender
        // membership (the terrain block doesn't change which NWPs the
        // builder ingests).
        var rich = PrecipRichFeatureBuilder.BuildSpec(blendersCfg, leadHours);
        var names = rich.FeatureNames.Concat(TerrainFeatureNames).ToList();
        return new BlenderSpec
        {
            Target = rich.Target,
            FeatureSet = SpecFeatureSet,
            LeadHours = rich.LeadHours,
            RequiredModels = rich.RequiredModels,
            OptionalModels = rich.OptionalModels,
            Models = rich.Models,
            FeatureNames = names,
            DataSource = rich.DataSource,
            Tier = SpecFeatureSet,
            UkvStrategy = rich.UkvStrategy,
        };
    }

    /// <summary>
    /// Build rich-oro training rows for one (station, lead). Per-row feature
    /// layout = rich vector (unchanged) || 8 terrain features.
    /// </summary>
    public static List<BinaryTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        OroStaticFeatures oro,
        int stationIndex,
        BlenderSpec richOroSpec,
        CancellationToken ct = default)
    {
        // The rich spec is what the underlying builder consumes. Reuse the
        // rich-oro spec's model membership to materialise a matching rich
        // spec — bit-identical to what PrecipRichFeatureBuilder.BuildSpec
        // would have produced for these models at this lead.
        var richSpec = new BlenderSpec
        {
            Target = richOroSpec.Target,
            FeatureSet = PrecipRichFeatureBuilder.SpecFeatureSet,
            LeadHours = richOroSpec.LeadHours,
            RequiredModels = richOroSpec.RequiredModels,
            OptionalModels = richOroSpec.OptionalModels,
            Models = richOroSpec.Models,
            FeatureNames = richOroSpec.FeatureNames
                .Take(richOroSpec.FeatureNames.Count - TerrainFeatureCount).ToList(),
            DataSource = richOroSpec.DataSource,
            Tier = PrecipRichFeatureBuilder.SpecFeatureSet,
            UkvStrategy = richOroSpec.UkvStrategy,
        };

        var richRows = PrecipRichFeatureBuilder.BuildForLead(
            forecastsPath, rainfallPath, locationName, stationName, richSpec, ct);
        if (richRows.Count == 0) return richRows;

        var aux = LoadAuxNwpMeans(forecastsPath, locationName, richSpec, ct);

        var richDim = richSpec.FeatureCount;
        var outDim  = richOroSpec.FeatureCount;
        if (outDim != richDim + TerrainFeatureCount)
            throw new InvalidOperationException(
                $"rich-oro spec dim mismatch: {outDim} != {richDim} + {TerrainFeatureCount}");

        var rows = new List<BinaryTrainingRow>(richRows.Count);
        foreach (var rr in richRows)
        {
            ct.ThrowIfCancellationRequested();
            if (!aux.TryGetValue(rr.ValidTimeUtc, out var a))
                continue; // No NWP-mean aux row — drop, matches rich's row filter style.

            var terrain = ComposeTerrainBlock(oro, a, stationIndex);

            var features = new float[outDim];
            Array.Copy(rr.Features, features, richDim);
            for (int i = 0; i < TerrainFeatureCount; i++)
                features[richDim + i] = terrain[i];

            rows.Add(new BinaryTrainingRow
            {
                ValidTimeUtc = rr.ValidTimeUtc,
                Features = features,
                Label = rr.Label,
                TruthMmHour = rr.TruthMmHour,
            });
        }
        return rows;
    }

    // -----------------------------------------------------------------------
    // Terrain block composition (8 features)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Pack the 9-feature terrain block for one row.
    ///   0: oro_elevation_vs_cell_m   (static, per-site)
    ///   1: oro_relief_5km_m          (static, per-site)
    ///   2: oro_ruggedness_5km_m      (static, per-site)
    ///   3: oro_wind_sin              (NWP-mean of sin(wind_dir_from))
    ///   4: oro_wind_cos              (NWP-mean of cos(wind_dir_from))
    ///   5: oro_upwind_gain_per_wind_5km_m  (8-sector lookup at NWP-mean dir)
    ///   6: oro_uplift_m_per_s        (max(0, u·dx + v·dy), u/v from NWP-mean)
    ///   7: oro_uplift_x_q_g_per_kg   (uplift × specific humidity, g/kg)
    ///   8: oro_station_id            (integer 0..N-1, ordered by caller)
    /// </summary>
    public static float[] ComposeTerrainBlock(OroStaticFeatures oro, NwpMeanRow a, int stationIndex)
    {
        // Mean wind direction "from" in radians (atan2 handles wrap correctly).
        var windDirRad = (double.IsNaN(a.WindSin) || double.IsNaN(a.WindCos))
            ? double.NaN
            : Math.Atan2(a.WindSin, a.WindCos);
        var upwindGain = oro.UpwindGainAt(windDirRad);

        // Wind vector (u east-component, v north-component). Convention:
        // WindDirection10m = direction wind is FROM. So a wind "from west" (270°)
        // blows toward east: u_east = +ws, v_north = 0.
        //   u_east  = -ws · sin(dir_from)
        //   v_north = -ws · cos(dir_from)
        double uplift;
        if (double.IsNaN(a.WindSpeed) || double.IsNaN(a.WindSin) || double.IsNaN(a.WindCos))
        {
            uplift = 0.0;
        }
        else
        {
            var uEast  = -a.WindSpeed * a.WindSin;
            var vNorth = -a.WindSpeed * a.WindCos;
            var w = uEast * oro.TerrainGradientDx + vNorth * oro.TerrainGradientDy;
            uplift = Math.Max(0.0, w);
        }

        // Specific humidity q (g/kg) from Magnus saturation vapor pressure at
        // dewpoint, divided by surface pressure. NaN-safe: any missing input
        // collapses q to 0 (no humidity contribution).
        double q;
        if (double.IsNaN(a.DewC) || double.IsNaN(a.PresHpa) || a.PresHpa <= 0)
        {
            q = 0.0;
        }
        else
        {
            var eHpa = 6.112 * Math.Exp(17.62 * a.DewC / (a.DewC + 243.12));
            var qKgKg = 0.622 * eHpa / (a.PresHpa - 0.378 * eHpa);
            q = Math.Max(0.0, qKgKg * 1000.0); // g/kg
        }

        var block = new float[TerrainFeatureCount];
        block[0] = (float)oro.ElevationVsCellM;
        block[1] = (float)oro.Relief5kmM;
        block[2] = (float)oro.TerrainRuggedness5kmM;
        block[3] = (float)(double.IsNaN(a.WindSin) ? 0.0 : a.WindSin);
        block[4] = (float)(double.IsNaN(a.WindCos) ? 0.0 : a.WindCos);
        block[5] = (float)upwindGain;
        block[6] = (float)uplift;
        block[7] = (float)(uplift * q);
        block[8] = (float)stationIndex;
        return block;
    }

    /// <summary>NWP-ensemble-mean covariates per valid_time, used to compose
    /// the dynamic terrain features.</summary>
    public sealed record NwpMeanRow(
        double WindSin,
        double WindCos,
        double WindSpeed,
        double TempC,
        double DewC,
        double PresHpa);

    // -----------------------------------------------------------------------
    // Aux SQL pass
    // -----------------------------------------------------------------------

    private static Dictionary<DateTime, NwpMeanRow> LoadAuxNwpMeans(
        string forecastsPath, string locationName, BlenderSpec spec, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";

        var sb = new StringBuilder();
        sb.AppendLine("WITH latest AS (");
        sb.AppendLine("    SELECT ValidTimeUtc, Model,");
        sb.AppendLine("           WindDirection10m, WindSpeed10m, Temperature2m, DewPoint2m, SurfacePressure,");
        sb.AppendLine("           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn");
        sb.AppendLine($"    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)");
        sb.AppendLine($"    WHERE LocationName = '{escLocation}'");
        sb.AppendLine("      AND RunTimeSource = 'offset_day'");
        sb.AppendLine($"      AND LeadHours = {spec.LeadHours}");
        sb.AppendLine($"      AND Model IN {modelInClause}");
        sb.AppendLine(")");
        sb.AppendLine("SELECT ValidTimeUtc,");
        sb.AppendLine("       AVG(SIN(WindDirection10m * pi() / 180.0)) AS wind_sin,");
        sb.AppendLine("       AVG(COS(WindDirection10m * pi() / 180.0)) AS wind_cos,");
        sb.AppendLine("       AVG(WindSpeed10m)    AS wind_speed,");
        sb.AppendLine("       AVG(Temperature2m)   AS temp_c,");
        sb.AppendLine("       AVG(DewPoint2m)      AS dew_c,");
        sb.AppendLine("       AVG(SurfacePressure) AS pres_hpa");
        sb.AppendLine("FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc ORDER BY ValidTimeUtc;");

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sb.ToString();
        using var r = cmd.ExecuteReader();

        var map = new Dictionary<DateTime, NwpMeanRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var t = r.GetDateTime(0);
            var windSin = r.IsDBNull(1) ? double.NaN : r.GetDouble(1);
            var windCos = r.IsDBNull(2) ? double.NaN : r.GetDouble(2);
            var ws      = r.IsDBNull(3) ? double.NaN : r.GetDouble(3);
            var tc      = r.IsDBNull(4) ? double.NaN : r.GetDouble(4);
            var td      = r.IsDBNull(5) ? double.NaN : r.GetDouble(5);
            var p       = r.IsDBNull(6) ? double.NaN : r.GetDouble(6);
            map[t] = new NwpMeanRow(windSin, windCos, ws, tc, td, p);
        }
        return map;
    }
}
