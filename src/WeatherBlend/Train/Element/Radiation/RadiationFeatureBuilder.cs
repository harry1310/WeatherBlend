using DuckDB.NET.Data;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Radiation;

/// <summary>
/// SQL pivot + row composition for the shortwave-radiation blender.
///
/// Required (must be non-null per row): GFS, ECMWF, ICON, MF, GEM shortwave radiation.
/// Optional: UKMO — present at lead 24h ~74% of the time, always-null at lead 48/72h.
/// We keep UKMO columns and let them be NULL → NaN. LightGBM handles missing values
/// natively, so the blender silently down-weights UKMO at long lead.
///
/// Direct + diffuse follow the same required/optional split as their parent SW value
/// per model — if a model returns SW it returns the triplet.
/// </summary>
public static class RadiationFeatureBuilder
{
    public static readonly IReadOnlyList<string> PublicFeatureNames = new[]
    {
        "sw_gfs", "sw_ecmwf", "sw_icon", "sw_mf", "sw_ukmo", "sw_gem",
        "direct_gfs", "direct_ecmwf", "direct_icon", "direct_mf", "direct_ukmo", "direct_gem",
        "diffuse_gfs", "diffuse_ecmwf", "diffuse_icon", "diffuse_mf", "diffuse_ukmo", "diffuse_gem",
        "sw_mean", "sw_std", "sw_range",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    public static readonly IReadOnlyList<string> InternalFeatureColumns = new[]
    {
        nameof(RadiationRow.SwGfs), nameof(RadiationRow.SwEcmwf), nameof(RadiationRow.SwIcon),
        nameof(RadiationRow.SwMf), nameof(RadiationRow.SwUkmo), nameof(RadiationRow.SwGem),
        nameof(RadiationRow.DirectGfs), nameof(RadiationRow.DirectEcmwf), nameof(RadiationRow.DirectIcon),
        nameof(RadiationRow.DirectMf), nameof(RadiationRow.DirectUkmo), nameof(RadiationRow.DirectGem),
        nameof(RadiationRow.DiffuseGfs), nameof(RadiationRow.DiffuseEcmwf), nameof(RadiationRow.DiffuseIcon),
        nameof(RadiationRow.DiffuseMf), nameof(RadiationRow.DiffuseUkmo), nameof(RadiationRow.DiffuseGem),
        nameof(RadiationRow.SwMean), nameof(RadiationRow.SwStd), nameof(RadiationRow.SwRange),
        nameof(RadiationRow.HourSin), nameof(RadiationRow.HourCos),
        nameof(RadiationRow.DoySin), nameof(RadiationRow.DoyCos),
    };

    public static readonly IReadOnlyList<ElementTrainerHarness.ModelAccessor<RadiationRow>> ModelAccessors = new[]
    {
        new ElementTrainerHarness.ModelAccessor<RadiationRow>("gfs_seamless",         r => r.SwGfs),
        new ElementTrainerHarness.ModelAccessor<RadiationRow>("ecmwf_ifs025",         r => r.SwEcmwf),
        new ElementTrainerHarness.ModelAccessor<RadiationRow>("icon_seamless",        r => r.SwIcon),
        new ElementTrainerHarness.ModelAccessor<RadiationRow>("meteofrance_seamless", r => r.SwMf),
        new ElementTrainerHarness.ModelAccessor<RadiationRow>("ukmo_seamless",        r => r.SwUkmo),
        new ElementTrainerHarness.ModelAccessor<RadiationRow>("gem_seamless",         r => r.SwGem),
    };

    /// <summary>
    /// Per-lead model membership. UKMO radiation is excluded at every lead — its
    /// ShortwaveRadiation field is essentially never populated in either backfill
    /// (≥48h: 99.99% null, 24h: 25% null) or live collects (92–100% null per
    /// 2026-04-26 audit). A 6-model trainer would learn to lean on UKMO during
    /// training and find nothing to feed it at predict time. Unlike the other
    /// elements, UKMO is genuinely missing data here, not a train-time-poisoning
    /// artefact, so 5-model is the right choice. No date filter needed for the
    /// same reason — there's no UKMO column to be poisoned by the pre-2024-09 block.
    /// </summary>
    public static IReadOnlyList<string> ModelsForLead(int leadHours) =>
        new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless" };

    public static List<RadiationRow> BuildForLead(
        string forecastsPath, string era5Path, string locationName, int leadHours, CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob  = Norm(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = Norm(Path.Combine(era5Path, "**", "*.parquet"));

        var modelList = string.Join(",", ModelsForLead(leadHours).Select(m => $"'{m}'"));
        // UKMO excluded at every lead now (was 26% null at 24h, 100% at 48/72h — inconsistency
        // hurt the 24h blender). Pivot returns NULL for UKMO columns so the row schema stays
        // constant across leads.
        var ukmoPivotClauses = @"
        CAST(NULL AS DOUBLE) AS sw_ukmo,
        CAST(NULL AS DOUBLE) AS dr_ukmo,
        CAST(NULL AS DOUBLE) AS df_ukmo,";

        // Required-models WHERE clause (5 models excluding UKMO; UKMO is optional at
        // 24h via NaN, structurally absent at 48/72h).
        var fcWhere =
            $"LocationName = '{locationName}' " +
            $"AND RunTimeSource = 'offset_day' " +
            $"AND LeadHours = {leadHours} " +
            $"AND Model IN ({modelList})";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model,
           ShortwaveRadiation AS sw, DirectRadiation AS dr, DiffuseRadiation AS df,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN sw END) AS sw_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN sw END) AS sw_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN sw END) AS sw_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN sw END) AS sw_mf,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN sw END) AS sw_gem,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN dr END) AS dr_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN dr END) AS dr_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN dr END) AS dr_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN dr END) AS dr_mf,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN dr END) AS dr_gem,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN df END) AS df_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN df END) AS df_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN df END) AS df_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN df END) AS df_mf,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN df END) AS df_gem,{ukmoPivotClauses}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, ShortwaveRadiation AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND ShortwaveRadiation IS NOT NULL
)
SELECT
    p.ValidTimeUtc,
    p.sw_gfs, p.sw_ecmwf, p.sw_icon, p.sw_mf, p.sw_ukmo, p.sw_gem,
    p.dr_gfs, p.dr_ecmwf, p.dr_icon, p.dr_mf, p.dr_ukmo, p.dr_gem,
    p.df_gfs, p.df_ecmwf, p.df_icon, p.df_mf, p.df_ukmo, p.df_gem,
    e.truth
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
WHERE p.sw_gfs IS NOT NULL AND p.sw_ecmwf IS NOT NULL AND p.sw_icon IS NOT NULL
  AND p.sw_mf  IS NOT NULL AND p.sw_gem   IS NOT NULL
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RadiationRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var sw = ReadFloats(r, 1, 6);
            var dr = ReadFloats(r, 7, 6);
            var df = ReadFloats(r, 13, 6);
            var truth = (float)r.GetDouble(19);
            rows.Add(ComposeRow(valid, sw, dr, df, truth));
        }
        return rows;
    }

    public static RadiationRow ComposeRow(
        DateTime validTimeUtc,
        ReadOnlySpan<float> sw, ReadOnlySpan<float> direct, ReadOnlySpan<float> diffuse,
        float era5Sw)
    {
        if (sw.Length      != 6) throw new ArgumentException("Expected 6 SW values", nameof(sw));
        if (direct.Length  != 6) throw new ArgumentException("Expected 6 direct values", nameof(direct));
        if (diffuse.Length != 6) throw new ArgumentException("Expected 6 diffuse values", nameof(diffuse));

        var spread = InterModelSpread.From(sw);
        var cal = CalendarFeatures.From(validTimeUtc);

        return new RadiationRow
        {
            ValidTimeUtc = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc),
            SwGfs = sw[0], SwEcmwf = sw[1], SwIcon = sw[2], SwMf = sw[3], SwUkmo = sw[4], SwGem = sw[5],
            DirectGfs = direct[0], DirectEcmwf = direct[1], DirectIcon = direct[2],
            DirectMf  = direct[3], DirectUkmo  = direct[4], DirectGem  = direct[5],
            DiffuseGfs = diffuse[0], DiffuseEcmwf = diffuse[1], DiffuseIcon = diffuse[2],
            DiffuseMf  = diffuse[3], DiffuseUkmo  = diffuse[4], DiffuseGem  = diffuse[5],
            SwMean = spread.Mean, SwStd = spread.Std, SwRange = spread.Range,
            HourSin = cal.HourSin, HourCos = cal.HourCos, DoySin = cal.DoySin, DoyCos = cal.DoyCos,
            Era5Sw = era5Sw,
        };
    }

    private static float[] ReadFloats(System.Data.IDataReader r, int start, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
        {
            var ord = start + i;
            arr[i] = r.IsDBNull(ord) ? float.NaN : (float)r.GetDouble(ord);
        }
        return arr;
    }

    private static string Norm(string p) => p.Replace('\\', '/').Replace("'", "''");
}
