using DuckDB.NET.Data;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Humidity;

/// <summary>
/// SQL pivot + row composition for the humidity blender.
/// Pulls per-model <c>relative_humidity_2m</c> and <c>dew_point_2m</c> from the
/// Previous Runs forecast tree; joins to ERA5 <c>relative_humidity_2m</c> as truth.
/// Audit 2026-04-25 confirmed all six models populate both columns at Bonehill.
/// </summary>
public static class HumidityFeatureBuilder
{
    public static readonly IReadOnlyList<string> PublicFeatureNames = new[]
    {
        "rh_gfs", "rh_ecmwf", "rh_icon", "rh_mf", "rh_ukmo", "rh_gem",
        "dp_gfs", "dp_ecmwf", "dp_icon", "dp_mf", "dp_ukmo", "dp_gem",
        "rh_mean", "rh_std", "rh_range",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    public static readonly IReadOnlyList<string> InternalFeatureColumns = new[]
    {
        nameof(HumidityRow.RhGfs), nameof(HumidityRow.RhEcmwf), nameof(HumidityRow.RhIcon),
        nameof(HumidityRow.RhMf), nameof(HumidityRow.RhUkmo), nameof(HumidityRow.RhGem),
        nameof(HumidityRow.DpGfs), nameof(HumidityRow.DpEcmwf), nameof(HumidityRow.DpIcon),
        nameof(HumidityRow.DpMf), nameof(HumidityRow.DpUkmo), nameof(HumidityRow.DpGem),
        nameof(HumidityRow.RhMean), nameof(HumidityRow.RhStd), nameof(HumidityRow.RhRange),
        nameof(HumidityRow.HourSin), nameof(HumidityRow.HourCos),
        nameof(HumidityRow.DoySin), nameof(HumidityRow.DoyCos),
    };

    public static readonly IReadOnlyList<ElementTrainerHarness.ModelAccessor<HumidityRow>> ModelAccessors = new[]
    {
        new ElementTrainerHarness.ModelAccessor<HumidityRow>("gfs_seamless",         r => r.RhGfs),
        new ElementTrainerHarness.ModelAccessor<HumidityRow>("ecmwf_ifs025",         r => r.RhEcmwf),
        new ElementTrainerHarness.ModelAccessor<HumidityRow>("icon_seamless",        r => r.RhIcon),
        new ElementTrainerHarness.ModelAccessor<HumidityRow>("meteofrance_seamless", r => r.RhMf),
        new ElementTrainerHarness.ModelAccessor<HumidityRow>("ukmo_seamless",        r => r.RhUkmo),
        new ElementTrainerHarness.ModelAccessor<HumidityRow>("gem_seamless",         r => r.RhGem),
    };

    /// <summary>
    /// Per-lead model membership.
    ///
    /// MétéoFrance live forecasts cut off ~36h → drop MF at 48/72h so train/predict
    /// shape match.
    ///
    /// UKMO excluded entirely. 4-way bake-off 2026-04-26 confirmed humidity behaves
    /// like temp/precip: the 6-model + restricted-window variant loses to 5-model +
    /// full window by 4–6% MAE on the same fixed test rows. UKMO's signal doesn't
    /// outweigh the ~30% training-data loss for humidity. (See docs/DESIGN.md UKMO
    /// matrix; cloud and wind reach the opposite conclusion and keep UKMO.)
    ///
    /// Net membership: 5 models at 24h (no UKMO), 4 at 48/72h (no UKMO + no MF).
    /// </summary>
    public static IReadOnlyList<string> ModelsForLead(int leadHours) =>
        leadHours == 24
            ? new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless" }
            : new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless" };

    public static List<HumidityRow> BuildForLead(
        string forecastsPath, string era5Path, string locationName, int leadHours, CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob  = Norm(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = Norm(Path.Combine(era5Path, "**", "*.parquet"));

        var modelList = string.Join(",", ModelsForLead(leadHours).Select(m => $"'{m}'"));
        var includeMf = leadHours == 24;
        // MF excluded at 48/72h (live API horizon ~36h); UKMO excluded everywhere
        // (4-way bake-off 2026-04-26 — see ModelsForLead docstring).
        var mfPivotClauses = includeMf
            ? @"
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN rh END) AS rh_mf,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN dp END) AS dp_mf,"
            : @"
        CAST(NULL AS DOUBLE) AS rh_mf,
        CAST(NULL AS DOUBLE) AS dp_mf,";
        var mfWhereClause = includeMf
            ? "AND p.rh_mf IS NOT NULL AND p.dp_mf IS NOT NULL"
            : "";

        var fcWhere =
            $"LocationName = '{locationName}' " +
            $"AND RunTimeSource = 'offset_day' " +
            $"AND LeadHours = {leadHours} " +
            $"AND RelativeHumidity2m IS NOT NULL " +
            $"AND DewPoint2m IS NOT NULL " +
            $"AND Model IN ({modelList})";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RelativeHumidity2m AS rh, DewPoint2m AS dp,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN rh END) AS rh_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN rh END) AS rh_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN rh END) AS rh_icon,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN rh END) AS rh_gem,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN dp END) AS dp_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN dp END) AS dp_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN dp END) AS dp_icon,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN dp END) AS dp_gem,
        CAST(NULL AS DOUBLE) AS rh_ukmo,
        CAST(NULL AS DOUBLE) AS dp_ukmo,{mfPivotClauses}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, RelativeHumidity2m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND RelativeHumidity2m IS NOT NULL
)
SELECT
    p.ValidTimeUtc,
    p.rh_gfs, p.rh_ecmwf, p.rh_icon, p.rh_mf, p.rh_ukmo, p.rh_gem,
    p.dp_gfs, p.dp_ecmwf, p.dp_icon, p.dp_mf, p.dp_ukmo, p.dp_gem,
    e.truth
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
WHERE p.rh_gfs IS NOT NULL AND p.rh_ecmwf IS NOT NULL AND p.rh_icon IS NOT NULL
  AND p.rh_gem IS NOT NULL
  AND p.dp_gfs IS NOT NULL AND p.dp_ecmwf IS NOT NULL AND p.dp_icon IS NOT NULL
  AND p.dp_gem IS NOT NULL
  {mfWhereClause}
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<HumidityRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var rh = ReadFloats(r, 1, 6);
            var dp = ReadFloats(r, 7, 6);
            var truth = (float)r.GetDouble(13);
            rows.Add(ComposeRow(valid, rh, dp, truth));
        }
        return rows;
    }

    private static float[] ReadFloats(System.Data.IDataReader r, int start, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = r.IsDBNull(start + i) ? float.NaN : (float)r.GetDouble(start + i);
        return arr;
    }

    public static HumidityRow ComposeRow(
        DateTime validTimeUtc, ReadOnlySpan<float> rh, ReadOnlySpan<float> dp, float era5Rh)
    {
        if (rh.Length != 6) throw new ArgumentException("Expected 6 RH values", nameof(rh));
        if (dp.Length != 6) throw new ArgumentException("Expected 6 dewpoint values", nameof(dp));

        var spread = InterModelSpread.From(rh);
        var cal = CalendarFeatures.From(validTimeUtc);

        return new HumidityRow
        {
            ValidTimeUtc = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc),
            RhGfs = rh[0], RhEcmwf = rh[1], RhIcon = rh[2], RhMf = rh[3], RhUkmo = rh[4], RhGem = rh[5],
            DpGfs = dp[0], DpEcmwf = dp[1], DpIcon = dp[2], DpMf = dp[3], DpUkmo = dp[4], DpGem = dp[5],
            RhMean = spread.Mean, RhStd = spread.Std, RhRange = spread.Range,
            HourSin = cal.HourSin, HourCos = cal.HourCos, DoySin = cal.DoySin, DoyCos = cal.DoyCos,
            Era5Rh = era5Rh,
        };
    }

    private static string Norm(string p) => p.Replace('\\', '/').Replace("'", "''");
}
