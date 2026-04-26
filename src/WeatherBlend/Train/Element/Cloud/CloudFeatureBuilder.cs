using DuckDB.NET.Data;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Cloud;

/// <summary>
/// SQL pivot + row composition for the cloud-cover blender.
/// Pulls per-model total <c>cloud_cover</c> (only — layered cloud is unavailable
/// in Open-Meteo Previous Runs at Bonehill, see <see cref="CloudRow"/> docstring).
/// All six models populate total cloud cover.
/// </summary>
public static class CloudFeatureBuilder
{
    public static readonly IReadOnlyList<string> PublicFeatureNames = new[]
    {
        "cc_gfs", "cc_ecmwf", "cc_icon", "cc_mf", "cc_ukmo", "cc_gem",
        "cc_mean", "cc_std", "cc_range",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    public static readonly IReadOnlyList<string> InternalFeatureColumns = new[]
    {
        nameof(CloudRow.CcGfs), nameof(CloudRow.CcEcmwf), nameof(CloudRow.CcIcon),
        nameof(CloudRow.CcMf), nameof(CloudRow.CcUkmo), nameof(CloudRow.CcGem),
        nameof(CloudRow.CcMean), nameof(CloudRow.CcStd), nameof(CloudRow.CcRange),
        nameof(CloudRow.HourSin), nameof(CloudRow.HourCos),
        nameof(CloudRow.DoySin), nameof(CloudRow.DoyCos),
    };

    public static readonly IReadOnlyList<ElementTrainerHarness.ModelAccessor<CloudRow>> ModelAccessors = new[]
    {
        new ElementTrainerHarness.ModelAccessor<CloudRow>("gfs_seamless",         r => r.CcGfs),
        new ElementTrainerHarness.ModelAccessor<CloudRow>("ecmwf_ifs025",         r => r.CcEcmwf),
        new ElementTrainerHarness.ModelAccessor<CloudRow>("icon_seamless",        r => r.CcIcon),
        new ElementTrainerHarness.ModelAccessor<CloudRow>("meteofrance_seamless", r => r.CcMf),
        new ElementTrainerHarness.ModelAccessor<CloudRow>("ukmo_seamless",        r => r.CcUkmo),
        new ElementTrainerHarness.ModelAccessor<CloudRow>("gem_seamless",         r => r.CcGem),
    };

    /// <summary>
    /// Per-lead model membership.
    ///
    /// MétéoFrance live forecasts cut off ~36h → drop at 48/72h so train/predict match.
    ///
    /// UKMO restored 2026-04-26 after restricted-window bake-off — see
    /// <see cref="TrainingWindow"/>. UKMO ships total cloud cover at every lead in live
    /// collects, so train/predict shape match.
    ///
    /// Net: 6 models at 24h, 5 at 48/72h (no MF).
    /// </summary>
    public static IReadOnlyList<string> ModelsForLead(int leadHours) =>
        leadHours == 24
            ? new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "ukmo_seamless", "gem_seamless" }
            : new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ukmo_seamless", "gem_seamless" };

    public static List<CloudRow> BuildForLead(
        string forecastsPath, string era5Path, string locationName, int leadHours, CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob  = Norm(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = Norm(Path.Combine(era5Path, "**", "*.parquet"));

        var modelList = string.Join(",", ModelsForLead(leadHours).Select(m => $"'{m}'"));
        var includeMf = leadHours == 24;
        // UKMO restored at every lead; MF excluded at 48/72h (live API horizon ~36h).
        var mfPivotClause = includeMf
            ? "MAX(CASE WHEN Model = 'meteofrance_seamless' THEN cc END) AS cc_mf,"
            : "CAST(NULL AS DOUBLE) AS cc_mf,";
        var mfWhereClause = includeMf ? "AND p.cc_mf IS NOT NULL" : "";

        var fcWhere =
            $"LocationName = '{locationName}' " +
            $"AND RunTimeSource = 'offset_day' " +
            $"AND LeadHours = {leadHours} " +
            $"AND CloudCover IS NOT NULL " +
            $"AND ValidTimeUtc >= TIMESTAMP '{TrainingWindow.UkmoCleanWindowStart}' " +
            $"AND Model IN ({modelList})";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, CloudCover AS cc,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN cc END) AS cc_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN cc END) AS cc_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN cc END) AS cc_icon,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN cc END) AS cc_gem,
        MAX(CASE WHEN Model = 'ukmo_seamless'        THEN cc END) AS cc_ukmo,
        {mfPivotClause}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, CloudCover AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND CloudCover IS NOT NULL
      AND ValidTimeUtc >= TIMESTAMP '{TrainingWindow.UkmoCleanWindowStart}'
)
SELECT
    p.ValidTimeUtc,
    p.cc_gfs, p.cc_ecmwf, p.cc_icon, p.cc_mf, p.cc_ukmo, p.cc_gem,
    e.truth
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
WHERE p.cc_gfs IS NOT NULL AND p.cc_ecmwf IS NOT NULL AND p.cc_icon IS NOT NULL
  AND p.cc_gem IS NOT NULL
  {mfWhereClause}
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<CloudRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var cc = new float[6];
            for (int i = 0; i < 6; i++) cc[i] = r.IsDBNull(1 + i) ? float.NaN : (float)r.GetDouble(1 + i);
            var truth = (float)r.GetDouble(7);
            rows.Add(ComposeRow(valid, cc, truth));
        }
        return rows;
    }

    public static CloudRow ComposeRow(DateTime validTimeUtc, ReadOnlySpan<float> cc, float era5Cc)
    {
        if (cc.Length != 6) throw new ArgumentException("Expected 6 cloud values", nameof(cc));

        var spread = InterModelSpread.From(cc);
        var cal = CalendarFeatures.From(validTimeUtc);

        return new CloudRow
        {
            ValidTimeUtc = validTimeUtc.Kind == DateTimeKind.Utc ? validTimeUtc : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc),
            CcGfs = cc[0], CcEcmwf = cc[1], CcIcon = cc[2], CcMf = cc[3], CcUkmo = cc[4], CcGem = cc[5],
            CcMean = spread.Mean, CcStd = spread.Std, CcRange = spread.Range,
            HourSin = cal.HourSin, HourCos = cal.HourCos, DoySin = cal.DoySin, DoyCos = cal.DoyCos,
            Era5Cc = era5Cc,
        };
    }

    private static string Norm(string p) => p.Replace('\\', '/').Replace("'", "''");
}
