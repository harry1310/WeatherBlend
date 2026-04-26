using DuckDB.NET.Data;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// SQL pivot + row composition for the wind blender.
/// Pulls per-model <c>wind_speed_10m</c> and <c>wind_direction_10m</c> from the
/// Previous Runs forecast tree and joins to ERA5 <c>wind_speed_10m</c> as truth.
///
/// MétéoFrance is excluded entirely: audit 2026-04-25 confirmed Open-Meteo's
/// Previous Runs API ships no MF wind data (100% null on both speed and direction).
/// Including MF columns would just mean five-of-six models contributing always-NaN
/// features — better to be explicit about the 5-model membership.
/// </summary>
public static class WindFeatureBuilder
{
    /// <summary>Public feature names persisted to feature_schema.json.</summary>
    public static readonly IReadOnlyList<string> PublicFeatureNames = new[]
    {
        "wind_spd_gfs", "wind_spd_ecmwf", "wind_spd_icon", "wind_spd_ukmo", "wind_spd_gem",
        "wind_dir_sin_gfs", "wind_dir_cos_gfs",
        "wind_dir_sin_ecmwf", "wind_dir_cos_ecmwf",
        "wind_dir_sin_icon", "wind_dir_cos_icon",
        "wind_dir_sin_ukmo", "wind_dir_cos_ukmo",
        "wind_dir_sin_gem", "wind_dir_cos_gem",
        "wind_spd_mean", "wind_spd_std", "wind_spd_range",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    };

    /// <summary>Internal C# property names ML.NET binds on. Order aligned 1:1 with PublicFeatureNames.</summary>
    public static readonly IReadOnlyList<string> InternalFeatureColumns = new[]
    {
        nameof(WindRow.SpdGfs), nameof(WindRow.SpdEcmwf), nameof(WindRow.SpdIcon),
        nameof(WindRow.SpdUkmo), nameof(WindRow.SpdGem),
        nameof(WindRow.DirSinGfs), nameof(WindRow.DirCosGfs),
        nameof(WindRow.DirSinEcmwf), nameof(WindRow.DirCosEcmwf),
        nameof(WindRow.DirSinIcon), nameof(WindRow.DirCosIcon),
        nameof(WindRow.DirSinUkmo), nameof(WindRow.DirCosUkmo),
        nameof(WindRow.DirSinGem), nameof(WindRow.DirCosGem),
        nameof(WindRow.SpdMean), nameof(WindRow.SpdStd), nameof(WindRow.SpdRange),
        nameof(WindRow.HourSin), nameof(WindRow.HourCos),
        nameof(WindRow.DoySin), nameof(WindRow.DoyCos),
    };

    /// <summary>Per-model accessors used by the harness's best-single-by-val-MAE picker.</summary>
    public static readonly IReadOnlyList<ElementTrainerHarness.ModelAccessor<WindRow>> ModelAccessors = new[]
    {
        new ElementTrainerHarness.ModelAccessor<WindRow>("gfs_seamless",  r => r.SpdGfs),
        new ElementTrainerHarness.ModelAccessor<WindRow>("ecmwf_ifs025",  r => r.SpdEcmwf),
        new ElementTrainerHarness.ModelAccessor<WindRow>("icon_seamless", r => r.SpdIcon),
        new ElementTrainerHarness.ModelAccessor<WindRow>("ukmo_seamless", r => r.SpdUkmo),
        new ElementTrainerHarness.ModelAccessor<WindRow>("gem_seamless",  r => r.SpdGem),
    };

    /// <summary>
    /// Per-lead model membership. Wind drops MF (Open-Meteo Previous Runs ships no MF
    /// wind data — confirmed by 2026-04-25 audit) but KEEPS UKMO. Wind is the only
    /// target where the 6-model + restricted-window variant beats 5-model + full
    /// window on the same fixed test rows (4-way bake-off 2026-04-26): UKMO's regional
    /// UK wind skill outweighs the ~30% training-data loss. Same 5-model membership
    /// at every lead.
    /// </summary>
    public static IReadOnlyList<string> ModelsForLead(int leadHours) =>
        new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ukmo_seamless", "gem_seamless" };

    public static List<WindRow> BuildForLead(
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

        // 5-model membership (UKMO kept, MF excluded). Training restricted to the
        // post-2024-09 clean window — required for the 6-model variant since UKMO is
        // all-NULL pre-2024-09 in the Open-Meteo archive.
        var fcWhere =
            $"LocationName = '{locationName}' " +
            $"AND RunTimeSource = 'offset_day' " +
            $"AND LeadHours = {leadHours} " +
            $"AND WindSpeed10m IS NOT NULL " +
            $"AND WindDirection10m IS NOT NULL " +
            $"AND ValidTimeUtc >= TIMESTAMP '{TrainingWindow.UkmoCleanWindowStart}' " +
            $"AND Model IN ('gfs_seamless','ecmwf_ifs025','icon_seamless','ukmo_seamless','gem_seamless')";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, WindSpeed10m AS spd, WindDirection10m AS dir,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'  THEN spd END) AS spd_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'  THEN spd END) AS spd_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless' THEN spd END) AS spd_icon,
        MAX(CASE WHEN Model = 'ukmo_seamless' THEN spd END) AS spd_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'  THEN spd END) AS spd_gem,
        MAX(CASE WHEN Model = 'gfs_seamless'  THEN dir END) AS dir_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'  THEN dir END) AS dir_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless' THEN dir END) AS dir_icon,
        MAX(CASE WHEN Model = 'ukmo_seamless' THEN dir END) AS dir_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'  THEN dir END) AS dir_gem
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, WindSpeed10m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND WindSpeed10m IS NOT NULL
      AND ValidTimeUtc >= TIMESTAMP '{TrainingWindow.UkmoCleanWindowStart}'
)
SELECT
    p.ValidTimeUtc,
    p.spd_gfs, p.spd_ecmwf, p.spd_icon, p.spd_ukmo, p.spd_gem,
    p.dir_gfs, p.dir_ecmwf, p.dir_icon, p.dir_ukmo, p.dir_gem,
    e.truth
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
WHERE p.spd_gfs   IS NOT NULL
  AND p.spd_ecmwf IS NOT NULL
  AND p.spd_icon  IS NOT NULL
  AND p.spd_gem   IS NOT NULL
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<WindRow>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();

            var valid = r.GetDateTime(0);
            var spd = new float[5];
            for (int i = 0; i < 5; i++) spd[i] = r.IsDBNull(1 + i) ? float.NaN : (float)r.GetDouble(1 + i);
            var dir = new float[5];
            for (int i = 0; i < 5; i++) dir[i] = r.IsDBNull(6 + i) ? float.NaN : (float)r.GetDouble(6 + i);
            var truth = (float)r.GetDouble(11);

            rows.Add(ComposeRow(valid, spd, dir, truth));
        }

        return rows;
    }

    /// <summary>
    /// Compose one row from raw per-model values + ERA5 truth. Pure function so the SQL
    /// loop stays a one-liner and the row-shape logic gets a unit test fixture.
    /// </summary>
    public static WindRow ComposeRow(
        DateTime validTimeUtc,
        ReadOnlySpan<float> speeds,    // 5: gfs, ecmwf, icon, ukmo, gem
        ReadOnlySpan<float> directions, // 5: gfs, ecmwf, icon, ukmo, gem (degrees)
        float era5WindSpeed)
    {
        if (speeds.Length != 5)     throw new ArgumentException("Expected 5 speeds (gfs/ecmwf/icon/ukmo/gem)", nameof(speeds));
        if (directions.Length != 5) throw new ArgumentException("Expected 5 directions (gfs/ecmwf/icon/ukmo/gem)", nameof(directions));

        var spread = InterModelSpread.From(speeds);
        var cal = CalendarFeatures.From(validTimeUtc);

        // Bearings → unit-vector components. North = 0°, increasing clockwise.
        static (float Sin, float Cos) Encode(float deg)
        {
            if (float.IsNaN(deg)) return (float.NaN, float.NaN);
            var rad = deg * (float)(Math.PI / 180.0);
            return ((float)Math.Sin(rad), (float)Math.Cos(rad));
        }
        var (s0, c0) = Encode(directions[0]);
        var (s1, c1) = Encode(directions[1]);
        var (s2, c2) = Encode(directions[2]);
        var (s3, c3) = Encode(directions[3]);
        var (s4, c4) = Encode(directions[4]);

        return new WindRow
        {
            ValidTimeUtc = validTimeUtc.Kind == DateTimeKind.Utc
                ? validTimeUtc
                : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc),
            SpdGfs   = speeds[0], SpdEcmwf = speeds[1], SpdIcon = speeds[2],
            SpdUkmo  = speeds[3], SpdGem   = speeds[4],
            DirSinGfs   = s0, DirCosGfs   = c0,
            DirSinEcmwf = s1, DirCosEcmwf = c1,
            DirSinIcon  = s2, DirCosIcon  = c2,
            DirSinUkmo  = s3, DirCosUkmo  = c3,
            DirSinGem   = s4, DirCosGem   = c4,
            SpdMean = spread.Mean, SpdStd = spread.Std, SpdRange = spread.Range,
            HourSin = cal.HourSin, HourCos = cal.HourCos,
            DoySin  = cal.DoySin,  DoyCos  = cal.DoyCos,
            Era5WindSpeed = era5WindSpeed,
        };
    }

    private static string NormaliseGlob(string path) => path.Replace('\\', '/').Replace("'", "''");
}
