"""Drill into UKMO FIELD-level missingness (the actual 26% gap).

The previous script established that UKMO ROWS exist for ~99.9% of (lead, valid)
slots — the 26% gap is that the temperature/precip/wind/cloud COLUMNS within
those rows are NULL. This script asks: of the UKMO rows that have NULL temp,
what's their structure? Specific RunTimeUtc? Specific valid-time hour? Specific
date ranges? Specific months?
"""
from __future__ import annotations

from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "data" / "forecasts" / "location=bonehill_rocks"
LEADS = (24, 48, 72)


def main() -> None:
    con = duckdb.connect()
    glob_ukmo = (FORECASTS / "model=ukmo_seamless" / "date=*" / "*.parquet").as_posix()
    con.execute(f"""
        CREATE OR REPLACE VIEW ukmo AS
            SELECT * FROM read_parquet('{glob_ukmo}', hive_partitioning=false, union_by_name=true)
            WHERE LeadHours IN ({','.join(map(str, LEADS))})
    """)

    print("=== UKMO row count by (RunTimeSource, has-temp) ===")
    print(con.execute("""
        SELECT RunTimeSource,
               (Temperature2m IS NOT NULL) AS has_temp,
               COUNT(*) AS n
        FROM ukmo GROUP BY 1, 2 ORDER BY 1, 2
    """).fetch_df().to_string(index=False))

    print("\n=== Of UKMO rows with NULL temp: by valid-time month ===")
    print(con.execute("""
        SELECT strftime(ValidTimeUtc, '%Y-%m') AS ym,
               COUNT(*) FILTER (WHERE Temperature2m IS NULL) AS null_temp,
               COUNT(*) AS total,
               (COUNT(*) FILTER (WHERE Temperature2m IS NULL))*100.0/COUNT(*) AS pct_null
        FROM ukmo GROUP BY 1 ORDER BY 1
    """).fetch_df().to_string(index=False))

    print("\n=== Of UKMO rows with NULL temp: by valid-time hour (UTC) ===")
    print(con.execute("""
        SELECT extract('hour' FROM ValidTimeUtc) AS valid_hour,
               COUNT(*) FILTER (WHERE Temperature2m IS NULL) AS null_temp,
               COUNT(*) AS total,
               (COUNT(*) FILTER (WHERE Temperature2m IS NULL))*100.0/COUNT(*) AS pct_null
        FROM ukmo GROUP BY 1 ORDER BY 1
    """).fetch_df().to_string(index=False))

    print("\n=== Of UKMO rows with NULL temp: by RunTime hour (cycle) ===")
    print(con.execute("""
        SELECT extract('hour' FROM RunTimeUtc) AS run_hour,
               COUNT(*) FILTER (WHERE Temperature2m IS NULL) AS null_temp,
               COUNT(*) AS total,
               (COUNT(*) FILTER (WHERE Temperature2m IS NULL))*100.0/COUNT(*) AS pct_null
        FROM ukmo WHERE RunTimeUtc IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetch_df().to_string(index=False))

    print("\n=== Of UKMO rows with NULL temp: by date — top 30 worst dates ===")
    print(con.execute("""
        SELECT strftime(ValidTimeUtc, '%Y-%m-%d') AS d,
               COUNT(*) FILTER (WHERE Temperature2m IS NULL) AS null_temp,
               COUNT(*) AS total,
               (COUNT(*) FILTER (WHERE Temperature2m IS NULL))*100.0/COUNT(*) AS pct_null
        FROM ukmo GROUP BY 1
        HAVING null_temp > 0
        ORDER BY pct_null DESC, total DESC
        LIMIT 30
    """).fetch_df().to_string(index=False))

    print("\n=== Per-variable column nullness (within UKMO rows) — by lead ===")
    print(con.execute("""
        SELECT LeadHours,
               COUNT(*) AS rows,
               (1 - AVG(CAST(Temperature2m IS NOT NULL AS INT)))*100 AS pct_null_temp,
               (1 - AVG(CAST(DewPoint2m IS NOT NULL AS INT)))*100 AS pct_null_dewpt,
               (1 - AVG(CAST(RelativeHumidity2m IS NOT NULL AS INT)))*100 AS pct_null_rh,
               (1 - AVG(CAST(Precipitation IS NOT NULL AS INT)))*100 AS pct_null_precip,
               (1 - AVG(CAST(WindSpeed10m IS NOT NULL AS INT)))*100 AS pct_null_wind,
               (1 - AVG(CAST(WindDirection10m IS NOT NULL AS INT)))*100 AS pct_null_winddir,
               (1 - AVG(CAST(CloudCover IS NOT NULL AS INT)))*100 AS pct_null_cloud,
               (1 - AVG(CAST(SurfacePressure IS NOT NULL AS INT)))*100 AS pct_null_msl,
               (1 - AVG(CAST(Cape IS NOT NULL AS INT)))*100 AS pct_null_cape,
               (1 - AVG(CAST(ShortwaveRadiation IS NOT NULL AS INT)))*100 AS pct_null_sw
        FROM ukmo GROUP BY LeadHours ORDER BY LeadHours
    """).fetch_df().to_string(index=False))

    print("\n=== Co-occurrence: do temp/precip/wind nulls travel together? (lead 24h) ===")
    print(con.execute("""
        SELECT
            (Temperature2m IS NULL) AS t_null,
            (Precipitation IS NULL) AS p_null,
            (WindSpeed10m IS NULL) AS w_null,
            COUNT(*) AS n
        FROM ukmo WHERE LeadHours = 24
        GROUP BY 1, 2, 3 ORDER BY n DESC
    """).fetch_df().to_string(index=False))


if __name__ == "__main__":
    main()
