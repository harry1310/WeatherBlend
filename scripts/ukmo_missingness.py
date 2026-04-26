"""Stratified UKMO presence audit on the WeatherBlend forecasts tree.

Answers Harry's question (2026-04-26): is the 26% UKMO gap a uniform random
miss, or is it concentrated in particular {leads, valid-time hours, run-time
hours, valid-time months, run-time sources}? If it's clumped, the per-row
"5-model wins" finding doesn't necessarily survive into the where-UKMO-is-
present subset — and a 6-model variant trained only on present-UKMO rows
might still beat 5-model in those same rows.

Reads from forecasts hive directly via DuckDB. Joins UKMO presence against
the full set of (Model='gfs_seamless', RunTimeSource='offset_day') anchor
rows — gfs_seamless is the most complete, so its row set is the closest
proxy to "every (run, lead, valid) we'd ever blend at".

Run with:
    .venv/Scripts/python.exe scripts/ukmo_missingness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "data" / "forecasts" / "location=bonehill_rocks"
LEADS = (24, 48, 72)


def main() -> None:
    if not FORECASTS.exists():
        sys.exit(f"forecast root not found: {FORECASTS}")

    con = duckdb.connect()
    glob_ukmo = (FORECASTS / "model=ukmo_seamless" / "date=*" / "*.parquet").as_posix()
    glob_gfs = (FORECASTS / "model=gfs_seamless"  / "date=*" / "*.parquet").as_posix()

    print("Loading UKMO + GFS rows…")
    con.execute(f"""
        CREATE OR REPLACE VIEW ukmo AS
            SELECT * FROM read_parquet('{glob_ukmo}', hive_partitioning=false, union_by_name=true)
            WHERE LeadHours IN ({','.join(map(str, LEADS))})
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW gfs AS
            SELECT * FROM read_parquet('{glob_gfs}', hive_partitioning=false, union_by_name=true)
            WHERE LeadHours IN ({','.join(map(str, LEADS))})
    """)

    print("\n=== Anchor row counts (gfs_seamless = the 'should-have' baseline) ===")
    print(con.execute(
        "SELECT RunTimeSource, COUNT(*) FROM gfs GROUP BY 1 ORDER BY 2 DESC"
    ).fetch_df().to_string(index=False))

    # --- 1. Aggregate presence ---
    print("\n=== UKMO presence overall (vs gfs_seamless anchors at same lead) ===")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE anchors AS
        SELECT g.ValidTimeUtc, g.LeadHours, g.RunTimeUtc, g.RunTimeSource,
               u.ValidTimeUtc IS NOT NULL AS ukmo_present,
               u.Temperature2m IS NOT NULL AS ukmo_temp_present,
               u.Precipitation IS NOT NULL AS ukmo_precip_present
        FROM gfs g
        LEFT JOIN ukmo u
          ON u.ValidTimeUtc = g.ValidTimeUtc
         AND u.LeadHours = g.LeadHours
         AND u.RunTimeSource = g.RunTimeSource
    """)
    print(con.execute("""
        SELECT COUNT(*) AS anchors,
               SUM(ukmo_present)::INT AS ukmo_rows,
               (1 - AVG(CAST(ukmo_present AS INT)))*100 AS pct_missing,
               (1 - AVG(CAST(ukmo_temp_present AS INT)))*100 AS pct_temp_missing,
               (1 - AVG(CAST(ukmo_precip_present AS INT)))*100 AS pct_precip_missing
        FROM anchors
    """).fetch_df().to_string(index=False))

    # --- 2. Per-lead ---
    print("\n=== UKMO presence by lead ===")
    print(con.execute("""
        SELECT LeadHours,
               COUNT(*) AS n,
               (1 - AVG(CAST(ukmo_present AS INT)))*100 AS pct_missing,
               (1 - AVG(CAST(ukmo_temp_present AS INT)))*100 AS pct_temp_missing,
               (1 - AVG(CAST(ukmo_precip_present AS INT)))*100 AS pct_precip_missing
        FROM anchors GROUP BY LeadHours ORDER BY LeadHours
    """).fetch_df().to_string(index=False))

    # --- 3. By RunTimeSource (offset_day = backfill, reported = live) ---
    print("\n=== UKMO presence by RunTimeSource (offset_day=backfill, reported=live) ===")
    print(con.execute("""
        SELECT RunTimeSource, LeadHours,
               COUNT(*) AS n,
               (1 - AVG(CAST(ukmo_present AS INT)))*100 AS pct_missing
        FROM anchors GROUP BY 1, 2 ORDER BY 1, 2
    """).fetch_df().to_string(index=False))

    # --- 4. Temporal: by valid-time year-month ---
    print("\n=== UKMO presence by valid-time year-month (top 24 most-recent + worst 5) ===")
    monthly = con.execute("""
        SELECT strftime(ValidTimeUtc, '%Y-%m') AS ym,
               COUNT(*) AS n,
               (1 - AVG(CAST(ukmo_present AS INT)))*100 AS pct_missing
        FROM anchors GROUP BY 1 ORDER BY 1
    """).fetch_df()
    print("recent 24 months:")
    print(monthly.tail(24).to_string(index=False))
    print("\nworst 5 months by missingness:")
    print(monthly.sort_values("pct_missing", ascending=False).head(5).to_string(index=False))

    # --- 5. Diurnal: by valid-time hour ---
    print("\n=== UKMO presence by valid-time hour (UTC) ===")
    print(con.execute("""
        SELECT extract('hour' FROM ValidTimeUtc) AS valid_hour,
               COUNT(*) AS n,
               (1 - AVG(CAST(ukmo_present AS INT)))*100 AS pct_missing
        FROM anchors GROUP BY 1 ORDER BY 1
    """).fetch_df().to_string(index=False))

    # --- 6. By RunTime hour (which UKMO cycles get archived?) ---
    print("\n=== UKMO presence by run-time hour (cycle hour) — among rows where RunTime is known ===")
    print(con.execute("""
        SELECT extract('hour' FROM RunTimeUtc) AS run_hour,
               COUNT(*) AS n,
               (1 - AVG(CAST(ukmo_present AS INT)))*100 AS pct_missing
        FROM anchors WHERE RunTimeUtc IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetch_df().to_string(index=False))

    # --- 7. Burstiness: are misses contiguous? ---
    print("\n=== Burstiness: longest contiguous UKMO-missing streaks (top 10, lead=24h) ===")
    streaks = con.execute("""
        WITH ordered AS (
            SELECT ValidTimeUtc, ukmo_present,
                   LAG(ukmo_present) OVER (ORDER BY ValidTimeUtc) AS prev
            FROM anchors WHERE LeadHours = 24
              AND RunTimeSource = 'offset_day'
        ),
        runs AS (
            SELECT ValidTimeUtc, ukmo_present,
                   SUM(CASE WHEN ukmo_present = prev THEN 0 ELSE 1 END)
                       OVER (ORDER BY ValidTimeUtc ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run_id
            FROM ordered
        )
        SELECT MIN(ValidTimeUtc) AS run_start,
               MAX(ValidTimeUtc) AS run_end,
               COUNT(*) AS run_hours
        FROM runs
        WHERE ukmo_present = false
        GROUP BY run_id
        ORDER BY run_hours DESC
        LIMIT 10
    """).fetch_df()
    print(streaks.to_string(index=False))

    # --- 8. Per-variable: temp vs precip missingness (within UKMO-present rows) ---
    print("\n=== Within UKMO-present rows: which fields are present? ===")
    print(con.execute("""
        SELECT LeadHours,
               COUNT(*) AS ukmo_rows,
               (1 - AVG(CAST(Temperature2m IS NOT NULL AS INT)))*100 AS pct_temp_null,
               (1 - AVG(CAST(Precipitation IS NOT NULL AS INT)))*100 AS pct_precip_null,
               (1 - AVG(CAST(WindSpeed10m IS NOT NULL AS INT)))*100 AS pct_wind_null,
               (1 - AVG(CAST(CloudCover IS NOT NULL AS INT)))*100 AS pct_cloud_null,
               (1 - AVG(CAST(ShortwaveRadiation IS NOT NULL AS INT)))*100 AS pct_sw_null
        FROM ukmo GROUP BY LeadHours ORDER BY LeadHours
    """).fetch_df().to_string(index=False))


if __name__ == "__main__":
    main()
