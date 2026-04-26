"""Audit UKMO live-collect (RunTimeSource='reported') field coverage for the
four Element variables before reverting Pattern 1 → 6-model.

Question: do live UKMO collects actually return wind / humidity / shortwave /
cloud, or only some? If a variable is essentially never present live, a
6-model trainer would learn to lean on UKMO during training but have nothing
to feed it at predict time, which would silently degrade live accuracy.
"""
from __future__ import annotations

from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "data" / "forecasts" / "location=bonehill_rocks"
LEADS = (24, 48, 72)


def main() -> None:
    con = duckdb.connect()
    glob = (FORECASTS / "model=ukmo_seamless" / "**" / "*.parquet").as_posix()
    con.execute(f"""
        CREATE OR REPLACE VIEW ukmo_live AS
            SELECT * FROM read_parquet('{glob}', hive_partitioning=false, union_by_name=true)
            WHERE RunTimeSource='reported'
              AND LeadHours IN ({','.join(map(str, LEADS))})
    """)

    n = con.execute("SELECT COUNT(*) FROM ukmo_live").fetchone()[0]
    print(f"=== Live (reported) UKMO rows in tree at leads {LEADS}: {n:,} ===\n")

    if n == 0:
        print("No live UKMO rows yet — the live collector hasn't accumulated history at these leads.")
        # Fall back to looking at any UKMO rows from the most recent dates.
        recent_glob = (FORECASTS / "model=ukmo_seamless" / "date=2026-04-2*" / "**" / "*.parquet").as_posix()
        con.execute(f"""
            CREATE OR REPLACE VIEW ukmo_recent AS
                SELECT * FROM read_parquet('{recent_glob}', hive_partitioning=false, union_by_name=true)
        """)
        n2 = con.execute("SELECT COUNT(*) FROM ukmo_recent").fetchone()[0]
        print(f"Falling back to recent (April 2026) UKMO rows of any source: {n2:,}\n")
        target = "ukmo_recent"
    else:
        target = "ukmo_live"

    print(f"=== Per-variable nullness across {target} rows, by lead ===")
    df = con.execute(f"""
        SELECT LeadHours, RunTimeSource,
               COUNT(*) AS n,
               (1 - AVG(CAST(Temperature2m IS NOT NULL AS INT)))*100 AS pct_null_temp,
               (1 - AVG(CAST(DewPoint2m IS NOT NULL AS INT)))*100 AS pct_null_dewpt,
               (1 - AVG(CAST(RelativeHumidity2m IS NOT NULL AS INT)))*100 AS pct_null_rh,
               (1 - AVG(CAST(Precipitation IS NOT NULL AS INT)))*100 AS pct_null_precip,
               (1 - AVG(CAST(WindSpeed10m IS NOT NULL AS INT)))*100 AS pct_null_wind,
               (1 - AVG(CAST(WindDirection10m IS NOT NULL AS INT)))*100 AS pct_null_winddir,
               (1 - AVG(CAST(WindGusts10m IS NOT NULL AS INT)))*100 AS pct_null_gust,
               (1 - AVG(CAST(CloudCover IS NOT NULL AS INT)))*100 AS pct_null_cloud,
               (1 - AVG(CAST(CloudCoverLow IS NOT NULL AS INT)))*100 AS pct_null_cloud_low,
               (1 - AVG(CAST(CloudCoverMid IS NOT NULL AS INT)))*100 AS pct_null_cloud_mid,
               (1 - AVG(CAST(CloudCoverHigh IS NOT NULL AS INT)))*100 AS pct_null_cloud_high,
               (1 - AVG(CAST(ShortwaveRadiation IS NOT NULL AS INT)))*100 AS pct_null_sw
        FROM {target} GROUP BY LeadHours, RunTimeSource ORDER BY LeadHours, RunTimeSource
    """).fetch_df()
    print(df.to_string(index=False))

    print("\n=== Recent date sweep (last 30 days, any RunTimeSource) — UKMO Element fields ===")
    recent_glob = (FORECASTS / "model=ukmo_seamless" / "date=*" / "*.parquet").as_posix()
    df2 = con.execute(f"""
        WITH r AS (
            SELECT * FROM read_parquet('{recent_glob}', hive_partitioning=false, union_by_name=true)
            WHERE date >= DATE '2026-03-26'
        )
        SELECT date, RunTimeSource, COUNT(*) AS n,
               (1 - AVG(CAST(WindSpeed10m IS NOT NULL AS INT)))*100 AS pct_null_wind,
               (1 - AVG(CAST(RelativeHumidity2m IS NOT NULL AS INT)))*100 AS pct_null_rh,
               (1 - AVG(CAST(CloudCover IS NOT NULL AS INT)))*100 AS pct_null_cloud,
               (1 - AVG(CAST(ShortwaveRadiation IS NOT NULL AS INT)))*100 AS pct_null_sw
        FROM r GROUP BY date, RunTimeSource
        ORDER BY date DESC, RunTimeSource
        LIMIT 60
    """).fetch_df()
    print(df2.to_string(index=False))

    print("\n=== Lead-dependence check for live UKMO Element fields (latest week) ===")
    df3 = con.execute(f"""
        WITH r AS (
            SELECT * FROM read_parquet('{recent_glob}', hive_partitioning=false, union_by_name=true)
            WHERE date >= DATE '2026-04-19' AND RunTimeSource='reported'
        )
        SELECT LeadHours, COUNT(*) AS n,
               (1 - AVG(CAST(WindSpeed10m IS NOT NULL AS INT)))*100 AS pct_null_wind,
               (1 - AVG(CAST(RelativeHumidity2m IS NOT NULL AS INT)))*100 AS pct_null_rh,
               (1 - AVG(CAST(DewPoint2m IS NOT NULL AS INT)))*100 AS pct_null_dewpt,
               (1 - AVG(CAST(CloudCover IS NOT NULL AS INT)))*100 AS pct_null_cloud,
               (1 - AVG(CAST(CloudCoverLow IS NOT NULL AS INT)))*100 AS pct_null_cloud_low,
               (1 - AVG(CAST(ShortwaveRadiation IS NOT NULL AS INT)))*100 AS pct_null_sw
        FROM r GROUP BY LeadHours
        HAVING n > 0
        ORDER BY LeadHours
    """).fetch_df()
    print(df3.to_string(index=False))


if __name__ == "__main__":
    main()
