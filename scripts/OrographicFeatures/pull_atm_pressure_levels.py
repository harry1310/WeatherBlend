"""
Pull historical pressure-level atmospheric data from Open-Meteo's archive-api
for each configured location, save as parquet under data/static/atm_history/.

Used downstream by `build_atm_climatology.py` to compute per-station
(wind_sector × month) climatology features.

Source: archive-api with `&models=gfs_seamless` — GFS pressure-level forecasts
go back to 2022-06 (vs ECMWF IFS only to 2024-06), so we use GFS for the
longest stable record. Climatology averages over thousands of samples so any
single-model bias is irrelevant.

Rate: archive-api has light rate-limiting (existing ERA5 backfill uses 2s
delay). Per location: ~48 monthly chunks × ~3s = ~2 min. Total for 2
locations: ~5 min. Negligible compared to forecast backfill.

Idempotent: re-running re-downloads any month whose parquet is older than
the most recent successful upload (or always, if --force). Skips months that
already have non-empty parquets unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "static" / "atm_history"

LOCATIONS = [
    ("bonehill_rocks", 50.5831, -3.7931),
    ("membury_devon",  50.8254, -3.0000),
]

# 12 pressure-level variables. Enough to compute stability (lapse rates between
# levels), moisture content, and upper-air wind/shear. Geopotential at 500 hPa
# is the synoptic-scale circulation indicator.
HOURLY_VARS = [
    "temperature_925hPa", "temperature_850hPa", "temperature_700hPa", "temperature_500hPa",
    "dew_point_850hPa", "dew_point_700hPa",
    "relative_humidity_850hPa",
    "wind_speed_850hPa", "wind_direction_850hPa",
    "wind_speed_500hPa", "wind_direction_500hPa",
    "geopotential_height_500hPa",
]

MODEL = "gfs_seamless"
START_DEFAULT = "2022-06-01"
DELAY_SECONDS = 2
TIMEOUT_SECONDS = 60


def month_chunks(start: date, end: date):
    """Yield (chunk_start, chunk_end) per calendar month, last chunk inclusive of `end`."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), end)
        chunk_start = max(cur, start)
        if chunk_start <= chunk_end:
            yield chunk_start, chunk_end
        cur = nxt


def fetch_month(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """One archive-api call for one (lat, lon, month). Returns a DataFrame of the
    hourly data with proper UTC timestamps. Empty df if all variables NULL."""
    qs = urllib.parse.urlencode({
        "latitude":   f"{lat:.4f}",
        "longitude":  f"{lon:.4f}",
        "hourly":     ",".join(HOURLY_VARS),
        "models":     MODEL,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "timezone":   "UTC",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "WeatherBlend-AtmHistory/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
        payload = json.load(r)
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame({k: hourly[k] for k in hourly})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START_DEFAULT, help="ISO date, default 2022-06-01")
    p.add_argument("--end", default=date.today().isoformat(), help="ISO date, default today")
    p.add_argument("--force", action="store_true", help="re-download even if parquet exists")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for loc_name, lat, lon in LOCATIONS:
        loc_dir = OUT_DIR / f"location={loc_name}"
        loc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {loc_name} ({lat:.4f}, {lon:.4f}) ===")
        for c_start, c_end in month_chunks(start, end):
            out_path = loc_dir / f"{c_start:%Y-%m}.parquet"
            if out_path.exists() and not args.force and out_path.stat().st_size > 0:
                print(f"  {c_start:%Y-%m}  exists, skip (use --force to refetch)")
                continue
            try:
                df = fetch_month(lat, lon, c_start, c_end)
                if df.empty:
                    print(f"  {c_start:%Y-%m}  NO DATA (skipping)")
                    time.sleep(DELAY_SECONDS)
                    continue
                df["LocationName"] = loc_name
                df["Source"] = MODEL
                df.to_parquet(out_path, index=False)
                # Show non-null counts for key variables
                nn_t850 = df["temperature_850hPa"].notna().sum()
                nn_w500 = df["wind_direction_500hPa"].notna().sum()
                print(f"  {c_start:%Y-%m}  {len(df)} rows  T850 nn={nn_t850}  Wdir500 nn={nn_w500}")
                total_rows += len(df)
            except Exception as e:
                print(f"  {c_start:%Y-%m}  ERROR: {e}", file=sys.stderr)
            time.sleep(DELAY_SECONDS)
    print(f"\nTotal new rows: {total_rows}")


if __name__ == "__main__":
    main()
