"""Met Office Global Deterministic 10km archive backfill.

Pulls historical Met Office Global UM forecasts from AWS Open Data
(`s3://met-office-atmospheric-model-data/global-deterministic-10km/`) and
writes them into WeatherBlend's hive forecast tree as `model=met_office_global`,
exactly like Open-Meteo / GFS rows. Once landed, the existing compare / verify
/ blender plumbing treats it as one more model — no schema changes needed.

Why this exists: Open-Meteo's UKMO archive is incomplete pre-2024-09-01 (the
7-month all-NULL block we diagnosed 2026-04-26). The AWS Open Data bucket
ships the same underlying UM Global model on a 2-year rolling window and is
the only practical route to a clean historical UKMO baseline.

Caveats:
  * 06Z and 18Z cycles only run short-range (20h) — useless for 24/48/72h
    leads. Default cycles list is therefore [0, 12].
  * Only `radiation_flux_in_shortwave_direct_downward_at_surface` is published
    (no total or diffuse split). Mapped to ForecastRow.DirectRadiation; the
    radiation Element blender's `ShortwaveRadiation` column will stay NULL
    from this source.
  * Single grid cell extraction at Bonehill (50.5831N -3.7931W) — the cell
    centre lands at ~50.578N -3.727W (10×10 km), close enough for a tor
    forecast.
  * Each NetCDF is the full global grid (~3-10 MB). For a single cell that's
    >99.9% wasted bandwidth — fine for a one-off backfill, but a future
    kerchunk index would let us byte-range only the chunks we want.

Usage:
    .venv/Scripts/python.exe scripts/met_office_archive_backfill.py \\
        --start 2024-04-26 --end 2024-08-31 --cycles 0,12 --leads 24,48,72

Backfill the pre-Open-Meteo-archive window, the bit we actually need:
    .venv/Scripts/python.exe scripts/met_office_archive_backfill.py \\
        --start 2024-04-26 --end 2024-08-31
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
FORECASTS_ROOT = ROOT / "data" / "forecasts" / "location=bonehill_rocks" / "model=met_office_global"

BUCKET_URL = "https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com"
DATASET = "global-deterministic-10km"

# Bonehill Rocks, Dartmoor — same target site as the rest of WeatherBlend.
BONEHILL_LAT = 50.5831
BONEHILL_LON = -3.7931

LOCATION_NAME = "bonehill_rocks"
MODEL_ID = "met_office_global"


@dataclass(frozen=True)
class VarSpec:
    """One Met Office NetCDF variable + how it maps to a ForecastRow column.

    `convert` is applied to the raw numeric value before assignment. None means
    pass-through. xarray gives us back numpy scalars; we cast to float for parquet.
    """
    nc_variable_name: str   # filename suffix, e.g. "temperature_at_screen_level"
    nc_data_var: str        # the actual variable inside the NetCDF (CF standard name)
    forecast_row_col: str   # ForecastRow column name (PascalCase, matches C# property)
    convert: callable | None = None


# Map of Met Office variables → ForecastRow columns. Order is just for log readability.
VARIABLE_MAP: list[VarSpec] = [
    VarSpec("temperature_at_screen_level",                        "air_temperature",       "Temperature2m",      lambda x: x - 273.15),
    VarSpec("temperature_of_dew_point_at_screen_level",           "dew_point_temperature", "DewPoint2m",         lambda x: x - 273.15),
    VarSpec("relative_humidity_at_screen_level",                  "relative_humidity",     "RelativeHumidity2m", lambda x: x * 100.0),  # 0..1 → %
    VarSpec("precipitation_rate",                                 "precipitation_flux",    "Precipitation",      lambda x: x * 3600.0),  # kg/m²/s → mm/h
    VarSpec("wind_speed_at_10m",                                  "wind_speed",            "WindSpeed10m",       None),
    VarSpec("wind_direction_at_10m",                              "wind_from_direction",   "WindDirection10m",   None),
    VarSpec("wind_gust_at_10m",                                   "wind_speed_of_gust",    "WindGusts10m",       None),
    VarSpec("cloud_amount_of_total_cloud",                        "cloud_area_fraction",   "CloudCover",         lambda x: x * 100.0),  # 0..1 → %
    VarSpec("cloud_amount_of_low_cloud",                          "low_type_cloud_area_fraction",    "CloudCoverLow",  lambda x: x * 100.0),
    VarSpec("cloud_amount_of_medium_cloud",                       "medium_type_cloud_area_fraction", "CloudCoverMid",  lambda x: x * 100.0),
    VarSpec("cloud_amount_of_high_cloud",                         "high_type_cloud_area_fraction",   "CloudCoverHigh", lambda x: x * 100.0),
    VarSpec("CAPE_surface",                                       "atmosphere_convective_available_potential_energy", "Cape", None),
    VarSpec("visibility_at_screen_level",                         "visibility_in_air",     "Visibility",         None),
    VarSpec("radiation_flux_in_shortwave_direct_downward_at_surface", "surface_direct_downwelling_shortwave_flux_in_air", "DirectRadiation", None),
]


def _file_url(cycle_dt: datetime, lead_h: int, var: VarSpec) -> str:
    valid_dt = cycle_dt + pd.Timedelta(hours=lead_h)
    cycle_str = cycle_dt.strftime("%Y%m%dT%H%MZ")
    valid_str = valid_dt.strftime("%Y%m%dT%H%MZ")
    return f"{BUCKET_URL}/{DATASET}/{cycle_str}/{valid_str}-PT{lead_h:04d}H00M-{var.nc_variable_name}.nc"


def _extract_cell(nc_bytes: bytes, var: VarSpec, lat: float, lon: float) -> float | None:
    """Open a NetCDF blob in memory, find the data variable, return the
    nearest-cell value at (lat, lon). Returns None on any failure."""
    try:
        with xr.open_dataset(io.BytesIO(nc_bytes), engine="h5netcdf") as ds:
            # Met Office files store the data var under its CF standard name.
            # Some variables may use a slightly different name — fall back to
            # the largest 2D data var if the expected name isn't present.
            if var.nc_data_var in ds.data_vars:
                da = ds[var.nc_data_var]
            else:
                candidates = [v for v in ds.data_vars if ds[v].ndim >= 2 and ds[v].size > 1000]
                if not candidates:
                    return None
                da = ds[candidates[0]]
            sel = da.sel(latitude=lat, longitude=lon, method="nearest")
            v = float(sel.values)
            if not np.isfinite(v):
                return None
            return var.convert(v) if var.convert else v
    except Exception:
        return None


def _fetch_variable_for_lead(
    session: requests.Session, cycle_dt: datetime, lead_h: int, var: VarSpec,
    lat: float, lon: float,
) -> tuple[VarSpec, float | None]:
    url = _file_url(cycle_dt, lead_h, var)
    try:
        r = session.get(url, timeout=60)
        if r.status_code == 404:
            return (var, None)
        r.raise_for_status()
    except requests.RequestException:
        return (var, None)
    return (var, _extract_cell(r.content, var, lat, lon))


def fetch_cycle(
    cycle_dt: datetime, leads: list[int],
    variables: list[VarSpec],
    lat: float, lon: float,
    parallelism: int = 8,
    log: callable = print,
) -> list[dict]:
    """Pull every (lead, variable) for one cycle and assemble per-lead rows."""
    session = requests.Session()
    rows: dict[int, dict] = {}
    work = [(lead, var) for lead in leads for var in variables]

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        fut_to_key = {
            pool.submit(_fetch_variable_for_lead, session, cycle_dt, lead, var, lat, lon): (lead, var)
            for lead, var in work
        }
        for fut in as_completed(fut_to_key):
            lead, var = fut_to_key[fut]
            _, value = fut.result()
            row = rows.setdefault(lead, {})
            row[var.forecast_row_col] = value

    out: list[dict] = []
    for lead in sorted(rows):
        valid_dt = cycle_dt + pd.Timedelta(hours=lead)
        present = [c for c, v in rows[lead].items() if v is not None]
        if not present:
            log(f"    lead {lead}h: ALL variables missing, skipping")
            continue
        row = {
            "LocationName": LOCATION_NAME,
            "Model": MODEL_ID,
            "RunTimeUtc": cycle_dt,
            "ValidTimeUtc": valid_dt,
            "LeadHours": lead,
            "RunTimeSource": "exact",
            **rows[lead],
        }
        out.append(row)
        log(f"    lead {lead}h: {len(present)}/{len(variables)} vars (missing={len(variables)-len(present)})")
    return out


def write_cycle_parquet(rows: list[dict], cycle_dt: datetime, log: callable = print) -> Path:
    """Write the cycle's rows under WB's hive layout:
    data/forecasts/location=bonehill_rocks/model=met_office_global/date=YYYY-MM-DD/run=HH.parquet."""
    date_str = cycle_dt.strftime("%Y-%m-%d")
    run_dir = FORECASTS_ROOT / f"date={date_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"run={cycle_dt.hour:02d}.parquet"
    df = pd.DataFrame(rows)
    # Force timestamp dtype so Parquet stores as TIMESTAMP_MILLIS / TIMESTAMP_MICROS
    # rather than int64 epoch — keeps round-trip happy with WB's Parquet.NET reader.
    df["RunTimeUtc"] = pd.to_datetime(df["RunTimeUtc"], utc=True)
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"], utc=True)
    df.to_parquet(path, index=False)
    log(f"  wrote {len(df)} rows → {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end",   required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--cycles", default="0,12",
                    help="Comma-separated cycle hours. 06Z/18Z only run 20h and are not useful for 24/48/72h leads.")
    ap.add_argument("--leads", default="24,48,72",
                    help="Comma-separated lead hours")
    ap.add_argument("--parallelism", type=int, default=8,
                    help="Concurrent NetCDF downloads per cycle")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cycles = [int(c) for c in args.cycles.split(",")]
    leads = [int(l) for l in args.leads.split(",")]

    print(f"Met Office Global Det archive backfill")
    print(f"  range:    {args.start} → {args.end}")
    print(f"  cycles:   {cycles}")
    print(f"  leads:    {leads}")
    print(f"  vars:     {len(VARIABLE_MAP)}")
    print(f"  out root: {FORECASTS_ROOT}")

    n_cycles_done = 0
    n_cycles_failed = 0
    t_start = time.time()

    cur = start
    while cur <= end:
        for cc in cycles:
            cycle_dt = cur.replace(hour=cc, minute=0, second=0, microsecond=0)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] cycle {cycle_dt:%Y-%m-%d %H}Z")
            try:
                rows = fetch_cycle(cycle_dt, leads, VARIABLE_MAP, BONEHILL_LAT, BONEHILL_LON,
                                   parallelism=args.parallelism)
                if rows:
                    write_cycle_parquet(rows, cycle_dt)
                    n_cycles_done += 1
                else:
                    print("  no rows produced (every lead failed); skipping write")
                    n_cycles_failed += 1
            except Exception as e:
                print(f"  cycle FAILED: {e!r}")
                n_cycles_failed += 1
        cur += pd.Timedelta(days=1)

    dur = time.time() - t_start
    print(f"\nDone. cycles ok={n_cycles_done} failed={n_cycles_failed}  elapsed={dur:.1f}s")
    return 0 if n_cycles_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
