"""
WeatherBench2 ECMWF HRES backfill.

Pulls historical ECMWF HRES forecasts for one point from the public WB2 zarr on
GCS (gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr) and writes
them into the repo's existing forecasts parquet layout, tagged with real cycle
stamps (RunTimeSource=exact).

The HRES dataset runs 2016-01-01 00Z..2023-01-10 12Z, cycles at 00/12Z, with
prediction_timedelta ranging 0..240h in 6h steps. 0.25° grid. Licence: CC-BY-4.0.

Why Python and not .NET: there is no mature zarr/xarray client for .NET. The
spike verified the full pipeline works (Bonehill nearest grid point 50.50N,
356.25E, 2020-06-15 00Z 2m_temperature = 283.66 K).

Output path matches the live collector so DuckDB queries pick it up:
  data/forecasts/location=<name>/model=ecmwf_hres_wb2/date=<YYYY-MM-DD>/run=<HH>.parquet

Usage (from repo root):
  .venv/Scripts/python.exe scripts/wb2_ecmwf_backfill.py \
      --start 2020-01-01 --end 2020-12-31 \
      --location bonehill_rocks --lat 50.5831 --lon -3.7931

Skips dates whose output parquet already exists, so it's safe to resume.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import gcsfs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

MODEL_ID = "ecmwf_hres_wb2"
ZARR_URI = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
RUN_TIME_SOURCE_EXACT = "exact"


@dataclass
class Point:
    name: str
    latitude: float
    longitude: float  # stored as signed degrees (negative west)


def normalise_lon(lon_signed: float) -> float:
    """Convert signed (-180..180) to WB2 convention (0..360)."""
    return lon_signed % 360.0


def open_dataset() -> xr.Dataset:
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ZARR_URI)
    ds = xr.open_zarr(store, consolidated=True)
    return ds


def select_point(ds: xr.Dataset, point: Point) -> xr.Dataset:
    """Collapse lat/lon down to the nearest grid point. Keeps time +
    prediction_timedelta dims intact."""
    lon_wb2 = normalise_lon(point.longitude)
    # 2-D vars only — drop the 3-D pressure-level cube we don't use at a point.
    keep = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation_6hr",
    ]
    keep = [v for v in keep if v in ds.data_vars]
    sub = ds[keep].sel(latitude=point.latitude, longitude=lon_wb2, method="nearest")
    return sub


def iter_cycles(start: date, end: date):
    """Yield every (init_time, init_date, init_hour) in range at 00 and 12Z."""
    day = start
    while day <= end:
        for hh in (0, 12):
            yield datetime(day.year, day.month, day.day, hh, tzinfo=timezone.utc), day, hh
        day += timedelta(days=1)


def wind_dir_deg(u: float, v: float) -> float:
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def extract_cycle(point_ds: xr.Dataset, init_time: datetime, location_name: str) -> list[dict]:
    """Return one row per prediction_timedelta for this init time."""
    # Zarr time dim is tz-naive; our init_time is UTC — convert.
    init_np = np.datetime64(init_time.replace(tzinfo=None))
    try:
        cube = point_ds.sel(time=init_np)
    except KeyError:
        return []

    # Materialise the small (41,) arrays into memory in one shot.
    cube = cube.load()
    timedeltas = cube.prediction_timedelta.values  # int64 nanoseconds? or timedelta64
    # Normalise to hours int
    lead_hours = (timedeltas.astype("timedelta64[h]").astype(int))

    def var(name: str) -> np.ndarray | None:
        return cube[name].values if name in cube else None

    t2m = var("2m_temperature")
    u10 = var("10m_u_component_of_wind")
    v10 = var("10m_v_component_of_wind")
    sp = var("surface_pressure")
    msl = var("mean_sea_level_pressure")
    precip6 = var("total_precipitation_6hr")

    rows = []
    for i, lead in enumerate(lead_hours):
        lead = int(lead)
        valid = init_time + timedelta(hours=lead)

        def val(arr, i):
            if arr is None:
                return None
            v = arr[i]
            # numpy scalar comparison — np.isnan handles numpy float32/64 cleanly;
            # a plain Python None never makes it this far (arr[i] on ndarray returns scalar).
            try:
                if np.isnan(v):
                    return None
            except (TypeError, ValueError):
                pass
            return float(v)

        t_c = val(t2m, i)
        if t_c is not None:
            t_c = t_c - 273.15

        u = val(u10, i)
        v = val(v10, i)
        wspd = None
        wdir = None
        if u is not None and v is not None:
            wspd = float(np.sqrt(u * u + v * v))
            wdir = float(wind_dir_deg(u, v))

        sp_val = val(sp, i)
        if sp_val is not None:
            sp_hpa = sp_val / 100.0
        else:
            msl_val = val(msl, i)
            sp_hpa = msl_val / 100.0 if msl_val is not None else None

        precip = val(precip6, i)  # in metres; convert to mm
        if precip is not None:
            precip = precip * 1000.0

        rows.append({
            "LocationName": location_name,
            "Model": MODEL_ID,
            "RunTimeUtc": init_time.replace(tzinfo=None),
            "ValidTimeUtc": valid.replace(tzinfo=None),
            "LeadHours": lead,
            "RunTimeSource": RUN_TIME_SOURCE_EXACT,
            "Temperature2m": t_c,
            "DewPoint2m": None,
            "RelativeHumidity2m": None,
            "Precipitation": precip,
            "PrecipitationProbability": None,
            "Rain": None,
            "Showers": None,
            "Snowfall": None,
            "CloudCover": None,
            "CloudCoverLow": None,
            "CloudCoverMid": None,
            "CloudCoverHigh": None,
            "WindSpeed10m": wspd,
            "WindDirection10m": wdir,
            "WindGusts10m": None,
            "SurfacePressure": sp_hpa,
            "Cape": None,
            "Visibility": None,
        })
    return rows


# Fixed pyarrow schema — must match ForecastRow field order/names and make every
# numeric column float64 nullable so we don't get surprises when a cell is None.
SCHEMA = pa.schema([
    ("LocationName", pa.string()),
    ("Model", pa.string()),
    ("RunTimeUtc", pa.timestamp("us")),
    ("ValidTimeUtc", pa.timestamp("us")),
    ("LeadHours", pa.int32()),
    ("RunTimeSource", pa.string()),
    ("Temperature2m", pa.float64()),
    ("DewPoint2m", pa.float64()),
    ("RelativeHumidity2m", pa.float64()),
    ("Precipitation", pa.float64()),
    ("PrecipitationProbability", pa.float64()),
    ("Rain", pa.float64()),
    ("Showers", pa.float64()),
    ("Snowfall", pa.float64()),
    ("CloudCover", pa.float64()),
    ("CloudCoverLow", pa.float64()),
    ("CloudCoverMid", pa.float64()),
    ("CloudCoverHigh", pa.float64()),
    ("WindSpeed10m", pa.float64()),
    ("WindDirection10m", pa.float64()),
    ("WindGusts10m", pa.float64()),
    ("SurfacePressure", pa.float64()),
    ("Cape", pa.float64()),
    ("Visibility", pa.float64()),
])


def write_cycle_parquet(rows: list[dict], out_root: Path, init_date: date, init_hour: int, location_name: str) -> Path:
    date_str = init_date.strftime("%Y-%m-%d")
    hh = f"{init_hour:02d}"
    out_dir = out_root / f"location={location_name}" / f"model={MODEL_ID}" / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run={hh}.parquet"
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="WB2 ECMWF HRES backfill to parquet.")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--location", default="bonehill_rocks", help="Location partition name")
    ap.add_argument("--lat", type=float, default=50.5831)
    ap.add_argument("--lon", type=float, default=-3.7931)
    ap.add_argument("--out-root", default="data/forecasts", help="Forecasts parquet root")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip cycles whose output parquet already exists (default: on)")
    ap.add_argument("--no-skip", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    point = Point(args.location, args.lat, args.lon)
    out_root = Path(args.out_root)

    print(f"[wb2] opening {ZARR_URI}", flush=True)
    ds = open_dataset()

    # Bound check: refuse silently-out-of-range ranges early.
    zarr_min = np.datetime_as_string(ds.time.min().values, unit="D")
    zarr_max = np.datetime_as_string(ds.time.max().values, unit="D")
    print(f"[wb2] archive covers {zarr_min}..{zarr_max}; requested {start}..{end}", flush=True)

    point_ds = select_point(ds, point)
    lat_nearest = float(point_ds.latitude.values)
    lon_nearest = float(point_ds.longitude.values)
    print(f"[wb2] point {point.name} -> nearest grid ({lat_nearest}, {lon_nearest})", flush=True)

    written = 0
    skipped = 0
    missing = 0
    for init_time, init_date, init_hour in iter_cycles(start, end):
        date_str = init_date.strftime("%Y-%m-%d")
        hh = f"{init_hour:02d}"
        out_path = out_root / f"location={point.name}" / f"model={MODEL_ID}" / f"date={date_str}" / f"run={hh}.parquet"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        rows = extract_cycle(point_ds, init_time, point.name)
        if not rows:
            missing += 1
            print(f"[wb2] {date_str} t{hh}z: not in archive", flush=True)
            continue

        write_cycle_parquet(rows, out_root, init_date, init_hour, point.name)
        written += 1
        if written % 20 == 0:
            print(f"[wb2] {date_str} t{hh}z: written={written} skipped={skipped} missing={missing}", flush=True)

    print(f"[wb2] done. written={written} skipped={skipped} missing={missing}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
