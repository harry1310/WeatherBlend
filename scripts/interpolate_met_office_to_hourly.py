"""Interpolate sparse Met Office (UKV / Global) parquet rows to hourly.

The AWS-direct Met Office pulls land at the cycle's native valid times only —
UKV at 03Z + 15Z, Global at 0Z + 12Z — leaving 22+ hours per day with no row
in the forecast tree. The blender's training pivot keys on
(ValidTimeUtc, LeadHours), so most rows end up with NaN for these models even
in `optional` mode. LightGBM gives the resulting feature ~0 split gain.

This script reads each existing model partition, groups by `LeadHours`, and
linearly interpolates each numeric column on a 1h grid spanning the data's
range. Results are written to a sibling partition named `<model>_hourly`,
preserving the raw cycle rows for any future use case that wants them.

What gets interpolated and what doesn't:
  * Continuous variables (temp, dewpoint, RH, pressure, cloud, wind speed,
    radiation): linear interp per LeadHours.
  * Wind direction: interpolated via sin/cos to handle wraparound.
  * Cumulative variables (Precipitation as mm/h is fine; rate, not accum,
    so linear interp is reasonable).
  * Identifiers (LocationName, Model, RunTimeUtc, RunTimeSource): the new
    rows carry Model = '<original>_hourly'; RunTimeUtc is set to the
    upstream cycle's run time when the interpolated valid time falls
    exactly on a sparse-row valid time, otherwise null (interpolated rows
    have no single "issuing cycle").

Intended usage:
    .venv/Scripts/python.exe scripts/interpolate_met_office_to_hourly.py \\
        --model met_office_ukv

The output sits at
data/forecasts/location=<loc>/model=met_office_ukv_hourly/date=YYYY-MM-DD/run=interp.parquet
(one parquet per UTC date, all leads in one file, run=interp marker).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FORECASTS_ROOT = ROOT / "data" / "forecasts" / "location=bonehill_rocks"

# Numeric columns we interpolate as plain linear-on-time. Anything not in
# this list AND not a special-case (wind direction, identifiers) is dropped.
LINEAR_COLS = [
    "Temperature2m", "DewPoint2m", "RelativeHumidity2m",
    "Precipitation", "WindSpeed10m", "WindGusts10m",
    "CloudCover", "CloudCoverLow", "CloudCoverMid", "CloudCoverHigh",
    "Cape", "Visibility",
    "DirectRadiation", "DiffuseRadiation", "ShortwaveRadiation",
    "MeanSeaLevelPressure", "SurfacePressure",
]


def interpolate_one_lead(group: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Given the sparse rows for a single LeadHours, return hourly rows
    spanning the same valid-time range, with each LINEAR_COLS column
    linearly interpolated. Wind direction handled via sin/cos.
    """
    g = group.sort_values("ValidTimeUtc").drop_duplicates("ValidTimeUtc", keep="last").copy()
    if len(g) < 2:
        # Need at least two anchors to interpolate; pass through what we have.
        return g

    g["ValidTimeUtc"] = pd.to_datetime(g["ValidTimeUtc"], utc=True)
    g = g.set_index("ValidTimeUtc").sort_index()

    # Hourly grid spanning min..max valid time.
    grid = pd.date_range(g.index.min().ceil("h"), g.index.max().floor("h"), freq="h", tz="UTC")
    out = pd.DataFrame(index=grid)

    for col in LINEAR_COLS:
        if col in g.columns:
            # pandas' interpolate('time') uses index timestamps as x-axis.
            combined = g[col].reindex(g.index.union(grid)).interpolate("time", limit_direction="both")
            out[col] = combined.reindex(grid)

    if "WindDirection10m" in g.columns:
        # Sin/cos interp avoids the 359 -> 0 wraparound discontinuity.
        rad = np.deg2rad(g["WindDirection10m"].astype(float))
        s = np.sin(rad).reindex(g.index.union(grid)).interpolate("time", limit_direction="both")
        c = np.cos(rad).reindex(g.index.union(grid)).interpolate("time", limit_direction="both")
        deg = np.rad2deg(np.arctan2(s.reindex(grid), c.reindex(grid))) % 360.0
        out["WindDirection10m"] = deg.values

    out = out.reset_index().rename(columns={"index": "ValidTimeUtc"})
    out["LeadHours"] = lead
    return out


def process_model(model_id: str, out_suffix: str = "_hourly") -> None:
    src_dir = FORECASTS_ROOT / f"model={model_id}"
    out_model = f"{model_id}{out_suffix}"
    dst_dir = FORECASTS_ROOT / f"model={out_model}"

    if not src_dir.exists():
        raise SystemExit(f"Source partition not found: {src_dir}")

    print(f"Reading sparse rows from {src_dir}")
    files = list(src_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquets under {src_dir}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"], utc=True)
    location = df["LocationName"].iloc[0]
    print(f"  loaded {len(df)} rows, leads = {sorted(df.LeadHours.unique().tolist())}")

    # Process each lead independently — the lead-24 series and lead-48 series
    # describe genuinely different forecasts and should never be mixed.
    out_rows: list[pd.DataFrame] = []
    for lead, sub in df.groupby("LeadHours"):
        interp = interpolate_one_lead(sub, int(lead))
        interp["LocationName"] = location
        interp["Model"] = out_model
        # New rows have no single issuing cycle — set RunTime to NaT and
        # source = "interp" so any downstream consumer can filter if needed.
        interp["RunTimeUtc"] = pd.NaT
        interp["RunTimeSource"] = "interp"
        out_rows.append(interp)
        print(f"  lead {lead}h: {len(sub)} sparse -> {len(interp)} hourly")

    full = pd.concat(out_rows, ignore_index=True)
    full = full.sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)

    # Write one parquet per UTC date. Matches the existing forecast tree's
    # date-partitioned layout so DuckDB / Parquet.NET readers behave identically.
    dst_dir.mkdir(parents=True, exist_ok=True)
    full["date_str"] = full["ValidTimeUtc"].dt.strftime("%Y-%m-%d")
    n_written = 0
    for date_str, day_df in full.groupby("date_str"):
        day_df = day_df.drop(columns=["date_str"])
        run_dir = dst_dir / f"date={date_str}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "run=interp.parquet"
        day_df["ValidTimeUtc"] = pd.to_datetime(day_df["ValidTimeUtc"], utc=True)
        # RunTimeUtc must be a real timestamp dtype even when all-null, else
        # Parquet.NET can't deserialise the column. NaT in datetime64 is fine.
        day_df["RunTimeUtc"] = pd.to_datetime(day_df["RunTimeUtc"], utc=True)
        day_df.to_parquet(path, index=False)
        n_written += 1
    print(f"Wrote {n_written} date partitions to {dst_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Source model id (e.g. met_office_ukv, met_office_global)")
    ap.add_argument("--out-suffix", default="_hourly",
                    help="Suffix for the new model id (default: _hourly)")
    args = ap.parse_args()

    process_model(args.model, args.out_suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
