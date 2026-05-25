"""
Compute per-(station × wind_sector × month) atmospheric climatology from the
pressure-level GFS history pulled by `pull_atm_pressure_levels.py`, and bake
into each location's static orographic JSON under data/static/orographic/.

Climatology features per (sector, month) bin:
  * lapse_850_500   — T_500 − T_850 (°C). More negative = unstable; positive = inversion.
  * lapse_700_500   — T_500 − T_700. Mid-tropospheric instability.
  * q_850           — typical specific humidity at 850 hPa (g/kg). Moisture proxy.
  * thickness_500_850 — geopotential difference (m). Larger = warmer column = more moisture capacity.
  * wind_500_speed  — typical mid-trop wind speed (m/s). Jet-stream proxy.
  * shear_850_500   — |wind_500 − wind_850| (m/s). Vertical shear.

Binning: by 850 hPa wind direction sector (matches the puller's variables;
sector definition mirrors `oro_static`'s 8-compass-bin convention). At predict
time the C# feature builder looks up by (NWP-mean surface wind sector × month);
the slight 850 hPa vs surface mismatch is acceptable for a climatological prior.

NULL handling: bins with < 100 samples get NaN values (the C# loader will
treat NaN as "no climatology available" — the model falls back to other features).

Idempotent: rewrites the per-station JSONs. Existing terrain / TPI / lee
fields are preserved; new `climatology_by_sector_month` field is added or
overwritten.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ATM_DIR = REPO_ROOT / "data" / "static" / "atm_history"
ORO_DIR = REPO_ROOT / "data" / "static" / "orographic"

# Match StationSlug + LocationConfig.Name. Stations alias to their parent
# location (same NWP-cell + atmospheric column).
STATION_TO_LOCATION = {
    "ea_bellever_dartmoor":       "bonehill_rocks",
    "ea_bovey_tracey":            "bonehill_rocks",
    "ea_dartmoor_nr_hexworthy":   "bonehill_rocks",
    "ea_princetown":              "bonehill_rocks",
    "ea_chards_snowdon_hill":     "membury_devon",
    "ea_goren":                   "membury_devon",
    "ea_raymonds_hill":           "membury_devon",
    "bonehill_rocks":             "bonehill_rocks",
    "membury_devon":              "membury_devon",
}

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
MIN_BIN_SAMPLES = 100

# Rough gas-constant for specific humidity from dewpoint + pressure.
RD_OVER_RV = 0.622


def sector_for_bearing(deg: float) -> str:
    """Map a 0-360 bearing to a compass sector. Matches OroStaticFeatures.UpwindGainAt
    in C#: nearest 45° bin starting at N."""
    if pd.isna(deg):
        return None  # type: ignore[return-value]
    deg = (deg % 360.0 + 360.0) % 360.0
    idx = int(math.floor((deg + 22.5) / 45.0)) % 8
    return SECTORS[idx]


def specific_humidity_g_per_kg(dew_c: pd.Series, p_hpa: float) -> pd.Series:
    """Magnus formula for saturation vapor pressure at dewpoint, then mixing-ratio
    formula. Returns g/kg."""
    es = 6.112 * np.exp(17.62 * dew_c / (dew_c + 243.12))
    q = RD_OVER_RV * es / (p_hpa - 0.378 * es)
    return q * 1000.0


def compute_location_climatology(location_name: str) -> dict:
    """Read all monthly parquets for a location, compute per-(sector, month)
    means of the climatology features. Returns nested dict keyed by sector."""
    loc_dir = ATM_DIR / f"location={location_name}"
    parquets = sorted(loc_dir.glob("*.parquet"))
    if not parquets:
        print(f"  no parquets under {loc_dir}, skipping")
        return {}

    frames = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(frames, ignore_index=True)
    print(f"  {location_name}: {len(df)} rows from {len(parquets)} months, "
          f"{df['time'].min()} -> {df['time'].max()}")

    # Derive sector + month
    df["sector"] = df["wind_direction_850hPa"].apply(sector_for_bearing)
    df["month"]  = df["time"].dt.month

    # Climatology features
    df["lapse_850_500"]   = df["temperature_500hPa"] - df["temperature_850hPa"]
    df["lapse_700_500"]   = df["temperature_500hPa"] - df["temperature_700hPa"]
    df["q_850"]           = specific_humidity_g_per_kg(df["dew_point_850hPa"], 850.0)
    df["wind_500_speed"]  = df["wind_speed_500hPa"]
    # Shear: simple magnitude of vector difference between 500 and 850 hPa wind.
    u_850 = -df["wind_speed_850hPa"] * np.sin(np.radians(df["wind_direction_850hPa"]))
    v_850 = -df["wind_speed_850hPa"] * np.cos(np.radians(df["wind_direction_850hPa"]))
    u_500 = -df["wind_speed_500hPa"] * np.sin(np.radians(df["wind_direction_500hPa"]))
    v_500 = -df["wind_speed_500hPa"] * np.cos(np.radians(df["wind_direction_500hPa"]))
    df["shear_850_500"]   = np.hypot(u_500 - u_850, v_500 - v_850)
    # Thickness — geopotential height proxy. We only have GPH_500; without GPH_850
    # the literal thickness can't be computed. Fall back to T_850 as a layer-mean
    # warmth proxy (warmer 850 ≈ more moisture-holding capacity in lower trop).
    df["thickness_proxy"] = df["temperature_850hPa"]

    feature_cols = [
        "lapse_850_500", "lapse_700_500", "q_850",
        "wind_500_speed", "shear_850_500", "thickness_proxy",
    ]

    # Bin → group → mean
    grouped = df.groupby(["sector", "month"])[feature_cols + ["time"]]
    out: dict = {}
    skipped_bins = 0
    for (sector, month), grp in grouped:
        if sector is None: continue
        n = len(grp)
        if n < MIN_BIN_SAMPLES:
            skipped_bins += 1
            continue
        bin_dict = {col: round(float(grp[col].mean()), 4) for col in feature_cols}
        bin_dict["n_samples"] = int(n)
        out.setdefault(sector, {})[str(int(month))] = bin_dict
    print(f"    bins emitted: {sum(len(v) for v in out.values())}; bins skipped (< {MIN_BIN_SAMPLES} samples): {skipped_bins}")
    return out


def main() -> None:
    # Compute once per LOCATION; stations alias to their parent location's climatology.
    by_location: dict[str, dict] = {}
    for loc in ("bonehill_rocks", "membury_devon"):
        print(f"\n=== {loc} ===")
        by_location[loc] = compute_location_climatology(loc)

    # Bake into each slug's JSON
    print("\n=== Writing climatology into per-slug JSONs ===")
    for slug, parent_loc in STATION_TO_LOCATION.items():
        json_path = ORO_DIR / f"{slug}.json"
        if not json_path.exists():
            print(f"  {slug}: no oro JSON found, skipping")
            continue
        d = json.loads(json_path.read_text())
        d["climatology_source"] = "gfs_seamless_atm_history_2022-06+_via_archive-api"
        d["climatology_by_sector_month"] = by_location.get(parent_loc, {})
        json_path.write_text(json.dumps(d, indent=2))
        n_bins = sum(len(v) for v in d["climatology_by_sector_month"].values())
        print(f"  {slug}: wrote {n_bins} bins")


if __name__ == "__main__":
    main()
