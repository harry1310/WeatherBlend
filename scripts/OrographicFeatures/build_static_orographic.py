"""
Build per-station + per-location static orographic features from SRTM.

Reads the CGIAR SRTM v4 90m tile (srtm_36_02 — covers SW England + East Devon)
at WB/data/static/dem/srtm_36_02.tif and computes, for every EA rainfall
station configured under any location in config.yaml plus every location
centroid:

  - site_elevation_m
  - nwp_cell_elevation_m            (mean elev in ±4.5 km box ~ 9 km cell)
  - elevation_vs_cell_m             (site - cell mean)
  - relief_{5,10,25}km_m            (max - min within radius)
  - terrain_ruggedness_5km_m        (mean |elev - site| within 5 km)
  - upwind_gain_{5,10}km            (8-sector lookup: mean elev gain over
                                     the upwind arc in each compass sector)
  - terrain_gradient_dx, _dy        (rise/run at the site, m per m east/north;
                                     for the predict-time orographic_uplift
                                     feature uplift = max(0, u·dx + v·dy))

Outputs one JSON per entity at data/static/orographic/{slug}.json.

Slug convention matches StationSlug.cs:
  - Stations: 'ea_' + lower-snake of station name
  - Locations: location.name from config.yaml

Run from the WeatherBlend repo root:
  python scripts/OrographicFeatures/build_static_orographic.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEM_PATH = REPO_ROOT / "data" / "static" / "dem" / "srtm_36_02.tif"
OUT_DIR = REPO_ROOT / "data" / "static" / "orographic"

# SRTM tile georeferencing (from srtm_36_02.tfw + .hdr).
# 5°×5° tile, top-left pixel CENTER at (-4.9996, 54.9996), 0.000833° per pixel.
TILE_LON_ORIGIN = -4.9995833333  # longitude of pixel-(0,0) centre
TILE_LAT_ORIGIN = 54.9995833333  # latitude  of pixel-(0,0) centre
PIXEL_DEG = 0.0008333333          # 3 arc-seconds

# Sites hard-coded for v1 — small and stable. If config.yaml grows new
# stations/locations the build script will need updating; that's intentional
# (it's a once-ish offline build, not a runtime path). Station coords from the
# EA Hydrology /measures endpoint (canonical, queried 2026-05-24).
STATIONS = [
    ("ea_bellever_dartmoor",       50.582553, -3.898638),
    ("ea_bovey_tracey",            50.592312, -3.716672),
    ("ea_dartmoor_nr_hexworthy",   50.548615, -3.938746),
    ("ea_princetown",              50.549567, -3.999400),
    ("ea_chards_snowdon_hill",     50.875716, -2.981262),
    ("ea_goren",                   50.816230, -3.085698),
    ("ea_raymonds_hill",           50.767425, -2.964048),
]
LOCATIONS = [
    ("bonehill_rocks",  50.5831, -3.7931),
    ("membury_devon",   50.8254, -3.0000),
]

# Pixel scale in metres at our latitudes (~50.7°N).
# 1° lat ≈ 111.32 km; 1° lon ≈ 111.32 km × cos(lat).
LAT_M_PER_DEG = 111_320.0
def lon_m_per_deg(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))

# Wind sector definitions (centre bearings, degrees from north, clockwise).
SECTORS = [
    ("N",  0.0),
    ("NE", 45.0),
    ("E",  90.0),
    ("SE", 135.0),
    ("S",  180.0),
    ("SW", 225.0),
    ("W",  270.0),
    ("NW", 315.0),
]
SECTOR_HALF_WIDTH_DEG = 22.5  # 45° bin, ±22.5°

# Window radius for everything (so we read one square subset of the DEM per
# site and derive all features from it).
MAX_RADIUS_KM = 25.0
HALF_NWP_CELL_KM = 4.5  # ±4.5 km box ≈ 9 km cell

# ---------------------------------------------------------------------------
# DEM access
# ---------------------------------------------------------------------------

SRTM_NODATA_SENTINEL = -32768  # CGIAR SRTM v4 over sea / voids

def load_dem() -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(DEM_PATH)
    arr = np.asarray(im, dtype=np.int32)
    if arr.shape != (6000, 6000):
        raise RuntimeError(f"unexpected DEM shape: {arr.shape}")
    return arr

def _valid_mask(window: np.ndarray) -> np.ndarray:
    """True for cells whose elevation should be trusted. SRTM uses
    -32768 for sea / void; treat anything below -500 m as nodata to
    catch both that sentinel and any future void-fill artefacts."""
    return window > -500

def latlon_to_px(lat: float, lon: float) -> tuple[int, int]:
    col = round((lon - TILE_LON_ORIGIN) / PIXEL_DEG)
    row = round((TILE_LAT_ORIGIN - lat) / PIXEL_DEG)
    return col, row

# ---------------------------------------------------------------------------
# Per-site feature derivation
# ---------------------------------------------------------------------------

def compute_features(dem: np.ndarray, lat: float, lon: float) -> dict:
    col, row = latlon_to_px(lat, lon)
    site_elev = int(dem[row, col])

    # Subset window: enough pixels for MAX_RADIUS_KM in every direction.
    # Use the worst-case (latitude) pixel size for radius -> half-window.
    px_per_km_lat = 1000.0 / (PIXEL_DEG * LAT_M_PER_DEG)
    px_per_km_lon = 1000.0 / (PIXEL_DEG * lon_m_per_deg(lat))
    half_px_lat = int(math.ceil(MAX_RADIUS_KM * px_per_km_lat))
    half_px_lon = int(math.ceil(MAX_RADIUS_KM * px_per_km_lon))

    r0 = max(0, row - half_px_lat); r1 = min(dem.shape[0], row + half_px_lat + 1)
    c0 = max(0, col - half_px_lon); c1 = min(dem.shape[1], col + half_px_lon + 1)
    window = dem[r0:r1, c0:c1].astype(np.float32)

    # Mesh of distance + bearing from the site for every pixel in the window.
    rows = np.arange(r0, r1) - row  # northward = negative row offset
    cols = np.arange(c0, c1) - col  # eastward  = positive col offset
    drow, dcol = np.meshgrid(rows, cols, indexing="ij")
    dn_m = (-drow) * (PIXEL_DEG * LAT_M_PER_DEG)        # metres north
    de_m = ( dcol) * (PIXEL_DEG * lon_m_per_deg(lat))   # metres east
    dist_m = np.hypot(dn_m, de_m)
    dist_km = dist_m / 1000.0

    # Bearing FROM site TO pixel, degrees clockwise from north.
    bearing_deg = (np.degrees(np.arctan2(de_m, dn_m)) + 360.0) % 360.0

    valid = _valid_mask(window)
    sea_pct = round(100.0 * float((~valid).sum()) / valid.size, 1)

    feats: dict = {
        "site_elevation_m": site_elev,
        "lat": lat,
        "lon": lon,
        "window_sea_pct": sea_pct,  # fraction of 25 km window that is sea/void
    }

    # NWP cell mean (±4.5 km square box — approximation of a 9 km grid cell).
    # Sea cells are treated as nodata, NOT zero, because zero would bias the
    # cell mean downward for coastal sites. NWP grids do average over sea so
    # this is an approximation; a future refinement could set sea = 0 m if a
    # bake-off says coastal blenders need that.
    mask_cell = (np.abs(dn_m) <= HALF_NWP_CELL_KM * 1000.0) & \
                (np.abs(de_m) <= HALF_NWP_CELL_KM * 1000.0) & valid
    cell_elev = float(window[mask_cell].mean()) if mask_cell.any() else float(site_elev)
    feats["nwp_cell_elevation_m"] = round(cell_elev, 1)
    feats["elevation_vs_cell_m"] = round(site_elev - cell_elev, 1)

    # Relief and ruggedness at multiple radii.
    for radius_km in (5.0, 10.0, 25.0):
        mask = (dist_km <= radius_km) & (dist_km > 0) & valid
        sub = window[mask]
        if sub.size == 0:
            feats[f"relief_{int(radius_km)}km_m"] = 0.0
        else:
            feats[f"relief_{int(radius_km)}km_m"] = round(float(sub.max() - sub.min()), 1)
    mask5 = (dist_km <= 5.0) & (dist_km > 0) & valid
    sub5 = window[mask5]
    feats["terrain_ruggedness_5km_m"] = (
        round(float(np.abs(sub5 - site_elev).mean()), 1) if sub5.size else 0.0
    )

    # Upwind gain per sector at 5 km and 10 km. "Upwind" sector S means the
    # arc of pixels lying in bearing band [S-22.5, S+22.5] from the site (i.e.
    # the wind comes FROM that direction; the upwind terrain sits there).
    # Sea cells excluded — "upwind sea" sets the sector's land-elevation gain
    # to zero in the limit of no land in that arc, which is the correct
    # behaviour for the orographic feature.
    for radius_km in (5.0, 10.0):
        sector_gain = {}
        for name, centre in SECTORS:
            lo = (centre - SECTOR_HALF_WIDTH_DEG) % 360.0
            hi = (centre + SECTOR_HALF_WIDTH_DEG) % 360.0
            if lo < hi:
                arc_mask = (bearing_deg >= lo) & (bearing_deg < hi)
            else:  # wrap around 0/360 (sector N)
                arc_mask = (bearing_deg >= lo) | (bearing_deg < hi)
            mask = arc_mask & (dist_km <= radius_km) & (dist_km > 0) & valid
            sub = window[mask]
            if sub.size == 0:
                sector_gain[name] = 0.0
            else:
                sector_gain[name] = round(float(sub.mean() - site_elev), 1)
        feats[f"upwind_gain_{int(radius_km)}km"] = sector_gain

    # Local terrain gradient at the site: rise/run east (dx) and north (dy),
    # m per m. Computed by central difference over a ±~500 m window (6 pixels
    # at this latitude) — smooths over single-pixel noise but stays local.
    half_px = 6
    east_elev = float(window[row - r0,  col - c0 + half_px])
    west_elev = float(window[row - r0,  col - c0 - half_px])
    north_elev = float(window[row - r0 - half_px, col - c0])
    south_elev = float(window[row - r0 + half_px, col - c0])
    de_m_grad = 2 * half_px * PIXEL_DEG * lon_m_per_deg(lat)
    dn_m_grad = 2 * half_px * PIXEL_DEG * LAT_M_PER_DEG
    feats["terrain_gradient_dx"] = round((east_elev - west_elev) / de_m_grad, 6)
    feats["terrain_gradient_dy"] = round((north_elev - south_elev) / dn_m_grad, 6)

    # =====================================================================
    # v2 DEM aggregations (added 2026-05-25, drives PrecipRichOroV2)
    # =====================================================================

    # Topographic Position Index at multiple scales. TPI = site_elev - mean(elev)
    # over each radius. Positive = local high point; negative = local hollow.
    # Captures the multi-scale topographic context that a single elevation_vs_cell
    # collapses — e.g. Bonehill is TPI ≫ 0 at 200 m (granite tor) but TPI ≈ 0
    # at 5 km (sits on high plateau), distinguishing it from a "high plateau
    # everywhere" site like Princetown which is TPI > 0 at every scale.
    for radius_km in (0.2, 1.0, 5.0, 25.0):
        mask = (dist_km <= radius_km) & valid
        sub = window[mask]
        if sub.size == 0:
            feats[f"tpi_{int(radius_km * 1000)}m"] = 0.0
        else:
            feats[f"tpi_{int(radius_km * 1000)}m"] = round(float(site_elev - sub.mean()), 1)

    # Lee / rain-shadow detector per sector. For each upwind sector, find the
    # MAXIMUM elevation along a ±22.5° arc within 10 km, and report
    # max_upwind − site_elev. Large positive = a ridge crest sits between the
    # site and the upwind boundary → site is partially shielded (rain-shadow).
    # Combined with mean upwind_gain (already in v1), the model can distinguish
    # "continuous upslope" (enhancement) from "ridge then drop" (lee effect).
    lee = {}
    for name, centre in SECTORS:
        lo = (centre - SECTOR_HALF_WIDTH_DEG) % 360.0
        hi = (centre + SECTOR_HALF_WIDTH_DEG) % 360.0
        if lo < hi:
            arc_mask = (bearing_deg >= lo) & (bearing_deg < hi)
        else:
            arc_mask = (bearing_deg >= lo) | (bearing_deg < hi)
        mask = arc_mask & (dist_km <= 10.0) & (dist_km > 0) & valid
        sub = window[mask]
        if sub.size == 0:
            lee[name] = 0.0
        else:
            lee[name] = round(float(sub.max() - site_elev), 1)
    feats["lee_obstruction_10km"] = lee

    # Mean slope + aspect dominance within 5 km. Aggregates per-pixel slope and
    # aspect over the surrounding terrain — gives the model regional terrain
    # character, distinct from the single-point site gradient already exposed.
    #
    # Compute per-pixel slope (m/m) and aspect (radians, 0=east, π/2=north)
    # via np.gradient on the window. Aspect dominance = magnitude of the
    # vector-mean of unit-aspect vectors over valid 5 km pixels — near 1 =
    # terrain mostly faces one direction (coherent slope), near 0 = omnidirectional
    # (peak / basin / complex). Steepness-weighted so flat pixels don't dilute.
    px_size_east_m  = PIXEL_DEG * lon_m_per_deg(lat)
    px_size_north_m = PIXEL_DEG * LAT_M_PER_DEG
    # np.gradient over axis 0 (row) gives δelev per row going SOUTH; flip sign for north.
    dh_dy = -np.gradient(window, axis=0) / px_size_north_m
    dh_dx =  np.gradient(window, axis=1) / px_size_east_m
    slope_per_px = np.hypot(dh_dx, dh_dy)        # m/m
    aspect_per_px = np.arctan2(dh_dy, dh_dx)     # radians, 0=east, π/2=north

    mask5 = (dist_km <= 5.0) & valid
    s5 = slope_per_px[mask5]
    a5 = aspect_per_px[mask5]
    if s5.size == 0:
        feats["mean_slope_5km"] = 0.0
        feats["aspect_dominance_5km"] = 0.0
    else:
        feats["mean_slope_5km"] = round(float(s5.mean()), 6)
        # Slope-weighted unit-aspect mean: flat-ground pixels contribute little.
        w = s5
        if w.sum() == 0:
            feats["aspect_dominance_5km"] = 0.0
        else:
            mean_cos = float((np.cos(a5) * w).sum() / w.sum())
            mean_sin = float((np.sin(a5) * w).sum() / w.sum())
            feats["aspect_dominance_5km"] = round(math.hypot(mean_cos, mean_sin), 4)

    return feats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"loading DEM from {DEM_PATH}")
    dem = load_dem()
    print(f"DEM shape={dem.shape}, range=[{dem.min()},{dem.max()}] m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [("station", *t) for t in STATIONS] + [("location", *t) for t in LOCATIONS]
    for kind, slug, lat, lon in targets:
        feats = compute_features(dem, lat, lon)
        feats = {
            "slug": slug,
            "kind": kind,
            "dem_source": "cgiar_srtm_v4_90m_srtm_36_02",
            **feats,
        }
        out_path = OUT_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(feats, indent=2))
        print(f"  wrote {out_path.name}: site={feats['site_elevation_m']}m  "
              f"vs_cell={feats['elevation_vs_cell_m']:+.1f}m  "
              f"relief5={feats['relief_5km_m']:.0f}m  "
              f"rugg5={feats['terrain_ruggedness_5km_m']:.0f}m")
    print(f"\n{len(targets)} files written under {OUT_DIR}")

if __name__ == "__main__":
    main()
