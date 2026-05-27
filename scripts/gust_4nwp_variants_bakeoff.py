"""Gust feature-variant bake-off on the 4-NWP production-with-gust scope.

Re-runs the variants from Temp/gust_bakeoff.py + gust_ratio_bakeoff.py but
restricted to the 4 Open-Meteo NWPs that have gust forecasts:
    gfs_seamless, icon_seamless, gem_seamless, ukmo_seamless.

The earlier numbers (0.98 - 1.04 MAE) were contaminated by archive NWPs
(gfs_ncep / met_office_global / met_office_ukv / ecmwf_ifs_oper) and
HARMONIE (KNMI/DMI). This run holds the NWP set constant at the
production scope so the variant comparison is honest.

Window: 2024-01-01 .. 2024-12-31, 70/15/15 chrono split. Truth: ERA5
WindGusts10m. Hyperparams match the original LightGBM
(n_estimators=600, lr=0.05, max_depth=6, num_leaves=31).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FCAST_GLOB = str(ROOT / "data/forecasts/location=bonehill_rocks/**/*.parquet").replace("\\", "/")
ERA5_GLOB = str(ROOT / "data/truth/era5/location=bonehill_rocks/**/*.parquet").replace("\\", "/")
ORO_PATH = ROOT / "data/static/orographic/bonehill_rocks.json"
START = "2024-01-01"
END = "2024-12-31 23:00:00"

GUST_NWPS = ("gfs_seamless", "icon_seamless", "gem_seamless", "ukmo_seamless")
nwps_in = "(" + ",".join(f"'{m}'" for m in GUST_NWPS) + ")"

# -- ORO setup (matches Temp/gust_bakeoff.py orographic block) ----------
oro = json.loads(ORO_PATH.read_text())
GRAD_DX, GRAD_DY = float(oro["terrain_gradient_dx"]), float(oro["terrain_gradient_dy"])
UPWIND_GAIN_5KM = oro["upwind_gain_5km"]
SECTOR_RAD = {
    "N": 0.0, "NE": math.pi / 4, "E": math.pi / 2, "SE": 3 * math.pi / 4,
    "S": math.pi, "SW": 5 * math.pi / 4, "W": 3 * math.pi / 2, "NW": 7 * math.pi / 4,
}

def upwind_gain_at(rad: float) -> float:
    if math.isnan(rad):
        return 0.0
    d = rad % (2 * math.pi)
    best, bd = "N", math.pi * 3
    for s, c in SECTOR_RAD.items():
        diff = min(abs(d - c), 2 * math.pi - abs(d - c))
        if diff < bd:
            bd, best = diff, s
    return float(UPWIND_GAIN_5KM[best])

def oro_dynamic(wsp: float, wd_sin: float, wd_cos: float, td: float, p: float) -> list[float]:
    rad = math.atan2(wd_sin, wd_cos) if not (math.isnan(wd_sin) or math.isnan(wd_cos)) else float("nan")
    gain = upwind_gain_at(rad)
    if math.isnan(wsp) or math.isnan(wd_sin) or math.isnan(wd_cos):
        uplift = 0.0
    else:
        uplift = max(0.0, (-wsp * wd_sin) * GRAD_DX + (-wsp * wd_cos) * GRAD_DY)
    if math.isnan(td) or math.isnan(p) or p <= 0:
        q = 0.0
    else:
        e = 6.112 * math.exp(17.62 * td / (td + 243.12))
        q = max(0.0, 0.622 * e / (p - 0.378 * e) * 1000.0)
    return [
        (0.0 if math.isnan(wd_sin) else wd_sin),
        (0.0 if math.isnan(wd_cos) else wd_cos),
        gain, uplift, uplift * q,
    ]

ORO_LEAN = ["oro_wind_sin", "oro_wind_cos", "oro_upwind_gain", "oro_uplift", "oro_uplift_x_q"]


# -- Data load (4-NWP only) ---------------------------------------------
print(f"[1/4] Loading data — 4 NWPs: {', '.join(GUST_NWPS)}")
con = duckdb.connect()
fc = con.execute(f"""
    WITH ranked AS (
        SELECT Model, ValidTimeUtc, WindSpeed10m, WindDirection10m, WindGusts10m,
               Temperature2m, DewPoint2m, SurfacePressure, CloudCover, RunTimeUtc,
               row_number() OVER (PARTITION BY Model, ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{FCAST_GLOB}', union_by_name=true)
        WHERE ValidTimeUtc >= TIMESTAMP '{START}'
          AND ValidTimeUtc <= TIMESTAMP '{END}'
          AND Model IN {nwps_in}
          AND WindGusts10m IS NOT NULL
          AND WindSpeed10m IS NOT NULL
    )
    SELECT * FROM ranked WHERE rn = 1
""").df()
print(f"   {len(fc):,} per-NWP rows ({fc.Model.nunique()} NWPs)")

era5 = con.execute(f"""
    SELECT ValidTimeUtc, WindGusts10m AS gust_era5
    FROM read_parquet('{ERA5_GLOB}', union_by_name=true)
    WHERE ValidTimeUtc >= TIMESTAMP '{START}'
      AND ValidTimeUtc <= TIMESTAMP '{END}'
""").df()
print(f"   ERA5 truth rows: {len(era5):,}")


def pivot(col: str) -> pd.DataFrame:
    p = fc.pivot(index="ValidTimeUtc", columns="Model", values=col).reset_index()
    p.columns = ["ValidTimeUtc"] + [f"{col}_{m}" for m in p.columns[1:]]
    return p

tabs = {n: pivot(n) for n in ("WindSpeed10m", "WindDirection10m", "WindGusts10m",
                              "Temperature2m", "DewPoint2m", "SurfacePressure", "CloudCover")}
df = tabs["WindSpeed10m"]
for k, t in tabs.items():
    if k != "WindSpeed10m":
        df = df.merge(t, on="ValidTimeUtc", how="inner")

wsp_cols = [c for c in tabs["WindSpeed10m"].columns if c != "ValidTimeUtc"]
wdir_cols = [c for c in tabs["WindDirection10m"].columns if c != "ValidTimeUtc"]
gust_cols = [c for c in tabs["WindGusts10m"].columns if c != "ValidTimeUtc"]
t_cols = [c for c in tabs["Temperature2m"].columns if c != "ValidTimeUtc"]
td_cols = [c for c in tabs["DewPoint2m"].columns if c != "ValidTimeUtc"]
p_cols = [c for c in tabs["SurfacePressure"].columns if c != "ValidTimeUtc"]
cc_cols = [c for c in tabs["CloudCover"].columns if c != "ValidTimeUtc"]
print(f"   Per-var col counts — gust:{len(gust_cols)} wsp:{len(wsp_cols)} "
      f"wdir:{len(wdir_cols)} t:{len(t_cols)} td:{len(td_cols)} p:{len(p_cols)} cc:{len(cc_cols)}")

# Cross-NWP means + ORO features (uses NWP-mean wsp/wdir/t/td/p for derivation)
df["wsp_xmean"] = df[wsp_cols].mean(axis=1, skipna=True)
df["wd_sin_xmean"] = np.nanmean(np.sin(np.radians(df[wdir_cols].values)), axis=1)
df["wd_cos_xmean"] = np.nanmean(np.cos(np.radians(df[wdir_cols].values)), axis=1)
df["td_xmean"] = df[td_cols].mean(axis=1, skipna=True)
df["p_xmean"] = df[p_cols].mean(axis=1, skipna=True)
arr = np.array([
    oro_dynamic(r.wsp_xmean, r.wd_sin_xmean, r.wd_cos_xmean, r.td_xmean, r.p_xmean)
    for r in df.itertuples()
])
for i, n in enumerate(ORO_LEAN):
    df[n] = arr[:, i]

# SPREAD features: mean + std for each of 6 vars (12 features total).
SPREAD: list[str] = []
for label, cols in [("wsp_spd", wsp_cols), ("gust_spd", gust_cols), ("t_spd", t_cols),
                    ("td_spd", td_cols), ("p_spd", p_cols), ("cc_spd", cc_cols)]:
    df[f"{label}_mean"] = df[cols].mean(axis=1, skipna=True)
    df[f"{label}_std"] = df[cols].std(axis=1, skipna=True)
    SPREAD.extend([f"{label}_mean", f"{label}_std"])

# Per-NWP gust ratio (gust / max(wsp, 0.5), clipped [0.5, 4.0])
RATIO_COLS: list[str] = []
for gc in gust_cols:
    nwp = gc.replace("WindGusts10m_", "")
    wc = f"WindSpeed10m_{nwp}"
    if wc not in df.columns:
        continue
    rc = f"gust_ratio_{nwp}"
    df[rc] = np.clip(df[gc].values / np.maximum(df[wc].values, 0.5), 0.5, 4.0)
    RATIO_COLS.append(rc)
df["gust_ratio_mean"] = df[RATIO_COLS].mean(axis=1, skipna=True)
df["gust_ratio_std"] = df[RATIO_COLS].std(axis=1, skipna=True)
RATIO_ALL = RATIO_COLS + ["gust_ratio_mean", "gust_ratio_std"]


# -- Split + ERA5 merge ------------------------------------------------
df = df.merge(era5, on="ValidTimeUtc", how="inner").dropna(subset=["gust_era5"])
n = len(df)
i_tr, i_va = int(n * 0.7), int(n * 0.85)
train, val, test = df.iloc[:i_tr].copy(), df.iloc[i_tr:i_va].copy(), df.iloc[i_va:].copy()
print(f"[2/4] {n:,} aligned rows -> train {len(train):,} / val {len(val):,} / test {len(test):,}")


# -- NWP baselines on test ---------------------------------------------
truth = test["gust_era5"].values
nwp_mean = test[gust_cols].mean(axis=1, skipna=True).values
nwp_mean_mae = float(np.mean(np.abs(nwp_mean - truth)))

per_nwp_mae: dict[str, float] = {}
for col in gust_cols:
    mask = test[col].notna()
    if mask.sum() < 100:
        continue
    mae = float(np.mean(np.abs(test.loc[mask, col].values - truth[mask])))
    per_nwp_mae[col.replace("WindGusts10m_", "")] = mae
best_single_name = min(per_nwp_mae, key=per_nwp_mae.get)
best_single_mae = per_nwp_mae[best_single_name]


# -- Variant runs ------------------------------------------------------
def fit_eval(label: str, feats: list[str]) -> tuple[float, float, list]:
    Xtr, ytr = train[feats].values, train["gust_era5"].values
    Xva, yva = val[feats].values, val["gust_era5"].values
    Xte, yte = test[feats].values, test["gust_era5"].values
    m = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_data_in_leaf=20, reg_alpha=0.1,
        reg_lambda=0.1, random_state=42, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = m.predict(Xte)
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    imp = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])
    print(f"   {label:<32s}  feats={len(feats):>3}  best={m.best_iteration_:>3}  "
          f"MAE={mae:.4f}  RMSE={rmse:.4f}")
    return mae, rmse, imp

print(f"[3/4] Variant bake-off (production-only, 4 NWPs)")
print(f"   NWP-mean baseline:                        MAE={nwp_mean_mae:.4f}")
print(f"   Best single NWP ({best_single_name}):                   MAE={best_single_mae:.4f}")
print()

variants = {
    "baseline (gust only)":      gust_cols,
    "+oro (gust + 5 oro)":       gust_cols + ORO_LEAN,
    "rich-no-oro (gust+wsp+wdir+spread)": gust_cols + wsp_cols + wdir_cols + SPREAD,
    "rich (incl. oro)":          gust_cols + wsp_cols + wdir_cols + ORO_LEAN + SPREAD,
    "rich + ratio":              gust_cols + wsp_cols + wdir_cols + ORO_LEAN + SPREAD + RATIO_ALL,
    "minimal (gust + ratio)":    gust_cols + RATIO_ALL,
}
results: dict[str, tuple[float, float, list]] = {}
for label, feats in variants.items():
    results[label] = fit_eval(label, feats)

print()
print("=" * 78)
print(f"[4/4] Summary — 4 OM NWPs, 2024 Bonehill, {len(test):,} test rows")
print("=" * 78)
print(f"  {'Variant':<36s}  {'feats':>5s}  {'MAE':>7s}  {'Δ vs NWP-mean':>14s}  {'Δ vs minimal':>13s}")
minimal_mae = results["minimal (gust + ratio)"][0]
for label, (mae, _, _) in results.items():
    delta_nwp = 100 * (mae - nwp_mean_mae) / nwp_mean_mae
    delta_min = 100 * (mae - minimal_mae) / minimal_mae
    print(f"  {label:<36s}  {len(variants[label]):>5d}  {mae:>7.4f}  {delta_nwp:>+12.2f}%  {delta_min:>+11.2f}%")

# Top-15 features of the best variant
winner = min(results, key=lambda k: results[k][0])
print()
print(f"  Winner: {winner}  (MAE {results[winner][0]:.4f})")
print(f"  Top-15 features:")
for rank, (name, gain) in enumerate(results[winner][2][:15], 1):
    print(f"    rank {rank:>2}  {name:<42s}  imp={gain}")
