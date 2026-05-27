"""Gust bake-off restricted to the 8 production NWPs (Nwp.BlenderModelIds).

Two things this answers, which the Temp/gust_ratio_bakeoff.py couldn't:
  1. Hourly vs 6-hourly cycle counts per NWP (the "data shape is different"
     concern) — counts rows/day in 2024 per Model, plus distinct cycle hours.
  2. Production-scope MAE for the minimal (gust + ratio + spread) variant.
     Same algorithm as gust_ratio_bakeoff.py, but with Model IN (production-only).
"""
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
START = "2024-01-01"
END = "2024-12-31 23:00:00"

PROD_NWPS = (
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless",
    "ukmo_seamless", "gem_seamless", "ecmwf_aifs025_single", "jma_seamless",
)
# Production + HARMONIE pair (KNMI + DMI). Both collected via Open-Meteo
# Previous Runs (hourly cadence), backfilled 2026-04-28, not yet wired into
# any blender. Worth re-testing for gust specifically — the 2026-04-28
# rejection was scoped to precip + dry-window targets.
PROD_PLUS_HARMONIE = PROD_NWPS + (
    "knmi_harmonie_arome_europe", "dmi_harmonie_arome_europe",
)
# Production + the 4 archive NWPs that drove the original bake-off's
# 0.9808 win. These are 6-hourly (or sparser) so LightGBM sees them as
# NaN at most rows — but where they ARE present, they may add gust-factor
# regime info the 4 production hourly NWPs don't carry. User's framing:
# "once every 6 hours we'd be more accurate". The deployment cost is
# resurrecting the archive collectors (currently removed) — only worth
# it if MAE moves meaningfully.
PROD_PLUS_ARCHIVES = PROD_NWPS + (
    "met_office_global", "met_office_ukv", "gfs_ncep", "ecmwf_ifs_oper",
)

con = duckdb.connect()

print("=" * 78)
print("[1/3] Cycle frequency per NWP (2024)")
print("=" * 78)
shape_q = f"""
SELECT Model,
       COUNT(DISTINCT ValidTimeUtc) AS distinct_valids,
       COUNT(DISTINCT date_trunc('day', ValidTimeUtc)) AS distinct_days,
       ROUND(COUNT(DISTINCT ValidTimeUtc)::DOUBLE /
             NULLIF(COUNT(DISTINCT date_trunc('day', ValidTimeUtc)), 0), 2) AS valids_per_day,
       COUNT(DISTINCT RunTimeUtc) AS distinct_cycles,
       COUNT(WindGusts10m) AS rows_with_gust,
       COUNT(WindSpeed10m) AS rows_with_wsp
FROM read_parquet('{FCAST_GLOB}', union_by_name=true)
WHERE ValidTimeUtc >= TIMESTAMP '{START}'
  AND ValidTimeUtc <= TIMESTAMP '{END}'
GROUP BY Model
HAVING COUNT(WindGusts10m) > 0
ORDER BY rows_with_gust DESC
"""
shape = con.execute(shape_q).df()
shape["in_prod"] = shape["Model"].isin(PROD_NWPS).map({True: "PROD", False: "ext "})
print(shape.to_string(index=False))

print()
print("=" * 78)
print("[2/3] Production-scope (8 NWPs) gust+wsp coverage in 2024")
print("=" * 78)
prod_in = "(" + ",".join(f"'{m}'" for m in PROD_PLUS_ARCHIVES) + ")"
prod_q = f"""
SELECT Model,
       COUNT(*) AS rows_total,
       COUNT(WindGusts10m) AS rows_with_gust,
       COUNT(WindSpeed10m) AS rows_with_wsp
FROM read_parquet('{FCAST_GLOB}', union_by_name=true)
WHERE ValidTimeUtc >= TIMESTAMP '{START}'
  AND ValidTimeUtc <= TIMESTAMP '{END}'
  AND Model IN {prod_in}
GROUP BY Model
ORDER BY Model
"""
print(con.execute(prod_q).df().to_string(index=False))

print()
print("=" * 78)
print("[3/3] Production-scope minimal bake-off")
print("=" * 78)
fc = con.execute(f"""
    WITH ranked AS (
        SELECT Model, ValidTimeUtc, WindSpeed10m, WindGusts10m, RunTimeUtc,
               row_number() OVER (PARTITION BY Model, ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{FCAST_GLOB}', union_by_name=true)
        WHERE ValidTimeUtc >= TIMESTAMP '{START}'
          AND ValidTimeUtc <= TIMESTAMP '{END}'
          AND Model IN {prod_in}
          AND WindGusts10m IS NOT NULL AND WindSpeed10m IS NOT NULL
    )
    SELECT * FROM ranked WHERE rn = 1
""").df()
print(f"   {len(fc):,} per-NWP rows ({fc.Model.nunique()} NWPs with gust)")

def pivot(col: str) -> pd.DataFrame:
    p = fc.pivot(index="ValidTimeUtc", columns="Model", values=col).reset_index()
    p.columns = ["ValidTimeUtc"] + [f"{col}_{m}" for m in p.columns[1:]]
    return p

wsp_df = pivot("WindSpeed10m")
gust_df = pivot("WindGusts10m")
df = wsp_df.merge(gust_df, on="ValidTimeUtc", how="inner")

gust_cols = [c for c in gust_df.columns if c != "ValidTimeUtc"]
ratio_cols: list[str] = []
for gc in gust_cols:
    nwp = gc.replace("WindGusts10m_", "")
    wc = f"WindSpeed10m_{nwp}"
    if wc not in df.columns:
        continue
    rc = f"gust_ratio_{nwp}"
    df[rc] = np.clip(df[gc].values / np.maximum(df[wc].values, 0.5), 0.5, 4.0)
    ratio_cols.append(rc)
df["gust_ratio_mean"] = df[ratio_cols].mean(axis=1, skipna=True)
df["gust_ratio_std"] = df[ratio_cols].std(axis=1, skipna=True)

era5 = con.execute(f"""
    SELECT ValidTimeUtc, WindGusts10m AS gust_era5
    FROM read_parquet('{ERA5_GLOB}', union_by_name=true)
    WHERE ValidTimeUtc >= TIMESTAMP '{START}'
      AND ValidTimeUtc <= TIMESTAMP '{END}'
""").df()
df = df.merge(era5, on="ValidTimeUtc", how="inner").dropna(subset=["gust_era5"])
n = len(df)
i_tr, i_va = int(n * 0.7), int(n * 0.85)
train, val, test = df.iloc[:i_tr].copy(), df.iloc[i_tr:i_va].copy(), df.iloc[i_va:].copy()
print(f"   {n:,} aligned rows -> train {len(train):,} / val {len(val):,} / test {len(test):,}")

feats = gust_cols + ratio_cols + ["gust_ratio_mean", "gust_ratio_std"]
print(f"   Feature count: {len(feats)} ({len(gust_cols)} gust + {len(ratio_cols)} ratio + 2 spread)")

m = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    num_leaves=31, min_data_in_leaf=20, reg_alpha=0.1,
    reg_lambda=0.1, random_state=42, verbose=-1,
)
m.fit(
    train[feats].values, train["gust_era5"].values,
    eval_set=[(val[feats].values, val["gust_era5"].values)],
    callbacks=[lgb.early_stopping(30, verbose=False)],
)
pred = m.predict(test[feats].values)
truth = test["gust_era5"].values
mae = float(np.mean(np.abs(pred - truth)))
rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
nwp_mean = test[gust_cols].mean(axis=1, skipna=True).values
nwp_mean_mae = float(np.mean(np.abs(nwp_mean - truth)))

print()
print(f"   NWP-mean baseline:              MAE = {nwp_mean_mae:.4f}")
print(f"   LightGBM production-scope:      MAE = {mae:.4f}  RMSE = {rmse:.4f}  "
      f"({100 * (mae - nwp_mean_mae) / nwp_mean_mae:+.2f}% vs NWP-mean)")
print()
print(f"   For reference: Temp/gust_ratio_bakeoff.py minimal (10 NWPs incl. archive) was 0.9808.")

# -- Feature importance ---------------------------------------------------
print()
print("=" * 78)
print(f"Feature importance (LightGBM split count, all {len(feats)} features)")
print("=" * 78)
imp = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])
for rank, (name, gain) in enumerate(imp, 1):
    if name.startswith("WindGusts10m_"):
        nwp = name.replace("WindGusts10m_", "")
        tag = "GUST"
    elif name.startswith("gust_ratio_") and name not in ("gust_ratio_mean", "gust_ratio_std"):
        nwp = name.replace("gust_ratio_", "")
        tag = "RATIO"
    else:
        nwp = ""
        tag = "SPREAD"
    in_prod = " [PROD]" if nwp in PROD_NWPS else (" [ext]" if nwp else "")
    print(f"  rank {rank:>2}  {tag:<6}  {name:42s}  imp={gain:>4}{in_prod}")

# Per-NWP importance roll-up (sum of gust + ratio per NWP)
print()
print("=" * 78)
print("Per-NWP importance roll-up (gust + ratio importance summed)")
print("=" * 78)
per_nwp: dict[str, int] = {}
for name, gain in zip(feats, m.feature_importances_):
    if name.startswith("WindGusts10m_"):
        nwp = name.replace("WindGusts10m_", "")
    elif name.startswith("gust_ratio_") and name not in ("gust_ratio_mean", "gust_ratio_std"):
        nwp = name.replace("gust_ratio_", "")
    else:
        continue
    per_nwp[nwp] = per_nwp.get(nwp, 0) + int(gain)
for nwp, total in sorted(per_nwp.items(), key=lambda x: -x[1]):
    tag = "[PROD]" if nwp in PROD_NWPS else "[ext]"
    print(f"  {nwp:35s}  total_imp={total:>4}  {tag}")
