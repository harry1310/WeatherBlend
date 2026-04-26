"""Bagging + feature_fraction hyperparameter grid for the temp 2b 6-model
blender on the post-2024-09 restricted window.

Each (bagging_fraction, feature_fraction) combo is evaluated across 5 seeds
to capture the variance bagging introduces. Reports mean MAE ± std and the
per-lead winner.
"""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from restricted_window_bakeoff import (  # noqa: E402
    FORECASTS, ERA5, WINDOW_START, LEADS, TEST_FRACTION,
    LGB_REG_PARAMS, NUM_ITERATIONS, EARLY_STOPPING,
    _calendar, _spread,
)
from temp_stability_bootstrap import load_lead  # noqa: E402

BAGGING_GRID = [1.0, 0.9, 0.8, 0.7]
FEATURE_GRID = [1.0, 0.9, 0.8]
SEEDS = [42, 7, 123, 2025, 9001]


def fit_predict(X_tr, y_tr, X_te, feat_cols, params):
    cut = max(int(len(X_tr) * 0.85), 10)
    train_set = lgb.Dataset(X_tr[:cut], label=y_tr[:cut], feature_name=feat_cols)
    val_set = lgb.Dataset(X_tr[cut:], label=y_tr[cut:], feature_name=feat_cols, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_ITERATIONS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return booster.predict(X_te, num_iteration=booster.best_iteration)


def main():
    con = duckdb.connect()
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()
    era_glob = (ERA5 / "**" / "*.parquet").as_posix()

    rows = []
    model_cols = ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem"]

    for lead in LEADS:
        df = load_lead(con, lead, fc_glob, era_glob)
        d = _spread(df, model_cols, "temp")
        feat_cols = model_cols + ["temp_mean", "temp_std", "temp_range",
                                  "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        split = int(len(d) * (1 - TEST_FRACTION))
        tr, te = d.iloc[:split], d.iloc[split:]
        X_tr = tr[feat_cols].to_numpy(dtype="float64")
        y_tr = tr["era5_temp"].to_numpy(dtype="float64")
        X_te = te[feat_cols].to_numpy(dtype="float64")
        y_te = te["era5_temp"].to_numpy(dtype="float64")

        print(f"\n--- Lead {lead}h --- (n_train={len(tr):,} n_test={len(te):,})")
        t0 = time.time()
        for bf, ff in product(BAGGING_GRID, FEATURE_GRID):
            maes = []
            for seed in SEEDS:
                params = dict(LGB_REG_PARAMS, seed=seed, bagging_fraction=bf,
                              bagging_freq=(1 if bf < 1.0 else 0), feature_fraction=ff)
                p = fit_predict(X_tr, y_tr, X_te, feat_cols, params)
                maes.append(float(np.mean(np.abs(p - y_te))))
            rows.append(dict(lead=lead, bagging=bf, feature=ff,
                             mean_mae=float(np.mean(maes)),
                             std_mae=float(np.std(maes)),
                             min_mae=float(min(maes)),
                             max_mae=float(max(maes))))
        print(f"  grid done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    print("\n=== Per-lead grid (mean ± std MAE across 5 seeds) ===")
    for lead in LEADS:
        sub = df[df.lead == lead].copy()
        pivot_mean = sub.pivot(index="bagging", columns="feature", values="mean_mae").round(4)
        pivot_std = sub.pivot(index="bagging", columns="feature", values="std_mae").round(4)
        print(f"\nLead {lead}h — mean MAE:")
        print(pivot_mean.to_string())
        print(f"\nLead {lead}h — std across seeds:")
        print(pivot_std.to_string())
        # Best
        best = sub.sort_values("mean_mae").iloc[0]
        baseline = sub[(sub.bagging == 1.0) & (sub.feature == 1.0)].iloc[0]
        delta_pct = (baseline.mean_mae - best.mean_mae) / baseline.mean_mae * 100
        print(f"\nLead {lead}h winner: bagging={best.bagging}, feature={best.feature}  "
              f"MAE={best.mean_mae:.4f} ± {best.std_mae:.4f}  "
              f"(vs deterministic {baseline.mean_mae:.4f}; Δ={delta_pct:+.2f}%)")

    print("\n=== Cross-lead 'safest single setting' (rank-sum across leads) ===")
    df["rank"] = df.groupby("lead").mean_mae.rank()
    by_combo = df.groupby(["bagging", "feature"]).agg(
        rank_sum=("rank", "sum"),
        mean_mae_avg=("mean_mae", "mean"),
        max_std=("std_mae", "max"),
    ).reset_index().sort_values("rank_sum")
    print(by_combo.to_string(index=False))


if __name__ == "__main__":
    main()
