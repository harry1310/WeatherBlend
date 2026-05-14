"""Phase 3q — LightGBM on the same 16 ensemble features 3p uses.

Hypothesis: 3p (L2 logistic regression) lost 3.8% to 3g on aggregate
because the features are highly correlated (f01/f02/f03 measure variants
of "member vote"; f04/f05/f06 measure variants of "longest dry run"; etc.)
and L2 spreads weight across redundant features instead of picking the
most informative one per regime. Tree-based learners pick the best single
feature per split — multicollinearity stops mattering.

Same input data and chronological split as 3p so the bake-off compares
on identical cells. PAV on val DISABLED by default — today's evidence
across 3g and 3p showed PAV-on-output hurts at ~120 val rows due to
overfitting at Bovey/Hexworthy. Re-enable per-station if a future run
shows different.

Outputs per (station, window):
  data/models/dry_window/{station}/window_{N}h/v..._phase3q/
    test_predictions.parquet
    training_metadata.json  (per-lead feature importances)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "data" / "features" / "dry_window_3p" / "features.parquet"
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# LightGBM hyperparameters tuned conservatively for the ~600-train-row-per-cell
# regime. num_leaves=15 + min_child_samples=20 + L2=0.1 keeps each tree small
# enough to avoid overfitting. Bagging adds row variance; feature_fraction
# decorrelates trees and reduces multicollinearity-amplifying overfit. Early
# stopping on val Brier caps trees automatically — 200 iter ceiling handles
# heavier-signal cells without ballooning trees on the easier ones.
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 20,
    "lambda_l2": 0.1,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "feature_fraction": 0.8,
    "random_state": SEED,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 20


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]

    if not FEATURES_PATH.exists():
        print(f"::error::features.parquet not found at {FEATURES_PATH}")
        print("Run scripts/DryWindowStartHour/dry_window_3p_features.py first.")
        return 1

    df = pd.read_parquet(FEATURES_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None)
    feature_cols = [c for c in df.columns if c.startswith("f") and "_" in c]
    print(f"Phase 3q (ensemble-feature LightGBM)")
    print(f"  features.parquet: {len(df):,} rows, {len(feature_cols)} features")
    print(f"  LightGBM params: num_leaves={LGB_PARAMS['num_leaves']}, max_depth={LGB_PARAMS['max_depth']}, "
          f"lr={LGB_PARAMS['learning_rate']}, n_est={LGB_PARAMS['n_estimators']}, "
          f"min_child={LGB_PARAMS['min_child_samples']}, L2={LGB_PARAMS['lambda_l2']}, "
          f"feat_frac={LGB_PARAMS['feature_fraction']}, bag_frac={LGB_PARAMS['bagging_fraction']}, "
          f"early_stop={EARLY_STOPPING_ROUNDS}")
    print(f"  PAV: DISABLED (today's evidence: PAV hurts at ~120 val rows)")

    overall_start = datetime.now(timezone.utc)
    ts = overall_start.strftime("%Y-%m-%d_%H%M%S")

    for station in stations:
        sd = df[df["station"] == station].copy().sort_values(["lead", "target_date"])
        if sd.empty:
            print(f"::warning::{station}: no features rows; skipping")
            continue
        print(f"\n=== {station} ===")

        for window in windows:
            version = f"v{ts}_phase3q"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []
            importance_by_lead: dict[int, dict] = {}
            best_iter_by_lead: dict[int, int] = {}

            for lead in leads:
                cell = sd[sd["lead"] == lead].sort_values("target_date").reset_index(drop=True)
                if len(cell) < 100:
                    print(f"  {station} window {window}h lead {lead}h: only {len(cell)} rows; skipping")
                    continue

                X = cell[feature_cols].to_numpy(dtype="float64")
                label_col = f"dry_{window}h"
                y = cell[label_col].to_numpy(dtype="float64")

                n = len(cell)
                tr_end = int(np.floor(n * TRAIN_FRAC))
                val_end = tr_end + int(np.floor(n * VAL_FRAC))
                X_tr, X_va, X_te = X[:tr_end], X[tr_end:val_end], X[val_end:]
                y_tr, y_va, y_te = y[:tr_end], y[tr_end:val_end], y[val_end:]
                dates_te = cell["target_date"].iloc[val_end:].tolist()

                model = lgb.LGBMClassifier(**LGB_PARAMS)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                best_iter_by_lead[lead] = int(model.best_iteration_ or LGB_PARAMS["n_estimators"])

                test_probs = model.predict_proba(X_te)[:, 1]
                test_brier = brier(test_probs, y_te)
                clim = float(y_tr.mean())
                clim_brier = brier(np.full(len(y_te), clim), y_te)
                bss = (clim_brier - test_brier) / clim_brier if clim_brier > 0 else float("nan")

                importance = dict(zip(feature_cols, [int(g) for g in model.booster_.feature_importance(importance_type="gain").tolist()]))
                top3 = sorted(importance.items(), key=lambda kv: -kv[1])[:3]
                importance_by_lead[lead] = importance

                print(f"  {station} window {window}h lead {lead}h: "
                      f"test Brier={test_brier:.4f}  clim={clim_brier:.4f}  BSS={bss:+.4f}  "
                      f"n_test={len(y_te)}  best_iter={model.best_iteration_}  "
                      f"top: {', '.join(f'{n}={g}' for n, g in top3)}")

                for d, p, lbl in zip(dates_te, test_probs, y_te):
                    test_rows.append({
                        "target_date": d, "station": station,
                        "window": window, "lead": lead,
                        "p_dry_window": float(p),
                        "observed_dry_window": np.uint8(int(lbl)),
                    })

            if test_rows:
                pd.DataFrame(test_rows).to_parquet(bundle_dir / "test_predictions.parquet", index=False)
                meta = {
                    "Version": version, "Target": "dry_window", "Phase": "3q",
                    "Architecture": "ensemble_feature_lightgbm",
                    "FeatureColumns": feature_cols,
                    "LightGBMParams": LGB_PARAMS,
                    "EarlyStoppingRounds": EARLY_STOPPING_ROUNDS,
                    "Seed": SEED,
                    "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                    "ImportanceByLead": importance_by_lead,
                    "BestIterByLead": best_iter_by_lead,
                    "WindowHours": window, "Leads": leads,
                    "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
                }
                (bundle_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
                print(f"  wrote {len(test_rows)} rows -> {bundle_dir / 'test_predictions.parquet'}")

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
