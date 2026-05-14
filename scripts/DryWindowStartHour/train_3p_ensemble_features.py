"""Phase 3p trainer — ensemble-feature meta-model.

Reads the per-(station, lead, target_date) feature matrix written by
dry_window_3p_features.py and trains a per-(station, window, lead)
logistic regression that maps the 16 ensemble features to
P(dry-window N).

Same chronological 70/15/15 split as 3b/3j/3n so the bake-off compares
on the same test cells. Writes test_predictions.parquet in the
standard dry-window bundle layout so the existing bake-off picks it up
via a new find_3p_test_predictions discovery function.

L2-regularised LR (C=1.0) by default — small models for ~600 train
days × 16 features (≈ 38:1 examples:params). PAV calibration on val is
applied if val sample count ≥ 30; gated otherwise.

Outputs per (station, window):
  data/models/dry_window/{station}/window_{N}h/v..._phase3p/
    test_predictions.parquet
    training_metadata.json  (coefficients per lead + per-lead val PAV)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import has_contiguous_dry_block  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "data" / "features" / "dry_window_3p" / "features.parquet"
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
PAV_MIN_VAL = 30


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and "_" in c and not c.startswith("frac")]


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    C = args.C

    if not FEATURES_PATH.exists():
        print(f"::error::features.parquet not found at {FEATURES_PATH}")
        print("Run scripts/DryWindowStartHour/dry_window_3p_features.py first.")
        return 1

    df = pd.read_parquet(FEATURES_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None)
    feature_cols = [c for c in df.columns if c.startswith("f") and "_" in c]
    print(f"Phase 3p (ensemble-feature LR meta-model)")
    print(f"  features.parquet: {len(df):,} rows, {len(feature_cols)} features")
    print(f"  features: {feature_cols}")
    print(f"  L2 strength: C={C}; train_frac={TRAIN_FRAC}; val_frac={VAL_FRAC}; PAV gated by val_n>={PAV_MIN_VAL}")

    overall_start = datetime.now(timezone.utc)
    ts = overall_start.strftime("%Y-%m-%d_%H%M%S")

    for station in stations:
        sd = df[df["station"] == station].copy().sort_values(["lead", "target_date"])
        if sd.empty:
            print(f"::warning::{station}: no features rows; skipping")
            continue
        print(f"\n=== {station} ===")

        for window in windows:
            version = f"v{ts}_phase3p"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []
            coefs_by_lead: dict[int, dict] = {}
            pav_by_lead: dict[int, dict] = {}

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

                # Standardise on train (LR is L2-regularised so scaling matters).
                mean = X_tr.mean(axis=0)
                std = X_tr.std(axis=0)
                std = np.where(std < 1e-6, 1.0, std)
                X_tr_s = (X_tr - mean) / std
                X_va_s = (X_va - mean) / std
                X_te_s = (X_te - mean) / std

                lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=SEED)
                lr.fit(X_tr_s, y_tr)
                val_raw = lr.predict_proba(X_va_s)[:, 1]
                test_raw = lr.predict_proba(X_te_s)[:, 1]

                # PAV on val output if we have enough rows.
                pav = None
                if len(val_raw) >= PAV_MIN_VAL:
                    pav = IsotonicRegression(out_of_bounds="clip").fit(val_raw, y_va)
                    test_probs = pav.predict(test_raw)
                else:
                    test_probs = test_raw

                test_brier = brier(test_probs, y_te)
                clim = float(y_tr.mean())
                clim_brier = brier(np.full(len(y_te), clim), y_te)
                bss = (clim_brier - test_brier) / clim_brier if clim_brier > 0 else float("nan")
                raw_brier = brier(test_raw, y_te)

                coefs_by_lead[lead] = dict(zip(feature_cols, [float(c) for c in lr.coef_[0]]))
                if pav is not None:
                    pav_by_lead[lead] = {
                        "X": [float(x) for x in pav.X_thresholds_],
                        "Y": [float(y) for y in pav.y_thresholds_],
                    }

                print(f"  {station} window {window}h lead {lead}h: "
                      f"test Brier={test_brier:.4f}  raw={raw_brier:.4f}  clim={clim_brier:.4f}  "
                      f"BSS={bss:+.4f}  n_test={len(y_te)}  intercept={float(lr.intercept_[0]):+.3f}"
                      f"{'  [PAV]' if pav is not None else ''}")

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
                    "Version": version, "Target": "dry_window", "Phase": "3p",
                    "Architecture": "ensemble_feature_logistic_regression",
                    "FeatureColumns": feature_cols,
                    "C": C, "Seed": SEED,
                    "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                    "CoefByLead": coefs_by_lead,
                    "PavByLead": pav_by_lead,
                    "WindowHours": window, "Leads": leads,
                    "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
                }
                (bundle_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
                print(f"  wrote {len(test_rows)} rows -> {bundle_dir / 'test_predictions.parquet'}")

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
