"""Phase 3l — hand-crafted q-vector features + per-cell logistic regression.

Bypasses MC entirely. From the 9-hour daytime q-vector (3a's hourly P(wet)),
extract ~10 summary features and train a logistic regression per
(station, window, lead) to predict the dry-window label directly.

Features (per day):
    mean_q, max_q, min_q, var_q
    n_low  = count(q < 0.20)
    n_high = count(q > 0.50)
    longest_low_run  = longest contiguous run of q < 0.25
    longest_high_run = longest contiguous run of q > 0.50
    morning_mean  = mean(q[0:3])
    midday_mean   = mean(q[3:6])
    afternoon_mean = mean(q[6:9])
    q_first, q_last  (= the two ends — wet-ending or wet-starting days are
                       hostile to long dry blocks regardless of middle)

12 features. With ~500 train days per cell that's 40:1 examples:params for
a plain LR — plenty of room. L2 regularised to be safe.

Tests the hypothesis: if the GRU's job is just to learn run-length statistics
from a 9-hour sequence, a logistic regression on hand-crafted run-statistics
should do at least as well at this data scale. (3i with raw NWP failed; 3l
with engineered features on a pre-blended q should compete.)

Same 70/15/15 split as 3j/3k. Outputs to
data/models/dry_window/{station}/window_{N}h/v..._phase3l/.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours, has_contiguous_dry_block  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"
REPLAY_ROOT = ROOT / "data" / "predictions" / "precipitation_replay"

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.50


def find_replay_dir(station: str) -> Path | None:
    station_dir = REPLAY_ROOT / station
    if not station_dir.is_dir():
        return None
    cands = [d for d in station_dir.iterdir() if d.is_dir() and "phase" not in d.name]
    return max(cands, key=lambda d: d.name) if cands else None


def build_daytime_cells(replay_dir: Path, lead: int) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    df = pd.read_parquet(replay_dir / f"lead_{lead}h.parquet")
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    dates: list[pd.Timestamp] = []
    qs: list[np.ndarray] = []
    obs: list[np.ndarray] = []
    for target_date, grp in df.groupby("target_date"):
        target_ts = pd.Timestamp(target_date)
        s, e = daytime_utc_hours(target_ts)
        n = e - s
        sub = grp[(grp["hour"] >= s) & (grp["hour"] < e)].sort_values("valid_time")
        if len(sub) != n:
            continue
        dates.append(target_ts)
        qs.append(sub["ProbWet"].to_numpy(dtype="float64"))
        obs.append(sub["Label"].astype(np.int32).to_numpy())
    if not dates:
        return [], np.zeros((0, 0)), np.zeros((0, 0), dtype="int32")
    return dates, np.stack(qs), np.stack(obs)


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest contiguous True run in a boolean array."""
    run, longest = 0, 0
    for v in mask:
        if v:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return longest


def build_features(q_arr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return (n_days, n_features) + feature names from a (n_days, 9) q array."""
    n = q_arr.shape[0]
    names = [
        "mean_q", "max_q", "min_q", "var_q",
        "n_low", "n_high",
        "longest_low_run", "longest_high_run",
        "morning_mean", "midday_mean", "afternoon_mean",
        "q_first", "q_last",
    ]
    feats = np.zeros((n, len(names)), dtype="float64")
    for i in range(n):
        q = q_arr[i]
        feats[i, 0] = q.mean()
        feats[i, 1] = q.max()
        feats[i, 2] = q.min()
        feats[i, 3] = q.var()
        feats[i, 4] = int((q < LOW_THRESHOLD).sum())
        feats[i, 5] = int((q > HIGH_THRESHOLD).sum())
        feats[i, 6] = longest_run(q < LOW_THRESHOLD)
        feats[i, 7] = longest_run(q > HIGH_THRESHOLD)
        feats[i, 8] = q[:3].mean()
        feats[i, 9] = q[3:6].mean()
        feats[i, 10] = q[6:9].mean()
        feats[i, 11] = q[0]
        feats[i, 12] = q[-1]
    return feats, names


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--C", type=float, default=1.0,
                    help="Inverse L2 regularisation strength (sklearn LR convention). "
                         "Smaller C = more regularisation.")
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    C = args.C

    print(f"Phase 3l (engineered q-vector features + LR) — stations={stations} leads={leads} windows={windows}")
    print(f"  features: mean, max, min, var, n_low<{LOW_THRESHOLD}, n_high>{HIGH_THRESHOLD}, "
          f"longest_low_run, longest_high_run, morning/midday/afternoon means, q_first, q_last")
    print(f"  classifier: sklearn LogisticRegression(C={C}, solver='lbfgs', max_iter=1000)")

    overall_start = datetime.now(timezone.utc)

    for station in stations:
        replay_dir = find_replay_dir(station)
        if replay_dir is None:
            print(f"::warning::{station}: no replay; skipping")
            continue
        print(f"\n=== {station} (replay {replay_dir.name}) ===")

        cells_by_lead: dict[int, tuple[list[pd.Timestamp], np.ndarray, np.ndarray]] = {}
        for lead in leads:
            cells_by_lead[lead] = build_daytime_cells(replay_dir, lead)

        for window in windows:
            ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
            version = f"v{ts}_phase3l"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []
            coefs_by_lead: dict[int, dict] = {}

            for lead in leads:
                dates, q_arr, obs_arr = cells_by_lead[lead]
                if not dates:
                    continue

                feats, names = build_features(q_arr)
                labels = np.array([1 if has_contiguous_dry_block(obs_arr[i], window) else 0 for i in range(len(dates))],
                                  dtype="int32")

                n = len(dates)
                tr_end = int(np.floor(n * TRAIN_FRAC))
                val_end = tr_end + int(np.floor(n * VAL_FRAC))
                X_train = feats[:val_end]   # train+val combined for LR (fixed-form, no early stop)
                y_train = labels[:val_end]
                X_test = feats[val_end:]
                y_test = labels[val_end:]

                # Standardise features on train+val only. LR is scale-sensitive
                # for L2 regularisation.
                mean = X_train.mean(axis=0)
                std = X_train.std(axis=0)
                std = np.where(std < 1e-6, 1.0, std)
                X_train_s = (X_train - mean) / std
                X_test_s  = (X_test  - mean) / std

                lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=SEED)
                lr.fit(X_train_s, y_train)
                test_probs = lr.predict_proba(X_test_s)[:, 1]
                test_b = brier(test_probs, y_test.astype(float))

                # Climatology reference
                clim = float(y_train.mean())
                clim_pred = np.full(len(y_test), clim)
                clim_b = brier(clim_pred, y_test.astype(float))
                bss = (clim_b - test_b) / clim_b if clim_b > 0 else float("nan")

                coefs_by_lead[lead] = dict(zip(names, [float(c) for c in lr.coef_[0]]))

                print(f"  {station} window {window}h lead {lead}h: "
                      f"LR test Brier={test_b:.4f}  clim={clim_b:.4f}  BSS={bss:+.4f}  "
                      f"n_test={len(y_test)}  intercept={float(lr.intercept_[0]):+.3f}")

                for td, p, y in zip(dates[val_end:], test_probs, y_test):
                    test_rows.append({
                        "target_date": td, "station": station,
                        "window": window, "lead": lead,
                        "p_dry_window": float(p),
                        "observed_dry_window": np.uint8(int(y)),
                    })

            if test_rows:
                pd.DataFrame(test_rows).to_parquet(bundle_dir / "test_predictions.parquet", index=False)
                meta = {
                    "Version": version, "Target": "dry_window", "Phase": "3l",
                    "Architecture": "logistic_regression_qfeatures",
                    "Features": names, "LowThreshold": LOW_THRESHOLD, "HighThreshold": HIGH_THRESHOLD,
                    "Seed": SEED, "C": C,
                    "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                    "CoefByLead": coefs_by_lead,
                    "WindowHours": window, "Leads": leads,
                    "ReplayVersion": replay_dir.name,
                    "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
                }
                (bundle_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
                print(f"  wrote {len(test_rows)} rows -> {bundle_dir / 'test_predictions.parquet'}")

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
