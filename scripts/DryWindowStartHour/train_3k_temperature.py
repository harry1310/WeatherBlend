"""Phase 3k — temperature-scaled hourly q before iid MC dry-window.

3g samples iid Bernoullis from 3a's raw hourly q. The diagnostic shows 3a's
marginals are implicitly calibrated for the iid+independence assumption but
the calibration isn't optimal — at some (station, window, lead) cells 3g
under- or over-predicts dry-window probability systematically. 3k inserts a
single-parameter sharpness knob:
    q' = q^t / (q^t + (1-q)^t)   (a.k.a. logit + temperature)
fit per (station, window, lead) by grid-searching t ∈ [0.4, 2.5] against
VAL dry-window Brier under the same iid MC sampler 3g uses. t < 1 softens
marginals; t > 1 sharpens; t = 1 reduces to 3g exactly. Picking t per cell
amounts to "let 3g calibrate itself for the dry-window objective."

Same chronological 70/15/15 split as 3j (the copula MC) so the bake-off's
test slice aligns and head-to-head against 3g is fair.

Inputs: 3a replay parquets under data/predictions/precipitation_replay/.
Outputs: data/models/dry_window/{station}/window_{N}h/v..._phase3k/
    test_predictions.parquet + training_metadata.json (the fitted t per
    lead, plus val Brier vs 3g for the comparison).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours, has_contiguous_dry_block  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"
REPLAY_ROOT = ROOT / "data" / "predictions" / "precipitation_replay"

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
MC_SAMPLES = 1000
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Temperature grid — log-spaced around 1.0 so both sharpening (t>1) and
# softening (t<1) get equal coverage.
T_GRID = np.geomspace(0.4, 2.5, num=21)


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
        qs.append(np.clip(sub["ProbWet"].to_numpy(dtype="float64"), 1e-6, 1 - 1e-6))
        obs.append(sub["Label"].astype(np.int32).to_numpy())
    if not dates:
        return [], np.zeros((0, 0)), np.zeros((0, 0), dtype="int32")
    return dates, np.stack(qs), np.stack(obs)


def temperature_transform(q: np.ndarray, t: float) -> np.ndarray:
    """q' = q^t / (q^t + (1-q)^t). t=1 -> identity. t>1 sharpens, t<1 softens."""
    qt = np.power(q, t)
    return qt / (qt + np.power(1 - q, t))


def mc_dry_window(q: np.ndarray, window: int, n_samples: int, rng: np.random.Generator) -> float:
    n = len(q)
    if window > n:
        return 0.0
    samples = (rng.random((n_samples, n)) < q).astype(np.int32)
    hits = 0
    for s in samples:
        run, longest = 0, 0
        for v in s:
            if v == 0:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
        if longest >= window:
            hits += 1
    return hits / n_samples


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def evaluate_temperature(q_arr: np.ndarray, obs_arr: np.ndarray, window: int, t: float,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return MC dry-window probs + labels for a given t."""
    n = len(q_arr)
    probs = np.zeros(n)
    labels = np.zeros(n, dtype="float64")
    for i in range(n):
        q_t = temperature_transform(q_arr[i], t)
        probs[i] = mc_dry_window(q_t, window, MC_SAMPLES, rng)
        labels[i] = 1.0 if has_contiguous_dry_block(obs_arr[i], window) else 0.0
    return probs, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]

    print(f"Phase 3k (temperature-scaled iid MC) — stations={stations} leads={leads} windows={windows}")
    print(f"  MC samples: {MC_SAMPLES}; seed: {SEED}; temperature grid: {T_GRID.round(2).tolist()}")

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
            version = f"v{ts}_phase3k"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []
            t_by_lead: dict[int, float] = {}
            val_brier_by_lead: dict[int, dict] = {}

            for lead in leads:
                dates, q_arr, obs_arr = cells_by_lead[lead]
                if not dates:
                    continue
                n = len(dates)
                tr_end = int(np.floor(n * TRAIN_FRAC))
                val_end = tr_end + int(np.floor(n * VAL_FRAC))

                # Fit t on TRAIN+VAL combined dry-window labels — using val
                # only would give us 90 days, marginal noise. The 3a marginals
                # themselves were fit on train so using train+val for the
                # t grid search isn't leakage of the same kind 3a saw at fit.
                # Test stays held-out.
                fit_q = q_arr[:val_end]
                fit_obs = obs_arr[:val_end]

                best_t = 1.0
                best_b = float("inf")
                grid_results: list[tuple[float, float]] = []
                for t in T_GRID:
                    # Single MC pass with fixed seed → deterministic comparison
                    rng = np.random.default_rng(SEED)
                    probs, labels = evaluate_temperature(fit_q, fit_obs, window, t, rng)
                    b = brier(probs, labels)
                    grid_results.append((float(t), b))
                    if b < best_b:
                        best_b = b
                        best_t = float(t)
                t_by_lead[lead] = best_t

                # Also evaluate t=1 explicitly as the 3g reference
                rng = np.random.default_rng(SEED)
                probs_iid, labels_iid = evaluate_temperature(fit_q, fit_obs, window, 1.0, rng)
                iid_b = brier(probs_iid, labels_iid)
                val_brier_by_lead[lead] = {"3g_ref_t1": iid_b, "best_t": best_t, "best_brier": best_b,
                                           "improvement_pct": float(100 * (iid_b - best_b) / iid_b)}

                # Score the held-out test slice at the fitted t.
                test_q = q_arr[val_end:]
                test_obs = obs_arr[val_end:]
                rng = np.random.default_rng(SEED + 1)  # different seed for test
                test_probs, test_labels = evaluate_temperature(test_q, test_obs, window, best_t, rng)
                test_b = brier(test_probs, test_labels)
                # Reference: 3g (iid, t=1) on the same test slice
                rng = np.random.default_rng(SEED + 1)
                ref_probs, _ = evaluate_temperature(test_q, test_obs, window, 1.0, rng)
                ref_b = brier(ref_probs, test_labels)

                print(f"  {station} window {window}h lead {lead}h: "
                      f"best_t={best_t:.2f} (val Brier {best_b:.4f} vs t=1 {iid_b:.4f}, +{100*(iid_b - best_b)/iid_b:.1f}% on val); "
                      f"TEST Brier={test_b:.4f}  3g_test_ref={ref_b:.4f}  "
                      f"({100*(ref_b - test_b)/ref_b:+.1f}% vs 3g on test, n={len(test_q)})")

                for td, p, y in zip(dates[val_end:], test_probs, test_labels):
                    test_rows.append({
                        "target_date": td, "station": station,
                        "window": window, "lead": lead,
                        "p_dry_window": float(p),
                        "observed_dry_window": np.uint8(int(y)),
                    })

            if test_rows:
                pd.DataFrame(test_rows).to_parquet(bundle_dir / "test_predictions.parquet", index=False)
                meta = {
                    "Version": version, "Target": "dry_window", "Phase": "3k",
                    "Architecture": "temperature_scaled_iid_mc",
                    "MCSamples": MC_SAMPLES, "Seed": SEED,
                    "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                    "TemperatureGrid": T_GRID.round(3).tolist(),
                    "FittedTemperatureByLead": t_by_lead,
                    "ValBrierByLead": val_brier_by_lead,
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
