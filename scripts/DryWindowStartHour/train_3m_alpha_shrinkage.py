"""Phase 3m — α-shrinkage interpolation between 3j (copula MC) and 3g (iid MC).

3j's Gaussian copula beats 3g at 3h windows by ~5% but loses at 6h by 11%. The
hypothesis: the right amount of dependence varies by window length. Build a
single-parameter family
    Σ_α = α · Σ + (1 - α) · I
that smoothly interpolates between iid (α=0) and full copula (α=1). Per
(station, window, lead): grid-search α on train+val dry-window Brier, evaluate
on test.

α=0 reduces exactly to 3g (identity correlation → iid Bernoullis).
α=1 reduces exactly to 3j (full fitted correlation).
Intermediate α blends dependence strength.

Inputs: 3a replay parquets under data/predictions/precipitation_replay/.
Outputs: data/models/dry_window/{station}/window_{N}h/v..._phase3m/
    test_predictions.parquet + training_metadata.json (fitted α per lead +
    val Brier vs 3g/3j for transparency).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

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
ALPHA_GRID = np.linspace(0.0, 1.0, 11)   # 0.0, 0.1, ..., 1.0


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


def fit_correlation(obs_seqs: np.ndarray) -> np.ndarray:
    """9x9 Pearson correlation on observed binary sequences. Jitter the diagonal
    to keep it strictly PSD for Cholesky."""
    if len(obs_seqs) < 10:
        return np.eye(obs_seqs.shape[1])
    corr = np.corrcoef(obs_seqs.T)
    corr = corr + 1e-6 * np.eye(corr.shape[0])
    return corr


def copula_dry_window(q: np.ndarray, L: np.ndarray, window: int,
                      n_samples: int, rng: np.random.Generator) -> float:
    """MC under copula with Cholesky L (= chol(Σ_α))."""
    n = len(q)
    if window > n:
        return 0.0
    z_iid = rng.standard_normal((n_samples, n))
    z = z_iid @ L.T
    u = norm.cdf(z)
    samples = (u < q[None, :]).astype(np.int32)
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


def evaluate_alpha(q_arr: np.ndarray, obs_arr: np.ndarray, window: int,
                   alpha: float, corr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """MC over the shrunk-correlation copula on (q_arr, obs_arr). Returns
    (probs, labels)."""
    n_hours = corr.shape[0]
    sigma_alpha = alpha * corr + (1 - alpha) * np.eye(n_hours)
    # Cholesky with PSD-jitter fallback.
    try:
        L = np.linalg.cholesky(sigma_alpha)
    except np.linalg.LinAlgError:
        sigma_alpha = sigma_alpha + 1e-5 * np.eye(n_hours)
        L = np.linalg.cholesky(sigma_alpha)
    probs = np.zeros(len(q_arr))
    labels = np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        probs[i] = copula_dry_window(q_arr[i], L, window, MC_SAMPLES, rng)
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

    print(f"Phase 3m (alpha-shrinkage 3j<->3g) — stations={stations} leads={leads} windows={windows}")
    print(f"  MC samples: {MC_SAMPLES}; seed: {SEED}; alpha grid: {ALPHA_GRID.round(2).tolist()}")

    overall_start = datetime.now(timezone.utc)

    for station in stations:
        replay_dir = find_replay_dir(station)
        if replay_dir is None:
            print(f"::warning::{station}: no replay; skipping")
            continue
        print(f"\n=== {station} (replay {replay_dir.name}) ===")

        cells_by_lead: dict[int, tuple[list[pd.Timestamp], np.ndarray, np.ndarray]] = {}
        corr_by_lead: dict[int, np.ndarray] = {}
        for lead in leads:
            dates, q_arr, obs_arr = build_daytime_cells(replay_dir, lead)
            cells_by_lead[lead] = (dates, q_arr, obs_arr)
            if dates:
                tr_end = int(np.floor(len(dates) * TRAIN_FRAC))
                corr_by_lead[lead] = fit_correlation(obs_arr[:tr_end])

        for window in windows:
            ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
            version = f"v{ts}_phase3m"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []
            alpha_by_lead: dict[int, float] = {}
            val_brier_by_lead: dict[int, dict] = {}

            for lead in leads:
                dates, q_arr, obs_arr = cells_by_lead[lead]
                corr = corr_by_lead.get(lead)
                if not dates or corr is None:
                    continue
                n = len(dates)
                tr_end = int(np.floor(n * TRAIN_FRAC))
                val_end = tr_end + int(np.floor(n * VAL_FRAC))

                # Grid search alpha on train+val combined.
                fit_q = q_arr[:val_end]
                fit_obs = obs_arr[:val_end]
                best_a = 0.0
                best_b = float("inf")
                grid_results = []
                for a in ALPHA_GRID:
                    rng = np.random.default_rng(SEED)
                    probs, labels = evaluate_alpha(fit_q, fit_obs, window, float(a), corr, rng)
                    b = brier(probs, labels)
                    grid_results.append({"alpha": float(a), "val_brier": float(b)})
                    if b < best_b:
                        best_b = b
                        best_a = float(a)
                alpha_by_lead[lead] = best_a

                # Reference 3g (alpha=0) and 3j (alpha=1) val Brier explicitly
                # so we can SEE the curvature.
                iid_brier  = next(r["val_brier"] for r in grid_results if r["alpha"] == 0.0)
                full_brier = next(r["val_brier"] for r in grid_results if r["alpha"] == 1.0)
                val_brier_by_lead[lead] = {
                    "best_alpha": best_a, "best_brier": best_b,
                    "alpha_0_brier": iid_brier, "alpha_1_brier": full_brier,
                    "improvement_pct_vs_3g_val": float(100 * (iid_brier - best_b) / iid_brier),
                    "improvement_pct_vs_3j_val": float(100 * (full_brier - best_b) / full_brier),
                    "grid": grid_results,
                }

                # Held-out test scoring at the fitted alpha.
                test_q = q_arr[val_end:]
                test_obs = obs_arr[val_end:]
                rng = np.random.default_rng(SEED + 1)
                test_probs, test_labels = evaluate_alpha(test_q, test_obs, window, best_a, corr, rng)
                test_b = brier(test_probs, test_labels)
                # 3g reference on the same test slice
                rng = np.random.default_rng(SEED + 1)
                iid_test_probs, _ = evaluate_alpha(test_q, test_obs, window, 0.0, corr, rng)
                iid_test_b = brier(iid_test_probs, test_labels)
                # 3j reference on the same test slice
                rng = np.random.default_rng(SEED + 1)
                full_test_probs, _ = evaluate_alpha(test_q, test_obs, window, 1.0, corr, rng)
                full_test_b = brier(full_test_probs, test_labels)

                print(f"  {station} window {window}h lead {lead}h: "
                      f"best_alpha={best_a:.2f}  val Brier={best_b:.4f} (3g={iid_brier:.4f}, 3j={full_brier:.4f}); "
                      f"TEST {test_b:.4f}  (3g_test={iid_test_b:.4f}, 3j_test={full_test_b:.4f})  "
                      f"{100*(iid_test_b - test_b)/iid_test_b:+.1f}% vs 3g  /  "
                      f"{100*(full_test_b - test_b)/full_test_b:+.1f}% vs 3j")

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
                    "Version": version, "Target": "dry_window", "Phase": "3m",
                    "Architecture": "alpha_shrinkage_copula",
                    "MCSamples": MC_SAMPLES, "Seed": SEED,
                    "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                    "AlphaGrid": ALPHA_GRID.round(3).tolist(),
                    "FittedAlphaByLead": alpha_by_lead,
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
