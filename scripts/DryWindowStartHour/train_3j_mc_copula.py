"""Phase 3j — Markov / Gaussian-copula MC over the daytime 9-hour q-vector.

3g samples iid Bernoullis per hour. Real rain clusters, so independence is
wrong. 3j fits a 9x9 Pearson-correlation matrix Σ from historical observed
daytime wet/dry sequences, then samples each test day's 9-hour binary
sequence from a Gaussian copula:
    Z ~ N(0, Σ);  U_h = Φ(Z_h);  X_h = 1[U_h < q_h]
This yields marginals P(X_h=1) = q_h (= 3a's hourly P(wet)) PLUS a
dependence structure that matches the historical autocorrelation pattern.

Same chronological 70/15/15 split as 3h: fit Σ from the TRAIN slice's
observed daytime binary sequences, evaluate on the TEST slice.

Inputs: 3a replay parquets (full history of hourly ProbWet + Label) under
    data/predictions/precipitation_replay/{station}/v..._unsuffixed/
Outputs: data/models/dry_window/{station}/window_{N}h/v..._phase3j/
    test_predictions.parquet + training_metadata.json with the per-station
    fitted Σ saved for inspection.

Bake-off discovery via find_3j_test_predictions in dry_window_4way_bakeoff.py.
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

# SciPy is in the venv (5a uses it).
from scipy.stats import norm

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


def find_replay_dir(station: str) -> Path | None:
    station_dir = REPLAY_ROOT / station
    if not station_dir.is_dir():
        return None
    cands = [d for d in station_dir.iterdir() if d.is_dir() and "phase" not in d.name]
    return max(cands, key=lambda d: d.name) if cands else None


def build_daytime_cells(replay_dir: Path, lead: int) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    """Per (target_date) at this lead: daytime q-vector + truth vector."""
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
    """9x9 Pearson correlation on the observed binary daytime sequences.
    obs_seqs shape: (n_days, 9). Adds small jitter to keep PSD."""
    if len(obs_seqs) < 10:
        return np.eye(obs_seqs.shape[1])
    # Pearson on binary is the phi-coefficient; cheap proxy for tetrachoric.
    corr = np.corrcoef(obs_seqs.T)
    # Jitter the diagonal to ensure PSD for Cholesky (handles rank-deficient
    # cases where two daytime hours happen to be 100% correlated in train).
    corr = corr + 1e-6 * np.eye(corr.shape[0])
    return corr


def copula_sample_dry_window(q: np.ndarray, corr: np.ndarray, window: int,
                              n_samples: int, rng: np.random.Generator) -> float:
    """Gaussian copula: sample n_samples 9-hour Bernoulli sequences with
    marginal P(X_h=1) = q_h and Gaussian-correlation Σ = corr. Return the
    fraction of samples whose binary sequence contains a contiguous run
    of >= window zeros."""
    n = len(q)
    if window > n:
        return 0.0
    # Cholesky of correlation matrix; fallback to identity if not PSD.
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.eye(n)
    # Generate iid standard normals then correlate.
    z_iid = rng.standard_normal((n_samples, n))
    z = z_iid @ L.T   # row k = L @ z_iid_k => has cov = L L^T = corr
    # Φ(z) ~ Uniform[0,1]; X_h = 1[U < q_h] gives P(X_h=1) = q_h.
    u = norm.cdf(z)
    samples = (u < q[None, :]).astype(np.int32)
    # Count runs of zeros >= window in each row. Vectorised.
    return float(np.mean([has_contiguous_dry_block(s, window) for s in samples]))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--correlation-pool", default="per-lead", choices=["per-lead", "per-station"],
                    help="Fit Σ per (station, lead) or pool across leads per station. "
                         "Per-lead is more accurate; per-station is more stable. Default: per-lead.")
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    pool = args.correlation_pool

    print(f"Phase 3j (Gaussian copula MC) — stations={stations} leads={leads} windows={windows}")
    print(f"  MC samples: {MC_SAMPLES}; seed: {SEED}; corr fit: train slice; pool: {pool}")

    rng = np.random.default_rng(SEED)
    overall_start = datetime.now(timezone.utc)

    for station in stations:
        replay_dir = find_replay_dir(station)
        if replay_dir is None:
            print(f"::warning::{station}: no 3a replay; skipping")
            continue
        print(f"\n=== {station} (3a replay {replay_dir.name}) ===")

        # Load all leads once; index by lead.
        cells_by_lead: dict[int, tuple[list[pd.Timestamp], np.ndarray, np.ndarray]] = {}
        for lead in leads:
            cells_by_lead[lead] = build_daytime_cells(replay_dir, lead)

        # Fit correlation matrix. Train slice = first 70% of (chronologically
        # ordered) daytime-complete days. Per-lead fit by default.
        corr_by_lead: dict[int, np.ndarray] = {}
        if pool == "per-lead":
            for lead in leads:
                dates, _, obs_arr = cells_by_lead[lead]
                if not dates:
                    continue
                n = len(dates)
                tr_end = int(np.floor(n * TRAIN_FRAC))
                corr_by_lead[lead] = fit_correlation(obs_arr[:tr_end])
                print(f"  lead {lead}h: corr fit on {tr_end} train days; "
                      f"mean off-diagonal = {corr_by_lead[lead][np.triu_indices_from(corr_by_lead[lead], k=1)].mean():.3f}")
        else:
            # Pool across leads: stack train slices, fit ONE 9x9 Σ.
            train_obs: list[np.ndarray] = []
            for lead in leads:
                dates, _, obs_arr = cells_by_lead[lead]
                if not dates:
                    continue
                tr_end = int(np.floor(len(dates) * TRAIN_FRAC))
                train_obs.append(obs_arr[:tr_end])
            corr_one = fit_correlation(np.concatenate(train_obs)) if train_obs else np.eye(9)
            for lead in leads:
                corr_by_lead[lead] = corr_one
            print(f"  pooled corr fit on {sum(len(x) for x in train_obs)} train days; "
                  f"mean off-diagonal = {corr_one[np.triu_indices_from(corr_one, k=1)].mean():.3f}")

        # For each (window): walk the test slice (last 15%) of each lead and
        # compute the copula-sampled dry-window probability.
        for window in windows:
            ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
            version = f"v{ts}_phase3j"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []

            for lead in leads:
                dates, q_arr, obs_arr = cells_by_lead[lead]
                if not dates:
                    continue
                corr = corr_by_lead.get(lead)
                if corr is None:
                    continue
                n = len(dates)
                tr_end = int(np.floor(n * TRAIN_FRAC))
                val_end = tr_end + int(np.floor(n * VAL_FRAC))
                test_dates = dates[val_end:]
                test_q = q_arr[val_end:]
                test_obs = obs_arr[val_end:]
                if len(test_dates) == 0:
                    continue

                # Per-test-day: copula MC. Reuse rng so seed across cells stays
                # deterministic but each cell gets independent draws.
                probs = np.zeros(len(test_dates))
                labels = np.zeros(len(test_dates), dtype="int32")
                for i in range(len(test_dates)):
                    probs[i] = copula_sample_dry_window(test_q[i], corr, window, MC_SAMPLES, rng)
                    labels[i] = 1 if has_contiguous_dry_block(test_obs[i], window) else 0

                b = brier(probs, labels.astype(float))
                clim = float(np.mean([1 if has_contiguous_dry_block(obs_arr[k], window) else 0
                                      for k in range(tr_end)]))
                clim_b = brier(np.full(len(labels), clim), labels.astype(float))
                bss = (clim_b - b) / clim_b if clim_b > 0 else float("nan")
                print(f"  {station} window {window}h lead {lead}h: "
                      f"copula Brier={b:.4f}  clim={clim_b:.4f}  BSS={bss:+.4f}  n_test={len(test_dates)}")

                for td, p, y in zip(test_dates, probs, labels):
                    test_rows.append({
                        "target_date": td,
                        "station": station,
                        "window": window,
                        "lead": lead,
                        "p_dry_window": float(p),
                        "observed_dry_window": np.uint8(int(y)),
                    })

            if test_rows:
                pd.DataFrame(test_rows).to_parquet(bundle_dir / "test_predictions.parquet", index=False)
                # Save the fitted correlation matrix per lead for inspection.
                meta = {
                    "Version": version,
                    "Target": "dry_window",
                    "Phase": "3j",
                    "Architecture": "gaussian_copula_mc",
                    "MCSamples": MC_SAMPLES,
                    "Seed": SEED,
                    "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                    "CorrelationPool": pool,
                    "WindowHours": window,
                    "Leads": leads,
                    "ReplayVersion": replay_dir.name,
                    "FittedCorrelationByLead": {
                        str(lead): corr_by_lead.get(lead, np.eye(9)).round(4).tolist()
                        for lead in leads
                    },
                    "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
                }
                (bundle_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
                print(f"  wrote {len(test_rows)} rows -> {bundle_dir / 'test_predictions.parquet'}")

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
