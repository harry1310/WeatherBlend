"""Phase 5b — Bayesian hierarchical logistic regression on 3p ensemble features.

Last attempt at the ensemble-features idea before we park it.

The 3p (sklearn LR) and 3q (LightGBM) variants both lost on aggregate
despite the features carrying strong AUC signal (0.85-0.91). Hypothesis:
per-cell models can't extract that signal at ~600 train days, and
information that could be shared across cells (the per-feature
coefficient structure) is being relearned 27 times.

5b fits ONE model per window (3 total) with:
  - Cell-specific intercepts α_c (partial pool toward μ_α)
  - Shared coefficients β (one β per feature, all cells use it)

That's the simplest hierarchical form: pools nothing on the slopes
(strongest pooling) but lets each cell have its own baseline. If
random slopes per cell are needed later, this code is a clean starting
point.

Inputs: data/features/dry_window_3p/features.parquet (16 features +
3 binary labels per (station, lead, target_date)).

Outputs per (station, window, lead):
  data/models/dry_window/{station}/window_{N}h/v..._phase5b/
    test_predictions.parquet
    training_metadata.json  (posterior summary, sampler diagnostics)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Silence deprecation chatter from pymc / pytensor on numpy 2.x.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pymc as pm  # noqa: E402

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

# NUTS sampler config — kept conservative because ~6000 rows × 16 features
# fits quickly. 1000 tune + 1000 draws × 2 chains is enough for convergence
# on this kind of conjugate-ish problem.
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 2


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    args = ap.parse_args()
    windows = [int(s) for s in args.windows.split(",")]

    if not FEATURES_PATH.exists():
        print(f"::error::features.parquet not found at {FEATURES_PATH}")
        return 1

    df = pd.read_parquet(FEATURES_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None)
    feature_cols = [c for c in df.columns if c.startswith("f") and "_" in c]
    print(f"Phase 5b (Bayesian hierarchical LR — shared β, per-cell α)")
    print(f"  features.parquet: {len(df):,} rows, {len(feature_cols)} features")
    print(f"  NUTS: tune={N_TUNE}, draws={N_DRAWS}, chains={N_CHAINS}, seed={SEED}")

    overall_start = datetime.now(timezone.utc)
    ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
    version = f"v{ts}_phase5b"

    # Cell encoding: stable ordering across all stations × leads.
    cells = sorted({(s, L) for s in df["station"].unique() for L in df["lead"].unique()})
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    n_cells = len(cells)

    for window in windows:
        print(f"\n=== window {window}h ===")
        label_col = f"dry_{window}h"

        # Build the per-cell chronological split, concat into one big train/val/test.
        rows_train: list[pd.DataFrame] = []
        rows_test: list[pd.DataFrame] = []
        for (s, L), c_idx in cell_to_idx.items():
            cell = df[(df["station"] == s) & (df["lead"] == L)].sort_values("target_date").reset_index(drop=True)
            n = len(cell)
            tr_end = int(np.floor(n * TRAIN_FRAC))
            val_end = tr_end + int(np.floor(n * VAL_FRAC))
            # We don't actually NEED val for vanilla hierarchical LR — no early
            # stopping, no PAV (PAV reverted today). Pool train+val into one
            # training set for more posterior data.
            train_chunk = cell.iloc[:val_end].copy()
            test_chunk = cell.iloc[val_end:].copy()
            train_chunk["cell"] = c_idx
            test_chunk["cell"] = c_idx
            rows_train.append(train_chunk)
            rows_test.append(test_chunk)

        train_df = pd.concat(rows_train, ignore_index=True)
        test_df = pd.concat(rows_test, ignore_index=True)
        print(f"  train rows: {len(train_df):,} across {n_cells} cells; test rows: {len(test_df):,}")

        # Standardise features on train.
        mu = train_df[feature_cols].mean().to_numpy()
        sd = train_df[feature_cols].std().to_numpy()
        sd = np.where(sd < 1e-6, 1.0, sd)
        X_train = ((train_df[feature_cols].to_numpy() - mu) / sd).astype("float64")
        X_test = ((test_df[feature_cols].to_numpy() - mu) / sd).astype("float64")
        y_train = train_df[label_col].to_numpy().astype("int32")
        y_test = test_df[label_col].to_numpy().astype("int32")
        cell_train = train_df["cell"].to_numpy().astype("int32")
        cell_test = test_df["cell"].to_numpy().astype("int32")

        # Build the model.
        with pm.Model() as model:
            # Hyperpriors.
            mu_alpha = pm.Normal("mu_alpha", 0.0, 2.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", 2.0)
            # Cell-level intercepts (partial pool around mu_alpha).
            alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, shape=n_cells)
            # Shared coefficients across cells.
            beta = pm.Normal("beta", 0.0, 1.0, shape=len(feature_cols))
            # Linear predictor + Bernoulli likelihood.
            eta = alpha[cell_train] + pm.math.dot(X_train, beta)
            pm.Bernoulli("y_obs", logit_p=eta, observed=y_train)

            # Sample via nutpie (production-default per memory). Falls back
            # to numpy-NUTS if nutpie throws (Windows occasionally does).
            try:
                trace = pm.sample(
                    draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, target_accept=0.9,
                    random_seed=SEED, progressbar=False, nuts_sampler="nutpie")
            except Exception as e:
                print(f"  nutpie failed ({e}); falling back to default NUTS")
                trace = pm.sample(
                    draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, target_accept=0.9,
                    random_seed=SEED, progressbar=False)

        # Posterior predictive at test rows: average σ(α_c + β · x_d) over draws.
        alpha_post = trace.posterior["alpha"].to_numpy()  # (chains, draws, n_cells)
        beta_post  = trace.posterior["beta"].to_numpy()   # (chains, draws, n_features)
        a_flat = alpha_post.reshape(-1, n_cells)
        b_flat = beta_post.reshape(-1, len(feature_cols))
        # Predict in mini-batches to keep memory predictable.
        test_probs = np.zeros(len(X_test))
        batch = 1000
        for i0 in range(0, len(X_test), batch):
            X_b = X_test[i0:i0 + batch]
            c_b = cell_test[i0:i0 + batch]
            # eta_draws: (n_draws, batch). a_flat[:, c_b] is (n_draws, batch); X_b @ b_flat.T is (batch, n_draws) → transpose.
            eta_draws = a_flat[:, c_b] + (X_b @ b_flat.T).T
            test_probs[i0:i0 + batch] = (1.0 / (1.0 + np.exp(-eta_draws))).mean(axis=0)

        # Per-cell Brier reporting.
        per_cell_brier: dict[str, float] = {}
        for (s, L), c_idx in cell_to_idx.items():
            mask = cell_test == c_idx
            if mask.sum() < 10:
                continue
            b = brier(test_probs[mask], y_test[mask].astype("float64"))
            per_cell_brier[f"{s}/lead_{L}h"] = round(b, 4)
            clim = float(y_train[cell_train == c_idx].mean())
            clim_b = brier(np.full(mask.sum(), clim), y_test[mask].astype("float64"))
            print(f"  {s} window {window}h lead {L}h: Brier={b:.4f}  clim={clim_b:.4f}  BSS={(clim_b - b)/clim_b if clim_b > 0 else float('nan'):+.4f}  n_test={int(mask.sum())}")

        # Write bundles per (station, window) — one file per window contains
        # all that station's test rows.
        for (s, L), c_idx in cell_to_idx.items():
            bundle_dir = DRY_WINDOW_MODELS_ROOT / s / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
        # The bake-off expects test_predictions per (station, window) bundle
        # covering all leads for that station. Group rows accordingly.
        for s in df["station"].unique():
            for w_local in [window]:
                bundle_dir = DRY_WINDOW_MODELS_ROOT / s / f"window_{w_local}h" / version
                rows: list[dict] = []
                for L in df["lead"].unique():
                    c_idx = cell_to_idx[(s, int(L))]
                    mask = (cell_test == c_idx)
                    if mask.sum() == 0:
                        continue
                    sub_test = test_df[mask].reset_index(drop=True)
                    sub_probs = test_probs[mask]
                    for td, p, lbl in zip(sub_test["target_date"], sub_probs, sub_test[label_col]):
                        rows.append({
                            "target_date": td, "station": s,
                            "window": w_local, "lead": int(L),
                            "p_dry_window": float(p),
                            "observed_dry_window": np.uint8(int(lbl)),
                        })
                if rows:
                    pd.DataFrame(rows).to_parquet(bundle_dir / "test_predictions.parquet", index=False)
                    meta = {
                        "Version": version, "Target": "dry_window", "Phase": "5b",
                        "Architecture": "bayesian_hierarchical_lr_partial_pool",
                        "FeatureColumns": feature_cols,
                        "NDraws": N_DRAWS, "NTune": N_TUNE, "NChains": N_CHAINS, "Seed": SEED,
                        "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                        "PerCellBrier": per_cell_brier,
                        "WindowHours": w_local, "Leads": [int(L) for L in df["lead"].unique()],
                        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
                    }
                    (bundle_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
                    print(f"  wrote {len(rows)} rows -> {bundle_dir / 'test_predictions.parquet'}")

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
