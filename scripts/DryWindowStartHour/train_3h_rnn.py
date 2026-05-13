"""Phase 3h — RNN over 3a's daytime hourly P(wet) sequence for P(dry-window).

Trains a small bidirectional GRU per (station, window, lead) that consumes the
9-hour daytime P(wet) vector from 3a's replay parquet and outputs P(∃ contiguous
N-hour dry block in the daytime window). Tests whether explicit temporal modeling
beats iid Monte Carlo (= 3g) on the same input.

Input: precipitation_replay parquets (hourly ProbWet + Label across full
historical period, written by PrecipReplayCommand for 3g training). Path:
    data/predictions/precipitation_replay/{station}/{precip3aVersion}/
        lead_{L}h.parquet
Schema: ValidTimeUtc, LeadHours, ProbWet, Label  (one row per (valid_time, lead))

For each (station, lead, target_date):
  1. Find the daytime UTC hour range (DST-aware via zoneinfo).
  2. Extract the 9-hour ProbWet vector for that range (skip days with any gap).
  3. Extract the 9-hour binary truth vector for the same range.
  4. Derive the dry-window label per N: contiguous run of >=N zeros in the
     daytime range.

Same chronological 70/15/15 split as 3b/3f so the held-out test slice aligns
with the existing bake-off (~2025-12-29 -> end of replay).

Output: data/models/dry_window/{station}/window_{N}h/v..._phase3h/
    rnn_lead_24h.pt
    rnn_lead_48h.pt
    rnn_lead_72h.pt
    preprocess.json    (input normalisation + RNN config)
    test_predictions.parquet  (same schema as 3b/3f - target_date, station,
                                window, lead, p_dry_window, observed_dry_window)
    training_metadata.json

Bake-off pickup: dry_window_4way_bakeoff.py's find_dry_window_test_predictions
with phase_suffix="phase3h" reads it for free.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Allow `import dry_window_4way_bakeoff` so we can reuse the daytime helper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours  # type: ignore

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"
REPLAY_ROOT = ROOT / "data" / "predictions" / "precipitation_replay"
DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
SEED = 42

# Architecture sized for the tabular small-N regime, same prior as 3f:
# ~700 day rows per (station, window, lead) cell.
HIDDEN = 16
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def find_replay_dir(station: str, phase_suffix: str | None) -> Path | None:
    """Return the newest replay dir for this station matching the given phase
    suffix. ``phase_suffix=None`` selects the unsuffixed 3a champion convention;
    ``"phase3e"`` selects the 3e MLP replay. Newest by directory name wins."""
    station_dir = REPLAY_ROOT / station
    if not station_dir.is_dir():
        return None
    candidates: list[Path] = []
    for d in station_dir.iterdir():
        if not d.is_dir():
            continue
        if phase_suffix:
            if phase_suffix not in d.name:
                continue
        else:
            if "phase" in d.name:
                continue
        candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def has_contiguous_dry_block(binary: np.ndarray, window: int) -> bool:
    """True if the binary sequence (1=wet, 0=dry) contains >=window consecutive zeros."""
    run = 0
    for v in binary:
        if v == 0:
            run += 1
            if run >= window:
                return True
        else:
            run = 0
    return False


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def build_daytime_cells(replay_dir: Path, lead: int) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    """Read replay parquet for one lead, slice to daytime, return:
       - target_dates: list of pd.Timestamp (UTC date)
       - q: ndarray (n_days, 9) of P(wet)
       - obs: ndarray (n_days, 9) of 0/1 observed_wet"""
    df = pd.read_parquet(replay_dir / f"lead_{lead}h.parquet")
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    cells_dates: list[pd.Timestamp] = []
    cells_q: list[np.ndarray] = []
    cells_obs: list[np.ndarray] = []
    for target_date, grp in df.groupby("target_date"):
        target_ts = pd.Timestamp(target_date)
        start_utc, end_utc = daytime_utc_hours(target_ts)
        n_expected = end_utc - start_utc
        sub = grp[(grp["hour"] >= start_utc) & (grp["hour"] < end_utc)].sort_values("valid_time")
        if len(sub) != n_expected:
            continue
        q_vec = sub["ProbWet"].to_numpy(dtype="float32")
        obs_vec = sub["Label"].astype(np.int32).to_numpy()
        cells_dates.append(target_ts)
        cells_q.append(q_vec)
        cells_obs.append(obs_vec)
    return cells_dates, np.stack(cells_q), np.stack(cells_obs)


class DryWindowGRU(nn.Module):
    """Bidirectional GRU over the 9-hour P(wet) sequence, single output via
    sigmoid. Hidden=16, dropout=0.3 — same small-N regime as 3f's MLP. The
    bidirection matters because at predict time the whole 9-hour vector is
    known up-front; no streaming."""

    def __init__(self, input_dim: int = 1, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden, num_layers=1,
                           batch_first=True, bidirectional=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq=9, input_dim=1) -> output (batch,)
        out, _ = self.gru(x)
        last = out[:, -1, :]   # final timestep concat of fwd + bwd hidden states
        last = self.dropout(last)
        logit = self.head(last).squeeze(-1)
        return logit  # raw logit; caller applies sigmoid


def train_one_cell(station: str, lead: int, window: int,
                   dates: list[pd.Timestamp], q_arr: np.ndarray, obs_arr: np.ndarray,
                   verbose: bool = True) -> tuple[np.ndarray, list[pd.Timestamp], np.ndarray]:
    """Train a GRU for (station, lead, window). Return test predictions +
    matching target_dates + truth labels for the test slice."""
    n = len(dates)
    train_end = int(np.floor(n * TRAIN_FRAC))
    val_end = train_end + int(np.floor(n * VAL_FRAC))

    labels = np.array([1 if has_contiguous_dry_block(obs_arr[i], window) else 0 for i in range(n)],
                      dtype="float32")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train = torch.tensor(q_arr[:train_end][:, :, None])
    y_train = torch.tensor(labels[:train_end])
    x_val   = torch.tensor(q_arr[train_end:val_end][:, :, None])
    y_val   = torch.tensor(labels[train_end:val_end])
    x_test  = torch.tensor(q_arr[val_end:][:, :, None])
    y_test  = torch.tensor(labels[val_end:])

    if verbose:
        print(f"  split: train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]} "
              f"(label rate train={labels[:train_end].mean():.2f}, test={labels[val_end:].mean():.2f})")

    model = DryWindowGRU()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    best_val_brier = float("inf")
    best_state: dict | None = None
    epochs_since_best = 0
    epochs_run = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val)).numpy()
        val_brier = float(np.mean((val_probs - labels[train_end:val_end]) ** 2))

        if val_brier < best_val_brier:
            best_val_brier = val_brier
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(x_test)).numpy()

    if verbose:
        clim = float(labels[:train_end].mean())
        clim_pred = np.full_like(labels[val_end:], clim, dtype="float32")
        clim_brier = float(np.mean((clim_pred - labels[val_end:]) ** 2))
        test_brier = float(np.mean((test_probs - labels[val_end:]) ** 2))
        bss = (clim_brier - test_brier) / clim_brier if clim_brier > 0 else float("nan")
        print(f"  GRU test Brier={test_brier:.4f}  clim={clim_brier:.4f}  BSS={bss:+.4f}  "
              f"best_val={best_val_brier:.4f}  epochs={epochs_run}")

    return test_probs, dates[val_end:], labels[val_end:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--source", default="3a", choices=["3a", "3e"],
                    help="Which replay parquet feeds the GRU's hourly input — 3a (unsuffixed) or 3e (MLP).")
    ap.add_argument("--phase-tag", default=None,
                    help="Override the bundle suffix (default: 'phase3h' for source=3a, 'phase3h_3e' for source=3e). "
                         "Lets the bake-off discriminate between the two GRU variants.")
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    source = args.source
    replay_phase_suffix = "phase3e" if source == "3e" else None
    # Default phase tags: phase3h for 3a-sourced GRU (the original 3h), phase3h_3e
    # for the 3e-sourced variant. Override via --phase-tag if running a
    # different experiment.
    phase_tag = args.phase_tag or ("phase3h_3e" if source == "3e" else "phase3h")

    print(f"Phase 3h RNN training — stations={stations} leads={leads} windows={windows}")
    print(f"  source: {source} replay (phase_suffix={replay_phase_suffix or '(unsuffixed champion = 3a)'})")
    print(f"  bundle tag: {phase_tag}")
    print(f"  arch=GRU(hidden={HIDDEN}, dropout={DROPOUT}) lr={LR} bs={BATCH_SIZE} "
          f"max_epochs={MAX_EPOCHS} early_stop={EARLY_STOP_PATIENCE}")

    overall_start = datetime.now(timezone.utc)

    for station in stations:
        replay_dir = find_replay_dir(station, replay_phase_suffix)
        if replay_dir is None:
            print(f"::warning::{station}: no {source} replay dir under {REPLAY_ROOT}; skipping")
            continue
        print(f"\n=== {station} (replay {replay_dir.name}) ===")

        for window in windows:
            ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
            version = f"v{ts}_{phase_tag}"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)

            test_rows: list[dict] = []
            for lead in leads:
                print(f"-- {station} window {window}h lead {lead}h --")
                replay_lead = replay_dir / f"lead_{lead}h.parquet"
                if not replay_lead.exists():
                    print(f"  (skip) no replay parquet at {replay_lead}")
                    continue

                dates, q_arr, obs_arr = build_daytime_cells(replay_dir, lead)
                if len(dates) < 100:
                    print(f"  (skip) only {len(dates)} daytime-complete days")
                    continue

                test_probs, test_dates, test_labels = train_one_cell(
                    station, lead, window, dates, q_arr, obs_arr,
                )
                for td, p, y in zip(test_dates, test_probs, test_labels):
                    test_rows.append({
                        "target_date": td,
                        "station": station,
                        "window": window,
                        "lead": lead,
                        "p_dry_window": float(p),
                        "observed_dry_window": np.uint8(int(y)),
                    })

            if test_rows:
                out_path = bundle_dir / "test_predictions.parquet"
                pd.DataFrame(test_rows).to_parquet(out_path, index=False)
                print(f"  wrote {len(test_rows)} test_predictions -> {out_path}")

                # Minimal training_metadata for bake-off discovery.
                (bundle_dir / "training_metadata.json").write_text(json.dumps({
                    "Version": version,
                    "Target": "dry_window",
                    "Phase": phase_tag.replace("phase", ""),   # "3h" or "3h_3e"
                    "Architecture": "bidirectional_gru",
                    "InputSource": source,
                    "Hyperparameters": {
                        "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                        "early_stop_patience": EARLY_STOP_PATIENCE,
                        "seed": SEED, "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC,
                    },
                    "ReplayVersion": replay_dir.name,
                    "WindowHours": window,
                    "Leads": leads,
                    "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
                }, indent=2))
            else:
                print(f"  no test predictions for {station} window {window}h — bundle empty")

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
