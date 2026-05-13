"""Phase 3i — GRU over raw per-NWP daytime hourly precip sequences.

3h's input was a single hourly P(wet) channel (3a or 3e blender output) — a
pre-blended summary of the NWPs. 3i feeds the GRU the RAW per-NWP precip
rates as a multi-channel sequence: (9 daytime hours, N NWPs) per
(station, lead, target_date). Tests whether the GRU can learn a better
hourly blend + temporal model end-to-end than the existing pipeline
(3a hourly LightGBM blender + iid MC sampling at the daytime step).

Inputs:
  - data/forecasts/  (offset_day forecasts per (location, model, date, run))
  - data/truth/rainfall/  (15-min EA gauge readings → hourly wet/dry)

Per (station, lead, target_date) the script:
  1. Picks the daytime UTC hour range (DST-aware via zoneinfo).
  2. For each daytime hour, picks the latest offset_day forecast per Model
     with LeadHours = {lead} and ValidTimeUtc matching the target hour.
     This mirrors the SQL in DryWindowFeatureBuilder + the existing 3a /
     3e replay queries.
  3. Stacks the per-Model precip into a 9 x N matrix (N = configured NWPs).
     Missing models for a slot leave NaN → imputed to 0 before standardisation.
  4. Computes daytime dry-window label from hourly EA rainfall (>= window
     consecutive dry hours).

Same 70/15/15 chronological split as 3h so the bake-off's test slice
aligns (replay parquet end-date ~ 2026-04-28).

Output: data/models/dry_window/{station}/window_{N}h/v..._phase3i/
    rnn_lead_*.pt + preprocess.json + test_predictions.parquet +
    training_metadata.json

Bake-off auto-discovers via dry_window_4way_bakeoff.py's find_3i pattern
(added in the same commit as this script).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"
FORECASTS_ROOT = ROOT / "data" / "forecasts"
RAINFALL_ROOT  = ROOT / "data" / "truth" / "rainfall"

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
DEFAULT_LOCATION = "bonehill_rocks"

# NWP precip-source set. Mirrors DryWindowFeatureBuilder's OptionalModels for
# Phase 3b — every model with offset_day forecasts at the relevant leads.
# The full row is 7 channels at typical (station, lead) cells; AIFS may be
# missing for older dates. NaN handled per-row via mean imputation post-norm.
DEFAULT_MODELS = [
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "meteofrance_seamless",
    "gem_seamless",
    "ecmwf_aifs025_single",
    "jma_seamless",
]
WET_THRESHOLD_MM = 0.1

SEED = 42
HIDDEN = 16
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def map_station_to_friendly(slug: str) -> str:
    """Slug -> config friendly name. Hard-coded because the C#
    ResolveFriendlyStationName reads from AppConfig and exact case matters
    for the rainfall truth join (StationName='Dartmoor nr Hexworthy' has
    lowercase 'nr', a naive title-case map breaks the lookup -> 0 cells)."""
    overrides = {
        "ea_bellever_dartmoor":         "Bellever Dartmoor",
        "ea_bovey_tracey":              "Bovey Tracey",
        "ea_dartmoor_nr_hexworthy":     "Dartmoor nr Hexworthy",
    }
    if slug in overrides:
        return overrides[slug]
    bare = slug[3:] if slug.startswith("ea_") else slug
    return " ".join(w.capitalize() for w in bare.split("_"))


def has_contiguous_dry_block(binary: np.ndarray, window: int) -> bool:
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


# ----------------------------------------------------------------------
# Data load
# ----------------------------------------------------------------------

def load_offset_day_precip(location: str, models: list[str], leads: list[int]) -> pd.DataFrame:
    """Load offset_day forecasts for all models + leads in one query, picking
    the latest RunTime per (ValidTime, Model, Lead). Returns long-form:
        valid_time, lead, model, precip
    """
    fc_glob = str(FORECASTS_ROOT / "**" / "*.parquet").replace("\\", "/")
    models_in = "(" + ",".join(f"'{m}'" for m in models) + ")"
    leads_in = "(" + ",".join(str(L) for L in leads) + ")"
    sql = f"""
WITH latest AS (
    SELECT ValidTimeUtc, Model, LeadHours, Precipitation,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model, LeadHours
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{location}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours IN {leads_in}
      AND Model IN {models_in}
)
SELECT ValidTimeUtc AS valid_time, LeadHours AS lead,
       Model AS model, Precipitation AS precip
FROM latest WHERE rn = 1
ORDER BY valid_time, lead, model
"""
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
    return df


def load_hourly_truth(location: str, friendly_station: str) -> pd.DataFrame:
    """Load 15-min EA rainfall, aggregate to hourly with 4-of-4 quality gate
    (same rule as DryWindowFeatureBuilder.LoadHourlyTruth)."""
    rf_glob = str(RAINFALL_ROOT / "**" / "*.parquet").replace("\\", "/")
    sql = f"""
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{rf_glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{location}'
  AND StationName  = '{friendly_station}'
  AND Value15MinMm IS NOT NULL
GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1
"""
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
    df["wet"] = (df["mm_hour"] >= WET_THRESHOLD_MM).astype(np.int32)
    return df


def build_cells_for_lead(precip_long: pd.DataFrame, hourly_truth: pd.DataFrame,
                         lead: int, models: list[str]) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    """For one lead: per (target_date) build a (9, N_models) precip matrix
    over the daytime UTC hours and a (9,) wet/dry truth vector.
    Drop days that are missing any required hour in EITHER axis.
    Returns: target_dates, precip_tensor (n_days, 9, N), truth_tensor (n_days, 9).
    """
    sub_precip = precip_long[precip_long["lead"] == lead].copy()
    sub_precip["target_date"] = sub_precip["valid_time"].dt.normalize()
    sub_precip["hour"] = sub_precip["valid_time"].dt.hour

    truth_lookup: dict[pd.Timestamp, int] = dict(
        zip(hourly_truth["valid_time"], hourly_truth["wet"]),
    )

    # Build per-(target_date) day-of-precip matrix
    model_idx = {m: i for i, m in enumerate(models)}
    n_models = len(models)

    days: list[pd.Timestamp] = []
    precip_mats: list[np.ndarray] = []
    truth_vecs: list[np.ndarray] = []

    for target_date, grp in sub_precip.groupby("target_date"):
        target_ts = pd.Timestamp(target_date)
        start_utc, end_utc = daytime_utc_hours(target_ts)
        n_hours = end_utc - start_utc

        # Forecast matrix per (hour_offset, model) — start with NaN
        mat = np.full((n_hours, n_models), np.nan, dtype="float32")
        truth = np.full(n_hours, -1, dtype="int32")
        all_truth_present = True

        for h_idx, h in enumerate(range(start_utc, end_utc)):
            slot_time = target_ts + pd.Timedelta(hours=h)
            slot_rows = grp[grp["hour"] == h]
            for _, row in slot_rows.iterrows():
                mi = model_idx.get(row["model"])
                if mi is None:
                    continue
                p = row["precip"]
                if pd.notna(p):
                    mat[h_idx, mi] = p

            # Truth for this hour
            t = truth_lookup.get(slot_time)
            if t is None:
                all_truth_present = False
                break
            truth[h_idx] = t

        if not all_truth_present:
            continue
        # Require at least one channel has any non-NaN value (otherwise the
        # day is genuinely useless). Don't require EVERY model to be present
        # — AIFS only landed 2026-04-27 (memory: project_aifs_shipped), so a
        # strict "every model" filter would drop the entire pre-April 2026
        # train period. Missing channels stay NaN here; the standardiser
        # imputes them to channel-mean (= 0 after z-score), so the GRU sees
        # a neutral signal for those slots rather than rejecting the row.
        if np.isnan(mat).all():
            continue

        days.append(target_ts)
        precip_mats.append(mat)
        truth_vecs.append(truth)

    if not days:
        return [], np.zeros((0, 0, 0), dtype="float32"), np.zeros((0, 0), dtype="int32")
    return days, np.stack(precip_mats), np.stack(truth_vecs)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class DryWindowGruNwp(nn.Module):
    """Bidirectional GRU consuming the (9-hour, N-channel) per-NWP precip
    sequence. Pre-standardised inputs (z-score per channel). NaNs → 0
    after standardisation (= channel mean), same dead-column logic as
    the MLP standardiser."""

    def __init__(self, input_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden, num_layers=1,
                           batch_first=True, bidirectional=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score using train slice, NaN-aware. Returns standardised
    tensor + mean + scale. Dead channels (all-NaN in train) get (mean=0, scale=1)
    so post-standardisation NaN→0 imputation is the channel "mean" — same
    rule the MLP standardiser shipped 2026-05-13."""
    n_channels = x.shape[-1]
    flat = x.reshape(-1, n_channels)
    mean = np.zeros(n_channels, dtype="float32")
    scale = np.ones(n_channels, dtype="float32")
    for k in range(n_channels):
        col = flat[:, k]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        mean[k] = valid.mean()
        s = valid.std()
        scale[k] = max(s, 1e-6)
    standardised = (x - mean) / scale
    standardised = np.where(np.isnan(standardised), 0.0, standardised).astype("float32")
    return standardised, mean, scale


def train_one_cell(station: str, lead: int, window: int,
                   dates: list[pd.Timestamp], precip_arr: np.ndarray, truth_arr: np.ndarray,
                   verbose: bool = True) -> tuple[np.ndarray, list[pd.Timestamp], np.ndarray]:
    n = len(dates)
    train_end = int(np.floor(n * TRAIN_FRAC))
    val_end = train_end + int(np.floor(n * VAL_FRAC))

    labels = np.array([1 if has_contiguous_dry_block(truth_arr[i], window) else 0 for i in range(n)],
                      dtype="float32")

    # Standardise per-channel on TRAIN slice ONLY (no test leakage).
    x_train_raw = precip_arr[:train_end]
    x_train_s, mean, scale = standardise(x_train_raw)
    x_val_raw = precip_arr[train_end:val_end]
    x_val_s = ((x_val_raw - mean) / scale)
    x_val_s = np.where(np.isnan(x_val_s), 0.0, x_val_s).astype("float32")
    x_test_raw = precip_arr[val_end:]
    x_test_s = ((x_test_raw - mean) / scale)
    x_test_s = np.where(np.isnan(x_test_s), 0.0, x_test_s).astype("float32")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_train = torch.tensor(x_train_s)
    y_train = torch.tensor(labels[:train_end])
    x_val   = torch.tensor(x_val_s)
    y_val_arr = labels[train_end:val_end]
    x_test  = torch.tensor(x_test_s)

    if verbose:
        print(f"  split: train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]} "
              f"(label rate train={labels[:train_end].mean():.2f}, test={labels[val_end:].mean():.2f})")

    n_channels = precip_arr.shape[-1]
    model = DryWindowGruNwp(input_dim=n_channels)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    best_val = float("inf")
    best_state: dict | None = None
    epochs_since_best = 0
    epochs_run = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val)).numpy()
        val_b = float(np.mean((val_probs - y_val_arr) ** 2))
        if val_b < best_val:
            best_val = val_b
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
    test_labels = labels[val_end:]
    if verbose:
        clim = float(labels[:train_end].mean())
        clim_pred = np.full_like(test_labels, clim, dtype="float32")
        clim_b = float(np.mean((clim_pred - test_labels) ** 2))
        test_b = float(np.mean((test_probs - test_labels) ** 2))
        bss = (clim_b - test_b) / clim_b if clim_b > 0 else float("nan")
        print(f"  GRU-NWP test Brier={test_b:.4f}  clim={clim_b:.4f}  BSS={bss:+.4f}  "
              f"best_val={best_val:.4f}  epochs={epochs_run}")

    return test_probs, dates[val_end:], test_labels


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--location", default=DEFAULT_LOCATION)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    location = args.location
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Phase 3i (RNN on raw per-NWP daytime hourly precip) — stations={stations} "
          f"leads={leads} windows={windows}")
    print(f"  arch=GRU({len(models)}-channel input, hidden={HIDDEN}, dropout={DROPOUT})")
    print(f"  models: {models}")
    print(f"  loading offset_day precip for all leads at location={location} ...")
    precip_long = load_offset_day_precip(location, models, leads)
    print(f"  precip rows: {len(precip_long):,}; "
          f"date range {precip_long['valid_time'].min():%Y-%m-%d} -> {precip_long['valid_time'].max():%Y-%m-%d}")

    overall_start = datetime.now(timezone.utc)
    for station in stations:
        friendly = map_station_to_friendly(station)
        print(f"\n=== {station} (friendly: '{friendly}') ===")
        hourly_truth = load_hourly_truth(location, friendly)
        print(f"  truth rows: {len(hourly_truth):,}")

        for window in windows:
            ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
            version = f"v{ts}_phase3i"
            bundle_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h" / version
            bundle_dir.mkdir(parents=True, exist_ok=True)
            test_rows: list[dict] = []

            for lead in leads:
                print(f"-- {station} window {window}h lead {lead}h --")
                dates, precip_arr, truth_arr = build_cells_for_lead(
                    precip_long, hourly_truth, lead, models,
                )
                if len(dates) < 100:
                    print(f"  (skip) only {len(dates)} daytime-complete days")
                    continue
                test_probs, test_dates, test_labels = train_one_cell(
                    station, lead, window, dates, precip_arr, truth_arr,
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
                (bundle_dir / "training_metadata.json").write_text(json.dumps({
                    "Version": version,
                    "Target": "dry_window",
                    "Phase": "3i",
                    "Architecture": "bidirectional_gru_nwp",
                    "Models": models,
                    "Hyperparameters": {
                        "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                        "early_stop_patience": EARLY_STOP_PATIENCE,
                        "seed": SEED, "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC,
                    },
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
