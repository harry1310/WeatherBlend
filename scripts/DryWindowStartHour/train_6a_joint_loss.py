"""Phase 6a — hourly P(wet) MLP trained with joint loss.

3a's hourly model optimises only hourly Brier (BCE against observed_wet
per hour). The dry-window MC sits downstream and doesn't influence 3a's
training. The diagnostic + 3k/3m experiments show 3a's marginals are
slightly miscalibrated for the dry-window task — but post-hoc warping
overfits at the rare 6h regime.

6a fixes this at training time. Same lean NWP feature set as 3a (22
features per hour: 7 per-NWP precip + 4 spread + 7 envelope means + 4
calendar), same offset_day source. Loss:

    L = alpha * mean_hourly[BCE(q_h, observed_h)]
      + sum_{N in {3,4,6}} beta * BCE(P_iid(N-block, q_daytime), label_N)

P_iid(N-block, q) is computed by a finite-state automaton over the
9-hour q-vector — see dry_window_prob(). Fully differentiable in q, so
gradients flow from the day-level dry-window task back into the per-hour
model weights. The hourly term keeps q sensible for the chart; the dry-
window term shifts q in directions that improve dry-window MC output.

3a stays untouched. 6a writes its own bundles under
data/models/precipitation/{station}/v..._phase6a/, plus TWO downstream
dry-window bundles per (station, window, lead):
    phase6a_iid    — MC over 6a's q-vectors under independence
    phase6a_copula — MC over 6a's q-vectors with the train-fitted Σ
                     correlation matrix (same shape as 3j).
The bake-off picks up both phases for head-to-head against 3g/3j.

Same chronological 70/15/15 split as 3j/3k/3m so test slices align.
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
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours, has_contiguous_dry_block  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"
PRECIP_MODELS_ROOT     = ROOT / "data" / "models" / "precipitation"
FORECASTS_ROOT         = ROOT / "data" / "forecasts"
RAINFALL_ROOT          = ROOT / "data" / "truth" / "rainfall"

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
DEFAULT_LOCATION = "bonehill_rocks"

# Match 3a's lean feature set: 7 NWPs in CanonicalModelOrder.
MODELS = [
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "meteofrance_seamless",
    "gem_seamless",
    "ecmwf_aifs025_single",
    "jma_seamless",
]
MODEL_SHORT = {
    "gfs_seamless": "gfs",
    "ecmwf_ifs025": "ecmwf",
    "icon_seamless": "icon",
    "meteofrance_seamless": "mf",
    "gem_seamless": "gem",
    "ecmwf_aifs025_single": "aifs",
    "jma_seamless": "jma",
}
WET_THRESHOLD_MM = 0.1
N_FEATURES_PER_HOUR = 22   # 7 precip + 4 spread + 7 envelope + 4 calendar

# Training hyperparameters. Small MLP for the small-N regime (~700 daytime
# days per (station, lead) cell after the chronological 70/15/15 split).
HIDDEN = 16
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 32          # batch in DAYS (each carries 9 hours)
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 25
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ALPHA_HOURLY = 1.0       # weight on hourly BCE (averaged over 9 hours)
BETA_DRY = 3.0           # weight on each window's dry-window BCE (3 windows -> total weight 9)
MC_SAMPLES_TEST = 1000   # downstream test-time MC samples


# ----------------------------------------------------------------------
# Data load — port the lean feature SQL from PrecipFeatureBuilder.cs.
# ----------------------------------------------------------------------

STATION_FRIENDLY = {
    "ea_bellever_dartmoor":     "Bellever Dartmoor",
    "ea_bovey_tracey":          "Bovey Tracey",
    "ea_dartmoor_nr_hexworthy": "Dartmoor nr Hexworthy",
}


def load_hourly_features_and_truth(location: str, friendly: str, lead: int) -> pd.DataFrame:
    """Port of PrecipFeatureBuilder.BuildForLead — long-form (valid_time,
    feature columns, observed_wet) at the requested lead, joined to EA truth."""
    fc_glob = str(FORECASTS_ROOT / "**" / "*.parquet").replace("\\", "/")
    rn_glob = str(RAINFALL_ROOT / "**" / "*.parquet").replace("\\", "/")
    models_in = "(" + ",".join(f"'{m}'" for m in MODELS) + ")"
    precip_pivot = ",\n    ".join(
        f"MAX(CASE WHEN Model = '{m}' THEN Precipitation END) AS precip_{MODEL_SHORT[m]}"
        for m in MODELS
    )
    precip_cols = ", ".join(f"p.precip_{MODEL_SHORT[m]}" for m in MODELS)
    any_not_null = "(" + " OR ".join(f"p.precip_{MODEL_SHORT[m]} IS NOT NULL" for m in MODELS) + ")"
    sql = f"""
WITH hourly_truth AS (
    SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
           SUM(Value15MinMm) AS precip_mm_hour
    FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{location}'
      AND StationName  = '{friendly}'
      AND Value15MinMm IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) = 4
),
latest AS (
    SELECT ValidTimeUtc, Model,
           Precipitation,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           Cape, WindSpeed10m,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{location}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {lead}
      AND Model IN {models_in}
),
pivoted AS (
    SELECT ValidTimeUtc,
           {precip_pivot},
           AVG(RelativeHumidity2m)         AS rh_mean,
           AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
           AVG(CloudCoverLow)              AS cloud_low_mean,
           AVG(CloudCoverMid)              AS cloud_mid_mean,
           AVG(CloudCoverHigh)             AS cloud_high_mean,
           AVG(Cape)                       AS cape_mean,
           AVG(WindSpeed10m)               AS wind_speed_mean
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
)
SELECT p.ValidTimeUtc AS valid_time,
       {precip_cols},
       p.rh_mean, p.dew_depression_mean,
       p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean,
       p.cape_mean, p.wind_speed_mean,
       t.precip_mm_hour
FROM pivoted p
JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
WHERE {any_not_null}
ORDER BY p.ValidTimeUtc
"""
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
    df["observed_wet"] = (df["precip_mm_hour"] >= WET_THRESHOLD_MM).astype(np.int32)
    return df


def compose_feature_row(row: pd.Series, valid_time: pd.Timestamp) -> np.ndarray:
    """22-feature vector matching PrecipFeatureBuilder.ComposeRow's lean layout."""
    n = len(MODELS)
    feats = np.zeros(N_FEATURES_PER_HOUR, dtype="float32")
    # 7 per-model precip
    precip = np.array([row[f"precip_{MODEL_SHORT[m]}"] for m in MODELS], dtype="float64")
    for i, m in enumerate(MODELS):
        feats[i] = float(precip[i]) if not np.isnan(precip[i]) else float("nan")
    # 4 spread stats — NaN-safe across the 7 models
    valid = ~np.isnan(precip)
    if valid.any():
        present = precip[valid]
        mean_p = float(present.mean())
        std_p  = float(present.std()) if len(present) > 1 else 0.0
        max_p  = float(present.max())
        agree  = float((present >= WET_THRESHOLD_MM).mean())
    else:
        mean_p = std_p = max_p = agree = float("nan")
    feats[n] = mean_p
    feats[n + 1] = std_p
    feats[n + 2] = max_p
    feats[n + 3] = agree
    # 7 envelope means
    feats[n + 4] = float(row["rh_mean"]) if not pd.isna(row["rh_mean"]) else float("nan")
    feats[n + 5] = float(row["dew_depression_mean"]) if not pd.isna(row["dew_depression_mean"]) else float("nan")
    feats[n + 6] = float(row["cloud_low_mean"]) if not pd.isna(row["cloud_low_mean"]) else float("nan")
    feats[n + 7] = float(row["cloud_mid_mean"]) if not pd.isna(row["cloud_mid_mean"]) else float("nan")
    feats[n + 8] = float(row["cloud_high_mean"]) if not pd.isna(row["cloud_high_mean"]) else float("nan")
    feats[n + 9] = float(row["cape_mean"]) if not pd.isna(row["cape_mean"]) else float("nan")
    feats[n + 10] = float(row["wind_speed_mean"]) if not pd.isna(row["wind_speed_mean"]) else float("nan")
    # 4 calendar
    h = valid_time.hour
    doy = valid_time.dayofyear
    feats[n + 11] = float(np.sin(2 * np.pi * h / 24.0))
    feats[n + 12] = float(np.cos(2 * np.pi * h / 24.0))
    feats[n + 13] = float(np.sin(2 * np.pi * (doy - 1) / 365.0))
    feats[n + 14] = float(np.cos(2 * np.pi * (doy - 1) / 365.0))
    return feats


def build_daytime_dataset(df: pd.DataFrame) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    """Per target_date: stack 9 daytime hours into (9, 22) features + (9,) obs.
    Drop days that are missing any daytime hour. Returns (dates, X, Y)."""
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    dates: list[pd.Timestamp] = []
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    for target_date, grp in df.groupby("target_date"):
        target_ts = pd.Timestamp(target_date)
        s, e = daytime_utc_hours(target_ts)
        n_h = e - s
        sub = grp[(grp["hour"] >= s) & (grp["hour"] < e)].sort_values("valid_time")
        if len(sub) != n_h:
            continue
        feats = np.stack([compose_feature_row(row, pd.Timestamp(row["valid_time"]))
                          for _, row in sub.iterrows()])
        obs = sub["observed_wet"].astype(np.int32).to_numpy()
        dates.append(target_ts)
        X_list.append(feats)
        Y_list.append(obs)
    if not dates:
        return [], np.zeros((0, 0, 0), dtype="float32"), np.zeros((0, 0), dtype="int32")
    return dates, np.stack(X_list).astype("float32"), np.stack(Y_list).astype("float32")


# ----------------------------------------------------------------------
# Model + losses
# ----------------------------------------------------------------------

class HourlyMlp(nn.Module):
    """Per-hour MLP: 22 features -> P(wet). Applied independently per hour
    inside a daytime window. The day-level loss is computed externally over
    the resulting 9-hour q-vector."""

    def __init__(self, n_features: int = N_FEATURES_PER_HOUR, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_features) -> (..., ) raw logit
        return self.net(x).squeeze(-1)


def dry_window_prob_iid(q: torch.Tensor, N: int) -> torch.Tensor:
    """Analytical P(>= N-hour dry block) under iid Bernoulli, given the q
    vector. Forward DP over the 9-hour sequence tracking current dry-run
    length. Differentiable. q shape: (batch, T)."""
    batch, T = q.shape
    # Cap state at N: state k means "current dry run length = k, NO N-block
    # has formed yet". k in {0..N-1}. Total mass per row = P(no N-block so far).
    f = torch.zeros(batch, N, device=q.device, dtype=q.dtype)
    # Initial step: hour 0 wet -> run 0; hour 0 dry -> run 1.
    f[:, 0] = q[:, 0]
    if N > 1:
        f[:, 1] = 1.0 - q[:, 0]
    for i in range(1, T):
        wet = q[:, i]
        dry = 1.0 - q[:, i]
        new_f = torch.zeros_like(f)
        # New wet: all states collapse to run = 0.
        new_f[:, 0] = f.sum(dim=-1) * wet
        # New dry: shift run length up by 1, dropping the k=N-1 mass (which
        # would become run >= N — absorbed into the failure state we don't
        # track explicitly; missing mass is precisely P(N-block exists)).
        if N > 1:
            new_f[:, 1:] = f[:, :-1] * dry.unsqueeze(-1)
        f = new_f
    p_no_block = f.sum(dim=-1)
    return 1.0 - p_no_block


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def has_dry_window(seq: np.ndarray, N: int) -> bool:
    return has_contiguous_dry_block(seq, N)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score on the train slice (NaN-aware). Drops NaN to 0
    post-standardisation. Returns (X_std, mean, scale)."""
    n_feat = X.shape[-1]
    flat = X.reshape(-1, n_feat)
    mean = np.zeros(n_feat, dtype="float32")
    scale = np.ones(n_feat, dtype="float32")
    for k in range(n_feat):
        vals = flat[:, k]
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        mean[k] = valid.mean()
        s = valid.std()
        scale[k] = max(s, 1e-6)
    Xs = (X - mean) / scale
    Xs = np.where(np.isnan(Xs), 0.0, Xs).astype("float32")
    return Xs, mean, scale


def fit_correlation(obs_seqs: np.ndarray) -> np.ndarray:
    if len(obs_seqs) < 10:
        return np.eye(obs_seqs.shape[1])
    corr = np.corrcoef(obs_seqs.T)
    return corr + 1e-6 * np.eye(corr.shape[0])


def train_6a(X_train, Y_train, X_val, Y_val, windows: list[int]) -> tuple[HourlyMlp, dict]:
    """Joint-loss training over a stack of daytime windows."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(Y_train)
    x_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(Y_val)

    # Day-level dry-window labels.
    def day_labels(Y: np.ndarray, N: int) -> torch.Tensor:
        return torch.tensor([1.0 if has_dry_window(Y[i], N) else 0.0 for i in range(len(Y))])
    val_dw_labels = {N: day_labels(Y_val, N) for N in windows}

    model = HourlyMlp()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
    bce_prob   = nn.BCELoss(reduction="mean")

    n_train = len(X_train)
    indices = np.arange(n_train)
    best_val = float("inf")
    best_state = None
    epochs_since_best = 0
    epochs_run = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        np.random.shuffle(indices)
        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            xb = x_train_t[batch_idx]    # (B, 9, F)
            yb = y_train_t[batch_idx]    # (B, 9)

            optim.zero_grad()
            # Hourly logits/probs
            logits = model(xb)            # (B, 9)
            q = torch.sigmoid(logits)
            # Hourly BCE (per hour, averaged)
            hourly_loss = bce_logits(logits, yb)
            # Day-level dry-window losses
            dry_loss = torch.tensor(0.0)
            for N in windows:
                dw_pred = dry_window_prob_iid(q, N)
                dw_label = torch.tensor([1.0 if has_dry_window(yb[i].numpy().astype(np.int32), N) else 0.0
                                         for i in range(len(yb))], dtype=q.dtype)
                # Clip to avoid log(0)
                dw_pred = torch.clamp(dw_pred, 1e-6, 1 - 1e-6)
                dry_loss = dry_loss + bce_prob(dw_pred, dw_label)
            loss = ALPHA_HOURLY * hourly_loss + BETA_DRY * dry_loss
            loss.backward()
            optim.step()

        # Val
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_q = torch.sigmoid(val_logits)
            val_hourly = bce_logits(val_logits, y_val_t).item()
            val_dry = 0.0
            for N in windows:
                p = dry_window_prob_iid(val_q, N)
                p = torch.clamp(p, 1e-6, 1 - 1e-6)
                val_dry += bce_prob(p, val_dw_labels[N]).item()
            val_total = ALPHA_HOURLY * val_hourly + BETA_DRY * val_dry

        if val_total < best_val - 1e-5:
            best_val = val_total
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_run": epochs_run, "best_val_total": float(best_val)}


def mc_dry_window_iid(q: np.ndarray, N: int, n_samples: int, rng: np.random.Generator) -> float:
    samples = (rng.random((n_samples, len(q))) < q).astype(np.int32)
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
        if longest >= N:
            hits += 1
    return hits / n_samples


def mc_dry_window_copula(q: np.ndarray, L: np.ndarray, N: int, n_samples: int, rng: np.random.Generator) -> float:
    z_iid = rng.standard_normal((n_samples, len(q)))
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
        if longest >= N:
            hits += 1
    return hits / n_samples


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    global ALPHA_HOURLY, BETA_DRY
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--location", default=DEFAULT_LOCATION)
    ap.add_argument("--alpha-hourly", type=float, default=ALPHA_HOURLY)
    ap.add_argument("--beta-dry", type=float, default=BETA_DRY)
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    location = args.location
    ALPHA_HOURLY = args.alpha_hourly
    BETA_DRY = args.beta_dry

    print(f"Phase 6a (joint hourly+dry-window loss MLP) — stations={stations} leads={leads} windows={windows}")
    print(f"  loss weights: alpha_hourly={ALPHA_HOURLY}, beta_dry={BETA_DRY} (per window, sum across {len(windows)} windows)")
    print(f"  arch: MLP {N_FEATURES_PER_HOUR}->{HIDDEN}->{HIDDEN}->1; dropout={DROPOUT}; lr={LR}; bs={BATCH_SIZE} days; max_epochs={MAX_EPOCHS}")

    overall_start = datetime.now(timezone.utc)

    for station in stations:
        friendly = STATION_FRIENDLY[station]
        print(f"\n=== {station} (friendly: '{friendly}') ===")

        # Accumulators are per-station, per-window. Bundles are
        # (station, window) — every lead's test rows go into the SAME bundle's
        # test_predictions.parquet (3a/3b/etc. all do this; the bake-off
        # inner-joins on (target_date, lead) so it needs the full lead set
        # present in one file). The earlier nested-write version overwrote
        # the parquet per lead and only kept the last lead — that broke the
        # 2026-05-13 bake-off and caused a fake "MC-3e wins" headline.
        rows_iid_by_window: dict[int, list[dict]] = {w: [] for w in windows}
        rows_cop_by_window: dict[int, list[dict]] = {w: [] for w in windows}
        per_lead_info: dict[int, dict] = {}
        corr_by_lead: dict[int, np.ndarray] = {}
        clim_by_window: dict[int, float] = {}

        for lead in leads:
            print(f"\n--- lead {lead}h ---")
            df = load_hourly_features_and_truth(location, friendly, lead)
            print(f"  hourly feature rows: {len(df):,}")
            dates, X, Y = build_daytime_dataset(df)
            print(f"  daytime-complete days: {len(dates):,}")
            if len(dates) < 100:
                print("  (skip) not enough days")
                continue

            n = len(dates)
            tr_end = int(np.floor(n * TRAIN_FRAC))
            val_end = tr_end + int(np.floor(n * VAL_FRAC))

            # Standardise on TRAIN only.
            X_train_raw = X[:tr_end]
            X_val_raw   = X[tr_end:val_end]
            X_test_raw  = X[val_end:]
            X_train, mean, scale = standardise(X_train_raw)
            X_val_arr   = np.where(np.isnan((X_val_raw - mean) / scale), 0.0,
                                   (X_val_raw - mean) / scale).astype("float32")
            X_test_arr  = np.where(np.isnan((X_test_raw - mean) / scale), 0.0,
                                   (X_test_raw - mean) / scale).astype("float32")

            Y_train = Y[:tr_end]; Y_val = Y[tr_end:val_end]; Y_test = Y[val_end:]
            test_dates = dates[val_end:]

            # Fit Σ for copula MC from TRAIN observed sequences only.
            corr = fit_correlation(Y_train.astype(np.int32))
            try:
                L_chol = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                L_chol = np.linalg.cholesky(corr + 1e-5 * np.eye(corr.shape[0]))
            corr_by_lead[lead] = corr

            # Train.
            model, train_info = train_6a(X_train, Y_train, X_val_arr, Y_val, windows)
            print(f"  trained: epochs={train_info['epochs_run']}, val_total_loss={train_info['best_val_total']:.4f}")
            per_lead_info[lead] = {
                "epochs_run": int(train_info["epochs_run"]),
                "best_val_total": float(train_info["best_val_total"]),
            }

            # Score on test slice.
            model.eval()
            with torch.no_grad():
                test_q = torch.sigmoid(model(torch.tensor(X_test_arr))).numpy()
            # test_q: (n_test_days, 9)

            # For each window: append this lead's test rows to the per-window
            # accumulator. Per-(lead, window) Brier still printed as a sanity
            # check; the parquet write is deferred until after the lead loop.
            for window in windows:
                rng_iid = np.random.default_rng(SEED + 1)
                rng_cop = np.random.default_rng(SEED + 2)
                lead_rows_iid: list[dict] = []
                lead_rows_cop: list[dict] = []
                for i, td in enumerate(test_dates):
                    q = np.clip(test_q[i], 1e-6, 1 - 1e-6)
                    p_iid = mc_dry_window_iid(q, window, MC_SAMPLES_TEST, rng_iid)
                    p_cop = mc_dry_window_copula(q, L_chol, window, MC_SAMPLES_TEST, rng_cop)
                    label = 1 if has_dry_window(Y_test[i].astype(np.int32), window) else 0
                    lead_rows_iid.append({
                        "target_date": td, "station": station,
                        "window": window, "lead": lead,
                        "p_dry_window": float(p_iid),
                        "observed_dry_window": np.uint8(label),
                    })
                    lead_rows_cop.append({
                        "target_date": td, "station": station,
                        "window": window, "lead": lead,
                        "p_dry_window": float(p_cop),
                        "observed_dry_window": np.uint8(label),
                    })
                rows_iid_by_window[window].extend(lead_rows_iid)
                rows_cop_by_window[window].extend(lead_rows_cop)

                b_iid = brier(np.array([r["p_dry_window"] for r in lead_rows_iid]),
                              np.array([r["observed_dry_window"] for r in lead_rows_iid], dtype=float))
                b_cop = brier(np.array([r["p_dry_window"] for r in lead_rows_cop]),
                              np.array([r["observed_dry_window"] for r in lead_rows_cop], dtype=float))
                clim = float(np.mean([1 if has_dry_window(Y_train[k].astype(np.int32), window) else 0
                                      for k in range(len(Y_train))]))
                clim_b = float(np.mean([(clim - r["observed_dry_window"]) ** 2 for r in lead_rows_iid]))
                clim_by_window[window] = clim  # last lead wins, only used for metadata
                print(f"  window {window}h: 6a-iid test Brier={b_iid:.4f}  6a-copula={b_cop:.4f}  "
                      f"clim={clim_b:.4f}  n_test={len(lead_rows_iid)}")

        # All leads done for this station. Write one bundle per window with
        # ALL leads' test rows present.
        ts = overall_start.strftime("%Y-%m-%d_%H%M%S")
        for window in windows:
            rows_iid = rows_iid_by_window[window]
            rows_cop = rows_cop_by_window[window]
            if not rows_iid:
                continue
            base = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h"
            bundle_iid    = base / f"v{ts}_phase6a_iid"
            bundle_copula = base / f"v{ts}_phase6a_copula"
            bundle_iid.mkdir(parents=True, exist_ok=True)
            bundle_copula.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows_iid).to_parquet(bundle_iid / "test_predictions.parquet", index=False)
            pd.DataFrame(rows_cop).to_parquet(bundle_copula / "test_predictions.parquet", index=False)
            meta_common = {
                "Target": "dry_window", "WindowHours": window,
                "Leads": sorted(per_lead_info.keys()),
                "Station": station,
                "Hyperparameters": {
                    "alpha_hourly": ALPHA_HOURLY, "beta_dry": BETA_DRY,
                    "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                    "batch_size_days": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                    "early_stop_patience": EARLY_STOP_PATIENCE, "seed": SEED,
                    "n_features_per_hour": N_FEATURES_PER_HOUR,
                },
                "PerLeadTrainInfo": per_lead_info,
                "Climatology": clim_by_window.get(window),
                "MCSamplesTest": MC_SAMPLES_TEST,
                "TrainFrac": TRAIN_FRAC, "ValFrac": VAL_FRAC,
                "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
            }
            (bundle_iid / "training_metadata.json").write_text(json.dumps(
                {**meta_common, "Phase": "6a_iid", "Architecture": "joint_loss_mlp_iid_mc",
                 "Version": f"v{ts}_phase6a_iid"}, indent=2))
            (bundle_copula / "training_metadata.json").write_text(json.dumps(
                {**meta_common, "Phase": "6a_copula", "Architecture": "joint_loss_mlp_copula_mc",
                 "Version": f"v{ts}_phase6a_copula",
                 "CorrelationMatrixByLead": {str(L): corr_by_lead[L].round(4).tolist()
                                              for L in sorted(corr_by_lead.keys())}}, indent=2))

    print(f"\nDone in {(datetime.now(timezone.utc) - overall_start).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
