"""Dry-window 4-way bake-off — 3b vs 3g vs MC-over-calibrated-3a vs mean(3b, 3g).

All four predictors evaluated on the SAME (station, window, lead, target_date)
test cells via inner-join across:
  - 3b's test_predictions.parquet  (LightGBM dry-window blender, day-level)
  - 3a's test_predictions.parquet  (hourly P(wet) ->MC-sampled to dry-window)
  - 3a_isotonic's calibration.json (per-lead PAV maps applied to 3a hourly P(wet))

The four predictions:
  3b                : direct LightGBM dry-window output (p_dry_window from 3b)
  3g                : MC over 3a's raw hourly q vector (= current production 3g)
  MC-cal-3a         : MC over PAV-calibrated hourly q (3a_isotonic's per-lead
                      knots applied to each hour's p_wet before sampling).
                      Only available for stations that have a 3a_isotonic
                      bundle on disk; Bovey skipped (no bundle).
  mean(3b, 3g)      : arithmetic mean of 3b and 3g predictions

Same MC algorithm as the existing 3g source bake-off: independent-hour Bernoulli
sampling against the 24-hour q vector, n_samples=1000, seed=42.

Output: per-(station, window, lead) Brier table + aggregate means + per-window
summary; CSV to reports/4way_bakeoff_{ts}.csv.

Usage::

    python scripts/DryWindowStartHour/dry_window_4way_bakeoff.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Production dry-window question is "is there a contiguous N-hour dry block
# in the DAYTIME wall-clock window?", not "anywhere in the UTC day".
# Defaults match config.yaml's DryWindow.AllowedWindow defaults
# (StartLocalHour=9, EndLocalHour=18, Tz=Europe/London) — see
# DaytimeWindow.cs. Tests that mix sources MUST agree on this or the
# Brier numbers are evaluating different questions and don't reconcile
# with the deployed 3g panel on the site.
DAYTIME_LOCAL_START = 9
DAYTIME_LOCAL_END = 18
DAYTIME_TZ = ZoneInfo("Europe/London")


def daytime_utc_hours(target_date: pd.Timestamp) -> tuple[int, int]:
    """Return (start_utc_hour, end_utc_hour_exclusive) for the daytime
    wall-clock window on this target UTC date. Mirrors
    DaytimeWindow.UtcHourRangeFor — handles BST/GMT cleanly via
    zoneinfo. Europe/London 9-18 ends up as UTC 9-18 in GMT and UTC
    8-17 in BST."""
    start_local = pd.Timestamp(
        year=target_date.year, month=target_date.month, day=target_date.day,
        hour=DAYTIME_LOCAL_START, tz=DAYTIME_TZ,
    )
    end_local = pd.Timestamp(
        year=target_date.year, month=target_date.month, day=target_date.day,
        hour=DAYTIME_LOCAL_END, tz=DAYTIME_TZ,
    )
    return int(start_local.tz_convert("UTC").hour), int(end_local.tz_convert("UTC").hour)

ROOT = Path(__file__).resolve().parent.parent.parent
PRECIP_MODELS_ROOT = ROOT / "data" / "models" / "precipitation"
DRY_WINDOW_MODELS_ROOT = ROOT / "data" / "models" / "dry_window"
DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
MC_SAMPLES = 1000
SEED = 42


@dataclass
class CellResult:
    station: str
    window: int
    lead: int
    n_days: int
    obs_rate: float                    # fraction of days with observed dry-window = True
    brier_3b: float
    brier_3g: float
    brier_mc_cal_3a: float | None      # None when station has no 3a_isotonic bundle
    brier_mean_3b_3g: float
    brier_3f: float | None             # None when station has no 3f bundle
    brier_mean_3b_3f_3g: float | None  # 3-way ensemble, None when 3f absent
    brier_mc_3c: float | None          # MC over 3c's hourly P(wet) — None when 3c absent
    brier_mc_3e: float | None          # MC over 3e's hourly P(wet) — daytime-sliced (the previous
                                       # bake-off used 24h UTC + 24h labels, which evaluated the
                                       # wrong question; re-test with daytime is what this column is).
    brier_3h: float | None             # Bidirectional GRU on the same 9-hour 3a-replay P(wet) sequence
                                       # 3g/MC variants iid-sample. None when 3h bundle absent.
    brier_3h_3e: float | None          # Same GRU architecture but trained on the 3e-replay P(wet)
                                       # sequence. Tests whether the MLP's hourly P(wet) carries more
                                       # sequence-modellable signal than 3a's LightGBM hourly P(wet).
    brier_3i: float | None             # GRU on raw per-NWP daytime hourly precip (9 hours x N models).
                                       # Tests whether the GRU can learn its own blend + temporal model
                                       # end-to-end, vs feeding it a pre-blended single-channel P(wet).
    brier_3j: float | None             # Gaussian-copula MC over 3a's daytime q-vector with
                                       # train-fitted 9x9 wet/dry correlation. Tests whether relaxing
                                       # 3g's independence assumption recovers the MC ceiling.
    brier_3k: float | None             # Temperature-scaled iid MC — single t per cell fit by
                                       # grid search on val dry-window Brier. Tests whether 3a's
                                       # marginals are mis-calibrated for the dry-window objective.
    brier_3l: float | None             # Hand-crafted q-vector features (mean/max/run-stats) +
                                       # per-cell logistic regression. Tests whether sequence
                                       # learning is necessary or simple feature engineering suffices.
    brier_3m: float | None             # Alpha-shrinkage interpolation between 3j (copula) and 3g
                                       # (iid). Per-cell grid-search picks the dependence strength.
    brier_6a_iid: float | None         # 6a — hourly MLP trained with joint hourly+dry-window loss,
                                       # then iid MC downstream (same sampler as 3g).
    brier_6a_copula: float | None      # Same 6a model, copula MC downstream (3j-style dependence).
    brier_3n: float | None             # Regime-conditioned copula MC — 3j with TWO Σs (settled vs
                                       # unsettled) split by NWP-ensemble agreement. Tests whether
                                       # 3j's pooled Σ underfits both regimes.


# ----------------------------------------------------------------------
# Bundle discovery
# ----------------------------------------------------------------------

def find_latest_with_test_predictions(root: Path, station: str, phase_suffix: str | None) -> Path | None:
    """Find newest (station, phase) bundle that has a test_predictions.parquet.
    ``phase_suffix=None`` selects the unsuffixed 3a champion convention."""
    station_dir = root / station
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
        if (d / "test_predictions.parquet").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name) / "test_predictions.parquet"


def find_dry_window_test_predictions(station: str, window: int, phase_suffix: str | None) -> Path | None:
    """Bundle layout: data/models/dry_window/{station}/window_{N}h/{version}/.
    ``phase_suffix=None`` selects the unsuffixed 3b champion convention;
    pass ``"phase3f"`` (etc.) to filter for that phase only. Newest matching
    bundle with a test_predictions.parquet wins."""
    composite_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h"
    if not composite_dir.is_dir():
        return None
    candidates: list[Path] = []
    for d in composite_dir.iterdir():
        if not d.is_dir():
            continue
        if phase_suffix:
            if phase_suffix not in d.name:
                continue
        else:
            # Unsuffixed 3b — reject any *_phase* directory.
            if "phase" in d.name:
                continue
        if (d / "test_predictions.parquet").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name) / "test_predictions.parquet"


def find_3b_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix=None)


def find_3f_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3f")


def find_3h_test_predictions(station: str, window: int) -> Path | None:
    # 3h = GRU on 3a hourly P(wet) replay. Bundle suffix "_phase3h" only —
    # NOT "_phase3h_3e" (that's the 3e-sourced sibling, see find_3h_3e).
    composite_dir = DRY_WINDOW_MODELS_ROOT / station / f"window_{window}h"
    if not composite_dir.is_dir():
        return None
    candidates = [
        d for d in composite_dir.iterdir()
        if d.is_dir() and d.name.endswith("_phase3h")
           and (d / "test_predictions.parquet").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name) / "test_predictions.parquet"


def find_3h_3e_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3h_3e")


def find_3i_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3i")


def find_3j_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3j")


def find_3n_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3n")


def find_3k_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3k")


def find_3l_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3l")


def find_3m_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase3m")


def find_6a_iid_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase6a_iid")


def find_6a_copula_test_predictions(station: str, window: int) -> Path | None:
    return find_dry_window_test_predictions(station, window, phase_suffix="phase6a_copula")


def find_3c_test_predictions(station: str) -> Path | None:
    return find_latest_with_test_predictions(PRECIP_MODELS_ROOT, station, phase_suffix="phase3c")


def find_3e_test_predictions(station: str) -> Path | None:
    return find_latest_with_test_predictions(PRECIP_MODELS_ROOT, station, phase_suffix="phase3e")


def find_3a_isotonic_calibration(station: str) -> Path | None:
    """3a_isotonic bundle: data/models/precipitation/{station}/v*_phase3a_isotonic/.
    File of interest is calibration.json (per-lead PAV knots).
    Returns None for stations without a 3a_isotonic bundle (e.g. Bovey)."""
    station_dir = PRECIP_MODELS_ROOT / station
    if not station_dir.is_dir():
        return None
    candidates = [
        d for d in station_dir.iterdir()
        if d.is_dir() and "phase3a_isotonic" in d.name and (d / "calibration.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name) / "calibration.json"


# ----------------------------------------------------------------------
# PAV calibrator — load + apply (mirrors IsotonicCalibrator semantics)
# ----------------------------------------------------------------------

@dataclass
class IsotonicMap:
    """Per-lead PAV map. xs is the sorted raw-prob breakpoints; ys is the
    matching calibrated values. Apply via np.interp with end-clamping.
    Equivalent to IsotonicCalibrator.Predict in MlpTrainer / production C#."""
    xs: np.ndarray   # ascending
    ys: np.ndarray   # parallel array of calibrated values

    def apply(self, raw_p: np.ndarray) -> np.ndarray:
        # np.interp clamps to ys[0] / ys[-1] outside the fitted range — same as
        # IsotonicCalibrator.Predict's "extrapolation would invent signal we
        # don't have" behaviour.
        return np.interp(raw_p, self.xs, self.ys)


def load_isotonic_by_lead(calibration_json_path: Path) -> dict[int, IsotonicMap]:
    """Read the calibration.json written by Phase3aIsotonicCommand. Schema:
        { "ByLead": { "24": { "Knots": [{X, XMax, Y}, ...] }, "48": ..., ... } }
    X and XMax are usually equal (single-point knots from PAV merge); when
    they differ we use the block midpoint as the breakpoint (mirrors how
    IsotonicCalibrator.Fit collapses merged blocks). Knots arrive sorted
    by X so we trust that ordering."""
    with calibration_json_path.open("r") as f:
        data = json.load(f)
    out: dict[int, IsotonicMap] = {}
    for lead_str, block in data["ByLead"].items():
        knots = block["Knots"]
        xs = np.array([(k["X"] + k["XMax"]) / 2.0 for k in knots], dtype="float64")
        ys = np.array([k["Y"] for k in knots], dtype="float64")
        # Defensive sort — production writes sorted but pinning here costs
        # nothing and protects against any drift in the JSON format.
        order = np.argsort(xs, kind="stable")
        out[int(lead_str)] = IsotonicMap(xs=xs[order], ys=ys[order])
    return out


# ----------------------------------------------------------------------
# Dry-window MC primitives — mirror DryWindow3gPredictor.ProbDryWindow
# ----------------------------------------------------------------------

def has_contiguous_dry_block(binary: np.ndarray, window: int) -> bool:
    """True if the binary sequence (1 = wet, 0 = dry) contains a contiguous
    run of >= ``window`` zeros."""
    run = 0
    for v in binary:
        if v == 0:
            run += 1
            if run >= window:
                return True
        else:
            run = 0
    return False


def prob_dry_window_mc(q: np.ndarray, window: int, n_samples: int, rng: np.random.Generator) -> float:
    """Monte Carlo P(∃ contiguous run of >= window dry hours) given the hourly
    P(wet) marginals q. Independent-hour sampling — same assumption
    DryWindow3gPredictor.ProbDryWindow makes."""
    n = len(q)
    if window > n:
        return 0.0
    samples = (rng.random((n_samples, n)) < q).astype(np.int32)
    hits = 0
    for s in samples:
        if has_contiguous_dry_block(s, window):
            hits += 1
    return hits / n_samples


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# ----------------------------------------------------------------------
# Per-station evaluation
# ----------------------------------------------------------------------

def _build_daytime_q_index(df: pd.DataFrame) -> tuple[dict[tuple[int, pd.Timestamp], np.ndarray], set[tuple[int, int]]]:
    """Slice an hourly P(wet) test_predictions parquet to daytime UTC hours
    per target_date, return a (lead, target_date) -> q-vector dict + the
    set of (start, end) UTC ranges that fired. Same daytime-only rule
    DryWindow3gPredictor.ExtractDaytimeQ enforces in production."""
    df = df[["valid_time", "lead", "p_wet"]].copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    out: dict[tuple[int, pd.Timestamp], np.ndarray] = {}
    ranges: set[tuple[int, int]] = set()
    for (lead, target_date), grp in df.groupby(["lead", "target_date"]):
        target_ts = pd.Timestamp(target_date)
        start_utc, end_utc = daytime_utc_hours(target_ts)
        ranges.add((start_utc, end_utc))
        n_expected = end_utc - start_utc
        daytime_rows = grp[(grp["hour"] >= start_utc) & (grp["hour"] < end_utc)]
        if len(daytime_rows) != n_expected:
            continue
        out[(int(lead), target_ts)] = daytime_rows.sort_values("valid_time")["p_wet"].to_numpy(dtype="float64")
    return out, ranges


def evaluate_station(
    station: str,
    df_3a: pd.DataFrame,
    df_3b_by_window: dict[int, pd.DataFrame],
    df_3f_by_window: dict[int, pd.DataFrame],
    df_3h_by_window: dict[int, pd.DataFrame],
    df_3h_3e_by_window: dict[int, pd.DataFrame],
    df_3i_by_window: dict[int, pd.DataFrame],
    df_3j_by_window: dict[int, pd.DataFrame],
    df_3n_by_window: dict[int, pd.DataFrame],
    df_3k_by_window: dict[int, pd.DataFrame],
    df_3l_by_window: dict[int, pd.DataFrame],
    df_3m_by_window: dict[int, pd.DataFrame],
    df_6a_iid_by_window: dict[int, pd.DataFrame],
    df_6a_copula_by_window: dict[int, pd.DataFrame],
    df_3c: pd.DataFrame | None,
    df_3e: pd.DataFrame | None,
    isotonic_by_lead: dict[int, IsotonicMap] | None,
    leads: list[int],
    windows: list[int],
    rng: np.random.Generator,
) -> list[CellResult]:
    """For one station: group 3a's 24-hourly P(wet) per (lead, target_date),
    inner-join to 3b's per-(window, lead, target_date) prediction, MC-sample
    3g (raw) + MC-cal (calibrated) + MC-3c variants, compute Brier per cell."""
    # 3a -> per-day DAYTIME-ONLY q vector. Production 3g
    # (DryWindow3gPredictor.ExtractDaytimeQ) slices the hourly q vector to
    # the daytime UTC hour range BEFORE MC-sampling — the dry-window label
    # is "is there a contiguous N-hour dry block in the daytime?", not
    # "anywhere in the 24-hour UTC day". Without this slice the MC computes
    # a different question than the labels score, blowing up the Brier vs
    # the deployed 3g panel. Discovered 2026-05-13 when bake-off Brier was
    # ~2x the site's quoted figures.
    by_lead_date, daytime_range_seen = _build_daytime_q_index(df_3a)
    # Same slicing for 3c when present — different model, same hourly grid,
    # so MC over 3c is a head-to-head with 3g on the dry-window question.
    by_lead_date_3c: dict[tuple[int, pd.Timestamp], np.ndarray] = {}
    if df_3c is not None:
        by_lead_date_3c, _ = _build_daytime_q_index(df_3c)
    # Same for 3e — re-test the 2026-05-12 negative result on the CORRECT
    # daytime question. The prior bake-off used 24h UTC + 24h labels, which
    # measured a different question than the deployed dry-window panel.
    by_lead_date_3e: dict[tuple[int, pd.Timestamp], np.ndarray] = {}
    if df_3e is not None:
        by_lead_date_3e, _ = _build_daytime_q_index(df_3e)
    ranges_str = ", ".join(f"{s:02d}-{e:02d} UTC (n={e-s}h)"
                            for s, e in sorted(daytime_range_seen))
    print(f"  daytime ranges used for MC sampling (3g + MC-cal + MC-3c + MC-3e): {ranges_str}")

    results: list[CellResult] = []
    for window in windows:
        df_3b_w = df_3b_by_window.get(window)
        if df_3b_w is None:
            print(f"  {station} window {window}h: no 3b test_predictions parquet")
            continue
        df_3b_w = df_3b_w[["target_date", "lead", "p_dry_window", "observed_dry_window"]].copy()
        df_3b_w["target_date"] = pd.to_datetime(df_3b_w["target_date"], utc=True).dt.tz_localize(None)

        # 3f's parquet for this station/window (None if 3f wasn't trained yet).
        # Index by (lead, target_date) for fast lookup inside the per-cell loop.
        df_3f_w = df_3f_by_window.get(window)
        p_3f_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3f_w is not None:
            df_3f_w = df_3f_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3f_w["target_date"] = pd.to_datetime(df_3f_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3f_w.iterrows():
                p_3f_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])

        # Same shape for 3h (the GRU). Built via train_3h_rnn.py, writes the
        # same DryWindowTestPredictionRow schema as 3b/3f. None when the GRU
        # hasn't been trained for this (station, window).
        df_3h_w = df_3h_by_window.get(window)
        p_3h_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3h_w is not None:
            df_3h_w = df_3h_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3h_w["target_date"] = pd.to_datetime(df_3h_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3h_w.iterrows():
                p_3h_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 3h_3e — same GRU trained on 3e's replay hourly P(wet) instead of 3a's.
        df_3h_3e_w = df_3h_3e_by_window.get(window)
        p_3h_3e_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3h_3e_w is not None:
            df_3h_3e_w = df_3h_3e_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3h_3e_w["target_date"] = pd.to_datetime(df_3h_3e_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3h_3e_w.iterrows():
                p_3h_3e_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 3i — GRU on raw per-NWP daytime precip sequence.
        df_3i_w = df_3i_by_window.get(window)
        p_3i_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3i_w is not None:
            df_3i_w = df_3i_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3i_w["target_date"] = pd.to_datetime(df_3i_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3i_w.iterrows():
                p_3i_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 3j — Gaussian-copula MC over 3a daytime q (== 3g with dependence).
        df_3j_w = df_3j_by_window.get(window)
        p_3j_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3j_w is not None:
            df_3j_w = df_3j_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3j_w["target_date"] = pd.to_datetime(df_3j_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3j_w.iterrows():
                p_3j_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 3k — temperature-scaled iid MC.
        df_3k_w = df_3k_by_window.get(window)
        p_3k_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3k_w is not None:
            df_3k_w = df_3k_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3k_w["target_date"] = pd.to_datetime(df_3k_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3k_w.iterrows():
                p_3k_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 3l — engineered q-vector features + LR.
        df_3l_w = df_3l_by_window.get(window)
        p_3l_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3l_w is not None:
            df_3l_w = df_3l_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3l_w["target_date"] = pd.to_datetime(df_3l_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3l_w.iterrows():
                p_3l_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 3m — alpha-shrunk copula MC.
        df_3m_w = df_3m_by_window.get(window)
        p_3m_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3m_w is not None:
            df_3m_w = df_3m_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3m_w["target_date"] = pd.to_datetime(df_3m_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3m_w.iterrows():
                p_3m_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        # 6a-iid and 6a-copula — joint-loss MLP downstream MC.
        df_6a_iid_w = df_6a_iid_by_window.get(window)
        p_6a_iid_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_6a_iid_w is not None:
            df_6a_iid_w = df_6a_iid_w[["target_date", "lead", "p_dry_window"]].copy()
            df_6a_iid_w["target_date"] = pd.to_datetime(df_6a_iid_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_6a_iid_w.iterrows():
                p_6a_iid_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        df_6a_cop_w = df_6a_copula_by_window.get(window)
        p_6a_cop_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_6a_cop_w is not None:
            df_6a_cop_w = df_6a_cop_w[["target_date", "lead", "p_dry_window"]].copy()
            df_6a_cop_w["target_date"] = pd.to_datetime(df_6a_cop_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_6a_cop_w.iterrows():
                p_6a_cop_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])
        df_3n_w = df_3n_by_window.get(window)
        p_3n_by_key: dict[tuple[int, pd.Timestamp], float] = {}
        if df_3n_w is not None:
            df_3n_w = df_3n_w[["target_date", "lead", "p_dry_window"]].copy()
            df_3n_w["target_date"] = pd.to_datetime(df_3n_w["target_date"], utc=True).dt.tz_localize(None)
            for _, row in df_3n_w.iterrows():
                p_3n_by_key[(int(row["lead"]), pd.Timestamp(row["target_date"]))] = float(row["p_dry_window"])

        for lead in leads:
            sub_3b = df_3b_w[df_3b_w["lead"] == lead]
            if sub_3b.empty:
                continue

            # Inner-join 3b's day cells to 3a's daytime q vectors + 3f's
            # per-cell P. Drop days where 3a doesn't have a matching slice.
            # 3f matches by (lead, target_date); when 3f bundle is absent
            # (e.g. first run before retrain), 3f columns stay None per
            # cell and the aggregate skips them on the matched-subset path.
            preds_3b: list[float] = []
            preds_3g: list[float] = []
            preds_mc_cal: list[float] | None = [] if isotonic_by_lead is not None else None
            preds_3f: list[float] | None = [] if df_3f_w is not None else None
            preds_mc_3c: list[float] | None = [] if df_3c is not None else None
            preds_mc_3e: list[float] | None = [] if df_3e is not None else None
            preds_3h: list[float] | None = [] if df_3h_w is not None else None
            preds_3h_3e: list[float] | None = [] if df_3h_3e_w is not None else None
            preds_3i: list[float] | None = [] if df_3i_w is not None else None
            preds_3j: list[float] | None = [] if df_3j_w is not None else None
            preds_3k: list[float] | None = [] if df_3k_w is not None else None
            preds_3l: list[float] | None = [] if df_3l_w is not None else None
            preds_3m: list[float] | None = [] if df_3m_w is not None else None
            preds_6a_iid: list[float] | None = [] if df_6a_iid_w is not None else None
            preds_6a_cop: list[float] | None = [] if df_6a_cop_w is not None else None
            preds_3n: list[float] | None = [] if df_3n_w is not None else None
            labels: list[int] = []
            for _, row in sub_3b.iterrows():
                td = pd.Timestamp(row["target_date"])
                key = (lead, td)
                if key not in by_lead_date:
                    continue
                # When 3f is in play, also require a 3f prediction for this
                # cell — keeps the same-cells contract honest. Without this,
                # 3f's reported Brier could be on a different cell set than
                # 3b/3g and the numbers wouldn't reconcile.
                if preds_3f is not None and key not in p_3f_by_key:
                    continue
                # Same contract for MC-3c / MC-3e: require a daytime q-vector
                # at this (lead, target_date). All three precip phases share
                # the test slice with 3a, so misses are rare — but be strict.
                if preds_mc_3c is not None and key not in by_lead_date_3c:
                    continue
                if preds_mc_3e is not None and key not in by_lead_date_3e:
                    continue
                # 3h trained on replay parquet with its own chronological split,
                # so its test slice may be SHORTER than 3b's (replay ends earlier
                # than 3a's test_predictions). Skip the cell if 3h doesn't have
                # a prediction there — keeps the comparison subset honest.
                if preds_3h is not None and key not in p_3h_by_key:
                    continue
                if preds_3h_3e is not None and key not in p_3h_3e_by_key:
                    continue
                if preds_3i is not None and key not in p_3i_by_key:
                    continue
                if preds_3j is not None and key not in p_3j_by_key:
                    continue
                if preds_3k is not None and key not in p_3k_by_key:
                    continue
                if preds_3l is not None and key not in p_3l_by_key:
                    continue
                if preds_3m is not None and key not in p_3m_by_key:
                    continue
                if preds_6a_iid is not None and key not in p_6a_iid_by_key:
                    continue
                if preds_6a_cop is not None and key not in p_6a_cop_by_key:
                    continue
                if preds_3n is not None and key not in p_3n_by_key:
                    continue
                q_raw = by_lead_date[key]
                preds_3b.append(float(row["p_dry_window"]))
                preds_3g.append(prob_dry_window_mc(q_raw, window, MC_SAMPLES, rng))
                if preds_3f is not None:
                    preds_3f.append(p_3f_by_key[key])
                if preds_3h is not None:
                    preds_3h.append(p_3h_by_key[key])
                if preds_3h_3e is not None:
                    preds_3h_3e.append(p_3h_3e_by_key[key])
                if preds_3i is not None:
                    preds_3i.append(p_3i_by_key[key])
                if preds_3j is not None:
                    preds_3j.append(p_3j_by_key[key])
                if preds_3k is not None:
                    preds_3k.append(p_3k_by_key[key])
                if preds_3l is not None:
                    preds_3l.append(p_3l_by_key[key])
                if preds_3m is not None:
                    preds_3m.append(p_3m_by_key[key])
                if preds_6a_iid is not None:
                    preds_6a_iid.append(p_6a_iid_by_key[key])
                if preds_6a_cop is not None:
                    preds_6a_cop.append(p_6a_cop_by_key[key])
                if preds_3n is not None:
                    preds_3n.append(p_3n_by_key[key])
                if preds_mc_3c is not None:
                    q_3c = by_lead_date_3c[key]
                    preds_mc_3c.append(prob_dry_window_mc(q_3c, window, MC_SAMPLES, rng))
                if preds_mc_3e is not None:
                    q_3e = by_lead_date_3e[key]
                    preds_mc_3e.append(prob_dry_window_mc(q_3e, window, MC_SAMPLES, rng))
                if preds_mc_cal is not None:
                    iso = isotonic_by_lead.get(lead)  # type: ignore[union-attr]
                    if iso is None:
                        # No calibrator for this lead in the bundle — skip cal
                        # variant entirely for this station/lead. Future runs
                        # can fall back to identity, but we want the bake-off
                        # to flag this explicitly rather than silently degrade.
                        preds_mc_cal = None
                    else:
                        q_cal = np.clip(iso.apply(q_raw), 0.0, 1.0)
                        preds_mc_cal.append(prob_dry_window_mc(q_cal, window, MC_SAMPLES, rng))
                labels.append(int(row["observed_dry_window"]))

            if not labels:
                continue

            arr_3b = np.array(preds_3b, dtype="float64")
            arr_3g = np.array(preds_3g, dtype="float64")
            arr_lbl = np.array(labels, dtype="float64")
            arr_mean = (arr_3b + arr_3g) / 2.0

            b3b   = brier(arr_3b, arr_lbl)
            b3g   = brier(arr_3g, arr_lbl)
            bmean = brier(arr_mean, arr_lbl)
            bcal: float | None
            if preds_mc_cal is not None and len(preds_mc_cal) == len(labels):
                bcal = brier(np.array(preds_mc_cal, dtype="float64"), arr_lbl)
            else:
                bcal = None
            b3f: float | None
            b_mean3: float | None
            if preds_3f is not None and len(preds_3f) == len(labels):
                arr_3f = np.array(preds_3f, dtype="float64")
                b3f = brier(arr_3f, arr_lbl)
                b_mean3 = brier((arr_3b + arr_3f + arr_3g) / 3.0, arr_lbl)
            else:
                b3f = None
                b_mean3 = None
            b_mc_3c: float | None
            if preds_mc_3c is not None and len(preds_mc_3c) == len(labels):
                b_mc_3c = brier(np.array(preds_mc_3c, dtype="float64"), arr_lbl)
            else:
                b_mc_3c = None
            b_mc_3e: float | None
            if preds_mc_3e is not None and len(preds_mc_3e) == len(labels):
                b_mc_3e = brier(np.array(preds_mc_3e, dtype="float64"), arr_lbl)
            else:
                b_mc_3e = None
            b_3h: float | None
            if preds_3h is not None and len(preds_3h) == len(labels):
                b_3h = brier(np.array(preds_3h, dtype="float64"), arr_lbl)
            else:
                b_3h = None
            b_3h_3e: float | None
            if preds_3h_3e is not None and len(preds_3h_3e) == len(labels):
                b_3h_3e = brier(np.array(preds_3h_3e, dtype="float64"), arr_lbl)
            else:
                b_3h_3e = None
            b_3i: float | None
            if preds_3i is not None and len(preds_3i) == len(labels):
                b_3i = brier(np.array(preds_3i, dtype="float64"), arr_lbl)
            else:
                b_3i = None
            b_3j: float | None
            if preds_3j is not None and len(preds_3j) == len(labels):
                b_3j = brier(np.array(preds_3j, dtype="float64"), arr_lbl)
            else:
                b_3j = None
            b_3k: float | None
            if preds_3k is not None and len(preds_3k) == len(labels):
                b_3k = brier(np.array(preds_3k, dtype="float64"), arr_lbl)
            else:
                b_3k = None
            b_3l: float | None
            if preds_3l is not None and len(preds_3l) == len(labels):
                b_3l = brier(np.array(preds_3l, dtype="float64"), arr_lbl)
            else:
                b_3l = None
            b_3m: float | None
            if preds_3m is not None and len(preds_3m) == len(labels):
                b_3m = brier(np.array(preds_3m, dtype="float64"), arr_lbl)
            else:
                b_3m = None
            b_6a_iid: float | None
            if preds_6a_iid is not None and len(preds_6a_iid) == len(labels):
                b_6a_iid = brier(np.array(preds_6a_iid, dtype="float64"), arr_lbl)
            else:
                b_6a_iid = None
            b_6a_cop: float | None
            if preds_6a_cop is not None and len(preds_6a_cop) == len(labels):
                b_6a_cop = brier(np.array(preds_6a_cop, dtype="float64"), arr_lbl)
            else:
                b_6a_cop = None
            b_3n: float | None
            if preds_3n is not None and len(preds_3n) == len(labels):
                b_3n = brier(np.array(preds_3n, dtype="float64"), arr_lbl)
            else:
                b_3n = None

            results.append(CellResult(
                station=station, window=window, lead=lead,
                n_days=len(labels),
                obs_rate=float(arr_lbl.mean()),
                brier_3b=b3b,
                brier_3g=b3g,
                brier_mc_cal_3a=bcal,
                brier_mean_3b_3g=bmean,
                brier_3f=b3f,
                brier_mean_3b_3f_3g=b_mean3,
                brier_mc_3c=b_mc_3c,
                brier_mc_3e=b_mc_3e,
                brier_3h=b_3h,
                brier_3h_3e=b_3h_3e,
                brier_3i=b_3i,
                brier_3j=b_3j,
                brier_3k=b_3k,
                brier_3l=b_3l,
                brier_3m=b_3m,
                brier_6a_iid=b_6a_iid,
                brier_6a_copula=b_6a_cop,
                brier_3n=b_3n,
            ))
    return results


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_summary(results: list[CellResult]) -> None:
    print()
    print("=" * 160)
    print("Per-(station, window, lead) Brier — dry-window bake-off (3b / 3g / MC-cal / 3f / MC-3c / ensembles)")
    print("=" * 160)
    print(f"{'station':<28} {'win':>4} {'lead':>5} {'n':>5} {'obs':>5}  "
          f"{'3b':>8} {'3g':>8} {'MC-cal':>8} {'MC-3c':>8} {'MC-3e':>8} {'3f':>8} {'3h(3a)':>8} {'3h(3e)':>8} {'3i':>8} {'3j':>8} {'3k':>8} {'3l':>8} {'3m':>8} {'mean2':>8} {'mean3':>8}  {'best':>14}")
    print("-" * 180)
    for r in sorted(results, key=lambda r: (r.station, r.window, r.lead)):
        cells = [("3b", r.brier_3b), ("3g", r.brier_3g), ("mean2", r.brier_mean_3b_3g)]
        if r.brier_mc_cal_3a is not None:
            cells.append(("MC-cal", r.brier_mc_cal_3a))
        if r.brier_mc_3c is not None:
            cells.append(("MC-3c", r.brier_mc_3c))
        if r.brier_mc_3e is not None:
            cells.append(("MC-3e", r.brier_mc_3e))
        if r.brier_3f is not None:
            cells.append(("3f", r.brier_3f))
        if r.brier_3h is not None:
            cells.append(("3h", r.brier_3h))
        if r.brier_3h_3e is not None:
            cells.append(("3h-3e", r.brier_3h_3e))
        if r.brier_3i is not None:
            cells.append(("3i", r.brier_3i))
        if r.brier_3j is not None:
            cells.append(("3j", r.brier_3j))
        if r.brier_3k is not None:
            cells.append(("3k", r.brier_3k))
        if r.brier_3l is not None:
            cells.append(("3l", r.brier_3l))
        if r.brier_3m is not None:
            cells.append(("3m", r.brier_3m))
        if r.brier_mean_3b_3f_3g is not None:
            cells.append(("mean3", r.brier_mean_3b_3f_3g))
        which, best_val = min(cells, key=lambda kv: kv[1])
        cal_cell   = f"{r.brier_mc_cal_3a:>8.4f}" if r.brier_mc_cal_3a is not None else f"{'—':>8}"
        c3c_cell   = f"{r.brier_mc_3c:>8.4f}"     if r.brier_mc_3c     is not None else f"{'—':>8}"
        e3e_cell   = f"{r.brier_mc_3e:>8.4f}"     if r.brier_mc_3e     is not None else f"{'—':>8}"
        f_cell     = f"{r.brier_3f:>8.4f}"        if r.brier_3f        is not None else f"{'—':>8}"
        h_cell     = f"{r.brier_3h:>8.4f}"        if r.brier_3h        is not None else f"{'—':>8}"
        he_cell    = f"{r.brier_3h_3e:>8.4f}"     if r.brier_3h_3e     is not None else f"{'—':>8}"
        i_cell     = f"{r.brier_3i:>8.4f}"        if r.brier_3i        is not None else f"{'—':>8}"
        j_cell     = f"{r.brier_3j:>8.4f}"        if r.brier_3j        is not None else f"{'—':>8}"
        k_cell     = f"{r.brier_3k:>8.4f}"        if r.brier_3k        is not None else f"{'—':>8}"
        l_cell     = f"{r.brier_3l:>8.4f}"        if r.brier_3l        is not None else f"{'—':>8}"
        m_cell     = f"{r.brier_3m:>8.4f}"        if r.brier_3m        is not None else f"{'—':>8}"
        mean3_cell = f"{r.brier_mean_3b_3f_3g:>8.4f}" if r.brier_mean_3b_3f_3g is not None else f"{'—':>8}"
        print(f"{r.station:<28} {r.window:>3}h {r.lead:>4}h {r.n_days:>5d} {r.obs_rate:>5.2f}  "
              f"{r.brier_3b:>8.4f} {r.brier_3g:>8.4f} {cal_cell} {c3c_cell} {e3e_cell} {f_cell} {h_cell} {he_cell} {i_cell} {j_cell} {k_cell} {l_cell} {m_cell} "
              f"{r.brier_mean_3b_3g:>8.4f} {mean3_cell}  {which}={best_val:.4f}")

    print()
    print("=" * 150)
    print("Aggregate mean Brier across all (station, window, lead) cells")
    print("=" * 150)
    def mean(field: str) -> float:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    b3b = mean("brier_3b")
    b3g = mean("brier_3g")
    bmean = mean("brier_mean_3b_3g")
    bcal = mean("brier_mc_cal_3a")
    b3f = mean("brier_3f")
    bmean3 = mean("brier_mean_3b_3f_3g")
    cells_with_cal = [r for r in results if r.brier_mc_cal_3a is not None]
    cells_with_3f  = [r for r in results if r.brier_3f is not None]
    print(f"  3b              {b3b:.4f}    (baseline)")
    print(f"  3g              {b3g:.4f}    ({100 * (b3b - b3g) / b3b:+.1f}% vs 3b)")
    print(f"  mean(3b, 3g)    {bmean:.4f}    ({100 * (b3b - bmean) / b3b:+.1f}% vs 3b)")
    if not np.isnan(bcal):
        sub3b = float(np.mean([r.brier_3b for r in cells_with_cal]))
        sub3g = float(np.mean([r.brier_3g for r in cells_with_cal]))
        submean = float(np.mean([r.brier_mean_3b_3g for r in cells_with_cal]))
        print(f"  MC-cal-3a       {bcal:.4f}    ({100 * (sub3b - bcal) / sub3b:+.1f}% vs 3b — on {len(cells_with_cal)}/{len(results)} cells with 3a_isotonic available)")
        print(f"    (matched-subset baselines — 3b={sub3b:.4f}, 3g={sub3g:.4f}, mean={submean:.4f})")
    else:
        print("  MC-cal-3a       —         (no station had a 3a_isotonic bundle)")
    if not np.isnan(b3f):
        sub3b_f = float(np.mean([r.brier_3b for r in cells_with_3f]))
        sub3g_f = float(np.mean([r.brier_3g for r in cells_with_3f]))
        submean3 = float(np.mean([r.brier_mean_3b_3f_3g for r in cells_with_3f]))
        print(f"  3f (MLP)        {b3f:.4f}    ({100 * (sub3b_f - b3f) / sub3b_f:+.1f}% vs 3b — on {len(cells_with_3f)}/{len(results)} cells with 3f bundle available)")
        print(f"  mean(3b,3f,3g)  {submean3:.4f}    ({100 * (sub3b_f - submean3) / sub3b_f:+.1f}% vs 3b — same matched subset)")
        print(f"    (matched-subset baselines — 3b={sub3b_f:.4f}, 3g={sub3g_f:.4f})")
    else:
        print("  3f (MLP)        —         (no 3f bundle trained yet)")
    b_mc_3c = mean("brier_mc_3c")
    cells_with_3c = [r for r in results if r.brier_mc_3c is not None]
    if not np.isnan(b_mc_3c):
        sub3b_c = float(np.mean([r.brier_3b for r in cells_with_3c]))
        sub3g_c = float(np.mean([r.brier_3g for r in cells_with_3c]))
        print(f"  MC-3c           {b_mc_3c:.4f}    ({100 * (sub3b_c - b_mc_3c) / sub3b_c:+.1f}% vs 3b — on {len(cells_with_3c)}/{len(results)} cells with 3c bundle available)")
        print(f"    (matched-subset baselines — 3b={sub3b_c:.4f}, 3g={sub3g_c:.4f}; head-to-head: 3g {sub3g_c:.4f} vs MC-3c {b_mc_3c:.4f} -> {100*(sub3g_c - b_mc_3c)/sub3g_c:+.1f}% vs 3g)")
    else:
        print("  MC-3c           —         (no 3c bundle available for any station)")
    b_mc_3e = mean("brier_mc_3e")
    cells_with_3e = [r for r in results if r.brier_mc_3e is not None]
    if not np.isnan(b_mc_3e):
        sub3b_e = float(np.mean([r.brier_3b for r in cells_with_3e]))
        sub3g_e = float(np.mean([r.brier_3g for r in cells_with_3e]))
        print(f"  MC-3e           {b_mc_3e:.4f}    ({100 * (sub3b_e - b_mc_3e) / sub3b_e:+.1f}% vs 3b — on {len(cells_with_3e)}/{len(results)} cells with 3e bundle available)")
        print(f"    (matched-subset baselines — 3b={sub3b_e:.4f}, 3g={sub3g_e:.4f}; head-to-head: 3g {sub3g_e:.4f} vs MC-3e {b_mc_3e:.4f} -> {100*(sub3g_e - b_mc_3e)/sub3g_e:+.1f}% vs 3g)")
    else:
        print("  MC-3e           —         (no 3e bundle available for any station)")
    b_3h = mean("brier_3h")
    cells_with_3h = [r for r in results if r.brier_3h is not None]
    if not np.isnan(b_3h):
        sub3b_h = float(np.mean([r.brier_3b for r in cells_with_3h]))
        sub3g_h = float(np.mean([r.brier_3g for r in cells_with_3h]))
        print(f"  3h (GRU on 3a)  {b_3h:.4f}    ({100 * (sub3b_h - b_3h) / sub3b_h:+.1f}% vs 3b — on {len(cells_with_3h)}/{len(results)} cells with 3h bundle available)")
        print(f"    (matched-subset baselines — 3b={sub3b_h:.4f}, 3g={sub3g_h:.4f}; head-to-head: 3g {sub3g_h:.4f} vs 3h {b_3h:.4f} -> {100*(sub3g_h - b_3h)/sub3g_h:+.1f}% vs 3g)")
    else:
        print("  3h (GRU on 3a)  —         (no 3h bundle trained yet)")
    b_3h_3e = mean("brier_3h_3e")
    cells_with_3h_3e = [r for r in results if r.brier_3h_3e is not None]
    if not np.isnan(b_3h_3e):
        sub3b_he = float(np.mean([r.brier_3b for r in cells_with_3h_3e]))
        sub3g_he = float(np.mean([r.brier_3g for r in cells_with_3h_3e]))
        sub3h_he = float(np.mean([r.brier_3h for r in cells_with_3h_3e if r.brier_3h is not None]))
        print(f"  3h (GRU on 3e)  {b_3h_3e:.4f}    ({100 * (sub3b_he - b_3h_3e) / sub3b_he:+.1f}% vs 3b — on {len(cells_with_3h_3e)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b_he:.4f}, 3g={sub3g_he:.4f}, 3h-on-3a={sub3h_he:.4f}; head-to-head GRU sources: 3h-on-3a {sub3h_he:.4f} vs 3h-on-3e {b_3h_3e:.4f} -> {100*(sub3h_he - b_3h_3e)/sub3h_he:+.1f}% vs 3h-on-3a)")
    else:
        print("  3h (GRU on 3e)  —         (3h_3e bundle not trained yet)")
    b_3i = mean("brier_3i")
    cells_with_3i = [r for r in results if r.brier_3i is not None]
    if not np.isnan(b_3i):
        sub3b_i = float(np.mean([r.brier_3b for r in cells_with_3i]))
        sub3g_i = float(np.mean([r.brier_3g for r in cells_with_3i]))
        sub3h_i = float(np.mean([r.brier_3h for r in cells_with_3i if r.brier_3h is not None]))
        print(f"  3i (GRU on NWP) {b_3i:.4f}    ({100 * (sub3b_i - b_3i) / sub3b_i:+.1f}% vs 3b — on {len(cells_with_3i)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b_i:.4f}, 3g={sub3g_i:.4f}, 3h-on-3a={sub3h_i:.4f}; head-to-head: 3g {sub3g_i:.4f} vs 3i {b_3i:.4f} -> {100*(sub3g_i - b_3i)/sub3g_i:+.1f}% vs 3g)")
    else:
        print("  3i (GRU on NWP) —         (3i bundle not trained yet)")
    b_3j = mean("brier_3j")
    cells_with_3j = [r for r in results if r.brier_3j is not None]
    if not np.isnan(b_3j):
        sub3b_j = float(np.mean([r.brier_3b for r in cells_with_3j]))
        sub3g_j = float(np.mean([r.brier_3g for r in cells_with_3j]))
        print(f"  3j (copula MC)  {b_3j:.4f}    ({100 * (sub3b_j - b_3j) / sub3b_j:+.1f}% vs 3b — on {len(cells_with_3j)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b_j:.4f}, 3g={sub3g_j:.4f}; head-to-head: 3g {sub3g_j:.4f} vs 3j {b_3j:.4f} -> {100*(sub3g_j - b_3j)/sub3g_j:+.1f}% vs 3g  [POSITIVE = copula beats iid])")
    else:
        print("  3j (copula MC)  —         (3j bundle not run yet)")
    b_3k = mean("brier_3k")
    cells_with_3k = [r for r in results if r.brier_3k is not None]
    if not np.isnan(b_3k):
        sub3b_k = float(np.mean([r.brier_3b for r in cells_with_3k]))
        sub3g_k = float(np.mean([r.brier_3g for r in cells_with_3k]))
        print(f"  3k (temp-iid)   {b_3k:.4f}    ({100 * (sub3b_k - b_3k) / sub3b_k:+.1f}% vs 3b — on {len(cells_with_3k)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b_k:.4f}, 3g={sub3g_k:.4f}; head-to-head: 3g {sub3g_k:.4f} vs 3k {b_3k:.4f} -> {100*(sub3g_k - b_3k)/sub3g_k:+.1f}% vs 3g)")
    else:
        print("  3k (temp-iid)   —         (3k bundle not run yet)")
    b_3l = mean("brier_3l")
    cells_with_3l = [r for r in results if r.brier_3l is not None]
    if not np.isnan(b_3l):
        sub3b_l = float(np.mean([r.brier_3b for r in cells_with_3l]))
        sub3g_l = float(np.mean([r.brier_3g for r in cells_with_3l]))
        print(f"  3l (feat-LR)    {b_3l:.4f}    ({100 * (sub3b_l - b_3l) / sub3b_l:+.1f}% vs 3b — on {len(cells_with_3l)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b_l:.4f}, 3g={sub3g_l:.4f}; head-to-head: 3g {sub3g_l:.4f} vs 3l {b_3l:.4f} -> {100*(sub3g_l - b_3l)/sub3g_l:+.1f}% vs 3g)")
    else:
        print("  3l (feat-LR)    —         (3l bundle not run yet)")
    b_3m = mean("brier_3m")
    cells_with_3m = [r for r in results if r.brier_3m is not None]
    if not np.isnan(b_3m):
        sub3b_m = float(np.mean([r.brier_3b for r in cells_with_3m]))
        sub3g_m = float(np.mean([r.brier_3g for r in cells_with_3m]))
        sub3j_m = float(np.mean([r.brier_3j for r in cells_with_3m if r.brier_3j is not None]))
        print(f"  3m (alpha-shr)  {b_3m:.4f}    ({100 * (sub3b_m - b_3m) / sub3b_m:+.1f}% vs 3b — on {len(cells_with_3m)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b_m:.4f}, 3g={sub3g_m:.4f}, 3j={sub3j_m:.4f}; "
              f"head-to-head: 3g {sub3g_m:.4f} vs 3m {b_3m:.4f} -> {100*(sub3g_m - b_3m)/sub3g_m:+.1f}% vs 3g; "
              f"3j {sub3j_m:.4f} vs 3m -> {100*(sub3j_m - b_3m)/sub3j_m:+.1f}% vs 3j)")
    else:
        print("  3m (alpha-shr)  —         (3m bundle not run yet)")
    b_6a_iid = mean("brier_6a_iid")
    cells_6ai = [r for r in results if r.brier_6a_iid is not None]
    if not np.isnan(b_6a_iid):
        sub3b  = float(np.mean([r.brier_3b for r in cells_6ai]))
        sub3g  = float(np.mean([r.brier_3g for r in cells_6ai]))
        print(f"  6a-iid          {b_6a_iid:.4f}    ({100 * (sub3b - b_6a_iid) / sub3b:+.1f}% vs 3b — on {len(cells_6ai)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b:.4f}, 3g={sub3g:.4f}; head-to-head: 3g {sub3g:.4f} vs 6a-iid {b_6a_iid:.4f} -> {100*(sub3g - b_6a_iid)/sub3g:+.1f}% vs 3g)")
    else:
        print("  6a-iid          —         (6a bundle not trained yet)")
    b_6a_cop = mean("brier_6a_copula")
    cells_6ac = [r for r in results if r.brier_6a_copula is not None]
    if not np.isnan(b_6a_cop):
        sub3b  = float(np.mean([r.brier_3b for r in cells_6ac]))
        sub3g  = float(np.mean([r.brier_3g for r in cells_6ac]))
        sub3j  = float(np.mean([r.brier_3j for r in cells_6ac if r.brier_3j is not None]))
        print(f"  6a-copula       {b_6a_cop:.4f}    ({100 * (sub3b - b_6a_cop) / sub3b:+.1f}% vs 3b — on {len(cells_6ac)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b:.4f}, 3g={sub3g:.4f}, 3j={sub3j:.4f}; head-to-head: 3g {sub3g:.4f} vs 6a-cop {b_6a_cop:.4f} -> {100*(sub3g - b_6a_cop)/sub3g:+.1f}% vs 3g; 3j {sub3j:.4f} vs 6a-cop -> {100*(sub3j - b_6a_cop)/sub3j:+.1f}% vs 3j)")
    else:
        print("  6a-copula       —         (6a bundle not trained yet)")
    b_3n = mean("brier_3n")
    cells_3n = [r for r in results if r.brier_3n is not None]
    if not np.isnan(b_3n):
        sub3b  = float(np.mean([r.brier_3b for r in cells_3n]))
        sub3g  = float(np.mean([r.brier_3g for r in cells_3n]))
        sub3j  = float(np.mean([r.brier_3j for r in cells_3n if r.brier_3j is not None]))
        print(f"  3n (regime-MC)  {b_3n:.4f}    ({100 * (sub3b - b_3n) / sub3b:+.1f}% vs 3b — on {len(cells_3n)}/{len(results)} cells)")
        print(f"    (matched-subset — 3b={sub3b:.4f}, 3g={sub3g:.4f}, 3j={sub3j:.4f}; "
              f"head-to-head: 3g {sub3g:.4f} vs 3n {b_3n:.4f} -> {100*(sub3g - b_3n)/sub3g:+.1f}% vs 3g; "
              f"3j {sub3j:.4f} vs 3n -> {100*(sub3j - b_3n)/sub3j:+.1f}% vs 3j)")
    else:
        print("  3n (regime-MC)  —         (3n bundle not trained yet)")

    print()
    print("=" * 150)
    print("Per-window aggregate (averaged across stations + leads)")
    print("=" * 150)
    for w in sorted({r.window for r in results}):
        subset = [r for r in results if r.window == w]
        b3b   = float(np.mean([r.brier_3b for r in subset]))
        b3g   = float(np.mean([r.brier_3g for r in subset]))
        bmean = float(np.mean([r.brier_mean_3b_3g for r in subset]))
        cal_sub = [r.brier_mc_cal_3a for r in subset if r.brier_mc_cal_3a is not None]
        f_sub   = [r.brier_3f for r in subset if r.brier_3f is not None]
        m3_sub  = [r.brier_mean_3b_3f_3g for r in subset if r.brier_mean_3b_3f_3g is not None]
        c3c_sub = [r.brier_mc_3c for r in subset if r.brier_mc_3c is not None]
        e3e_sub = [r.brier_mc_3e for r in subset if r.brier_mc_3e is not None]
        h_sub   = [r.brier_3h for r in subset if r.brier_3h is not None]
        he_sub  = [r.brier_3h_3e for r in subset if r.brier_3h_3e is not None]
        i_sub   = [r.brier_3i for r in subset if r.brier_3i is not None]
        j_sub   = [r.brier_3j for r in subset if r.brier_3j is not None]
        k_sub   = [r.brier_3k for r in subset if r.brier_3k is not None]
        l_sub   = [r.brier_3l for r in subset if r.brier_3l is not None]
        m_sub   = [r.brier_3m for r in subset if r.brier_3m is not None]
        ai_sub  = [r.brier_6a_iid for r in subset if r.brier_6a_iid is not None]
        ac_sub  = [r.brier_6a_copula for r in subset if r.brier_6a_copula is not None]
        n_sub   = [r.brier_3n for r in subset if r.brier_3n is not None]
        cal_str = f"{np.mean(cal_sub):.4f}" if cal_sub else "—"
        f_str   = f"{np.mean(f_sub):.4f}"   if f_sub   else "—"
        m3_str  = f"{np.mean(m3_sub):.4f}"  if m3_sub  else "—"
        c3c_str = f"{np.mean(c3c_sub):.4f}" if c3c_sub else "—"
        e3e_str = f"{np.mean(e3e_sub):.4f}" if e3e_sub else "—"
        h_str   = f"{np.mean(h_sub):.4f}"   if h_sub   else "—"
        he_str  = f"{np.mean(he_sub):.4f}"  if he_sub  else "—"
        i_str   = f"{np.mean(i_sub):.4f}"   if i_sub   else "—"
        j_str   = f"{np.mean(j_sub):.4f}"   if j_sub   else "—"
        k_str   = f"{np.mean(k_sub):.4f}"   if k_sub   else "—"
        l_str   = f"{np.mean(l_sub):.4f}"   if l_sub   else "—"
        m_str   = f"{np.mean(m_sub):.4f}"   if m_sub   else "—"
        ai_str  = f"{np.mean(ai_sub):.4f}"  if ai_sub  else "—"
        ac_str  = f"{np.mean(ac_sub):.4f}"  if ac_sub  else "—"
        n_str   = f"{np.mean(n_sub):.4f}"   if n_sub   else "—"
        print(f"  window {w}h:  3b={b3b:.4f}  3g={b3g:.4f}  3j={j_str}  3n={n_str}  3m={m_str}  6a-iid={ai_str}  6a-cop={ac_str}  3f={f_str}  3h(3a)={h_str}  mean2={bmean:.4f}  mean3={m3_str}")


def write_csv(results: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "station": r.station, "window": r.window, "lead": r.lead,
            "n_days": r.n_days, "obs_rate": r.obs_rate,
            "brier_3b": r.brier_3b, "brier_3g": r.brier_3g,
            "brier_mc_cal_3a": r.brier_mc_cal_3a if r.brier_mc_cal_3a is not None else "",
            "brier_mc_3c": r.brier_mc_3c if r.brier_mc_3c is not None else "",
            "brier_mc_3e": r.brier_mc_3e if r.brier_mc_3e is not None else "",
            "brier_3h": r.brier_3h if r.brier_3h is not None else "",
            "brier_3h_3e": r.brier_3h_3e if r.brier_3h_3e is not None else "",
            "brier_3i": r.brier_3i if r.brier_3i is not None else "",
            "brier_3j": r.brier_3j if r.brier_3j is not None else "",
            "brier_3k": r.brier_3k if r.brier_3k is not None else "",
            "brier_3l": r.brier_3l if r.brier_3l is not None else "",
            "brier_3m": r.brier_3m if r.brier_3m is not None else "",
            "brier_6a_iid": r.brier_6a_iid if r.brier_6a_iid is not None else "",
            "brier_6a_copula": r.brier_6a_copula if r.brier_6a_copula is not None else "",
            "brier_mean_3b_3g": r.brier_mean_3b_3g,
            "brier_3f": r.brier_3f if r.brier_3f is not None else "",
            "brier_mean_3b_3f_3g": r.brier_mean_3b_3f_3g if r.brier_mean_3b_3f_3g is not None else "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nWrote {len(rows)} rows ->{path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]

    print(f"4-way dry-window bake-off — stations={stations}, leads={leads}, windows={windows}")
    print(f"  MC samples per cell: {MC_SAMPLES}, seed: {SEED}\n")

    rng = np.random.default_rng(SEED)
    all_results: list[CellResult] = []
    for station in stations:
        p_3a = find_latest_with_test_predictions(PRECIP_MODELS_ROOT, station, phase_suffix=None)
        if p_3a is None:
            print(f"::warning::skipping {station} — no 3a test_predictions parquet on disk")
            continue
        # Per-window 3b parquets: load each into a dict keyed by window int.
        df_3b_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3b = find_3b_test_predictions(station, w)
            if p_3b is None:
                print(f"::warning::{station} window {w}h: no 3b test_predictions parquet (retrain 3b?)")
                continue
            df_3b_by_window[w] = pd.read_parquet(p_3b)
        if not df_3b_by_window:
            print(f"::warning::skipping {station} — no 3b parquets found for any window")
            continue

        # Per-window 3f parquets — same shape as 3b; absent if 3f hasn't
        # been retrained for that (station, window).
        df_3f_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3f = find_3f_test_predictions(station, w)
            if p_3f is None:
                continue
            df_3f_by_window[w] = pd.read_parquet(p_3f)
        # Per-window 3h parquets — bidirectional GRU over the 9-hour 3a
        # replay sequence. Written by train_3h_rnn.py, same schema.
        df_3h_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3h = find_3h_test_predictions(station, w)
            if p_3h is None:
                continue
            df_3h_by_window[w] = pd.read_parquet(p_3h)
        # Same GRU but trained on 3e's replay. --source 3e on train_3h_rnn.py.
        df_3h_3e_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3h_3e = find_3h_3e_test_predictions(station, w)
            if p_3h_3e is None:
                continue
            df_3h_3e_by_window[w] = pd.read_parquet(p_3h_3e)
        # 3i: GRU on raw per-NWP daytime hourly precip — train_3i_rnn_nwp.py
        df_3i_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3i = find_3i_test_predictions(station, w)
            if p_3i is None:
                continue
            df_3i_by_window[w] = pd.read_parquet(p_3i)
        # 3j: Gaussian-copula MC over 3a daytime q — train_3j_mc_copula.py
        df_3j_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3j = find_3j_test_predictions(station, w)
            if p_3j is None:
                continue
            df_3j_by_window[w] = pd.read_parquet(p_3j)
        # 3k: temperature-scaled iid MC — train_3k_temperature.py
        df_3k_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3k = find_3k_test_predictions(station, w)
            if p_3k is None:
                continue
            df_3k_by_window[w] = pd.read_parquet(p_3k)
        # 3l: engineered features + LR — train_3l_feature_lr.py
        df_3l_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3l = find_3l_test_predictions(station, w)
            if p_3l is None:
                continue
            df_3l_by_window[w] = pd.read_parquet(p_3l)
        # 3m: alpha-shrinkage copula <-> iid — train_3m_alpha_shrinkage.py
        df_3m_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3m = find_3m_test_predictions(station, w)
            if p_3m is None:
                continue
            df_3m_by_window[w] = pd.read_parquet(p_3m)
        # 6a: joint-loss MLP — train_6a_joint_loss.py. Two outputs per cell:
        # iid MC and copula MC over the same trained hourly q-vectors.
        df_6a_iid_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p = find_6a_iid_test_predictions(station, w)
            if p is None:
                continue
            df_6a_iid_by_window[w] = pd.read_parquet(p)
        df_6a_copula_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p = find_6a_copula_test_predictions(station, w)
            if p is None:
                continue
            df_6a_copula_by_window[w] = pd.read_parquet(p)
        # 3n: regime-conditioned copula MC — DryWindowTrainCommand.RunPhase3nAsync.
        df_3n_by_window: dict[int, pd.DataFrame] = {}
        for w in windows:
            p_3n = find_3n_test_predictions(station, w)
            if p_3n is None:
                continue
            df_3n_by_window[w] = pd.read_parquet(p_3n)

        # 3c hourly P(wet) — feeds MC-3c (parallel to 3a-driven 3g).
        # Test slice should be identical to 3a's (both from the same
        # 70/15/15 chronological split of the same offset_day source).
        p_3c = find_3c_test_predictions(station)
        df_3c = pd.read_parquet(p_3c) if p_3c is not None else None
        # 3e hourly P(wet) — feeds MC-3e. Yesterday's 3e bake-off scored
        # 24-UTC vectors against 24-UTC labels; today we re-test on the
        # daytime question, the same one production 3g serves.
        p_3e = find_3e_test_predictions(station)
        df_3e = pd.read_parquet(p_3e) if p_3e is not None else None

        cal_path = find_3a_isotonic_calibration(station)
        isotonic_by_lead = load_isotonic_by_lead(cal_path) if cal_path else None
        cal_note = f"  3a_isotonic: {cal_path.parent.name}" if cal_path else "  3a_isotonic: (none — MC-cal will be omitted for this station)"

        df_3a = pd.read_parquet(p_3a)
        df_3a["valid_time"] = pd.to_datetime(df_3a["valid_time"], utc=True).dt.tz_localize(None)
        a_min, a_max = df_3a["valid_time"].min(), df_3a["valid_time"].max()
        print(f"{station}:")
        print(f"  3a: {len(df_3a):,} rows, test range {a_min:%Y-%m-%d} ->{a_max:%Y-%m-%d} ({p_3a.parent.name})")
        # Per-window 3b test slices — log explicitly so we can SEE same-date-range
        # alignment between 3a and 3b (both should be the trailing 15% chronological
        # chunk of the same offset_day source). The user-facing same-NWP-data
        # check leans on these prints matching.
        for w, df in df_3b_by_window.items():
            df["target_date"] = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None)
            b_min, b_max = df["target_date"].min(), df["target_date"].max()
            print(f"  3b window {w}h: {len(df):,} rows, test range {b_min:%Y-%m-%d} ->{b_max:%Y-%m-%d} ({find_3b_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        for w, df in df_3f_by_window.items():
            f_min, f_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3f window {w}h: {len(df):,} rows, test range {f_min:%Y-%m-%d} ->{f_max:%Y-%m-%d} ({find_3f_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3f_by_window:
            print("  3f: (none — MLP not yet trained for this station)")
        for w, df in df_3h_by_window.items():
            h_min, h_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3h window {w}h: {len(df):,} rows, test range {h_min:%Y-%m-%d} ->{h_max:%Y-%m-%d} ({find_3h_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3h_by_window:
            print("  3h: (none — GRU not yet trained for this station)")
        for w, df in df_3h_3e_by_window.items():
            he_min, he_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3h(3e) window {w}h: {len(df):,} rows, test range {he_min:%Y-%m-%d} ->{he_max:%Y-%m-%d} ({find_3h_3e_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3h_3e_by_window:
            print("  3h(3e): (none — GRU-on-3e not yet trained for this station)")
        for w, df in df_3i_by_window.items():
            i_min, i_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3i window {w}h: {len(df):,} rows, test range {i_min:%Y-%m-%d} ->{i_max:%Y-%m-%d} ({find_3i_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3i_by_window:
            print("  3i: (none — GRU-on-raw-NWP not yet trained for this station)")
        for w, df in df_3j_by_window.items():
            j_min, j_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3j window {w}h: {len(df):,} rows, test range {j_min:%Y-%m-%d} ->{j_max:%Y-%m-%d} ({find_3j_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3j_by_window:
            print("  3j: (none — Gaussian-copula MC not yet run for this station)")
        for w, df in df_3k_by_window.items():
            k_min, k_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3k window {w}h: {len(df):,} rows, test range {k_min:%Y-%m-%d} ->{k_max:%Y-%m-%d} ({find_3k_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3k_by_window:
            print("  3k: (none — temperature-MC not yet run for this station)")
        for w, df in df_3l_by_window.items():
            l_min, l_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3l window {w}h: {len(df):,} rows, test range {l_min:%Y-%m-%d} ->{l_max:%Y-%m-%d} ({find_3l_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3l_by_window:
            print("  3l: (none — engineered-feature LR not yet run for this station)")
        for w, df in df_3m_by_window.items():
            m_min, m_max = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  3m window {w}h: {len(df):,} rows, test range {m_min:%Y-%m-%d} ->{m_max:%Y-%m-%d} ({find_3m_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_3m_by_window:
            print("  3m: (none — alpha-shrinkage not yet run for this station)")
        for w, df in df_6a_iid_by_window.items():
            mn, mx = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  6a-iid window {w}h: {len(df):,} rows, test range {mn:%Y-%m-%d} ->{mx:%Y-%m-%d} ({find_6a_iid_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_6a_iid_by_window:
            print("  6a-iid: (none — joint-loss MLP not yet trained for this station)")
        for w, df in df_6a_copula_by_window.items():
            mn, mx = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None).max()
            print(f"  6a-copula window {w}h: {len(df):,} rows, test range {mn:%Y-%m-%d} ->{mx:%Y-%m-%d} ({find_6a_copula_test_predictions(station, w).parent.name})")  # type: ignore[union-attr]
        if not df_6a_copula_by_window:
            print("  6a-copula: (none — joint-loss MLP not yet trained for this station)")
        if df_3c is not None and p_3c is not None:
            c_min, c_max = pd.to_datetime(df_3c["valid_time"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df_3c["valid_time"], utc=True).dt.tz_localize(None).max()
            print(f"  3c: {len(df_3c):,} rows, test range {c_min:%Y-%m-%d} ->{c_max:%Y-%m-%d} ({p_3c.parent.name})")
        else:
            print("  3c: (none — MC-3c will be omitted for this station)")
        if df_3e is not None and p_3e is not None:
            e_min, e_max = pd.to_datetime(df_3e["valid_time"], utc=True).dt.tz_localize(None).min(), pd.to_datetime(df_3e["valid_time"], utc=True).dt.tz_localize(None).max()
            print(f"  3e: {len(df_3e):,} rows, test range {e_min:%Y-%m-%d} ->{e_max:%Y-%m-%d} ({p_3e.parent.name})")
        else:
            print("  3e: (none — MC-3e will be omitted for this station)")
        print(cal_note)

        all_results.extend(evaluate_station(
            station, df_3a, df_3b_by_window, df_3f_by_window, df_3h_by_window, df_3h_3e_by_window,
            df_3i_by_window, df_3j_by_window, df_3n_by_window,
            df_3k_by_window, df_3l_by_window, df_3m_by_window,
            df_6a_iid_by_window, df_6a_copula_by_window,
            df_3c, df_3e, isotonic_by_lead, leads, windows, rng,
        ))

    if not all_results:
        print("::error::no cells produced a result")
        return 3

    print_summary(all_results)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    write_csv(all_results, ROOT / "reports" / f"4way_bakeoff_{ts}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
