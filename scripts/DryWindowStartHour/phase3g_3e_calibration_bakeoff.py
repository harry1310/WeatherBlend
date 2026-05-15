"""3g hourly-source bake-off — does CALIBRATING 3e's hourly P(wet) make
MC-over-3e beat 3g (MC-over-3a) on the dry-window question?

Context
-------
3e (TorchSharp MLP) has better hourly P(wet) Brier than 3a, but the
2026-05-12 3e-source bake-off found MC-over-3e worse for dry windows.
That bake-off ran on the *24h UTC day*, not the daytime window — the
wrong question (see reference_dry_window_daytime_question). On the
corrected daytime question the 2026-05-14 4-way bake-off shows
MC-over-3e ~= 3g aggregate, and 3e edging 3g at the 6h window.

Hypothesis (re-tested here): 3e is an MLP, so its hourly P(wet) is
probably miscalibrated (sharp but overconfident). The dry-window MC
compounds per-hour calibration error multiplicatively over the window,
so fixing 3e's hourly reliability — judged on *dry-window* Brier, not
hourly Brier — could turn the 6h tie/edge into a real win.

Note: MC-over-calibrated-*3a* already lost (4-way `mc_cal_3a` 0.1413 vs
3g 0.1254) — but 3a is natively well-calibrated ("no PAV needed", per
the 3g docstring), so that result does NOT transfer to 3e.

Variants
--------
  3g        MC over 3a's daytime hourly P(wet)         (champion baseline)
  3e-raw    MC over 3e's daytime hourly P(wet)         (confirm the 6h edge)
  3e-iso    MC over isotonic-calibrated 3e
  3e-beta   MC over beta-calibrated 3e (Kull et al.)

Design
------
3e bundles persist only test_predictions.parquet (no val slice). A naive
chronological fit/score split fails twice: it scores only a seasonal
sub-slice (Brier not comparable to prior bake-offs) AND fits the
calibrator on winter while scoring spring (a seasonal confound on the
calibration verdict). Instead: out-of-fold (K-fold by contiguous date
block) calibration over the FULL test slice — each day's calibrated
prediction comes from a calibrator fit on the other folds, so there's
no leakage, no seasonal mismatch, and the dry-window Brier is scored on
all 125 days (directly comparable to the 4-way bake-off, 3g ~= 0.125).
Calibrators are fit per lead (lead-72 forecasts are softer than
lead-24), pooled across stations. Daytime slicing mirrors
DryWindow3gPredictor.ExtractDaytimeQ. MC at 10000 samples x 4 seeds so
the 6h verdict isn't swamped by sampling noise.

Outputs per-window Brier per variant + a dry-tail reliability/ECE table.
"""

from __future__ import annotations

import glob
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sys.exit("scikit-learn not installed — `pip install scikit-learn` and re-run.")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRECIP = os.path.join(REPO, "data", "models", "precipitation")
STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
LEADS = [24, 48, 72]
WINDOWS = [3, 4, 6]
DAYTIME_TZ = ZoneInfo("Europe/London")
DAYTIME_LOCAL_START, DAYTIME_LOCAL_END = 9, 18
MC_SAMPLES = 10_000
SEEDS = [42, 1, 7, 99]
N_FOLDS = 5  # out-of-fold calibration: contiguous date blocks
EPS = 1e-6


def daytime_utc_hours(target_date: pd.Timestamp) -> tuple[int, int]:
    """(start, end_exclusive) UTC hours for 09-18 Europe/London on this
    date — mirrors DaytimeWindow.UtcHourRangeFor. 9-18 in GMT, 8-17 in BST."""
    start = pd.Timestamp(year=target_date.year, month=target_date.month,
                         day=target_date.day, hour=DAYTIME_LOCAL_START, tz=DAYTIME_TZ)
    end = pd.Timestamp(year=target_date.year, month=target_date.month,
                       day=target_date.day, hour=DAYTIME_LOCAL_END, tz=DAYTIME_TZ)
    return int(start.tz_convert("UTC").hour), int(end.tz_convert("UTC").hour)


def latest_test_predictions(station: str, phase: str | None) -> str | None:
    """Newest bundle's test_predictions.parquet. phase=None = 3a champion
    (unsuffixed dir)."""
    sdir = os.path.join(PRECIP, station)
    if not os.path.isdir(sdir):
        return None
    hits = []
    for d in sorted(glob.glob(os.path.join(sdir, "v*"))):
        name = os.path.basename(d)
        if phase:
            if f"phase{phase}" not in name:
                continue
        elif "phase" in name:
            continue
        p = os.path.join(d, "test_predictions.parquet")
        if os.path.exists(p):
            hits.append(p)
    return hits[-1] if hits else None


def load_hourly(station: str, phase: str | None) -> pd.DataFrame | None:
    p = latest_test_predictions(station, phase)
    if p is None:
        return None
    df = pd.read_parquet(p)[["valid_time", "lead", "p_wet", "observed_wet"]].copy()
    # 3e/3a test_predictions carry leads 24/48/72/96/120; the dry-window
    # question is scoped to {24,48,72}. Filter here so every variant scores
    # the SAME 9 cells (3 stations x 3 leads) — otherwise the per-lead
    # calibrators (24/48/72 only) and the raw variants drift onto different
    # cell sets and the aggregate comparison is apples-to-oranges.
    df = df[df["lead"].isin(LEADS)].copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    df["station"] = station
    return df


def daytime_cells(df: pd.DataFrame) -> dict[tuple[str, int, pd.Timestamp], pd.DataFrame]:
    """(station, lead, target_date) -> daytime-sliced hourly rows, only for
    days with every daytime hour present."""
    out: dict[tuple[str, int, pd.Timestamp], pd.DataFrame] = {}
    for (station, lead, td), grp in df.groupby(["station", "lead", "target_date"]):
        s, e = daytime_utc_hours(pd.Timestamp(td))
        day = grp[(grp["hour"] >= s) & (grp["hour"] < e)].sort_values("valid_time")
        if len(day) != e - s:
            continue
        out[(station, int(lead), pd.Timestamp(td))] = day
    return out


def beta_calibration_fit(p: np.ndarray, y: np.ndarray):
    """Kull et al. beta calibration: logistic regression on
    [ln p, -ln(1-p)] -> y. Returns a callable p -> calibrated p."""
    p = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)

    def apply(q: np.ndarray) -> np.ndarray:
        q = np.clip(q, EPS, 1 - EPS)
        Xq = np.column_stack([np.log(q), -np.log(1 - q)])
        return lr.predict_proba(Xq)[:, 1]

    return apply


def isotonic_fit(p: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return lambda q: iso.predict(np.clip(q, 0.0, 1.0))


def has_dry_block(samples: np.ndarray, window: int) -> np.ndarray:
    """Vectorised: per MC sample (rows; 1=wet, 0=dry) does a contiguous
    run of >= window dry hours exist?"""
    n_hours = samples.shape[1]
    if window > n_hours:
        return np.zeros(samples.shape[0], dtype=bool)
    hit = np.zeros(samples.shape[0], dtype=bool)
    for i in range(n_hours - window + 1):
        hit |= samples[:, i:i + window].sum(axis=1) == 0
    return hit


def mc_prob(q: np.ndarray, window: int, rng: np.random.Generator) -> float:
    samples = (rng.random((MC_SAMPLES, len(q))) < q).astype(np.int8)
    return float(has_dry_block(samples, window).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def dry_tail_ece(p: np.ndarray, y: np.ndarray, hi: float = 0.30, bins: int = 6) -> float:
    """Expected calibration error restricted to predictions in [0, hi]."""
    mask = p <= hi
    if mask.sum() < 20:
        return float("nan")
    p, y = p[mask], y[mask]
    edges = np.linspace(0, hi, bins + 1)
    ece, n = 0.0, len(p)
    for b in range(bins):
        sel = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if sel.sum() == 0:
            continue
        ece += sel.sum() / n * abs(p[sel].mean() - y[sel].mean())
    return ece


def main() -> None:
    # ---- load + daytime-slice ------------------------------------------
    df_3a = pd.concat([d for s in STATIONS if (d := load_hourly(s, None)) is not None])
    df_3e = pd.concat([d for s in STATIONS if (d := load_hourly(s, "3e")) is not None])
    cells_3a = daytime_cells(df_3a)
    cells_3e = daytime_cells(df_3e)
    common = sorted(set(cells_3a) & set(cells_3e))
    if not common:
        sys.exit("no overlapping (station, lead, target_date) daytime cells")

    # ---- fold assignment: contiguous date blocks -----------------------
    # Out-of-fold calibration. Each unique target_date -> one of N_FOLDS
    # contiguous blocks. A day's calibrated q comes from a calibrator fit
    # on the OTHER folds — no leakage, and (unlike a single chrono split)
    # every day is scored, so dry-window Brier is comparable to the 4-way
    # bake-off and not biased to one season.
    dates = sorted({td for (_, _, td) in common})
    fold_of = {d: min(i * N_FOLDS // len(dates), N_FOLDS - 1)
               for i, d in enumerate(dates)}
    print(f"days: {len(dates)}  out-of-fold calibration, {N_FOLDS} contiguous blocks")

    # ---- out-of-fold calibrated 3e q-vectors ---------------------------
    # q3e_iso[k] / q3e_beta[k] for every key, each from a calibrator that
    # never saw key k's fold. Fit per (lead, fold), pooled across stations.
    q3e_iso: dict = {}
    q3e_beta: dict = {}
    for lead in LEADS:
        lead_keys = [k for k in common if k[1] == lead]
        for fold in range(N_FOLDS):
            fit_keys = [k for k in lead_keys if fold_of[k[2]] != fold]
            apply_keys = [k for k in lead_keys if fold_of[k[2]] == fold]
            if not fit_keys or not apply_keys:
                continue
            fit_df = pd.concat([cells_3e[k] for k in fit_keys])
            p = fit_df["p_wet"].to_numpy(dtype=float)
            y = fit_df["observed_wet"].to_numpy(dtype=float)
            iso = isotonic_fit(p, y)
            beta = beta_calibration_fit(p, y)
            for k in apply_keys:
                raw = cells_3e[k]["p_wet"].to_numpy(dtype=float)
                q3e_iso[k] = iso(raw)
                q3e_beta[k] = beta(raw)
        print(f"  lead {lead}h: {N_FOLDS}-fold calibrators fit "
              f"({len(lead_keys)} day-cells)")

    # ---- score on ALL cells --------------------------------------------
    variants = ["3g", "3e-raw", "3e-iso", "3e-beta"]
    per_cell = []  # (station, lead, window, variant, brier)
    rel = {v: {"p": [], "y": []} for v in ["3a", "3e-raw", "3e-iso", "3e-beta"]}

    by_lead_station: dict = {}
    for k in common:
        by_lead_station.setdefault((k[0], k[1]), []).append(k)

    for (station, lead), keys in sorted(by_lead_station.items()):
        keys.sort(key=lambda k: k[2])
        q3a = {k: cells_3a[k]["p_wet"].to_numpy(dtype=float) for k in keys}
        q3e = {k: cells_3e[k]["p_wet"].to_numpy(dtype=float) for k in keys}
        qiso = {k: q3e_iso[k] for k in keys if k in q3e_iso}
        qbeta = {k: q3e_beta[k] for k in keys if k in q3e_beta}
        obs = {k: cells_3e[k]["observed_wet"].to_numpy(dtype=np.int8) for k in keys}

        for k in keys:
            rel["3a"]["p"].append(q3a[k]); rel["3a"]["y"].append(obs[k])
            rel["3e-raw"]["p"].append(q3e[k]); rel["3e-raw"]["y"].append(obs[k])
            if k in qiso:
                rel["3e-iso"]["p"].append(qiso[k]); rel["3e-iso"]["y"].append(obs[k])
            if k in qbeta:
                rel["3e-beta"]["p"].append(qbeta[k]); rel["3e-beta"]["y"].append(obs[k])

        for window in WINDOWS:
            truth = np.array([has_dry_block(obs[k][None, :], window)[0] for k in keys],
                             dtype=float)
            for vname, qmap in [("3g", q3a), ("3e-raw", q3e),
                                ("3e-iso", qiso), ("3e-beta", qbeta)]:
                # all variants must cover the same keys for a fair Brier
                scored = [k for k in keys if k in qmap]
                if len(scored) != len(keys):
                    continue
                seed_briers = []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    preds = np.array([mc_prob(qmap[k], window, rng) for k in keys])
                    seed_briers.append(brier(preds, truth))
                per_cell.append(dict(
                    station=station, lead=lead, window=window, variant=vname,
                    n=len(keys), brier=float(np.mean(seed_briers)),
                    brier_sd=float(np.std(seed_briers)),
                ))

    res = pd.DataFrame(per_cell)

    # ---- report ---------------------------------------------------------
    print("\n=== aggregate Brier by variant (mean over 27 cells, 4-seed mean) ===")
    for v in variants:
        sub = res[res["variant"] == v]
        if sub.empty:
            continue
        print(f"  {v:9s} {sub['brier'].mean():.4f}   "
              f"(seed sd ~{sub['brier_sd'].mean():.4f})")

    print("\n=== per-window Brier ===")
    base = res[res["variant"] == "3g"].set_index(["station", "lead", "window"])["brier"]
    for window in WINDOWS:
        line = f"  {window}h: "
        for v in variants:
            sub = res[(res["variant"] == v) & (res["window"] == window)]
            if sub.empty:
                continue
            m = sub["brier"].mean()
            if v == "3g":
                line += f"3g={m:.4f}  "
            else:
                b3g = res[(res["variant"] == "3g") & (res["window"] == window)]["brier"].mean()
                line += f"{v}={m:.4f}({100*(m-b3g)/b3g:+.1f}%)  "
        print(line)

    print("\n=== 6h window — per cell (the one we care about) ===")
    six = res[res["window"] == 6]
    piv = six.pivot_table(index=["station", "lead"], columns="variant", values="brier")
    for v in variants:
        if v not in piv:
            piv[v] = np.nan
    print(piv[variants].round(4).to_string())
    wins = (piv["3e-iso"] < piv["3g"]).sum() if "3e-iso" in piv else 0
    winsb = (piv["3e-beta"] < piv["3g"]).sum() if "3e-beta" in piv else 0
    winsr = (piv["3e-raw"] < piv["3g"]).sum() if "3e-raw" in piv else 0
    print(f"\n  beats 3g at 6h:  3e-raw {winsr}/9   3e-iso {wins}/9   3e-beta {winsb}/9")

    print("\n=== dry-tail reliability (hourly score set, predictions <= 0.30) ===")
    for v, acc in rel.items():
        if not acc["p"]:
            continue
        p = np.concatenate(acc["p"]); y = np.concatenate(acc["y"]).astype(float)
        ece = dry_tail_ece(p, y)
        tail = p <= 0.30
        bias = p[tail].mean() - y[tail].mean()
        print(f"  {v:9s} dry-tail ECE={ece:.4f}  "
              f"mean_pred-mean_obs={bias:+.4f}  (n_tail={int(tail.sum())})")


if __name__ == "__main__":
    main()
