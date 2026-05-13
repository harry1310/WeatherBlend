"""Phase 3n pre-flight — does NWP agreement separate Σ meaningfully?

The hypothesis: when NWPs agree on per-hour wet/dry, observed sequences
show strong within-day correlation (settled atmosphere → wet hours cluster);
when NWPs disagree, observed sequences are more iid-like (uncertain/showery
regime). If true, regime-conditioned Σ should beat single-Σ 3j at long
windows.

This diagnostic checks the antecedent: do the two Σs (settled vs
unsettled, by median split on NWP agreement) actually differ enough to
matter? Cheap test: ~15 min wallclock for one (station, lead) cell.

Reports (for one cell, Bellever lead 24):
  - mean off-diagonal correlation per bucket
  - Frobenius norm of (Σ_settled - Σ_unsettled)
  - max absolute entry-wise difference
  - both 9×9 matrices side-by-side (rounded)

Threshold for "go build it": Frobenius norm > 1.5 or max abs diff > 0.20.
Below those, the regime axis isn't informative and 3n isn't worth ~1 day
of plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
REPLAY_ROOT = ROOT / "data" / "predictions" / "precipitation_replay"
FORECASTS_GLOB = (ROOT / "data" / "forecasts" / "location=bonehill_rocks" /
                  "model=*" / "**" / "*.parquet").as_posix()

STATION = "ea_bellever_dartmoor"
LEAD = 24
WET_THRESHOLD_MM_H = 0.1  # matches 3a's wet/dry labelling


def find_replay_dir() -> Path:
    station_dir = REPLAY_ROOT / STATION
    cands = [d for d in station_dir.iterdir() if d.is_dir() and "phase" not in d.name]
    if not cands:
        raise FileNotFoundError(f"no replay dir under {station_dir}")
    return max(cands, key=lambda d: d.name)


def load_observed_daytime(replay_dir: Path) -> dict[pd.Timestamp, np.ndarray]:
    """target_date -> 9-hour daytime observed binary sequence."""
    df = pd.read_parquet(replay_dir / f"lead_{LEAD}h.parquet")
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour

    out: dict[pd.Timestamp, np.ndarray] = {}
    for target_date, grp in df.groupby("target_date"):
        target_ts = pd.Timestamp(target_date)
        s, e = daytime_utc_hours(target_ts)
        n = e - s
        sub = grp[(grp["hour"] >= s) & (grp["hour"] < e)].sort_values("valid_time")
        if len(sub) != n:
            continue
        out[target_ts] = sub["Label"].astype(np.int32).to_numpy()
    return out


def load_per_nwp_daytime_wet(target_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, np.ndarray]:
    """target_date -> (n_models, 9) binary wet matrix from raw NWP forecasts.

    Reads offset_day forecasts at the requested lead, thresholds each model's
    precip to a wet/dry binary at WET_THRESHOLD_MM_H, and slices to the
    daytime window. Returns days where ALL configured models have all 9
    daytime hours populated — anything else is dropped (we want clean
    apples-to-apples agreement).
    """
    sql = f"""
    SELECT
      Model,
      ValidTimeUtc,
      CAST(Precipitation AS DOUBLE) AS precip_mm_h
    FROM read_parquet('{FORECASTS_GLOB}', hive_partitioning=true, union_by_name=true)
    WHERE LeadHours = {LEAD}
      AND (RunTimeSource IS NULL OR RunTimeSource = 'offset_day')
      AND Precipitation IS NOT NULL
    """
    conn = duckdb.connect(":memory:")
    df = conn.execute(sql).fetchdf()
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    df["wet"] = (df["precip_mm_h"] >= WET_THRESHOLD_MM_H).astype(np.int8)

    print(f"forecast rows: {len(df):,}  models: {sorted(df['Model'].unique().tolist())}")

    models = sorted(df["Model"].unique().tolist())
    out: dict[pd.Timestamp, np.ndarray] = {}
    for target_ts in target_dates:
        s, e = daytime_utc_hours(target_ts)
        n = e - s
        day = df[(df["target_date"] == target_ts.normalize())
                 & (df["hour"] >= s) & (df["hour"] < e)]
        if day.empty:
            continue
        # (n_models, n_hours) — fill missing model-hours with NaN, drop day if any NaN.
        wet = (day.pivot_table(index="Model", columns="hour", values="wet")
                  .reindex(index=models, columns=list(range(s, e))))
        if wet.isnull().any().any():
            continue
        out[target_ts] = wet.to_numpy(dtype=np.int8)
    return out


def fit_corr(binary_sequences: np.ndarray) -> np.ndarray:
    """9×9 Pearson correlation from (n_days, 9) binary observed sequences."""
    n_days, n = binary_sequences.shape
    mean = binary_sequences.mean(axis=0)
    sigma = np.zeros((n, n))
    for h1 in range(n):
        for h2 in range(n):
            sigma[h1, h2] = np.mean(
                (binary_sequences[:, h1] - mean[h1]) *
                (binary_sequences[:, h2] - mean[h2]))
    stdev = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
    corr = sigma / np.outer(stdev, stdev)
    np.fill_diagonal(corr, 1.0)
    return corr


def main() -> int:
    print(f"=== Σ-diff diagnostic: {STATION} lead {LEAD}h ===\n")

    replay_dir = find_replay_dir()
    print(f"replay dir: {replay_dir.name}")

    observed = load_observed_daytime(replay_dir)
    print(f"observed daytime-complete days: {len(observed):,}")

    per_nwp = load_per_nwp_daytime_wet(list(observed.keys()))
    print(f"NWP-complete days (all 7 models, all 9 hours): {len(per_nwp):,}")

    common = sorted(set(observed.keys()) & set(per_nwp.keys()))
    print(f"intersection: {len(common):,} days\n")

    if len(common) < 200:
        print(f"::warning:: only {len(common)} matched days — diagnostic may be noisy.")

    # Per-day NWP agreement: mean across the 9 hours of
    # max(p_wet, 1-p_wet) where p_wet = fraction of NWPs calling that hour wet.
    # 1.0 = perfect consensus every hour; 0.5 = max disagreement every hour.
    agreement_by_day: dict[pd.Timestamp, float] = {}
    for d in common:
        wet_mat = per_nwp[d]  # (n_models, 9)
        p_wet = wet_mat.mean(axis=0)  # (9,)
        consensus_per_hour = np.maximum(p_wet, 1.0 - p_wet)
        agreement_by_day[d] = float(consensus_per_hour.mean())

    agrees = np.array([agreement_by_day[d] for d in common])
    print(f"agreement distribution:")
    print(f"  min  {agrees.min():.3f}")
    print(f"  Q25  {np.quantile(agrees, 0.25):.3f}")
    print(f"  med  {np.median(agrees):.3f}")
    print(f"  Q75  {np.quantile(agrees, 0.75):.3f}")
    print(f"  max  {agrees.max():.3f}")
    print()

    # Median split — also chronological 70/15/15 split awareness: fit on
    # the first 70% to mirror 3j's train slice. Same split point lets us
    # talk about "what would 3n have fit" exactly.
    n = len(common)
    tr_end = int(np.floor(n * 0.70))
    train_days = common[:tr_end]
    print(f"train slice: {len(train_days)} days "
          f"({train_days[0].date()} → {train_days[-1].date()})")

    train_agree = np.array([agreement_by_day[d] for d in train_days])
    threshold = float(np.median(train_agree))
    print(f"median-split threshold (train): {threshold:.3f}\n")

    train_obs = np.array([observed[d] for d in train_days])

    settled_mask = train_agree >= threshold
    unsettled_mask = ~settled_mask
    print(f"settled (agreement ≥ {threshold:.3f}):    {settled_mask.sum()} days")
    print(f"unsettled (agreement < {threshold:.3f}):  {unsettled_mask.sum()} days\n")

    sigma_settled = fit_corr(train_obs[settled_mask])
    sigma_unsettled = fit_corr(train_obs[unsettled_mask])
    sigma_pool = fit_corr(train_obs)

    def mean_off_diag(m: np.ndarray) -> float:
        ix = ~np.eye(m.shape[0], dtype=bool)
        return float(m[ix].mean())

    print(f"mean off-diagonal correlation:")
    print(f"  pooled (current 3j):  {mean_off_diag(sigma_pool):.3f}")
    print(f"  settled:              {mean_off_diag(sigma_settled):.3f}")
    print(f"  unsettled:            {mean_off_diag(sigma_unsettled):.3f}\n")

    diff = sigma_settled - sigma_unsettled
    fro = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    print(f"Σ_settled − Σ_unsettled:")
    print(f"  Frobenius norm:  {fro:.3f}    (>1.5 → meaningful; <0.5 → essentially identical)")
    print(f"  max abs entry:   {max_abs:.3f}   (>0.20 → meaningful)")

    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"\nΣ_settled:\n{sigma_settled}")
    print(f"\nΣ_unsettled:\n{sigma_unsettled}")
    print(f"\nΔ = Σ_settled - Σ_unsettled:\n{diff}")

    if fro > 1.5 and max_abs > 0.20:
        verdict = "GO: regime axis is meaningfully informative → build Phase 3n"
    elif fro < 0.5 and max_abs < 0.10:
        verdict = "STOP: regime axis isn't separating Σ — 3n unlikely to help"
    else:
        verdict = "MIXED: marginal signal — consider K=3 buckets or a different regime descriptor"
    print(f"\n→ Verdict: {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
