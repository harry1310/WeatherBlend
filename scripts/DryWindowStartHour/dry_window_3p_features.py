"""Phase 3p pre-flight — ensemble-feature extractor + univariate signal diagnostic.

Builds a feature matrix per (station, lead, target_date) from per-NWP
daytime precip trajectories, with binary dry-window labels for windows
{3, 4, 6}h. Mirrors the 3j/3n train slice (same offset_day source, same
canonical 8-NWP set, same daytime UTC hour range).

Features (16 total) capture the ensemble-spread information that gets
collapsed when 3a blends NWPs into a single q-vector. See the
2026-05-14 plan doc / chat thread for the rationale per feature.

This script is read-only on the training data — safe to run while the
dotnet trainer is still computing PAV bundles. Outputs:
  data/features/dry_window_3p/features.parquet

After writing the parquet, prints a univariate signal diagnostic per
feature × per window: Pearson r, AUC, single-feature Brier vs
climatology. Tells us which features carry signal BEFORE we build the
full meta-trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import daytime_utc_hours, has_contiguous_dry_block  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
REPLAY_ROOT = ROOT / "data" / "predictions" / "precipitation_replay"
FORECASTS_GLOB = (ROOT / "data" / "forecasts" / "location=bonehill_rocks" /
                  "model=*" / "**" / "*.parquet").as_posix()
OUT_DIR = ROOT / "data" / "features" / "dry_window_3p"
OUT_PATH = OUT_DIR / "features.parquet"

# Canonical production NWPs (matches DryWindowNwpAgreement in C#).
CANONICAL_MODELS = [
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "meteofrance_seamless",
    "ukmo_seamless",
    "gem_seamless",
    "ecmwf_aifs025_single",
    "jma_seamless",
]
WET_THRESHOLD_MM_H = 0.1
MIN_MODELS_FOR_DAY = 5  # below this, can't compute reliable features

STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
LEADS = [24, 48, 72]
WINDOWS = [3, 4, 6]


def find_replay_dir(station: str) -> Path | None:
    station_dir = REPLAY_ROOT / station
    if not station_dir.is_dir():
        return None
    cands = [d for d in station_dir.iterdir() if d.is_dir() and "phase" not in d.name]
    return max(cands, key=lambda d: d.name) if cands else None


def load_observed_daytime(station: str, lead: int) -> dict[pd.Timestamp, np.ndarray]:
    """target_date → 9-hour daytime observed binary sequence (from 3a replay's Label column)."""
    replay = find_replay_dir(station)
    if replay is None:
        raise FileNotFoundError(f"No 3a replay for {station}")
    df = pd.read_parquet(replay / f"lead_{lead}h.parquet")
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


def load_per_nwp_daytime(lead: int) -> dict[pd.Timestamp, np.ndarray]:
    """target_date → (n_models, n_daytime_hours) precip matrix in mm/h, NaN for missing."""
    sql = f"""
    SELECT Model, ValidTimeUtc, CAST(Precipitation AS DOUBLE) AS precip
    FROM read_parquet('{FORECASTS_GLOB}', hive_partitioning = false, union_by_name = true)
    WHERE LeadHours = {lead}
      AND (RunTimeSource IS NULL OR RunTimeSource = 'offset_day')
      AND Precipitation IS NOT NULL
    """
    conn = duckdb.connect(":memory:")
    df = conn.execute(sql).fetchdf()
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour

    out: dict[pd.Timestamp, np.ndarray] = {}
    for target_date, grp in df.groupby("target_date"):
        target_ts = pd.Timestamp(target_date)
        s, e = daytime_utc_hours(target_ts)
        n = e - s
        day = grp[(grp["hour"] >= s) & (grp["hour"] < e)]
        if day.empty:
            continue
        # Pivot to (model, hour) and reindex to canonical model order.
        mat = (day.pivot_table(index="Model", columns="hour", values="precip", aggfunc="last")
                  .reindex(index=CANONICAL_MODELS, columns=list(range(s, e))))
        if mat.notna().any(axis=1).sum() < MIN_MODELS_FOR_DAY:
            continue
        out[target_ts] = mat.to_numpy(dtype="float64")
    return out


def longest_run(mask: np.ndarray) -> int:
    """Length of longest contiguous True run."""
    run, longest = 0, 0
    for v in mask:
        if v:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return longest


def compute_features(precip_mat: np.ndarray) -> dict[str, float]:
    """Compute 16 ensemble features from a (model × hour) precip matrix.
    NaN cells mean "this model didn't run for this hour"; they're tolerated
    (rolled up via nanmean-style aggregation per feature).
    """
    n_models, n_hours = precip_mat.shape
    wet_mat = (precip_mat >= WET_THRESHOLD_MM_H).astype(np.float64)
    # Mark missing model-hour pairs as NaN in wet_mat too so they don't
    # falsely contribute as "dry".
    wet_mat[np.isnan(precip_mat)] = np.nan

    # Per-member sequences — keep only members with full coverage for run-stats.
    member_dry_runs: list[int] = []      # longest dry block per member
    member_wet_runs: list[int] = []      # longest wet block per member
    member_argmin_q: list[int] = []      # hour index where each member is driest
    member_patterns: list[tuple] = []    # 9-bit wet/dry tuple per member
    member_has_dry: dict[int, list[int]] = {n: [] for n in WINDOWS}

    for m in range(n_models):
        row_precip = precip_mat[m]
        if np.isnan(row_precip).any():
            continue
        wet = (row_precip >= WET_THRESHOLD_MM_H)
        dry = ~wet
        dry_run = longest_run(dry)
        wet_run = longest_run(wet)
        member_dry_runs.append(dry_run)
        member_wet_runs.append(wet_run)
        member_argmin_q.append(int(np.argmin(row_precip)))
        member_patterns.append(tuple(wet.astype(np.int8).tolist()))
        for w in WINDOWS:
            member_has_dry[w].append(1 if dry_run >= w else 0)

    # Per-hour ensemble probability of wet (fraction of valid members calling wet).
    with np.errstate(invalid="ignore"):
        per_hour_p_wet = np.nanmean(wet_mat, axis=0)
    consensus_per_hour = np.maximum(per_hour_p_wet, 1.0 - per_hour_p_wet)
    # Shannon entropy of the per-hour wet/dry coin in nats (use clip to avoid log(0)).
    p = np.clip(per_hour_p_wet, 1e-9, 1 - 1e-9)
    entropy_per_hour = -p * np.log(p) - (1 - p) * np.log(1 - p)

    arr_dry_runs = np.array(member_dry_runs, dtype="float64") if member_dry_runs else np.array([np.nan])
    arr_wet_runs = np.array(member_wet_runs, dtype="float64") if member_wet_runs else np.array([np.nan])
    arr_blockiness = arr_dry_runs + arr_wet_runs
    arr_argmin_q = np.array(member_argmin_q, dtype="float64") if member_argmin_q else np.array([np.nan])

    return {
        "f01_frac_members_dry_3h": float(np.mean(member_has_dry[3])) if member_has_dry[3] else float("nan"),
        "f02_frac_members_dry_4h": float(np.mean(member_has_dry[4])) if member_has_dry[4] else float("nan"),
        "f03_frac_members_dry_6h": float(np.mean(member_has_dry[6])) if member_has_dry[6] else float("nan"),
        "f04_max_longest_dry_run":  float(np.nanmax(arr_dry_runs)),
        "f05_min_longest_dry_run":  float(np.nanmin(arr_dry_runs)),
        "f06_mean_longest_dry_run": float(np.nanmean(arr_dry_runs)),
        "f07_std_longest_dry_run":  float(np.nanstd(arr_dry_runs)) if arr_dry_runs.size > 1 else 0.0,
        "f08_mean_ensemble_q":      float(np.nanmean(per_hour_p_wet)),
        "f09_min_per_hour_q":       float(np.nanmin(per_hour_p_wet)),
        "f10_max_per_hour_q":       float(np.nanmax(per_hour_p_wet)),
        "f11_mean_hourly_consensus": float(np.nanmean(consensus_per_hour)),
        "f12_mean_hourly_entropy":  float(np.nanmean(entropy_per_hour)),
        "f13_mean_blockiness":      float(np.nanmean(arr_blockiness)),
        "f14_std_blockiness":       float(np.nanstd(arr_blockiness)) if arr_blockiness.size > 1 else 0.0,
        "f15_std_argmin_q_hour":    float(np.nanstd(arr_argmin_q)) if arr_argmin_q.size > 1 else 0.0,
        "f16_n_unique_patterns":    float(len(set(member_patterns))),
    }


def build_features() -> pd.DataFrame:
    rows: list[dict] = []
    for station in STATIONS:
        observed = load_observed_daytime(station, LEADS[0])  # any lead's labels — same observed sequence
        for lead in LEADS:
            per_nwp = load_per_nwp_daytime(lead)
            print(f"  {station} lead {lead}h: {len(per_nwp):,} forecast days × {len(observed):,} observed days, "
                  f"intersection {len(set(per_nwp) & set(observed)):,}")
            for date in sorted(set(per_nwp) & set(observed)):
                feats = compute_features(per_nwp[date])
                if any(pd.isna(v) for v in feats.values()):
                    continue
                obs_seq = observed[date]
                row = {
                    "station": station,
                    "lead": lead,
                    "target_date": date,
                    **feats,
                    "dry_3h": int(has_contiguous_dry_block(obs_seq, 3)),
                    "dry_4h": int(has_contiguous_dry_block(obs_seq, 4)),
                    "dry_6h": int(has_contiguous_dry_block(obs_seq, 6)),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def diagnostic(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c.startswith("f") and "_" in c]
    print("\n" + "=" * 110)
    print("Univariate signal diagnostic — per feature × per window")
    print("=" * 110)
    print(f"{'feature':<32} {'window':>8} {'r':>8} {'AUC':>8} {'brier_skill':>14}")
    print("-" * 110)
    for window_col, window_label in [("dry_3h", "3h"), ("dry_4h", "4h"), ("dry_6h", "6h")]:
        labels = df[window_col].to_numpy(dtype="float64")
        clim = labels.mean()
        clim_brier = float(np.mean((clim - labels) ** 2))
        print(f"-- window {window_label} (label rate {clim:.3f}, climatology Brier {clim_brier:.4f}) --")
        scored: list[tuple[str, float, float, float]] = []
        for col in feature_cols:
            x = df[col].to_numpy(dtype="float64")
            r = float(np.corrcoef(x, labels)[0, 1])
            try:
                auc = float(roc_auc_score(labels, x))
            except ValueError:
                auc = float("nan")
            # Single-feature Brier — min-max scale feature to [0,1], use as
            # probability proxy. Crude but a cheap "does this feature alone
            # carry signal" check.
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            if x_max > x_min:
                x01 = (x - x_min) / (x_max - x_min)
                # If correlation is negative, flip so higher feature = higher P(dry-window)
                if r < 0:
                    x01 = 1.0 - x01
                bs = float(np.mean((x01 - labels) ** 2))
                bss = (clim_brier - bs) / clim_brier
            else:
                bs = clim_brier
                bss = 0.0
            scored.append((col, r, auc, bss))
        for name, r, auc, bss in sorted(scored, key=lambda t: -abs(t[1])):
            print(f"  {name:<30} {window_label:>8} {r:>+7.3f} {auc:>8.3f} {bss:>+13.3f}")
        print()


def main() -> int:
    print(f"Phase 3p feature extractor → {OUT_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_features()
    print(f"\nTotal rows: {len(df):,}")
    print(f"  per station × lead breakdown:")
    print(df.groupby(["station", "lead"]).size().to_string())
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote → {OUT_PATH}")
    diagnostic(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
