"""Per-lead LightGBM meta-learner bake-off across precip phases.

Asks: can a LightGBM meta-learner over (NWP features + phase P(wet))
beat the equal-pool stacking baseline from the 2026-05-12 bake-off?

Two variants compared:
  A: NWP features + 4a P(wet) + 3e P(wet)
     "Honest baseline" — can LightGBM compete with 3c+4a+3e equal-pool
     by being smarter? Has to re-learn the LightGBM-over-NWPs signal.
  B: NWP features + 3c P(wet) + 4a P(wet) + 3e P(wet)
     "Everything available" — meta-learner gets all three component
     probabilities and just decides how much to lean on each.

Pooling: per-lead (5 LightGBM models, one per lead). Each model pools
across the 3 Bonehill stations for ~9k rows of training data — best
signal-to-overfit ratio at this dataset size. Per-cell was tried in
the original bake-off and overfit hard.

Leakage prevention: each phase's `test_predictions.parquet` carries
HELD-OUT predictions from training (out-of-fold by construction).
We inner-join all three, fetch the NWP features at the same
(valid_time, lead) using DuckDB, then chronologically split the
joined set 60% fit / 40% eval. The meta-learner never sees a
4a/3e/3c prediction that came from in-sample training rows.

Baselines from 2026-05-12 stack bake-off (same data slice):
  3c alone:        0.0870
  4a alone:        0.0848
  3e alone:        0.0844
  Equal-pool:      0.0830 (the target to beat)

Run::

    python scripts/StackBakeoff/lgbm_meta_bakeoff.py

Output: pretty-printed tables to stdout, full per-(variant, lead)
CSV to ``reports/lgbm_meta_bakeoff_{timestamp}.csv``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

# Repo root is two levels up from this script (scripts/StackBakeoff/).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "StackBakeoff"))
from stack_bakeoff import load_phase_predictions  # noqa: E402

DEFAULT_MODELS_ROOT = ROOT / "data" / "models" / "precipitation"
DEFAULT_FORECASTS_ROOT = ROOT / "data" / "forecasts"
DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72, 96, 120]
FIT_FRACTION = 0.60

# The 8 NWPs that have precip in the forecast tree at Bonehill.
NWP_MODELS = [
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless",
    "meteofrance_seamless", "ukmo_seamless", "gem_seamless",
    "ecmwf_aifs025_single", "jma_seamless",
]

# LightGBM hyperparameters — match 3c's production config so this is a
# clean "same trainer, different feature set" comparison rather than a
# benchmark search. Numbers cribbed from the 3c BlenderSpec defaults.
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}
N_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30

# Variant A: NWP features + 4a + 3e
# Variant B: NWP features + 3c + 4a + 3e
META_PHASES_A = ["4a", "3e"]
META_PHASES_B = ["3c", "4a", "3e"]
ALL_PHASES = ["3c", "4a", "3e"]   # always loaded; variant A just drops 3c


@dataclass
class VariantResult:
    variant: str
    lead: int
    n_fit: int
    n_eval: int
    feature_names: list[str]
    eval_brier: float
    component_brier: dict[str, float]      # eval Brier of each meta-phase alone
    equal_pool_brier: float                # eval Brier of equal-pool over the variant's meta-phases


# ----------------------------------------------------------------------
# NWP feature fetch
# ----------------------------------------------------------------------

def fetch_nwp_features(
    valid_times: pd.Series, lead: int, forecasts_root: Path,
) -> pd.DataFrame:
    """For each ValidTimeUtc in ``valid_times`` at the given ``lead``,
    pull the per-NWP precip rate from the latest RunTime that matches
    ``LeadHours == lead`` (so the forecast was issued exactly ``lead``
    hours before valid time — same lead-bucket pattern 3a/3c use at
    training). Pivots to wide so each row is one (valid_time, lead).

    Adds ensemble stats (mean / std / max / agreement_wet01) over the
    8 NWP columns. Missing NWPs (e.g. JMA pre-2026-04-28) come through
    as NaN; LightGBM handles them natively, no imputation needed.
    """
    glob = str(forecasts_root / "location=bonehill_rocks" / "**" / "*.parquet").replace("\\", "/")
    vt_min = valid_times.min()
    vt_max = valid_times.max()
    model_list = ", ".join(f"'{m}'" for m in NWP_MODELS)

    sql = f"""
    WITH latest AS (
      SELECT ValidTimeUtc, LeadHours, Model, Precipitation,
             ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours, Model
                                ORDER BY RunTimeUtc DESC) AS rn
      FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
      WHERE LocationName = 'bonehill_rocks'
        AND Model IN ({model_list})
        AND LeadHours = {lead}
        AND ValidTimeUtc >= timestamp '{vt_min}'
        AND ValidTimeUtc <= timestamp '{vt_max}'
    )
    SELECT ValidTimeUtc AS valid_time, Model, Precipitation
    FROM latest WHERE rn = 1
    """
    con = duckdb.connect(":memory:")
    long_df = con.execute(sql).fetch_df()
    if long_df.empty:
        return pd.DataFrame()

    # Pivot to wide: one row per valid_time, one column per model.
    wide = long_df.pivot_table(
        index="valid_time", columns="Model", values="Precipitation", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    rename_map = {m: f"precip_{m}" for m in NWP_MODELS}
    wide = wide.rename(columns=rename_map)

    # Some NWPs may have no rows in the window (JMA pre-2026-04-28,
    # AIFS pre-2026-04-27). Add as NaN columns so the feature matrix
    # has a stable shape across leads.
    for m in NWP_MODELS:
        col = f"precip_{m}"
        if col not in wide.columns:
            wide[col] = np.nan

    # Filter to only the valid_times we actually asked for (the SQL
    # range may include extras — e.g. weekend gaps in the test slice).
    wide = wide[wide["valid_time"].isin(valid_times)]

    # Ensemble stats over available NWPs per row (NaN-aware).
    precip_cols = [f"precip_{m}" for m in NWP_MODELS]
    wide["precip_mean"] = wide[precip_cols].mean(axis=1, skipna=True)
    wide["precip_std"]  = wide[precip_cols].std(axis=1, skipna=True)
    wide["precip_max"]  = wide[precip_cols].max(axis=1, skipna=True)
    # Agreement: fraction of non-null NWPs whose precip ≥ 0.1 mm/h (the
    # wet threshold the blender targets). NaN-ignoring division.
    wet_mask = wide[precip_cols] >= 0.1
    non_null_count = wide[precip_cols].notna().sum(axis=1).replace(0, np.nan)
    wide["precip_agreement_wet01"] = wet_mask.sum(axis=1) / non_null_count

    return wide.reset_index(drop=True)


# ----------------------------------------------------------------------
# Training + evaluation
# ----------------------------------------------------------------------

def brier(probs: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((probs - truth) ** 2))


def train_and_score_variant(
    joined: pd.DataFrame,
    variant_name: str,
    meta_phases: list[str],
    nwp_feature_cols: list[str],
    lead: int,
) -> VariantResult | None:
    """Train one LightGBM on the variant's feature set for a single
    lead, return Brier on the eval slice plus the component Briers."""
    cell = joined[joined["lead"] == lead].sort_values("valid_time").reset_index(drop=True)
    n = len(cell)
    if n < 200:
        print(f"  lead {lead}h: only {n} rows — skipping (need ≥200 for a credible split)")
        return None

    # Drop rows where any required feature is fully missing — usually
    # the NWP fetch dropped a row whose NWP data was completely absent.
    # NaN within a NWP column is fine (LightGBM handles missing); fully
    # missing rows would have NaN agreement_wet01 (no NWPs at all).
    cell = cell.dropna(subset=["precip_agreement_wet01"])
    n = len(cell)
    if n < 200:
        print(f"  lead {lead}h: only {n} usable rows after NaN drop — skipping")
        return None

    cut = int(n * FIT_FRACTION)
    fit = cell.iloc[:cut]
    ev  = cell.iloc[cut:]

    feature_cols = nwp_feature_cols + [f"p_wet_{p}" for p in meta_phases]
    X_fit = fit[feature_cols].astype("float64").to_numpy()
    y_fit = fit["observed_wet"].astype("int32").to_numpy()
    X_ev  = ev[feature_cols].astype("float64").to_numpy()
    y_ev  = ev["observed_wet"].astype("float64").to_numpy()

    # Use a tail of the fit slice as validation for early stopping —
    # 10% of fit, chronologically the most-recent within fit. Same
    # holdout-within-train pattern 3c uses at production train time.
    val_cut = int(len(X_fit) * 0.9)
    X_tr, y_tr = X_fit[:val_cut], y_fit[:val_cut]
    X_va, y_va = X_fit[val_cut:], y_fit[val_cut:]

    train_ds = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_ds   = lgb.Dataset(X_va, label=y_va, feature_name=feature_cols, reference=train_ds)
    booster = lgb.train(
        LGB_PARAMS, train_ds,
        num_boost_round=N_BOOST_ROUND,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    probs_ev = booster.predict(X_ev, num_iteration=booster.best_iteration)
    probs_ev = np.clip(probs_ev, 1e-6, 1 - 1e-6)
    variant_brier = brier(probs_ev, y_ev)

    # Component Briers + equal-pool baseline on the SAME eval slice so
    # the comparison is apples-to-apples.
    component_brier = {p: brier(ev[f"p_wet_{p}"].to_numpy(), y_ev) for p in meta_phases}
    equal = np.stack([ev[f"p_wet_{p}"].to_numpy() for p in meta_phases]).mean(axis=0)
    equal_brier = brier(equal, y_ev)

    return VariantResult(
        variant=variant_name, lead=lead,
        n_fit=cut, n_eval=n - cut,
        feature_names=feature_cols,
        eval_brier=variant_brier,
        component_brier=component_brier,
        equal_pool_brier=equal_brier,
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_summary(results: list[VariantResult]) -> None:
    by_variant: dict[str, list[VariantResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    print()
    print("=" * 110)
    print("Per-lead LightGBM meta-learner eval Brier (chronological 60/40 split, pooled across 3 Bonehill stations)")
    print("=" * 110)
    for variant, rs in by_variant.items():
        rs = sorted(rs, key=lambda r: r.lead)
        print(f"\n--- Variant {variant} ({rs[0].feature_names[-len(META_PHASES_A if variant=='A' else META_PHASES_B):]}) ---")
        header = f"{'lead':>5} {'n_fit':>7} {'n_eval':>7} {'lgbm Brier':>12} {'equal-pool':>12} {'lift vs eq':>12}"
        for p in (META_PHASES_A if variant == "A" else META_PHASES_B):
            header += f"  {('comp ' + p):>10}"
        print(header)
        for r in rs:
            lift = r.equal_pool_brier - r.eval_brier
            pct = 100 * lift / r.equal_pool_brier if r.equal_pool_brier > 0 else 0
            row = (f"{r.lead:>5d} {r.n_fit:>7d} {r.n_eval:>7d} "
                   f"{r.eval_brier:>12.4f} {r.equal_pool_brier:>12.4f} "
                   f"{lift:>+8.4f} ({pct:+.1f}%)")
            for p, b in r.component_brier.items():
                row += f"  {b:>10.4f}"
            print(row)
        mean_lgbm = np.mean([r.eval_brier for r in rs])
        mean_eq   = np.mean([r.equal_pool_brier for r in rs])
        lift_pct  = 100 * (mean_eq - mean_lgbm) / mean_eq
        print(f"      {'mean':>27} {mean_lgbm:>12.4f} {mean_eq:>12.4f}  "
              f"{(mean_eq - mean_lgbm):>+8.4f} ({lift_pct:+.1f}%)")

    print()
    print("=" * 110)
    print("Reference baselines from 2026-05-12 stack bake-off (same data slice):")
    print("  3c alone: 0.0870   4a alone: 0.0848   3e alone: 0.0844   equal-pool 3c+4a+3e: 0.0830")
    print("=" * 110)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS),
                    help="Comma-separated EA slugs (default: 3 Bonehill stations)")
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS),
                    help="Comma-separated leads (default: 24,48,72,96,120)")
    ap.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    ap.add_argument("--forecasts-root", default=str(DEFAULT_FORECASTS_ROOT))
    ap.add_argument("--variants", default="A,B",
                    help="Which variants to run: 'A', 'B', or 'A,B' (default both)")
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    models_root = Path(args.models_root)
    forecasts_root = Path(args.forecasts_root)
    variants = [v.strip() for v in args.variants.split(",")]

    print(f"LightGBM meta-learner bake-off")
    print(f"  variants: {variants}")
    print(f"  stations: {stations}")
    print(f"  leads:    {leads}")
    print(f"  models_root:    {models_root}")
    print(f"  forecasts_root: {forecasts_root}\n")

    # 1) Load each phase's test_predictions + inner-join on (valid_time, station, lead).
    print("Loading test_predictions for each phase:")
    frames = {}
    for phase in ALL_PHASES:
        df = load_phase_predictions(models_root, stations, phase)
        if df.empty:
            print(f"::error:: no test_predictions for phase {phase} — aborting")
            return 2
        frames[phase] = df

    wide = None
    for phase, df in frames.items():
        keep = df[["valid_time", "station", "lead", "p_wet", "observed_wet"]].copy()
        keep = keep.rename(columns={"p_wet": f"p_wet_{phase}"})
        if wide is None:
            wide = keep
        else:
            wide = wide.merge(keep.drop(columns=["observed_wet"]),
                              on=["valid_time", "station", "lead"], how="inner")
    if wide is None or wide.empty:
        print("::error:: inner-join across phases produced no aligned rows")
        return 3
    print(f"\nInner-join: {len(wide):,} aligned rows ({wide['station'].nunique()} stations × {wide['lead'].nunique()} leads).")

    # 2) For each lead, fetch NWP features once + join into the wide frame.
    print("\nFetching NWP features per lead:")
    nwp_per_lead: dict[int, pd.DataFrame] = {}
    for lead in leads:
        vt = wide.loc[wide["lead"] == lead, "valid_time"].drop_duplicates()
        if vt.empty:
            print(f"  lead {lead}h: no rows in joined set — skipping")
            continue
        nwp = fetch_nwp_features(vt, lead, forecasts_root)
        if nwp.empty:
            print(f"  lead {lead}h: no NWP rows fetched — skipping")
            continue
        nwp_per_lead[lead] = nwp
        print(f"  lead {lead}h: fetched {len(nwp):,} NWP feature rows")

    # 3) Combine NWP features with the joined predictions, per lead.
    print("\nTraining per-lead variants:")
    nwp_feature_cols = (
        [f"precip_{m}" for m in NWP_MODELS]
        + ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet01"]
    )
    results: list[VariantResult] = []
    for lead, nwp in nwp_per_lead.items():
        joined_lead = wide[wide["lead"] == lead].merge(nwp, on="valid_time", how="left")
        for variant in variants:
            meta_phases = META_PHASES_A if variant == "A" else META_PHASES_B
            print(f"  lead {lead}h, variant {variant} (meta={meta_phases}):", end=" ")
            r = train_and_score_variant(
                joined_lead, variant_name=variant,
                meta_phases=meta_phases,
                nwp_feature_cols=nwp_feature_cols,
                lead=lead,
            )
            if r is not None:
                results.append(r)
                lift = r.equal_pool_brier - r.eval_brier
                pct = 100 * lift / r.equal_pool_brier if r.equal_pool_brier > 0 else 0
                print(f"Brier {r.eval_brier:.4f} (eq-pool {r.equal_pool_brier:.4f}, lift {pct:+.1f}%)")

    if not results:
        print("::error:: no variant/lead pairs produced a result")
        return 4

    print_summary(results)

    # 4) CSV.
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"lgbm_meta_bakeoff_{ts}.csv"
    rows = []
    for r in results:
        row = {
            "variant": r.variant, "lead": r.lead,
            "n_fit": r.n_fit, "n_eval": r.n_eval,
            "lgbm_brier": r.eval_brier,
            "equal_pool_brier": r.equal_pool_brier,
        }
        for p, b in r.component_brier.items():
            row[f"comp_brier_{p}"] = b
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote per-(variant,lead) CSV -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
