"""Phase 3g hourly-source bake-off — does swapping 3g's MC marginals
from 3a's hourly P(wet) to 3e (or a stack of the two) improve dry-
window Brier on the held-out test slice?

3g is the parameter-free dry-window predictor that runs Monte Carlo
sampling over an hourly P(wet) sequence to estimate
P(∃ contiguous N-hour dry block in the day) for N ∈ {3, 4, 6}.
Currently sources its hourly marginals from Phase 3a (LightGBM lean,
27 features). Phase 3e (TorchSharp MLP) beat 3a at every hourly lead
in the 2026-05-12 stacking bake-off (mean Brier 0.0849 vs 0.0865),
so a better hourly source SHOULD propagate through the MC simulation
to give better dry-window probabilities.

This script tests three variants on the same data slice:
  3a-only   (baseline — current 3g source)
  3e-only   (best single hourly we have)
  mean-3a3e (synthesised 2-way stack at hourly resolution)

Inputs: each phase's test_predictions.parquet (canonical schema:
valid_time, station, lead, p_wet, observed_wet). Inner-join across
3a + 3e drops rows that aren't covered by both, then group by
(station, lead, target_date) and require all 24 hours to be present
(consistent with 3g's existing ExtractDaytimeQ "no gap" rule).

Truth label per (station, target_date, window): does the actual
24-hour observed_wet sequence contain a contiguous N-hour run of
zeros? Computed from the same parquet's observed_wet column —
no external rainfall query needed.

Predicted P(dry-window) per cell: MC sampling over the 24-hour
hourly q vector, mirroring DryWindow3gPredictor.ProbDryWindow's
algorithm in C# (n_samples=1000, default seed for reproducibility).

Output: per-(station, window, lead) Brier table for each variant
plus aggregate means; CSV to reports/3g_source_bakeoff_{ts}.csv.

Usage::

    python scripts/DryWindowStartHour/dry_window_3g_source_bakeoff.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_ROOT = ROOT / "data" / "models" / "precipitation"
DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72, 96, 120]
DEFAULT_WINDOWS = [3, 4, 6]
MC_SAMPLES = 1000
SEED = 42


@dataclass
class CellResult:
    station: str
    lead: int
    window: int
    n_days: int
    brier_3a_only: float
    brier_3e_only: float
    brier_mean_3a3e: float
    obs_wet_rate: float          # fraction of (day) cells with observed dry-window = True


# ----------------------------------------------------------------------
# Bundle discovery
# ----------------------------------------------------------------------

def find_latest_test_predictions(models_root: Path, station: str, phase: str) -> Path | None:
    """Find the newest local bundle for (station, phase) with a
    test_predictions.parquet. ``phase=None`` selects the unsuffixed
    3a champion convention."""
    station_dir = models_root / station
    if not station_dir.is_dir():
        return None
    suffix = f"phase{phase}" if phase else None
    candidates = []
    for d in station_dir.iterdir():
        if not d.is_dir():
            continue
        if suffix:
            if suffix not in d.name:
                continue
        else:
            if "phase" in d.name:
                continue
        if (d / "test_predictions.parquet").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name) / "test_predictions.parquet"


# ----------------------------------------------------------------------
# Dry-window primitives — mirror DryWindow3gPredictor.ProbDryWindow
# ----------------------------------------------------------------------

def has_contiguous_dry_block(binary: np.ndarray, window: int) -> bool:
    """True if the binary sequence (1 = wet, 0 = dry) contains a
    contiguous run of >= ``window`` zeros."""
    run = 0
    for v in binary:
        if v == 0:
            run += 1
            if run >= window:
                return True
        else:
            run = 0
    return False


def prob_dry_window(q: np.ndarray, window: int, n_samples: int, rng: np.random.Generator) -> float:
    """Monte Carlo P(∃ contiguous run of >= window dry hours) given
    the hourly P(wet) marginals q. Independent-hour sampling — same
    assumption DryWindow3gPredictor.ProbDryWindow makes."""
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
# Main
# ----------------------------------------------------------------------

def evaluate_station(
    station: str,
    df_3a: pd.DataFrame,
    df_3e: pd.DataFrame,
    leads: list[int],
    windows: list[int],
    rng: np.random.Generator,
) -> list[CellResult]:
    """For one station: inner-join 3a + 3e by (valid_time, lead), group
    by (lead, target_date), require 24 hours per group, run MC for
    each variant. Returns one CellResult per (station, lead, window)."""
    keep_cols = ["valid_time", "lead", "p_wet", "observed_wet"]
    df_3a = df_3a[keep_cols].rename(columns={"p_wet": "p_3a"})
    df_3e = df_3e[keep_cols].rename(columns={"p_wet": "p_3e"})

    # Inner-join: keep only rows where both phases predicted. Truth
    # labels (observed_wet) should agree — assert that to catch any
    # bundle-version drift.
    j = df_3a.merge(df_3e, on=["valid_time", "lead"], suffixes=("", "_e"))
    if not (j["observed_wet"] == j["observed_wet_e"]).all():
        raise RuntimeError(
            f"{station}: 3a + 3e disagree on observed_wet for "
            f"{(j['observed_wet'] != j['observed_wet_e']).sum()} rows — "
            f"different truth source between bundles?",
        )
    j = j.drop(columns=["observed_wet_e"])
    j["valid_time"] = pd.to_datetime(j["valid_time"], utc=True).dt.tz_localize(None)
    j["target_date"] = j["valid_time"].dt.date

    results: list[CellResult] = []
    for lead in leads:
        sub = j[j["lead"] == lead]
        if sub.empty:
            print(f"  {station} lead {lead}h: no joined rows")
            continue

        # Per-day groups; require all 24 hours of the UTC day present.
        groups = []
        for date, day_df in sub.groupby("target_date"):
            if len(day_df) != 24:
                continue
            day_df = day_df.sort_values("valid_time")
            groups.append((
                date,
                day_df["p_3a"].to_numpy(),
                day_df["p_3e"].to_numpy(),
                day_df["observed_wet"].astype(np.int32).to_numpy(),
            ))
        if not groups:
            print(f"  {station} lead {lead}h: no full-day cells (need 24 h each)")
            continue

        # MC predictions per day per variant + observed labels per window.
        for window in windows:
            preds_3a   = np.zeros(len(groups))
            preds_3e   = np.zeros(len(groups))
            preds_mean = np.zeros(len(groups))
            labels     = np.zeros(len(groups), dtype=np.int32)
            for i, (date, q3a, q3e, obs) in enumerate(groups):
                preds_3a[i]   = prob_dry_window(q3a, window, MC_SAMPLES, rng)
                preds_3e[i]   = prob_dry_window(q3e, window, MC_SAMPLES, rng)
                preds_mean[i] = prob_dry_window((q3a + q3e) / 2.0, window, MC_SAMPLES, rng)
                labels[i]     = 1 if has_contiguous_dry_block(obs, window) else 0

            results.append(CellResult(
                station=station, lead=lead, window=window,
                n_days=len(groups),
                brier_3a_only=brier(preds_3a, labels),
                brier_3e_only=brier(preds_3e, labels),
                brier_mean_3a3e=brier(preds_mean, labels),
                obs_wet_rate=float(labels.mean()),
            ))
    return results


def print_summary(results: list[CellResult]) -> None:
    print()
    print("=" * 110)
    print("Per-(station, window, lead) Brier — 3g hourly source bake-off")
    print("=" * 110)
    print(f"{'station':<28} {'window':>6} {'lead':>5} {'n_days':>7} "
          f"{'3a':>10} {'3e':>10} {'mean(3a,3e)':>12} {'best':>10}  base_rate")
    print("-" * 110)
    for r in sorted(results, key=lambda r: (r.station, r.window, r.lead)):
        best = min(r.brier_3a_only, r.brier_3e_only, r.brier_mean_3a3e)
        which = ("3a" if best == r.brier_3a_only
                 else "3e" if best == r.brier_3e_only
                 else "mean")
        print(f"{r.station:<28} {r.window:>4}h  {r.lead:>4}h {r.n_days:>7d} "
              f"{r.brier_3a_only:>10.4f} {r.brier_3e_only:>10.4f} {r.brier_mean_3a3e:>12.4f} "
              f"{which + '=' + f'{best:.4f}':>10}  {r.obs_wet_rate:.2f}")

    print()
    print("=" * 110)
    print("Aggregate mean Brier across all (station, window, lead) cells")
    print("=" * 110)
    a = np.mean([r.brier_3a_only   for r in results])
    e = np.mean([r.brier_3e_only   for r in results])
    m = np.mean([r.brier_mean_3a3e for r in results])
    print(f"  3a-only         {a:.4f}")
    print(f"  3e-only         {e:.4f}    ({100*(a-e)/a:+.1f}% vs 3a baseline)")
    print(f"  mean(3a, 3e)    {m:.4f}    ({100*(a-m)/a:+.1f}% vs 3a baseline)")

    print()
    print("=" * 110)
    print("Per-window aggregate (averaged across stations + leads)")
    print("=" * 110)
    for w in sorted({r.window for r in results}):
        subset = [r for r in results if r.window == w]
        a = np.mean([r.brier_3a_only for r in subset])
        e = np.mean([r.brier_3e_only for r in subset])
        m = np.mean([r.brier_mean_3a3e for r in subset])
        print(f"  window {w}h:  3a={a:.4f}  3e={e:.4f}  mean={m:.4f}  "
              f"(3e {100*(a-e)/a:+.1f}%, mean {100*(a-m)/a:+.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS),
                    help=f"Comma-separated station slugs (default: 3 Bonehill stations)")
    ap.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS),
                    help=f"Comma-separated leads (default: 24,48,72,96,120)")
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS),
                    help=f"Comma-separated dry-window lengths (default: 3,4,6)")
    ap.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s) for s in args.leads.split(",")]
    windows = [int(s) for s in args.windows.split(",")]
    models_root = Path(args.models_root)

    print(f"3g source bake-off — stations={stations}, leads={leads}, windows={windows}")
    print(f"  models_root: {models_root}")
    print(f"  MC samples per cell: {MC_SAMPLES}, seed: {SEED}\n")

    rng = np.random.default_rng(SEED)
    all_results: list[CellResult] = []
    for station in stations:
        p_3a = find_latest_test_predictions(models_root, station, phase=None)   # 3a unsuffixed
        p_3e = find_latest_test_predictions(models_root, station, phase="3e")
        if p_3a is None or p_3e is None:
            print(f"::warning::skipping {station} — missing 3a or 3e test_predictions")
            continue
        df_3a = pd.read_parquet(p_3a)
        df_3e = pd.read_parquet(p_3e)
        print(f"{station}: 3a={len(df_3a):,} rows ({p_3a.parent.name}), "
              f"3e={len(df_3e):,} rows ({p_3e.parent.name})")
        all_results.extend(evaluate_station(station, df_3a, df_3e, leads, windows, rng))

    if not all_results:
        print("::error::no cells produced a result")
        return 3

    print_summary(all_results)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"3g_source_bakeoff_{ts}.csv"
    rows = [
        {
            "station": r.station, "window": r.window, "lead": r.lead,
            "n_days": r.n_days, "obs_wet_rate": r.obs_wet_rate,
            "brier_3a_only": r.brier_3a_only,
            "brier_3e_only": r.brier_3e_only,
            "brier_mean_3a3e": r.brier_mean_3a3e,
        }
        for r in all_results
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote per-cell CSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
