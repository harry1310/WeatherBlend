"""Per-lead linear-pool + logit-pool stacking bake-off across precip phases.

Reads test_predictions.parquet from the latest local bundle of each
specified phase × station, inner-joins on (valid_time, station, lead),
splits chronologically (60% fit / 40% eval), fits per-lead optimal
weights, reports lift vs each component.

Usage::

    python scripts/StackBakeoff/stack_bakeoff.py \\
        --phases 3c,3e,4a \\
        --stations ea_bellever_dartmoor,ea_bovey_tracey,ea_dartmoor_nr_hexworthy

Defaults match the 2026-05-11/12 bake-off setup.

Output: pretty-printed tables to stdout; full per-(station,lead) CSV
to ``reports/stack_bakeoff_{timestamp}.csv``.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this script (scripts/StackBakeoff/).
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_ROOT = ROOT / "data" / "models" / "precipitation"
DEFAULT_PHASES = ["3c", "3e", "4a"]
DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72, 96, 120]
FIT_FRACTION = 0.60
TINY = 1e-9


@dataclass
class CellResult:
    station: str
    lead: int
    n_fit: int
    n_eval: int
    component_brier: dict[str, float]      # {phase: eval Brier}
    equal_brier: float
    linear_brier: float
    linear_weights: dict[str, float]
    logit_brier: float
    logit_weights: dict[str, float]


# ----------------------------------------------------------------------
# Bundle discovery + parquet loading
# ----------------------------------------------------------------------

def find_latest_bundle(models_root: Path, station: str, phase: str) -> Path | None:
    """Return the newest local bundle for (station, phase) that has a
    test_predictions.parquet file. Returns None if none found."""
    station_dir = models_root / station
    if not station_dir.exists():
        return None
    pattern = f"phase{phase}" if phase != "3a" else None
    candidates = []
    for d in station_dir.iterdir():
        if not d.is_dir():
            continue
        if pattern and pattern not in d.name:
            continue
        if phase == "3a" and "phase" in d.name:  # 3a champion has no phase suffix
            continue
        if (d / "test_predictions.parquet").exists():
            candidates.append(d)
    return max(candidates, default=None, key=lambda d: d.name)


def load_phase_predictions(models_root: Path, stations: list[str], phase: str) -> pd.DataFrame:
    """Load + concat test_predictions across stations for a single phase.
    Returns an empty DataFrame if no bundle is found for any station."""
    frames = []
    for station in stations:
        bundle = find_latest_bundle(models_root, station, phase)
        if bundle is None:
            print(f"  WARN: no {phase} bundle with test_predictions for {station}")
            continue
        df = pd.read_parquet(bundle / "test_predictions.parquet")
        # Normalise column names — every phase writes the same canonical
        # snake_case schema, but defensive in case.
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["bundle_version"] = bundle.name
        df["phase"] = phase
        frames.append(df)
        print(f"  {phase}/{station}: {len(df)} rows from {bundle.name}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Stacking primitives
# ----------------------------------------------------------------------

def brier(probs: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((probs - truth) ** 2))


def fit_linear_pool(probs_per_phase: dict[str, np.ndarray], truth: np.ndarray) -> tuple[dict[str, float], float]:
    """Fit weights w_i >= 0, sum(w_i) = 1 by minimising Brier on the
    given (probs, truth) slice. Tiny problem (3-5 phases × constraint)
    — we use a 0.05-step grid search to avoid SciPy dep + because grid
    on 3 phases is ~231 evaluations, sub-millisecond."""
    phases = list(probs_per_phase.keys())
    P = np.stack([probs_per_phase[p] for p in phases], axis=1)  # [n, n_phases]
    n_phases = P.shape[1]

    if n_phases == 1:
        return {phases[0]: 1.0}, brier(P[:, 0], truth)

    best = (None, float("inf"))
    # Generate weight tuples summing to 1.0 in 0.05 steps.
    step = 0.05
    grid_resolution = int(round(1.0 / step))
    for combo in _iter_weight_grid(n_phases, grid_resolution):
        w = np.array(combo, dtype="float64") / grid_resolution
        pooled = P @ w
        b = brier(pooled, truth)
        if b < best[1]:
            best = (w, b)
    weights = {phases[i]: float(best[0][i]) for i in range(n_phases)}
    return weights, float(best[1])


def fit_logit_pool(probs_per_phase: dict[str, np.ndarray], truth: np.ndarray) -> tuple[dict[str, float], float]:
    """Same constrained pool but on log-odds. Components are clipped to
    [TINY, 1-TINY] before logit transform to avoid ±inf at saturation."""
    phases = list(probs_per_phase.keys())
    logits_per_phase = {
        p: np.log(np.clip(probs_per_phase[p], TINY, 1 - TINY) /
                  np.clip(1 - probs_per_phase[p], TINY, 1 - TINY))
        for p in phases
    }
    L = np.stack([logits_per_phase[p] for p in phases], axis=1)
    n_phases = L.shape[1]
    if n_phases == 1:
        return {phases[0]: 1.0}, brier(probs_per_phase[phases[0]], truth)

    best = (None, float("inf"))
    step = 0.05
    grid_resolution = int(round(1.0 / step))
    for combo in _iter_weight_grid(n_phases, grid_resolution):
        w = np.array(combo, dtype="float64") / grid_resolution
        pooled_logits = L @ w
        pooled_probs = 1 / (1 + np.exp(-pooled_logits))
        b = brier(pooled_probs, truth)
        if b < best[1]:
            best = (w, b)
    weights = {phases[i]: float(best[0][i]) for i in range(n_phases)}
    return weights, float(best[1])


def _iter_weight_grid(n: int, resolution: int):
    """Yield integer tuples summing to `resolution` over n positions.
    Each tuple / resolution = a weight vector summing to 1."""
    if n == 1:
        yield (resolution,)
        return
    for first in range(resolution + 1):
        for rest in _iter_weight_grid(n - 1, resolution - first):
            yield (first,) + rest


def apply_linear(weights: dict[str, float], probs_per_phase: dict[str, np.ndarray]) -> np.ndarray:
    P = np.stack([probs_per_phase[p] for p in weights.keys()], axis=1)
    w = np.array(list(weights.values()), dtype="float64")
    return P @ w


def apply_logit(weights: dict[str, float], probs_per_phase: dict[str, np.ndarray]) -> np.ndarray:
    L = np.stack([
        np.log(np.clip(probs_per_phase[p], TINY, 1 - TINY) /
               np.clip(1 - probs_per_phase[p], TINY, 1 - TINY))
        for p in weights.keys()
    ], axis=1)
    w = np.array(list(weights.values()), dtype="float64")
    pooled_logits = L @ w
    return 1 / (1 + np.exp(-pooled_logits))


# ----------------------------------------------------------------------
# Per-cell evaluation
# ----------------------------------------------------------------------

def evaluate_cell(joined: pd.DataFrame, station: str, lead: int, phases: list[str]) -> CellResult | None:
    """Per-(station, lead) fit + eval. joined has columns
    valid_time, station, lead, observed_wet, p_wet_<phase>."""
    cell = joined[(joined["station"] == station) & (joined["lead"] == lead)]
    cell = cell.sort_values("valid_time").reset_index(drop=True)
    n = len(cell)
    if n < 50:
        return None
    cut = int(n * FIT_FRACTION)
    fit = cell.iloc[:cut]
    ev  = cell.iloc[cut:]

    truth_fit = fit["observed_wet"].astype("float64").to_numpy()
    truth_ev  = ev["observed_wet"].astype("float64").to_numpy()
    probs_fit = {p: fit[f"p_wet_{p}"].to_numpy() for p in phases}
    probs_ev  = {p: ev[f"p_wet_{p}"].to_numpy()  for p in phases}

    component_brier = {p: brier(probs_ev[p], truth_ev) for p in phases}
    equal = np.stack([probs_ev[p] for p in phases]).mean(axis=0)
    equal_brier = brier(equal, truth_ev)

    lin_w, _ = fit_linear_pool(probs_fit, truth_fit)
    lin_eval = apply_linear(lin_w, probs_ev)
    lin_brier = brier(lin_eval, truth_ev)

    log_w, _ = fit_logit_pool(probs_fit, truth_fit)
    log_eval = apply_logit(log_w, probs_ev)
    log_brier = brier(log_eval, truth_ev)

    return CellResult(
        station=station, lead=lead, n_fit=cut, n_eval=n - cut,
        component_brier=component_brier,
        equal_brier=equal_brier,
        linear_brier=lin_brier, linear_weights=lin_w,
        logit_brier=log_brier,  logit_weights=log_w,
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def fmt_w(w: dict[str, float]) -> str:
    return " ".join(f"{p}:{v:.2f}" for p, v in w.items())


def print_summary(cells: list[CellResult], phases: list[str]) -> None:
    print()
    print("=" * 100)
    print("Per-cell stacking results (eval Brier on held-out 40% slice)")
    print("=" * 100)
    print(f"{'station':<28} {'lead':>5} {'n_fit':>6} {'n_ev':>5}  "
          + " ".join(f"{p:>8}" for p in phases)
          + f"  {'equal':>8}  {'linear':>8}  {'logit':>8}  {'best stack lift':>20}")
    print("-" * 130)
    for c in cells:
        comp_str = " ".join(f"{c.component_brier[p]:>8.4f}" for p in phases)
        best_component = min(c.component_brier.values())
        best_stack = min(c.equal_brier, c.linear_brier, c.logit_brier)
        lift = best_component - best_stack
        lift_pct = 100 * lift / best_component if best_component > 0 else 0
        lift_str = f"{lift:+.4f} ({lift_pct:+.1f}%)"
        print(f"{c.station:<28} {c.lead:>5d} {c.n_fit:>6d} {c.n_eval:>5d}  {comp_str}"
              f"  {c.equal_brier:>8.4f}  {c.linear_brier:>8.4f}  {c.logit_brier:>8.4f}  {lift_str:>20}")

    print()
    print("=" * 100)
    print("Linear-pool weights (per cell, fit on first 60%)")
    print("=" * 100)
    for c in cells:
        print(f"{c.station:<28} +{c.lead:>3}h  {fmt_w(c.linear_weights)}")

    print()
    print("=" * 100)
    print("Aggregate (mean Brier across all cells)")
    print("=" * 100)
    for p in phases:
        avg = np.mean([c.component_brier[p] for c in cells])
        print(f"  {p:<8} mean Brier = {avg:.4f}")
    eq = np.mean([c.equal_brier for c in cells])
    li = np.mean([c.linear_brier for c in cells])
    lo = np.mean([c.logit_brier for c in cells])
    print(f"  {'equal':<8} mean Brier = {eq:.4f}")
    print(f"  {'linear':<8} mean Brier = {li:.4f}")
    print(f"  {'logit':<8} mean Brier = {lo:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--phases", default=",".join(DEFAULT_PHASES),
                   help=f"Comma-separated list (default: {','.join(DEFAULT_PHASES)})")
    p.add_argument("--stations", default=",".join(DEFAULT_STATIONS),
                   help=f"Comma-separated EA slugs (default: {','.join(DEFAULT_STATIONS)})")
    p.add_argument("--leads", default=",".join(str(L) for L in DEFAULT_LEADS),
                   help=f"Comma-separated leads (default: {','.join(str(L) for L in DEFAULT_LEADS)})")
    p.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT),
                   help="Path to data/models/precipitation")
    args = p.parse_args()

    phases = [s.strip() for s in args.phases.split(",")]
    stations = [s.strip() for s in args.stations.split(",")]
    leads = [int(s.strip()) for s in args.leads.split(",")]
    models_root = Path(args.models_root)

    print(f"Stack bake-off: phases={phases}, stations={stations}, leads={leads}")
    print(f"Models root: {models_root}\n")

    print("Loading per-phase test_predictions:")
    frames = []
    for phase in phases:
        df = load_phase_predictions(models_root, stations, phase)
        if df.empty:
            print(f"::error:: No test_predictions found for phase {phase} — aborting")
            return 2
        frames.append(df.assign(phase=phase))

    # Wide-pivot so each row is one (valid_time, station, lead) and
    # columns are p_wet_<phase>. Inner-join on the keys ensures every
    # row has a prediction from every phase (drops un-aligned rows).
    wide = None
    for phase, df in zip(phases, frames):
        keep = df[["valid_time", "station", "lead", "p_wet", "observed_wet"]].copy()
        keep = keep.rename(columns={"p_wet": f"p_wet_{phase}"})
        if wide is None:
            wide = keep
        else:
            # Drop observed_wet from the right side (already in left); keep p_wet_<phase>.
            wide = wide.merge(keep.drop(columns=["observed_wet"]),
                              on=["valid_time", "station", "lead"], how="inner")
    if wide is None or wide.empty:
        print("::error:: Inner-join produced no aligned rows. Check test_predictions schemas.")
        return 3
    print(f"\nInner-joined: {len(wide):,} rows across {wide['station'].nunique()} stations × "
          f"{wide['lead'].nunique()} leads × {len(phases)} phases.")

    cells = []
    for station in stations:
        for lead in leads:
            res = evaluate_cell(wide, station, lead, phases)
            if res is not None:
                cells.append(res)
            else:
                print(f"  skipped {station} lead {lead}h: <50 aligned rows")

    if not cells:
        print("::error:: No cells with enough rows to evaluate.")
        return 4

    print_summary(cells, phases)

    # Persist a CSV.
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"stack_bakeoff_{ts}.csv"
    rows = []
    for c in cells:
        row = {"station": c.station, "lead": c.lead, "n_fit": c.n_fit, "n_eval": c.n_eval,
               "equal_brier": c.equal_brier,
               "linear_brier": c.linear_brier,
               "logit_brier": c.logit_brier}
        for p, b in c.component_brier.items():
            row[f"brier_{p}"] = b
        for p, w in c.linear_weights.items():
            row[f"linear_w_{p}"] = w
        for p, w in c.logit_weights.items():
            row[f"logit_w_{p}"] = w
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    # ASCII arrow — Windows cp1252 console can't encode the unicode →.
    print(f"\nWrote per-cell CSV -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
