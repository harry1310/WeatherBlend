"""PAV vs raw A/B comparison for 3g/3j/3n.

For each (phase, station, window, lead) cell where we have BOTH a raw
C# bundle and a PAV C# bundle, compute Brier on the bundle's
test_predictions.parquet (which is the matched test slice from
DryWindowDataset.Split). Same fitted Σ (or no-Σ for 3g), same test
rows — the ONLY difference between the two bundles is whether PAV
calibration was applied to the test predictions.

Output: per-cell delta table, per-phase aggregate, per-window
aggregate, summary verdict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Projects\Weather\WeatherBlend")
DRY_WINDOW_ROOT = ROOT / "data" / "models" / "dry_window"
STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
WINDOWS = [3, 4, 6]
LEADS = [24, 48, 72]


def is_pav_bundle(d: Path) -> bool:
    return (d / "calibrator_24h.json").exists()


def list_phase_bundles(station: str, window: int, phase: str) -> dict[str, list[Path]]:
    """Returns {'raw': [...], 'pav': [...]} sorted by name."""
    d = DRY_WINDOW_ROOT / station / f"window_{window}h"
    out: dict[str, list[Path]] = {"raw": [], "pav": []}
    if not d.exists():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir() or phase not in sub.name:
            continue
        if not (sub / "test_predictions.parquet").exists():
            continue
        if is_pav_bundle(sub):
            out["pav"].append(sub)
        else:
            out["raw"].append(sub)
    return out


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def find_5b_bundle(station: str, window: int) -> Path | None:
    """5b only has one bundle per cell (no raw/PAV split — PAV is baked in
    via the val-fit isotonic step at training time)."""
    d = DRY_WINDOW_ROOT / station / f"window_{window}h"
    if not d.exists():
        return None
    bundles = sorted([p for p in d.iterdir() if "phase5b" in p.name and p.is_dir()
                      and (p / "test_predictions.parquet").exists()])
    return bundles[-1] if bundles else None


def load_5b_briers(station: str, window: int) -> dict[int, tuple[int, float]]:
    """Return {lead: (n, brier)} for the 5b bundle of this (station, window)."""
    b = find_5b_bundle(station, window)
    if b is None:
        return {}
    df = pd.read_parquet(b / "test_predictions.parquet")
    out: dict[int, tuple[int, float]] = {}
    for lead in LEADS:
        sub = df[df["lead"] == lead]
        if len(sub) < 10:
            continue
        y = sub["observed_dry_window"].to_numpy(dtype="float64")
        out[lead] = (len(sub), brier(sub["p_dry_window"].to_numpy(), y))
    return out


def main() -> int:
    rows: list[dict] = []
    missing_pairs: list[str] = []

    # Pre-load 5b briers for context columns.
    fivem_b_by_key: dict[tuple[str, int, int], tuple[int, float]] = {}
    for s in STATIONS:
        for w in WINDOWS:
            for lead, (n, br) in load_5b_briers(s, w).items():
                fivem_b_by_key[(s.replace("ea_", ""), w, lead)] = (n, br)

    for phase in ["phase3g", "phase3j", "phase3n"]:
        for s in STATIONS:
            for w in WINDOWS:
                bundles = list_phase_bundles(s, w, phase)
                if not bundles["raw"] or not bundles["pav"]:
                    missing_pairs.append(f"{phase} {s}/w{w}h: raw={len(bundles['raw'])} pav={len(bundles['pav'])}")
                    continue
                raw_bundle = bundles["raw"][-1]   # newest raw
                pav_bundle = bundles["pav"][-1]   # newest pav

                df_raw = pd.read_parquet(raw_bundle / "test_predictions.parquet")
                df_pav = pd.read_parquet(pav_bundle / "test_predictions.parquet")

                merged = (df_raw.rename(columns={"p_dry_window": "p_raw"})
                          .merge(df_pav.rename(columns={"p_dry_window": "p_pav"})[
                              ["target_date", "lead", "p_pav"]],
                              on=["target_date", "lead"], how="inner"))
                for lead in LEADS:
                    sub = merged[merged["lead"] == lead]
                    if len(sub) < 10:
                        continue
                    y_col = "observed_dry_window_x" if "observed_dry_window_x" in sub.columns else "observed_dry_window"
                    y = sub[y_col].to_numpy(dtype="float64")
                    br = brier(sub["p_raw"].to_numpy(), y)
                    bp = brier(sub["p_pav"].to_numpy(), y)
                    key = (s.replace("ea_", ""), w, lead)
                    n5, b5 = fivem_b_by_key.get(key, (0, float("nan")))
                    rows.append({
                        "phase": phase.replace("phase", ""),
                        "station": s.replace("ea_", ""),
                        "window": w,
                        "lead": lead,
                        "n": len(sub),
                        "brier_raw": br,
                        "brier_pav": bp,
                        "delta_pav_minus_raw": bp - br,
                        "brier_5b": b5,        # context (same test cells if 5b's test slice matches)
                        "n_5b": n5,
                    })

    if missing_pairs:
        print("MISSING raw/pav pairs (skipped):")
        for m in missing_pairs:
            print(f"  {m}")
        print()

    if not rows:
        print("No cells with both raw and PAV bundles found.")
        return 1

    df = pd.DataFrame(rows)

    # Per-cell table
    print("=" * 142)
    print("PER-CELL: raw vs PAV Brier on identical matched test rows (5b context — different test slice possible)")
    print("=" * 142)
    print(f"{'phase':<7} {'station':<25} {'win':<4} {'lead':<5} {'n':<5} {'brier_raw':>10} {'brier_pav':>10} {'delta':>9} {'verdict':<12} {'5b':>9}")
    print("-" * 142)
    for _, r in df.iterrows():
        verdict = ("PAV helps ★" if r.delta_pav_minus_raw < -0.005
                   else "PAV helps" if r.delta_pav_minus_raw < -0.0005
                   else "PAV hurts ★" if r.delta_pav_minus_raw > 0.005
                   else "PAV hurts" if r.delta_pav_minus_raw > 0.0005
                   else "tied")
        b5_str = f"{r.brier_5b:.4f}" if not np.isnan(r.brier_5b) else "—"
        print(f"{r.phase:<7} {r.station:<25} {r.window}h   {r.lead:<5} {r.n:<5} "
              f"{r.brier_raw:>10.4f} {r.brier_pav:>10.4f} {r.delta_pav_minus_raw:>+9.4f} {verdict:<12} {b5_str:>9}")

    # Per-phase aggregate
    print()
    print("=" * 100)
    print("PER-PHASE aggregate (mean across all cells, with 5b context)")
    print("=" * 100)
    for phase in df["phase"].unique():
        sub = df[df["phase"] == phase]
        n_help = (sub["delta_pav_minus_raw"] < -0.0005).sum()
        n_hurt = (sub["delta_pav_minus_raw"] > 0.0005).sum()
        n_tied = len(sub) - n_help - n_hurt
        mean_raw = sub["brier_raw"].mean()
        mean_pav = sub["brier_pav"].mean()
        rel = 100 * (mean_pav - mean_raw) / mean_raw
        b5_mean = sub["brier_5b"].dropna().mean() if sub["brier_5b"].notna().any() else float("nan")
        b5_str = f"5b {b5_mean:.4f}" if not np.isnan(b5_mean) else "5b —"
        print(f"  {phase}: {len(sub)} cells | help {n_help}, hurt {n_hurt}, tied {n_tied} | "
              f"raw {mean_raw:.4f} -> PAV {mean_pav:.4f} ({rel:+.1f}%) | {b5_str}")

    # Per-window per-phase
    print()
    print("=" * 96)
    print("PER-WINDOW × PHASE (mean Brier across stations + leads, with 5b context)")
    print("=" * 96)
    print(f"{'phase':<7} {'win':<4} {'cells':<6} {'raw':>9} {'pav':>9} {'delta':>9} {'5b':>9}")
    for phase in df["phase"].unique():
        for w in WINDOWS:
            sub = df[(df["phase"] == phase) & (df["window"] == w)]
            if sub.empty:
                continue
            b5_mean = sub["brier_5b"].dropna().mean() if sub["brier_5b"].notna().any() else float("nan")
            b5_str = f"{b5_mean:.4f}" if not np.isnan(b5_mean) else "—"
            print(f"  {phase:<5} {w}h    {len(sub):<6} {sub.brier_raw.mean():>9.4f} {sub.brier_pav.mean():>9.4f} "
                  f"{sub.brier_pav.mean() - sub.brier_raw.mean():>+9.4f} {b5_str:>9}")

    # Per-station per-phase
    print()
    print("=" * 80)
    print("PER-STATION × PHASE (mean Brier across windows + leads)")
    print("=" * 80)
    print(f"{'phase':<7} {'station':<25} {'cells':<6} {'raw':>9} {'pav':>9} {'delta':>9}")
    for phase in df["phase"].unique():
        for s in [st.replace("ea_", "") for st in STATIONS]:
            sub = df[(df["phase"] == phase) & (df["station"] == s)]
            if sub.empty:
                continue
            print(f"  {phase:<5} {s:<25} {len(sub):<6} {sub.brier_raw.mean():>9.4f} {sub.brier_pav.mean():>9.4f} "
                  f"{sub.brier_pav.mean() - sub.brier_raw.mean():>+9.4f}")

    # Write CSV
    out = ROOT / "reports" / "pav_ab_2026-05-14.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote per-cell rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
