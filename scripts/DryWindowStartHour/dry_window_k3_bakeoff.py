"""K=2 vs K=3 bake-off for 3n.

For each (station, window, lead): inner-join the K=2 raw 3n test_predictions
with the K=3 raw 3n test_predictions on (target_date, lead). Same test slice
from DryWindowDataset.Split (date-based). Compute Brier on identical rows.

Also pulls in 3g raw and 3j raw bundles for context (apples-to-apples
matched to whichever overlap exists).
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


def latest_raw_bundle(station: str, window: int, phase_token: str) -> Path | None:
    d = DRY_WINDOW_ROOT / station / f"window_{window}h"
    if not d.exists():
        return None
    cands = sorted([p for p in d.iterdir()
                    if p.is_dir()
                    and p.name.endswith(phase_token)
                    and not is_pav_bundle(p)
                    and (p / "test_predictions.parquet").exists()])
    return cands[-1] if cands else None


def load_predictions(bundle: Path) -> pd.DataFrame:
    df = pd.read_parquet(bundle / "test_predictions.parquet")
    df["target_date"] = pd.to_datetime(df["target_date"], utc=True).dt.tz_localize(None)
    return df[["target_date", "lead", "p_dry_window", "observed_dry_window"]]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main() -> int:
    rows = []
    for s in STATIONS:
        for w in WINDOWS:
            phase_bundles = {}
            for tag, token in [("3g", "phase3g"),
                               ("3j", "phase3j"),
                               ("3n_k2", "phase3n"),
                               ("3n_k3", "phase3n_k3")]:
                b = latest_raw_bundle(s, w, token)
                if b is None:
                    print(f"MISSING {tag} bundle for {s}/w{w}h")
                    continue
                phase_bundles[tag] = b

            if "3n_k2" not in phase_bundles or "3n_k3" not in phase_bundles:
                continue

            # Inner-join all four on (target_date, lead) for honest comparison
            dfs = {tag: load_predictions(b).rename(
                columns={"p_dry_window": f"p_{tag}"})
                   for tag, b in phase_bundles.items()}
            base = dfs["3n_k2"]
            for tag, df in dfs.items():
                if tag == "3n_k2":
                    continue
                base = base.merge(df[["target_date", "lead", f"p_{tag}"]],
                                  on=["target_date", "lead"], how="inner")

            for lead in LEADS:
                sub = base[base["lead"] == lead]
                if len(sub) < 10:
                    continue
                y = sub["observed_dry_window"].to_numpy(dtype="float64")
                row = {"station": s.replace("ea_", ""),
                       "window": w, "lead": lead, "n": len(sub),
                       "obs": float(y.mean())}
                for tag in phase_bundles:
                    row[f"brier_{tag}"] = brier(sub[f"p_{tag}"].to_numpy(), y)
                rows.append(row)

    if not rows:
        print("No rows.")
        return 1
    df = pd.DataFrame(rows)
    df["delta_k3_minus_k2"] = df["brier_3n_k3"] - df["brier_3n_k2"]

    print()
    print("=" * 130)
    print("PER-CELL: 3n K=3 vs K=2 (raw), with 3g/3j context (all inner-joined on matched test rows)")
    print("=" * 130)
    print(f"{'station':<23} {'win':<4} {'lead':<5} {'n':<5} {'obs':<5} {'3g':>8} {'3j':>8} {'3n_K2':>8} {'3n_K3':>8} {'Δ(K3-K2)':>10} verdict")
    print("-" * 130)
    for _, r in df.iterrows():
        v = ("K=3 helps ★" if r.delta_k3_minus_k2 < -0.005
             else "K=3 helps" if r.delta_k3_minus_k2 < -0.0005
             else "K=3 hurts ★" if r.delta_k3_minus_k2 > 0.005
             else "K=3 hurts" if r.delta_k3_minus_k2 > 0.0005
             else "tied")
        g = f"{r.brier_3g:.4f}" if "brier_3g" in df.columns and not pd.isna(r.get("brier_3g", np.nan)) else "—"
        j = f"{r.brier_3j:.4f}" if "brier_3j" in df.columns and not pd.isna(r.get("brier_3j", np.nan)) else "—"
        print(f"{r.station:<23} {r.window}h   {int(r.lead):<5} {int(r.n):<5} {r.obs:.2f}  "
              f"{g:>8} {j:>8} {r.brier_3n_k2:>8.4f} {r.brier_3n_k3:>8.4f} {r.delta_k3_minus_k2:>+10.4f} {v}")

    # Aggregate across all 27 cells
    print()
    print("=" * 80)
    print("AGGREGATE (mean Brier across all matched cells, lower = better)")
    print("=" * 80)
    n_cells = len(df)
    print(f"  cells matched: {n_cells}")
    for col, name in [("brier_3g", "3g"), ("brier_3j", "3j"),
                      ("brier_3n_k2", "3n (K=2)"), ("brier_3n_k3", "3n (K=3)")]:
        if col not in df.columns:
            continue
        m = df[col].mean()
        print(f"  {name:<12} {m:.4f}")
    helps = (df["delta_k3_minus_k2"] < -0.0005).sum()
    hurts = (df["delta_k3_minus_k2"] > 0.0005).sum()
    ties = n_cells - helps - hurts
    k2 = df["brier_3n_k2"].mean()
    k3 = df["brier_3n_k3"].mean()
    rel = 100 * (k3 - k2) / k2
    print(f"  K=3 vs K=2: helps {helps} | hurts {hurts} | tied {ties} | delta {k3-k2:+.4f} ({rel:+.2f}%)")

    # Per-window
    print()
    print("PER-WINDOW")
    for w in WINDOWS:
        sub = df[df["window"] == w]
        if sub.empty:
            continue
        print(f"  {w}h ({len(sub)} cells): 3g={sub.brier_3g.mean():.4f}  3j={sub.brier_3j.mean():.4f}  "
              f"K=2={sub.brier_3n_k2.mean():.4f}  K=3={sub.brier_3n_k3.mean():.4f}  Δ={sub.delta_k3_minus_k2.mean():+.4f}")

    # Per-station
    print()
    print("PER-STATION")
    for s in df["station"].unique():
        sub = df[df["station"] == s]
        print(f"  {s:<25}: K=2={sub.brier_3n_k2.mean():.4f}  K=3={sub.brier_3n_k3.mean():.4f}  Δ={sub.delta_k3_minus_k2.mean():+.4f}")

    out = ROOT / "reports" / "k3_bakeoff_2026-05-14.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
