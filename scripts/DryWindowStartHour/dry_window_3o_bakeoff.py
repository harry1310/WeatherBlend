"""3o (Markov run-length) bake-off vs 3g/3j/3n raw bundles.

Inner-joins test_predictions across all four phases per (station, window,
lead) and computes Brier on matched rows.
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


def is_pav(d: Path) -> bool:
    return (d / "calibrator_24h.json").exists()


def latest_raw_bundle(station: str, window: int, phase_token: str) -> Path | None:
    d = DRY_WINDOW_ROOT / station / f"window_{window}h"
    if not d.exists():
        return None
    cands = sorted([p for p in d.iterdir()
                    if p.is_dir() and p.name.endswith(phase_token)
                    and not is_pav(p)
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
            bundles = {}
            for tag, token in [("3g", "phase3g"),
                               ("3j", "phase3j"),
                               ("3n", "phase3n"),
                               ("3o", "phase3o")]:
                b = latest_raw_bundle(s, w, token)
                if b is None:
                    print(f"MISSING {tag} for {s}/w{w}h")
                    continue
                bundles[tag] = b

            if "3o" not in bundles:
                continue

            dfs = {tag: load_predictions(b).rename(
                columns={"p_dry_window": f"p_{tag}"})
                   for tag, b in bundles.items()}
            base = dfs["3o"]
            for tag, df in dfs.items():
                if tag == "3o":
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
                for tag in bundles:
                    row[f"brier_{tag}"] = brier(sub[f"p_{tag}"].to_numpy(), y)
                rows.append(row)

    if not rows:
        print("No rows.")
        return 1
    df = pd.DataFrame(rows)

    print()
    print("=" * 120)
    print("PER-CELL Brier on inner-joined matched test rows")
    print("=" * 120)
    print(f"{'station':<23} {'win':<4} {'lead':<5} {'n':<5} {'obs':<5} {'3g':>8} {'3j':>8} {'3n':>8} {'3o':>8}  best")
    print("-" * 120)
    for _, r in df.iterrows():
        cols = [("3g", r.get("brier_3g")), ("3j", r.get("brier_3j")),
                ("3n", r.get("brier_3n")), ("3o", r.get("brier_3o"))]
        cols = [(t, v) for t, v in cols if v is not None and not pd.isna(v)]
        best = min(cols, key=lambda x: x[1])
        print(f"{r.station:<23} {r.window}h   {int(r.lead):<5} {int(r.n):<5} {r.obs:.2f}  "
              f"{r.get('brier_3g', float('nan')):>8.4f} {r.get('brier_3j', float('nan')):>8.4f} "
              f"{r.get('brier_3n', float('nan')):>8.4f} {r.brier_3o:>8.4f}  {best[0]}")

    print()
    print("=" * 80)
    print("AGGREGATE — mean Brier (lower = better)")
    print("=" * 80)
    for col, name in [("brier_3g", "3g"), ("brier_3j", "3j"),
                      ("brier_3n", "3n"), ("brier_3o", "3o")]:
        if col not in df.columns:
            continue
        m = df[col].mean()
        n = df[col].notna().sum()
        print(f"  {name}: {m:.4f} ({n} cells)")
    g, o = df["brier_3g"].mean(), df["brier_3o"].mean()
    print(f"  3o vs 3g: {o-g:+.4f} ({100*(o-g)/g:+.2f}%)")

    print()
    print("PER-WINDOW")
    for w in WINDOWS:
        sub = df[df["window"] == w]
        if sub.empty:
            continue
        print(f"  {w}h: 3g={sub.brier_3g.mean():.4f}  3j={sub.brier_3j.mean():.4f}  "
              f"3n={sub.brier_3n.mean():.4f}  3o={sub.brier_3o.mean():.4f}  "
              f"3o-3g={sub.brier_3o.mean() - sub.brier_3g.mean():+.4f}")

    print()
    print("PER-STATION")
    for s in df["station"].unique():
        sub = df[df["station"] == s]
        print(f"  {s:<25}: 3g={sub.brier_3g.mean():.4f}  3o={sub.brier_3o.mean():.4f}  "
              f"Δ={sub.brier_3o.mean() - sub.brier_3g.mean():+.4f}")

    print()
    print("WINS BY PHASE per window")
    for w in WINDOWS:
        sub = df[df["window"] == w]
        wins = {tag: 0 for tag in ["3g", "3j", "3n", "3o"]}
        for _, r in sub.iterrows():
            cells = [(t, r.get(f"brier_{t}")) for t in wins.keys()
                     if r.get(f"brier_{t}") is not None and not pd.isna(r.get(f"brier_{t}"))]
            best = min(cells, key=lambda x: x[1])
            wins[best[0]] += 1
        print(f"  {w}h: " + ", ".join(f"{k}={v}" for k, v in wins.items()))

    out = ROOT / "reports" / "3o_bakeoff_2026-05-14.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
