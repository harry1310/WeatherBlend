"""Per-(station, lead) linear-pool bake-off: lead-pooled 4a vs per-cell BART.

Same model class (BART), same hyperparameters (NTREE=500, K=3, K3=200,
NDPOST=1000), same features (22-feat 3a base + 3 synoptic). The ONLY
difference is the architectural scope:

  - Lead-pooled 4a: one BART per station, trained on ~70k rows pooled
    across all 5 leads (24/48/72/96/120). Learns cross-lead structure
    via the `lead` feature column.
  - Per-cell BART: 9 separate BARTs (3 stations × 3 leads), each
    trained on ~14k rows from ONE (station, lead). Specialises hard
    per cell, no cross-lead exchange.

Question: does the lead-pooled 4a's cross-lead structure encode
ANYTHING that per-cell BART misses? If w fits ≈ 0, per-cell dominates
and lead-pooling has no extra info. If w fits > 0.1, lead-pooling
captures real residual signal that per-cell loses by specialising.

Restricted to 9 cells (leads 24/48/72 × 3 stations) — the per-cell
BART parquet's scope.

Usage: python scripts/linear_pool_4a_4apc.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MODELS_PRECIP = ROOT / "data" / "models" / "precipitation"
BART_9CELL = ROOT.parent / "WeatherProbabilistic" / "reports" / "phase6_artefacts" / "_9cell_full" / "test_predictions.parquet"

REPORT_DIR = ROOT / "data" / "reports" / "linear_pool_4a_4apc"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _latest_4a_with_predictions(station_dir: Path) -> Path | None:
    """Pick the most-recent lead-pooled 4a bundle with test_predictions.parquet."""
    candidates = []
    for vdir in sorted(station_dir.glob("v*phase4a*"), reverse=True):
        meta = vdir / "training_metadata.json"
        pred = vdir / "test_predictions.parquet"
        if not (meta.exists() and pred.exists()):
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        if m.get("Phase") == "4a":
            candidates.append((m.get("TrainedAtUtc", ""), vdir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _brier(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def _optimal_w(a, b, y):
    """Closed-form Brier-minimising w for P = w·a + (1-w)·b, clipped to [0,1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = a - b
    denom = float(np.mean(d * d))
    if denom < 1e-12:
        return 0.5
    w = float(np.mean((y - b) * d) / denom)
    return float(np.clip(w, 0.0, 1.0))


def main() -> None:
    if not BART_9CELL.exists():
        print(f"Per-cell BART parquet not found at {BART_9CELL}.")
        print("Run scripts/run_phase6_bart_9cell.py in WeatherProbabilistic first.")
        sys.exit(1)
    if not MODELS_PRECIP.is_dir():
        print(f"No models tree at {MODELS_PRECIP}.")
        sys.exit(1)

    stations = sorted(p.name for p in MODELS_PRECIP.iterdir() if p.is_dir())
    print(f"Stations on disk: {stations}\n")

    con = duckdb.connect()

    union_parts = []
    for station in stations:
        sdir = MODELS_PRECIP / station
        b4 = _latest_4a_with_predictions(sdir)
        if b4 is None:
            print(f"  {station}: missing lead-pooled 4a test_predictions — skipping.")
            continue
        print(f"  {station}: lead-pooled 4a={b4.name}")
        union_parts.append(f"""
            SELECT '{station}' AS station, p4.valid_time, p4.lead,
                   p4.p_wet AS p_4a_pooled, bart.p_wet AS p_4a_percell,
                   p4.observed_wet AS y
            FROM read_parquet('{(b4 / "test_predictions.parquet").as_posix()}') p4
            JOIN read_parquet('{BART_9CELL.as_posix()}') bart
              ON bart.station = '{station}'
             AND bart.valid_time = p4.valid_time
             AND bart.lead = p4.lead
        """)
    if not union_parts:
        print("\nNo lead-pooled 4a bundles with test_predictions on disk.")
        sys.exit(1)

    sql = " UNION ALL ".join(union_parts) + " ORDER BY station, lead, valid_time"
    rows = con.execute(sql).fetchall()
    if not rows:
        print("\nNo joined rows. Either 4a's test window doesn't overlap the "
              "9-cell BART's (valid_time, lead) cells, or the parquet has no "
              "matching rows.")
        sys.exit(1)

    by_cell: dict[tuple[str, int], list] = {}
    for r in rows:
        station, vt, lead, p_pooled, p_percell, y = r
        by_cell.setdefault((station, int(lead)), []).append((vt, p_pooled, p_percell, y))

    print(f"\nJoined {len(rows):,} rows across {len(by_cell)} (station, lead) cells.\n")

    results = []
    for (station, lead), records in sorted(by_cell.items()):
        records.sort(key=lambda x: x[0])
        n = len(records)
        if n < 50:
            print(f"  {station} lead={lead}h — only {n} rows, skipping.")
            continue

        # a = lead-pooled 4a, b = per-cell BART (so w → 1 means trust pool)
        a = np.array([r[1] for r in records], dtype=np.float64)
        b = np.array([r[2] for r in records], dtype=np.float64)
        y = np.array([r[3] for r in records], dtype=np.float64)

        split_idx = int(0.7 * n)
        a_fit, a_test = a[:split_idx], a[split_idx:]
        b_fit, b_test = b[:split_idx], b[split_idx:]
        y_fit, y_test = y[:split_idx], y[split_idx:]

        w_star = _optimal_w(a_fit, b_fit, y_fit)
        p_blend = w_star * a_test + (1 - w_star) * b_test
        p_5050  = 0.5 * a_test + 0.5 * b_test

        results.append({
            "station": station, "lead": lead,
            "n_total": n, "n_fit": split_idx, "n_test": n - split_idx,
            "wet_rate_test":  float(y_test.mean()),
            "w_star":         w_star,
            "brier_4a_pooled":   _brier(a_test, y_test),
            "brier_4a_percell":  _brier(b_test, y_test),
            "brier_5050":        _brier(p_5050, y_test),
            "brier_blend":       _brier(p_blend, y_test),
        })

    print(f"{'station':<28} {'lead':>4} {'n_test':>7} {'wet%':>5} {'w*':>5} "
          f"{'B(pooled)':>10} {'B(percell)':>11} {'B(5050)':>8} {'B(blend)':>9} "
          f"{'Δ vs pcell':>11}")
    print("-" * 125)
    for r in results:
        d_pc = (100.0 * (r["brier_blend"] - r["brier_4a_percell"]) / r["brier_4a_percell"]) if r["brier_4a_percell"] > 0 else 0.0
        print(
            f"{r['station']:<28} {r['lead']:>4} {r['n_test']:>7,} "
            f"{100*r['wet_rate_test']:>4.1f}% {r['w_star']:>5.2f} "
            f"{r['brier_4a_pooled']:>10.4f} {r['brier_4a_percell']:>11.4f} "
            f"{r['brier_5050']:>8.4f} {r['brier_blend']:>9.4f} "
            f"{d_pc:>+10.1f}%"
        )

    total_n = sum(r["n_test"] for r in results)
    if total_n > 0:
        agg = {k: sum(r[k] * r["n_test"] for r in results) / total_n
               for k in ("brier_4a_pooled", "brier_4a_percell", "brier_5050", "brier_blend")}
        agg_w = sum(r["w_star"] * r["n_test"] for r in results) / total_n
        print("-" * 125)
        print(
            f"{'AGGREGATE (n-weighted)':<28} {'':>4} {total_n:>7,} {'':>5} {agg_w:>5.2f} "
            f"{agg['brier_4a_pooled']:>10.4f} {agg['brier_4a_percell']:>11.4f} "
            f"{agg['brier_5050']:>8.4f} {agg['brier_blend']:>9.4f} "
            f"{100*(agg['brier_blend']-agg['brier_4a_percell'])/agg['brier_4a_percell']:>+10.1f}%"
        )
        print()
        print(f"  Lead-pooled 4a vs per-cell BART:   "
              f"{100*(agg['brier_4a_pooled']-agg['brier_4a_percell'])/agg['brier_4a_percell']:+.1f}% Brier "
              f"(positive = per-cell wins, as expected)")
        print(f"  Naive 50/50 vs per-cell BART:      "
              f"{100*(agg['brier_5050']-agg['brier_4a_percell'])/agg['brier_4a_percell']:+.1f}% Brier")
        print(f"  Fitted-w blend vs per-cell BART:   "
              f"{100*(agg['brier_blend']-agg['brier_4a_percell'])/agg['brier_4a_percell']:+.1f}% Brier")
        print(f"  Aggregate w* on lead-pooled:       {agg_w:.2f}")
        if agg_w < 0.05:
            print("\n  Interpretation: w* ≈ 0 — lead-pooled contributes nothing, "
                  "per-cell BART captures all the signal.")
        elif agg_w < 0.2:
            print("\n  Interpretation: w* small but non-zero — lead-pooled has some "
                  "residual cross-lead info worth a few percent of the pool weight.")
        else:
            print("\n  Interpretation: w* meaningful — lead-pooled has real "
                  "complementary info; the blend captures genuine architectural diversity.")

    csv_path = REPORT_DIR / "results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        cols = ["station", "lead", "n_total", "n_fit", "n_test",
                "wet_rate_test", "w_star",
                "brier_4a_pooled", "brier_4a_percell", "brier_5050", "brier_blend"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote per-cell results → {csv_path}")


if __name__ == "__main__":
    main()
