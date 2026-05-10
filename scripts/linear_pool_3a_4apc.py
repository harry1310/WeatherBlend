"""Per-(station, lead) linear-pool bake-off: 3a + per-cell BART.

Mirror of ``linear_pool_3a_4a.py`` but reads the per-cell BART
predictions from
``WeatherProbabilistic/reports/phase6_artefacts/_9cell_full/test_predictions.parquet``
(produced by ``run_phase6_bart_9cell.py``) instead of the deployed
lead-pooled 4a's per-station bundles.

Asks: does a linear pool of 3a + per-cell BART beat the lead-pooled
3a + 4a pool we already measured? If the per-cell BART recovers the
~5.7% headline edge that lead-pooling threw away, the per-cell pool
should win meaningfully over the lead-pooled pool.

Same closed-form Brier-optimising weight, same chronological 70/30
split, same Bellever/Bovey/Hexworthy stations. Limited to leads
{24, 48, 72} because that's the 9-cell research scope.

Usage: python scripts/linear_pool_3a_4apc.py
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
# 9-cell BART parquet lives in the sibling repo; the bake-off only runs
# locally so the relative path resolves cleanly when both repos are
# cloned alongside (the standard layout).
BART_9CELL = ROOT.parent / "WeatherProbabilistic" / "reports" / "phase6_artefacts" / "_9cell_full" / "test_predictions.parquet"

REPORT_DIR = ROOT / "data" / "reports" / "linear_pool_3a_4apc"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _latest_3a_with_predictions(station_dir: Path) -> Path | None:
    """Mirror of the 3a lookup in linear_pool_3a_4a.py."""
    candidates = []
    for vdir in sorted(station_dir.glob("v*"), reverse=True):
        meta = vdir / "training_metadata.json"
        pred = vdir / "test_predictions.parquet"
        if not (meta.exists() and pred.exists()):
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        if m.get("Phase") == "3a":
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
    """Closed-form Brier-minimising weight, clipped to [0, 1]."""
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
        print(f"BART 9-cell parquet not found at {BART_9CELL}.")
        print("Run scripts/run_phase6_bart_9cell.py in WeatherProbabilistic first.")
        sys.exit(1)
    if not MODELS_PRECIP.is_dir():
        print(f"No 3a models tree at {MODELS_PRECIP}. Pull from R2 first.")
        sys.exit(1)

    stations = sorted(p.name for p in MODELS_PRECIP.iterdir() if p.is_dir())
    print(f"Stations on disk: {stations}\n")

    con = duckdb.connect()

    # 3a per-station × 4a-per-cell joined on (valid_time, lead). The 9-cell
    # parquet carries station as a column (one big union); 3a's parquet
    # is per-station so we filter by the station name when joining.
    union_parts = []
    for station in stations:
        sdir = MODELS_PRECIP / station
        b3 = _latest_3a_with_predictions(sdir)
        if b3 is None:
            print(f"  {station}: missing 3a test_predictions — skipping.")
            continue
        print(f"  {station}: 3a={b3.name}")
        union_parts.append(f"""
            SELECT '{station}' AS station, p3.valid_time, p3.lead,
                   p3.p_wet AS p_3a, bart.p_wet AS p_4a,
                   p3.observed_wet AS y
            FROM read_parquet('{(b3 / "test_predictions.parquet").as_posix()}') p3
            JOIN read_parquet('{BART_9CELL.as_posix()}') bart
              ON bart.station = '{station}'
             AND bart.valid_time = p3.valid_time
             AND bart.lead = p3.lead
        """)
    if not union_parts:
        print("\nNo 3a stations with test_predictions on disk.")
        sys.exit(1)

    sql = " UNION ALL ".join(union_parts) + " ORDER BY station, lead, valid_time"
    rows = con.execute(sql).fetchall()
    if not rows:
        print("\nNo joined rows. Either 3a's test windows don't overlap "
              "the 9-cell BART's (valid_time, lead) cells, or the 9-cell "
              "parquet has no rows for any 3a-stationed cell.")
        sys.exit(1)

    by_cell: dict[tuple[str, int], list] = {}
    for r in rows:
        station, vt, lead, p3, p4, y = r
        by_cell.setdefault((station, int(lead)), []).append((vt, p3, p4, y))

    print(f"\nJoined {len(rows):,} rows across {len(by_cell)} (station, lead) cells.\n")

    results = []
    for (station, lead), records in sorted(by_cell.items()):
        records.sort(key=lambda x: x[0])
        n = len(records)
        if n < 50:
            print(f"  {station} lead={lead}h — only {n} rows, skipping.")
            continue

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
            "wet_rate_test": float(y_test.mean()),
            "w_star":       w_star,
            "brier_3a":     _brier(a_test, y_test),
            "brier_4apc":   _brier(b_test, y_test),
            "brier_5050":   _brier(p_5050, y_test),
            "brier_blend":  _brier(p_blend, y_test),
        })

    print(f"{'station':<28} {'lead':>4} {'n_test':>7} {'wet%':>5} {'w*':>5} "
          f"{'B(3a)':>7} {'B(4apc)':>8} {'B(5050)':>8} {'B(blend)':>9} {'Δ vs 4apc':>10} {'Δ vs 3a':>9}")
    print("-" * 120)
    for r in results:
        d_pc = (100.0 * (r["brier_blend"] - r["brier_4apc"]) / r["brier_4apc"]) if r["brier_4apc"] > 0 else 0.0
        d_3a = (100.0 * (r["brier_4apc"]  - r["brier_3a"])   / r["brier_3a"])   if r["brier_3a"]   > 0 else 0.0
        print(
            f"{r['station']:<28} {r['lead']:>4} {r['n_test']:>7,} "
            f"{100*r['wet_rate_test']:>4.1f}% {r['w_star']:>5.2f} "
            f"{r['brier_3a']:>7.4f} {r['brier_4apc']:>8.4f} "
            f"{r['brier_5050']:>8.4f} {r['brier_blend']:>9.4f} "
            f"{d_pc:>+9.1f}% {d_3a:>+8.1f}%"
        )

    total_n = sum(r["n_test"] for r in results)
    if total_n > 0:
        agg = {k: sum(r[k] * r["n_test"] for r in results) / total_n
               for k in ("brier_3a", "brier_4apc", "brier_5050", "brier_blend")}
        agg_w = sum(r["w_star"] * r["n_test"] for r in results) / total_n
        print("-" * 120)
        print(
            f"{'AGGREGATE (n-weighted)':<28} {'':>4} {total_n:>7,} {'':>5} {agg_w:>5.2f} "
            f"{agg['brier_3a']:>7.4f} {agg['brier_4apc']:>8.4f} "
            f"{agg['brier_5050']:>8.4f} {agg['brier_blend']:>9.4f} "
            f"{100*(agg['brier_blend']-agg['brier_4apc'])/agg['brier_4apc']:>+9.1f}% "
            f"{100*(agg['brier_4apc']-agg['brier_3a'])/agg['brier_3a']:>+8.1f}%"
        )
        print(f"\n  4a-per-cell vs 3a:        {100*(agg['brier_4apc']-agg['brier_3a'])/agg['brier_3a']:+.1f}% Brier")
        print(f"  3a + per-cell pool vs 3a:    {100*(agg['brier_blend']-agg['brier_3a'])/agg['brier_3a']:+.1f}% Brier")
        print(f"  3a + per-cell pool vs 4apc:  {100*(agg['brier_blend']-agg['brier_4apc'])/agg['brier_4apc']:+.1f}% Brier")

    csv_path = REPORT_DIR / "results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        cols = ["station", "lead", "n_total", "n_fit", "n_test",
                "wet_rate_test", "w_star",
                "brier_3a", "brier_4apc", "brier_5050", "brier_blend"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote per-cell results → {csv_path}")


if __name__ == "__main__":
    main()
