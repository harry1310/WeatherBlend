"""Per-(station, lead) linear-pool bake-off: 3a + 4a P(wet).

Asks: does a fitted convex combination of 3a and 4a beat either single
model on held-out Brier? Per (station, lead) cell::

    P_blend = w · P_3a + (1 - w) · P_4a,    w fit on val slice

closed-form on Brier:

    w* = mean((y - b) · (a - b)) / mean((a - b)²),    a = P_3a, b = P_4a

clipped to [0, 1]. Compares four candidates on the held-out test slice:
3a alone, 4a alone, naive 50/50, fitted-w blend.

Data source: each training bundle saves a per-row
``test_predictions.parquet`` with schema
``{valid_time, station, lead, p_wet, observed_wet}``. We pick the
latest 3a + 4a per station, inner-join on (valid_time, lead), and split
chronologically 70/30 — fit `w` on the older 70%, score on the newer
30%. Both models genuinely never saw their respective test slices, and
the inner-join restricts us to rows that fall within BOTH models'
held-out windows, so no train-set leakage.

Run after a fresh training cycle has produced
``data/models/precipitation/{station}/{version}/test_predictions.parquet``
for both phases (Sunday auto-retrain on/after 2026-05-17 — the slice 1
Python and slice 1c .NET commits 2026-05-10 introduce the parquet save).

Usage: python scripts/linear_pool_3a_4a.py
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

REPORT_DIR = ROOT / "data" / "reports" / "linear_pool_3a_4a"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _latest_bundle_with_predictions(station_dir: Path, phase: str) -> Path | None:
    """Pick the most-recent version dir under ``station_dir`` whose
    training_metadata.Phase matches ``phase`` and that has a
    test_predictions.parquet on disk. Returns None if no such bundle
    exists yet (typically: pre-2026-05-10 bundles, or guard-failed
    runs that left an orphan dir without a summary).
    """
    candidates = []
    for vdir in sorted(station_dir.glob("v*"), reverse=True):
        meta_path = vdir / "training_metadata.json"
        pred_path = vdir / "test_predictions.parquet"
        if not (meta_path.exists() and pred_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("Phase") == phase:
            candidates.append((meta.get("TrainedAtUtc", ""), vdir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _brier(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def _optimal_w(a, b, y):
    """Closed-form Brier-minimising weight for P = w·a + (1-w)·b. Clipped to [0,1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = a - b
    denom = float(np.mean(d * d))
    if denom < 1e-12:
        # 3a and 4a effectively agree — w undefined, any blend = same point.
        return 0.5
    w = float(np.mean((y - b) * d) / denom)
    return float(np.clip(w, 0.0, 1.0))


# ----- log-linear (logit) pool -----------------------------------------
#
# logit(P_blend) = w · logit(P_3a) + (1 - w) · logit(P_4a)
#
# No closed form for the Brier-optimal w (the loss is non-quadratic in w),
# so 1D bounded Brent search on [0, 1]. Same chronological 70/30 split as
# the linear pool — fit on the older 70%, score on the newer 30%.

_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit_blend(a, b, w):
    return _sigmoid(w * _logit(a) + (1 - w) * _logit(b))


def _optimal_w_logit(a, b, y):
    """Brent search for the Brier-minimising w in the logit pool. Bounded
    to [0, 1] — same convex-combination semantics as the linear pool."""
    from scipy.optimize import minimize_scalar
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    al = _logit(a)
    bl = _logit(b)
    def loss(w):
        p = _sigmoid(w * al + (1 - w) * bl)
        return float(np.mean((p - y) ** 2))
    res = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def main() -> None:
    if not MODELS_PRECIP.is_dir():
        print(f"No models tree at {MODELS_PRECIP}. Pull from R2 first.")
        sys.exit(1)

    stations = sorted(p.name for p in MODELS_PRECIP.iterdir() if p.is_dir())
    print(f"Stations on disk: {stations}\n")

    con = duckdb.connect()
    pairs: list[tuple[str, Path, Path]] = []
    for station in stations:
        sdir = MODELS_PRECIP / station
        b3 = _latest_bundle_with_predictions(sdir, "3a")
        b4 = _latest_bundle_with_predictions(sdir, "4a")
        if b3 is None or b4 is None:
            print(f"  {station}: missing test_predictions for "
                  f"{'3a' if b3 is None else ''}{' + ' if (b3 is None and b4 is None) else ''}{'4a' if b4 is None else ''}"
                  f" — skipping (need a fresh retrain with the slice-1 trainer changes).")
            continue
        print(f"  {station}: 3a={b3.name}, 4a={b4.name}")
        pairs.append((station, b3, b4))

    if not pairs:
        print("\nNo (station, 3a, 4a) bundles with test_predictions.parquet on disk.")
        print("This bake-off needs the slice-1 / slice-1c trainer save (2026-05-10) to have")
        print("run on a recent retrain cycle — e.g. the Sunday auto-retrain on or after 2026-05-17.")
        sys.exit(0)

    # Build a per-station joined table via DuckDB. Each station's 3a +
    # 4a parquets are inner-joined on (valid_time, lead); the union
    # across stations is a single fetchall().
    union_parts = []
    for station, b3, b4 in pairs:
        union_parts.append(f"""
            SELECT '{station}' AS station, p3.valid_time, p3.lead,
                   p3.p_wet AS p_3a, p4.p_wet AS p_4a,
                   p3.observed_wet AS y
            FROM read_parquet('{(b3 / "test_predictions.parquet").as_posix()}') p3
            JOIN read_parquet('{(b4 / "test_predictions.parquet").as_posix()}') p4
              USING (valid_time, lead)
        """)
    sql = " UNION ALL ".join(union_parts) + " ORDER BY station, lead, valid_time"
    rows = con.execute(sql).fetchall()
    if not rows:
        print("\nNo joined rows. The 3a + 4a test windows don't overlap on (valid_time, lead) — "
              "is one model's held-out window strictly outside the other's?")
        sys.exit(1)

    # Bucket by (station, lead).
    by_cell: dict[tuple[str, int], list] = {}
    for r in rows:
        station, vt, lead, p_3a, p_4a, y = r
        by_cell.setdefault((station, int(lead)), []).append((vt, p_3a, p_4a, y))

    print(f"\nJoined {len(rows):,} rows across {len(by_cell)} (station, lead) cells.\n")

    results = []
    for (station, lead), records in sorted(by_cell.items()):
        records.sort(key=lambda x: x[0])
        n = len(records)
        if n < 50:
            print(f"  {station} lead={lead}h — only {n} rows, skipping.")
            continue

        a = np.array([r[1] for r in records], dtype=np.float64)  # P_3a
        b = np.array([r[2] for r in records], dtype=np.float64)  # P_4a
        y = np.array([r[3] for r in records], dtype=np.float64)  # truth

        split_idx = int(0.7 * n)
        a_fit, a_test = a[:split_idx], a[split_idx:]
        b_fit, b_test = b[:split_idx], b[split_idx:]
        y_fit, y_test = y[:split_idx], y[split_idx:]

        w_star = _optimal_w(a_fit, b_fit, y_fit)
        p_blend = w_star * a_test + (1 - w_star) * b_test
        p_5050  = 0.5 * a_test + 0.5 * b_test
        # Logit-pool: separate w fit on logit-of-prob loss surface.
        w_logit = _optimal_w_logit(a_fit, b_fit, y_fit)
        p_logit = _logit_blend(a_test, b_test, w_logit)

        results.append({
            "station": station, "lead": lead,
            "n_total": n, "n_fit": split_idx, "n_test": n - split_idx,
            "fit_dates":  f"{records[0][0].date()} → {records[split_idx-1][0].date()}",
            "test_dates": f"{records[split_idx][0].date()} → {records[-1][0].date()}",
            "wet_rate_test": float(y_test.mean()),
            "w_star":     w_star,
            "w_logit":    w_logit,
            "brier_3a":   _brier(a_test, y_test),
            "brier_4a":   _brier(b_test, y_test),
            "brier_5050": _brier(p_5050, y_test),
            "brier_blend":_brier(p_blend, y_test),
            "brier_logit":_brier(p_logit, y_test),
        })

    # Per-cell table — adds B(logit) + w_logit alongside the linear-pool
    # columns so logit vs linear is easy to read off at a glance.
    print(f"{'station':<28} {'lead':>4} {'n_test':>7} {'wet%':>5} "
          f"{'w*':>5} {'wL':>5} "
          f"{'B(3a)':>7} {'B(4a)':>7} {'B(5050)':>8} {'B(lin)':>7} {'B(log)':>7} "
          f"{'Δlin/4a':>8} {'Δlog/4a':>8}")
    print("-" * 125)
    for r in results:
        d_lin = (100.0 * (r["brier_blend"] - r["brier_4a"]) / r["brier_4a"]) if r["brier_4a"] > 0 else 0.0
        d_log = (100.0 * (r["brier_logit"] - r["brier_4a"]) / r["brier_4a"]) if r["brier_4a"] > 0 else 0.0
        print(
            f"{r['station']:<28} {r['lead']:>4} {r['n_test']:>7,} "
            f"{100*r['wet_rate_test']:>4.1f}% "
            f"{r['w_star']:>5.2f} {r['w_logit']:>5.2f} "
            f"{r['brier_3a']:>7.4f} {r['brier_4a']:>7.4f} "
            f"{r['brier_5050']:>8.4f} {r['brier_blend']:>7.4f} {r['brier_logit']:>7.4f} "
            f"{d_lin:>+7.1f}% {d_log:>+7.1f}%"
        )

    # n-weighted aggregate including logit pool.
    total_n = sum(r["n_test"] for r in results)
    if total_n > 0:
        agg = {k: sum(r[k] * r["n_test"] for r in results) / total_n
               for k in ("brier_3a", "brier_4a", "brier_5050", "brier_blend", "brier_logit")}
        agg_w     = sum(r["w_star"]  * r["n_test"] for r in results) / total_n
        agg_wlog  = sum(r["w_logit"] * r["n_test"] for r in results) / total_n
        print("-" * 125)
        print(
            f"{'AGGREGATE (n-weighted)':<28} {'':>4} {total_n:>7,} {'':>5} "
            f"{agg_w:>5.2f} {agg_wlog:>5.2f} "
            f"{agg['brier_3a']:>7.4f} {agg['brier_4a']:>7.4f} "
            f"{agg['brier_5050']:>8.4f} {agg['brier_blend']:>7.4f} {agg['brier_logit']:>7.4f} "
            f"{100*(agg['brier_blend']-agg['brier_4a'])/agg['brier_4a']:>+7.1f}% "
            f"{100*(agg['brier_logit']-agg['brier_4a'])/agg['brier_4a']:>+7.1f}%"
        )
        print(f"\n  Naive 50/50  vs 4a: {100*(agg['brier_5050']-agg['brier_4a'])/agg['brier_4a']:+.1f}% Brier")
        print(f"  Linear pool  vs 4a: {100*(agg['brier_blend']-agg['brier_4a'])/agg['brier_4a']:+.1f}% Brier")
        print(f"  Logit pool   vs 4a: {100*(agg['brier_logit']-agg['brier_4a'])/agg['brier_4a']:+.1f}% Brier")
        print(f"  Linear pool  vs 3a: {100*(agg['brier_blend']-agg['brier_3a'])/agg['brier_3a']:+.1f}% Brier")
        print(f"  Logit pool   vs 3a: {100*(agg['brier_logit']-agg['brier_3a'])/agg['brier_3a']:+.1f}% Brier")

    # CSV out (now includes logit columns).
    csv_path = REPORT_DIR / "results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        cols = ["station", "lead", "n_total", "n_fit", "n_test", "fit_dates", "test_dates",
                "wet_rate_test", "w_star", "w_logit",
                "brier_3a", "brier_4a", "brier_5050", "brier_blend", "brier_logit"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote per-cell results → {csv_path}")


if __name__ == "__main__":
    main()
