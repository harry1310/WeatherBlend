"""Rolling verification of the LIVE radar engine over the Met Office ODIM archive, at the 3 Dartmoor EA
gauges. Runs the SAME `_engine` the live product uses (so it validates production, not a different method),
scores radar P(wet next hour) vs gauge truth, and is the place to compare radar vs the NWP blend's gauge
forecasts. Scheduled (NOT armed-gated) via radar-verify.yml — see docs/RADAR_NOWCAST_PLAN.md.

Scope note: the rigorous *historical* radar-vs-NWP backtest is phase2_vs_nwp.py (NIMROD, cached). This is the
forward, engine-matched rolling check over recent ODIM. Default window = last 30 days.

  python scripts/radar/verify_archive.py [DAYS]   # default 30
"""
import os
import sys
import json
import glob
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_odim
import _engine

RAIN = "data/truth/rainfall"
LIVE_DIR = "data/radar/archive"
LOC = "bonehill_rocks"
CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_bonehill.json")
LEAD_MIN, DT_MIN, NBHD_KM = 60, 15, 2
WET = _engine.WET
# the 3 EA gauges (name as in the rainfall parquet -> lat/lon). Crag has no truth, so it's verify-only here.
GAUGES = {
    "Bellever Dartmoor":     (50.582381, -3.898151),
    "Bovey Tracey":          (50.592312, -3.716672),
    "Dartmoor nr Hexworthy": (50.548615, -3.938746),
}


def hour_key(ts):
    return int(np.datetime64(ts, "h").astype("int64"))


def gauge_truth(station):
    """{hour_key -> wet 0/1} from EA 15-min truth (>=0.1mm in the hour)."""
    q = f"""SELECT date_trunc('hour', ObservedTimeUtc) h, SUM(Value15MinMm) mm
            FROM read_parquet('{RAIN}/location={LOC}/**/*.parquet', hive_partitioning=false, union_by_name=true)
            WHERE StationName = '{station}' GROUP BY 1"""
    return {hour_key(np.datetime64(h)): (1 if (mm is not None and mm >= WET) else 0)
            for h, mm in duckdb.sql(q).fetchall()}


def load_blend_pwet(station):
    """TODO (the one piece needing the blend schema): the deployed precip blend's P(wet) at this gauge,
    per target hour. Source = the predictions parquet the verify.yml path already reads (3c/3o champion).
    Returning {} skips the blend column until wired — don't guess the schema. See CLAUDE.md 'never guess'."""
    return {}


def day_keys(d):
    keys, _ = fetch_odim._list(f"radar/{d:%Y/%m/%d}/", maxkeys=400)
    return sorted(keys)


def p_wet(accum, cal):
    return float(1.0 / (1.0 + np.exp(-(cal["a"] + cal["b"] * np.log1p(accum / cal["wet"])))))


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cal = json.load(open(CAL))
    truth = {g: gauge_truth(g) for g in GAUGES}
    blend = {g: load_blend_pwet(g) for g in GAUGES}
    have_blend = any(blend.values())

    # contingency [hits, miss, fa] + brier accumulators per (gauge, method)
    cont = defaultdict(lambda: np.zeros(3))
    brier = defaultdict(lambda: [0.0, 0])  # [sum_sq, n]
    end = datetime.now(timezone.utc).date()
    georef = None
    done = 0
    for i in range(days):
        d = end - timedelta(days=i)
        keys = day_keys(d)
        if len(keys) < 2:
            continue
        # group by valid hour-minute; we need, per issue hour H:00, the frame at H:00 and H-? for flow
        by_ts = {os.path.basename(k)[:12]: k for k in keys}  # YYYYMMDDhhmm -> key
        for ts, key in by_ts.items():
            if ts[10:12] != "00":                            # issue only at the top of the hour
                continue
            prev_ts = (datetime.strptime(ts, "%Y%m%d%H%M") - timedelta(minutes=DT_MIN)).strftime("%Y%m%d%H%M")
            if prev_ts not in by_ts:
                continue
            issue = hour_key(np.datetime64(datetime.strptime(ts, "%Y%m%d%H%M")))
            vh = issue + 1                                   # +1h target hour
            if not any(vh in truth[g] for g in GAUGES):
                continue
            try:
                p0, _, _ = _engine.load_odim(fetch_odim.download(by_ts[prev_ts], LIVE_DIR))
                p1, georef, _ = _engine.load_odim(fetch_odim.download(key, LIVE_DIR))
            except Exception:
                continue
            res = _engine.nowcast([p0, p1], georef, GAUGES, lead_min=LEAD_MIN, dt_min=DT_MIN,
                                  trend_gain=0.0, nbhd_km=NBHD_KM)
            for g in GAUGES:
                t = truth[g].get(vh)
                if t is None:
                    continue
                pr = p_wet(res[g]["accum_mm"], cal)
                pw = res[g]["accum_mm"] >= WET
                c = cont[(g, "radar")]
                c[0] += pw and t; c[1] += t and not pw; c[2] += pw and not t
                brier[(g, "radar")][0] += (pr - t) ** 2; brier[(g, "radar")][1] += 1
                if have_blend and vh in blend[g]:
                    bp = blend[g][vh]
                    bc = cont[(g, "blend")]
                    bpw = bp >= 0.5
                    bc[0] += bpw and t; bc[1] += t and not bpw; bc[2] += bpw and not t
                    brier[(g, "blend")][0] += (bp - t) ** 2; brier[(g, "blend")][1] += 1
        done += 1

    def csi(c):
        d = c.sum()
        return c[0] / d if d else float("nan")

    out = [f"# Radar live-engine verification vs EA gauges ({LOC}) — last {days} days, {done} days with data", "",
           "- SAME engine as the live product (ODIM + `_engine`); radar P(wet) calibrated; truth = EA gauge hourly wet(>=0.1mm).",
           ("- blend column: PENDING (wire `load_blend_pwet` to the deployed precip blend's gauge P(wet))."
            if not have_blend else "- blend = deployed precip blend P(wet) at the gauge."), "",
           "| gauge | method | n | CSI | Brier |", "|---|---|---:|---:|---:|"]
    for g in GAUGES:
        for m in (["radar", "blend"] if have_blend else ["radar"]):
            c = cont[(g, m)]; b = brier[(g, m)]
            bs = b[0] / b[1] if b[1] else float("nan")
            out.append(f"| {g} | {m} | {int(c.sum())} | {csi(c):.3f} | {bs:.4f} |")
    os.makedirs("data/reports", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    rep = f"data/reports/radar_live_verify_{stamp}.md"
    open(rep, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out)); print("\nwrote", rep)


if __name__ == "__main__":
    main()
