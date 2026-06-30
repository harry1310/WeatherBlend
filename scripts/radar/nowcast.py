"""Live Bonehill radar nowcast: fetch latest ODIM frames -> advection engine -> calibrate -> JSON.

Reads the Bonehill crag (display) + the 3 Dartmoor EA gauges (real-time cross-check). Writes
`data/radar/nowcast/bonehill.json` (the workflow pushes it to R2; the site card reads it client-side).

v1 = pure advection (trend_gain=0) to stay consistent with the calibration (fit on pure-advection accums).
The growth/decay term in _engine is built but stays OFF until re-calibrated via the verification track.
"""
import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_odim
import _engine

SITES = {
    "Bonehill crag":         (50.5831,   -3.7931),    # display target (no ground truth)
    "Bellever Dartmoor":     (50.582381, -3.898151),  # EA gauge (verify)
    "Bovey Tracey":          (50.592312, -3.716672),  # EA gauge
    "Dartmoor nr Hexworthy": (50.548615, -3.938746),  # EA gauge
}
CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_bonehill.json")
LIVE_DIR = "data/radar/live"
OUT = "data/radar/nowcast/bonehill.json"
LEAD_MIN, DT_MIN, NBHD_KM, TREND_GAIN = 60, 15, 2, 0.0
import numpy as np


def p_wet(accum, cal):
    z = np.log1p(accum / cal["wet"])
    return float(1.0 / (1.0 + np.exp(-(cal["a"] + cal["b"] * z))))


def start_stop(series, dt, valid_epoch):
    """First wet onset / first dry-after-onset within the window, as clock times (epoch-derived)."""
    wet = [s >= _engine.WET for s in series]
    start = next((i + 1 for i, w in enumerate(wet) if w), None)
    stop = next((i + 1 for i, w in enumerate(wet) if start and (i + 1) > start and not w), None)
    fmt = lambda k: datetime.fromtimestamp(valid_epoch + k * dt * 60, timezone.utc).strftime("%H:%M") if k else None
    return (start * dt if start else None), fmt(start), fmt(stop)


def compute(frame_paths, cal, sites=SITES, now=None):
    """Pipeline core (testable, no network): frame paths -> engine -> calibrate -> output dict."""
    frames, georef, valid = [], None, None
    for p in frame_paths:
        r, georef, valid = _engine.load_odim(p)
        frames.append(r)
    res = _engine.nowcast(frames, georef, sites, lead_min=LEAD_MIN, dt_min=DT_MIN,
                          trend_gain=TREND_GAIN, nbhd_km=NBHD_KM)
    now = now or datetime.now(timezone.utc)
    valid_epoch = int(valid.astype("datetime64[s]").astype("int64"))
    out = {
        "frame_valid": str(valid), "computed_at": now.isoformat(timespec="seconds"),
        "frame_age_min": round((now.timestamp() - valid_epoch) / 60.0, 1),
        "lead_min": LEAD_MIN, "motion_kmh": round(res["_motion"]["median_kmh"], 1),
        "attribution": "Contains Met Office data © Crown copyright, CC BY-SA", "sites": {},
    }
    for n in sites:
        s = res[n]
        eta_min, start_clk, stop_clk = start_stop(s["rate_series"], DT_MIN, valid_epoch)
        out["sites"][n] = {
            "accum_mm": round(s["accum_mm"], 3), "p_wet": round(p_wet(s["accum_mm"], cal), 3),
            "max_rate_mmh": round(s["max_rate"], 2),
            "rain_from": start_clk, "rain_until": stop_clk, "onset_in_min": eta_min,
        }
    return out


def main():
    cal = json.load(open(CAL))
    out = compute(fetch_odim.latest_frames(4, LIVE_DIR), cal)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
