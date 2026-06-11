"""Tide sanity check: Open-Meteo sea_level_height_msl vs the Newlyn tide gauge.

Answers SENNEN_SEA_STATE_PLAN.md open question 3: is the marine API's tide
signal at the pinned Sennen sea cell honest enough to drive the climbability
chip, or do we need our own Newlyn harmonic prediction?

Truth: EA flood-monitoring API, station E72239 (Newlyn, the UK's reference
tide station, ~8 km E of Sennen), 15-min observed water level in mAOD.
Ordnance Datum IS mean sea level at Newlyn (1915-21 epoch), so the two
series share a datum up to ~10-20 cm of sea-level rise since that epoch —
expect a small constant offset, which we report rather than hide.

Run:  .venv python (WeatherProbabilistic) — needs requests + pandas + numpy.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

SENNEN_MARINE = (50.0417, -5.7083)
NEWLYN_MEASURE = "E72239-level-tidal_level-Mean-15_min-mAOD"
DAYS = 28


def fetch_newlyn(start: datetime, end: datetime) -> pd.Series:
    url = (
        f"https://environment.data.gov.uk/flood-monitoring/id/measures/{NEWLYN_MEASURE}/readings"
        f"?since={start:%Y-%m-%dT%H:%M:%SZ}&_limit=10000&_sorted"
    )
    items = requests.get(url, timeout=60).json()["items"]
    s = pd.Series(
        {pd.Timestamp(i["dateTime"]): float(i["value"]) for i in items if "value" in i}
    ).sort_index()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s[(s.index >= start.replace(tzinfo=None)) & (s.index <= end.replace(tzinfo=None))]


def fetch_openmeteo(start: datetime, end: datetime) -> pd.Series:
    lat, lon = SENNEN_MARINE
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}"
        f"&hourly=sea_level_height_msl&timezone=UTC"
        f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
    )
    h = requests.get(url, timeout=60).json()["hourly"]
    s = pd.Series(h["sea_level_height_msl"], index=pd.to_datetime(h["time"]), dtype=float)
    return s.dropna()


def main() -> int:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=DAYS)

    newlyn = fetch_newlyn(start, end)
    om = fetch_openmeteo(start, end)
    print(f"Newlyn 15-min rows: {len(newlyn)}   Open-Meteo hourly rows: {len(om)}")
    if len(newlyn) < 500 or len(om) < 200:
        print("Not enough overlapping data — aborting.")
        return 1

    # The 15-min gauge lets us test sub-hourly phase: shift Open-Meteo by
    # -90..+90 min in 15-min steps and keep the best-correlating lag.
    # (Tide moves up to ~1.5 m/h at springs here, so even 15 min of phase
    # error is ~0.4 m of level error at mid-tide — phase is the thing to check.)
    newlyn_15 = newlyn.resample("15min").mean().interpolate(limit=2)
    best = None
    for lag_min in range(-90, 91, 15):
        om_shifted = om.copy()
        om_shifted.index = om_shifted.index + pd.Timedelta(minutes=lag_min)
        joined = pd.concat([newlyn_15, om_shifted], axis=1, keys=["newlyn", "om"]).dropna()
        if len(joined) < 100:
            continue
        r = joined["newlyn"].corr(joined["om"])
        if best is None or r > best[1]:
            best = (lag_min, r, joined)
    lag_min, r, joined = best

    offset = (joined["om"] - joined["newlyn"]).mean()
    resid = joined["om"] - joined["newlyn"] - offset
    rmse = float(np.sqrt((resid**2).mean()))
    amp_ratio = float(joined["om"].std() / joined["newlyn"].std())

    # High/low water: daily extremes comparison (the chip cares most about
    # how high the water actually gets, not mid-tide tracking).
    daily_hi = joined.resample("1D").max().dropna()
    daily_lo = joined.resample("1D").min().dropna()
    hi_err = (daily_hi["om"] - daily_hi["newlyn"] - offset).abs().mean()
    lo_err = (daily_lo["om"] - daily_lo["newlyn"] - offset).abs().mean()

    print(f"\nWindow: {start:%Y-%m-%d} .. {end:%Y-%m-%d}  ({len(joined)} aligned 15-min points)")
    print(f"Best phase lag: OM {'+' if lag_min >= 0 else ''}{lag_min} min vs Newlyn (r = {r:.4f})")
    print(f"Mean datum offset (OM − Newlyn): {offset:+.3f} m   <- expect ~+0.1..0.2 m (MSL rise since the 1915-21 OD epoch)")
    print(f"RMSE after removing offset:      {rmse:.3f} m")
    print(f"Amplitude ratio (OM/Newlyn):     {amp_ratio:.3f}")
    print(f"Daily HIGH-water |error|:        {hi_err:.3f} m")
    print(f"Daily LOW-water  |error|:        {lo_err:.3f} m")

    verdict = (
        "PASS — usable for the chip as-is"
        if r > 0.98 and rmse < 0.30 and abs(lag_min) <= 30
        else "MARGINAL/FAIL — build the Newlyn harmonics module"
    )
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
