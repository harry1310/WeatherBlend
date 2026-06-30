"""Lightweight radar-advection nowcast engine — shared by the live nowcast and the archive verification.

Method: half-res Farneback optical flow (OpenCV) for motion + semi-Lagrangian advection, plus an optional
Lagrangian intensity-TREND term for growth/decay (the one thing pure advection lacks). Builds everywhere
(no compiler) — pySTEPS is the measured v2 upgrade (CI-only on the Windows dev box). See docs/RADAR_NOWCAST_PLAN.md.

Units: ODIM `rainrate` is mm/h. `nodata` -> NaN, `undetect`(0) -> dry.
"""
import numpy as np
import cv2
import h5py
from pyproj import Transformer

WET = 0.1  # mm/h wet threshold (matches the backtest + the EA gauge truth)
_TF = {}   # proj-string -> Transformer (cached)


def load_odim(path):
    """ODIM HDF5 -> (rate mm/h grid with NaN for nodata, georef dict, valid datetime64)."""
    with h5py.File(path, "r") as f:
        w = f["where"].attrs
        georef = dict(proj=w["projdef"].decode(), xs=float(w["xscale"]), ys=float(w["yscale"]),
                      ul_lat=float(w["UL_lat"]), ul_lon=float(w["UL_lon"]))
        dw = f["dataset1/data1/what"].attrs
        gain, off, nodata = float(dw["gain"]), float(dw["offset"]), float(dw["nodata"])
        raw = f["dataset1/data1/data"][:].astype(np.float32)
        date = f["what"].attrs["date"].decode(); time = f["what"].attrs["time"].decode()
    rate = raw * gain + off
    rate[raw == nodata] = np.nan
    valid = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}")
    return rate, georef, valid


def pixel_of(georef, lat, lon):
    """(row, col) in the full ODIM grid for a lat/lon, via the file's own projection + UL corner."""
    proj = georef["proj"]
    tf = _TF.get(proj)
    if tf is None:
        tf = _TF[proj] = Transformer.from_crs(4326, proj, always_xy=True)
    x_ul, y_ul = tf.transform(georef["ul_lon"], georef["ul_lat"])
    x, y = tf.transform(lon, lat)
    return int(round((y_ul - y) / georef["ys"])), int(round((x - x_ul) / georef["xs"]))


def to_u8(R):
    """Rain rate (mm/h) -> 0..255 dBR-ish image for optical flow. NaN treated as dry."""
    R = np.nan_to_num(R, nan=0.0)
    x = 10 * np.log10(np.clip(R, 0.05, None))
    x[R < WET] = -15.0
    return np.clip((x + 15.0) / 45.0 * 255.0, 0, 255).astype(np.uint8)


def flow_between(prev, cur):
    """Half-res Farneback between two frames -> per-FRAME-INTERVAL pixel displacement field (full res)."""
    ph, ch = cv2.pyrDown(to_u8(prev)), cv2.pyrDown(to_u8(cur))
    fl = cv2.calcOpticalFlowFarneback(ph, ch, None, 0.5, 4, 21, 3, 5, 1.2, 0)
    return cv2.resize(fl, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0


def advect(field, flow, frac):
    """Semi-Lagrangian backtrack: where `field` lands after `frac` frame-intervals of motion."""
    H, W = field.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    mx = (X - frac * flow[..., 0]).astype(np.float32)
    my = (Y - frac * flow[..., 1]).astype(np.float32)
    return cv2.remap(np.nan_to_num(field).astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0)


def nowcast(frames, georef, sites, lead_min=60, dt_min=15, trend_gain=0.0, nbhd_km=2):
    """Advect the latest field over the next `lead_min` and read each site.

    frames: list of full rate grids, OLDEST..NEWEST (>=2). georef: from load_odim. sites: {name:(lat,lon)}.
    trend_gain: 0 = pure advection; >0 adds a damped Lagrangian growth/decay trend (residual of advection).
    Returns {name: {accum_mm, max_rate, rate_series, wet}} plus a "_motion" stats entry.
    """
    cur, prev = frames[-1], frames[-2]
    flow_full = flow_between(prev, cur)

    # crop to a window covering all sites + inflow margin (≈150 km), in full-grid coords
    rc = {n: pixel_of(georef, la, lo) for n, (la, lo) in sites.items()}
    rows = [r for r, _ in rc.values()]; cols = [c for _, c in rc.values()]
    HW = 150
    r0 = max(0, min(rows) - HW); r1 = min(cur.shape[0], max(rows) + HW + 1)
    c0 = max(0, min(cols) - HW); c1 = min(cur.shape[1], max(cols) + HW + 1)
    cur_c = cur[r0:r1, c0:c1]
    flow_c = flow_full[r0:r1, c0:c1]
    trend = None
    if trend_gain:
        prev_c = prev[r0:r1, c0:c1]
        trend = cur_c - advect(prev_c, flow_c, 1.0)   # growth(+)/decay(-) not explained by advection, per dt_min

    nsub = max(1, lead_min // dt_min)                  # sub-steps across the lead window
    rad = max(0, int(round(nbhd_km)))
    out = {}
    for n, (r, c) in rc.items():
        gr, gc = r - r0, c - c0
        series = []
        for k in range(1, nsub + 1):
            fld = advect(cur_c, flow_c, k)
            if trend is not None:
                damp = max(0.0, 1.0 - (k - 1) / nsub)  # trend fades over the window
                fld = fld + trend_gain * damp * advect(trend, flow_c, k)
            box = fld[max(0, gr - rad):gr + rad + 1, max(0, gc - rad):gc + rad + 1]
            series.append(float(max(0.0, box.max())))
        accum = sum(v * (dt_min / 60.0) for v in series)  # mm over the lead window
        out[n] = dict(accum_mm=accum, max_rate=max(series), rate_series=series, wet=accum >= WET)

    mag = np.hypot(flow_c[..., 0], flow_c[..., 1])
    out["_motion"] = dict(median_kmh=float(np.median(mag)) * 60.0 / dt_min,
                          p95_kmh=float(np.percentile(mag, 95)) * 60.0 / dt_min,
                          crop=(int(r0), int(r1), int(c0), int(c1)))
    return out
