"""Offline sanity check for the Sennen cliff-face rock-temp margins (S5).

Runs the same Force-Restore physics as RockSurfacePhysics.cs (incl. the S3
face-projected shortwave and S4 sea-longwave terms) over Sennen's ERA5 hourly
archive, and reports the per-face condensation-margin distribution + greasy
tier rates. Purpose: a first look at how the maritime dew-point margin behaves
BEFORE the product goes live — informs the greasyMarginC choice (the Bonehill
3.0degC band may be permanently amber at the coast) and gives the IR-gun
visits a baseline to argue with. NOT truth-validated: ERA5 forcing in, physics
out; treat distributions as indicative shape, not absolutes.

Usage (WP venv has duckdb+numpy):
  C:/Projects/Weather/WeatherProbabilistic/.venv/Scripts/python.exe \
      scripts/sennen_rock_margin_sanity.py [--start 2024-06-01] [--end 2026-05-30]
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np

ERA5_GLOB = "C:/Projects/Weather/WeatherBlend/data/truth/era5/location=sennen_cove/**/*.parquet"
LAT, LON = 50.0786, -5.7044

# RockSurfaceConfig values as resolved for Sennen (global block + override).
P = dict(albedo=0.30, eps_rock=0.95, lam=3.0, rho=2650.0, cp=790.0,
         tau_day=86400.0, tau_long=86400.0, f_sky=0.5,
         lw_clear_k=1.0, lw_cloud_k=0.54, mu_scale=0.3, substeps=6)
GREASY_MARGIN_C = 3.0
SIGMA = 5.670374419e-8
FACES = {"west": 270.0, "southwest": 225.0, "south": 180.0}
SLOPE_DEG = 90.0
MIN_ELEV_DIRECT = 3.0
MAX_DNI = 1100.0


def solar_position(dt_utc: datetime, lat: float, lon: float) -> tuple[float, float]:
    """NOAA solar calculator (mirror of SolarGeometry.cs). Returns (elev, az) deg."""
    epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
    oadate = (dt_utc - epoch).total_seconds() / 86400.0
    jd = oadate + 2415018.5
    jc = (jd - 2451545.0) / 36525.0
    rad, deg = math.radians, math.degrees
    mean_lon = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    eq_centre = (math.sin(rad(mean_anom)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
                 + math.sin(rad(2 * mean_anom)) * (0.019993 - 0.000101 * jc)
                 + math.sin(rad(3 * mean_anom)) * 0.000289)
    true_lon = mean_lon + eq_centre
    app_lon = true_lon - 0.00569 - 0.00478 * math.sin(rad(125.04 - 1934.136 * jc))
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * math.cos(rad(125.04 - 1934.136 * jc))
    decl = deg(math.asin(math.sin(rad(obliq)) * math.sin(rad(app_lon))))
    var_y = math.tan(rad(obliq / 2)) ** 2
    eot = 4.0 * deg(var_y * math.sin(2 * rad(mean_lon))
                    - 2 * ecc * math.sin(rad(mean_anom))
                    + 4 * ecc * var_y * math.sin(rad(mean_anom)) * math.cos(2 * rad(mean_lon))
                    - 0.5 * var_y * var_y * math.sin(4 * rad(mean_lon))
                    - 1.25 * ecc * ecc * math.sin(2 * rad(mean_anom)))
    minutes = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60.0
    tst = (minutes + eot + 4.0 * lon + 1440.0) % 1440.0
    ha = tst / 4.0 + 180.0 if tst / 4.0 < 0 else tst / 4.0 - 180.0
    cos_zen = (math.sin(rad(lat)) * math.sin(rad(decl))
               + math.cos(rad(lat)) * math.cos(rad(decl)) * math.cos(rad(ha)))
    zen = deg(math.acos(max(-1.0, min(1.0, cos_zen))))
    elev = 90.0 - zen
    denom = math.cos(rad(lat)) * math.sin(rad(zen))
    if abs(denom) < 1e-9:
        az = 180.0
    else:
        cos_az = (math.sin(rad(lat)) * math.cos(rad(zen)) - math.sin(rad(decl))) / denom
        acos = deg(math.acos(max(-1.0, min(1.0, cos_az))))
        az = (acos + 180.0) % 360.0 if ha > 0 else (540.0 - acos) % 360.0
    return elev, az


def face_incident_sw(ghi: float, direct_frac: float, elev: float, az: float,
                     aspect: float, slope: float, f_sky: float) -> float:
    ghi = max(ghi, 0.0)
    frac = max(0.0, min(1.0, direct_frac))
    sky_diffuse = ghi * (1.0 - frac) * f_sky
    if elev <= MIN_ELEV_DIRECT:
        return sky_diffuse
    sin_elev = math.sin(math.radians(elev))
    dni = min(ghi * frac / sin_elev, MAX_DNI)
    cos_inc = (math.cos(math.radians(elev)) * math.sin(math.radians(slope))
               * math.cos(math.radians(az - aspect))
               + sin_elev * math.cos(math.radians(slope)))
    return dni * max(0.0, cos_inc) + sky_diffuse


def integrate(times, ta, td, cloud_frac, wind, sw, sea, p):
    """Force-Restore march (mirror of RockSurfacePhysics.Integrate)."""
    omega = 2.0 * math.pi / p["tau_day"]
    mu = math.sqrt(p["lam"] * p["rho"] * p["cp"] / (2.0 * omega)) * p["mu_scale"]
    sub = p["substeps"]
    dt = 3600.0 / sub
    two_pi_tau = 2.0 * math.pi / p["tau_day"]
    inv_tau_long = 1.0 / p["tau_long"]
    ts = ta[0]
    tdeep = float(np.mean(ta[: min(24, len(ta))]))
    out = np.empty(len(times))
    for i in range(len(times)):
        e_a = 6.112 * math.exp(17.62 * td[i] / (243.12 + td[i]))
        air_k = ta[i] + 273.15
        e_clear = 1.24 * (e_a / air_k) ** (1.0 / 7.0)
        c = max(0.0, min(1.0, cloud_frac[i]))
        eps_sky = max(0.0, min(1.0, p["lw_clear_k"] * e_clear + p["lw_cloud_k"] * (1.0 - e_clear) * c))
        lw_down = eps_sky * SIGMA * air_k ** 4
        h_conv = 5.7 + 3.8 * max(wind[i], 0.0)
        sea_emit = SIGMA * (sea[i] + 273.15) ** 4 if sea[i] is not None and not np.isnan(sea[i]) else None
        sw_i = max(sw[i], 0.0)
        for _ in range(sub):
            ts_k4 = (ts + 273.15) ** 4
            g = (1.0 - p["albedo"]) * sw_i + p["eps_rock"] * p["f_sky"] * (lw_down - SIGMA * ts_k4)
            if sea_emit is not None:
                g += p["eps_rock"] * (1.0 - p["f_sky"]) * (sea_emit - SIGMA * ts_k4)
            g -= h_conv * (ts - ta[i])
            ts += (g / mu - two_pi_tau * (ts - tdeep)) * dt
            tdeep += (ts - tdeep) * inv_tau_long * dt
        out[i] = ts
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2026-05-30")
    args = ap.parse_args()

    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT ValidTimeUtc, Temperature2m, DewPoint2m, CloudCover, WindSpeed10m,
               ShortwaveRadiation, DirectRadiation, DiffuseRadiation, SoilTemperature0to7cm
        FROM read_parquet('{ERA5_GLOB}', union_by_name = true)
        WHERE ValidTimeUtc BETWEEN TIMESTAMP '{args.start}' AND TIMESTAMP '{args.end}'
          AND Temperature2m IS NOT NULL AND DewPoint2m IS NOT NULL
        ORDER BY ValidTimeUtc
        """
    ).fetch_df()
    print(f"ERA5 rows: {len(df)}  ({df.ValidTimeUtc.min()} .. {df.ValidTimeUtc.max()})")
    if len(df) == 0:
        return

    times = [t.to_pydatetime().replace(tzinfo=timezone.utc) for t in df.ValidTimeUtc]
    ta = df.Temperature2m.to_numpy(float)
    td = df.DewPoint2m.to_numpy(float)
    cloud = np.nan_to_num(df.CloudCover.to_numpy(float), nan=50.0) / 100.0
    wind = np.nan_to_num(df.WindSpeed10m.to_numpy(float), nan=4.0)
    ghi = np.nan_to_num(df.ShortwaveRadiation.to_numpy(float), nan=0.0)
    direct = df.DirectRadiation.to_numpy(float)
    diffuse = df.DiffuseRadiation.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where((direct + diffuse) > 1e-9, direct / (direct + diffuse), 0.0)
    frac = np.nan_to_num(frac, nan=0.0)

    # SST stand-in: no SST in the land ERA5 pull, so use a 10-day trailing
    # mean of air temperature as a smooth proxy of coastal sea temperature
    # (the Atlantic at Sennen tracks slow seasonal air temp within ~1-2degC).
    # Good enough for a margin-distribution shape check; the live pipeline
    # uses real best_match SST.
    kernel = np.ones(240) / 240.0
    sea = np.convolve(ta, kernel, mode="same")

    sw_dt = [t - timedelta(minutes=30) for t in times]
    sol = [solar_position(t, LAT, LON) for t in sw_dt]

    print(f"\nface        margin_p05  p25  p50  greasy%(<= {GREASY_MARGIN_C}C)  condensation%(<=0)")
    for name, aspect in FACES.items():
        sw_face = np.array([
            face_incident_sw(ghi[i], frac[i], sol[i][0], sol[i][1], aspect, SLOPE_DEG, P["f_sky"])
            for i in range(len(times))
        ])
        ts = integrate(times, ta, td, cloud, wind, sw_face, sea, P)
        margin = ts[240:] - td[240:]  # skip 10d spin-up
        greasy = float(np.mean(margin <= GREASY_MARGIN_C) * 100)
        cond = float(np.mean(margin <= 0) * 100)
        p05, p25, p50 = np.percentile(margin, [5, 25, 50])
        print(f"{name:<10}  {p05:9.2f} {p25:5.2f} {p50:5.2f}   {greasy:7.1f}%          {cond:6.1f}%")

    # Reference: the same physics with NO sea term + fSky 1 (Bonehill-style),
    # horizontal SW — shows how much the cliff treatment changes the answer.
    p_ref = dict(P, f_sky=1.0)
    ts_ref = integrate(times, ta, td, cloud, wind, ghi, [None] * len(times), p_ref)
    margin_ref = ts_ref[240:] - td[240:]
    greasy = float(np.mean(margin_ref <= GREASY_MARGIN_C) * 100)
    cond = float(np.mean(margin_ref <= 0) * 100)
    p05, p25, p50 = np.percentile(margin_ref, [5, 25, 50])
    print(f"{'horiz-ref':<10}  {p05:9.2f} {p25:5.2f} {p50:5.2f}   {greasy:7.1f}%          {cond:6.1f}%")

    # Threshold table for choosing Sennen's greasyMarginC: amber-rate at
    # candidate cut points, all-hours and the overview's visible window
    # (06-21 UTC) where the badge is actually seen.
    hours = np.array([t.hour for t in times[240:]])
    visible = (hours >= 6) & (hours < 21)
    print("\namber-rate by greasyMarginC candidate (west face):")
    west_sw2 = np.array([
        face_incident_sw(ghi[i], frac[i], sol[i][0], sol[i][1], FACES["west"], SLOPE_DEG, P["f_sky"])
        for i in range(len(times))
    ])
    m_west = integrate(times, ta, td, cloud, wind, west_sw2, sea, P)[240:] - td[240:]
    print("  threshold   all-hours   visible(06-21Z)")
    for thr in (0.5, 1.0, 1.5, 2.0, 3.0):
        print(f"  {thr:7.1f}   {np.mean(m_west <= thr) * 100:8.1f}%   {np.mean(m_west[visible] <= thr) * 100:8.1f}%")
    print(f"  cond(<=0)  {np.mean(m_west <= 0) * 100:8.1f}%   {np.mean(m_west[visible] <= 0) * 100:8.1f}%")

    # ERA5 soil rail for very rough plausibility (vegetated cell, weak proxy).
    soil = df.SoilTemperature0to7cm.to_numpy(float)
    ok = ~np.isnan(soil[240:])
    if ok.any():
        west_sw = np.array([
            face_incident_sw(ghi[i], frac[i], sol[i][0], sol[i][1], FACES["west"], SLOPE_DEG, P["f_sky"])
            for i in range(len(times))
        ])
        ts_west = integrate(times, ta, td, cloud, wind, west_sw, sea, P)
        corr = np.corrcoef(ts_west[240:][ok], soil[240:][ok])[0, 1]
        print(f"\nwest-face Ts vs ERA5 soil(0-7cm) correlation: {corr:.3f}  (weak proxy, direction only)")


if __name__ == "__main__":
    main()
