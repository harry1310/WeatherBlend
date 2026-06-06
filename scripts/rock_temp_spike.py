"""
Rock surface temperature — P0 offline spike (Force-Restore).

Estimates granite boulder surface temperature `Ts` at Bonehill from ERA5
hourly forcing (the best-estimate "what actually happened" series, so this
isolates the physics from forecast error), then derives the condensation
margin `m = Ts - Td` and flags condensation when m <= 0.

There is NO observed rock-surface-temperature truth, so this spike CANNOT
prove accuracy. Its job is to show the model is physically CREDIBLE:
  1. physical-behaviour checks (diurnal swing, day Ts>Ta, clear-calm-night
     Ts<Ta, cloudy/windy night Ts~=Ta),
  2. directional sensitivity (perturb cloud/wind/SW -> correct response sign),
  3. cloud-quality sensitivity of the condensation flag (the key question:
     does a cloud-blender-sized error change the flag much?),
  4. a weak independent rail: condensation nights vs ERA5 near-saturation.
Absolute accuracy is deferred to on-site calibration (IR gun / logger); an
optional CSV hook (--ir-csv) overlays any real spot readings when available.

Run (WP venv has numpy/pandas/duckdb/matplotlib):
  ../WeatherProbabilistic/.venv/Scripts/python.exe scripts/rock_temp_spike.py
"""
from __future__ import annotations
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 chokes on Δ/—
except Exception:
    pass
import os
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W/m^2/K^4

# ----------------------------- granite / model params (the calibratable knobs)
PARAMS = dict(
    albedo=0.30,        # granite shortwave albedo
    eps_rock=0.95,      # longwave emissivity
    lam=3.0,            # thermal conductivity W/m/K
    rho=2650.0,         # density kg/m^3
    cp_rock=790.0,      # specific heat J/kg/K
    tau_day=86400.0,    # diurnal restore period s
    tau_long=86400.0,   # deep-reservoir drift timescale s (canonical force-restore = ~1 day)
    f_sky=1.0,          # sky-view factor (1 = fully exposed boulder top; <1 = shaded/boulder field)
    lw_clear_k=1.0,     # clear-sky emissivity scale (GFS calibration 2026-06-04: ~1.016, keep 1.0)
    lw_cloud_k=0.54,    # cloud-enhancement scale. RECALIBRATED 2026-06-04 from 1.0 -> 0.54 by
                        # least-squares fit of LW-down to GFS DLWRF: the original full-to-1.0
                        # enhancement over-stated cloudy-sky LW by ~22 W/m2 (warm bias -> under-
                        # predicted condensation). 0.54 removes the bias (resid -0.3 W/m2) at full
                        # coverage, no GFS dependency. Set 1.0 to reproduce raw Brunt.
    mu_scale=0.3,       # multiplier on canonical skin heat capacity (THE main knob).
                        # 0.3 (~3cm radiating skin) is the P0 default: the sweep showed
                        # this is where clear-calm nights cool below air (-1.7C), sunny
                        # days run +10C, overcast nights ~coupled. The true value is a
                        # calibration target for on-site IR/logger data (see plan P2).
    substeps=6,         # ODE sub-steps per hour (stability)
)

def mu_from_props(p):
    """Canonical force-restore areal heat capacity Cg = sqrt(lam*rho*cp/(2*omega)),
    omega = 2*pi/tau_day (Deardorff 1978 / Hu & Islam 1995). Scaled by mu_scale:
    the effective RADIATING skin is thinner than the full diurnal-damping depth,
    so mu_scale<1 lets the surface respond faster to the nighttime longwave
    deficit (i.e. actually cool below air). Returns (mu, equivalent depth)."""
    omega = 2 * np.pi / p["tau_day"]
    cg = np.sqrt(p["lam"] * p["rho"] * p["cp_rock"] / (2 * omega)) * p["mu_scale"]
    depth = cg / (p["rho"] * p["cp_rock"])
    return cg, depth

# ----------------------------- forcing
def load_forcing(era5_glob, start, end):
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT ValidTimeUtc AS t, Temperature2m AS ta, DewPoint2m AS td,
               RelativeHumidity2m AS rh, CloudCover AS cc, WindSpeed10m AS v,
               ShortwaveRadiation AS sw, SoilTemperature0to7cm AS soil
        FROM read_parquet('{era5_glob}', hive_partitioning=false, union_by_name=true)
        WHERE Temperature2m IS NOT NULL AND DewPoint2m IS NOT NULL
          AND CloudCover IS NOT NULL AND WindSpeed10m IS NOT NULL
          AND ShortwaveRadiation IS NOT NULL
          AND ValidTimeUtc >= TIMESTAMP '{start}' AND ValidTimeUtc < TIMESTAMP '{end}'
        ORDER BY t
    """).df()
    con.close()
    df["t"] = pd.to_datetime(df["t"])
    # contiguous hourly index; interpolate any small gaps
    df = df.set_index("t").asfreq("1h")
    df[["ta","td","rh","cc","v","sw"]] = df[["ta","td","rh","cc","v","sw"]].interpolate(limit=6)
    # soil is the (damped, biased) validation rail only — interpolate but DON'T
    # let its gaps drop forcing rows (subset the dropna to the core forcing).
    if "soil" in df.columns:
        df["soil"] = df["soil"].interpolate(limit=6)
    df = df.dropna(subset=["ta","td","rh","cc","v","sw"]).reset_index()
    df["cfrac"] = (df["cc"].clip(0, 100) / 100.0)
    return df

# ----------------------------- NWP longwave (GFS DLWRF) as a substitute for the
# Brunt/cloud LW parameterisation. Best-estimate "actual" LW-down = the shortest
# available lead per valid hour from the GFS exact archive (closest to analysis).
# Only GFS carries DownwardLongwaveRadiation (OM/GEFS/ECMWF do not). NEEDS the
# gfs exact-archive backfill to have landed.
def load_nwp_lw(fc_glob, model, start, end):
    con = duckdb.connect()
    df = con.execute(f"""
        WITH ranked AS (
          SELECT ValidTimeUtc AS t, DownwardLongwaveRadiation AS lw, LeadHours,
                 ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc
                                    ORDER BY LeadHours ASC, RunTimeUtc DESC) rn
          FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
          WHERE Model='{model}' AND DownwardLongwaveRadiation IS NOT NULL
            AND ValidTimeUtc >= TIMESTAMP '{start}' AND ValidTimeUtc < TIMESTAMP '{end}'
        )
        SELECT t, lw, LeadHours AS lead FROM ranked WHERE rn=1 ORDER BY t
    """).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t")
    return df["lw"], df["lead"]

def vapour_pressure_hpa(td_c):
    return 6.112 * np.exp(17.62 * td_c / (243.12 + td_c))

def clear_sky_emissivity(td_c, ta_k):
    """Brutsaert (1975) clear-sky emissivity."""
    e = vapour_pressure_hpa(td_c)
    return 1.24 * (e / ta_k) ** (1.0 / 7.0)

# ----------------------------- Force-Restore integrator
def integrate(df, p, cloud_override=None, lw_down_override=None):
    mu, _ = mu_from_props(p)
    n = len(df)
    ta = df["ta"].to_numpy()
    td = df["td"].to_numpy()
    sw = df["sw"].to_numpy().clip(min=0)
    v  = df["v"].to_numpy().clip(min=0)
    cf = (cloud_override if cloud_override is not None else df["cfrac"].to_numpy()).clip(0, 1)

    ta_k = ta + 273.15
    eps_clear = clear_sky_emissivity(td, ta_k)
    # Recalibrated emissivity (GFS-fit cloud-enhancement scale; see PARAMS lw_cloud_k).
    eps_sky = np.clip(p.get("lw_clear_k", 1.0) * eps_clear
                      + p.get("lw_cloud_k", 1.0) * (1.0 - eps_clear) * cf, 0.0, 1.0)
    lw_down = eps_sky * SIGMA * ta_k ** 4
    # Substitute measured NWP LW-down where supplied; Brunt fills any gaps so the
    # integration stays continuous. (Task: does real LW beat the cloud param?)
    if lw_down_override is not None:
        ov = np.asarray(lw_down_override, dtype=float)
        lw_down = np.where(np.isnan(ov), lw_down, ov)
    h_conv = 5.7 + 3.8 * v  # McAdams, W/m^2/K

    ts = np.empty(n); tdeep = np.empty(n)
    # spin-up initial state: surface = air, deep = first-day mean air temp
    ts_c = ta[0]
    tdeep_c = float(np.nanmean(ta[:min(24, n)]))
    sub = p["substeps"]; dt = 3600.0 / sub
    a = p["albedo"]; er = p["eps_rock"]; fs = p["f_sky"]
    two_pi_tau = 2 * np.pi / p["tau_day"]
    inv_tau_long = 1.0 / p["tau_long"]

    for i in range(n):
        for _ in range(sub):
            ts_k = ts_c + 273.15
            sw_abs = (1 - a) * sw[i]
            lw_net = er * fs * (lw_down[i] - SIGMA * ts_k ** 4)
            h_loss = h_conv[i] * (ts_c - ta[i])
            g_net = sw_abs + lw_net - h_loss
            dts = g_net / mu - two_pi_tau * (ts_c - tdeep_c)
            dtd = (ts_c - tdeep_c) * inv_tau_long
            ts_c += dts * dt
            tdeep_c += dtd * dt
        ts[i] = ts_c
        tdeep[i] = tdeep_c

    out = df.copy()
    out["ts"] = ts
    out["tdeep"] = tdeep
    out["lw_down"] = lw_down                    # effective LW-down used this run
    out["margin"] = out["ts"] - out["td"]      # <=0 => condensation
    out["dts_minus_ta"] = out["ts"] - out["ta"]
    return out

# ----------------------------- masks
def masks(df):
    night = df["sw"] < 5.0
    day_sun = df["sw"] > 300.0
    clear = df["cfrac"] < 0.25
    overcast = df["cfrac"] > 0.75
    calm = df["v"] < 2.0
    windy = df["v"] > 6.0
    return dict(night=night, day_sun=day_sun, clear=clear, overcast=overcast,
                calm=calm, windy=windy)

# ----------------------------- physical-behaviour checks
def physical_checks(df, spinup_hours=48):
    d = df.iloc[spinup_hours:].copy()
    m = masks(d)
    checks = []
    def add(name, cond, detail):
        checks.append((name, bool(cond), detail))

    cn = d[m["night"] & m["clear"] & m["calm"]]["dts_minus_ta"]
    add("clear-calm night: rock COOLER than air (radiative cooling)",
        cn.mean() < -0.2, f"mean Ts-Ta = {cn.mean():+.2f} C  (n={len(cn)})")

    sun = d[m["day_sun"]]["dts_minus_ta"]
    add("sunny day: rock WARMER than air (solar heating)",
        sun.mean() > 1.0, f"mean Ts-Ta = {sun.mean():+.2f} C  (n={len(sun)})")

    on = d[m["night"] & m["overcast"]]["dts_minus_ta"]
    add("overcast night: rock ~ air (well-coupled, less cooling than clear)",
        abs(on.mean()) < abs(cn.mean()) if len(cn) and len(on) else False,
        f"mean Ts-Ta = {on.mean():+.2f} C vs clear-night {cn.mean():+.2f} C")

    wn = d[m["night"] & m["clear"] & m["windy"]]["dts_minus_ta"]
    add("windy clear night: less cooling than calm clear night (convective coupling)",
        (wn.mean() > cn.mean()) if len(wn) and len(cn) else False,
        f"windy {wn.mean():+.2f} C vs calm {cn.mean():+.2f} C  (n_windy={len(wn)})")

    # diurnal amplitude: mean daily (max-min) of Ts vs Ta
    g = d.set_index("t")
    amp_ts = g["ts"].resample("1D").agg(lambda x: x.max()-x.min()).mean()
    amp_ta = g["ta"].resample("1D").agg(lambda x: x.max()-x.min()).mean()
    add("diurnal swing of rock WIDER than air",
        amp_ts > amp_ta, f"mean daily range Ts={amp_ts:.1f} C vs Ta={amp_ta:.1f} C")
    return checks

# ----------------------------- directional sensitivity
def sensitivity(df, p, spinup_hours=48):
    base = integrate(df, p)
    night = base["sw"] < 5.0
    res = {}
    # +cloud (0.3 toward overcast) -> nighttime should WARM (less LW loss)
    cl = np.clip(df["cfrac"].to_numpy() + 0.3, 0, 1)
    res["cloud +0.3 -> night Ts"] = (integrate(df, p, cloud_override=cl)["ts"] - base["ts"])[night].iloc[spinup_hours:].mean()
    # wind x2 -> Ts toward Ta (smaller |Ts-Ta|); report night |Ts-Ta| change
    p2 = dict(p);
    dfw = df.copy(); dfw["v"] = df["v"] * 2.0
    bw = integrate(dfw, p)
    res["wind x2 -> night |Ts-Ta|"] = (bw["dts_minus_ta"].abs() - base["dts_minus_ta"].abs())[night].iloc[spinup_hours:].mean()
    # SW x1.2 -> daytime should WARM
    dfs = df.copy(); dfs["sw"] = df["sw"] * 1.2
    bs = integrate(dfs, p)
    daym = df["sw"] > 300
    res["SW x1.2 -> day Ts"] = (bs["ts"] - base["ts"])[daym].iloc[spinup_hours:].mean()
    return res

# ----------------------------- cloud-quality sensitivity of the condensation flag
def cloud_flag_sensitivity(df, p, spinup_hours=48, seed=0):
    """Degrade cloud by a blender-sized error (~15% abs MAE + small bias) and
    measure how many condensation NIGHTS flip vs the ERA5-cloud baseline.
    Answers: does cloud-blend quality actually matter for the flag?"""
    rng = np.random.default_rng(seed)
    base = integrate(df, p).iloc[spinup_hours:].copy()
    # nightly condensation flag (any overnight hour with margin<=0)
    def nightly_flag(o):
        o = o.copy()
        o["night"] = o["sw"] < 5.0
        o["noct"] = (o["t"] - pd.Timedelta(hours=12)).dt.date  # group dusk..dawn
        nf = o[o["night"]].groupby("noct")["margin"].min()
        return (nf <= 0.0)
    base_flag = nightly_flag(base)

    flips = []
    for s in range(5):
        rng = np.random.default_rng(seed + s)
        noise = rng.normal(0, 18.0, len(df)) / 100.0  # ~15-18% abs cloud MAE
        bias = 0.05 * (1 if s % 2 else -1)             # small systematic bias
        cl = np.clip(df["cfrac"].to_numpy() + noise + bias, 0, 1)
        deg = integrate(df, p, cloud_override=cl).iloc[spinup_hours:].copy()
        deg_flag = nightly_flag(deg)
        j = pd.concat([base_flag.rename("base"), deg_flag.rename("deg")], axis=1).dropna()
        flips.append((j["base"] != j["deg"]).mean())
    return base_flag.mean(), float(np.mean(flips)), float(np.std(flips))

# ----------------------------- plots
def make_plots(df, p, outdir, spinup_hours=48):
    os.makedirs(outdir, exist_ok=True)
    base = integrate(df, p).iloc[spinup_hours:].copy()

    # 1. clearest dry week — time series Ts/Ta/Td + condensation crossings
    g = base.set_index("t")
    weekly_cloud = g["cfrac"].resample("7D").mean()
    wk_start = weekly_cloud.idxmin()
    wk = base[(base["t"] >= wk_start) & (base["t"] < wk_start + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wk["t"], wk["ta"], label="air temp Ta", color="tab:blue", lw=1.2)
    ax.plot(wk["t"], wk["ts"], label="rock surface Ts", color="tab:red", lw=1.5)
    ax.plot(wk["t"], wk["td"], label="dew point Td", color="tab:green", lw=1.0, ls="--")
    cond = wk[wk["margin"] <= 0]
    ax.scatter(cond["t"], cond["ts"], s=14, color="purple", zorder=5, label="condensation (Ts<=Td)")
    ax.set_title(f"Rock surface temp — clearest week from {wk_start.date()}")
    ax.set_ylabel("deg C"); ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout(); fig.savefig(f"{outdir}/01_clear_week.png", dpi=110); plt.close(fig)

    # 2. night-time Ts-Ta vs cloud fraction (binned)
    night = base[base["sw"] < 5.0].copy()
    night["cbin"] = (night["cfrac"] * 10).round() / 10
    b = night.groupby("cbin")["dts_minus_ta"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(b.index, b.values, "o-", color="tab:red")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("cloud fraction"); ax.set_ylabel("night mean Ts - Ta (C)")
    ax.set_title("Nighttime rock cooling vs cloud (should rise toward 0 as cloud increases)")
    fig.tight_layout(); fig.savefig(f"{outdir}/02_night_cooling_vs_cloud.png", dpi=110); plt.close(fig)

    # 3. Ts vs Ta scatter coloured by SW
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(base["ta"], base["ts"], c=base["sw"], s=3, cmap="viridis", alpha=0.4)
    lim = [base["ta"].min()-2, base["ta"].max()+8]
    ax.plot(lim, lim, "k--", lw=0.8, label="Ts=Ta")
    ax.set_xlabel("air temp Ta (C)"); ax.set_ylabel("rock surface Ts (C)")
    ax.set_title("Ts vs Ta (colour = shortwave W/m2)"); ax.legend()
    fig.colorbar(sc, label="SW W/m2"); fig.tight_layout()
    fig.savefig(f"{outdir}/03_ts_vs_ta.png", dpi=110); plt.close(fig)
    return f"{outdir}/01_clear_week.png"

# ----------------------------- IR-gun overlay hook (optional)
def ir_compare(df, p, ir_csv, spinup_hours=48):
    if not ir_csv or not os.path.exists(ir_csv):
        return None
    ir = pd.read_csv(ir_csv, parse_dates=["timestamp"])  # cols: timestamp, rock_temp_c
    base = integrate(df, p)[["t", "ts"]].rename(columns={"t": "timestamp"})
    m = pd.merge_asof(ir.sort_values("timestamp"), base.sort_values("timestamp"),
                      on="timestamp", tolerance=pd.Timedelta("30min"), direction="nearest").dropna()
    if len(m) == 0:
        return None
    mae = (m["ts"] - m["rock_temp_c"]).abs().mean()
    bias = (m["ts"] - m["rock_temp_c"]).mean()
    return dict(n=len(m), mae=mae, bias=bias)

# ----------------------------- main
def compare_lw(df, p, nwp_lw, nwp_lead, spinup_hours=48):
    """Brunt/cloud LW vs measured NWP (GFS DLWRF) LW: quantify how much LW-down,
    Ts and the condensation flag move, and whether daily-mean Ts tracks the ERA5
    soil-temp rail more tightly in a DIRECTIONAL sense (day-to-day change + anomaly
    correlation). Absolute Ts-vs-soil offset is meaningless (rock != damped soil),
    so we score movement linkage only. No rock truth → this informs, not decides."""
    if len(nwp_lw) == 0:
        print("  no NWP DLWRF rows found — has the gfs exact-archive backfill landed?")
        return
    nwp = nwp_lw.reindex(df["t"]).to_numpy()
    lead = nwp_lead.reindex(df["t"]).to_numpy()
    cov = float(np.isfinite(nwp).mean())
    print(f"  NWP LW coverage of forcing hours: {cov:.1%}"
          + (f"   (median lead {np.nanmedian(lead[np.isfinite(lead)]):.0f}h)" if cov > 0 else ""))
    if cov < 0.2:
        print("  !! coverage too low to compare — aborting (backfill incomplete?).")
        return

    p_raw = {**p, "lw_clear_k": 1.0, "lw_cloud_k": 1.0}  # show RAW Brunt (model default is now calibrated)
    base = integrate(df, p_raw)                        # raw Brunt/cloud LW
    run  = integrate(df, p_raw, lw_down_override=nwp)  # NWP LW (raw Brunt fills gaps)
    b = base.iloc[spinup_hours:].reset_index(drop=True)
    r = run.iloc[spinup_hours:].reset_index(drop=True)
    nwp_s = nwp[spinup_hours:]
    m = np.isfinite(nwp_s)

    # 1) LW-down itself: NWP vs Brunt, where NWP is present
    brunt_lw = b["lw_down"].to_numpy()
    diff = nwp_s[m] - brunt_lw[m]
    print(f"  LW-down NWP vs Brunt: bias {diff.mean():+.1f} W/m2  rmse {np.sqrt((diff**2).mean()):.1f}  "
          f"corr {np.corrcoef(nwp_s[m], brunt_lw[m])[0,1]:+.3f}")

    # 2) Ts movement when the LW source is swapped
    night = (b["sw"] < 5.0).to_numpy()
    dts = (r["ts"] - b["ts"])
    print(f"  Ts(NWP)-Ts(Brunt): all {dts.mean():+.2f}C  night {dts[night].mean():+.2f}C  max|Δ| {dts.abs().max():.2f}C")

    # 3) condensation-flag flips (nightly: any night hour with margin<=0)
    bb = b.set_index("t"); rr = r.set_index("t")
    bf = (bb["margin"] <= 0)[bb["sw"] < 5.0].resample("1D").max()
    rf = (rr["margin"] <= 0)[rr["sw"] < 5.0].resample("1D").max()
    j = pd.concat([bf.rename("br"), rf.rename("nwp")], axis=1).dropna()
    if len(j):
        print(f"  condensation nights: Brunt {j['br'].mean():.1%}  NWP {j['nwp'].mean():.1%}  "
              f"flag FLIPS {(j['br'] != j['nwp']).mean():.1%} of nights")

    # 4) ERA5 soil-temp rail — DIRECTIONAL only (movement, not absolute level)
    if "soil" in bb.columns and bb["soil"].notna().any():
        soil_d = bb["soil"].resample("1D").mean()
        def dcorr(series):  # day-to-day change correlation
            x = series.resample("1D").mean().diff(); y = soil_d.diff()
            k = x.notna() & y.notna(); return x[k].corr(y[k]) if k.any() else float("nan")
        def acorr(series):  # 7d-anomaly correlation
            a = series.resample("1D").mean()
            xa = a - a.rolling(7, center=True, min_periods=3).mean()
            ya = soil_d - soil_d.rolling(7, center=True, min_periods=3).mean()
            k = xa.notna() & ya.notna(); return xa[k].corr(ya[k]) if k.any() else float("nan")
        print(f"  soil-temp rail (directional): day-to-day Δ corr  Brunt {dcorr(bb['ts']):+.3f}  NWP {dcorr(rr['ts']):+.3f}")
        print(f"                                7d-anomaly  corr  Brunt {acorr(bb['ts']):+.3f}  NWP {acorr(rr['ts']):+.3f}")
        print("    (higher = Ts movement tracks soil better; absolute offset ignored)")
    else:
        print("  soil-temp rail unavailable (SoilTemperature0to7cm null — era5 soil backfill landed?)")


def calibrate_lw(df, p, nwp_lw, spinup_hours=48):
    """Remove the Brunt warm bias by least-squares fitting the two LW terms
    (clear-sky + cloud-enhancement) to GFS DLWRF, then re-run the rock model with
    the RECALIBRATED LW (full coverage — no GFS dependency) and compare condensation
    + soil-rail tracking vs raw Brunt and vs the (partial-coverage) NWP run.

    lw = a*clear + b*cloud   where  clear = eps_clear*sig*Ta^4,
                                    cloud = (1-eps_clear)*cf*sig*Ta^4
    (a=b=1 reproduces the current parameterisation exactly)."""
    if len(nwp_lw) == 0:
        print("  no NWP DLWRF — cannot calibrate."); return
    ta = df["ta"].to_numpy(); td = df["td"].to_numpy(); cf = df["cfrac"].to_numpy().clip(0, 1)
    ta_k = ta + 273.15
    eps_clear = clear_sky_emissivity(td, ta_k)
    clear = eps_clear * SIGMA * ta_k ** 4
    cloud = (1.0 - eps_clear) * cf * SIGMA * ta_k ** 4
    nwp = nwp_lw.reindex(df["t"]).to_numpy()
    m = np.isfinite(nwp)
    if m.sum() < 500:
        print("  NWP coverage too low to calibrate."); return

    a, b = np.linalg.lstsq(np.column_stack([clear[m], cloud[m]]), nwp[m], rcond=None)[0]
    # recalibrated LW, clipped to the black-body ceiling (physical), full coverage
    eps_recal = np.clip(a * eps_clear + b * (1.0 - eps_clear) * cf, 0.0, 1.0)
    lw_recal = eps_recal * SIGMA * ta_k ** 4
    resid = lw_recal[m] - nwp[m]
    print(f"  fit  lw = {a:.3f}*clear + {b:.3f}*cloud   (was 1.000 / 1.000)")
    print(f"  recalibrated LW vs NWP: bias {resid.mean():+.1f} W/m2  rmse {np.sqrt((resid**2).mean()):.1f}  "
          f"(raw Brunt bias was +22.0)")

    p_raw = {**p, "lw_clear_k": 1.0, "lw_cloud_k": 1.0}  # baseline = RAW Brunt (model default is now calibrated)
    base = integrate(df, p_raw)                          # raw Brunt
    recal = integrate(df, p_raw, lw_down_override=lw_recal)  # recalibrated (full coverage)
    run = integrate(df, p_raw, lw_down_override=nwp)         # NWP where present, raw Brunt elsewhere

    def cond_rate(o):
        oo = o.iloc[spinup_hours:].set_index("t")
        return (oo["margin"] <= 0)[oo["sw"] < 5.0].resample("1D").max().mean()
    print(f"  condensation nights:  Brunt {cond_rate(base):.1%}   recal {cond_rate(recal):.1%}   NWP {cond_rate(run):.1%}")

    if "soil" in df.columns and df["soil"].notna().any():
        soil_d = base.iloc[spinup_hours:].set_index("t")["soil"].resample("1D").mean()
        def dcorr(o):
            x = o.iloc[spinup_hours:].set_index("t")["ts"].resample("1D").mean().diff(); y = soil_d.diff()
            k = x.notna() & y.notna(); return x[k].corr(y[k])
        print(f"  soil-rail day-to-day Δ corr:  Brunt {dcorr(base):+.3f}   recal {dcorr(recal):+.3f}   NWP {dcorr(run):+.3f}")
    print(f"  -> recalibrated Brunt keeps 100% coverage + no GFS dependency; check it captures the NWP gain.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-glob", default="data/truth/era5/location=bonehill_rocks/**/*.parquet")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--outdir", default="data/reports/rock_temp_spike")
    ap.add_argument("--ir-csv", default=None, help="optional CSV: timestamp,rock_temp_c")
    ap.add_argument("--mu-scale", type=float, default=PARAMS["mu_scale"],
                    help="multiplier on canonical skin heat capacity (smaller = cools faster)")
    ap.add_argument("--tau-long-days", type=float, default=PARAMS["tau_long"]/86400.0,
                    help="deep-reservoir drift timescale in days")
    ap.add_argument("--nwp-lw-glob", default=None,
                    help="forecasts glob to pull NWP DLWRF from (e.g. "
                         "data/forecasts/location=bonehill_rocks/**/*.parquet); "
                         "enables the Brunt-vs-NWP LW comparison")
    ap.add_argument("--nwp-lw-model", default="gfs_ncep",
                    help="model id carrying DownwardLongwaveRadiation (only GFS does)")
    ap.add_argument("--lw-clear-k", type=float, default=PARAMS["lw_clear_k"],
                    help="clear-sky LW emissivity scale (GFS-calibrated default)")
    ap.add_argument("--lw-cloud-k", type=float, default=PARAMS["lw_cloud_k"],
                    help="cloud-enhancement LW scale; 1.0 = raw Brunt, 0.54 = GFS-calibrated default")
    args = ap.parse_args()

    p = dict(PARAMS)
    p["mu_scale"] = args.mu_scale
    p["tau_long"] = args.tau_long_days * 86400.0
    p["lw_clear_k"] = args.lw_clear_k
    p["lw_cloud_k"] = args.lw_cloud_k
    mu, depth = mu_from_props(p)
    print("=" * 78)
    print("ROCK SURFACE TEMPERATURE — P0 Force-Restore spike (ERA5 forcing, Bonehill)")
    print("=" * 78)
    print(f"window {args.start} .. {args.end}")
    print(f"granite: albedo={p['albedo']} eps={p['eps_rock']} lam={p['lam']} "
          f"rho={p['rho']} cp={p['cp_rock']}")
    print(f"derived diurnal active depth = {depth*100:.1f} cm   areal heat cap mu = {mu:.3e} J/m2/K")
    print(f"tau_long={p['tau_long']/86400:.0f} d   f_sky={p['f_sky']}")
    print()

    df = load_forcing(args.era5_glob, args.start, args.end)
    print(f"loaded {len(df)} contiguous hourly ERA5 rows "
          f"({df['t'].iloc[0]} .. {df['t'].iloc[-1]})")
    print()

    base_full = integrate(df, p)

    print("--- 1. PHYSICAL-BEHAVIOUR CHECKS " + "-" * 44)
    npass = 0
    for name, ok, detail in physical_checks(base_full):
        flag = "PASS" if ok else "FAIL"
        npass += ok
        print(f"  [{flag}] {name}\n         {detail}")
    print()

    print("--- 2. DIRECTIONAL SENSITIVITY (signs must be physical) " + "-" * 21)
    s = sensitivity(df, p)
    print(f"  cloud +0.3  -> night Ts change : {s['cloud +0.3 -> night Ts']:+.2f} C  (expect >0, warming)")
    print(f"  wind  x2    -> night |Ts-Ta|    : {s['wind x2 -> night |Ts-Ta|']:+.2f} C  (expect <0, toward air)")
    print(f"  SW    x1.2  -> day Ts change    : {s['SW x1.2 -> day Ts']:+.2f} C  (expect >0, warming)")
    print()

    print("--- 3. CLOUD-QUALITY SENSITIVITY OF THE CONDENSATION FLAG " + "-" * 19)
    base_rate, flip_mean, flip_sd = cloud_flag_sensitivity(df, p)
    print(f"  baseline condensation-night rate (ERA5 cloud) : {base_rate:.1%}")
    print(f"  nights whose flag FLIPS under ~15-18% cloud error: {flip_mean:.1%} +/- {flip_sd:.1%}")
    print("  -> low flip rate = flag robust to cloud-blend error; high = cloud quality matters")
    print()

    print("--- 4. WEAK RAIL: condensation vs ERA5 near-saturation " + "-" * 22)
    base = integrate(df, p).iloc[48:].copy()
    base["night"] = base["sw"] < 5.0
    base["noct"] = (base["t"] - pd.Timedelta(hours=12)).dt.date
    nf = base[base["night"]].groupby("noct").agg(cond=("margin", lambda x: (x <= 0).any()),
                                                 rh_max=("rh", "max"))
    overlap = (nf["cond"] & (nf["rh_max"] >= 97)).sum()
    print(f"  condensation nights: {nf['cond'].sum()} / {len(nf)}   "
          f"of which ERA5 air also hit RH>=97%: {overlap} "
          f"({overlap/max(1,nf['cond'].sum()):.0%})")
    print("  (rock can sweat below air saturation, so partial overlap is expected, not 100%)")
    print()

    if args.nwp_lw_glob:
        print("--- 5. NWP LW SUBSTITUTION (Brunt/cloud vs GFS DLWRF) " + "-" * 23)
        nwp_lw, nwp_lead = load_nwp_lw(args.nwp_lw_glob, args.nwp_lw_model, args.start, args.end)
        compare_lw(df, p, nwp_lw, nwp_lead)
        print()
        print("--- 6. RECALIBRATE BRUNT LW to GFS (remove warm bias, full coverage) " + "-" * 8)
        calibrate_lw(df, p, nwp_lw)
        print()

    plot1 = make_plots(df, p, args.outdir)
    print(f"--- plots -> {args.outdir}/  (01_clear_week, 02_night_cooling_vs_cloud, 03_ts_vs_ta)")

    ir = ir_compare(df, p, args.ir_csv)
    if ir:
        print(f"\n--- IR-gun overlay: n={ir['n']}  MAE={ir['mae']:.2f} C  bias={ir['bias']:+.2f} C")
    else:
        print("\n--- IR-gun overlay: no --ir-csv supplied (absolute accuracy uncalibrated)")

    print()
    print("=" * 78)
    print(f"VERDICT: {npass}/5 physical checks passed. "
          "P0 establishes CREDIBILITY, not accuracy — absolute level needs on-site truth.")
    print("=" * 78)

if __name__ == "__main__":
    main()
