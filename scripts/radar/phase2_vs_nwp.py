#!/usr/bin/env python
"""Phase 2 — radar nowcast vs OUR NWP vs persistence, scored on the EA gauge (FAST, +1h only).

The decision question: does radar advection beat our NWP's own short-lead forecast at the only
lead where it can — +1h? At issue time T = H:00 (top of the hour):
  * RADAR    : advect the radar field forward into the +1h target hour, read the gauge box.
  * NWP      : the exact-runtime ensemble-mean precip for the +1h target hour (leak-safe).
  * PERSIST  : the gauge's radar wet-state at the issue hour, held (Eulerian).
All scored against the EA gauge's hourly wet/dry. We sweep the neighbourhood box size to check the
+1h radar edge is real, not an artifact of how big a box we let radar match the gauge over.

Speed design (so re-runs are seconds, not ~an hour):
  * +1h lead ONLY (the verdict turns on it; +3/+6 already lose decisively).
  * the optical flow is computed on a +/-HW km WINDOW around the gauge (big enough to keep every
    realistic +1h inflow), and we only REMAP the small (2*HALF+1)^2 box at the gauge — never the
    whole field. No inflow is lost; we just stop advecting pixels we never read.
  * the one slow optical-flow pass is CACHED: it writes the advected gauge box (+ matched NWP and
    truth) per issue-time to a tiny npz. Every box-size / threshold sweep then reads the cache.

  python scripts/radar/phase2_vs_nwp.py [LOCATION] [--rebuild]   # default bonehill_rocks
"""
import sys, glob, os
import numpy as np, cv2
import duckdb
from collections import defaultdict
from pyproj import Transformer

FIELD = "data/radar/extracted/field"
FC = "data/forecasts"
RAIN = "data/truth/rainfall"
CACHE_DIR = "data/radar/cache"
LEAD = 1                            # hours — the ONLY lead we test now
LEADS_H = [LEAD]                    # (load_nwp_exact wants a list)
WET = 0.1                          # mm/hr (rate) and mm (hourly accumulation) threshold
EVENT_WETPX = 500                  # region rain-system gate (full-field, same as Phase 1/v1)
RADII = [0, 1, 2, 3]               # neighbourhood radii (1km cells) for the box-size sweep -> 1,3,5,7 km boxes
HW = 150                           # half-width (1km cells) of the flow window around the gauge.
                                   # +1h accumulation reaches +1h45 from issue; ~150km covers inflow
                                   # up to ~85 km/h — beyond that advection is unreliable anyway.
HALF = max(RADII)                  # we only need the gauge box out to the widest neighbourhood
# primary gauge per location (the truth) + its EA-Hydrology lat/lon so we read the radar AT THE
# GAUGE pixel (co-located with the truth) — not the climbing-site pixel km away (the Phase-2 v1
# confound). Persistence is then radar-held at that pixel (Eulerian), the deployable baseline.
GAUGE = {"bonehill_rocks": "Bellever Dartmoor", "membury_devon": "Chards Snowdon Hill",
         "sennen_cove": "Trengwainton"}   # EA Hydrology gauge, ~10km E of Sennen (dodges the WeatherLink path)
GAUGE_COORDS = {"Bellever Dartmoor": (50.582381, -3.898151),
                "Trengwainton": (50.12719, -5.568903)}  # EA gauge ~10km E of Sennen; extend per location as fetched

_TF = Transformer.from_crs(4326, 27700, always_xy=True)  # WGS84 -> OSGB (the NIMROD grid CRS)


def gauge_pixel(d, lat, lon):
    """(row, col) of a lat/lon in the SW-box field, via the npz georeference."""
    y0, x0, dxy = float(d["y0"]), float(d["x0"]), float(d["dxy"])
    r0, c0 = float(d["r0"]), float(d["c0"])
    e, n = _TF.transform(lon, lat)
    return int(round((y0 - n) / dxy - r0)), int(round((e - x0) / dxy - c0))


def to_u8(R):
    x = 10 * np.log10(np.clip(R, 0.05, None)); x[R < WET] = -15.0
    return np.clip((x + 15.0) / 45.0 * 255.0, 0, 255).astype(np.uint8)


def flow_halfres(prev_u8, cur_u8):
    """Farneback at HALF resolution then upscaled — ~4x cheaper, full spatial extent kept (no inflow
    lost). Precip motion is smooth, so coarsening the flow grid costs almost no accuracy. The /3 turns
    the 15-min (3-frame) flow into per-5-min; the *2 restores full-res pixel magnitude after upscaling."""
    ph, ch = cv2.pyrDown(prev_u8), cv2.pyrDown(cur_u8)
    fl = cv2.calcOpticalFlowFarneback(ph, ch, None, 0.5, 4, 21, 3, 5, 1.2, 0) / 3.0
    return cv2.resize(fl, (cur_u8.shape[1], cur_u8.shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0


def advect_box(cur, flow, steps, gr, gc, half):
    """Advect `cur` forward `steps` and return ONLY the (2*half+1)^2 box at (gr,gc).
    Source pixels may sit anywhere in `cur` (inflow up to `steps*flow` away), so the field
    must be the full window — we just restrict the OUTPUT to the gauge box."""
    rows = np.arange(gr - half, gr + half + 1)
    cols = np.arange(gc - half, gc + half + 1)
    cc, rr = np.meshgrid(cols, rows)
    fsub = flow[gr - half:gr + half + 1, gc - half:gc + half + 1, :]
    mx = (cc - steps * fsub[..., 0]).astype(np.float32)     # source col
    my = (rr - steps * fsub[..., 1]).astype(np.float32)     # source row
    return cv2.remap(cur.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0)


def hour_key(ts):  # numpy datetime64 -> integer hour-since-epoch
    return int(np.datetime64(ts, "h").astype("int64"))


def load_gauge_hourly(loc, gauge):
    """{hour_key -> wet(0/1)} from EA 15-min truth (>=0.1mm in the hour = wet)."""
    q = f"""SELECT date_trunc('hour', ObservedTimeUtc) h, SUM(Value15MinMm) mm
            FROM read_parquet('{RAIN}/location={loc}/**/*.parquet', hive_partitioning=false, union_by_name=true)
            WHERE StationName = '{gauge}' GROUP BY 1"""
    out = {}
    for h, mm in duckdb.sql(q).fetchall():
        if mm is not None:
            out[hour_key(np.datetime64(h))] = 1 if mm >= WET else 0
    return out


def load_nwp_exact(loc):
    """{(hour_key, lead_h) -> ensemble-mean precip mm} from the exact-runtime source."""
    leads = ",".join(str(l) for l in LEADS_H)
    q = f"""SELECT date_trunc('hour', ValidTimeUtc) h, LeadHours l, AVG(Precipitation) p
            FROM read_parquet('{FC}/location={loc}/**/*.parquet', hive_partitioning=false, union_by_name=true)
            WHERE RunTimeSource='exact' AND LeadHours IN ({leads}) AND Precipitation IS NOT NULL
            GROUP BY 1,2"""
    out = {}
    for h, l, p in duckdb.sql(q).fetchall():
        out[(hour_key(np.datetime64(h)), int(l))] = p
    return out


def load_nowcast_cached(loc):
    """{hour_key(valid) -> ensemble-mean precip} from hist_forecast lead-0 = the OM 'nowcast'.
    This is the freshest-per-valid-hour forecast (issued ~at the valid hour), so vs a +1h radar
    nowcast it has a freshness EDGE radar never had — a deliberately HARD baseline for radar.
    Cached to npz so it's pulled once."""
    cp = f"{CACHE_DIR}/phase2_nowcast_{loc}.npz"
    if os.path.exists(cp):
        z = np.load(cp)
        return {int(h): float(p) for h, p in zip(z["h"], z["p"])}
    print("  scanning parquet for hist_forecast nowcast (one-time, cached after) ...", flush=True)
    q = f"""SELECT date_trunc('hour', ValidTimeUtc) h, AVG(Precipitation) p
            FROM read_parquet('{FC}/location={loc}/**/*.parquet', hive_partitioning=false, union_by_name=true)
            WHERE RunTimeSource='hist_forecast' AND LeadHours=0 AND Precipitation IS NOT NULL
            GROUP BY 1"""
    out = {hour_key(np.datetime64(h)): p for h, p in duckdb.sql(q).fetchall()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cp, h=np.array(list(out.keys()), dtype=np.int64),
                        p=np.array(list(out.values()), dtype=np.float32))
    return out


def load_stale_cached(loc):
    """{hour_key(valid) -> ensemble-mean precip} from offset_day lead-24 = the STALE end of the
    freshness ladder (forecast issued ~24h before the target hour). Both sites have this. Cached."""
    cp = f"{CACHE_DIR}/phase2_stale24_{loc}.npz"
    if os.path.exists(cp):
        z = np.load(cp)
        return {int(h): float(p) for h, p in zip(z["h"], z["p"])}
    print("  scanning parquet for offset_day +24h (one-time, cached after) ...", flush=True)
    q = f"""SELECT date_trunc('hour', ValidTimeUtc) h, AVG(Precipitation) p
            FROM read_parquet('{FC}/location={loc}/**/*.parquet', hive_partitioning=false, union_by_name=true)
            WHERE RunTimeSource='offset_day' AND LeadHours=24 AND Precipitation IS NOT NULL
            GROUP BY 1"""
    out = {hour_key(np.datetime64(h)): p for h, p in duckdb.sql(q).fetchall()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cp, h=np.array(list(out.keys()), dtype=np.int64),
                        p=np.array(list(out.values()), dtype=np.float32))
    return out


def csi(c):
    d = c[0] + c[1] + c[2]
    return c[0] / d if d else float("nan")


def corr(p, t):
    p, t = np.array(p), np.array(t)
    if len(p) < 30 or p.std() < 1e-9 or t.std() < 1e-9: return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def cache_path(loc):
    return f"{CACHE_DIR}/phase2_{loc}_hw{HW}_lead{LEAD}.npz"


def load_truth_nwp_cached(loc, gauge):
    """gauge_wet + nwp dicts, with the slow DuckDB parquet scans cached to npz so rebuilds skip them."""
    cp = f"{CACHE_DIR}/phase2_truthnwp_{loc}_lead{LEAD}.npz"
    if os.path.exists(cp):
        z = np.load(cp)
        gw = {int(h): int(v) for h, v in zip(z["gw_h"], z["gw_v"])}
        nw = {(int(h), LEAD): float(p) for h, p in zip(z["nwp_h"], z["nwp_p"])}
        print(f"  truth+nwp from cache ({len(gw)} gauge hrs, {len(nw)} nwp)", flush=True)
        return gw, nw
    print("  scanning parquet for gauge truth + NWP (one-time, cached after) ...", flush=True)
    gw, nw = load_gauge_hourly(loc, gauge), load_nwp_exact(loc)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cp,
                        gw_h=np.array(list(gw.keys()), dtype=np.int64),
                        gw_v=np.array(list(gw.values()), dtype=np.int8),
                        nwp_h=np.array([k[0] for k in nw], dtype=np.int64),
                        nwp_p=np.array([nw[k] for k in nw], dtype=np.float32))
    return gw, nw


def build_cache(loc, gauge, glat, glon):
    """The one slow pass: optical flow on the gauge window + advect the gauge box per issue-time.
    Writes the advected boxes (+ matched NWP value and truth) to a tiny cache for instant re-analysis."""
    print(f"location={loc} gauge='{gauge}'; building cache (one slow pass) ...", flush=True)
    gauge_wet, nwp = load_truth_nwp_cached(loc, gauge)
    if not nwp:
        print("  no exact-runtime NWP for this location — Phase 2 needs it. Stop."); return False
    days = sorted(glob.glob(f"{FIELD}/*/*.npz"))
    mx = int(os.environ.get("PHASE2_MAXDAYS", "0"))         # >0 = quick validation on first N days only
    if mx: days = days[:mx]; print(f"  [PHASE2_MAXDAYS={mx}] limiting to {len(days)} days", flush=True)
    grow, gcol = gauge_pixel(np.load(days[0]), glat, glon)
    print(f"  gauge pixel (row,col)=({grow},{gcol}); flow window +/-{HW} cells; gauge box +/-{HALF}", flush=True)
    rec = {"h": [], "persist": [], "inst": [], "acc": [], "nwp": [], "truth": []}
    ndone = nev = 0
    for dp in days:
        d = np.load(dp)
        f = d["field"].astype(np.float32); f[f < 0] = 0; R = f / 32.0
        times = d["times"]; tsec = times.astype("datetime64[s]").astype("int64")
        wetpx = (R > WET).sum(axis=(1, 2))                  # full-field regional gate (same as v1)
        Hf, Wf = R.shape[1], R.shape[2]
        r_lo, r_hi = max(0, grow - HW), min(Hf, grow + HW + 1)
        c_lo, c_hi = max(0, gcol - HW), min(Wf, gcol + HW + 1)
        gr, gc = grow - r_lo, gcol - c_lo                   # gauge position inside the window
        for i in range(3, len(R)):
            if int(np.datetime64(times[i], "m").astype(int)) % 60 != 0: continue  # top of hour only
            if wetpx[i] < EVENT_WETPX: continue
            if int(tsec[i]) - int(tsec[i - 3]) != 900: continue
            H = hour_key(times[i]); vh = H + LEAD
            truth = gauge_wet.get(vh)
            if truth is None: continue                      # no gauge truth for the +1h target hour
            cim3 = R[i - 3, r_lo:r_hi, c_lo:c_hi]
            ci = R[i, r_lo:r_hi, c_lo:c_hi]
            flow = flow_halfres(to_u8(cim3), to_u8(ci))
            # advect across the whole +1h target hour [+60, +120) min in 15-min sub-steps (5-min flow units)
            inst = acc = None
            for j in range(4):
                box = advect_box(ci, flow, LEAD * 12 + j * 3, gr, gc, HALF)
                if j == 0: inst = box                       # start-of-hour instant = the v1 sample
                acc = box * 0.25 if acc is None else acc + box * 0.25   # 4 x 0.25h = 1h of mm
            rec["h"].append(H)
            rec["persist"].append(float(ci[gr, gc]))        # radar held at the gauge (Eulerian)
            rec["inst"].append(inst.astype(np.float32))
            rec["acc"].append(acc.astype(np.float32))
            rec["nwp"].append(np.float32(nwp.get((vh, LEAD), np.nan)))
            rec["truth"].append(np.int8(truth))
            nev += 1
        ndone += 1
        if ndone % 100 == 0: print(f"  build {ndone}/{len(days)} days, {nev} issue-times ...", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = cache_path(loc)
    np.savez_compressed(cp, h=np.array(rec["h"], dtype=np.int64),
                        persist=np.array(rec["persist"], dtype=np.float32),
                        inst=np.array(rec["inst"], dtype=np.float32),
                        acc=np.array(rec["acc"], dtype=np.float32),
                        nwp=np.array(rec["nwp"], dtype=np.float32),
                        truth=np.array(rec["truth"], dtype=np.int8))
    print(f"  cached {nev} issue-times -> {cp}", flush=True)
    return True


def analyze(loc, gauge):
    """Instant pass from cache. Each NWP head-to-head is scored on its MATCHED event set (only the
    hours where that comparator exists), so radar vs exact and radar vs nowcast are like-for-like —
    the exact source is sparse/recent, so comparing it to radar's full set would confound period."""
    z = np.load(cache_path(loc))
    persist, inst, acc, nwp, truth, hh = z["persist"], z["inst"], z["acc"], z["nwp"], z["truth"], z["h"]
    nowcast = load_nowcast_cached(loc)                      # hist_forecast lead-0 = OM nowcast comparator
    ctr = HALF
    tw = truth.astype(bool)                                 # gauge wet/dry
    # per-event prediction vectors (vectorised box-max over the cached gauge boxes)
    box = lambda B, rad: B[:, ctr - rad:ctr + rad + 1, ctr - rad:ctr + rad + 1].max(axis=(1, 2))
    P = {"persist": persist.astype(float), "radar": inst[:, ctr, ctr].astype(float)}
    for rad in RADII:
        P[f"nbhd{rad}"] = box(inst, rad); P[f"both{rad}"] = box(acc, rad)
    P["exact"] = nwp.astype(float)
    P["nowcast"] = np.array([nowcast.get(int(h) + LEAD, np.nan) for h in hh], dtype=float)
    stale = load_stale_cached(loc)                          # offset_day +24h = stale end of the ladder
    P["stale24"] = np.array([stale.get(int(h) + LEAD, np.nan) for h in hh], dtype=float)

    def csi_on(pred, mask):
        v = mask & ~np.isnan(pred)                          # scored only where present + in subset
        pw = (pred >= WET) & v; tt = tw & v
        hits = int((pw & tt).sum()); miss = int((~pw & tt).sum()); fa = int((pw & ~tt).sum())
        d = hits + miss + fa
        return (hits / d if d else float("nan")), int(v.sum())

    kmh = lambda rad: f"{2 * rad + 1}km"
    radar_cols = ["persist", "radar"] + [f"both{r}" for r in RADII]
    radar_hdr = "persist | radar v1 | " + " | ".join(f"both@{kmh(r)}" for r in RADII)

    def block(title, comparator, note):
        mask = ~np.isnan(P[comparator])
        cc, n = csi_on(P[comparator], mask)
        cells = " | ".join(f"{csi_on(P[m], mask)[0]:.3f}" for m in radar_cols)
        lines = [f"### {title}  (matched subset: {n} events)", note,
                 f"| {comparator} CSI | {radar_hdr} |",
                 "|---:|---:|---:|" + "---:|" * len(RADII),
                 f"| **{cc:.3f}** | {cells} |", ""]
        return lines

    out = [f"# Radar advection — Phase 2: +{LEAD}h radar vs NWP on the EA gauge ({loc})", "",
           f"- radar read AT the gauge pixel; persist = radar-held (Eulerian); truth = gauge('{gauge}') hourly wet(>={WET}mm). Flow window +/-{HW}km; cached.",
           "- **both@Xkm** = hour-accumulated radar, max over an X-by-X box at the gauge (1km = single pixel).",
           "- Each block is scored ONLY on hours where that NWP comparator exists, so radar and the NWP are on the SAME events.", ""]
    out += block("vs EXACT +1h NWP — the FAIR test (forecast issued at T, same info radar had)",
                 "exact", "_If radar (esp. both@3–7km) ≥ exact here, radar genuinely beats our leak-safe short-lead forecast._")
    out += block("vs NOWCAST (hist_forecast lead-0) — a HARD test (NWP issued ~at the target hour, a freshness edge radar lacks)",
                 "nowcast", "_Radar beating this = strong (overcame a freshness handicap); radar losing = inconclusive, not a refutation._")
    # freshness ladder on the common subset: validates that the FAIR +1h sits between stale(+24h)
    # and fresh(lead-0), and measures where (fraction f) — so Sennen's missing +1h can be bracketed.
    m3 = ~np.isnan(P["stale24"]) & ~np.isnan(P["exact"]) & ~np.isnan(P["nowcast"])
    n3 = int(m3.sum())
    s, e, f = (csi_on(P[x], m3)[0] for x in ("stale24", "exact", "nowcast"))
    frac = (e - s) / (f - s) if (f - s) else float("nan")
    out += [f"### Freshness ladder — same {n3} events (does the FAIR +1h sit between the two ends Sennen also has?)",
            f"_If stale ≤ exact ≤ fresh, Sennen's missing +1h can be estimated at fraction f={frac:.2f} of the way from offset\\_day+24h up to the lead-0 nowcast._",
            "| rung | lead | CSI |", "|---|---|---:|",
            f"| offset_day (stale) | +24h | {s:.3f} |",
            f"| **exact (FAIR)** | **+1h** | **{e:.3f}** |",
            f"| nowcast (fresh) | lead-0 | {f:.3f} |",
            f"| _radar both@5km_ | _+1h_ | _{csi_on(P['both2'], m3)[0]:.3f}_ |", ""]
    out.append("- neighbourhood-only (instant, no accumulation), on the exact-subset — how much the accumulation adds:")
    mask_e = ~np.isnan(P["exact"])
    out.append("| " + " | ".join(f"nbhd@{kmh(r)}" for r in RADII) + " |")
    out.append("|---:|" + "---:|" * (len(RADII) - 1))
    out.append("| " + " | ".join(f"{csi_on(P[f'nbhd{r}'], mask_e)[0]:.3f}" for r in RADII) + " |")
    os.makedirs("data/reports", exist_ok=True)
    rep = f"data/reports/radar_phase2_{loc}.md"
    open(rep, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out)); print(f"\nwrote {rep}", flush=True)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")           # report has ≥/→/– ; don't die on cp1252 console
    except Exception: pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    loc = args[0] if args else "bonehill_rocks"
    rebuild = "--rebuild" in sys.argv
    gauge = GAUGE[loc]
    if gauge not in GAUGE_COORDS:
        print(f"no lat/lon for gauge '{gauge}' — add it to GAUGE_COORDS. Stop."); return
    glat, glon = GAUGE_COORDS[gauge]
    if rebuild or not os.path.exists(cache_path(loc)):
        if not build_cache(loc, gauge, glat, glon): return
    else:
        print(f"using cache {cache_path(loc)} (pass --rebuild to recompute)", flush=True)
    analyze(loc, gauge)


if __name__ == "__main__":
    main()
