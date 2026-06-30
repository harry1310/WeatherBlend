#!/usr/bin/env python
"""Phase 1 radar-advection backtest — does optical-flow advection beat Eulerian persistence
(and climatology) at Bonehill / Membury / Sennen, and out to what lead, in which regime?

For sampled rain-event times T (a rain system is in view of the SW box), advect the radar
field forward with Farneback dBR optical flow + semi-Lagrangian warp, and score the SITE
rain-rate prediction at +30..+180 min against the observed radar, vs persistence (rain@T held)
and the site climatology. Strictly leak-free: only frames <= T feed each prediction.

Headline metric = CSI (Critical Success Index) at the wet threshold (>0.1 mm/hr, matching 3c);
also reports rate correlation. Stratified by site x season (DJF/JJA = frontal/convective proxy).

  python scripts/radar/advection_backtest.py [YEARFILTER]
Output: data/reports/radar_advection_phase1.md
"""
import numpy as np, cv2, glob, os, sys
from collections import defaultdict

FIELD = "data/radar/extracted/field"
LEADS = [6, 12, 18, 24, 36]            # 30/60/90/120/180 min in 5-min steps
LMIN = {6: 30, 12: 60, 18: 90, 24: 120, 36: 180}
WET = 0.1                              # mm/hr wet threshold
EVENT_WETPX = 500                     # region must have >= this many wet px to be an "event"
MAX_EV_PER_DAY = 25
SITES = ["bonehill", "membury", "sennen"]
SEASON = lambda m: ["DJF", "MAM", "JJA", "SON"][(m % 12) // 3]


def to_u8(R):  # dBR transform + 0-255 normalise (so Farneback has gradients to track)
    x = 10 * np.log10(np.clip(R, 0.05, None)); x[R < WET] = -15.0
    return np.clip((x + 15.0) / 45.0 * 255.0, 0, 255).astype(np.uint8)


def flow_for(R, i):  # per-5min-step motion from the 15-min baseline (i-3 -> i)
    return cv2.calcOpticalFlowFarneback(to_u8(R[i - 3]), to_u8(R[i]), None, 0.5, 4, 31, 3, 7, 1.5, 0) / 3.0


def advect(cur, flow, leads):  # semi-Lagrangian backward warp
    H, W = cur.shape; X, Y = np.meshgrid(np.arange(W), np.arange(H))
    cur = cur.astype(np.float32); out = {}
    for t in leads:
        mx = (X - t * flow[..., 0]).astype(np.float32); my = (Y - t * flow[..., 1]).astype(np.float32)
        out[t] = cv2.remap(cur, mx, my, cv2.INTER_LINEAR, borderValue=0)
    return out


def csi(c):  # c = [hits, misses, false_alarms]
    d = c[0] + c[1] + c[2]
    return c[0] / d if d else float("nan")


def corr(p, t):
    if len(p) < 30: return float("nan")
    p, t = np.array(p), np.array(t)
    if p.std() < 1e-9 or t.std() < 1e-9: return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def main():
    days = sorted(glob.glob(f"{FIELD}/*/*.npz"))
    if len(sys.argv) > 1:
        days = [d for d in days if sys.argv[1] in d]
    cont = defaultdict(lambda: np.zeros(3))     # (site,lead,seas,method) -> [hits,miss,fa]
    rate = defaultdict(lambda: ([], []))        # (site,lead,seas,method) -> (preds,truths)
    base = defaultdict(lambda: [0, 0])          # (site,seas) -> [wet, total]  (climatology)
    nday = nev = 0
    for dp in days:
        d = np.load(dp)
        f = d["field"].astype(np.float32); f[f < 0] = 0; R = f / 32.0
        times = d["times"]; locr = d["loc_rows"]; locc = d["loc_cols"]
        wetpx = (R > WET).sum(axis=(1, 2))
        tsec = times.astype("datetime64[s]").astype("int64")          # timestamp index (handles missing scans)
        tidx = {int(s): k for k, s in enumerate(tsec)}
        last = int(tsec[-1])
        cand = [i for i in range(3, len(R))
                if wetpx[i] >= EVENT_WETPX
                and int(tsec[i]) - int(tsec[i - 3]) == 900            # gap-free 15-min history for the flow
                and int(tsec[i]) + 36 * 300 <= last]                  # room for +180min within the day
        if not cand: continue
        if len(cand) > MAX_EV_PER_DAY:
            cand = [cand[k] for k in np.linspace(0, len(cand) - 1, MAX_EV_PER_DAY).astype(int)]
        nday += 1
        for i in cand:
            nev += 1
            se = SEASON(int(str(times[i])[5:7]))
            fc = advect(R[i], flow_for(R, i), LEADS)
            t0 = int(tsec[i])
            for k, site in enumerate(SITES):
                r, c = int(locr[k]), int(locc[k]); p0 = R[i, r, c]
                for t in LEADS:
                    j = tidx.get(t0 + t * 300)                        # truth scan at exactly T+lead
                    if j is None: continue                            # missing scan -> skip this lead
                    truth = R[j, r, c]; tw = truth > WET
                    base[(site, se)][0] += int(tw); base[(site, se)][1] += 1
                    for meth, pred in (("adv", float(fc[t][r, c])), ("persist", float(p0))):
                        pw = pred > WET; key = (site, t, se, meth)
                        if pw and tw: cont[key][0] += 1
                        elif tw and not pw: cont[key][1] += 1
                        elif pw and not tw: cont[key][2] += 1
                        rate[key][0].append(pred); rate[key][1].append(truth)
        if nday % 100 == 0:
            print(f"  {nday} days, {nev} events ...")

    # aggregate helper across seasons
    def agg(site, t, meth, seasons):
        c = np.zeros(3); p, tr = [], []
        for se in seasons:
            c += cont[(site, t, se, meth)]
            p += rate[(site, t, se, meth)][0]; tr += rate[(site, t, se, meth)][1]
        return csi(c), corr(p, tr), int(c.sum())

    os.makedirs("data/reports", exist_ok=True)
    out = ["# Radar advection — Phase 1: does optical-flow advection beat persistence at the sites?", ""]
    out.append(f"- {nday} rain-event days, {nev} event-times sampled (region >= {EVENT_WETPX} wet px; <= {MAX_EV_PER_DAY}/day).")
    out.append("- Method: Farneback dBR optical flow (15-min baseline) + semi-Lagrangian warp. Truth = observed radar at T+lead.")
    out.append("- CSI = hits/(hits+miss+false-alarm) at >0.1 mm/hr. corr = Pearson on rain-rate. Persistence = rain@T held.")
    out.append("- **Advection earns its keep where CSI(adv) > CSI(persist); the lead where that gap closes = the skill horizon.**")
    out.append("")
    GROUPS = [("ALL", ["DJF", "MAM", "JJA", "SON"]), ("DJF (frontal)", ["DJF"]), ("JJA (convective)", ["JJA"])]
    for site in SITES:
        bw, bt = sum(base[(site, s)][0] for s in ["DJF","MAM","JJA","SON"]), sum(base[(site, s)][1] for s in ["DJF","MAM","JJA","SON"])
        out.append(f"## {site}  (event wet-rate {bw/bt:.1%})")
        out.append("")
        for gname, seas in GROUPS:
            out.append(f"### {gname}")
            out.append("| lead | n | CSI adv | CSI persist | ΔCSI | corr adv | corr persist |")
            out.append("|---|---:|---:|---:|---:|---:|---:|")
            for t in LEADS:
                ca, ra, n = agg(site, t, "adv", seas)
                cp, rp, _ = agg(site, t, "persist", seas)
                dc = ca - cp
                out.append(f"| +{LMIN[t]}m | {n} | {ca:.3f} | {cp:.3f} | {dc:+.3f} | {ra:.3f} | {rp:.3f} |")
            out.append("")
    rep = "data/reports/radar_advection_phase1.md"
    open(rep, "w", encoding="utf-8").write("\n".join(out))
    print(f"\nwrote {rep}  ({nday} days, {nev} events)")


if __name__ == "__main__":
    main()
