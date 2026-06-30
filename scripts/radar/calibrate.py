"""Fit advected-accumulation -> P(rain next hour) from the cached backtest data.

v1 source = the Farneback/NIMROD/gauge backtest cache (`data/radar/cache/phase2_*.npz`, which holds the
per-event accumulated advected box + gauge wet/dry truth). The live engine (ODIM/crag) reuses this mapping:
accum->P(wet) is a physical relationship that transfers reasonably across engine/location. ASSUMPTION flagged
in docs/RADAR_NOWCAST_PLAN.md — re-fit engine-matched once the ODIM archive-verify track exists.

Model: logistic on log1p(accum/WET). Compact (2 params), monotonic, smooth.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize

CACHE = "data/radar/cache/phase2_bonehill_rocks_hw150_lead1.npz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_bonehill.json")
NBHD_RAD = 2  # 5km neighbourhood-max predictor (matches the live engine's nbhd_km=2)
WET = 0.1


def main():
    z = np.load(CACHE)
    acc, truth = z["acc"], z["truth"].astype(float)
    ctr = acc.shape[1] // 2
    x = acc[:, ctr - NBHD_RAD:ctr + NBHD_RAD + 1, ctr - NBHD_RAD:ctr + NBHD_RAD + 1].max(axis=(1, 2))
    zt = np.log1p(x / WET)

    def nll(p):
        lo = p[0] + p[1] * zt
        pr = np.clip(1.0 / (1.0 + np.exp(-lo)), 1e-6, 1 - 1e-6)
        return -np.mean(truth * np.log(pr) + (1 - truth) * np.log(1 - pr))

    a, b = minimize(nll, [0.0, 1.0], method="Nelder-Mead").x
    pr = 1.0 / (1.0 + np.exp(-(a + b * zt)))
    brier, base = float(np.mean((pr - truth) ** 2)), float(truth.mean())
    print(f"N={len(truth)}  base wet-rate={base:.3f}")
    print(f"logistic a={a:.3f} b={b:.3f}  Brier={brier:.4f}  (climatology Brier {base * (1 - base):.4f})")
    print("calibration curve:")
    for av in [0.0, 0.05, 0.1, 0.3, 1.0, 3.0]:
        p = 1.0 / (1.0 + np.exp(-(a + b * np.log1p(av / WET))))
        print(f"  accum {av:>4.2f} mm -> P(wet) {p:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(model="logistic", transform="log1p(accum/0.1)", a=float(a), b=float(b),
                   nbhd_rad=NBHD_RAD, wet=WET, source=os.path.basename(CACHE), n=int(len(truth)),
                   brier=brier, climatology_brier=base * (1 - base)), open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
