"""3g hourly-source bake-off, copula edition — do the 3j/3n simulation
techniques help when run over 3e's hourly P(wet) instead of 3a's?

Context
-------
MC-over-3e-raw ties 3g (iid MC over 3a) and marginally edges it at the 6h
window. The calibration angle is dead (phase3g_3e_calibration: per-hour
calibration HURTS the iid MC). Open question: does swapping the *sampler*
— from 3g's independent-hour Bernoulli to 3j's Gaussian copula, or 3n's
regime-conditioned copula — unlock 3e's hourly skill at the 6h window?

  3j  Gaussian copula MC: draw the 9 daytime hours from N(0, Sigma),
      Phi-transform to uniform, threshold against the hourly marginals.
      Sigma is a 9x9 wet/dry correlation fit on OBSERVED daytime
      sequences (model-independent — same Sigma for 3a or 3e marginals).
  3n  Regime-conditioned copula: two Sigmas per (station, lead) —
      settled vs unsettled — picked per day by NWP-ensemble agreement.

Sigma matrices are loaded from the existing 3j / 3n bundles' correlation.json
(fit on each bundle's train slice — disjoint from the test slice scored
here, so no leakage). NWP agreement for the 3n regime label is recomputed
from the offset_day forecast tree, mirroring DryWindowNwpAgreement.ComputePerDay.

Variants scored (window 6h headline; 3h/4h for context):
  3g       iid MC over 3a            (champion baseline)
  3e-iid   iid MC over 3e            (the known ~tie)
  3j-3a    copula MC over 3a         (validates the copula impl vs the 3j bundle)
  3j-3e    copula MC over 3e         <- hypothesis
  3n-3a    regime copula over 3a
  3n-3e    regime copula over 3e     <- hypothesis

Scored on the full ~125-day 3a/3e test slice (comparable to the 4-way
bake-off: 3g ~= 0.125 aggregate, 0.131 at 6h).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3g_3e_calibration_bakeoff import (  # noqa: E402
    LEADS, MC_SAMPLES, PRECIP, SEEDS, STATIONS, WINDOWS,
    brier, daytime_cells, daytime_utc_hours, has_dry_block, load_hourly, mc_prob,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRY_WINDOW = os.path.join(REPO, "data", "models", "dry_window")
FORECASTS = os.path.join(REPO, "data", "forecasts")
CANONICAL_NWPS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless",
                  "meteofrance_seamless", "ukmo_seamless", "gem_seamless",
                  "ecmwf_aifs025_single", "jma_seamless"]
WET_MM_H = 0.1
MIN_MODELS_PER_HOUR = 3


# ----------------------------------------------------------------------
# Sigma loading (from the 3j / 3n bundles' correlation.json — UTF-8 BOM)
# ----------------------------------------------------------------------

def _latest_bundle(station: str, suffix: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(DRY_WINDOW, station, "window_6h", f"v*_{suffix}")))
    return hits[-1] if hits else None


def load_sigma_3j(station: str) -> dict[int, np.ndarray]:
    d = _latest_bundle(station, "phase3j")
    if d is None:
        return {}
    c = json.load(open(os.path.join(d, "correlation.json"), encoding="utf-8-sig"))
    return {int(ld): np.asarray(v["Sigma"], dtype=float)
            for ld, v in c["ByLead"].items()}


def load_sigma_3n(station: str) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    d = _latest_bundle(station, "phase3n")
    if d is None:
        return {}
    c = json.load(open(os.path.join(d, "correlation.json"), encoding="utf-8-sig"))
    out = {}
    for ld, v in c["ByLead"].items():
        out[int(ld)] = (np.asarray(v["Sigma_settled"], dtype=float),
                        np.asarray(v["Sigma_unsettled"], dtype=float),
                        float(v["Threshold"]))
    return out


def safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Cholesky factor, with a nearest-PSD repair if the fitted Sigma
    isn't quite positive-definite (binary-data correlation fits can drift
    slightly off the PSD cone)."""
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(sigma)
        w = np.clip(w, 1e-8, None)
        psd = (V * w) @ V.T
        dg = np.sqrt(np.diag(psd))
        psd = psd / np.outer(dg, dg)  # renormalise to unit diagonal
        return np.linalg.cholesky(psd)


# ----------------------------------------------------------------------
# Copula Monte Carlo
# ----------------------------------------------------------------------

def copula_mc(q: np.ndarray, chol: np.ndarray, window: int,
              rng: np.random.Generator) -> float:
    """P(exists contiguous >= window dry-hour run) via Gaussian copula MC.
    Draw Z ~ N(0, Sigma) (Sigma = chol @ chol.T), Phi-transform to uniform,
    wet iff U < q (so P(wet hour h) = q[h] marginally, hours correlated)."""
    n = len(q)
    eps = rng.standard_normal((MC_SAMPLES, n))
    Z = eps @ chol.T
    U = ndtr(Z)
    wet = (U < q).astype(np.int8)
    return float(has_dry_block(wet, window).mean())


# ----------------------------------------------------------------------
# NWP agreement (3n regime label) — mirrors DryWindowNwpAgreement.ComputePerDay
# ----------------------------------------------------------------------

def nwp_agreement_by_date(lead: int) -> dict[pd.Timestamp, float]:
    """Per-target-date NWP consensus from offset_day forecasts at this lead.
    1.0 = unanimous wet/dry every daytime hour; 0.5 = max disagreement.
    Location is bonehill_rocks (all 3 EA stations share its NWP grid cell)."""
    import duckdb
    glob_path = os.path.join(FORECASTS, "location=bonehill_rocks", "model=*",
                             "**", "*.parquet").replace("\\", "/")
    model_list = ",".join(f"'{m}'" for m in CANONICAL_NWPS)
    sql = f"""
        SELECT Model, ValidTimeUtc, CAST(Precipitation AS DOUBLE) AS precip
        FROM read_parquet('{glob_path}', hive_partitioning=false, union_by_name=true)
        WHERE LeadHours = {lead}
          AND (RunTimeSource IS NULL OR RunTimeSource = 'offset_day')
          AND Precipitation IS NOT NULL
          AND Model IN ({model_list})
    """
    df = duckdb.query(sql).to_df()
    if df.empty:
        return {}
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])
    df["target_date"] = df["ValidTimeUtc"].dt.normalize()
    df["hour"] = df["ValidTimeUtc"].dt.hour

    out: dict[pd.Timestamp, float] = {}
    for td, day in df.groupby("target_date"):
        s, e = daytime_utc_hours(pd.Timestamp(td))
        consensus, valid = 0.0, 0
        for h in range(s, e):
            hr = day[day["hour"] == h]
            total = hr["Model"].nunique()
            if total < MIN_MODELS_PER_HOUR:
                continue
            wet = (hr.groupby("Model")["precip"].max() >= WET_MM_H).sum()
            p = wet / total
            consensus += max(p, 1.0 - p)
            valid += 1
        out[pd.Timestamp(td)] = consensus / valid if valid else float("nan")
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    df_3a = pd.concat([d for s in STATIONS if (d := load_hourly(s, None)) is not None])
    df_3e = pd.concat([d for s in STATIONS if (d := load_hourly(s, "3e")) is not None])
    cells_3a = daytime_cells(df_3a)
    cells_3e = daytime_cells(df_3e)
    common = sorted(set(cells_3a) & set(cells_3e))
    if not common:
        sys.exit("no overlapping daytime cells")
    print(f"{len({k[2] for k in common})} test days, {len(common)} (station,lead,date) cells")

    # Sigma per station; NWP agreement per lead (shared across stations).
    sigma_3j = {s: load_sigma_3j(s) for s in STATIONS}
    sigma_3n = {s: load_sigma_3n(s) for s in STATIONS}
    agreement = {ld: nwp_agreement_by_date(ld) for ld in LEADS}
    for ld in LEADS:
        ag = [v for v in agreement[ld].values() if not np.isnan(v)]
        print(f"  lead {ld}h: agreement computed for {len(ag)} days "
              f"(mean {np.mean(ag):.3f})" if ag else f"  lead {ld}h: no agreement")

    variants = ["3g", "3e-iid", "3j-3a", "3j-3e", "3n-3a", "3n-3e"]
    per_cell = []

    by_ls: dict = {}
    for k in common:
        by_ls.setdefault((k[0], k[1]), []).append(k)

    for (station, lead), keys in sorted(by_ls.items()):
        keys.sort(key=lambda k: k[2])
        q3a = {k: cells_3a[k]["p_wet"].to_numpy(float) for k in keys}
        q3e = {k: cells_3e[k]["p_wet"].to_numpy(float) for k in keys}
        obs = {k: cells_3e[k]["observed_wet"].to_numpy(np.int8) for k in keys}

        # 3j: one Sigma per (station, lead)
        sj = sigma_3j.get(station, {}).get(lead)
        chol_3j = safe_cholesky(sj) if sj is not None else None

        # 3n: two Sigmas + per-day regime from NWP agreement
        sn = sigma_3n.get(station, {}).get(lead)
        if sn is not None:
            sig_set, sig_uns, thresh = sn
            chol_set, chol_uns = safe_cholesky(sig_set), safe_cholesky(sig_uns)
        else:
            chol_set = chol_uns = thresh = None

        def regime_chol(td: pd.Timestamp):
            if chol_set is None:
                return None
            a = agreement.get(lead, {}).get(td, float("nan"))
            # unclassifiable day -> settled (matches 3n's predict default)
            return chol_uns if (not np.isnan(a) and a < thresh) else chol_set

        for window in WINDOWS:
            truth = np.array([has_dry_block(obs[k][None, :], window)[0] for k in keys],
                             dtype=float)
            for vname in variants:
                if vname in ("3j-3a", "3j-3e") and chol_3j is None:
                    continue
                if vname in ("3n-3a", "3n-3e") and chol_set is None:
                    continue
                seed_briers = []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    preds = np.empty(len(keys))
                    for i, k in enumerate(keys):
                        if vname == "3g":
                            preds[i] = mc_prob(q3a[k], window, rng)
                        elif vname == "3e-iid":
                            preds[i] = mc_prob(q3e[k], window, rng)
                        elif vname == "3j-3a":
                            preds[i] = copula_mc(q3a[k], chol_3j, window, rng)
                        elif vname == "3j-3e":
                            preds[i] = copula_mc(q3e[k], chol_3j, window, rng)
                        elif vname == "3n-3a":
                            preds[i] = copula_mc(q3a[k], regime_chol(k[2]), window, rng)
                        elif vname == "3n-3e":
                            preds[i] = copula_mc(q3e[k], regime_chol(k[2]), window, rng)
                    seed_briers.append(brier(preds, truth))
                per_cell.append(dict(
                    station=station, lead=lead, window=window, variant=vname,
                    n=len(keys), brier=float(np.mean(seed_briers)),
                    brier_sd=float(np.std(seed_briers))))

    res = pd.DataFrame(per_cell)

    print("\n=== aggregate Brier (mean over 27 cells, 4-seed mean) ===")
    for v in variants:
        sub = res[res["variant"] == v]
        if not sub.empty:
            print(f"  {v:8s} {sub['brier'].mean():.4f}  (seed sd ~{sub['brier_sd'].mean():.4f})")

    print("\n=== per-window Brier (% vs 3g) ===")
    for window in WINDOWS:
        b3g = res[(res.variant == "3g") & (res.window == window)]["brier"].mean()
        line = f"  {window}h: 3g={b3g:.4f}  "
        for v in variants[1:]:
            sub = res[(res.variant == v) & (res.window == window)]
            if not sub.empty:
                m = sub["brier"].mean()
                line += f"{v}={m:.4f}({100*(m-b3g)/b3g:+.1f}%)  "
        print(line)

    print("\n=== 6h window — per cell ===")
    six = res[res.window == 6]
    piv = six.pivot_table(index=["station", "lead"], columns="variant", values="brier")
    print(piv[[v for v in variants if v in piv]].round(4).to_string())
    for v in variants[1:]:
        if v in piv:
            print(f"  {v} beats 3g at 6h: {(piv[v] < piv['3g']).sum()}/9")


if __name__ == "__main__":
    main()
