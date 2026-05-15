"""Start-hour bake-off — do the 3j/3n copula MC techniques predict WHERE a
dry window starts better than 3g's independence MC?

Context
-------
The dry-window OCCURRENCE question ("is there a 6h dry block") prefers
independence — 3g beats every copula at 6h, because positive within-day
autocorrelation inflates long-run probabilities. But the START question
("when does the dry window begin") is the opposite shape: localising the
dry block within the day is exactly what within-day correlation structure
should help. The 2026-05-03 start-hour bake-off already hinted at this —
an AR(1) copula (option B) beat independence MC (option C). But that only
tested a single-rho AR(1) copula; the real within-day correlation is
fatter-tailed than geometric (3j's Sigma plateaus near 0.45 rather than
decaying like AR(1)). 3j (full 9x9 Sigma) and 3n (regime-conditioned)
postdate that bake-off and were never tested on the start question.

Method
------
For each (station, window, lead, target_date) MC-sample the 9-hour daytime
wet/dry sequence — iid (3g), Gaussian copula (3j), or regime-conditioned
copula (3n) — and record, per sample, every hour a window-N dry block
*starts* at. Aggregate to pi = P(dry window starts at hour h). Score pi
against the observed start-hour set with categorical Brier / LogLoss /
Top-1, informative-day filter (1 <= |truth| < |candidates|).

9-hour DST-aware daytime window (09-18 Europe/London) — consistent with
the shipped dry-window definition and reuses 3j/3n's 9x9 Sigma directly.
Sigma comes from the existing 3j/3n bundles (fit on each bundle's train
slice — disjoint from the test slice scored here).

Variants: 3g / 3j / 3n MC, each over 3a and over 3e hourly marginals.
Hypothesis: 3j/3n beat 3g on start-hour Brier/Top-1 — the inverse of the
occurrence result.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3g_3e_copula_bakeoff import (  # noqa: E402
    LEADS, MC_SAMPLES, SEEDS, STATIONS,
    daytime_cells, load_hourly, load_sigma_3j, load_sigma_3n,
    nwp_agreement_by_date, safe_cholesky,
)

WINDOWS = [3, 4, 6]
EPS = 1e-12


def draw_wet(q: np.ndarray, kind: str, chol: np.ndarray | None,
             rng: np.random.Generator) -> np.ndarray:
    """(MC_SAMPLES, n_hours) wet/dry samples — 1=wet, 0=dry.
    kind='iid' = independent Bernoulli; 'copula' = Gaussian copula via chol."""
    n = len(q)
    if kind == "iid":
        return (rng.random((MC_SAMPLES, n)) < q).astype(np.int8)
    Z = rng.standard_normal((MC_SAMPLES, n)) @ chol.T
    return (ndtr(Z) < q).astype(np.int8)


def start_hour_pi(wet: np.ndarray, window: int) -> np.ndarray:
    """pi over candidate start hours: pi[h] = P(a window-N dry block starts
    at hour h). Counts every block-start in every sample, then normalises.
    Degenerate (no block in any sample) -> uniform."""
    n_cand = wet.shape[1] - window + 1
    counts = np.array([
        (wet[:, h:h + window].sum(axis=1) == 0).sum()
        for h in range(n_cand)
    ], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else np.full(n_cand, 1.0 / n_cand)


def start_hour_truth(obs: np.ndarray, window: int) -> list[int]:
    """Observed start hours where a window-N dry block actually begins."""
    n_cand = len(obs) - window + 1
    return [h for h in range(n_cand) if obs[h:h + window].sum() == 0]


def score(pi: np.ndarray, truth: list[int]) -> tuple[float, float, int]:
    """Categorical Brier + LogLoss vs the uniform-over-truth target, and
    Top-1 (argmax pi lands on a truth-valid start)."""
    n = len(pi)
    t = np.zeros(n)
    for h in truth:
        t[h] = 1.0 / len(truth)
    brier = float(np.sum((pi - t) ** 2))
    logloss = float(-np.sum(t * np.log(np.clip(pi, EPS, 1.0))))
    top1 = int(np.argmax(pi) in truth)
    return brier, logloss, top1


def main() -> None:
    df_3a = pd.concat([d for s in STATIONS if (d := load_hourly(s, None)) is not None])
    df_3e = pd.concat([d for s in STATIONS if (d := load_hourly(s, "3e")) is not None])
    cells_3a = daytime_cells(df_3a)
    cells_3e = daytime_cells(df_3e)
    common = sorted(set(cells_3a) & set(cells_3e))
    if not common:
        sys.exit("no overlapping daytime cells")
    print(f"{len({k[2] for k in common})} test days, {len(common)} (station,lead,date) cells")

    sigma_3j = {s: load_sigma_3j(s) for s in STATIONS}
    sigma_3n = {s: load_sigma_3n(s) for s in STATIONS}
    agreement = {ld: nwp_agreement_by_date(ld) for ld in LEADS}

    variants = ["3g-3a", "3g-3e", "3j-3a", "3j-3e", "3n-3a", "3n-3e"]
    rows = []  # (station, lead, window, variant, brier, logloss, top1, n_inf)

    by_ls: dict = {}
    for k in common:
        by_ls.setdefault((k[0], k[1]), []).append(k)

    for (station, lead), keys in sorted(by_ls.items()):
        keys.sort(key=lambda k: k[2])
        q3a = {k: cells_3a[k]["p_wet"].to_numpy(float) for k in keys}
        q3e = {k: cells_3e[k]["p_wet"].to_numpy(float) for k in keys}
        obs = {k: cells_3e[k]["observed_wet"].to_numpy(np.int8) for k in keys}

        sj = sigma_3j.get(station, {}).get(lead)
        chol_3j = safe_cholesky(sj) if sj is not None else None
        sn = sigma_3n.get(station, {}).get(lead)
        if sn is not None:
            sig_set, sig_uns, thresh = sn
            chol_set, chol_uns = safe_cholesky(sig_set), safe_cholesky(sig_uns)
        else:
            chol_set = chol_uns = thresh = None

        def regime_chol(td):
            if chol_set is None:
                return None
            a = agreement.get(lead, {}).get(td, float("nan"))
            return chol_uns if (not np.isnan(a) and a < thresh) else chol_set

        # spec per variant: (marginal source, kind, chol-or-None / 'regime')
        def variant_spec(v, k):
            src = q3e if v.endswith("3e") else q3a
            if v.startswith("3g"):
                return src[k], "iid", None
            if v.startswith("3j"):
                return src[k], "copula", chol_3j
            return src[k], "copula", regime_chol(k[2])  # 3n

        for window in WINDOWS:
            # accumulate per variant across days x seeds
            acc = {v: {"brier": [], "logloss": [], "top1": []} for v in variants}
            for k in keys:
                truth = start_hour_truth(obs[k], window)
                n_cand = len(obs[k]) - window + 1
                if not (1 <= len(truth) < n_cand):  # informative-day filter
                    continue
                for v in variants:
                    if v.startswith(("3j", "3n")) and chol_3j is None and "3j" in v:
                        continue
                    q, kind, chol = variant_spec(v, k)
                    if kind == "copula" and chol is None:
                        continue
                    for seed in SEEDS:
                        rng = np.random.default_rng(seed)
                        wet = draw_wet(q, kind, chol, rng)
                        pi = start_hour_pi(wet, window)
                        b, ll, t1 = score(pi, truth)
                        acc[v]["brier"].append(b)
                        acc[v]["logloss"].append(ll)
                        acc[v]["top1"].append(t1)
            for v in variants:
                if not acc[v]["brier"]:
                    continue
                rows.append(dict(
                    station=station, lead=lead, window=window, variant=v,
                    n_inf=len(acc[v]["brier"]) // len(SEEDS),
                    brier=float(np.mean(acc[v]["brier"])),
                    logloss=float(np.mean(acc[v]["logloss"])),
                    top1=float(np.mean(acc[v]["top1"])),
                ))

    res = pd.DataFrame(rows)

    print("\n=== aggregate start-hour metrics (mean over cells, 4-seed mean) ===")
    print(f"  {'variant':8s} {'Brier':>8s} {'LogLoss':>9s} {'Top-1':>8s}")
    for v in variants:
        sub = res[res.variant == v]
        if not sub.empty:
            print(f"  {v:8s} {sub.brier.mean():8.4f} {sub.logloss.mean():9.4f} "
                  f"{sub.top1.mean():7.1%}")

    for window in WINDOWS:
        print(f"\n=== window {window}h — Brier / Top-1 (% vs 3g-3a) ===")
        b0 = res[(res.variant == "3g-3a") & (res.window == window)]["brier"].mean()
        for v in variants:
            sub = res[(res.variant == v) & (res.window == window)]
            if sub.empty:
                continue
            b, t = sub.brier.mean(), sub.top1.mean()
            tag = "" if v == "3g-3a" else f" ({100*(b-b0)/b0:+.1f}% Brier)"
            print(f"  {v:8s} Brier={b:.4f}  Top-1={t:.1%}{tag}")

    print("\n=== 6h window — per cell Brier ===")
    six = res[res.window == 6]
    piv = six.pivot_table(index=["station", "lead"], columns="variant", values="brier")
    print(piv[[v for v in variants if v in piv]].round(4).to_string())
    for v in variants:
        if v != "3g-3a" and v in piv:
            print(f"  {v} beats 3g-3a at 6h: {(piv[v] < piv['3g-3a']).sum()}/{len(piv)}")


if __name__ == "__main__":
    main()
