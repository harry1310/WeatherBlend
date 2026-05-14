"""PAV block-size diagnostic — does a bin-size floor cleanly separate
PAV-helps-cells from PAV-hurts-cells?

For each (phase, station, window, lead) cell:
  1. Recompute val raw probabilities (iid MC for 3g, copula MC for 3j,
     regime MC for 3n — using each bundle's saved Σ / regime threshold)
  2. Manually fit PAV with block-size tracking (the standard
     pool-adjacent-violators algorithm, but we don't collapse block
     metadata — we keep the count of original samples per merged block)
  3. Tabulate per cell: min block size, total blocks, PAV-helped-or-hurt,
     whether thresholds N=10/15/20/25 would gate this cell off

Outputs a flat table you can scan for a clean separating threshold.

Val date set: chronological 15% slice immediately preceding the test
dates we see in the raw bundle. Approximate but consistent across
phases (all three use DryWindowDataset.Split → same val rows).
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(r"C:\Projects\Weather\WeatherBlend")
DRY_WINDOW_ROOT = ROOT / "data" / "models" / "dry_window"
PRECIP_REPLAY_ROOT = ROOT / "data" / "predictions" / "precipitation_replay"
FORECASTS_GLOB = (ROOT / "data" / "forecasts" / "location=bonehill_rocks" /
                  "model=*" / "**" / "*.parquet").as_posix()

STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
WINDOWS = [3, 4, 6]
LEADS = [24, 48, 72]
MC_SAMPLES = 10000
WET_THRESHOLD = 0.1
CANONICAL_MODELS = [
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless",
    "ukmo_seamless", "gem_seamless", "ecmwf_aifs025_single", "jma_seamless",
]


# DST-aware daytime window (mirrors WeatherBlend's DaytimeWindow with
# start=09 / end=18 Europe/London). UTC hour range varies by DST.
from zoneinfo import ZoneInfo
from datetime import datetime as DT
LON = ZoneInfo("Europe/London")


def daytime_utc_hours(d: pd.Timestamp) -> tuple[int, int]:
    s_local = DT(d.year, d.month, d.day, 9, tzinfo=LON)
    e_local = DT(d.year, d.month, d.day, 18, tzinfo=LON)
    s_utc = s_local.astimezone(ZoneInfo("UTC")).hour
    e_utc = e_local.astimezone(ZoneInfo("UTC")).hour
    if e_utc == 0:
        e_utc = 24
    return s_utc, e_utc


def find_replay(station: str) -> Path:
    cands = [d for d in (PRECIP_REPLAY_ROOT / station).iterdir() if d.is_dir() and "phase" not in d.name]
    return max(cands, key=lambda d: d.name)


def load_replay(station: str, lead: int) -> pd.DataFrame:
    r = find_replay(station) / f"lead_{lead}h.parquet"
    df = pd.read_parquet(r)
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    return df


def daytime_q_and_label(replay_df: pd.DataFrame, target_date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray] | None:
    s, e = daytime_utc_hours(target_date)
    sub = replay_df[(replay_df["target_date"] == target_date) &
                    (replay_df["hour"] >= s) & (replay_df["hour"] < e)].sort_values("valid_time")
    if len(sub) != e - s:
        return None
    return (np.clip(sub["ProbWet"].to_numpy(dtype="float64"), 1e-6, 1 - 1e-6),
            sub["Label"].astype(np.int32).to_numpy())


def has_dry_block(seq: np.ndarray, window: int) -> bool:
    run = 0
    for v in seq:
        if v == 0:
            run += 1
            if run >= window: return True
        else:
            run = 0
    return False


def iid_mc(q: np.ndarray, window: int, n: int, rng: np.random.Generator) -> float:
    samples = (rng.random((n, len(q))) < q).astype(np.int32)
    hits = 0
    for s in samples:
        run = longest = 0
        for v in s:
            if v == 0:
                run += 1
                if run > longest: longest = run
            else:
                run = 0
        if longest >= window: hits += 1
    return hits / n


def copula_mc(q: np.ndarray, L: np.ndarray, window: int, n: int, rng: np.random.Generator) -> float:
    eps = rng.standard_normal((n, len(q)))
    z = eps @ L.T  # (n, k) where L is lower triangular
    u = 0.5 * (1.0 + erf_approx(z / np.sqrt(2.0)))
    samples = (u < q).astype(np.int32)
    hits = 0
    for s in samples:
        run = longest = 0
        for v in s:
            if v == 0:
                run += 1
                if run > longest: longest = run
            else:
                run = 0
        if longest >= window: hits += 1
    return hits / n


def erf_approx(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 (same as C# DryWindow3jPredictor).
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5 = -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * np.exp(-ax*ax)
    return sign * y


def chol(sigma: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(sigma)


def pav_with_blocks(probs: np.ndarray, labels: np.ndarray) -> list[int]:
    """Standard PAV; return list of block sample counts post-merging."""
    order = np.argsort(probs, kind="stable")
    blocks: list[list[float]] = []  # [sum_y, count]
    for i in order:
        new = [float(labels[i]), 1]
        while blocks and blocks[-1][0] / blocks[-1][1] >= new[0] / new[1]:
            top = blocks.pop()
            new = [top[0] + new[0], top[1] + new[1]]
        blocks.append(new)
    return [int(b[1]) for b in blocks]


# Load forecasts once for NWP agreement (used by 3n).
def load_forecasts(lead: int) -> pd.DataFrame:
    sql = f"""
    SELECT Model, ValidTimeUtc, CAST(Precipitation AS DOUBLE) AS precip
    FROM read_parquet('{FORECASTS_GLOB}', hive_partitioning = false, union_by_name = true)
    WHERE LeadHours = {lead}
      AND (RunTimeSource IS NULL OR RunTimeSource = 'offset_day')
      AND Precipitation IS NOT NULL
      AND Model IN ({", ".join(repr(m) for m in CANONICAL_MODELS)})
    """
    df = duckdb.connect(":memory:").execute(sql).fetchdf()
    df["valid_time"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    return df


def day_agreement(forecasts: pd.DataFrame, target_date: pd.Timestamp) -> float | None:
    s, e = daytime_utc_hours(target_date)
    day = forecasts[(forecasts["target_date"] == target_date) &
                    (forecasts["hour"] >= s) & (forecasts["hour"] < e)]
    if day.empty: return None
    wet = (day["precip"] >= WET_THRESHOLD).astype(int)
    by_hour = wet.groupby(day["hour"]).agg(["sum", "count"])
    if (by_hour["count"] < 3).any():
        return None
    p_wet = by_hour["sum"] / by_hour["count"]
    return float(np.mean(np.maximum(p_wet, 1 - p_wet)))


def main() -> int:
    rng_seed = 42
    rows = []

    for lead in LEADS:
        forecasts = load_forecasts(lead) if lead in LEADS else None
        for station in STATIONS:
            replay = load_replay(station, lead)
            all_dates = sorted(replay["target_date"].unique())
            for w in WINDOWS:
                for phase in ["phase3g", "phase3j", "phase3n"]:
                    bundles = sorted([p for p in (DRY_WINDOW_ROOT / station / f"window_{w}h").iterdir()
                                      if phase in p.name and (p / "test_predictions.parquet").exists()])
                    if not bundles: continue
                    raw_bundle = [b for b in bundles if not (b / "calibrator_24h.json").exists()]
                    pav_bundle = [b for b in bundles if (b / "calibrator_24h.json").exists()]
                    if not raw_bundle or not pav_bundle: continue
                    raw_test = pd.read_parquet(raw_bundle[-1] / "test_predictions.parquet")
                    test_dates_for_lead = sorted(pd.to_datetime(
                        raw_test[raw_test["lead"] == lead]["target_date"]).dt.tz_localize(None).unique())
                    if not test_dates_for_lead: continue
                    test_start = test_dates_for_lead[0]
                    # val = 15% of dates immediately before test_start
                    pre_test = [d for d in all_dates if d < test_start]
                    n_val = max(int(np.floor(len(pre_test) * 0.15 / 0.85)), 30)
                    val_dates = pre_test[-n_val:]

                    # Compute val raw probs per phase.
                    rng = np.random.default_rng(rng_seed)
                    val_probs: list[float] = []
                    val_labels: list[int] = []

                    if phase == "phase3j" or phase == "phase3n":
                        corr_path = pav_bundle[-1] / "correlation.json"
                        cj = json.loads(corr_path.read_text(encoding="utf-8-sig"))
                    if phase == "phase3j":
                        sigma = np.array(cj["ByLead"][str(lead)]["Sigma"])
                        L = chol(sigma)
                    elif phase == "phase3n":
                        sigma_s = np.array(cj["ByLead"][str(lead)]["Sigma_settled"])
                        sigma_u = np.array(cj["ByLead"][str(lead)]["Sigma_unsettled"])
                        L_s = chol(sigma_s); L_u = chol(sigma_u)
                        threshold = float(cj["ByLead"][str(lead)]["Threshold"])

                    for d in val_dates:
                        ql = daytime_q_and_label(replay, d)
                        if ql is None: continue
                        q, labels_arr = ql
                        if phase == "phase3g":
                            p = iid_mc(q, w, MC_SAMPLES, rng)
                        elif phase == "phase3j":
                            p = copula_mc(q, L, w, MC_SAMPLES, rng)
                        else:  # phase3n
                            agr = day_agreement(forecasts, d) if forecasts is not None else None
                            if agr is None: continue
                            L_use = L_s if agr >= threshold else L_u
                            p = copula_mc(q, L_use, w, MC_SAMPLES, rng)
                        val_probs.append(p)
                        val_labels.append(1 if has_dry_block(labels_arr, w) else 0)

                    if len(val_probs) < 30: continue
                    vp = np.array(val_probs); vy = np.array(val_labels)
                    block_sizes = pav_with_blocks(vp, vy)
                    # Val-improvement gate: fit PAV on val, score val raw vs val PAV
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(vp, vy)
                    vp_pav = iso.predict(vp)
                    val_br_raw = float(np.mean((vp - vy) ** 2))
                    val_br_pav = float(np.mean((vp_pav - vy) ** 2))
                    rows.append({
                        "phase": phase.replace("phase", ""),
                        "station": station.replace("ea_", ""),
                        "window": w, "lead": lead,
                        "n_val": len(val_probs),
                        "obs_rate": float(vy.mean()),
                        "n_blocks": len(block_sizes),
                        "min_block": min(block_sizes),
                        "median_block": int(np.median(block_sizes)),
                        "val_brier_raw": val_br_raw,
                        "val_brier_pav": val_br_pav,
                        "val_delta": val_br_pav - val_br_raw,
                    })

    if not rows:
        print("no rows"); return 1
    df = pd.DataFrame(rows)
    # Merge in PAV-help status from existing csv.
    pav_ab = pd.read_csv(ROOT / "reports" / "pav_ab_2026-05-14.csv")
    pav_ab["phase"] = pav_ab["phase"].astype(str)
    df = df.merge(pav_ab[["phase","station","window","lead","brier_raw","brier_pav","delta_pav_minus_raw"]],
                  on=["phase","station","window","lead"], how="left")
    df["pav_helps"] = df["delta_pav_minus_raw"] < -0.0005
    df["pav_hurts"] = df["delta_pav_minus_raw"] > 0.0005

    out = ROOT / "reports" / "pav_blocks_2026-05-14.csv"
    df.to_csv(out, index=False)

    # Tabulate
    print(f"{'phase':<5} {'station':<22} {'win':<3} {'lead':<4} {'val_raw':>8} {'val_pav':>8} {'val_d':>8} | {'tst_raw':>8} {'tst_pav':>8} {'tst_d':>8} | val→tst verdict")
    print("-" * 140)
    df_sorted = df.sort_values(["phase","station","window","lead"])
    for _, r in df_sorted.iterrows():
        v = "PAV helps" if r.pav_helps else "PAV hurts" if r.pav_hurts else "tied"
        val_helps = "helps" if r.val_delta < -0.0005 else "hurts" if r.val_delta > 0.0005 else "tied"
        print(f"{r.phase:<5} {r.station:<22} {int(r.window)}h  {int(r.lead):<4} "
              f"{r.val_brier_raw:>8.4f} {r.val_brier_pav:>8.4f} {r.val_delta:>+8.4f} | "
              f"{r.brier_raw:>8.4f} {r.brier_pav:>8.4f} {r.delta_pav_minus_raw:>+8.4f} | "
              f"val:{val_helps:<5} → test:{v}")

    # Threshold analysis: for various N, would we gate this cell off?
    print()
    print(f"{'threshold N':<14} {'cells gated':<12} {'gated helps':<12} {'gated hurts':<12} {'kept helps':<12} {'kept hurts':<12} {'classifier accuracy':<20}")
    print("-" * 110)
    helps = df["pav_helps"]
    hurts = df["pav_hurts"]
    for n_thresh in [5, 8, 10, 12, 15, 18, 20, 25, 30]:
        gated = df["min_block"] < n_thresh
        kept = ~gated
        # ideal: gate hurts, keep helps
        true_positive = (gated & hurts).sum()  # correctly gated hurt cells
        false_positive = (gated & helps).sum()  # wrongly gated help cells
        true_negative = (kept & helps).sum()   # correctly kept help cells
        false_negative = (kept & hurts).sum()  # wrongly kept hurt cells
        n_decided = helps.sum() + hurts.sum()
        accuracy = (true_positive + true_negative) / max(n_decided, 1)
        print(f"N<{n_thresh:<10}  {gated.sum():<12} {(gated & hurts).sum():<12} {(gated & helps).sum():<12} "
              f"{(kept & helps).sum():<12} {(kept & hurts).sum():<12} {accuracy:.2%}")

    # Val-improvement gate analysis
    print()
    print("VAL-IMPROVEMENT GATE: fit PAV on val, persist only if val_pav < val_raw - threshold")
    print(f"{'threshold':<14} {'fires':<7} {'kept_hurts':<12} {'kept_helps':<12} {'skip_hurts':<12} {'skip_helps':<12} {'agg_brier':>10} {'vs raw':>9}")
    print("-" * 110)
    helps_mask = df["pav_helps"].to_numpy()
    hurts_mask = df["pav_hurts"].to_numpy()
    test_raw = df["brier_raw"].to_numpy()
    test_pav = df["brier_pav"].to_numpy()
    n_cells = len(df)
    raw_agg = float(test_raw.mean())
    pav_all_agg = float(test_pav.mean())
    print(f"{'always raw':<14} {'-':<7} {'-':<12} {'-':<12} {'-':<12} {'-':<12} {raw_agg:>10.4f} {0.0:>+9.4%}")
    print(f"{'always PAV':<14} {n_cells:<7} {hurts_mask.sum():<12} {helps_mask.sum():<12} {'0':<12} {'0':<12} {pav_all_agg:>10.4f} {(pav_all_agg-raw_agg)/raw_agg:>+9.2%}")
    for t in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        fires = (df["val_delta"] < -t).to_numpy()
        chosen = np.where(fires, test_pav, test_raw)
        agg = float(chosen.mean())
        kept_hurts = int((fires & hurts_mask).sum())
        kept_helps = int((fires & helps_mask).sum())
        skip_hurts = int((~fires & hurts_mask).sum())
        skip_helps = int((~fires & helps_mask).sum())
        print(f"val_d<-{t:<6.4f}  {fires.sum():<7} {kept_hurts:<12} {kept_helps:<12} {skip_hurts:<12} {skip_helps:<12} {agg:>10.4f} {(agg-raw_agg)/raw_agg:>+9.2%}")

    # Per-station gate behaviour at val_d<0 threshold
    print()
    print("PER-STATION gate behaviour (threshold val_delta < 0):")
    for st in df["station"].unique():
        sub = df[df["station"] == st]
        fires = (sub["val_delta"] < 0).to_numpy()
        hp = sub["pav_helps"].to_numpy()
        ht = sub["pav_hurts"].to_numpy()
        print(f"  {st:<25} {fires.sum():>3}/{len(sub):<3} fires | "
              f"correctly kept helps {(fires & hp).sum()}/{hp.sum()}, "
              f"correctly skipped hurts {(~fires & ht).sum()}/{ht.sum()}")

    print(f"\nWrote per-cell block stats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
