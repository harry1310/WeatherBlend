"""Diagnostic — why does MC-over-3a beat MC-over-3e at dry-window Brier, when
3e has lower hourly P(wet) Brier? Quantifies the three candidate explanations:
  1. Daytime-only hourly Brier may flip vs full-24h Brier.
  2. Sharpness vs calibration tradeoff (3e may be sharper but less calibrated).
  3. Per-day MC bias decomposition: when does iid MC's dry-window prediction
     drift from truth, and how does that drift correlate with the q-vector
     statistics?

Inputs: existing 3a and 3e test_predictions parquets (hourly P(wet) + observed_wet)
plus a daytime slicer. Outputs: text summary to stdout + CSV under reports/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dry_window_4way_bakeoff import (  # type: ignore
    daytime_utc_hours,
    find_latest_with_test_predictions,
    PRECIP_MODELS_ROOT,
    has_contiguous_dry_block,
    prob_dry_window_mc,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LEADS = [24, 48, 72]
DEFAULT_WINDOWS = [3, 4, 6]
MC_SAMPLES = 1000
SEED = 42


def daytime_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'target_date'+'hour' cols and filter to daytime UTC hours.
    Groups by (target_date, lead) — the parquet carries hourly rows at
    multiple leads, so a target_date holds 9*N_leads daytime rows total."""
    df = df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
    df["target_date"] = df["valid_time"].dt.normalize()
    df["hour"] = df["valid_time"].dt.hour
    keep_rows: list[pd.DataFrame] = []
    for (target_date, lead), grp in df.groupby(["target_date", "lead"]):
        s, e = daytime_utc_hours(pd.Timestamp(target_date))
        dt_rows = grp[(grp["hour"] >= s) & (grp["hour"] < e)]
        if len(dt_rows) != (e - s):
            continue
        keep_rows.append(dt_rows)
    if not keep_rows:
        return df.iloc[0:0]
    return pd.concat(keep_rows, ignore_index=True)


def reliability(probs: np.ndarray, obs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: per bin → (predicted mean, observed rate, count)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = mask.sum()
        if n == 0:
            continue
        rows.append({
            "bin": b,
            "p_low": bins[b], "p_high": bins[b + 1],
            "pred_mean": float(probs[mask].mean()),
            "obs_rate":  float(obs[mask].mean()),
            "n": int(n),
        })
    return pd.DataFrame(rows)


def calibration_error(probs: np.ndarray, obs: np.ndarray, n_bins: int = 10) -> float:
    """ECE — sum of |pred_mean - obs_rate| weighted by bin count fraction."""
    rdf = reliability(probs, obs, n_bins)
    if rdf.empty:
        return float("nan")
    w = rdf["n"] / rdf["n"].sum()
    return float((w * (rdf["pred_mean"] - rdf["obs_rate"]).abs()).sum())


def brier(probs: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean((probs - obs) ** 2))


def main() -> int:
    stations = DEFAULT_STATIONS
    leads = DEFAULT_LEADS
    windows = DEFAULT_WINDOWS

    rng = np.random.default_rng(SEED)
    summary_rows: list[dict] = []
    print("=" * 90)
    print("3a vs 3e daytime hourly diagnostic")
    print("=" * 90)

    for station in stations:
        p_3a = find_latest_with_test_predictions(PRECIP_MODELS_ROOT, station, phase_suffix=None)
        p_3e = find_latest_with_test_predictions(PRECIP_MODELS_ROOT, station, phase_suffix="phase3e")
        if p_3a is None or p_3e is None:
            print(f"::warning::skipping {station}: 3a or 3e missing")
            continue
        df_3a = daytime_slice(pd.read_parquet(p_3a))
        df_3e = daytime_slice(pd.read_parquet(p_3e))

        # Inner-join on (valid_time, lead, station). 3a + 3e have identical test
        # slices in our case so this is mostly a no-op safety check.
        merged = df_3a.merge(
            df_3e[["valid_time", "lead", "p_wet", "observed_wet"]],
            on=["valid_time", "lead"], suffixes=("_3a", "_3e"),
        )
        if not (merged["observed_wet_3a"] == merged["observed_wet_3e"]).all():
            print(f"::warning::{station}: 3a and 3e disagree on observed_wet on some rows")
        merged = merged.rename(columns={"observed_wet_3a": "observed_wet"}).drop(columns=["observed_wet_3e"])

        print(f"\n--- {station}  (daytime rows: {len(merged):,}) ---")

        for lead in leads:
            sub = merged[merged["lead"] == lead]
            if len(sub) == 0:
                continue
            obs = sub["observed_wet"].astype(np.int32).to_numpy()
            p_a = sub["p_wet_3a"].to_numpy()
            p_e = sub["p_wet_3e"].to_numpy()

            b_a = brier(p_a, obs)
            b_e = brier(p_e, obs)
            ece_a = calibration_error(p_a, obs)
            ece_e = calibration_error(p_e, obs)
            # Sharpness = variance of predictions (higher = sharper).
            var_a = float(p_a.var())
            var_e = float(p_e.var())
            mean_a = float(p_a.mean()); mean_e = float(p_e.mean()); obs_rate = float(obs.mean())

            # Extreme prediction fractions (P < 0.1 = confidently dry).
            frac_low_a = float(np.mean(p_a < 0.1)); frac_low_e = float(np.mean(p_e < 0.1))
            frac_high_a = float(np.mean(p_a > 0.9)); frac_high_e = float(np.mean(p_e > 0.9))

            print(f"\n  lead {lead}h: n_hourly={len(sub):>5d}  obs_wet_rate={obs_rate:.3f}")
            print(f"    Brier   3a={b_a:.4f}  3e={b_e:.4f}  3e-3a={b_e - b_a:+.4f}")
            print(f"    ECE     3a={ece_a:.4f}  3e={ece_e:.4f}  3e-3a={ece_e - ece_a:+.4f}")
            print(f"    Mean    3a={mean_a:.3f}  3e={mean_e:.3f}  (truth={obs_rate:.3f}) -> 3a bias={mean_a - obs_rate:+.3f}, 3e bias={mean_e - obs_rate:+.3f}")
            print(f"    Var     3a={var_a:.4f}  3e={var_e:.4f}  (3e sharper if larger)")
            print(f"    %p<0.1  3a={frac_low_a:.2f}  3e={frac_low_e:.2f}")
            print(f"    %p>0.9  3a={frac_high_a:.2f}  3e={frac_high_e:.2f}")

            summary_rows.append({
                "station": station, "lead": lead, "n_hourly": len(sub),
                "obs_rate": obs_rate,
                "brier_3a": b_a, "brier_3e": b_e,
                "ece_3a": ece_a, "ece_3e": ece_e,
                "mean_3a": mean_a, "mean_3e": mean_e,
                "var_3a": var_a, "var_3e": var_e,
                "frac_low_3a": frac_low_a, "frac_low_3e": frac_low_e,
                "frac_high_3a": frac_high_a, "frac_high_3e": frac_high_e,
            })

        # --- Per-day MC bias decomposition ---
        # For each (lead, target_date): get the daytime q-vector + truth seq,
        # run iid MC from 3a's q and 3e's q for each window, compute prediction
        # error vs label, regress against q-statistics. We do this for window=4
        # only (representative middle case) to keep the output focused.
        print(f"\n  per-day MC error decomposition (window=4):")
        bias_rows: list[dict] = []
        for lead in leads:
            sub = merged[merged["lead"] == lead]
            for target_date, day in sub.groupby("target_date"):
                if len(day) != 9:
                    continue
                day = day.sort_values("valid_time")
                q_a = day["p_wet_3a"].to_numpy()
                q_e = day["p_wet_3e"].to_numpy()
                obs_seq = day["observed_wet"].astype(np.int32).to_numpy()
                label = 1 if has_contiguous_dry_block(obs_seq, 4) else 0
                p_a = prob_dry_window_mc(q_a, 4, MC_SAMPLES, rng)
                p_e = prob_dry_window_mc(q_e, 4, MC_SAMPLES, rng)
                bias_rows.append({
                    "lead": lead, "target_date": target_date,
                    "q_mean_3a": float(q_a.mean()), "q_mean_3e": float(q_e.mean()),
                    "q_var_3a": float(q_a.var()),  "q_var_3e": float(q_e.var()),
                    "q_max_3a": float(q_a.max()),  "q_max_3e": float(q_e.max()),
                    "p_dry_3a": p_a, "p_dry_3e": p_e,
                    "label": label,
                    "err_3a": p_a - label, "err_3e": p_e - label,
                })
        if bias_rows:
            bdf = pd.DataFrame(bias_rows)
            # Aggregate metrics
            print(f"    mean |err| 3a={bdf['err_3a'].abs().mean():.4f}  3e={bdf['err_3e'].abs().mean():.4f}")
            print(f"    mean err (bias) 3a={bdf['err_3a'].mean():+.4f}  3e={bdf['err_3e'].mean():+.4f}")
            # Compare error by q-vector sharpness bucket
            for q_label, expr in [("low-q days  (mean < 0.20)", bdf["q_mean_3a"] < 0.20),
                                  ("mid-q days  (0.20-0.40)",   (bdf["q_mean_3a"] >= 0.20) & (bdf["q_mean_3a"] < 0.40)),
                                  ("high-q days (mean >= 0.40)", bdf["q_mean_3a"] >= 0.40)]:
                bin_df = bdf[expr]
                if len(bin_df) > 0:
                    print(f"    {q_label:30s} n={len(bin_df):4d}: "
                          f"3a |err|={bin_df['err_3a'].abs().mean():.4f}  "
                          f"3e |err|={bin_df['err_3e'].abs().mean():.4f}  "
                          f"3a bias={bin_df['err_3a'].mean():+.4f}  "
                          f"3e bias={bin_df['err_3e'].mean():+.4f}")

    # Save the per-(station, lead) summary as CSV for later cross-ref.
    if summary_rows:
        out_path = ROOT / "reports" / "3a_vs_3e_daytime_diagnostic.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\nwrote {len(summary_rows)} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
