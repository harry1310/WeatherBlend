"""5-model vs 6-model bake-off restricted to the post-UKMO-archive window.

Background (2026-04-26): the 26% UKMO field-nullness in our backfill turned
out to be a contiguous 7-month time block (2024-01 → 2024-08-15) where
Open-Meteo had not yet started archiving UKMO weather fields. Our
chronological 80/20 split puts that block almost entirely in TRAIN, so the
6-model blender had to learn a dual regime ("UKMO=NaN for 7 months, then
UKMO=valid forever"). The previous "5-model wins" finding may be train-time
poisoning, not evidence that UKMO genuinely hurts.

This script retrains both 5-model and 6-model variants of the temp 2b lean
and precip 3a lean blenders on the **post-2024-09-01** window only, with
identical hyperparameters and chronological 80/20 split inside that window,
and reports head-to-head MAE / Brier per (target, station, lead).

Usage:
    .venv/Scripts/python.exe scripts/restricted_window_bakeoff.py --target temp
    .venv/Scripts/python.exe scripts/restricted_window_bakeoff.py --target precip
    .venv/Scripts/python.exe scripts/restricted_window_bakeoff.py --target both
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "data" / "forecasts" / "location=bonehill_rocks"
ERA5 = ROOT / "data" / "truth" / "era5" / "location=bonehill_rocks"
RAINFALL = ROOT / "data" / "truth" / "rainfall" / "location=bonehill_rocks"

WINDOW_START = "2024-09-01"
LEADS = (24, 48, 72)
WET_THRESHOLD_MM = 0.1
TEST_FRACTION = 0.2
STATIONS = ("Bellever Dartmoor", "Bovey Tracey", "Dartmoor nr Hexworthy")
STATION_CODES = {
    "Bellever Dartmoor": "Bellever",
    "Bovey Tracey": "Bovey",
    "Dartmoor nr Hexworthy": "Hexworthy",
}

ALL_MODELS_TEMP = (
    ("gfs_seamless", "temp_gfs"),
    ("ecmwf_ifs025", "temp_ecmwf"),
    ("icon_seamless", "temp_icon"),
    ("meteofrance_seamless", "temp_mf"),
    ("ukmo_seamless", "temp_ukmo"),
    ("gem_seamless", "temp_gem"),
)
ALL_MODELS_PRECIP = (
    ("gfs_seamless", "precip_gfs"),
    ("ecmwf_ifs025", "precip_ecmwf"),
    ("icon_seamless", "precip_icon"),
    ("meteofrance_seamless", "precip_mf"),
    ("ukmo_seamless", "precip_ukmo"),
    ("gem_seamless", "precip_gem"),
)

LGB_REG_PARAMS = dict(
    objective="regression_l1",
    metric="mae",
    num_leaves=31,
    learning_rate=0.05,
    min_data_in_leaf=20,
    lambda_l1=0.1,
    lambda_l2=0.1,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    verbose=-1,
    seed=42,
    num_threads=0,
)
LGB_BIN_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    num_leaves=31,
    learning_rate=0.05,
    min_data_in_leaf=20,
    lambda_l1=0.1,
    lambda_l2=0.1,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    verbose=-1,
    seed=42,
    num_threads=0,
)
NUM_ITERATIONS = 500
EARLY_STOPPING = 30


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h = df["ValidTimeUtc"].dt.hour
    doy = df["ValidTimeUtc"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    return df


def _spread(df: pd.DataFrame, model_cols: list[str], prefix: str) -> pd.DataFrame:
    """Compute mean/std/range across the listed model columns, ignoring NaN."""
    df = df.copy()
    arr = df[model_cols].to_numpy(dtype="float64")
    df[f"{prefix}_mean"] = np.nanmean(arr, axis=1)
    df[f"{prefix}_std"] = np.nanstd(arr, axis=1)  # population std, NaN-safe
    df[f"{prefix}_range"] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
    return df


def _train_eval_lgb(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    params: dict, feature_names: list[str], task: str,
) -> tuple[float, float, np.ndarray]:
    """Train LightGBM with early stopping. Returns (primary, secondary, predictions).
    primary = MAE for regression, Brier for classification."""
    cut = max(int(len(X_tr) * 0.85), 10)
    X_tr_main, X_val = X_tr[:cut], X_tr[cut:]
    y_tr_main, y_val = y_tr[:cut], y_tr[cut:]
    train_set = lgb.Dataset(X_tr_main, label=y_tr_main, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_ITERATIONS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    p = booster.predict(X_te, num_iteration=booster.best_iteration)
    if task == "reg":
        mae = float(np.mean(np.abs(p - y_te)))
        rmse = float(np.sqrt(np.mean((p - y_te) ** 2)))
        return mae, rmse, p
    else:
        b = float(brier_score_loss(y_te, p))
        clim = float(y_te.mean())
        clim_brier = float(np.mean((clim - y_te) ** 2))
        bss = 1.0 - b / clim_brier if clim_brier > 0 else 0.0
        return b, bss, p


# ---------------------------------------------------------------------------
# Temperature 2b lean
# ---------------------------------------------------------------------------

def temp_bakeoff(restrict_to_ukmo_present: bool = False) -> pd.DataFrame:
    """For each lead, train 5-model and 6-model lean LightGBM on the
    post-WINDOW_START window. If restrict_to_ukmo_present, also evaluate
    on the subset of test rows where UKMO is non-null (apples-to-apples on
    the row population both can score)."""
    print(f"\n=== Temperature 2b lean bake-off (window: {WINDOW_START}+) ===")
    con = duckdb.connect()
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()
    era_glob = (ERA5 / "**" / "*.parquet").as_posix()

    rows = []
    for lead in LEADS:
        sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model, Temperature2m,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND RunTimeSource='offset_day'
              AND LeadHours={lead} AND Temperature2m IS NOT NULL
              AND ValidTimeUtc >= TIMESTAMP '{WINDOW_START}'
        ),
        pivoted AS (
            SELECT ValidTimeUtc,
                MAX(CASE WHEN Model='gfs_seamless' THEN Temperature2m END) AS temp_gfs,
                MAX(CASE WHEN Model='ecmwf_ifs025' THEN Temperature2m END) AS temp_ecmwf,
                MAX(CASE WHEN Model='icon_seamless' THEN Temperature2m END) AS temp_icon,
                MAX(CASE WHEN Model='meteofrance_seamless' THEN Temperature2m END) AS temp_mf,
                MAX(CASE WHEN Model='ukmo_seamless' THEN Temperature2m END) AS temp_ukmo,
                MAX(CASE WHEN Model='gem_seamless' THEN Temperature2m END) AS temp_gem
            FROM latest WHERE rn=1 GROUP BY ValidTimeUtc
        ),
        era AS (
            SELECT ValidTimeUtc, Temperature2m AS era5_temp
            FROM read_parquet('{era_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND Temperature2m IS NOT NULL
              AND ValidTimeUtc >= TIMESTAMP '{WINDOW_START}'
        )
        SELECT p.ValidTimeUtc,
               p.temp_gfs, p.temp_ecmwf, p.temp_icon, p.temp_mf, p.temp_ukmo, p.temp_gem,
               e.era5_temp
        FROM pivoted p JOIN era e USING (ValidTimeUtc)
        WHERE p.temp_gfs IS NOT NULL AND p.temp_ecmwf IS NOT NULL
          AND p.temp_icon IS NOT NULL AND p.temp_mf IS NOT NULL AND p.temp_gem IS NOT NULL
        ORDER BY p.ValidTimeUtc
        """
        df = con.execute(sql).fetch_df()
        df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])

        n_total = len(df)
        n_ukmo_present = int(df["temp_ukmo"].notna().sum())
        print(f"\n--- Lead {lead}h ---  rows={n_total:,}  ukmo non-null={n_ukmo_present:,} ({n_ukmo_present/n_total*100:.1f}%)")

        df = _calendar(df)

        for variant, model_cols in [
            ("5-model", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_gem"]),
            ("6-model", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem"]),
        ]:
            d = df.copy()
            if variant == "5-model":
                # Drop UKMO column entirely so spread is over 5
                d2 = _spread(d, model_cols, "temp")
                feat_cols = model_cols + ["temp_mean", "temp_std", "temp_range",
                                          "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
            else:
                d2 = _spread(d, model_cols, "temp")
                feat_cols = model_cols + ["temp_mean", "temp_std", "temp_range",
                                          "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

            split = int(len(d2) * (1 - TEST_FRACTION))
            tr, te = d2.iloc[:split], d2.iloc[split:]
            X_tr = tr[feat_cols].to_numpy(dtype="float64")
            y_tr = tr["era5_temp"].to_numpy(dtype="float64")
            X_te = te[feat_cols].to_numpy(dtype="float64")
            y_te = te["era5_temp"].to_numpy(dtype="float64")

            t0 = time.time()
            mae_full, rmse_full, p_te = _train_eval_lgb(
                X_tr, y_tr, X_te, y_te, LGB_REG_PARAMS, feat_cols, task="reg")
            dt = time.time() - t0

            # On-UKMO-present subset (compare both variants on the same rows)
            mask = te["temp_ukmo"].notna().to_numpy()
            mae_subset = float(np.mean(np.abs(p_te[mask] - y_te[mask]))) if mask.sum() > 0 else float("nan")

            print(f"  {variant}:  test MAE all={mae_full:.4f}  RMSE={rmse_full:.4f}  "
                  f"MAE on UKMO-present subset={mae_subset:.4f} (n={int(mask.sum()):,})  fit={dt:.1f}s")
            rows.append(dict(
                target="temp", lead=lead, variant=variant,
                test_n=len(te), ukmo_present_in_test=int(mask.sum()),
                mae=mae_full, rmse=rmse_full, mae_ukmo_present=mae_subset,
            ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Precip 3a lean
# ---------------------------------------------------------------------------

def precip_bakeoff() -> pd.DataFrame:
    print(f"\n=== Precip 3a lean bake-off (window: {WINDOW_START}+) ===")
    con = duckdb.connect()
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()

    # Build the per-(valid_time, lead) feature frame ONCE for all leads
    # (precip pivot is the same shape for all leads — just filter by lead later).
    sql = f"""
    WITH latest AS (
        SELECT ValidTimeUtc, LeadHours, Model, Precipitation, PrecipitationProbability,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours, Model ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
        WHERE LocationName='bonehill_rocks' AND RunTimeSource='offset_day'
          AND LeadHours IN ({','.join(map(str, LEADS))}) AND Precipitation IS NOT NULL
          AND ValidTimeUtc >= TIMESTAMP '{WINDOW_START}'
    )
    SELECT ValidTimeUtc, LeadHours,
        MAX(CASE WHEN Model='gfs_seamless' THEN Precipitation END) AS precip_gfs,
        MAX(CASE WHEN Model='ecmwf_ifs025' THEN Precipitation END) AS precip_ecmwf,
        MAX(CASE WHEN Model='icon_seamless' THEN Precipitation END) AS precip_icon,
        MAX(CASE WHEN Model='meteofrance_seamless' THEN Precipitation END) AS precip_mf,
        MAX(CASE WHEN Model='ukmo_seamless' THEN Precipitation END) AS precip_ukmo,
        MAX(CASE WHEN Model='gem_seamless' THEN Precipitation END) AS precip_gem,
        MAX(CASE WHEN Model='gfs_seamless' THEN PrecipitationProbability END) AS prob_gfs,
        MAX(CASE WHEN Model='ecmwf_ifs025' THEN PrecipitationProbability END) AS prob_ecmwf,
        MAX(CASE WHEN Model='icon_seamless' THEN PrecipitationProbability END) AS prob_icon,
        MAX(CASE WHEN Model='meteofrance_seamless' THEN PrecipitationProbability END) AS prob_mf,
        MAX(CASE WHEN Model='ukmo_seamless' THEN PrecipitationProbability END) AS prob_ukmo,
        MAX(CASE WHEN Model='gem_seamless' THEN PrecipitationProbability END) AS prob_gem
    FROM latest WHERE rn=1
    GROUP BY ValidTimeUtc, LeadHours
    """
    forecasts = con.execute(sql).fetch_df()
    forecasts["ValidTimeUtc"] = pd.to_datetime(forecasts["ValidTimeUtc"])
    print(f"  forecasts loaded: {len(forecasts):,} (valid_time, lead) rows")

    rows = []
    for station in STATIONS:
        # Truth: hourly aggregated rainfall (same logic as data.py:_load_rainfall_truth)
        rain_path = (RAINFALL / f"station={station}" / "**" / "*.parquet").as_posix()
        truth_sql = f"""
            WITH r AS (
                SELECT date_trunc('hour', ObservedTimeUtc) AS hour_utc, Value15MinMm
                FROM read_parquet('{rain_path}', hive_partitioning=false, union_by_name=true)
                WHERE Value15MinMm IS NOT NULL
                  AND ObservedTimeUtc >= TIMESTAMP '{WINDOW_START}'
            )
            SELECT hour_utc AS ValidTimeUtc, SUM(Value15MinMm) AS precip_mm_h, COUNT(*) AS n
            FROM r GROUP BY hour_utc HAVING COUNT(*) = 4
            ORDER BY hour_utc
        """
        truth = con.execute(truth_sql).fetch_df()
        truth["ValidTimeUtc"] = pd.to_datetime(truth["ValidTimeUtc"])
        truth["observed_wet"] = (truth["precip_mm_h"] >= WET_THRESHOLD_MM).astype("int8")

        # Inner join — precip on (ValidTimeUtc) since forecasts have lead too
        df = truth[["ValidTimeUtc", "observed_wet"]].merge(forecasts, on="ValidTimeUtc", how="inner")
        df = df.dropna(subset=[c for c in df.columns if c.startswith("precip_") and not c.endswith("ukmo")])
        df = _calendar(df).sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)

        for lead in LEADS:
            sub = df[df["LeadHours"] == lead].reset_index(drop=True)
            if len(sub) < 200:
                print(f"  {station} lead {lead}h: only {len(sub)} rows, skipping")
                continue

            n_ukmo_present = int(sub["precip_ukmo"].notna().sum())
            print(f"\n--- {STATION_CODES[station]} lead {lead}h ---  rows={len(sub):,}  ukmo non-null={n_ukmo_present:,} ({n_ukmo_present/len(sub)*100:.1f}%)")

            split = int(len(sub) * (1 - TEST_FRACTION))
            tr, te = sub.iloc[:split], sub.iloc[split:]

            for variant, prec_cols, prob_cols in [
                ("5-model",
                 ["precip_gfs", "precip_ecmwf", "precip_icon", "precip_mf", "precip_gem"],
                 ["prob_gfs", "prob_ecmwf", "prob_icon", "prob_mf", "prob_gem"]),
                ("6-model",
                 ["precip_gfs", "precip_ecmwf", "precip_icon", "precip_mf", "precip_ukmo", "precip_gem"],
                 ["prob_gfs", "prob_ecmwf", "prob_icon", "prob_mf", "prob_ukmo", "prob_gem"]),
            ]:
                # Spread on precip cols
                trv = _spread(tr, prec_cols, "precip")
                tev = _spread(te, prec_cols, "precip")

                # Fill prob_* with column mean from train (probs are mostly NaN
                # but follow the C# convention of filling)
                for c in prob_cols:
                    m = trv[c].mean()
                    if np.isnan(m): m = 0.0
                    trv[c] = trv[c].fillna(m)
                    tev[c] = tev[c].fillna(m)

                feat_cols = prec_cols + prob_cols + ["precip_mean", "precip_std", "precip_range",
                                                    "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

                X_tr = trv[feat_cols].to_numpy(dtype="float64")
                y_tr = trv["observed_wet"].to_numpy()
                X_te = tev[feat_cols].to_numpy(dtype="float64")
                y_te = tev["observed_wet"].to_numpy()

                t0 = time.time()
                brier_full, bss_full, p_te = _train_eval_lgb(
                    X_tr, y_tr, X_te, y_te, LGB_BIN_PARAMS, feat_cols, task="bin")
                dt = time.time() - t0

                mask = te["precip_ukmo"].notna().to_numpy()
                if mask.sum() > 10:
                    brier_subset = float(brier_score_loss(y_te[mask], p_te[mask]))
                else:
                    brier_subset = float("nan")

                print(f"  {variant}:  Brier all={brier_full:.4f}  BSS={bss_full:+.3f}  "
                      f"Brier on UKMO-present subset={brier_subset:.4f} (n={int(mask.sum()):,})  fit={dt:.1f}s")
                rows.append(dict(
                    target="precip", station=STATION_CODES[station], lead=lead, variant=variant,
                    test_n=len(te), wet_rate=float(y_te.mean()),
                    ukmo_present_in_test=int(mask.sum()),
                    brier=brier_full, bss=bss_full, brier_ukmo_present=brier_subset,
                ))

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["temp", "precip", "both"], default="both")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target in ("temp", "both"):
        temp_results = temp_bakeoff()
        temp_results.to_csv(out_dir / "ukmo_restricted_bakeoff_temp.csv", index=False)
        print("\n=== Temp summary (test MAE: all rows / UKMO-present subset) ===")
        for lead in LEADS:
            sub = temp_results[temp_results.lead == lead]
            if len(sub) == 0: continue
            five = sub[sub.variant == "5-model"].iloc[0]
            six = sub[sub.variant == "6-model"].iloc[0]
            delta = (five.mae - six.mae) / five.mae * 100
            delta_subset = (five.mae_ukmo_present - six.mae_ukmo_present) / five.mae_ukmo_present * 100
            winner = "6-model" if six.mae < five.mae else "5-model"
            winner_sub = "6-model" if six.mae_ukmo_present < five.mae_ukmo_present else "5-model"
            print(f"  lead {lead}h:  5={five.mae:.4f}  6={six.mae:.4f}  Δ={delta:+.2f}% → {winner}    "
                  f"|  on UKMO-present:  5={five.mae_ukmo_present:.4f}  6={six.mae_ukmo_present:.4f}  Δ={delta_subset:+.2f}% → {winner_sub}")

    if args.target in ("precip", "both"):
        precip_results = precip_bakeoff()
        precip_results.to_csv(out_dir / "ukmo_restricted_bakeoff_precip.csv", index=False)
        print("\n=== Precip summary (test Brier: all rows / UKMO-present subset) ===")
        for station in sorted(set(precip_results.station)):
            for lead in LEADS:
                sub = precip_results[(precip_results.station == station) & (precip_results.lead == lead)]
                if len(sub) == 0: continue
                five = sub[sub.variant == "5-model"].iloc[0]
                six = sub[sub.variant == "6-model"].iloc[0]
                delta = (five.brier - six.brier) / five.brier * 100
                delta_subset = (five.brier_ukmo_present - six.brier_ukmo_present) / five.brier_ukmo_present * 100
                winner = "6-model" if six.brier < five.brier else "5-model"
                winner_sub = "6-model" if six.brier_ukmo_present < five.brier_ukmo_present else "5-model"
                print(f"  {station:9s} lead {lead}h:  5={five.brier:.4f}  6={six.brier:.4f}  Δ={delta:+.2f}% → {winner}    "
                      f"|  on UKMO-present:  5={five.brier_ukmo_present:.4f}  6={six.brier_ukmo_present:.4f}  Δ={delta_subset:+.2f}% → {winner_sub}")


if __name__ == "__main__":
    main()
