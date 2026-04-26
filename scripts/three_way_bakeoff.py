"""Apples-to-apples 3-way bake-off on fixed identical test rows.

Variants per (target, station/lead) cell:
  A. Prior     = 5-model + full backfill + no bagging  (the pre-migration champion)
  B. 6-only    = 6-model + restricted post-2024-09 + no bagging  (UKMO restored only)
  C. New       = 6-model + restricted post-2024-09 + bagging     (UKMO + bagging)

All three are trained on rows with ValidTime < cutoff and scored on rows with
ValidTime >= cutoff. The cutoff is the chronological 80% point of the
post-2024-09 window (so all three have access to the same test rows). The
prior variant gets MORE training data — that's the whole point: we want to
know whether the date-filter cost outweighs the UKMO benefit on the same
test set.
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
sys.path.insert(0, str(ROOT / "scripts"))
from restricted_window_bakeoff import (  # noqa: E402
    FORECASTS, ERA5, RAINFALL, WINDOW_START, LEADS, WET_THRESHOLD_MM,
    STATIONS, STATION_CODES, LGB_REG_PARAMS, LGB_BIN_PARAMS,
    NUM_ITERATIONS, EARLY_STOPPING, _calendar, _spread,
)

# Bagging knobs match the production change (TemperatureTrainer / PrecipOccurrenceTrainer):
LGB_BAGGED_REG = dict(LGB_REG_PARAMS, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8)
LGB_BAGGED_BIN = dict(LGB_BIN_PARAMS, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8)
TEST_FRACTION = 0.2


def _fit_predict(X_tr, y_tr, X_te, feat_cols, params):
    cut = max(int(len(X_tr) * 0.85), 10)
    train_set = lgb.Dataset(X_tr[:cut], label=y_tr[:cut], feature_name=feat_cols)
    val_set = lgb.Dataset(X_tr[cut:], label=y_tr[cut:], feature_name=feat_cols, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_ITERATIONS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return booster.predict(X_te, num_iteration=booster.best_iteration)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def temp_three_way():
    print("\n=== Temperature 2b lean: 3-way bake-off (fixed test set) ===")
    con = duckdb.connect()
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()
    era_glob = (ERA5 / "**" / "*.parquet").as_posix()

    rows_out = []
    model_cols_5 = ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_gem"]
    model_cols_6 = ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem"]

    for lead in LEADS:
        # Pull EVERY available row first (no date filter); we'll slice in pandas.
        sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model, Temperature2m,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND RunTimeSource='offset_day'
              AND LeadHours={lead} AND Temperature2m IS NOT NULL
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
        )
        SELECT p.ValidTimeUtc, p.temp_gfs, p.temp_ecmwf, p.temp_icon, p.temp_mf, p.temp_ukmo, p.temp_gem, e.era5_temp
        FROM pivoted p JOIN era e USING (ValidTimeUtc)
        WHERE p.temp_gfs IS NOT NULL AND p.temp_ecmwf IS NOT NULL
          AND p.temp_icon IS NOT NULL AND p.temp_mf IS NOT NULL AND p.temp_gem IS NOT NULL
        ORDER BY p.ValidTimeUtc
        """
        df = con.execute(sql).fetch_df()
        df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
        df = _calendar(df)

        # Fixed test cutoff: 80% point of the POST-WINDOW data, applied to ALL variants.
        post_window = df[df["ValidTimeUtc"] >= pd.Timestamp(WINDOW_START)].reset_index(drop=True)
        cut_idx = int(len(post_window) * (1 - TEST_FRACTION))
        cutoff_time = post_window.iloc[cut_idx]["ValidTimeUtc"]
        train_full  = df[df["ValidTimeUtc"] < cutoff_time].reset_index(drop=True)
        train_post  = df[(df["ValidTimeUtc"] >= pd.Timestamp(WINDOW_START)) &
                         (df["ValidTimeUtc"] < cutoff_time)].reset_index(drop=True)
        test_df     = df[df["ValidTimeUtc"] >= cutoff_time].reset_index(drop=True)
        y_te = test_df["era5_temp"].to_numpy(dtype="float64")

        print(f"\n--- Lead {lead}h --- "
              f"train_full={len(train_full):,}  train_post={len(train_post):,}  test={len(test_df):,}  "
              f"cutoff={cutoff_time:%Y-%m-%d %H:%M}Z")

        for variant_lbl, train_df, model_cols, params in [
            ("A_prior",  train_full, model_cols_5, LGB_REG_PARAMS),
            ("B_6model", train_post, model_cols_6, LGB_REG_PARAMS),
            ("C_new",    train_post, model_cols_6, LGB_BAGGED_REG),
            ("D_5mod_full_bag", train_full, model_cols_5, LGB_BAGGED_REG),
        ]:
            d_tr = _spread(train_df, model_cols, "temp")
            d_te = _spread(test_df,  model_cols, "temp")
            feat = model_cols + ["temp_mean", "temp_std", "temp_range",
                                 "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
            X_tr = d_tr[feat].to_numpy(dtype="float64")
            y_tr = d_tr["era5_temp"].to_numpy(dtype="float64")
            X_te = d_te[feat].to_numpy(dtype="float64")
            t0 = time.time()
            p = _fit_predict(X_tr, y_tr, X_te, feat, params)
            mae = float(np.mean(np.abs(p - y_te)))
            rmse = float(np.sqrt(np.mean((p - y_te) ** 2)))
            print(f"  {variant_lbl}:  MAE={mae:.4f}  RMSE={rmse:.4f}  fit={time.time()-t0:.1f}s")
            rows_out.append(dict(target="temp", lead=lead, variant=variant_lbl,
                                 train_n=len(train_df), test_n=len(test_df),
                                 mae=mae, rmse=rmse))
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------

def precip_three_way():
    print("\n=== Precip 3a lean: 3-way bake-off (fixed test set) ===")
    con = duckdb.connect()
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()

    sql = f"""
    WITH latest AS (
        SELECT ValidTimeUtc, LeadHours, Model, Precipitation, PrecipitationProbability,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours, Model ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
        WHERE LocationName='bonehill_rocks' AND RunTimeSource='offset_day'
          AND LeadHours IN ({','.join(map(str, LEADS))}) AND Precipitation IS NOT NULL
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
    FROM latest WHERE rn=1 GROUP BY ValidTimeUtc, LeadHours
    """
    forecasts = con.execute(sql).fetch_df()
    forecasts["ValidTimeUtc"] = pd.to_datetime(forecasts["ValidTimeUtc"], utc=True).dt.tz_localize(None)

    rows_out = []
    for station in STATIONS:
        rain_path = (RAINFALL / f"station={station}" / "**" / "*.parquet").as_posix()
        truth_sql = f"""
            WITH r AS (
                SELECT date_trunc('hour', ObservedTimeUtc) AS hour_utc, Value15MinMm
                FROM read_parquet('{rain_path}', hive_partitioning=false, union_by_name=true)
                WHERE Value15MinMm IS NOT NULL
            )
            SELECT hour_utc AS ValidTimeUtc, SUM(Value15MinMm) AS precip_mm_h, COUNT(*) AS n
            FROM r GROUP BY hour_utc HAVING COUNT(*) = 4 ORDER BY hour_utc
        """
        truth = con.execute(truth_sql).fetch_df()
        truth["ValidTimeUtc"] = pd.to_datetime(truth["ValidTimeUtc"], utc=True).dt.tz_localize(None)
        truth["observed_wet"] = (truth["precip_mm_h"] >= WET_THRESHOLD_MM).astype("int8")

        df = truth[["ValidTimeUtc", "observed_wet"]].merge(forecasts, on="ValidTimeUtc", how="inner")
        df = df.dropna(subset=[c for c in df.columns if c.startswith("precip_") and not c.endswith("ukmo")])
        df = _calendar(df).sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)

        for lead in LEADS:
            sub = df[df["LeadHours"] == lead].reset_index(drop=True)
            post_window = sub[sub["ValidTimeUtc"] >= pd.Timestamp(WINDOW_START)].reset_index(drop=True)
            cut_idx = int(len(post_window) * (1 - TEST_FRACTION))
            cutoff_time = post_window.iloc[cut_idx]["ValidTimeUtc"]
            train_full = sub[sub["ValidTimeUtc"] < cutoff_time].reset_index(drop=True)
            train_post = sub[(sub["ValidTimeUtc"] >= pd.Timestamp(WINDOW_START)) &
                             (sub["ValidTimeUtc"] < cutoff_time)].reset_index(drop=True)
            test_df    = sub[sub["ValidTimeUtc"] >= cutoff_time].reset_index(drop=True)
            y_te = test_df["observed_wet"].to_numpy()
            wet_rate = float(y_te.mean())
            clim_brier = float(np.mean((wet_rate - y_te) ** 2))

            print(f"\n--- {STATION_CODES[station]} lead {lead}h ---  "
                  f"train_full={len(train_full):,}  train_post={len(train_post):,}  test={len(test_df):,}  wet={wet_rate:.3f}")

            for variant_lbl, train_df_, prec_cols, prob_cols, params in [
                ("A_prior",  train_full,
                 ["precip_gfs","precip_ecmwf","precip_icon","precip_mf","precip_gem"],
                 ["prob_gfs","prob_ecmwf","prob_icon","prob_mf","prob_gem"], LGB_BIN_PARAMS),
                ("B_6model", train_post,
                 ["precip_gfs","precip_ecmwf","precip_icon","precip_mf","precip_ukmo","precip_gem"],
                 ["prob_gfs","prob_ecmwf","prob_icon","prob_mf","prob_ukmo","prob_gem"], LGB_BIN_PARAMS),
                ("C_new",    train_post,
                 ["precip_gfs","precip_ecmwf","precip_icon","precip_mf","precip_ukmo","precip_gem"],
                 ["prob_gfs","prob_ecmwf","prob_icon","prob_mf","prob_ukmo","prob_gem"], LGB_BAGGED_BIN),
                ("D_5mod_full_bag", train_full,
                 ["precip_gfs","precip_ecmwf","precip_icon","precip_mf","precip_gem"],
                 ["prob_gfs","prob_ecmwf","prob_icon","prob_mf","prob_gem"], LGB_BAGGED_BIN),
            ]:
                trv = _spread(train_df_, prec_cols, "precip")
                tev = _spread(test_df,   prec_cols, "precip")
                for c in prob_cols:
                    m = trv[c].mean()
                    if np.isnan(m): m = 0.0
                    trv[c] = trv[c].fillna(m); tev[c] = tev[c].fillna(m)
                feat = prec_cols + prob_cols + ["precip_mean","precip_std","precip_range",
                                                "hour_sin","hour_cos","doy_sin","doy_cos"]
                X_tr = trv[feat].to_numpy(dtype="float64")
                y_tr = trv["observed_wet"].to_numpy()
                X_te = tev[feat].to_numpy(dtype="float64")

                t0 = time.time()
                p = _fit_predict(X_tr, y_tr, X_te, feat, params)
                brier = float(brier_score_loss(y_te, p))
                bss = 1.0 - brier / clim_brier if clim_brier > 0 else 0.0
                print(f"  {variant_lbl}:  Brier={brier:.4f}  BSS={bss:+.3f}  fit={time.time()-t0:.1f}s")
                rows_out.append(dict(target="precip", station=STATION_CODES[station], lead=lead,
                                     variant=variant_lbl, train_n=len(train_df_), test_n=len(test_df),
                                     wet_rate=wet_rate, brier=brier, bss=bss))
    return pd.DataFrame(rows_out)


def _summary_table(df, target_label, metric_col, lower_is_better=True):
    print(f"\n=== {target_label} 3-way summary ===\n")
    keys = ["station", "lead"] if "station" in df.columns else ["lead"]
    pivot = df.pivot_table(index=keys, columns="variant", values=metric_col).round(4)
    pivot["B_vs_A_pct"] = (pivot["A_prior"] - pivot["B_6model"]) / pivot["A_prior"] * 100
    pivot["C_vs_A_pct"] = (pivot["A_prior"] - pivot["C_new"])    / pivot["A_prior"] * 100
    pivot["C_vs_B_pct"] = (pivot["B_6model"] - pivot["C_new"])   / pivot["B_6model"] * 100
    if not lower_is_better:
        pivot["B_vs_A_pct"] = -pivot["B_vs_A_pct"]
        pivot["C_vs_A_pct"] = -pivot["C_vs_A_pct"]
        pivot["C_vs_B_pct"] = -pivot["C_vs_B_pct"]
    pivot = pivot.round({"B_vs_A_pct": 2, "C_vs_A_pct": 2, "C_vs_B_pct": 2})
    print(pivot.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["temp", "precip", "both"], default="both")
    args = ap.parse_args()

    out = ROOT / "data" / "reports"
    out.mkdir(parents=True, exist_ok=True)

    if args.target in ("temp", "both"):
        temp_df = temp_three_way()
        temp_df.to_csv(out / "three_way_bakeoff_temp.csv", index=False)
        _summary_table(temp_df, "Temperature MAE °C (positive % = better than A_prior)", "mae")

    if args.target in ("precip", "both"):
        precip_df = precip_three_way()
        precip_df.to_csv(out / "three_way_bakeoff_precip.csv", index=False)
        _summary_table(precip_df, "Precip Brier (positive % = better than A_prior)", "brier")


if __name__ == "__main__":
    main()
