"""4-way bake-off for the three remaining Elements (wind / humidity / cloud)
on a fixed identical test set.

Variants per (target, lead) cell:
  A. Prior     = current pre-migration champion (5-model — actually 4-model for
                 wind which has never had MF — full backfill, no bagging)
  B. 6-restr   = the 6-model + post-2024-09 restricted window experiment
                 (5-model for wind since wind never has MF)
  C. C_new     = B + bagging (the migration we currently have on disk)
  D. 5+full+bag = roll back UKMO, keep full backfill, keep bagging

This mirrors the temp/precip 4-way; we want to confirm the same conclusion
holds for the Elements before completing the rollback.

Radiation skipped — its 5-model membership doesn't include UKMO, so the
"data-window vs UKMO" question doesn't apply to it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from restricted_window_bakeoff import (  # noqa: E402
    FORECASTS, ERA5, WINDOW_START, LEADS,
    LGB_REG_PARAMS, NUM_ITERATIONS, EARLY_STOPPING,
    _calendar, _spread,
)

LGB_BAGGED_REG = dict(LGB_REG_PARAMS, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8)
TEST_FRACTION = 0.2


def _fit_predict(X_tr, y_tr, X_te, feat, params):
    cut = max(int(len(X_tr) * 0.85), 10)
    train_set = lgb.Dataset(X_tr[:cut], label=y_tr[:cut], feature_name=feat)
    val_set = lgb.Dataset(X_tr[cut:], label=y_tr[cut:], feature_name=feat, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_ITERATIONS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return booster.predict(X_te, num_iteration=booster.best_iteration)


def _split(df: pd.DataFrame):
    """Return (train_full, train_post, test) — fixed-cutoff at 80% of post-window."""
    df = df.sort_values("ValidTimeUtc").reset_index(drop=True)
    post_window = df[df["ValidTimeUtc"] >= pd.Timestamp(WINDOW_START)].reset_index(drop=True)
    cut_idx = int(len(post_window) * (1 - TEST_FRACTION))
    cutoff_time = post_window.iloc[cut_idx]["ValidTimeUtc"]
    train_full = df[df["ValidTimeUtc"] < cutoff_time].reset_index(drop=True)
    train_post = df[(df["ValidTimeUtc"] >= pd.Timestamp(WINDOW_START)) &
                    (df["ValidTimeUtc"] < cutoff_time)].reset_index(drop=True)
    test = df[df["ValidTimeUtc"] >= cutoff_time].reset_index(drop=True)
    return train_full, train_post, test, cutoff_time


def _eval(train_df, test_df, model_cols, prefix, truth_col, params):
    d_tr = _spread(train_df, model_cols, prefix)
    d_te = _spread(test_df,  model_cols, prefix)
    feat = model_cols + [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_range",
                         "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    X_tr = d_tr[feat].to_numpy(dtype="float64")
    y_tr = d_tr[truth_col].to_numpy(dtype="float64")
    X_te = d_te[feat].to_numpy(dtype="float64")
    y_te = d_te[truth_col].to_numpy(dtype="float64")
    p = _fit_predict(X_tr, y_tr, X_te, feat, params)
    return float(np.mean(np.abs(p - y_te))), float(np.sqrt(np.mean((p - y_te) ** 2)))


# ---------------------------------------------------------------------------
# Wind — A: 4-model+full / B: 5-model+restricted / C: B+bag / D: A+bag
# ---------------------------------------------------------------------------

def wind_4way(con):
    print("\n=== Wind 4-way (truth = ERA5 wind_speed_10m) ===")
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()
    era_glob = (ERA5 / "**" / "*.parquet").as_posix()
    rows = []
    cols_4 = ["wind_gfs", "wind_ecmwf", "wind_icon", "wind_gem"]
    cols_5 = ["wind_gfs", "wind_ecmwf", "wind_icon", "wind_ukmo", "wind_gem"]

    for lead in LEADS:
        sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model, WindSpeed10m,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND RunTimeSource='offset_day'
              AND LeadHours={lead} AND WindSpeed10m IS NOT NULL
        ),
        pivoted AS (
            SELECT ValidTimeUtc,
                MAX(CASE WHEN Model='gfs_seamless'  THEN WindSpeed10m END) AS wind_gfs,
                MAX(CASE WHEN Model='ecmwf_ifs025'  THEN WindSpeed10m END) AS wind_ecmwf,
                MAX(CASE WHEN Model='icon_seamless' THEN WindSpeed10m END) AS wind_icon,
                MAX(CASE WHEN Model='ukmo_seamless' THEN WindSpeed10m END) AS wind_ukmo,
                MAX(CASE WHEN Model='gem_seamless'  THEN WindSpeed10m END) AS wind_gem
            FROM latest WHERE rn=1 GROUP BY ValidTimeUtc
        ),
        era AS (
            SELECT ValidTimeUtc, WindSpeed10m AS truth
            FROM read_parquet('{era_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND WindSpeed10m IS NOT NULL
        )
        SELECT p.ValidTimeUtc, p.wind_gfs, p.wind_ecmwf, p.wind_icon, p.wind_ukmo, p.wind_gem, e.truth
        FROM pivoted p JOIN era e USING (ValidTimeUtc)
        WHERE p.wind_gfs IS NOT NULL AND p.wind_ecmwf IS NOT NULL
          AND p.wind_icon IS NOT NULL AND p.wind_gem IS NOT NULL
        ORDER BY p.ValidTimeUtc
        """
        df = con.execute(sql).fetch_df()
        df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
        df = _calendar(df)
        train_full, train_post, test_df, cutoff = _split(df)
        print(f"\n--- Lead {lead}h --- train_full={len(train_full):,} train_post={len(train_post):,} test={len(test_df):,} cutoff={cutoff:%Y-%m-%d %H:%M}Z")

        for lbl, train_df, cols, params in [
            ("A_prior",          train_full, cols_4, LGB_REG_PARAMS),
            ("B_5mod_restr",     train_post, cols_5, LGB_REG_PARAMS),
            ("C_new",            train_post, cols_5, LGB_BAGGED_REG),
            ("D_4mod_full_bag",  train_full, cols_4, LGB_BAGGED_REG),
        ]:
            t0 = time.time()
            mae, rmse = _eval(train_df, test_df, cols, "wind", "truth", params)
            print(f"  {lbl}:  MAE={mae:.4f}  RMSE={rmse:.4f}  fit={time.time()-t0:.1f}s")
            rows.append(dict(target="wind", lead=lead, variant=lbl, mae=mae, rmse=rmse,
                             train_n=len(train_df), test_n=len(test_df)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Humidity — A: 5-mod+full(24h)/4-mod+full(48,72h) / B: +UKMO + restr / C: B+bag / D: A+bag
# ---------------------------------------------------------------------------

def _humid_or_cloud_4way(con, target: str, raw_var: str, prefix: str):
    print(f"\n=== {target} 4-way (truth = ERA5 {raw_var}) ===")
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()
    era_glob = (ERA5 / "**" / "*.parquet").as_posix()
    rows = []
    for lead in LEADS:
        # MF only at 24h
        include_mf = (lead == 24)
        prior_models  = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]
        if include_mf: prior_models.insert(3, "meteofrance_seamless")
        ukmo_models = prior_models + ["ukmo_seamless"]
        prior_cols = [f"{prefix}_{m.split('_')[0]}" for m in prior_models]
        ukmo_cols  = [f"{prefix}_{m.split('_')[0]}" for m in ukmo_models]

        # Build pivot for ALL 6 models; we'll select the column subset we want per variant.
        sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model, {raw_var} AS v,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND RunTimeSource='offset_day'
              AND LeadHours={lead} AND {raw_var} IS NOT NULL
        ),
        pivoted AS (
            SELECT ValidTimeUtc,
                MAX(CASE WHEN Model='gfs_seamless'         THEN v END) AS {prefix}_gfs,
                MAX(CASE WHEN Model='ecmwf_ifs025'         THEN v END) AS {prefix}_ecmwf,
                MAX(CASE WHEN Model='icon_seamless'        THEN v END) AS {prefix}_icon,
                MAX(CASE WHEN Model='meteofrance_seamless' THEN v END) AS {prefix}_meteofrance,
                MAX(CASE WHEN Model='ukmo_seamless'        THEN v END) AS {prefix}_ukmo,
                MAX(CASE WHEN Model='gem_seamless'         THEN v END) AS {prefix}_gem
            FROM latest WHERE rn=1 GROUP BY ValidTimeUtc
        ),
        era AS (
            SELECT ValidTimeUtc, {raw_var} AS truth
            FROM read_parquet('{era_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='bonehill_rocks' AND {raw_var} IS NOT NULL
        )
        SELECT p.ValidTimeUtc,
               p.{prefix}_gfs, p.{prefix}_ecmwf, p.{prefix}_icon,
               p.{prefix}_meteofrance, p.{prefix}_ukmo, p.{prefix}_gem,
               e.truth
        FROM pivoted p JOIN era e USING (ValidTimeUtc)
        ORDER BY p.ValidTimeUtc
        """
        df = con.execute(sql).fetch_df()
        df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
        df = _calendar(df)

        # Drop rows where any of the prior-models columns are NaN (matching the
        # production WHERE clause)
        df = df.dropna(subset=prior_cols).reset_index(drop=True)

        train_full, train_post, test_df, cutoff = _split(df)
        print(f"\n--- {target} lead {lead}h --- train_full={len(train_full):,} train_post={len(train_post):,} test={len(test_df):,} cutoff={cutoff:%Y-%m-%d %H:%M}Z")

        for lbl, train_df, cols, params in [
            ("A_prior",         train_full, prior_cols, LGB_REG_PARAMS),
            ("B_ukmo_restr",    train_post, ukmo_cols,  LGB_REG_PARAMS),
            ("C_new",           train_post, ukmo_cols,  LGB_BAGGED_REG),
            ("D_prior_bag",     train_full, prior_cols, LGB_BAGGED_REG),
        ]:
            t0 = time.time()
            mae, rmse = _eval(train_df, test_df, cols, prefix, "truth", params)
            print(f"  {lbl}:  MAE={mae:.4f}  RMSE={rmse:.4f}  ({len(cols)} models)  fit={time.time()-t0:.1f}s")
            rows.append(dict(target=target, lead=lead, variant=lbl, mae=mae, rmse=rmse,
                             train_n=len(train_df), test_n=len(test_df), n_models=len(cols)))
    return pd.DataFrame(rows)


def _summary(df, label):
    print(f"\n=== {label} 4-way summary (positive % = better than A_prior) ===\n")
    pivot = df.pivot_table(index="lead", columns="variant", values="mae").round(4)
    if "A_prior" in pivot.columns:
        for v in ("B", "B_ukmo_restr", "B_5mod_restr", "C_new", "D_4mod_full_bag", "D_prior_bag"):
            if v in pivot.columns:
                pivot[f"{v}_vs_A_pct"] = ((pivot["A_prior"] - pivot[v]) / pivot["A_prior"] * 100).round(2)
    print(pivot.to_string())


def main():
    con = duckdb.connect()
    out = ROOT / "data" / "reports"
    out.mkdir(parents=True, exist_ok=True)

    wind_df = wind_4way(con)
    wind_df.to_csv(out / "element_4way_wind.csv", index=False)
    _summary(wind_df, "Wind MAE m/s")

    humid_df = _humid_or_cloud_4way(con, "humidity", "RelativeHumidity2m", "rh")
    humid_df.to_csv(out / "element_4way_humidity.csv", index=False)
    _summary(humid_df, "Humidity MAE %")

    cloud_df = _humid_or_cloud_4way(con, "cloud", "CloudCover", "cc")
    cloud_df.to_csv(out / "element_4way_cloud.csv", index=False)
    _summary(cloud_df, "Cloud MAE %")


if __name__ == "__main__":
    main()
