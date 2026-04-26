"""Multi-seed stability check for the temp 2b bake-off result.

Re-runs the temperature bake-off (5-model vs 6-model on the post-2024-09
window) with 5 different LightGBM seeds per lead. Reports mean delta + std
so we can tell if the 24h "5-model wins" was noise or signal.
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
    FORECASTS, ERA5, WINDOW_START, LEADS, TEST_FRACTION,
    LGB_REG_PARAMS, NUM_ITERATIONS, EARLY_STOPPING,
    _calendar, _spread,
)

SEEDS = [42, 7, 123, 2025, 9001]


def run_one(X_tr, y_tr, X_te, y_te, feat_cols, seed):
    params = dict(LGB_REG_PARAMS, seed=seed)
    cut = max(int(len(X_tr) * 0.85), 10)
    X_tr_main, X_val = X_tr[:cut], X_tr[cut:]
    y_tr_main, y_val = y_tr[:cut], y_tr[cut:]
    train_set = lgb.Dataset(X_tr_main, label=y_tr_main, feature_name=feat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_ITERATIONS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    p = booster.predict(X_te, num_iteration=booster.best_iteration)
    return float(np.mean(np.abs(p - y_te)))


def main():
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
        df = _calendar(df)

        for variant, model_cols in [
            ("5-model", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_gem"]),
            ("6-model", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem"]),
        ]:
            d = _spread(df, model_cols, "temp")
            feat_cols = model_cols + ["temp_mean", "temp_std", "temp_range",
                                      "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
            split = int(len(d) * (1 - TEST_FRACTION))
            tr, te = d.iloc[:split], d.iloc[split:]
            X_tr = tr[feat_cols].to_numpy(dtype="float64")
            y_tr = tr["era5_temp"].to_numpy(dtype="float64")
            X_te = te[feat_cols].to_numpy(dtype="float64")
            y_te = te["era5_temp"].to_numpy(dtype="float64")

            for seed in SEEDS:
                t0 = time.time()
                mae = run_one(X_tr, y_tr, X_te, y_te, feat_cols, seed)
                rows.append(dict(lead=lead, variant=variant, seed=seed, mae=mae,
                                 fit_s=time.time() - t0))

    df = pd.DataFrame(rows)
    print("\n=== Per-seed MAE ===")
    pivot = df.pivot_table(index=["lead", "variant"], columns="seed", values="mae")
    print(pivot.round(4).to_string())

    print("\n=== Lead summary (mean ± std across 5 seeds, signed Δ = 5-model - 6-model) ===")
    summary = df.groupby(["lead", "variant"]).mae.agg(["mean", "std", "min", "max"]).round(4)
    print(summary.to_string())

    print("\n=== Per-seed delta (5-model MAE − 6-model MAE; positive ⇒ 6-model better) ===")
    pivot2 = df.pivot_table(index=["lead", "seed"], columns="variant", values="mae")
    pivot2["delta_5_minus_6"] = pivot2["5-model"] - pivot2["6-model"]
    pivot2["pct"] = pivot2["delta_5_minus_6"] / pivot2["5-model"] * 100
    print(pivot2.round(4).to_string())

    print("\n=== Aggregate winner per lead (across seeds) ===")
    wins = pivot2.groupby("lead").apply(
        lambda g: pd.Series({
            "n_seeds_6_wins": int((g.delta_5_minus_6 > 0).sum()),
            "mean_delta_pct": float(g.pct.mean()),
            "std_delta_pct": float(g.pct.std()),
            "verdict": ("6-model" if g.pct.mean() > 0 else "5-model")
                       + (" (stable)" if abs(g.pct.mean()) > 2 * g.pct.std() else " (within noise)"),
        }), include_groups=False,
    )
    print(wins.to_string())


if __name__ == "__main__":
    main()
