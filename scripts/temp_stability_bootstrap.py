"""Proper noise check on the temp 24h '5-model wins' result.

Two complementary tests, since LightGBM with feature_fraction=bagging_fraction=1
is fully deterministic and the seed has no effect:

  (A) Bootstrap of test residuals (1000 resamples) — answers "given this fitted
      pair of models, is the 1.28% delta real or within sampling noise of the
      test set?"

  (B) Bagged-LightGBM seed variance (10 seeds with bagging_fraction=0.8) —
      answers "is the *trained model itself* stable, or could a slightly
      different fit flip the result?"
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

N_BOOTSTRAP = 2000
BAGGED_SEEDS = list(range(10))
BAGGED_PARAMS = dict(LGB_REG_PARAMS, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.85)


def fit_predict(X_tr, y_tr, X_te, feat_cols, params):
    cut = max(int(len(X_tr) * 0.85), 10)
    train_set = lgb.Dataset(X_tr[:cut], label=y_tr[:cut], feature_name=feat_cols)
    val_set = lgb.Dataset(X_tr[cut:], label=y_tr[cut:], feature_name=feat_cols, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_ITERATIONS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return booster.predict(X_te, num_iteration=booster.best_iteration)


def load_lead(con, lead, fc_glob, era_glob):
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
    return _calendar(df)


def main():
    con = duckdb.connect()
    fc_glob = (FORECASTS / "**" / "*.parquet").as_posix()
    era_glob = (ERA5 / "**" / "*.parquet").as_posix()
    rng = np.random.default_rng(42)

    print("=" * 72)
    print("(A) Bootstrap of test residuals — fully deterministic params")
    print("=" * 72)

    for lead in LEADS:
        df = load_lead(con, lead, fc_glob, era_glob)
        split = int(len(df) * (1 - TEST_FRACTION))
        tr_df, te_df = df.iloc[:split], df.iloc[split:]

        preds = {}
        for variant, model_cols in [
            ("5", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_gem"]),
            ("6", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem"]),
        ]:
            d_tr = _spread(tr_df, model_cols, "temp")
            d_te = _spread(te_df, model_cols, "temp")
            feat_cols = model_cols + ["temp_mean", "temp_std", "temp_range",
                                      "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
            X_tr = d_tr[feat_cols].to_numpy(dtype="float64")
            y_tr = d_tr["era5_temp"].to_numpy(dtype="float64")
            X_te = d_te[feat_cols].to_numpy(dtype="float64")
            preds[variant] = fit_predict(X_tr, y_tr, X_te, feat_cols, LGB_REG_PARAMS)
        y_te = te_df["era5_temp"].to_numpy(dtype="float64")

        e5 = np.abs(preds["5"] - y_te)
        e6 = np.abs(preds["6"] - y_te)
        n = len(y_te)

        # Bootstrap MAE difference
        deltas = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            deltas[i] = e5[idx].mean() - e6[idx].mean()  # +ve = 6-model better

        mae5, mae6 = e5.mean(), e6.mean()
        ci = np.percentile(deltas, [2.5, 50, 97.5])
        pct_pos = (deltas > 0).mean() * 100
        print(f"\nLead {lead}h  (n_test={n:,})")
        print(f"  point estimate: 5={mae5:.4f}  6={mae6:.4f}  Δ(5-6)={mae5-mae6:+.4f} ({(mae5-mae6)/mae5*100:+.2f}%)")
        print(f"  bootstrap Δ:   median={ci[1]:+.4f}  95% CI=[{ci[0]:+.4f}, {ci[2]:+.4f}]")
        print(f"  P(6-model better in bootstrap)={pct_pos:.1f}%")
        sig = "REAL" if (ci[0] > 0 or ci[2] < 0) else "NOT statistically significant"
        print(f"  verdict: {sig}")

    print("\n" + "=" * 72)
    print(f"(B) Bagged LightGBM (bagging=0.8, feat=0.85), {len(BAGGED_SEEDS)} seeds")
    print("=" * 72)

    for lead in LEADS:
        df = load_lead(con, lead, fc_glob, era_glob)
        split = int(len(df) * (1 - TEST_FRACTION))
        tr_df, te_df = df.iloc[:split], df.iloc[split:]
        y_te = te_df["era5_temp"].to_numpy(dtype="float64")

        results = {"5": [], "6": []}
        for variant, model_cols in [
            ("5", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_gem"]),
            ("6", ["temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem"]),
        ]:
            d_tr = _spread(tr_df, model_cols, "temp")
            d_te = _spread(te_df, model_cols, "temp")
            feat_cols = model_cols + ["temp_mean", "temp_std", "temp_range",
                                      "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
            X_tr = d_tr[feat_cols].to_numpy(dtype="float64")
            y_tr = d_tr["era5_temp"].to_numpy(dtype="float64")
            X_te = d_te[feat_cols].to_numpy(dtype="float64")
            for seed in BAGGED_SEEDS:
                params = dict(BAGGED_PARAMS, seed=seed)
                p = fit_predict(X_tr, y_tr, X_te, feat_cols, params)
                results[variant].append(float(np.mean(np.abs(p - y_te))))

        m5, s5 = float(np.mean(results["5"])), float(np.std(results["5"]))
        m6, s6 = float(np.mean(results["6"])), float(np.std(results["6"]))
        # Per-seed delta — paired comparison, same seed for fair pairing
        per_seed_delta = [r5 - r6 for r5, r6 in zip(results["5"], results["6"])]
        n_six_wins = sum(1 for d in per_seed_delta if d > 0)
        print(f"\nLead {lead}h")
        print(f"  5-model MAE: mean={m5:.4f}  std={s5:.4f}  range=[{min(results['5']):.4f}, {max(results['5']):.4f}]")
        print(f"  6-model MAE: mean={m6:.4f}  std={s6:.4f}  range=[{min(results['6']):.4f}, {max(results['6']):.4f}]")
        print(f"  per-seed Δ(5-6) (paired): mean={np.mean(per_seed_delta):+.4f}  std={np.std(per_seed_delta):.4f}  6-model wins {n_six_wins}/{len(BAGGED_SEEDS)} seeds")


if __name__ == "__main__":
    main()
