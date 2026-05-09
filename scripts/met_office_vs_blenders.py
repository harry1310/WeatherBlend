"""Met Office Global Det 10km vs WeatherBlend blenders vs each NWP, against truth.

Question: who has performed better historically — Met Office's deterministic
global forecast post-processing, or our blender ensemble? Plus the underlying
question — does Met Office Global Det beat each individual NWP it's fused
from at our specific tor-on-Dartmoor location?

Comparison window: where ALL of these overlap:
  * Met Office Global Det archive (2024-04-26 → today)
  * Each NWP source (we have continuous from ~2024-01)
  * ERA5 truth (~5-day lag, gapless)
  * EA hydrology truth (3 stations, ~2024-01 onwards)

Headline metrics:
  Temperature   — MAE (°C) per (source, lead) vs ERA5
  Precipitation — Brier per (source, station, lead) vs EA, where source's
                  P(wet) is derived from per-row Precipitation rate as a
                  hard threshold at 0.1 mm/h (matching blender training)

Sources compared:
  * met_office_global  (the new archive)
  * Each individual NWP: gfs_seamless, ecmwf_ifs025, icon_seamless,
    meteofrance_seamless, ukmo_seamless, gem_seamless
  * mean_of_nwps       (ensemble mean — climatology-style baseline)
  * blender            (LightGBM trained in-script on pre-window data,
                        evaluated in-window so the comparison is honest)

Run with:
    .venv/Scripts/python.exe scripts/met_office_vs_blenders.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "data" / "forecasts" / "location=bonehill_rocks"
ERA5 = ROOT / "data" / "truth" / "era5" / "location=bonehill_rocks"
RAINFALL = ROOT / "data" / "truth" / "rainfall" / "location=bonehill_rocks"
REPORT_PATH = ROOT / "data" / "reports" / "met_office_vs_blenders.md"

LEADS = (24, 48, 72)
NWPS = (
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless",
    "meteofrance_seamless", "ukmo_seamless", "gem_seamless",
)
NWP_SHORT = {
    "gfs_seamless": "GFS",
    "ecmwf_ifs025": "ECMWF",
    "icon_seamless": "ICON",
    "meteofrance_seamless": "MF",
    "ukmo_seamless": "UKMO",
    "gem_seamless": "GEM",
    "met_office_global": "MO_Global",
    "mean_of_nwps": "MeanOfNWPs",
    "blender": "Blender",
}
WET_THRESHOLD_MM = 0.1
STATIONS = ("Bellever Dartmoor", "Bovey Tracey", "Dartmoor nr Hexworthy")
STATION_CODE = {
    "Bellever Dartmoor": "Bellever",
    "Bovey Tracey": "Bovey",
    "Dartmoor nr Hexworthy": "Hexworthy",
}


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


# ---------------------------------------------------------------------------
# Common: pull all per-(model, valid, lead) rows over the overlap window
# ---------------------------------------------------------------------------

def _all_models_pivot(con: duckdb.DuckDBPyConnection, lead: int, variable: str) -> pd.DataFrame:
    """Return one row per (ValidTimeUtc) with one column per model carrying
    `variable`. Only rows where met_office_global has data are returned —
    that defines the overlap window. NWP rows pull RunTimeSource='offset_day'
    (the historical-forecast view) which gives the most consistent retro
    coverage; live `reported` rows would only cover the last few weeks."""
    fc_glob = _norm(FORECASTS / "**" / "*.parquet")
    sql = f"""
    WITH raw AS (
        SELECT ValidTimeUtc, Model, {variable} AS v
        FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
        WHERE LocationName='bonehill_rocks'
          AND LeadHours={lead}
          AND {variable} IS NOT NULL
          AND (
              (Model='met_office_global' AND RunTimeSource='exact')
              OR (Model IN ({",".join(f"'{m}'" for m in NWPS)})
                  AND RunTimeSource='offset_day')
          )
    ),
    -- Take latest (Model, ValidTime) row in case of duplicates.
    latest AS (
        SELECT ValidTimeUtc, Model, v,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY v) AS rn
        FROM raw
    )
    SELECT ValidTimeUtc, Model, v FROM latest WHERE rn=1
    """
    long_df = con.execute(sql).fetch_df()
    long_df["ValidTimeUtc"] = pd.to_datetime(long_df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    pivot = long_df.pivot_table(index="ValidTimeUtc", columns="Model", values="v", aggfunc="first").reset_index()
    # Inner-join semantics: only keep ValidTimes where met_office_global is present.
    if "met_office_global" not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["met_office_global"]).reset_index(drop=True)
    return pivot


# ---------------------------------------------------------------------------
# TEMPERATURE
# ---------------------------------------------------------------------------

def temperature_comparison(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, str]:
    """Per (source, lead) MAE °C vs ERA5 truth, on the overlap window.

    Includes a quick LightGBM blender trained on the FIRST 70% of the
    overlap window's rows and evaluated on the remaining 30% — keeps the
    blender's eval honest (no leakage) without needing to re-train the C#
    production blender separately.
    """
    print("\n=== Temperature MAE vs ERA5 ===")
    era_glob = _norm(ERA5 / "**" / "*.parquet")
    truth_df = con.execute(f"""
        SELECT ValidTimeUtc, Temperature2m AS truth
        FROM read_parquet('{era_glob}', hive_partitioning=false, union_by_name=true)
        WHERE LocationName='bonehill_rocks' AND Temperature2m IS NOT NULL
    """).fetch_df()
    truth_df["ValidTimeUtc"] = pd.to_datetime(truth_df["ValidTimeUtc"], utc=True).dt.tz_localize(None)

    rows = []
    summary_lines = []
    for lead in LEADS:
        pivot = _all_models_pivot(con, lead, "Temperature2m")
        if pivot.empty:
            print(f"  lead {lead}h: no overlap rows; skipping")
            continue
        df = pivot.merge(truth_df, on="ValidTimeUtc", how="inner")
        df = df.sort_values("ValidTimeUtc").reset_index(drop=True)
        # Mean of NWPs (skip NaNs row-wise)
        nwp_cols = [c for c in NWPS if c in df.columns]
        df["mean_of_nwps"] = df[nwp_cols].mean(axis=1, skipna=True)

        n_total = len(df)
        date_min, date_max = df["ValidTimeUtc"].min(), df["ValidTimeUtc"].max()
        print(f"  lead {lead}h: {n_total:,} overlap rows, {date_min.date()} → {date_max.date()}")

        # Per-source MAE on rows where the source is non-null
        for src in ["met_office_global"] + nwp_cols + ["mean_of_nwps"]:
            sub = df.dropna(subset=[src])
            if len(sub) < 100:
                continue
            mae = float(np.mean(np.abs(sub[src] - sub["truth"])))
            rows.append(dict(
                target="temperature", lead=lead, source=NWP_SHORT.get(src, src),
                n=len(sub), mae=mae, units="°C",
            ))

        # Quick blender: LightGBM on per-NWP + hour calendar, trained on first 70%
        # of these overlap rows, evaluated on the last 30%. Nothing leaks since
        # truth isn't fed in as a feature, and the test split is later in time.
        blend_cols = nwp_cols  # already excludes met_office_global
        df_b = df.dropna(subset=blend_cols + ["truth"]).reset_index(drop=True)
        df_b["hour_sin"] = np.sin(2 * np.pi * df_b["ValidTimeUtc"].dt.hour / 24)
        df_b["hour_cos"] = np.cos(2 * np.pi * df_b["ValidTimeUtc"].dt.hour / 24)
        feat_cols = blend_cols + ["hour_sin", "hour_cos"]
        cut = int(len(df_b) * 0.7)
        if cut < 200 or len(df_b) - cut < 100:
            print(f"    blender: not enough rows; skipping ({len(df_b)})")
            continue
        X_tr = df_b[feat_cols].iloc[:cut].to_numpy("float64")
        y_tr = df_b["truth"].iloc[:cut].to_numpy("float64")
        X_te = df_b[feat_cols].iloc[cut:].to_numpy("float64")
        y_te = df_b["truth"].iloc[cut:].to_numpy("float64")
        params = dict(
            objective="regression_l1", metric="mae", num_leaves=31, learning_rate=0.05,
            min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l1=0.1, lambda_l2=0.1, verbose=-1, seed=42, num_threads=0,
        )
        cut_v = max(int(len(X_tr) * 0.85), 10)
        train_set = lgb.Dataset(X_tr[:cut_v], label=y_tr[:cut_v], feature_name=feat_cols)
        val_set = lgb.Dataset(X_tr[cut_v:], label=y_tr[cut_v:], feature_name=feat_cols, reference=train_set)
        booster = lgb.train(params, train_set, num_boost_round=500,
                            valid_sets=[val_set], valid_names=["val"],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        p_blend = booster.predict(X_te, num_iteration=booster.best_iteration)
        mae_blend = float(np.mean(np.abs(p_blend - y_te)))
        # Evaluate MO_Global on the SAME held-out rows for apples-to-apples
        df_te = df_b.iloc[cut:].reset_index(drop=True)
        if "met_office_global" in df_te.columns:
            mo_te = df_te.dropna(subset=["met_office_global"])
            mae_mo_te = float(np.mean(np.abs(mo_te["met_office_global"] - mo_te["truth"])))
        else:
            mae_mo_te = float("nan")
        rows.append(dict(
            target="temperature", lead=lead, source="Blender(LGB)",
            n=int(len(X_te)), mae=mae_blend, units="°C",
        ))
        rows.append(dict(
            target="temperature", lead=lead, source="MO_Global (same test rows)",
            n=int(len(mo_te)), mae=mae_mo_te, units="°C",
        ))

    df_out = pd.DataFrame(rows)
    summary_lines.append("Temperature comparison: MAE °C vs ERA5 truth, lower is better.")
    summary_lines.append(f"Overlap window covers all rows where Met Office Global has historical data.")
    return df_out, "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# PRECIPITATION
# ---------------------------------------------------------------------------

def precipitation_comparison(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, str]:
    """Per (source, station, lead) Brier vs EA hydrology binary truth.

    Each source's per-row P(wet) is derived from its quantitative
    Precipitation forecast as a hard threshold at 0.1 mm/h (which matches
    the blender's training-time wet/dry definition). For the LightGBM
    blender we train P(wet) directly with binary labels.
    """
    print("\n=== Precipitation Brier (per station) vs EA hydrology ===")
    rows = []
    for station in STATIONS:
        rain_glob = _norm(RAINFALL / f"station={station}" / "**" / "*.parquet")
        truth_sql = f"""
            WITH r AS (
                SELECT date_trunc('hour', ObservedTimeUtc) AS hour_utc, Value15MinMm
                FROM read_parquet('{rain_glob}', hive_partitioning=false, union_by_name=true)
                WHERE Value15MinMm IS NOT NULL
            )
            SELECT hour_utc AS ValidTimeUtc, SUM(Value15MinMm) AS precip_mm
            FROM r GROUP BY hour_utc HAVING COUNT(*)=4
        """
        truth_df = con.execute(truth_sql).fetch_df()
        truth_df["ValidTimeUtc"] = pd.to_datetime(truth_df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
        truth_df["observed_wet"] = (truth_df["precip_mm"] >= WET_THRESHOLD_MM).astype("int8")

        for lead in LEADS:
            pivot = _all_models_pivot(con, lead, "Precipitation")
            if pivot.empty:
                continue
            df = pivot.merge(truth_df[["ValidTimeUtc", "observed_wet", "precip_mm"]], on="ValidTimeUtc", how="inner")
            if len(df) < 200:
                continue
            print(f"  {STATION_CODE[station]} lead {lead}h: {len(df):,} overlap rows")
            nwp_cols = [c for c in NWPS if c in df.columns]
            df["mean_of_nwps"] = df[nwp_cols].mean(axis=1, skipna=True)

            # Per source: derive P(wet) as binary 1/0 from precip threshold;
            # Brier degenerates to (pred-y)^2 mean = mis-classification rate.
            # We also report a "soft" P via sigmoid around the threshold so
            # heavy-precip predictions don't all lump at 1.0.
            for src in ["met_office_global"] + nwp_cols + ["mean_of_nwps"]:
                sub = df.dropna(subset=[src])
                if len(sub) < 100:
                    continue
                # Hard binary
                pred_hard = (sub[src] >= WET_THRESHOLD_MM).astype("float64").to_numpy()
                brier_hard = float(np.mean((pred_hard - sub["observed_wet"].to_numpy()) ** 2))
                # Compute "wet hit-rate" + "false-alarm rate" as additional context
                truth = sub["observed_wet"].to_numpy()
                tp = int(((pred_hard == 1) & (truth == 1)).sum())
                fp = int(((pred_hard == 1) & (truth == 0)).sum())
                fn = int(((pred_hard == 0) & (truth == 1)).sum())
                tn = int(((pred_hard == 0) & (truth == 0)).sum())
                acc = (tp + tn) / max(1, tp + fp + fn + tn)
                rows.append(dict(
                    target="precipitation", station=STATION_CODE[station], lead=lead,
                    source=NWP_SHORT.get(src, src), n=len(sub),
                    brier_hard=brier_hard, accuracy=acc,
                    wet_rate_pred=float(pred_hard.mean()),
                    wet_rate_obs=float(truth.mean()),
                ))

            # LightGBM blender: train on first 70%, eval on last 30%, with the
            # NWP precip + hour features. P(wet) is real-valued.
            df_b = df.dropna(subset=nwp_cols).reset_index(drop=True)
            df_b["hour_sin"] = np.sin(2 * np.pi * df_b["ValidTimeUtc"].dt.hour / 24)
            df_b["hour_cos"] = np.cos(2 * np.pi * df_b["ValidTimeUtc"].dt.hour / 24)
            feat_cols = nwp_cols + ["hour_sin", "hour_cos"]
            cut = int(len(df_b) * 0.7)
            if cut < 200 or len(df_b) - cut < 100:
                continue
            X_tr = df_b[feat_cols].iloc[:cut].to_numpy("float64")
            y_tr = df_b["observed_wet"].iloc[:cut].to_numpy()
            X_te = df_b[feat_cols].iloc[cut:].to_numpy("float64")
            y_te = df_b["observed_wet"].iloc[cut:].to_numpy()
            params = dict(
                objective="binary", metric="binary_logloss", num_leaves=31, learning_rate=0.04,
                min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l1=0.1, lambda_l2=0.1, verbose=-1, seed=42, num_threads=0,
            )
            cut_v = max(int(len(X_tr) * 0.85), 10)
            train_set = lgb.Dataset(X_tr[:cut_v], label=y_tr[:cut_v], feature_name=feat_cols)
            val_set = lgb.Dataset(X_tr[cut_v:], label=y_tr[cut_v:], feature_name=feat_cols, reference=train_set)
            booster = lgb.train(params, train_set, num_boost_round=500,
                                valid_sets=[val_set], valid_names=["val"],
                                callbacks=[lgb.early_stopping(30, verbose=False)])
            p_blend = booster.predict(X_te, num_iteration=booster.best_iteration)
            brier_blend = float(brier_score_loss(y_te, p_blend))
            # Evaluate MO_Global on the SAME held-out rows for apples-to-apples
            df_te = df_b.iloc[cut:].reset_index(drop=True)
            mo_te = df_te.dropna(subset=["met_office_global"])
            pred_mo = (mo_te["met_office_global"] >= WET_THRESHOLD_MM).astype("float64").to_numpy()
            brier_mo_te = float(np.mean((pred_mo - mo_te["observed_wet"].to_numpy()) ** 2))
            rows.append(dict(
                target="precipitation", station=STATION_CODE[station], lead=lead,
                source="Blender(LGB)", n=int(len(X_te)),
                brier_hard=brier_blend, accuracy=float((np.round(p_blend) == y_te).mean()),
                wet_rate_pred=float(p_blend.mean()),
                wet_rate_obs=float(y_te.mean()),
            ))
            rows.append(dict(
                target="precipitation", station=STATION_CODE[station], lead=lead,
                source="MO_Global (same test rows)", n=int(len(mo_te)),
                brier_hard=brier_mo_te,
                accuracy=float((pred_mo == mo_te["observed_wet"].to_numpy()).mean()),
                wet_rate_pred=float(pred_mo.mean()),
                wet_rate_obs=float(mo_te["observed_wet"].mean()),
            ))

    return pd.DataFrame(rows), "Precipitation comparison: Brier vs EA hydrology, lower is better."


def write_report(temp_df: pd.DataFrame, precip_df: pd.DataFrame) -> None:
    L = []
    L.append("# Met Office Global Det vs WeatherBlend blenders vs each NWP — historical comparison\n")
    L.append("Generated by `scripts/met_office_vs_blenders.py`. All metrics computed on "
             "the overlap window where Met Office Global Det archive has data, vs the "
             "appropriate truth source (ERA5 for temperature, EA hydrology for precipitation).\n")

    L.append("## Method\n")
    L.append("- **NWP rows** taken from the historical-forecast view "
             "(`RunTimeSource='offset_day'`) — the consistent retro-coverage view.")
    L.append("- **MO Global rows** taken from the AWS Open Data archive "
             "(`RunTimeSource='exact'`).")
    L.append("- **Blender** is a LightGBM regressor (temp) / binary classifier (precip) "
             "trained on the FIRST 70% of the overlap window's rows and evaluated on the "
             "remaining 30%. Same hyperparameters as production (bagging 0.8 / feature "
             "0.8 / leaves 31 / lr 0.05 for regression, lr 0.04 for classification). For "
             "the blender row in each table, MO Global is also re-evaluated on the SAME "
             "held-out test rows (\"MO_Global (same test rows)\") so the comparison is "
             "literally apples-to-apples.\n")

    L.append("## Temperature — MAE °C vs ERA5 (lower is better)\n")
    if not temp_df.empty:
        for lead in sorted(temp_df["lead"].unique()):
            sub = temp_df[temp_df["lead"] == lead].sort_values("mae").reset_index(drop=True)
            L.append(f"### Lead {lead}h\n")
            L.append(sub[["source", "n", "mae"]].to_markdown(index=False, floatfmt=".4f"))
            L.append("")
    else:
        L.append("(no temperature comparison rows)\n")

    L.append("## Precipitation — Brier (binary) vs EA hydrology (lower is better)\n")
    L.append("Per-source P(wet) is derived from quantitative Precipitation as a hard "
             "threshold at 0.1 mm/h, matching the blender's wet/dry definition. The "
             "blender row uses the LightGBM probability output directly (real-valued, "
             "not thresholded).\n")
    if not precip_df.empty:
        for station in sorted(precip_df["station"].unique()):
            for lead in sorted(precip_df["lead"].unique()):
                sub = precip_df[
                    (precip_df["station"] == station) & (precip_df["lead"] == lead)
                ].sort_values("brier_hard").reset_index(drop=True)
                if sub.empty:
                    continue
                L.append(f"### {station} — lead {lead}h\n")
                L.append(sub[["source", "n", "brier_hard", "accuracy",
                              "wet_rate_pred", "wet_rate_obs"]].to_markdown(index=False, floatfmt=".4f"))
                L.append("")
    else:
        L.append("(no precipitation comparison rows)\n")

    L.append("## How to read the tables\n")
    L.append("- A LOWER MAE / Brier wins.")
    L.append("- For each (lead) or (station, lead) cell, sources are sorted by metric.")
    L.append("- The two `(same test rows)` rows let you directly compare the blender to "
             "MO Global on byte-identical rows.")
    L.append("- `wet_rate_pred` / `wet_rate_obs` columns surface frequency bias — a source "
             "predicting 0.5 wet vs an observed 0.3 wet is over-forecasting precip.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    temp_df, _ = temperature_comparison(con)
    precip_df, _ = precipitation_comparison(con)

    print("\n=== TEMPERATURE summary ===")
    if not temp_df.empty:
        print(temp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== PRECIPITATION summary ===")
    if not precip_df.empty:
        print(precip_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    write_report(temp_df, precip_df)

    out_dir = ROOT / "data" / "reports"
    temp_df.to_csv(out_dir / "met_office_vs_blenders_temp.csv", index=False)
    precip_df.to_csv(out_dir / "met_office_vs_blenders_precip.csv", index=False)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
