# Phase 3f — rainfall_amount distributional forecast (Membury-only)

Plan for productionising the NGBoost-LogNormal two-stage that beat every
other approach on Membury hourly intensity (CRPS) in the 2026-05-22/23
exploration. Pick up directly from this doc — it captures every decision
already taken so we don't re-litigate.

## What 3f is

NGBoost-LogNormal regressor trained on wet rows from EA hourly rainfall
truth, gated at predict time by Phase 3a's hourly P(wet) marginal. The
mixed predictive distribution per hour is

```
F(x) = (1 - π) · δ_0(x)   +   π · LogNormal(μ_log, σ_log)(x)
```

where π comes from 3a and (μ_log, σ_log) come from NGBoost.

Identity is fixed: **3f IS "NGBoost-LogNormal over Phase 3a"**. The source
binding is hardcoded (see `IntensityModelSources.cs` below), not a config
flip — same discipline as `DryWindowMcSources` for 3g/3j/3n/3s. If we ever
want NGBoost-over-3c or NGBoost-over-4a, that's a new phase id, not a
reconfigured 3f.

## Decisions already taken

| Decision | Choice |
|---|---|
| Target name | `rainfall_amount` |
| Phase id | `3f` |
| Locations (initial) | `membury_devon` only — Bonehill comes later, possibly never |
| Predict cadence | every 6 hours, chained off `predict-and-render` completion |
| Stage 1 source | Phase 3a (raw probabilities, NO PAV — sensitivity test 2026-05-23 showed PAV is a no-op) |
| Stage 1 binding | Hardcoded in `IntensityModelSources.cs`, not in phases.yaml |
| Stage 2 algorithm | NGBoost-LogNormal, LogScore (CRPS-Score not implemented in ngboost 0.5 for LogNormal) |
| Feature set (initial) | Lean 15-feat (7 NWP precip + 4 spread + 4 calendar) — same that fit best in the bake-off |
| Quantile alphas for derived output | `[0.025, 0.1, 0.5, 0.9, 0.975]` (config, can change without retrain) |
| Exceedance thresholds (mm) | `[0.1, 1.0, 5.0, 10.0]` (config) |
| Sky on rollout to Bonehill | Defer — Phase B/C/D pattern was Membury-only first, prove it, then add |

## Empirical baselines to beat (from 2026-05-22/23 bake-off)

Aggregate test CRPS, mean over 3 Membury stations, leads 24/48/72h:

| Variant | 24h | 48h | 72h |
|---|---:|---:|---:|
| NGBoost-LogNormal per-station (champion) | **0.1185** | **0.1296** | **0.1412** |
| NGBoost-Gamma per-station | 0.1189 | 0.1297 | 0.1428 |
| LightGBM quantile stitching | 0.1205 | 0.1319 | 0.1430 |
| 7-NWP raw ensemble (cheap baseline) | 0.1270 | 0.1403 | 0.1494 |
| equal-weight NWP-mean (deterministic) | 0.1675 | 0.1894 | 0.2041 |

Oracle (perfect stage-1) ceiling: CRPS 0.0913 / 0.0965 / 0.1007 — i.e.
~23-29% headroom theoretically exists from a better stage 1, of which we'd
realistically capture ~3-5% via 3c/3e/4a. Not worth shipping until we have
a Membury 4a (which currently doesn't exist).

## 1. C# side (WeatherBlend)

### 1.1. `phases.yaml` — new target block

```yaml
rainfall_amount:
  phases:
    - id: "3f"
      role: champion
      impl: python
      locations: ["membury_devon"]
```

### 1.2. New registry — `Train/PrecipIntensity/IntensityModelSources.cs`

Mirror of `DryWindowMcSources.cs`. Initial content:

```csharp
namespace WeatherBlend.Train.PrecipIntensity;

public static class IntensityModelSources
{
    private static readonly IReadOnlyDictionary<string, string> _sourcePhase =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["3f"] = "3a",
        };

    public static string SourcePhaseFor(string intensityPhase) =>
        _sourcePhase.TryGetValue(intensityPhase, out var src)
            ? src
            : throw new ArgumentException(
                $"Rainfall-amount phase '{intensityPhase}' has no registered " +
                "source phase. Add it to IntensityModelSources.", nameof(intensityPhase));

    public static bool IsIntensityPhase(string phase) => _sourcePhase.ContainsKey(phase);

    public static string SourceVersionKey(string intensityPhase)
        => $"precip_{SourcePhaseFor(intensityPhase)}_version";
}
```

### 1.3. `BlendersConfig` extension

Add `rainfall_amount` to the blender YAML loaded into `BlendersConfig`:

```yaml
# config.yaml (or its sibling)
blenders:
  rainfall_amount:
    "3f":
      feature_set: "lean"          # lean | rich; lean shipped
      ngboost:
        n_estimators: 500
        learning_rate: 0.01
        early_stopping_rounds: 30
      predict_output:
        quantile_alphas:  [0.025, 0.1, 0.5, 0.9, 0.975]
        exceedance_mm:    [0.1, 1.0, 5.0, 10.0]
```

`BlendersConfig` already gives WP the cross-repo config bus (per
`reference_audit_tier1_ships_2026-05-20`), so this block flows to
WeatherProbabilistic automatically.

### 1.4. Manifest / artefact paths

Standard layout under WB's models tree:

```
data/models/rainfall_amount/{station_slug}/v{ts}_phase3f/
  stage2_lognormal_lead{24,48,72,96,120}h.pkl
  feature_schema.json
  training_metadata.json   # includes precip_3a_version
```

Promotion via existing `PromoteStationVersion`; no new manifest schema —
the existing per-station-keyed manifest layout (post Phase 3 flat-manifest
retirement on 2026-05-19, [[project_p3_flat_manifest_retired]]) already
handles it.

## 2. WeatherProbabilistic side

### 2.1. `WP/scripts/train_3f.py`

Mirror of `train_4a.py`. Per (location from phases.yaml ∩ stations from
WB config ∩ leads {24,48,72,96,120}):

1. Pull wet-only training rows via DuckDB on R2-synced forecasts +
   rainfall trees (reuse the SQL from
   `scripts/run_membury_two_stage_ngboost.py` and friends).
2. 70/15/15 chronological split per station.
3. Standardise features on the training wet rows (StandardScaler).
4. Fit `NGBRegressor(Dist=LogNormal, Score=LogScore)` with early stopping
   on the val wet subset.
5. Compute training summary: n_rows, n_wet, val CRPS, best_iter, σ
   quantile distribution.
6. Run `RetrainGuard`: rows ±30%, val_CRPS ±30% of last week's,
   label-rate 0.10, NaN% absolute 0.20.
7. Resolve the current champion 3a version via the R2 manifest
   (`ModelArtifact.ResolveStationChampionVersion`-equivalent), stamp into
   `training_metadata.Hyperparameters.precip_3a_version`.
8. Pickle NGBRegressor to `stage2_lognormal_lead{N}h.pkl`.
9. Write `feature_schema.json`, `training_metadata.json`.
10. Promote via the existing manifest helpers.

### 2.2. `retrain-python.yml` integration

New matrix step parallel to `train-4a` and `run-5a`, gated on `3f` being
active in `phases.yaml`. Per-location matrix with `max-parallel: 1`
(Phase B convention, [[project_phase_b_plan]]).

Does NOT depend on `train-4a` or `run-5a` — all three train on truth
independently. They can run concurrently within the python retrain step.

### 2.3. `WP/scripts/predict_3f.py`

Mirror of `predict_4a.py`. Per cycle:

1. For each (location, station) with an Active 3f bundle:
   1. `find_latest_bundle(...)` for 3f, read `precip_3a_version` from
      `training_metadata.json`.
   2. Read the 3a *predictions* parquet for the current cycle from R2
      (`predictions/precipitation/{station}/cycle={cycle_ts}.parquet`).
      Use the version the bundle was trained against — do NOT re-resolve
      from the live manifest, so in-flight 3a retrains don't break this
      cycle's prediction.
   3. Build the 15-feature matrix for the live cycle valid_times.
   4. Load NGBRegressor pickle, predict per (valid_time, lead) → `(mu_log, sigma_log)`.
   5. Join `π` from the 3a predictions parquet by (valid_time, lead).
   6. Compute derived columns per config: median, mean, P_x, P(>threshold).
   7. Emit
      `data/predictions/rainfall_amount/{station}/cycle={cycle_ts}.parquet`.
2. Sync `data/predictions/rainfall_amount/` to R2.

### 2.4. Predict output schema

One row per (valid_time_utc, lead_hours, station_slug):

| Column | Source |
|---|---|
| `valid_time_utc`, `lead_hours`, `station_slug` | Join keys |
| `pi` | From 3a |
| `mu_log`, `sigma_log` | Stage 2 NGBoost raw |
| `mean_mm_per_hr` | π · exp(μ + σ²/2) |
| `median_mm_per_hr` | π · exp(μ) (preferred for headline display) |
| `p2_5`, `p10`, `p50`, `p90`, `p97_5_mm_per_hr` | scipy.stats.lognorm.ppf, mixed with δ_0 |
| `p_exceed_0_1`, `p_exceed_1`, `p_exceed_5`, `p_exceed_10` | π · (1 - CDF(threshold)) |
| `precip_3a_version`, `precip_3f_version` | Provenance |

Raw (μ, σ) preserved so consumers can compute additional quantiles
without retraining.

### 2.5. `WP/scripts/verify_3f.py`

Mirror of `verify_4a.py`. Per (station, lead) cell, against EA hourly
rainfall truth:

- **CRPS** — primary skill metric (use `crps_mixed` from
  `scripts/run_membury_two_stage_ngboost.py`).
- **MAE_wet** — secondary.
- **Coverage** for the announced 80% interval ([P10, P90]) — should be
  ~0.80.
- **PIT values** per row (Probability Integral Transform); accumulate into
  a histogram — should be flat if calibrated.
- **Reliability** on each configured exceedance threshold (binned
  predicted vs observed exceedance rate).

Write to `data/verify_history/rainfall_amount/{station}/lead_{N}h.parquet`
with the same shape as other verify-history sidecars + new CRPS /
coverage / PIT-mean columns. `DriftFlag` triggers at 1.5× rolling CRPS,
same multiplier as P(wet) drift uses for Brier. Existing `verify.yml`
exits 4 on any `DriftFlag=true` — 3f rows ride that machinery
automatically.

## 3. Cloudflare orchestration

Predict 3a is NOT cron'd directly. Per
`cloudflare/scheduler-worker/wrangler.toml` + `src/index.ts`, the actual
chain is:

```
HH:45   collect       (Cloudflare cron)
   └─ on success ─► predict-4a + predict-5a            (Hop C)

HH+1:05 s3-collect    (Cloudflare cron)
   └─ on completion ─► predict-and-render              (Hop D)
                       (contains the .NET predict-3a as part of predict-all)
```

So 3f, depending on 3a's emitted predictions, chains off
`predict-and-render`:

```
predict-and-render ─(success)─► predict-3f             (new Hop E)
```

Add to `handleWorkflowRun` in `scheduler-worker/src/index.ts`. Failure of
`predict-and-render` → 3f skips this cycle, previous cycle's 3f stays
current; predict_3f.py logs a structured error if 3a's parquet for the
current cycle is missing.

This also matches the established discipline:
[[reference_scheduling_cloudflare_only]] — all crons + chains in
Cloudflare, never GitHub-native cron.

## 4. Site components

### 4.1. Predictions page (per-location)

1. **Headline strip** — for each lead bucket (24/48/72h):
   "Median 0.4 mm/h • 80% interval 0.0-1.8 • 22% chance of rain". Use
   median (not mean) for the point — skew-robust on LogNormal.
2. **Hourly intensity ribbon chart** — x = valid_time, y = mm/h. Layers:
   P50 line, P10-P90 ribbon, P2.5-P97.5 lighter shadow.
3. **Exceedance dashboard** — small grid: P(>1mm), P(>5mm), P(>10mm)
   across next 72h. Cells coloured by probability magnitude.
4. **(Deferred)** distribution-explorer modal — click an hour → full
   LogNormal density curve.

### 4.2. Skill page (per-station, per-lead)

1. **CRPS rolling time-series** — 90-day rolling CRPS, equivalent to
   existing Brier-history chart.
2. **PIT histogram** — flat = well-calibrated; bumps at 0/1 = under/over-spread.
3. **Coverage chart** — rolling fraction of obs in announced 80% interval.
4. **Exceedance reliability diagrams** — one per configured threshold.
5. **(Deferred)** sharpness-vs-skill scatter.

**PIT and coverage** are new reusable widgets — worth building them
properly since 4a's predictive distribution (and any future distributional
phase) will reuse them.

## 5. Order of work (estimate: ~2 weeks solo)

1. **C# side** — `IntensityModelSources.cs`, `rainfall_amount` target in
   phases.yaml + blender config, any missing `ModelArtifact` helpers.
   **~1 day.**
2. **`train_3f.py` + R2 sync + retrain-python.yml matrix entry**.
   **~2 days**, mostly mirroring `train_4a.py`.
3. **`predict_3f.py` + Cloudflare worker chain hop**.
   **~2 days**, mostly mirroring `predict_4a.py`.
4. **`verify_3f.py` + verify-history schema extension**.
   **~1-2 days.**
5. **Site components** (predictions card + skill page widgets).
   **~3-5 days** — the bigger calendar chunk because PIT + coverage are
   new chart types.

## 6. Validation gates before going live

1. After step 2: train 3f locally on Membury, confirm CRPS matches the
   bake-off numbers above (0.1185 / 0.1296 / 0.1412) within ±2%.
2. After step 3: run predict_3f against a recent live cycle, sanity-check
   the predictive bands look reasonable on a known wet event and a known
   dry day.
3. After step 4: let verify run for 2 weeks of live cycles before showing
   skill metrics on the public site. Confirm PIT histogram is flat-ish
   and 80% interval coverage lands in [0.75, 0.85].
4. Only then enable the site components for public traffic.

## 7. Reference artefacts from the 2026-05-22/23 exploration

All under `WP/reports/membury_intensity_lgbm/`:

- `_precip_cache.parquet` — cached pulled data for fast iteration
- `summary.csv`, `aggregate.csv` — initial LightGBM intensity sweep
- `member_weighted_*.csv` — constrained blend bake-off (negative result)
- `two_stage_summary.csv`, `two_stage_aggregate.csv` — two-stage with
  CRPS, the inflection point
- `two_stage_emos_*.csv`, `two_stage_ngboost_*.csv` — distributional
  variant bake-off (NGBoost-LogNormal won)
- `two_stage_inla_*.csv` — pooled INLA experiment (negative result —
  per-station NGBoost beat it)
- `stage1_sensitivity_*.csv` — confirmed PAV is no-op and oracle gap
  is 23-29% (mostly irreducible)

Scripts (in `WP/scripts/`): `run_intensity_lgbm_membury.py`,
`run_membury_member_weighted.py`, `run_membury_two_stage.py`,
`run_membury_two_stage_emos.py`, `run_membury_two_stage_ngboost.py`,
`run_membury_two_stage_inla.py`, `run_membury_stage1_sensitivity.py`.

The NGBoost-LogNormal fit in `run_membury_two_stage_ngboost.py` is the
direct reference implementation for the stage-2 fit in `train_3f.py`.
