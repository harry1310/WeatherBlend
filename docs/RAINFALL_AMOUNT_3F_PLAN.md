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
flip — same discipline that the deleted-2026-05-25 `DryWindowMcSources`
established for the retired 3g/3j/3n/3s family. If we ever want
NGBoost-over-3c or NGBoost-over-4a, that's a new phase id, not a
reconfigured 3f.

## Decisions already taken

| Decision | Choice |
|---|---|
| Target name | `rainfall_amount` |
| Phase id | `3f` |
| Locations (initial) | `membury_devon` only — Bonehill comes later, possibly never |
| Predict cadence | every 6 hours, chained off `predict` completion (Hop E) |
| Stage 1 source | Phase 3a (raw probabilities, NO PAV — sensitivity test 2026-05-23 showed PAV is a no-op) |
| Stage 1 binding | Hardcoded in `IntensityModelSources.cs`, not in phases.yaml |
| Stage 1 upgrade path | Defer Membury 4a (would ship the ~3-5% oracle headroom but doubles Phase 3 scope; revisit after 3f has ~4 weeks of clean verify) |
| Stage 2 algorithm | NGBoost-LogNormal, LogScore (CRPS-Score not implemented in ngboost 0.5 for LogNormal) |
| Feature set (initial) | Lean 15-feat (7 NWP precip + 4 spread + 4 calendar) — same that fit best in the bake-off |
| Quantile alphas for derived output | `[0.025, 0.1, 0.5, 0.9, 0.975]` (config, can change without retrain) |
| Exceedance thresholds (mm) | `[0.1, 1.0, 5.0, 10.0]` (config) |
| Python + ngboost pinning | REQUIRED — pin both in WP requirements.txt before train_3f.py ships. Pickle stability across library minors is the same kind of trap that bit 4a on 2026-05-09 ([[reference_dbarts_serialize_caveat]]) |
| Site v1 scope | SHIP ALL WIDGETS at once — PIT histogram + 80% coverage + exceedance reliability + headline + intensity ribbon + CRPS rolling. Reusable widgets that future distributional phases will reuse |
| Validation gate | RELAXED — skill metrics visible from day 1 with a sample-size caveat in the UI (departs from the original 2-week-hide rule because UI labelling makes "n=5 cycles" honest enough) |
| Cloudflare chain | Split predict-and-render.yml into predict + render-site so 3f can sandwich between (same-cycle freshness on the site) |
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
~23-29% headroom theoretically exists from a better stage 1. Realistic
Membury upgrade paths post-cleanup-Phase-2:
  * Membury **4a** — BART per-cell, would ship the bulk of the headroom
    (~3-5%). Not yet built (4a is Bonehill-only today). DEFERRED per the
    2026-05-26 Phase 3 decisions memory — revisit after 3f has ~4 weeks
    of clean verify.
  * Membury **3o** — REJECTED at the 2026-05-25 stage-1 bake-off
    (Membury terrain pool too homogeneous, +0.36% CRPS vs 3a).
  * Membury **3c** stays as a reference (within 0.4% CRPS of 3a, so
    swapping it in or out is noise-level).
3e was the legacy MLP and is retired (cleanup Phase 1, 2026-05-25).

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
`cloudflare/scheduler-worker/wrangler.toml` + `src/index.ts`, the
pre-Phase-3 chain is:

```
HH:45   collect       (Cloudflare cron)
   └─ on success ─► predict-4a + predict-5a            (Hop C)

HH+1:05 s3-collect    (Cloudflare cron)
   └─ on completion ─► predict-and-render              (Hop D)
                       (fused .NET predict-all + 4b synthesise +
                        render-site + deploy)
```

For 3f to land on the **same cycle's render** (rather than the next
cycle, which is the default 4a/5a behaviour), the fused
`predict-and-render` is split into separate `predict` + `render-site`
workflows so 3f can be sandwiched between them. Decision logged
2026-05-26 — same-cycle freshness is worth the extra ~3 min of chain
latency because 3f IS the headline on the rainfall_amount predictions
card (median mm/h + 80% interval + exceedance probabilities), unlike
4a which is primarily a skill-metric phase. Resulting chain:

```
HH+1:05 s3-collect    (Cloudflare cron)
   └─ on completion ─► predict   (WB targets, push to R2)   (Hop D)
              │
              └─ on completion ─► predict-3f (WP, fresh 3o) (Hop E)
                            │
                            └─ on completion ─► render-site (Hop F)
                                              (read R2, render, deploy)
```

Add a new Hop E + reshape Hop D + add Hop F to `handleWorkflowRun` in
`scheduler-worker/src/index.ts`. Note the existing `predict.yml` and
`render-site.yml` workflows still exist in `.github/workflows/` as
manual-dispatch-only (their bodies were merged into
predict-and-render.yml; see that file's header comment). The split
brings them back as production workflows.

Failure modes:
- `predict` fails → 3f skips, predict-and-render-style failure (cycle
  emits no fresh WB predictions; previous cycle's render stays live).
- `predict-3f` fails → render still chains (Hop E success is OPTIONAL
  for Hop F); site renders with stale 3f rows from R2.
- `render-site` fails → published site stays at the previous deploy
  (Cloudflare Pages keeps the last successful deploy live).

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

0. **Predict + render workflow split** (Phase 3 prereq). Revive
   `predict.yml` + `render-site.yml` as production workflows;
   collapse predict-and-render.yml's responsibilities into them;
   update Cloudflare scheduler-worker chain. **~half a day.**
1. **C# side** — `IntensityModelSources.cs`, `rainfall_amount` target in
   phases.yaml + blender config, any missing `ModelArtifact` helpers.
   **~1 day.**
2. **`train_3f.py` + R2 sync + retrain-python.yml matrix entry**.
   Pin Python + ngboost versions in `requirements.txt` first.
   **~2 days**, mostly mirroring `train_4a.py`.
3. **`predict_3f.py` + Cloudflare worker Hop E**.
   **~2 days**, mostly mirroring `predict_4a.py`. Chain hop now lands
   between `predict` (Hop D) and `render-site` (Hop F).
4. **`verify_3f.py` + verify-history schema extension**.
   **~1-2 days.**
5. **Site components** (predictions card + skill page widgets,
   sample-size caveat pattern). All widgets in v1 (per the
   2026-05-26 decision): headline strip + intensity ribbon + CRPS
   rolling + PIT histogram + 80% coverage + exceedance reliability.
   **~3-5 days** — the bigger calendar chunk because PIT + coverage
   are new chart types.

## 6. Validation gates before going live

1. After step 2: train 3f locally on Membury, confirm CRPS matches the
   bake-off numbers above (0.1185 / 0.1296 / 0.1412) within ±2%.
2. After step 3: run predict_3f against a recent live cycle, sanity-check
   the predictive bands look reasonable on a known wet event and a known
   dry day.
3. After step 4: verify runs from day 1. Skill widgets (CRPS / PIT /
   coverage / reliability) are visible immediately with a sample-size
   caveat in the UI ("n=N cycles — stabilises after ~14 days") rather
   than hidden behind a 2-week gate. Once N ≥ 14 days, the caveat
   self-clears and the widgets read as proper skill metrics. Watch the
   first 2 weeks of PIT + coverage in case the bake-off CRPS doesn't
   translate to calibrated live output — PIT bumps at 0/1 are the
   clearest signal of under/over-spread.

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
