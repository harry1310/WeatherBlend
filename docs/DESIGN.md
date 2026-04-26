# Design notes

Running commentary on why things are the way they are. Useful when you come back
to this in three months and wonder what past-you was thinking.

## Why Parquet + DuckDB and not a database?

For a single-location PoC, yearly data volume is a few hundred MB. A database adds
ops burden (backups, migrations, a process to keep running) with no query speed
benefit at this size. Parquet partitioned by model/date, queried via DuckDB, is
faster to set up, trivial to back up (copy a folder), and trivial to move to cloud
storage later if the PoC expands to many locations.

If this ever grows to hundreds of locations with gridded data, revisit. Options
at that point: DuckDB over S3-backed Parquet (still no database, just bigger
storage), or ClickHouse. Still not Postgres.

## Why Open-Meteo first and not direct-to-source?

Direct GRIB parsing from ECMWF / NOAA is a rabbit hole of eccodes, cfgrib, and
wgrib2 that would eat a week before any ML. Open-Meteo gives us six models through
one consistent JSON API including a historical archive - exactly what's needed
to get moving. The tradeoffs:

- We get their interpolation to our lat/lon, not the raw grid. For a single point
  this doesn't matter much.
- We don't see raw ensemble members, only the deterministic run per model.
  This matters a lot for precip (see below).
- We have to trust they don't silently rewrite history in the archive.

Phase 2+ plan: go direct for ECMWF ENS (51 members, the real prize) via their
open data at data.ecmwf.int, and for HRRR/GFS via AWS Open Data S3 buckets.

## Why LightGBM for the blender?

- Handles missing values natively (some models will fail to return some variables).
- Handles heterogeneous features without scaling.
- Monotonic constraints available (e.g. force "forecast temp up -> predicted temp up").
- Quantile regression mode for prediction intervals and CRPS approximation.
- Microsoft.ML has a first-class LightGBM trainer so no Python needed.

Alternatives considered:

- **Linear/ridge regression**: often surprisingly close on temperature. Worth
  including as a baseline. Less good on precip.
- **Deep nets**: overkill for single-location tabular data with ~10k-50k rows.
- **ngboost / EMOS / BMA**: better for probabilistic precip but thin tooling in
  .NET. Plan to shell out to Python for this specific step in phase 5.

## Champion/challenger for feature-set experiments (phase 2c)

Feature-engineering ablations are hard to judge off a single training-time test
MAE — the real question is "does the new feature set generalise better in
production?", which can only be answered by running both models in parallel on
live forecasts and comparing against ERA5 over several weeks.

The manifest encodes this directly. `MANIFEST.Active` is a list of version dir
names (not a single pointer like `Current`), and `predict --model-version
current` iterates that list, writing one parquet partition per version under
`data/predictions/temperature/model_version=<v>/date=<d>/`. Verify reads the
whole tree, stratifies by `(version, lead)`, and — when two active versions have
different `Phase` tags (e.g. `2b` lean vs `2c` rich) — renders a direct MAE delta
section in the weekly report. The winner is whichever version has the smaller
rolling blend MAE at each lead; promotion is then a one-line manifest edit
(`SetActive` with the chosen version).

Two consequences to be aware of:

- Storage is linear in `|Active|` — each version materialises its own predictions
  parquet. At ~6 rows/day per version this is fine at PoC scale.
- `Current` is frozen while a challenger is running. Dropping a challenger from
  `Active` doesn't delete its predictions — they stay on disk as part of the
  historical record, just no new ones are written.

## Why per-lead-time models and not one big model?

Forecast skill characteristics change dramatically with lead time:

- At 1h: persistence dominates, models all agree, spread is tiny.
- At 24-72h: model disagreement is meaningful signal; blend adds most value here.
- At 144h+: skill collapses, everyone's bad, blender mostly just averages.

A single model trying to cover all of these either underfits the nuance at each
horizon or overfits the easy ones. Separate per-bucket models each see a
more homogeneous problem. Buckets: [1-3h], [6-12h], [24-36h], [48-72h],
[96-120h], [144-168h].

## Precipitation architecture (phase 3)

Don't train a single regression on precip (mm/h). You'll get near-zero predictions
most of the time that score well on MAE and are useless.

Two-stage:

1. **Occurrence classifier (phase 3a, shipped)**: P(precip ≥ 0.1mm) in hour H, per
   lead time. Target: binary from EA Hydrology gauge (15-min tips summed to hourly;
   hours with fewer than 4 readings are dropped to avoid boundary flips). **One
   blender per truth station** because Dartmoor stations differ enough in micro-
   climate that a single model would underfit both; slugs are `ea_bellever_dartmoor`,
   `ea_princetown` (the `ea_` prefix reserves space for a future Met Office Princetown).
   Metrics: Brier, BSS vs climatology, frequency bias @0.5, reliability diagram,
   persistence + best-single + mean-of-models baselines. Climatology baseline is
   P(wet) per (month, hour-of-day) from training rows, persisted as
   `climatology.json` alongside each version so predict/verify never need the
   training tree again.

2. **Intensity regression (originally phase 3b)**: E[precip | precip > threshold],
   conditional on the classifier firing. Train only on hours with observed precip.
   Replaced by the Phase 3b dry-window classifier after the user pivoted to "is
   there time to walk the dog dry?" — one binary classifier per (station, window)
   beats calibrating a conditional intensity regressor for that decision.

3. **Rich-feature occurrence blender (phase 3c, shipped)**: same LightGBM, same
   hyperparameters, same chronological split as 3a — the isolated variable is
   feature richness. 27 lean features + 28 extras: 18 per-model humidity
   (dewpoint, RH, T-Td depression), 6 per-model surface pressure, 4 EA-observation
   trailing-rainfall persistence features anchored at `run_time = valid - leadHours`
   (prev-24h mm, prev-72h mm, wet-hours-last-24h, trailing-dry-hours). 55 features
   total. Saved as a challenger alongside 3a via per-station `StationEntry.Active`
   so both artefacts produce predictions every cycle and the rolling verify path
   can score them side by side. Forecast-time trailing-precip persistence (H-1..-3
   of the same run) and pressure tendency were explicitly dropped: Phase 1's
   training parquet stores only leads {24, 48, 72} per `offset_day` run, so the
   intermediate-lead cells those features need don't exist in training data even
   though live cycles have them. Switching training to live cycles would break
   the "same split as 3a" guarantee and is out of scope for 3c. The tier ablation
   in `precip-ablate` shows the rich features are modestly helpful at best —
   each tier contributes at most ~0.002 Brier points, and the EA-persistence
   tier actively hurts at 2 of 3 stations.

4. **Dry-window improvements (phase 3d, shipped)**: two parallel challengers to
   the Phase 3b dry-window blender, sharing 3b's hyperparameters and split so
   each change is isolated. Both register alongside 3b via the per-composite
   `StationEntry.Active` list — 3b stays the production champion and predict +
   verify score every Active version per cycle.
     * **3d-shape** extends the 53-feature row with 7 within-day shape features
       derived from the ensemble-mean hourly precip vector
       (`first_wet_hour`, `last_wet_hour`, `longest_forecast_dry_run_hours`,
       `longest_forecast_wet_run_hours`, `n_rain_events`,
       `morning_precip_sum`, `afternoon_precip_sum`). Day-level aggregates
       used by 3b lose all timing information; the shape features re-introduce
       it in a small, interpretable tier and let the model condition on
       _whether_ a forecast wet day is "wet morning, dry afternoon" vs
       "constant drizzle". Trained via `--target dry-window --feature-set rich`.
     * **3d-calibrated** is a strict post-hoc reweighting of the 3b model.
       For each lead the 3b probabilities on the held-out validation partition
       are fit with pool-adjacent-violators isotonic regression (the same
       `IsotonicCalibrator` shared with 3a_isotonic) and the resulting knot
       table is saved as `calibration.json`. Predict loads the calibration
       when `metadata.Phase == "3d-calibrated"` and applies the per-lead
       mapping to the raw 3b probability before writing the row. Same model
       file, same features, same feature hash — only the mapping changes.
   Lessons from the Phase 3a → 3a_isotonic experiment apply by construction to
   3d-calibrated: PAV alone moves Brier by ≈0 when the upstream model is
   already well-calibrated. It is risk insurance against miscalibration drift,
   not a skill increase. The training-time comparison is generated by
   `dry-window-ablate`, which reads each artefact's `training_metadata.json`
   plus the 3d-shape `feature_importance.json` and emits a side-by-side
   3b/3d-shape/3d-cal Brier+BSS+freq-bias table per (station, window, lead),
   with shape-feature gain rank highlighted so it is easy to see whether the
   new tier is doing useful work.

Original phase-3 combine recipe
(`expected_precip = P(precip) * E[precip | precip > 0]`) is deferred indefinitely —
the dry-window framing is what the user actually consumes.

For probabilistic output (the real product), train quantile regressions at
e.g. q10/q25/q50/q75/q90 of the precip distribution. CRPS is the metric that
matters; pinball loss is what you minimise per-quantile.

## Ground truth hierarchy for precip

Best -> worst for training a precip model:

1. **Radar composite** (Met Office Nimrod, MRMS US, OPERA EU) - 1km gridded,
   captures spatial structure. Best for validating convective events.
2. **Quality-controlled gauge network** (Met Office MIDAS) - point truth,
   reliable intensity, gaps in space.
3. **Personal weather station network** (Netatmo, Weather Underground) - dense
   but noisy, needs QC.
4. **METAR present-weather codes** - occurrence only, no intensity, coarse
   timing. What we have in phase 1.

Decision: ship phase 1 with METAR, tag phase 3 as "source quantitative precip
truth" before touching the precip model. Don't train precip on phase 1 truth.

Phase 3a shipped with **EA Hydrology** (free, OGL v3, 15-min resolution, 1998+
at Bellever and Princetown) rather than Met Office DataHub — quality is close
enough for Dartmoor-scale verification and avoids DataHub's rate limits and
paid tier. Nimrod radar is still the right next step for convective events.

## Verification: the bit that matters most

The easiest mistake in this whole project is declaring victory on bad metrics.
Pre-commit to:

- **Time-based splits, never random.** Random shuffling leaks future into past.
- **Walk-forward validation** for final numbers (train to month N, test N+1,
  refit to N+1, test N+2, ...).
- **Proper baselines every time**: persistence, climatology, mean-of-models,
  best single model. The blend has to beat all four or it's not earning its keep.
- **Report per lead time, not aggregated.** Aggregated numbers hide where the
  blend helps and where it doesn't.
- **For precip: Brier + reliability diagrams alongside MAE.** An underconfident
  well-calibrated forecast is more useful than a sharp miscalibrated one.

## Element blenders for non-temperature variables (wind / humidity / radiation / cloud)

The four lean per-variable blenders established in 2026-04-25 follow the same
architectural pattern as the temperature 2b lean blender: per-lead LightGBM
regression, ERA5 truth at the Bonehill grid cell, chronological 70/15/15 split,
identical hyperparameters. The dispatcher (`ElementTrainCommand`,
`ElementPredictCommand`, `ElementVerifyCommand`) routes by `--target` to a
per-element pipeline; each pipeline owns its own SQL pivot, row class, and
feature shape, because the four elements share no useful schema (wind has
direction sin/cos, humidity has dewpoint, radiation has direct/diffuse, cloud
has only total available in our data).

**ERA5 truth across all elements (consistency, not convenience).** Using ERA5
for wind, humidity, radiation, and cloud — same as temperature — keeps the
training-truth contract identical across blenders. EGTE METAR is ~25 km away
and ~360 m below Bonehill; for variables where elevation matters (wind: tor
exposure enhances by 10–30%; cloud: orographic fog often shrouds the tor when
the surrounding terrain is clear) it would teach the blender to predict EGTE
rather than Bonehill. ERA5 has its own caveat — ~28×18 km grid cells average
across regional terrain — but it's the closest gridded estimate of conditions
at the tor available, and applying the same caveat consistently across all
five blenders is preferable to per-variable truth-source choices that would
create unexamined coupling. METAR is reserved for verify-side secondary checks.

**Rich variants deferred pending evidence of need.** This phase ships champion
lean blenders only. Cloud cover (loses at 72h) and shortwave radiation (wash
across all leads) underperformed the brief's expectations — both are candidates
for rich variants if/when investigation shows the lean feature set is the
problem. The brief's principle stands: "investigate before adding rich features
blindly". Likely investigation paths: solar-geometry features for radiation,
layered-cloud features for cloud cover (requires a collector extension first
since Open-Meteo Previous Runs ships only total cloud).

**Spatial-resolution caveat is inherited.** Same ~28×18 km ERA5 grid resolution
applies to all element blenders as it does to temperature; consequences vary by
variable (small for temperature, large for wind/cloud) but the architectural
choice is the same.

## UKMO handling: per-blender decisions (2026-04-26, revised)

### What we thought the problem was

Open-Meteo's Previous Runs API ships UKMO with ~26% of valid times missing across
every lead and every variable. We initially assumed this was random per-row gaps.
Three patterns considered:

| Pattern | Training rows kept | Training noise | Predict robustness |
|---|---|---|---|
| **1. Drop UKMO entirely** (UKMO column always NaN in training) | 100% | None — feature is constant | Always works |
| **2. Require UKMO non-null** (drop training rows where UKMO missing) | 74% | None — UKMO always present in training | Breaks if live UKMO degrades |
| **3. NaN-tolerant** (keep all rows, UKMO can be NaN) | 100% | Random per-row NaN noise | Always works but miscalibrated |

A first-pass apples-to-apples bake-off favoured Pattern 1 for the lean blenders
(temp 2b, precip 3a, all 4 Elements) — UKMO was dropped from those trainers.

### What the problem actually was

Field-level missingness audit (`scripts/ukmo_field_missingness.py`, 2026-04-26):
the 26% gap is **NOT** a per-row random pattern. It is a contiguous **7-month
time block** at the start of the archive (2024-01 → 2024-08, every UKMO field
NULL); from 2024-09 onwards UKMO is consistently populated. With our
chronological 80/20 train/test split, that block dominates training and the
6-model fit learns a dual regime ("UKMO=NaN for 7 months, then valid forever
after") that doesn't generalise to the held-out test set (where UKMO is ~100%
present). The 5-model wins were train-time poisoning, not evidence that UKMO
genuinely hurts.

### Restricted-window bake-off (the fix)

Re-running the same apples-to-apples bake-off with training restricted to
post-2024-09 (the clean window where UKMO is consistently present) flipped the
result almost everywhere. 6-model wins or ties at every (target, station, lead)
cell tested — clearest at 24h precip (+2.9–4.7% Brier across all 3 stations)
and 48h/72h temp (+1.1–1.2% MAE).

### Current per-blender handling (final, 2026-04-26 four-way bake-off)

The 6-model migration was rolled back for most targets after a 4-way bake-off
(prior 5-model+full vs 6-model+restricted vs +bagging vs prior+bagging) on
**fixed identical test rows** revealed the original migration bake-off was
flawed: it held the data window constant for both variants, hiding the real
trade-off. The 6-model variant's UKMO signal generally does NOT outweigh the
~30% training-data loss from restricting to the post-2024-09 window. Per
target the answers diverged though — wind and cloud genuinely benefit from
UKMO; temp/precip/humidity don't.

| Blender | Final variant | Models | Window | Bagging |
|---|---|---|---|---|
| Temperature 2b lean | **5-model + full + bag** | GFS, ECMWF, ICON, MF, GEM | full backfill | ✓ |
| Temperature 2c rich | Pattern 2 (existing) | unchanged | unchanged | unchanged |
| Precip 3a lean (×3 stations) | **5-model + full + bag** | as above | full backfill | ✓ |
| Precip 3c rich | Pattern 3 (existing) | unchanged | unchanged | unchanged |
| Dry-window 3b | Pattern 3 (existing) | unchanged | unchanged | unchanged |
| Element **wind** | **5-model + restricted, NO bag** | GFS, ECMWF, ICON, **UKMO**, GEM (no MF; never had MF wind) | post-2024-09 | ✗ (deliberate) |
| Element humidity | **5-model + full + bag** | GFS, ECMWF, ICON, MF, GEM (24h); no MF at 48/72h | full backfill | ✓ |
| Element **cloud** | **6-model + restricted + bag** | GFS, ECMWF, ICON, MF, **UKMO**, GEM (24h); 5 at 48/72h (no MF) | post-2024-09 | ✓ |
| Element radiation | 5-model + full + bag | GFS, ECMWF, ICON, MF, GEM | full backfill | ✓ |

**Why heterogeneous?** Met Office's UM has historically excelled at UK wind
(regional model resolves Bonehill's tor topography) and cloud (good
parameterisation, stratus detection) — for those targets UKMO is genuinely
informative. For temperature the global ensemble averages out individual model
biases and UKMO is just one of six. For per-station rainfall the truth has
its own gauge-specific bias structure UKMO doesn't help with. For humidity
similarly. Picking per blender, not per pattern.

**Wind opts out of bagging** specifically — the 4-way showed bagging slightly
HURTS wind at every lead (−0.16/−0.55/−0.42% vs no-bag), unique among
targets. `WindBlender.cs` overrides `Hyperparameters` to set
`SubsampleFraction=1.0, SubsampleFrequency=0, FeatureFraction=1.0` while
every other Element inherits the bagged defaults from
`TemperatureTrainer.Hyperparameters`.

The restricted-window cutoff is `TrainingWindow.UkmoCleanWindowStart`
(2024-09-01), used by wind and cloud only.

**Bagging defaults** (everywhere except wind): `SubsampleFraction=0.8,
SubsampleFrequency=1, FeatureFraction=0.8` in all three trainers. ML.NET's
defaults left LightGBM fully deterministic; bagging adds 0.5–2% MAE
improvement and genuine ensemble noise across temp/precip/humidity/cloud.

**Known unfixed**: cloud 24h blend still loses to best-single ECMWF (~−10.8%
MAE), even with the C variant. Cloud is structurally hard with the lean
13-feature shape — needs feature engineering (layered cloud, dew-point
depression, radiation cross-checks) rather than model selection. Tracked as
future work.

## UTCI: derived outdoor-comfort target (2026-04-26)

The Universal Thermal Climate Index (Bröde et al. 2012) is the first WeatherBlend
output that *isn't* itself a trained blender — it's a deterministic transform of
five upstream blender outputs (`temperature` lean 2b + `humidity` + `wind` +
`shortwave-radiation` + `cloud-cover`). UTCI exists because none of the five
inputs alone answers the actual user question: "is it comfortable to be outside
on Bonehill right now?"

**Why we blended radiation and cloud was UTCI all along.** Mean radiant
temperature (Tmrt) is what turns UTCI from "Ta + wind chill + humidity" into a
real outdoor-comfort number. Tmrt at Ta=25 °C with full clear-sky midday sun
(~800 W/m² shortwave) sits around 50–55 °C; under heavy cloud at the same Ta
it's ~28 °C. That delta is what pushes UTCI from "no thermal stress" into
"strong heat stress" bands. Without blended SW + cloud the calculator would
have to fall back on Tmrt = Ta and lose the entire heat-stress signal.

### Tmrt approach (radiation balance, simplified)

Hemisphere-weighted radiation balance for an upright unobstructed person
(Thorsson et al. 2007 / Lindberg et al. 2008 SOLWEIG-lite form):

```
Sstr = K_human · Kdown + 0.5 · εp · (Ldown + Lup)
Tmrt = (Sstr / (εp · σ))^0.25 - 273.15
```

| Term | Source | Notes |
|---|---|---|
| `Ta` | temperature 2b lean | air temperature, °C |
| `Kdown` | shortwave-radiation blender | total horizontal SW down, W/m² |
| `Ldown` | derived from cloud + Ta | Brutsaert clear-sky εclear = 1.24·(e/T)^(1/7), boosted by cloud (Crawford-Duchon: εeff = (1-cc)·εclear + cc) |
| `Lup` | derived from Ta | εg · σ · Ta⁴, εg=0.95 (grass) |
| `e` | derived from Ta + RH | Wexler/Hardy saturation expansion × RH/100 |
| `K_human` | constant 0.30 | Lumped αk · (0.25·fdir + 0.5·fdiff + 0.5·αs) for upright person on horizontal Kdown — calibrated to reproduce published clear-sky Tmrt response (Tmrt − Ta ≈ +25–30 K at midday clear sky) |
| `εp = 0.97`, `αk = 0.7`, `αs = 0.15` | constants | human emissivity / SW absorptivity / grass albedo |

Direct/diffuse split and solar-zenith-dependent projected-area factor are
deliberately **not** computed: our blenders give us total horizontal Kdown and
no direction info, so the lumped `K_human` is the honest one-coefficient
substitute. The simplification is documented next to the constant so a future
refinement (split SW + solar elevation) is a localised change.

### UTCI polynomial

Inputs `(Ta, Pa hPa, va10 m/s, Tmrt °C)` fed to the official Bröde 2012
6th-order regression. Wind input is at 10 m (the regression was fit on it,
not on body-height wind); a 1.1 m reduction (`× log(1.1/0.01)/log(10/0.01)
≈ 0.6809`) is provided as a separate column for display.

### Architecture choices

- **Derived target, not a blender.** No model_version registry, no manifest,
  no champion/challenger. Output uses fixed `model_version=v1` so a future
  Tmrt formulation upgrade can land as `v2` without rewriting the predict path.
- **Per-row provenance of every input version.** `TempModelVersion`,
  `HumidityModelVersion`, etc. let us trace "which radiation champion produced
  the SW input that drove this Tmrt".
- **Anchor-aligned join.** UTCI predict reads `data/predictions/{slug}/
  model_version={champion}/date={anchor}/predictions.parquet` for each input,
  joins by `(ValidTimeUtc, LeadHours)`, drops rows where any one input is
  missing — UTCI requires all five.
- **No verify wired in v1.** UTCI has no observational truth at Bonehill
  (would need a globe thermometer). Verification belongs upstream — each input
  blender already has its own `verify` against ERA5.

## Things explicitly out of scope for now

- Multi-location generalisation
- Gridded forecasts
- User-facing UI or API
- Real-time nowcasting from radar
- Commercial licensing (stay on non-commercial tiers)
