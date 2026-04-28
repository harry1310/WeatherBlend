# WeatherBlend

A proof-of-concept for blending multiple NWP (numerical weather prediction) models
against local observations to produce a better single-location forecast than any
constituent model.

**PoC location:** Bonehill Rocks, Dartmoor (50.5831°N, 3.7931°W, 393m).

## Approach

1. **Collect** hourly forecasts from ~6 free NWP models via Open-Meteo every run cycle.
2. **Collect** METAR observations from Exeter (EGTE) with Yeovilton (EGDY) fallback as ground truth.
3. **Backfill** ~12 months of historical forecasts via Open-Meteo's archive API
   so there's enough training data without waiting.
4. **Train** a per-lead-time blender (LightGBM) that learns which models to trust when.
5. **Evaluate** against baselines (persistence, climatology, mean-of-models, best single model)
   using MAE/RMSE for temperature and Brier / reliability / CRPS for precipitation.

Temperature first, precipitation second - precip is a fundamentally harder problem
(zero-inflated, non-Gaussian, timing errors dominate) and warrants its own model.

## Prerequisites

- .NET 10 SDK (`dotnet --version` should report 10.x)
- Windows, macOS, or Linux
- ~1 GB free disk for a year of backfill data

## Quick start

```powershell
# From the repo root:
dotnet restore
dotnet build

# Run one collection cycle (pulls latest forecasts + METAR)
dotnet run --project src/WeatherBlend -- collect

# See what's on disk
dotnet run --project src/WeatherBlend -- status

# Backfill the last year of historical forecasts (takes ~30 min, polite rate limiting)
dotnet run --project src/WeatherBlend -- backfill --start 2024-11-01 --end 2025-10-31

# Phase 2 commands (stubs for now)
dotnet run --project src/WeatherBlend -- train --target temperature
dotnet run --project src/WeatherBlend -- evaluate

# Phase 3a (lean): per-station precipitation occurrence blender (P(wet>=0.1mm/hour))
dotnet run --project src/WeatherBlend -- train --target precipitation --station "Bellever Dartmoor"
dotnet run --project src/WeatherBlend -- predict --target precipitation --truth-station all
dotnet run --project src/WeatherBlend -- verify --target precipitation --truth-station all

# Phase 3c (rich): same blender with 55 features (adds per-model humidity, surface
# pressure, and EA trailing-rainfall persistence). Saves a challenger alongside 3a.
dotnet run --project src/WeatherBlend -- train --target precipitation --feature-set rich --lead all
dotnet run --project src/WeatherBlend -- precip-ablate      # 3a-vs-3c table + 24h tier ablation

# Phase 3b: per-station, per-window dry-window blender — P(at least one
# contiguous N-hour dry block exists in target UTC day), N in {3, 4, 6}
dotnet run --project src/WeatherBlend -- dry-window-diagnostic
dotnet run --project src/WeatherBlend -- train --target dry-window --station "Bellever Dartmoor" --window all
dotnet run --project src/WeatherBlend -- dry-window-report
dotnet run --project src/WeatherBlend -- predict --target dry-window --truth-station all --window all
dotnet run --project src/WeatherBlend -- verify --target dry-window --truth-station all --window all

# Phase 3d: dry-window improvements alongside 3b. 3d-shape adds 7 within-day
# rain-structure features (60-feature variant); 3d-calibrated wraps each saved
# 3b model with isotonic (PAV) calibration. Both register as challengers via
# the per-(station, window) Active list.
dotnet run --project src/WeatherBlend -- train --target dry-window --feature-set rich --window all --lead all
dotnet run --project src/WeatherBlend -- dry-window-calibrate --truth-station all
dotnet run --project src/WeatherBlend -- dry-window-ablate          # 3b vs 3d-shape vs 3d-cal table + shape gain importance

# Element blenders (lean): per-variable blenders for wind speed, relative humidity,
# shortwave radiation, and total cloud cover. Same architecture as the temperature
# 2b lean blender — LightGBM regression per lead {24,48,72}h, ERA5 truth at the
# Bonehill grid cell, champion-only this phase. One dispatcher routes each target.
dotnet run --project src/WeatherBlend -- train   --target wind                --feature-set lean --lead all
dotnet run --project src/WeatherBlend -- train   --target humidity            --feature-set lean --lead all
dotnet run --project src/WeatherBlend -- train   --target shortwave-radiation --feature-set lean --lead all
dotnet run --project src/WeatherBlend -- train   --target cloud-cover         --feature-set lean --lead all
dotnet run --project src/WeatherBlend -- predict --target wind                # likewise for humidity / shortwave-radiation / cloud-cover
dotnet run --project src/WeatherBlend -- verify  --target wind
```

### Precipitation target specifics

- Truth comes from EA Hydrology rainfall gauges (15-min tips aggregated to hourly,
  dropping hours with fewer than 4 readings) rather than ERA5 or METAR.
- One blender per station because each site has its own orographic skill profile;
  artefacts live under `data/models/precipitation/{station_slug}/v{ts}/`. The slug
  is always prefixed `ea_` so a future Met Office Princetown station can coexist.
- Predict output: `data/predictions/precipitation/{station}/model_version={v}/date={yyyy-MM-dd}/predictions.parquet`
  with P(wet), per-model inputs, ensemble aggregates, run-times, and a SHA-256
  feature-vector hash for provenance.
- Verify stratifies by (station, version, lead) and reports Brier, climatology Brier,
  BSS, mean-of-models, best-single, persistence, frequency bias @0.5, and a 10-bin
  reliability table. Drift flags fire when rolling Brier > 1.5× training-test Brier.

### Dry-window target specifics (phase 3b)

- Daily binary classifier per (station, window-length): label is "is there at
  least one contiguous run of `WindowHours` UTC hours with all four 15-min
  rainfall readings ≤ 0.1 mm/h within the target UTC day". Window lengths in the
  POC: **3, 4, 6 hours** at leads **24, 48, 72 hours** = 18 models.
- Truth gating: hourly bins with `COUNT(*)=4` only. Cross-midnight windows are
  not currently modelled (UTC-day boundary; deferred). Daylight filtering is
  deferred to the application layer.
- Artefacts live under `data/models/dry_window/{station_slug}/window_{N}h/v{ts}/`
  with composite manifest keys `{slug}/window_{N}h`.
- Per-row feature vector is **53 floats** in a frozen schema (per-model day
  aggregates: precip-sum / max-hour / wet-hour-count / longest-dry-run /
  has-dry-window self-prediction / max prob; ensemble agreement; day-level
  meteorology means + extremes; doy sin/cos). The Phase 3d-shape variant
  extends this to **60 floats** by appending 7 within-day shape features
  derived from the ensemble-mean hourly precip vector
  (`first_wet_hour`, `last_wet_hour`, `longest_forecast_dry_run_hours`,
  `longest_forecast_wet_run_hours`, `n_rain_events`, `morning_precip_sum`,
  `afternoon_precip_sum`).
- Predict output: `data/predictions/dry_window/{station}/window_{N}h/model_version={v}/date={yyyy-MM-dd}/predictions.parquet`
  with `ProbHasDryWindow`, climatology baseline, agreement, per-model
  self-predictions + day totals (so verify can recompute mean-of-models without
  re-reading the forecast tree), and a SHA-256 feature hash.
- Verify pairs predictions against the truth labels rebuilt from EA hourly
  rainfall via `DryWindowLabelBuilder`, stratifies by (station, window, version,
  lead), and reports blend Brier vs climatology vs mean-of-models, BSS, freq
  bias, drift flag (rolling Brier > 1.5× training Brier). When more than one
  phase is present at a slice, the report also emits a "Phase comparison"
  headline section showing 3b / 3d-shape / 3d-cal Brier + BSS side by side.
- **Phase 3d champion/challenger:** `Phase 3d-shape` and `Phase 3d-calibrated`
  versions are appended to each composite manifest entry's `Active` list rather
  than replacing 3b. Predict iterates over every Active version per cycle
  (writing one parquet per version), and 3d-calibrated lookups load the
  per-lead `calibration.json` saved at `dry-window-calibrate` time and apply
  the PAV mapping to the raw 3b probability before writing the row.
  `dry-window-ablate` produces a side-by-side training-time comparison report
  reading each phase's `training_metadata.json`.
- **Train/predict distribution mismatch (known caveat):** training pulls
  `RunTimeSource='offset_day'` rows so each lead has an exact target-anchor
  pairing; live inference uses the most-recent live-cycle row per (valid-time,
  model). Feature shape is identical and the model applies cleanly, but the
  inference-time distribution is not exactly the training distribution.

### Element blenders (wind / humidity / shortwave-radiation / cloud-cover)

- **Truth source: ERA5 at the Bonehill grid cell**, consistent with the
  temperature blender. EGTE METAR is intended as a verify-side secondary
  sanity check; the wiring is deferred (each element needs bespoke handling
  — RH from T+Td via Magnus, wind direct, no METAR signal for radiation,
  parser extension for cloud cover).
- **Per-element feature shapes:** wind 22 features (5 model speeds × 1 + 5×2
  sin/cos directions + 3 spread + 4 calendar; MétéoFrance excluded — Open-Meteo
  Previous Runs ships no MF wind), humidity 19 (6×RH + 6×dewpoint + 3 spread +
  4 calendar), radiation 25 (6×SW + 6×direct + 6×diffuse + 3 spread + 4 calendar;
  UKMO carries NaN at lead ≥48h, LightGBM handles natively), cloud 13 (6×total
  + 3 spread + 4 calendar — layered cloud is 100% null in Open-Meteo Previous
  Runs and was dropped from the lean spec).
- **Artefacts:** `data/models/{element}/v{ts}/` with one `lead_{N}h.zip` per
  lead, `feature_schema.json`, `feature_importance.json`,
  `training_metadata.json`, and a single-active `MANIFEST.json`. Phase tags
  are per-target (`lean-wind`, `lean-humidity`, etc.) so the predict dispatcher
  is unambiguous; rich variants would land as `rich-{element}` if added later.
- **Predict output:** `data/predictions/{element}/model_version={v}/date={yyyy-MM-dd}/predictions.parquet`
  via the shared `ElementPredictionRow` schema (per-model values for all six
  slots, runtimes, ensemble aggregates, blend value, SHA-256 feature hash).
- **Verify:** ERA5 primary MAE per (version, lead) with stratification by month
  and truth-value quintile, drift flag (rolling MAE > 1.5× training-test MAE).
- **Live-forecast coverage gaps (production caveats):** shortwave radiation is
  100% null in the live forecast tree — predict produces no radiation rows
  until the live collector is extended to pull SW/direct/diffuse. MétéoFrance
  cuts off around 36h on the live endpoint, so humidity and cloud predict
  yield only the 24h lead in production.
- **Lean training-time results (Bonehill, 2026-04-25):** wind +20–24% vs best
  single (ECMWF), humidity +7–10%, cloud +10% / +2% / **−2%** at 24/48/72h,
  radiation **−1% / +4% / +1%**. See
  `data/reports/lean_blenders_phase_2026-04-25.md` for the full headline table
  and per-element analysis.

### Feels-like (derived outdoor-comfort target)

Two "feels-like" indices on every row, derived from the five upstream
blenders — no model training of its own. Joins the latest predict parquet
for temperature (lean 2b) + humidity + wind + shortwave-radiation +
cloud-cover at the same anchor date, computes mean radiant temperature via
a hemisphere-weighted radiation balance (Thorsson 2007 / Lindberg 2008
simplified form), then runs both:
  * **UTCI** (Bröde et al. 2012 6th-order polynomial) — fed
    `(Ta, Pa, va10, Tmrt)`; labels each row with the published thermal-
    stress band (`NoStress`, `ModerateHeat`, `StrongCold`, ...). Tmrt is
    the lever that turns UTCI from "Ta + wind chill + humidity" into a
    real outdoor-comfort number that responds to clear-sky midday sun
    (Tmrt ≫ Ta) — which is why shortwave-radiation and cloud-cover were
    blended in the first place.
  * **Steadman 1994** apparent temperature — the simpler shade-form
    `Ta + 0.33·e − 0.70·ws − 4` behind BoM's published AT and the BBC's
    public "feels like" chip. Less aggressive on wind chill than UTCI;
    used as the publicly recognisable companion number on the home card.

Output: `data/predictions/feels_like/model_version=v1/date={yyyy-MM-dd}/predictions.parquet`
with per-row inputs, derived radiation terms, both indices, the UTCI band,
and the source model_version of each input for full provenance.

```powershell
# Pre-requisite: the five inputs already predicted for the same anchor.
dotnet run --project src/WeatherBlend -- predict --target temperature
dotnet run --project src/WeatherBlend -- predict --target humidity
dotnet run --project src/WeatherBlend -- predict --target wind
dotnet run --project src/WeatherBlend -- predict --target shortwave-radiation
dotnet run --project src/WeatherBlend -- predict --target cloud-cover
dotnet run --project src/WeatherBlend -- predict --target feels-like
```

## Scheduling on Windows

Register the `collect` command to run every 3 hours via Task Scheduler:

```powershell
$action = New-ScheduledTaskAction `
    -Execute "dotnet" `
    -Argument "run --project C:\path\to\WeatherBlend\src\WeatherBlend -- collect" `
    -WorkingDirectory "C:\path\to\WeatherBlend"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName "WeatherBlend-Collect" -Action $action -Trigger $trigger
```

Alternatively, `dotnet publish -c Release -r win-x64 --self-contained false` and schedule
the resulting `weatherblend.exe` directly - more robust than `dotnet run`.

## Project layout

```
WeatherBlend.slnx
src/WeatherBlend/
  Program.cs                       CLI wiring, DI host
  config.yaml                      all tunables live here
  Config/AppConfig.cs              config POCOs
  Models/                          ForecastRow, ObservationRow
  Collect/OpenMeteoClient.cs       forecast pulling (live + historical)
  Collect/MetarClient.cs           aviationweather.gov METAR parsing
  Storage/ParquetWriter.cs         hive-partitioned Parquet output
  Commands/                        Collect, Backfill, Status, Train, Evaluate
tests/WeatherBlend.Tests/          xUnit smoke tests
data/                              (gitignored) Parquet data lives here
```

## Data layout on disk

```
data/forecasts/
  location=bonehill_rocks/
    model=gfs_seamless/
      date=2025-11-15/
        run=12.parquet            one file per model cycle
data/observations/
  location=bonehill_rocks/
    station=EGTE/
      date=2025-11-15/
        observations.parquet      appended through the day, deduped by time
data/truth/rainfall/
  location=bonehill_rocks/
    station=Bellever Dartmoor/
      date=2025-11-15/
        rainfall.parquet          EA Hydrology 15-min tips
data/models/precipitation/
  ea_bellever_dartmoor/
    v2026-04-23_071842/
      lead_24h.zip lead_48h.zip lead_72h.zip
      training_metadata.json climatology.json
data/models/dry_window/
  MANIFEST.json                    composite keys {slug}/window_{N}h
  ea_bellever_dartmoor/
    window_3h/
      v2026-04-23_101107/
        lead_24h.zip lead_48h.zip lead_72h.zip
        training_metadata.json dry_window_climatology.json
        feature_schema.json feature_importance.json
data/predictions/precipitation/
  ea_bellever_dartmoor/
    model_version=v2026-04-23_071842/
      date=2026-04-23/
        predictions.parquet       P(wet) per lead, deduped by (predicted-at, lead)
data/predictions/dry_window/
  ea_bellever_dartmoor/
    window_3h/
      model_version=v2026-04-23_101107/
        date=2026-04-23/
          predictions.parquet     P(dry window) per lead, deduped by (predicted-at, lead)
```

DuckDB reads this natively:

```sql
SELECT * FROM read_parquet(
    'data/forecasts/**/*.parquet',
    hive_partitioning = true,
    union_by_name = true
);
```

## Known limitations

- **METAR precip is coded, not quantitative.** Fine for a precip occurrence signal,
  insufficient for intensity training. Phase 2: add Met Office DataHub for gauge data
  or radar-derived composite from the Nimrod archive.
- **Lowland METAR as truth for a 393m tor.** Exeter and Yeovilton are both near sea level
  ~30-55km away. Expect systematic biases - the blender can learn these, but a station
  closer to the moor (Princetown, 414m) would be better. Worth registering with Met Office
  DataHub in phase 2.
- **Open-Meteo historical API approximation.** The archive returns best-available forecasts
  per valid-time rather than rigorous "as issued at run T" forecasts. Good enough for PoC
  baselines; go direct to ECMWF/NOAA GRIB archives if you want rigorous verification.
- **Run time is approximated** in the live collector as "most recent hour". Open-Meteo
  doesn't return the exact model cycle timestamp in its JSON response.

## Roadmap

- **Phase 1:** collector, storage, status tooling, 12-month backfill. **Done.**
- **Phase 2:** temperature blender - LightGBM per lead-time bucket, beat best single model. **Done (phase 2b, rolling verify shipped).**
- **Phase 2c:** rich-feature temperature blender — expands from 13 to 88 features (lean + per-model dew/RH/cloud/wind/pressure secondaries + 9 cross-model aggregates). Trains via `--feature-set rich`, saves alongside 2b as a champion/challenger pair: both versions live in `MANIFEST.Active` and produce predictions every cycle; rolling verify reports a 2b-vs-2c MAE delta per lead. **Done — on held-out test, rich beats lean by 0.01–0.05°C at 24/48/72h.**
- **Phase 3a:** precip occurrence blender. Per-station P(wet≥0.1mm/h) classifier trained on EA Hydrology gauges (Bellever, Princetown). Per-lead, same temperature pipeline. **Done — predict + verify live.**
- **Phase 3b:** per-station dry-window classifier — P(at least one contiguous N-hour dry block in target UTC day) for N ∈ {3, 4, 6} at leads 24/48/72h. Replaces the original intensity-regressor plan after the user pivot to "is there time to walk the dog dry?". Per-station per-window LightGBM, climatology baseline, predict + verify wired into the daily/weekly CI alongside temperature + precipitation. **Done — predict + verify live.**
- **Phase 3c:** rich-feature precipitation occurrence blender — 27 lean + 28 extras (18 per-model humidity, 6 per-model surface pressure, 4 EA-observation trailing-rainfall persistence) = 55 features. Same hyperparameters and split as 3a so feature richness is the isolated variable; saved as challenger alongside 3a via the per-station `Active` manifest. Forecast-time precip-persistence and pressure-tendency tiers were dropped — the training parquet only stores leads {24,48,72} per `offset_day` run, so the H-1/H-2/H-3 cells those tiers need don't exist. **Done — Bellever 24/48/72h Brier drops 0.006/0.014/0.013; Princetown 0.002/0.008/0.011 vs 3a. Tier ablation (`precip-ablate`) shows the gains are modest — dropping any one tier costs at most ~0.002 Brier points, and dropping EA persistence actually _improves_ Brier at Bellever + Hexworthy.**
- **Phase 3d:** dry-window improvements alongside the 3b champion. Two challengers, both registered via the per-(station, window) `Active` list so 3b stays in production and the rolling verify scores all three side by side. (1) **3d-shape** — same hyperparameters, same split as 3b, but the 53-feature row is extended with 7 within-day shape features computed from the ensemble-mean hourly precip vector (first/last wet hour, longest dry/wet runs, n rain events, morning/afternoon precip sums) so the blender can see _when_ the rain falls inside a UTC day rather than only daily aggregates. (2) **3d-calibrated** — the 3b model is reused unchanged; a per-lead PAV isotonic regression fit on the validation partition is saved as `calibration.json` and applied at predict time as a strict reweighting of the raw probability. Lessons from the 3a → 3a_isotonic experiment carry over: PAV is risk insurance against miscalibration, not a skill increase. **Done — predict + verify + ablate live.**
- **Intensity regression (deferred indefinitely):** `expected_precip = P(precip) × E[precip | precip > 0]` is the original phase-3 plan — the dry-window framing covers the user-facing probabilistic question without the calibration headaches of conditional precip, so it is unlikely to be revisited.
- **Phase 4:** add ML models as inputs (GraphCast, AIFS - both now published by ECMWF).

## License

Personal project. Respect the terms of service of all upstream data providers
(Open-Meteo, aviationweather.gov, Met Office).
