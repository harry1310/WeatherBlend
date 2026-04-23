# Phase 3b — infrastructure audit

Snapshot taken before any 3b code is written. Captures what's reusable from 3a,
what has to be built new, and the concrete module layout chosen for the phase.

## What's reusable as-is

- **`Evaluate/Precip/PrecipMetrics`** — Brier, BSS, Reliability (N-bin), frequency
  bias, CRPS, 2×2 contingency. Operates on `(probability, binary-truth)` arrays
  with NaN-skip. Nothing precipitation-specific in the maths.
- **`Train/ModelArtifact`** — per-version folders, per-lead zips, JSON manifest
  with per-target / per-station entries, feature-importance + schema JSON. One
  small additive helper for the extra `window_{N}h` level below.
- **`Predict/PredictAnchor`** — anchor-time policy (live vs `--for-date`).
- **`Predict/PredictForecastFilters.LiveCycleAsOf`** — "latest run that covers
  valid-time V at anchor A, excluding `offset_day`" SQL.
- **`Predict/FeatureHashing`** — SHA-256 of a float-array for provenance.
- **R2 storage pattern** — `rclone copy` additive tree under `data/`.
  `data/models/dry_window/…` and `data/predictions/dry_window/…` inherit the
  same sync contract without any workflow-level new plumbing.

## What has to be built new (3a analogue in parens)

- **`Train/DryWindow/DryWindowLabelBuilder`** (no 3a analogue) — consumes EA
  hourly rainfall, emits one binary label per `(UTC-day, window_N)`. Drops days
  with any missing/partial hour.
- **`Train/DryWindow/DryWindowClimatology`** (`PrecipClimatology`) — month-keyed
  only, not `(month, hour)`. Written as `dry_window_climatology.json` alongside
  each version.
- **`Train/DryWindow/DryWindowTrainingRow`** (`PrecipTrainingRow`) — ~50-60
  day-level features; different shape so 3a loaders stay untouched.
- **`Train/DryWindow/DryWindowFeatureBuilder`** (`PrecipFeatureBuilder`) — DuckDB
  aggregation of per-model hourly forecasts to day-level plus join to labels.
- **`Train/DryWindow/DryWindowDataset`** (`PrecipDataset`) — chronological
  70/15/15 split on **date** (not valid-time), same ordering invariants.
- **`Train/DryWindow/DryWindowTrainer`** (`PrecipOccurrenceTrainer`) — LightGBM
  binary with `UnbalancedSets=false`; calibration rationale copied verbatim.
  New concrete type because ML.NET's reflection can't share a trainer across
  training-row classes.
- **`Evaluate/DryWindow/DryWindowBaselines`** — climatology-per-month,
  persistence (yesterday's truth label), each single model's implied daily
  forecast (apply dry-window construction to forecast precip rather than truth).
- **`Evaluate/DryWindow/DryWindowVerifier` + `DryWindowVerifyReporter`** — thin
  wrappers; all metric maths delegates to `PrecipMetrics`.
- **`Commands/DryWindowTrain/Predict/VerifyCommand`** — kept separate from the
  combined `TrainCommand` so the phase-3a temperature+precip dispatcher stays
  simple. CLI dispatch via `--target dry-window`.
- **`Models/DryWindowPredictionRow`** — day-level row with station, window,
  lead, probability, climatology reference, per-model implied day labels, and
  a SHA-256 feature hash.

## Decision — parallel build, not in-place extension

Extending `PrecipTrainingRow` to carry a day-level feature vector would break
the 3a feature hash, force conditional SQL through the feature builder, and
put two very different trainers inside one class. The cost of writing new
types is smaller than the cost of mixing hour-rows and day-rows through the
same pipeline. `PrecipMetrics` and `ModelArtifact` are the only shared leaves.

## On-disk layout

```
data/models/dry_window/
  MANIFEST.json
  ea_bellever_dartmoor/
    window_3h/
      v{yyyy-MM-dd_HHmmss}/
        lead_24h.zip lead_48h.zip lead_72h.zip
        training_metadata.json
        dry_window_climatology.json
        feature_schema.json
        feature_importance.json
    window_4h/...
    window_6h/...
  ea_princetown/...

data/predictions/dry_window/
  {ea_station}/
    window_{N}h/
      model_version={v}/
        date={yyyy-MM-dd}/
          predictions.parquet
```

`MANIFEST.json` uses composite keys inside the existing `Stations` dict:
`"ea_bellever_dartmoor/window_3h"`. This lets `ModelArtifact.ListStations`
and `ResolveStationVersionDir` work without modification; two tiny helpers
encode/decode the composite key at the command layer.

## Brief deltas (for the log)

- **"Python evaluation subprocess"** in the ground rules is a mis-statement
  about 3a — 3a is pure .NET, metrics live in `PrecipMetrics.cs`. No Python
  to reuse; nothing new added.
- **Stations**: config has Bellever + Princetown + Dartmoor-nr-Hexworthy; the
  brief says Bellever primary, Princetown secondary. **Hexworthy is skipped**
  in 3b. 18 models = 3 windows × 3 leads × 2 stations.
- **Leads in hours, 3-tuple {24,48,72}**: same convention as 3a. For a target
  UTC-day D, training rows read forecasts with `LeadHours ∈ {lead..lead+23}`
  from the forecast run whose anchor sits `lead` hours before midnight D.
- **Predict-row schema** (brief truncated): mirrors 3a with `window_hours`
  and `predicted_for_date` added, same provenance fields.
