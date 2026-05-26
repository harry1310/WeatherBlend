# Smoke test harness — rough plan

**Goal:** catch the wiring / typing / env-var / SQL-shape class of bugs in
<30s **locally**, before pushing to GH Actions. The 2026-05-26 ship-day
walked into ~6 of these (WB_LOCATION hardcoded, train_features list-vs-
ndarray, predict_3f join dedup, NaN-JSON, climatology-gate ordering,
schema-mismatch in old bundles) — each cost ~10 min of CI iteration to
catch when they should have been ~30s on the laptop.

**Non-goal:** data-quality / regression-skill testing. Those need real
parquet trees and stay in CI. The smoke harness only proves the code
PATH works end-to-end on synthetic data — bundle reads, parquet schemas,
env-var resolution, SQL filter shapes, manifest plumbing.

## Common shape

Both halves of the harness:

1. `tempdir` scoped via `WEATHERBLEND_DATA_ROOT` (already env-driven).
2. Synthetic fixture writers that produce parquet trees with the same
   schemas/partitions the production code expects.
3. Invoke the production entry point against the fixture.
4. Assert non-empty output + correct schema + load-roundtrip.
5. Cleanup tempdir.

Test runtime budget: **< 30s per phase** including pytest/xUnit setup.

## WP side — `tests/test_smoke_3f.py` (pytest)

**Phases covered first:** 3f (train + predict end-to-end). Extends to 4a
+ 5a in a follow-up.

Fixture builder (`tests/_smoke_fixtures.py`):

- `make_forecast_tree(root, location, date_range, models, lead_hours)` —
  writes `data/forecasts/location={loc}/model={m}/date={d}/...parquet`
  with the schema train_3f reads (ValidTimeUtc, RunTimeUtc, RunTimeSource,
  LeadHours, LocationName, Model, Precipitation, RelativeHumidity2m,
  Temperature2m, DewPoint2m, CloudCover{Low/Mid/High}, Cape,
  WindSpeed10m, WindDirection10m, SurfacePressure). Realistic
  distributions (Beta for precip, Gaussian for temp, etc.) so NGBoost
  has signal to fit on.
- `make_rainfall_truth(root, location, station, date_range)` — writes
  `data/truth/rainfall/location={loc}/station={st}/date={d}/rainfall.parquet`
  with ObservedTimeUtc, Value15MinMm, LocationName, StationName.
  Realistic wet/dry pattern (~25% wet rate, occasional storms).
- `make_3a_predictions(root, station, anchor, leads)` — writes
  `data/predictions/precipitation/{station}/model_version={v}/date={anchor}/predictions.parquet`
  matching the PrecipPredictionRow schema. ProbWet drawn from a Beta
  matching the truth wet rate.
- `make_manifest(root, target, station, active_versions)` — writes
  `data/models/{target}/MANIFEST.json` with the Active list train_3f
  reads to resolve the bound 3a champion.

Tests:

1. `test_train_3f_end_to_end_smoke` — fixture covers 180 days × 3
   Membury stations × 7 NWPs × 5 leads, runs `train_3f.main()` via
   subprocess (isolates env), asserts bundle dir exists with all 5
   pickled NGBRegressors + valid training_metadata.json + preprocess.json.
2. `test_predict_3f_end_to_end_smoke` — same fixture + a bundle from
   the train smoke + bound 3a predictions parquet, runs
   `predict_3f.main()`, asserts
   `data/predictions/rainfall_amount/{station}/.../predictions.parquet`
   exists with ≥1 row per lead and the right column set.
3. `test_predict_3f_dedupes_bound_3a_parquet` — already shipped as
   `test_predict_3f_pi_join.py` after the 2026-05-26 broadcast-shape
   bug. Same pattern but at the predict_main level.
4. `test_train_3f_writes_ndarray_train_features` — regression for
   the 4th-attempt list-vs-ndarray bug. Inspect the in-process
   `train_one_station` return for `train_features` type.

Follow-up phases (4a + 5a) take the same fixture builders but invoke
`train_4a.main` / `run_phase5_bayesian.main`.

## WB side — `tests/WeatherBlend.Tests/PredictSmokeTests.cs` (xUnit)

**Phases covered first:** 3a (PrecipPredictCommand), 3p
(DryWindowPredictCommand → RunPhase3pAsync), 4b (Phase4bPredictCommand).
Extends to 3o + 3c when their predict paths grow new branches.

Fixture builder (`tests/WeatherBlend.Tests/SmokeFixtures.cs`):

- Same shape as WP: synthetic forecast + truth + predictions trees in a
  temp dir.
- For phases that need a real LightGBM `model.zip`: train a 100-row toy
  binary classifier in test setup (~1s with `ML.NET LightGbmBinaryTrainer`)
  and save it to the fixture's `model_version={v}/lead_{N}h.zip`.
- For phases that only need synthetic parquets (3p, 4b): no toy LGBM
  needed.

Tests (key invariant: **predictions parquet has > 0 rows at the live
anchor**):

1. `Precip_3a_RunAsync_writes_non_empty_predictions` — fixture with
   reported live forecasts, toy 3a bundle in manifest Active, run
   `PrecipPredictCommand.RunAsync` for one station, assert the
   `model_version=v..._phase3a/date={anchor}/predictions.parquet`
   exists with ≥1 row per lead.
2. `Precip_3o_RunAsync_queries_live_tree_not_offset_day` — regression
   for the 2026-05-26 `LoadAuxNwpMeans` bug. Fixture has BOTH
   `RunTimeSource='reported'` and `'offset_day'` rows; predict must
   produce non-empty output because the live tree has data even if
   offset_day doesn't.
3. `DryWindow_3p_RunPhase3pAsync_skips_climatology_check` — regression
   for the 2026-05-26 climatology-gate bug. Fixture 3p bundle has
   `correlation.json` + `training_metadata.json` only (NO
   `dry_window_climatology.json`); predict must NOT reject it.
4. `Phase4b_RunAsync_joins_4a_and_3o_and_writes` — fixture has tiny 4a
   and 3o predictions parquets; phase4b-predict must produce a
   `model_version=v..._phase4b/` parquet with the union.

## Out of scope

- LightGBM hyperparameter search (real CI handles this).
- Real EA / Open-Meteo data fetching (those go via the collector
  workflows which already have unit tests).
- Worker (TypeScript) testing — that's a separate harness, the
  `PredictAndRenderEquivalenceTests.cs` we already shipped covers
  workflow-YAML divergence.

## Wiring + run pattern

Both halves:

- Run locally: `pytest tests/test_smoke_3f.py -v` and `dotnet test
  tests/WeatherBlend.Tests --filter "FullyQualifiedName~PredictSmokeTests"`.
- Run in CI: extend the existing test-runner workflows to include the
  smoke files — they're cheap enough not to gate them out.
- Failure mode: a smoke failure on PR / push is BLOCKING. The whole
  point is to catch these before they hit production.

## Estimated build cost

- WP side: ~3-4 hours including fixture builders, the 4 tests, and
  initial debugging on this laptop.
- WB side: ~4-6 hours including the toy LightGBM training step and
  fixture builders for the more complex 3o + 4b paths.

Total: roughly 1 working day. Recoverable from the next CI iteration
that catches a wiring bug (so 1-2 such avoided bugs already pays back
the build cost).
