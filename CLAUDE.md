# WeatherBlend

## NEVER GUESS — investigate, or ask

When answering "where does X run", "what does Y consume", "what's the predict /
retrain chain order", or any other question about how the system is wired
together: **read the code first.** Do not infer from filenames. Do not assume
based on naming patterns. Do not extend "this is how 3f works" to "this is
how wind_mvn must work" without checking.

Authoritative sources for common questions:

| Question | Read this first |
|---|---|
| Predict/retrain chain order, what fires what | `cloudflare/scheduler-worker/src/index.ts` (the `handleWorkflowRun` hops are the single source of truth; comments in `wrangler.toml` are a useful index but the code is canonical) |
| What a workflow actually does | `.github/workflows/<name>.yml` and its composite actions in `.github/actions/` |
| What features a Python predictor consumes | The relevant `scripts/predict_*.py` and `scripts/_shared.py` MODELS_LEAN / similar lists |
| Which Python script Cloudflare dispatches | `WORKFLOW_FOR_CRON` in the worker, plus the workflow_run hops below it |
| What `collect` collects | `src/WeatherBlend/Commands/CollectCommand.cs` |
| What `s3-collect` collects | `src/WeatherBlend/Commands/S3CollectCommand.cs` |
| Active phase set + their roles | `src/WeatherBlend/Config/phases.yaml` (loaded via `PhaseRegistry`) |

If you cannot quickly check a fact you need, **ask** rather than infer. "I'd
need to read X before answering — want me to?" is the right move. Inventing
a plausible answer and shipping it as fact has burnt trust at least twice
(2026-05-07 feels-like sparsity, 2026-05-27 wind-chain incident). Don't do it.

## Project

PoC for blending ~6 free NWP models against ERA5 reanalysis (training truth)
and METAR observations (verification truth) to produce a better single-location
forecast than any single model. Target site: **Bonehill Rocks, Dartmoor**
(50.5831°N, 3.7931°W, 393m).

Stack: .NET 10 console app, Parquet + DuckDB on disk (no database), Microsoft.ML
LightGBM for the blender. Open-Meteo for forecasts + ERA5; aviationweather.gov
for live METAR; OGIMET for historical METAR.

## Layout

```
src/WeatherBlend/
  Program.cs              CLI + DI host
  config.yaml             all tunables (location, models, variables, paths)
  Config/AppConfig.cs     POCOs bound from yaml
  Models/                 ForecastRow, ObservationRow, Era5Row
  Collect/                OpenMeteoClient, MetarClient, Era5Client, OgimetClient
  Storage/ParquetWriter   hive-partitioned writer (forecasts/observations/era5)
  Commands/               Collect, Backfill, Status, Inspect, Compare, Train, Evaluate
tests/WeatherBlend.Tests/
data/                     (gitignored) parquet output
```

Hive paths:
- `data/forecasts/location=<n>/model=<id>/date=<yyyy-MM-dd>/run=<HH>.parquet` — NWP predictions (live + historical-forecast API, same schema)
- `data/truth/era5/location=<n>/date=<yyyy-MM-dd>/data.parquet` — reanalysis, gapless training truth
- `data/truth/metar/location=<n>/station=<icao>/date=<yyyy-MM-dd>/observations.parquet` — station obs, verification truth

## Commands I'll use most

```powershell
dotnet build
./scripts/test-fast.ps1                                             # inner-loop tests (~20s, excludes smokes)
./scripts/test-smoke.ps1                                            # integration smokes only (~20min, pre-push)
dotnet run --project src/WeatherBlend -- collect                    # one cycle (live forecasts + recent METAR)
dotnet run --project src/WeatherBlend -- status                     # disk summary
dotnet run --project src/WeatherBlend -- inspect --path <parquet>   # dump one file
dotnet run --project src/WeatherBlend -- compare --glob "<glob>"    # cross-model agreement view
dotnet run --project src/WeatherBlend -- backfill --source all --start 2023-04-19 --end 2026-04-18
dotnet run --project src/WeatherBlend -- train --target temperature # phase 2 stub
dotnet run --project src/WeatherBlend -- evaluate                   # phase 2 stub
```

**Tests: fast inner loop vs smokes.** `dotnet test` runs all ~910, but ~97% of
the wall-time is ~13 `[Trait("Category","Smoke")]` integration tests (end-to-end
train→sync→predict→parquet chains; they catch the hard-to-spot wiring bugs). For
the inner loop run `test-fast.ps1` (`--filter "Category!=Smoke"`, ~20s); run
`test-smoke.ps1` as the final check before pushing major work. The smokes are
slow for structural reasons (bash sync script + rclone + DuckDB/parquet IO +
native cold-start), NOT model math — LightGBM iterations are already capped via
`WB_SMOKE_ITER` in the test `EnvScope`, so don't "fix" them by cutting leads or
the sync path (that's the coverage).

`backfill --source` accepts `previous-runs | era5 | metar | rainfall | all`. `collect`
is what Task Scheduler should fire on a cycle (currently every 3h per README; 6h is
also reasonable).

## Phased roadmap

1. **Phase 1:** collector, storage, status, ERA5 + OGIMET backfill. **Done.**
2. **Phase 2:** temperature blender — LightGBM per lead-time bucket, trained on ERA5,
   verified on METAR; beats persistence/climatology/mean-of-models/best single. **Done (2b: rolling verify shipped).**
3. **Phase 3a:** per-station P(wet ≥ 0.1mm/h) classifier on EA Hydrology gauges
   (Bellever, Bovey Tracey, Hexworthy). Per-lead, same temperature pipeline. **Done — predict + verify live.**
4. **Phase 3b:** per-station, per-window dry-window classifier — P(at least one
   contiguous N-hour dry block in target UTC day) for N ∈ {3, 4, 6} at leads 24/48/72h.
   Replaces the original intensity-regressor plan after the user pivot to "is there
   time to walk the dog dry?". **Done — predict + verify live, 18 models in CI.**
5. **Phase 3d:** dry-window improvements alongside 3b. **3d-shape** adds 7 within-day
   shape features (60-feature variant, `--feature-set rich`); **3d-calibrated** wraps
   each saved 3b model with per-lead PAV isotonic calibration via `dry-window-calibrate`.
   Both register as challengers in the per-(station, window) `Active` list; predict +
   verify score all three side by side; `dry-window-ablate` emits the comparison report.
6. **Phase 4:** add ML models (GraphCast, AIFS) as inputs.

## Key design decisions (see docs/DESIGN.md for full reasoning)

- **Parquet + DuckDB, no database.** Single-location volume is tens of MB/year;
  a DB adds ops burden with no speed benefit at this size.
- **Open-Meteo for forecasts + ERA5.** Six models + reanalysis through one consistent
  JSON API with the same query pattern. Tradeoff: their interpolation, deterministic runs only.
- **Two truth sources:** ERA5 for *training* (gapless, quantitative precip);
  METAR for *verification* (real obs, sanity check that the blend beats reality, not just reanalysis).
- **LightGBM blender.** Native missing-value handling (models drop in/out per
  lead-time), monotonic constraints, quantile mode, first-class .NET trainer.
- **Per-lead-time models, not one big model.** Skill characteristics differ wildly
  across horizons. Buckets: [1-3h], [6-12h], [24-36h], [48-72h], [96-120h], [144-168h].
- **Dry-window over intensity (phase 3b):** the original two-stage P(wet) ×
  E[mm | wet] plan was scrapped after the user reframed the question as
  "is there a long-enough dry block today?". One classifier per (station, window)
  beats calibrating a conditional intensity regressor for the actual decision.
- **Verification rules pre-committed:** time-based splits only; walk-forward
  validation; report per lead time; Brier + reliability for precip alongside MAE.

## Auto-retrain (Sunday weekly sweep)

`Config/phases.yaml` is the canonical active-phase registry; `PhaseRegistry`
loads it at startup and `ActivePhasePolicy` is a thin static façade for
back-compat. The site, train workflows, and predict/verify all read from
the same list — adding a new phase (e.g. 5b) is mostly one entry there.

**Cadence.** The Cloudflare scheduler worker's 12:00 UTC noon tick fires
`previous-runs-refresh.yml` (alongside `truth-refresh.yml`). The worker
then chains the retrain off `workflow_run` completion webhooks, strictly
serially: on a **Sunday** `previous-runs-refresh` success it dispatches
`retrain-python.yml` (WeatherProbabilistic), and on `retrain-python`'s
completion it dispatches `retrain-blenders.yml` (WeatherBlend) — so the
three run refresh → python → blenders. Serial so retrain-blenders' 4b
mint reads this cycle's fresh 4a rather than racing it. Both retrain
hops dispatch via `workflow_dispatch` with `force=true`. Manual override:
`gh workflow run retrain-python.yml -f force=true` (or `retrain-blenders.yml`).

**Pre-train sanity gate (RetrainGuard).** Each trainer (TrainCommand,
DryWindowTrainCommand, ElementTrainerHarness on the .NET side; train_4a.py
on the Python side) computes a `training_summary.json` post-fit and
compares against the previous run's
on disk. Defaults: rows ±30%, NaN% absolute 0.20, label-rate 0.10,
features-effective 0 (any change aborts). Fail = log structured breach,
exit 4, skip the manifest promotion. Orphan version dirs sit on disk
but never become live (predict + verify never see them).

**Drift alerting.** `verify.yml` exits 4 on any `DriftFlag=true` row in
the freshly-emitted history; the existing GH App webhook auto-files
`[ci-fail] verify` issues. Cooldown = App's built-in run-failure-signature
de-dupe.

**On-call playbook.**
- `[ci-fail] retrain-python` or `[ci-fail] retrain-blenders` issue: open
  the failed run's logs, find the "Retrain guard FAIL" line. Triage
  options: real upstream issue (collector failed → fix and re-run),
  legitimate distribution shift (raise the tolerance band for that cell
  via `data/models/retrain_tolerances.json` overrides — once that's
  shipped), or schema change (update phases.yaml + the trainer's
  feature builder).
- `[ci-fail] verify` issue: open `data/reports/verify_*.md` from the
  workflow artifact, identify the (target, station, lead) cells that
  drifted. If only one cell, likely a data-side issue (truth source,
  upstream model). If many cells across one target, suspect a real
  regime shift — wait for next Sunday retrain to absorb.
- A **partial retrain** (some phases pass, others fail) still pushes
  the passing bundles to R2. The previous version stays Current for the
  failing phases; verify the next day will scope-down the issue.

## Known limitations

- Lowland METAR (EGTE / EGDY) as verification truth for a 393m tor — systematic
  biases. ERA5's 0.25° grid + elevation interpolation is closer to representative.
- Open-Meteo historical-forecast archive returns best-available per valid-time,
  not rigorous "as issued at run T". Good enough for PoC baselines.
- Live collector approximates run time as "most recent hour" — historical rows
  end up with negative `LeadHours`. Filter `WHERE LeadHours >= 0` for analysis.
- OGIMET is community-funded and rate-limits aggressively (5s between requests
  enforced in `BackfillCommand`). Hammer it and the IP gets blocked.

## Gotchas worth knowing now

- `config.yaml` is copied to build output via `<CopyToOutputDirectory>`; loaded
  from `AppContext.BaseDirectory` (or `WEATHERBLEND_CONFIG` env var).
- YAML provider is `NetEscapades.Configuration.Yaml` (the
  `Microsoft.Extensions.Configuration.Yaml` package only ships a 2.0.0-preview).
- `ObservationRow`/`Era5Row` have `[SetsRequiredMembers]` parameterless ctors so
  `ParquetSerializer.DeserializeAsync<T>` satisfies its `new()` constraint.
- DuckDB queries use `hive_partitioning = false` because the hive keys
  `model`/`station` collide with in-file `Model`/`Station` columns under
  case-insensitive resolution; in-file columns are authoritative.
- OGIMET response is CSV-prefixed: `ICAO,YYYY,MM,DD,HH,mm,METAR_TEXT`. NIL
  reports are skipped. The minimal METAR parser covers routine bodies only —
  TAF amendments, RVR, runway state, etc. are ignored.
