# WeatherBlend

A proof-of-concept for blending eight free NWP (numerical weather prediction)
models against local observations to produce a better single-location forecast
than any constituent model.

**PoC location:** Bonehill Rocks, Dartmoor (50.5831°N, 3.7931°W, 393 m).

**Live site:** rendered HTML deploys to Cloudflare Pages on every predict cycle
+ a 2-hourly render-only cron so rolling-MAE and observed-rainfall annotations
stay current. CI runs on GitHub Actions; data lives on Cloudflare R2.

## What it predicts

| Target | Truth | Leads | Phases |
|---|---|---|---|
| Temperature 2 m | ERA5 reanalysis | 24 / 48 / 72 / 96 / 120 h | 2b lean (13 ft), 2c rich (88 ft) |
| Precipitation occurrence P(wet ≥ 0.1 mm/h) | EA Hydrology rainfall (Bellever, Princetown, Hexworthy) | 24 / 48 / 72 / 96 / 120 h | 3a lean (27 ft), 3c rich (55 ft) |
| Dry-window per UTC day, N ∈ {3, 4, 6} h | EA Hydrology rainfall (Bellever, Princetown) | 24 / 48 / 72 h | 3b lean (53 ft), 3d-shape rich (60 ft) |
| Element blenders: humidity, wind, shortwave radiation, cloud cover | ERA5 reanalysis | 24 / 48 / 72 h | lean only — feed feels-like |
| Feels-like: UTCI (Bröde 2012) + Steadman 1994 apparent temp | derived (no training) | inherits from temperature lead set | n/a |

Each blender is a **per-lead LightGBM model**. Champion + challenger phases run
side by side, both populating the prediction parquet; the rendered site shows
the latest per family on the Models page and side-by-side comparisons on the
Skill pages.

## NWP inputs

GFS (NCEP), ECMWF IFS, DWD ICON, Météo-France, UK Met Office UM Global,
Environment Canada GEM, ECMWF AIFS (GraphCast-class AI model), and JMA Global.
All routed through Open-Meteo's live + historical-forecast API; no model-specific
GRIB plumbing. Each NWP carries different optional-availability — JMA contributes
to precipitation only; AIFS is required everywhere except dry-window.

## Prerequisites

- .NET 10 SDK (`dotnet --version` should report 10.x)
- Windows, macOS, or Linux
- ~1 GB free disk for a year of backfill
- For full CI parity: `rclone` configured against an R2 (or S3) bucket — see
  `.github/workflows/predict.yml` for the env-var contract

## Quick start

```bash
# Build
dotnet build src/WeatherBlend/WeatherBlend.csproj -c Release

# One collection cycle (live forecasts + METAR)
dotnet run --project src/WeatherBlend -- collect

# What's on disk
dotnet run --project src/WeatherBlend -- status

# Backfill historical training data (~30 min, polite rate-limiting)
dotnet run --project src/WeatherBlend -- backfill --source all --start 2024-11-01 --end 2025-10-31
```

### Train a target

```bash
# Temperature: lean (Phase 2b) + rich (Phase 2c) champion/challenger
dotnet run --project src/WeatherBlend -- train --target temperature --feature-set lean --lead all
dotnet run --project src/WeatherBlend -- train --target temperature --feature-set rich --lead all

# Precipitation occurrence per station: 3a (lean) + 3c (rich)
dotnet run --project src/WeatherBlend -- train --target precipitation --feature-set lean
dotnet run --project src/WeatherBlend -- train --target precipitation --feature-set rich

# Dry-window per (station, window-length): 3b champion + 3d-shape challenger
dotnet run --project src/WeatherBlend -- train --target dry-window --feature-set lean --window all --lead all
dotnet run --project src/WeatherBlend -- train --target dry-window --feature-set rich --window all --lead all

# Element blenders feeding feels-like
dotnet run --project src/WeatherBlend -- train --target wind                --lead all
dotnet run --project src/WeatherBlend -- train --target humidity            --lead all
dotnet run --project src/WeatherBlend -- train --target shortwave-radiation --lead all
dotnet run --project src/WeatherBlend -- train --target cloud-cover         --lead all
```

### Predict + verify

```bash
# Predict every active version of every target. Idempotent re-run.
dotnet run --project src/WeatherBlend -- predict --target temperature
dotnet run --project src/WeatherBlend -- predict --target precipitation --truth-station all
dotnet run --project src/WeatherBlend -- predict --target dry-window    --truth-station all --window all
dotnet run --project src/WeatherBlend -- predict --target wind
dotnet run --project src/WeatherBlend -- predict --target humidity
dotnet run --project src/WeatherBlend -- predict --target shortwave-radiation
dotnet run --project src/WeatherBlend -- predict --target cloud-cover
dotnet run --project src/WeatherBlend -- predict --target feels-like   # joins the five above

# Verify (weekly in CI)
dotnet run --project src/WeatherBlend -- verify --target temperature
```

### Render the static site locally

```bash
# Reads R2-synced parquet trees, writes data/site/*.html
dotnet run --project src/WeatherBlend -- render-site --output data/site
```

## Project layout

```
WeatherBlend.slnx
src/WeatherBlend/
  Program.cs                       CLI wiring, DI host
  config.yaml                      all tunables
  Config/                          POCOs bound from yaml
  Models/                          row schemas (PredictionRow, PrecipPredictionRow, ...)
  Collect/                         clients (OpenMeteo, METAR, EA Hydrology, MetOffice, ...)
  Storage/                         ParquetWriter, ParquetReader
  Train/                           feature builders + blender trainers
    Element/                       per-element blenders (humidity, wind, radiation, cloud)
    DryWindow/                     phase 3b / 3d-shape feature builder + climatology
  Predict/
    FeelsLike/                     UTCI + Steadman calculator + the join pipeline
  Commands/                        CLI subcommands (Collect, Train, Predict, Verify, RenderSite, ...)
  Site/                            HTML renderers (no I/O, takes pre-loaded objects)
tests/WeatherBlend.Tests/          xUnit
data/                              gitignored — Parquet output + rendered site + reports
.github/workflows/                 collect / predict / render-site / verify / era5-refresh
```

## Data layout on disk

```
data/forecasts/location=bonehill_rocks/model={nwp}/date=YYYY-MM-DD/{run.parquet | previous_runs.parquet}
data/observations/location=bonehill_rocks/station={icao}/date=YYYY-MM-DD/observations.parquet
data/truth/era5/location=bonehill_rocks/date=YYYY-MM-DD/data.parquet
data/truth/rainfall/location=bonehill_rocks/station={Name}/date=YYYY-MM-DD/rainfall.parquet
data/models/{target}/[{station}/[window_{N}h/]]v{ts}_{phase}/
data/predictions/{target}/[{station}/[window_{N}h/]]model_version={v}/date=YYYY-MM-DD/predictions.parquet
data/predictions/feels_like/model_version=v1/date=YYYY-MM-DD/predictions.parquet
data/reports/                      training-time markdown reports + ablation CSVs
data/site/                         rendered HTML, deployed to Cloudflare Pages
```

DuckDB reads the partitioned trees natively:

```sql
SELECT * FROM read_parquet(
    'data/predictions/temperature/**/*.parquet',
    hive_partitioning = false,        -- in-file ModelVersion column wins over hive key
    union_by_name = true              -- handle schema drift across phases
);
```

## CI cadence

| Workflow | Cron (UTC) | Job |
|---|---|---|
| `collect` | `:15 02, 08, 14, 20` (4× daily) | pull fresh forecasts + METAR + EA rainfall, push to R2 |
| `predict` | `:45 02, 08, 14, 20` (4× daily) | run every active blender, push predictions |
| `render-site` | on `predict` / `verify` completion + `0 */2 * * *` | regenerate HTML, deploy to Pages |
| `era5-refresh` | `0 6 * * *` (daily) | fetch the latest ERA5 release, push to R2 |
| `verify` | `30 9 * * 1` (Mondays) | rolling MAE / Brier vs training-test, drift flag |

The repo is public, so GitHub Actions minutes are unlimited. R2 storage is well
inside Cloudflare's free tier.

## Known caveats

- **ERA5 grid-cell vs point.** ERA5 is 0.25° (~25 km) gridded reanalysis. The
  blender learns the systematic offset between the cell average and Bonehill's
  393 m elevation, but you'd want a downscaled product for tighter ground truth.
- **METAR is a sanity check, not truth.** Exeter (EGTE) and Yeovilton (EGDY)
  are 30–55 km away and at sea level. Useful as a cross-check that the blender
  beats real observations, not as a fitting target.
- **Open-Meteo historical-forecast approximation.** Their archive returns
  best-available per valid-time, not rigorous "as issued at run T". Fine for
  PoC training; insufficient for publication-grade reverification.
- **Train/predict distribution mismatch (precip + dry-window).** Training pulls
  `RunTimeSource='offset_day'` rows so each lead has an exact target-anchor
  pairing; live predict uses the most-recent live-cycle row per (valid-time,
  model). Identical feature shape, slightly different distribution. Documented
  in `docs/PHASE3B_AUDIT.md`.

## Roadmap

| Phase | Status |
|---|---|
| 1: collector, storage, status, 12-month backfill | done |
| 2b: temperature lean LightGBM | done — predict + rolling verify live |
| 2c: temperature rich (88 ft) | done — runs alongside 2b as challenger |
| 3a: precipitation occurrence (lean) | done — per-station P(wet≥0.1 mm/h) |
| 3b: dry-window (lean) | done — per-(station, window) classifier |
| 3c: precipitation rich (55 ft) | done — Brier −0.006 to −0.014 vs 3a per station |
| 3d-shape: dry-window with within-day shape features | done — runs as 3b challenger |
| 4: ML / AI NWPs as inputs | done — ECMWF AIFS shipped 2026-04-27, JMA Global 2026-04-28 |
| Feels-like: UTCI + Steadman, derived (no training) | done — both indices on the home card |
| Element blenders (humidity / wind / radiation / cloud) | done — feed feels-like |

Calibration phases (3a-isotonic, 3d-calibrated) were trained, scored against
their non-calibrated counterparts, and retired 2026-04-29 — PAV calibration
didn't move test Brier, so they no longer ship.

## License

Personal project. Respect the terms of service of every upstream data provider
(Open-Meteo, aviationweather.gov, Met Office DataHub, EA Hydrology, ECMWF).
