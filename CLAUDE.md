# WeatherBlend

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
dotnet run --project src/WeatherBlend -- collect                    # one cycle (live forecasts + recent METAR)
dotnet run --project src/WeatherBlend -- status                     # disk summary
dotnet run --project src/WeatherBlend -- inspect --path <parquet>   # dump one file
dotnet run --project src/WeatherBlend -- compare --glob "<glob>"    # cross-model agreement view
dotnet run --project src/WeatherBlend -- backfill --source all --start 2023-04-19 --end 2026-04-18
dotnet run --project src/WeatherBlend -- train --target temperature # phase 2 stub
dotnet run --project src/WeatherBlend -- evaluate                   # phase 2 stub
```

`backfill --source` accepts `previous-runs | era5 | metar | rainfall | all`. `collect`
is what Task Scheduler should fire on a cycle (currently every 3h per README; 6h is
also reasonable).

## Phased roadmap

1. **Phase 1:** collector, storage, status, ERA5 + OGIMET backfill. **Done.**
2. **Phase 2:** temperature blender — LightGBM per lead-time bucket, trained on ERA5,
   verified on METAR; beats persistence/climatology/mean-of-models/best single. **Done (2b: rolling verify shipped).**
3. **Phase 3a:** per-station P(wet ≥ 0.1mm/h) classifier on EA Hydrology gauges
   (Bellever, Princetown). Per-lead, same temperature pipeline. **Done — predict + verify live.**
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
