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

# Phase 3a: per-station precipitation occurrence blender (P(wet>=0.1mm/hour))
dotnet run --project src/WeatherBlend -- train --target precipitation --station "Bellever Dartmoor"
dotnet run --project src/WeatherBlend -- predict --target precipitation --truth-station all
dotnet run --project src/WeatherBlend -- verify --target precipitation --truth-station all
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
WeatherBlend.sln
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
data/predictions/precipitation/
  ea_bellever_dartmoor/
    model_version=v2026-04-23_071842/
      date=2026-04-23/
        predictions.parquet       P(wet) per lead, deduped by (predicted-at, lead)
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
- **Phase 3a:** precip occurrence blender. Per-station P(wet≥0.1mm/h) classifier trained on EA Hydrology gauges (Bellever, Princetown). Per-lead, same temperature pipeline. **Done — predict + verify live.**
- **Phase 3b:** precip intensity regression E[mm | wet] per station, combined to expected_precip = P(wet)·E[mm|wet].
- **Phase 3c:** quantile regressions for probabilistic thresholds (P>1mm, P>5mm, P>10mm); CRPS reporting.
- **Phase 4:** add ML models as inputs (GraphCast, AIFS - both now published by ECMWF).

## License

Personal project. Respect the terms of service of all upstream data providers
(Open-Meteo, aviationweather.gov, Met Office).
