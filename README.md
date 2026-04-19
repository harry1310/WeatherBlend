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

- **Phase 1 (current):** collector, storage, status tooling. Accumulate data.
- **Phase 2:** temperature blender - LightGBM per lead-time bucket, beat best single model.
- **Phase 3:** quantitative precip data source (Met Office DataHub / Nimrod radar).
- **Phase 4:** precip occurrence classifier.
- **Phase 5:** precip intensity + probabilistic thresholds (P>0.1mm, P>1mm, P>5mm, P>10mm).
- **Phase 6:** add ML models as inputs (GraphCast, AIFS - both now published by ECMWF).

## License

Personal project. Respect the terms of service of all upstream data providers
(Open-Meteo, aviationweather.gov, Met Office).
