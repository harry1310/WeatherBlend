# Forecast data sources — what each one returns and why it matters

Three independent sources feed the forecast tree under
`data/forecasts/location=.../model=<id>/date=<yyyy-MM-dd>/run=<HH>.parquet`.
They share the same `ForecastRow` schema but the (RunTime, LeadHours) axis
they fill is _very_ different. Most surprises about forecast density at a
given lead come from picking the wrong source for the job.

The three are tagged in-row via `RunTimeSource ∈ {offset_day, reported,
exact}` so downstream code can tell them apart without inspecting paths.

---

## 1. Open-Meteo `previous_runs` API

`OpenMeteoClient.FetchPreviousRunsAsync` →
`https://previous-runs-api.open-meteo.com/v1/forecast`

**What it returns.** A `(valid_time × lead-bucket)` matrix expressed as
hourly rows with one column per `var_previous_dayN` for `N ∈ {1, …, 7}`.
Each `previous_dayN` column at a given hourly valid time is "what the
run from N days before this valid time predicted for this hour".

**Lead axis: discrete, multiples of 24 only.** The `N` index quantises
lead to `LeadHours ∈ {24, 48, 72, 96, 120, 144, 168}`. There is no
lead-25 or lead-37 column. Stored as `LeadHours = 24·N`,
`RunTimeUtc = ValidTimeUtc − 24·N h`, `RunTimeSource = offset_day`.

**Valid-time axis: hourly, full window.** A request for `start_date=A`,
`end_date=B` returns one row per UTC hour in `[A, B+1)`, every one of
which has 7 lead-bucket forecast values populated (those rows are then
expanded to 7 `ForecastRow`s in our code).

**Density at a specific lead.** Lead-24 valid times exist at **every
hour of the day, every day in the window**. So at lead 24 a 30-day
backfill produces ≈ 720 rows per (location, model). That hourly-at-lead-24
density is what the Phase 2/3 training datasets are built on.

**Used by.** `archive-backfill.yml` (Phase 2 + Phase 3 training data,
including the WeatherProbabilistic Bayesian classifier).

**The catch.** The lead axis is _exactly_ {24, 48, 72, …}. Nothing in
between. If you wanted a lead-30 model you couldn't train it off this
endpoint — you'd have to fall back to live cycles (#2) or AWS (#3).

---

## 2. Open-Meteo live forecasts

`OpenMeteoClient.FetchLiveAsync` →
`https://api.open-meteo.com/v1/forecast`

**What it returns.** The model's most-recently-published cycle, expanded
hourly out to `forecast_days × 24` rows. The run-time itself comes from
a sibling metadata call (`/data/{model}/static/meta.json` →
`last_run_initialisation_time`) and is tagged `RunTimeSource = reported`.
If the metadata call fails we fall back to wall-clock-floored-to-hour
and tag `synthesised`.

**Lead axis: hourly, dense, but scoped to one cycle at a time.** A
single call returns one cycle's worth of hourly forecasts at
`LeadHours ∈ {1, 2, …, forecast_days·24}`. Calling at HH:45 captures
whatever cycle Open-Meteo has just finished publishing.

**Valid-time axis: hourly, anchored at the run.** Valid times start at
or just after `RunTimeUtc` and run forward.

**Density at a specific lead.** One row per call. To get multiple
lead-24 rows for the same model in a day you have to call the API
multiple times, hoping each call hits a different freshly-published
cycle. In practice we collect at HH:45 ∈ {02, 08, 14, 20} UTC = ≤ 4
distinct `RunTimeUtc`s per model per day — and if a publisher only
issues 2 cycles/day (ECMWF IFS oper) you only see 2.

**Used by.** `collect.yml` (the 2b/2c blender's input refresh), and
critically, the WeatherProbabilistic live predict
(`predict_live_with_ci.py` reading the same parquet tree).

**The catch — and it's the catch behind the "1 Bayesian point per day at
lead 24h" question.** The Bayesian model was _trained_ off source #1
(hourly density at lead 24). At predict time we read source #2 (one
row per cycle), filter to `LeadHours ∈ {24, 48, 72}`, and inner-join
across 5 models on `(ValidTimeUtc, LeadHours)`. The intersection of
4–5 cycles/day across 5 models, restricted to ECMWF oper's 00Z + 12Z
issuance, leaves 1–2 lead-24 valid times per day. Hence one chart
point per day on the 24h-lead CI band, even though the underlying
training data is hourly. The fix is either to predict from source #1
(re-using the same endpoint as training), or to enrich source #2
(more cycles, e.g. via ECMWF scda for the 06/18Z runs); see the
`feedback_bayesian_chart_per_day` thread.

---

## 3. AWS Open Data (S3 GRIB / NetCDF), via `s3-collect.yml`

`GfsClient` (`s3://noaa-gfs-bdp-pds`),
`EcmwfClient` (`s3://ecmwf-forecasts`),
`MetOfficeUkvArchiveCollector` /
`MetOfficeGlobalArchiveCollector` (`s3://met-office-…`).

**What it returns.** One physical GRIB2 (NOAA, ECMWF) or NetCDF (Met
Office) file per `(model, cycle_date, cycle_hour, lead_hour)`. We HTTP
Range-request just the byte slice for the variable+level we want and
extract the point value via `wgrib2`. Result: one `ForecastRow` per
file with `RunTimeUtc`, `LeadHours` taken straight from the filename —
no inference, no metadata round-trip. Tagged `RunTimeSource = exact`.

**Lead axis: per-file, sparse on purpose.** We collect a hand-picked
lead set per source rather than the full forecast horizon, because each
lead is a separate HTTP fetch:

- GFS / ECMWF IFS / AIFS: `{12, 24, 36, 48, …}`
- UKV: `{9, 12, 15, 21, 24, 27}` (averaging-around-target picker for 3d
  precip, strict-time picker for 2d temp — see
  `Exact12hFeatureBuilder.UkvPickStrategy`)
- Met Office Global: `{1, 3, 6, 12, 24, 36, 48, 72, 96, 120}`

**Valid-time axis: implied by `RunTimeUtc + LeadHours`.** A given
valid time is reached only if a cycle whose `cycle + lead` lands
exactly there exists in the lead set above. There is no interpolation
across cycle hours.

**Cycle axis.** Real, publisher-defined: GFS / GEM / ICON / MétéoFrance
at 00/06/12/18Z; ECMWF IFS oper at 00/12Z and scda at 06/18Z (a recent
addition, see `project_ifs_scda_plan`); Met Office UKV at every hour
where a cycle is published, gated by `MinCycleAgeHours = 0` since
2026-05-06.

**Used by.** Phase 2d (temperature exact-runtime) and Phase 3d
(precipitation exact-runtime) blenders. The `s3-collect.yml` workflow
fires at HH:45 ∈ {02, 08, 14, 20} UTC and each call walks a 3-day
rolling window of cycles per source.

**The catch.** Only the cycles + leads explicitly enumerated in the
collectors land on disk. If a future blender wants e.g. UKV lead-3,
that's not there — would need a config change + backfill.

---

## Cheat-sheet: which source for which job

| Want to … | Use | Why |
|---|---|---|
| Train a per-lead blender at lead 24/48/72 | #1 (`previous_runs`) | Hourly density × 7 lead buckets per call |
| Refresh 2b / 2c blender inputs hourly | #2 (live) | One call per cron tick → fresh forecast curve |
| Train or predict at exact (cycle, lead) for 2d / 3d | #3 (S3) | Real cycle stamps, no synthesis or inference |
| Get the Bayesian CI hourly at lead 24h | **(currently broken — see #2 catch)** | Predict path needs to switch to #1 or absorb scda cycles |

---

## Schema invariants worth remembering

`ForecastRow` columns are identical across all three sources, so a
DuckDB query over the union just works. The differences are:

- **`RunTimeSource`** (`exact` / `reported` / `offset_day` /
  `synthesised`) — the discriminator. Filter by this when you care which
  source a row came from.
- **`LeadHours`** is always `(ValidTimeUtc − RunTimeUtc).TotalHours`,
  but for `offset_day` rows that's a bucket lower edge (training code
  treats it as exactly 24·N), for `reported`/`exact` rows it's the
  literal cycle-to-valid offset.
- **Negative `LeadHours`** can show up on `synthesised` rows
  (wall-clock-floored run time vs. valid times in the past).
  `WHERE LeadHours >= 0` is the standard analysis filter.
