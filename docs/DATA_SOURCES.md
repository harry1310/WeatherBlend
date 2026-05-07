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
cycle. In practice we collect at HH:30 ∈ {02, 08, 14, 20} UTC = ≤ 4
distinct `RunTimeUtc`s per model per day. Most Open-Meteo models that
back our blender expose 4 cycles/day (00/06/12/18Z) — `ecmwf_ifs025`
even includes scda upstream so it gives 4, not 2. **The exception is
`gem_seamless`, which only publishes 2 cycles/day (00Z + 12Z).** This
matters more than it looks (see the catch below).

**Used by.** `collect.yml` (the 2b/2c blender's input refresh), and
critically, the WeatherProbabilistic live predict
(`predict_live_with_ci.py` reading the same parquet tree).

**The catch — the actual story behind "1 Bayesian point per day at
lead 24h" (corrected 2026-05-07 after empirical trace).** The Bayesian
model was _trained_ off source #1 (hourly density at lead 24). At
predict time we read source #2 (one row per cycle), filter to
`LeadHours ∈ {24, 48, 72}`, and inner-join across 5 models on
`(ValidTimeUtc, LeadHours)`. **GEM Seamless's 2 cycles/day caps the
intersection at 2 lead-24 valid_times per day** (00Z and 12Z) — every
06Z/18Z lead-24 valid time gets dropped because GEM has no 06Z/18Z
cycle to match. Real-world data gaps thin that further to ~1/day in
recent anchors (the live ECMWF collector occasionally misses the 12Z
cycle when 12Z hasn't published yet at the 14:30 UTC collect tick, so
the 5-way intersection on that valid_time fails too).

Two real fixes:
1. **Drop GEM** from the Bayesian model and retrain on the remaining 4
   sources. Lead-24 ceiling rises to 4/day. Likely small skill cost.
2. **Lead-as-feature retrain** (Phase 3b): re-train so the posterior
   takes lead as a continuous covariate. Then any forecast row at any
   hour can be scored against the single posterior — no more (cycle,
   lead) intersection bottleneck. Ceiling = 24/day from any model with
   hourly forecast emission.

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
(cycle, lead) set per source rather than the full forecast horizon,
because each lead is a separate HTTP fetch. The current pulls track
what the 2d/3d exact-runtime blenders consume — extending coverage is
a constant change + backfill, not a re-architecture.

**Valid-time axis: implied by `RunTimeUtc + LeadHours`.** A given
valid time is reached only if a cycle whose `cycle + lead` lands
exactly there exists in the (cycle, lead) set we collect. There is no
interpolation across cycle hours.

### Per-source detail (publisher offer vs what we currently pull)

**GFS — NOAA `noaa-gfs-bdp-pds`**
- Publisher: 4 cycles/day (00 / 06 / 12 / 18Z). Hourly to T+120, 3-hourly
  to T+240, 6-hourly to T+384.
- We collect: cycles `{0, 6, 12, 18}` × leads
  `{1, 3, 6, 12, 24, 36, 48, 72, 96, 120}` (`GfsBackfillCommand`).
  Capped at 120h by design — the inline comment notes "beyond f120
  single-model NWP skill drops off enough that we don't care for the
  blender".

**ECMWF IFS — `ecmwf-forecasts` (two streams)**
- Publisher:
  - `oper` stream: cycles **00Z** and **12Z**, 3h step to T+144, 6h step
    to T+240.
  - `scda` stream: cycles **06Z** and **18Z**, same lead structure as
    `oper`, published as a separate AWS path. **AWS scda coverage
    empirically starts 2024-02-28** — backfill chunks covering
    2023-01-18 → 2024-02-27 return 404s on every probe. Backfill in
    flight 2026-05-07 to populate the scda gap from 2024-02-28 onwards.
- We collect: cycles `{0, 6, 12, 18}` × leads
  `{6, 12, 24, 36, 48, 72, 96, 120}` (`EcmwfBackfillCommand`). Same 120h
  cap as GFS. Lands under `model=ecmwf_ifs_oper` for both streams (one
  partition per `(date, cycle)`); `model=ecmwf_ifs025` is the separate
  Open-Meteo path (#1 / #2 above), not the raw S3 archive.

**ECMWF AIFS — `ecmwf-forecasts`**
- Publisher: 4 cycles (00 / 06 / 12 / 18Z), 6h step to T+360.
- We collect: cycles `{0, 6, 12, 18}` × leads
  `{6, 12, 24, 36, 48, 72, 96, 120}`. Same path machinery as IFS,
  separate `model=ecmwf_aifs_oper` partition.

**Met Office Global / UM Global — Met Office AWS**
- Publisher: 4 cycles (00 / 06 / 12 / 18Z), forecasts to T+168 (some
  variables shorter).
- We collect: cycles `{0, 6, 12, 18}` × leads
  `{1, 3, 6, 12, 24, 36, 48, 72, 96, 120}`
  (`MetOfficeGlobalArchiveCollector`). Capped at 120h.

**Met Office UKV — Met Office AWS**
- Publisher: 24 cycles/day (every hour). Forecast length varies sharply
  per cycle:
  - **03Z and 15Z cycles → leads 0–120h hourly** (full long-range).
  - **The other 22 cycles → cap at ~T+54h.** Empirically re-verified
    2026-05-05; the python collector's `--cycles` default is `3,15`
    precisely for this reason.
- We collect: cycles `{0, 3, 6, 12, 15, 18}` × leads
  `{9, 12, 15, 21, 24, 27}` (`MetOfficeUkvArchiveCollector`). This is a
  deliberate narrowing to the 2d/3d exact-runtime pickers' needs — see
  `Exact12hFeatureBuilder.UkvPickStrategy` after the 2026-05-06 bake-off
  (temp Strict on `{0,6,12,18}` × `{12,24}`, precip Averaging on
  `{3,15}` × `{9,15,21,27}`).
  - **UKV's 0–120h availability from 03Z/15Z is NOT currently pulled.**
    Extending the lead set to e.g. `{1, 3, 6, 12, 24, 36, 48, 72, 96, 120}`
    while keeping cycles `{3, 15}` would put UKV on the same lead
    horizon as the other long-range S3 sources. One constant change in
    `MetOfficeUkvArchiveCollector` + a chunked backfill against the
    python script's existing long-range defaults.

**Used by.** Phase 2d (temperature exact-runtime) and Phase 3d
(precipitation exact-runtime) blenders. The `s3-collect.yml` workflow
fires at HH:45 ∈ {02, 08, 14, 20} UTC and each call walks a 3-day
rolling window of cycles per source.

**The catch.** Only the (cycle, lead) tuples explicitly enumerated in
the collectors land on disk. If a future blender wants e.g. UKV at
lead 48, or any source at T+144+, that's a config + backfill change,
not a download-on-demand fallback.

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
