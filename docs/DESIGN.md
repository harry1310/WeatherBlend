# Design notes

Running commentary on why things are the way they are. Useful when you come back
to this in three months and wonder what past-you was thinking.

## Why Parquet + DuckDB and not a database?

For a single-location PoC, yearly data volume is a few hundred MB. A database adds
ops burden (backups, migrations, a process to keep running) with no query speed
benefit at this size. Parquet partitioned by model/date, queried via DuckDB, is
faster to set up, trivial to back up (copy a folder), and trivial to move to cloud
storage later if the PoC expands to many locations.

If this ever grows to hundreds of locations with gridded data, revisit. Options
at that point: DuckDB over S3-backed Parquet (still no database, just bigger
storage), or ClickHouse. Still not Postgres.

## Why Open-Meteo first and not direct-to-source?

Direct GRIB parsing from ECMWF / NOAA is a rabbit hole of eccodes, cfgrib, and
wgrib2 that would eat a week before any ML. Open-Meteo gives us six models through
one consistent JSON API including a historical archive - exactly what's needed
to get moving. The tradeoffs:

- We get their interpolation to our lat/lon, not the raw grid. For a single point
  this doesn't matter much.
- We don't see raw ensemble members, only the deterministic run per model.
  This matters a lot for precip (see below).
- We have to trust they don't silently rewrite history in the archive.

Phase 2+ plan: go direct for ECMWF ENS (51 members, the real prize) via their
open data at data.ecmwf.int, and for HRRR/GFS via AWS Open Data S3 buckets.

## Why LightGBM for the blender?

- Handles missing values natively (some models will fail to return some variables).
- Handles heterogeneous features without scaling.
- Monotonic constraints available (e.g. force "forecast temp up -> predicted temp up").
- Quantile regression mode for prediction intervals and CRPS approximation.
- Microsoft.ML has a first-class LightGBM trainer so no Python needed.

Alternatives considered:

- **Linear/ridge regression**: often surprisingly close on temperature. Worth
  including as a baseline. Less good on precip.
- **Deep nets**: overkill for single-location tabular data with ~10k-50k rows.
- **ngboost / EMOS / BMA**: better for probabilistic precip but thin tooling in
  .NET. Plan to shell out to Python for this specific step in phase 5.

## Champion/challenger for feature-set experiments (phase 2c)

Feature-engineering ablations are hard to judge off a single training-time test
MAE — the real question is "does the new feature set generalise better in
production?", which can only be answered by running both models in parallel on
live forecasts and comparing against ERA5 over several weeks.

The manifest encodes this directly. `MANIFEST.Active` is a list of version dir
names (not a single pointer like `Current`), and `predict --model-version
current` iterates that list, writing one parquet partition per version under
`data/predictions/temperature/model_version=<v>/date=<d>/`. Verify reads the
whole tree, stratifies by `(version, lead)`, and — when two active versions have
different `Phase` tags (e.g. `2b` lean vs `2c` rich) — renders a direct MAE delta
section in the weekly report. The winner is whichever version has the smaller
rolling blend MAE at each lead; promotion is then a one-line manifest edit
(`SetActive` with the chosen version).

Two consequences to be aware of:

- Storage is linear in `|Active|` — each version materialises its own predictions
  parquet. At ~6 rows/day per version this is fine at PoC scale.
- `Current` is frozen while a challenger is running. Dropping a challenger from
  `Active` doesn't delete its predictions — they stay on disk as part of the
  historical record, just no new ones are written.

## Why per-lead-time models and not one big model?

Forecast skill characteristics change dramatically with lead time:

- At 1h: persistence dominates, models all agree, spread is tiny.
- At 24-72h: model disagreement is meaningful signal; blend adds most value here.
- At 144h+: skill collapses, everyone's bad, blender mostly just averages.

A single model trying to cover all of these either underfits the nuance at each
horizon or overfits the easy ones. Separate per-bucket models each see a
more homogeneous problem. Buckets: [1-3h], [6-12h], [24-36h], [48-72h],
[96-120h], [144-168h].

## Precipitation architecture (phase 3)

Don't train a single regression on precip (mm/h). You'll get near-zero predictions
most of the time that score well on MAE and are useless.

Two-stage:

1. **Occurrence classifier (phase 3a, shipped)**: P(precip ≥ 0.1mm) in hour H, per
   lead time. Target: binary from EA Hydrology gauge (15-min tips summed to hourly;
   hours with fewer than 4 readings are dropped to avoid boundary flips). **One
   blender per truth station** because Dartmoor stations differ enough in micro-
   climate that a single model would underfit both; slugs are `ea_bellever_dartmoor`,
   `ea_princetown` (the `ea_` prefix reserves space for a future Met Office Princetown).
   Metrics: Brier, BSS vs climatology, frequency bias @0.5, reliability diagram,
   persistence + best-single + mean-of-models baselines. Climatology baseline is
   P(wet) per (month, hour-of-day) from training rows, persisted as
   `climatology.json` alongside each version so predict/verify never need the
   training tree again.

2. **Intensity regression (originally phase 3b)**: E[precip | precip > threshold],
   conditional on the classifier firing. Train only on hours with observed precip.
   Replaced by the Phase 3b dry-window classifier after the user pivoted to "is
   there time to walk the dog dry?" — one binary classifier per (station, window)
   beats calibrating a conditional intensity regressor for that decision.

3. **Rich-feature occurrence blender (phase 3c, shipped)**: same LightGBM, same
   hyperparameters, same chronological split as 3a — the isolated variable is
   feature richness. 27 lean features + 28 extras: 18 per-model humidity
   (dewpoint, RH, T-Td depression), 6 per-model surface pressure, 4 EA-observation
   trailing-rainfall persistence features anchored at `run_time = valid - leadHours`
   (prev-24h mm, prev-72h mm, wet-hours-last-24h, trailing-dry-hours). 55 features
   total. Saved as a challenger alongside 3a via per-station `StationEntry.Active`
   so both artefacts produce predictions every cycle and the rolling verify path
   can score them side by side. Forecast-time trailing-precip persistence (H-1..-3
   of the same run) and pressure tendency were explicitly dropped: Phase 1's
   training parquet stores only leads {24, 48, 72} per `offset_day` run, so the
   intermediate-lead cells those features need don't exist in training data even
   though live cycles have them. Switching training to live cycles would break
   the "same split as 3a" guarantee and is out of scope for 3c. The tier ablation
   in `precip-ablate` shows the rich features are modestly helpful at best —
   each tier contributes at most ~0.002 Brier points, and the EA-persistence
   tier actively hurts at 2 of 3 stations.

Original phase-3 combine recipe
(`expected_precip = P(precip) * E[precip | precip > 0]`) is deferred indefinitely —
the dry-window framing is what the user actually consumes.

For probabilistic output (the real product), train quantile regressions at
e.g. q10/q25/q50/q75/q90 of the precip distribution. CRPS is the metric that
matters; pinball loss is what you minimise per-quantile.

## Ground truth hierarchy for precip

Best -> worst for training a precip model:

1. **Radar composite** (Met Office Nimrod, MRMS US, OPERA EU) - 1km gridded,
   captures spatial structure. Best for validating convective events.
2. **Quality-controlled gauge network** (Met Office MIDAS) - point truth,
   reliable intensity, gaps in space.
3. **Personal weather station network** (Netatmo, Weather Underground) - dense
   but noisy, needs QC.
4. **METAR present-weather codes** - occurrence only, no intensity, coarse
   timing. What we have in phase 1.

Decision: ship phase 1 with METAR, tag phase 3 as "source quantitative precip
truth" before touching the precip model. Don't train precip on phase 1 truth.

Phase 3a shipped with **EA Hydrology** (free, OGL v3, 15-min resolution, 1998+
at Bellever and Princetown) rather than Met Office DataHub — quality is close
enough for Dartmoor-scale verification and avoids DataHub's rate limits and
paid tier. Nimrod radar is still the right next step for convective events.

## Verification: the bit that matters most

The easiest mistake in this whole project is declaring victory on bad metrics.
Pre-commit to:

- **Time-based splits, never random.** Random shuffling leaks future into past.
- **Walk-forward validation** for final numbers (train to month N, test N+1,
  refit to N+1, test N+2, ...).
- **Proper baselines every time**: persistence, climatology, mean-of-models,
  best single model. The blend has to beat all four or it's not earning its keep.
- **Report per lead time, not aggregated.** Aggregated numbers hide where the
  blend helps and where it doesn't.
- **For precip: Brier + reliability diagrams alongside MAE.** An underconfident
  well-calibrated forecast is more useful than a sharp miscalibrated one.

## Things explicitly out of scope for now

- Multi-location generalisation
- Gridded forecasts
- User-facing UI or API
- Real-time nowcasting from radar
- Commercial licensing (stay on non-commercial tiers)
