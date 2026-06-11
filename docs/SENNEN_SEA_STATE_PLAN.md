# Sennen sea-state — "can you actually climb here today?"

Status: PHASE 0 IN PROGRESS 2026-06-11 (evening). Owner: Harry.
Origin: end-of-refactor "fun next step" discussion — picked as the #1 idea.
Reframed 2026-06-11 (Harry): this is NOT a wave-height product with a chip
on top; it is a **"will water be on the rock face?"** product. See §Product.

## Why

Sennen is a sea cliff, and rain is not what gates a Sennen day — **water on
the rock** is: wave wash on the lower pitches, spray keeping the granite
greasy well above the waterline, wind-driven chop. No general weather
product answers "are the bottom pitches climbable this afternoon?". We can.

A bonus that matters beyond fun: Sennen is our truth-sparse location (EA
gauges 11–21 km inland, METAR at Culdrose). Sea-state comes with REAL
nearby observations — wave buoys — giving Sennen its first genuinely local
verification truth for anything.

## Product (the reframe)

The headline variable is not Hs. It is something like **wetting elevation**:

    wetting elevation ≈ tide height
                      + swell run-up   (≈ k·√Hs·T, direction-windowed)
                      + wind-spray increment (onshore wind component)

compared against where the routes actually start. Three interacting inputs:

1. **Waves — Hs AND period AND direction.** Period is not merely a feature
   for predicting Hs; it is half the answer. 1.5 m at 16 s runs far higher
   up the rock than 1.5 m at 7 s (run-up scales ~√(Hs·L₀), L₀ ∝ T²). So the
   forecast must OUTPUT period + direction at valid time, not just consume
   them. Cheap version: blend Hs properly; pass period/direction through
   from the locally-best raw model (Phase 1 decides which); blend them as
   separate targets only if pass-through proves noisy.
2. **Tide — CORE, not optional.** The same swell at low neaps vs high
   springs is the difference between climbable and washed; the chip is
   barely meaningful without it. It is also the cheapest input: closed-form
   (Newlyn harmonics ~8 km away) AND Open-Meteo's marine API serves
   `sea_level_height_msl` directly (verified 2026-06-11, hindcast to
   ≥mid-2023) — collect it from day one, cross-check against Newlyn
   harmonics in-session, keep whichever is honest.
3. **Wind onto the face.** Onshore wind drives spray and chop that keep the
   granite greasy well above the swell line. Inputs are already solved
   (wind speed/direction forecasts; Sennen weather collection live since
   2026-06-05) — the new work is only the combination: an
   onshore-component term (speed × how squarely it blows into the cove).

Rain and rock-surface-temp chips bolt on from the existing systems.

**The combiner is NOT ML.** There is no truth data for "water on the face";
it has to be a physics-flavoured heuristic (run-up formula + tide datum +
spray term) with tiers thresholded on Harry's domain knowledge (which swell
direction wraps into the cove, what height washes the first belay) and
calibrated against real sessions — exactly like the rock-greasiness tiers.
Architecture splits cleanly: ML/blending for the stochastic inputs (waves;
wind already done), closed-form for tide, heuristic combiner on top.

## Data sources (VERIFIED by live probes 2026-06-11)

1. **Forecasts — Open-Meteo Marine API** (`marine-api.open-meteo.com/v1/marine`):
   - The cliff coordinate (50.0786, -5.7044) resolves to a VALID sea cell
     at **(50.0417, -5.7083)** — ~4 km SSW of the cliff, elevation 0.0,
     real data. Pinned explicitly in config as the location's `marine:`
     point (separate from the site's weather point; keep both).
   - Per-model wave components (height/period/direction × total/wind-wave/
     swell): model ids `meteofrance_wave` (MFWAM), `ecmwf_wam025`, `gwam`,
     `ewam`, `ncep_gfswave025` all valid. `best_match` additionally serves
     secondary swell, `sea_surface_temperature`, and `sea_level_height_msl`
     (these are "undefined" on per-model requests).
   - **`_previous_dayN` columns work on the live endpoint per model**
     (verified meteofrance_wave + ecmwf_wam025) — proper per-lead labelled
     rows, same offset_day convention as the weather pipeline. BUT they are
     NULL in the hindcast archive and there is no marine previous-runs API
     (404) — per-lead history only accrues from the day collection starts.
     Hence Phase 0 urgency. CAVEAT (first live cycle, 2026-06-11): offsets
     exist ONLY for the total-wave triple (wave_height/period/direction) —
     swell + wind-wave components have no previous_day variants on any
     model. So per-lead training features are the total triple; swell
     decomposition is available lead-unlabelled (live rows + hindcast)
     only. Fine for an Hs target; revisit if a swell-component blend is
     ever wanted. Also: ecmwf_wam025 serves ONLY the total triple even
     live (no swell split); ewam's horizon is ~4 days.
   - Hindcast (best-available, lead-unlabelled, RunTimeSource=hist_forecast
     analogue): best_match/MFWAM back to ≥2022-01; ewam + meteofrance_wave
     ≥2023-01; gwam + ecmwf_wam025 by 2024-06 (null 2023-01); gfswave null
     even 2024-06 (live-only). sea_level_height_msl null 2022-06, present
     2023-06.
2. **Training truth — `era5_ocean` via the SAME marine API** (`models=era5_ocean`):
   wave_height + wave_period + wave_direction verified non-null back to
   2020 (gapless ERA5 wave reanalysis, 0.5° cell at 50.0, -5.5). Swell
   decomposition is null — truth is the total-wave triple, which matches
   training Hs (+period/dir) targets. NOTE: the weather `archive-api`
   ERA5 endpoint snaps to LAND cells near the coast (all-null waves even
   with `cell_selection=sea` giving a cell 5 km off) — use the marine API
   era5_ocean route, NOT an Era5Client variable extension (plan revised
   from the original "extend Era5Client" idea after probing).
3. **Verification truth — wave buoys** (researched + endpoints verified
   2026-06-11; wired into config as `marine.buoys`):
   - **PRIMARY: Sevenstones Light Vessel** (WaveNet id 4/EXT, platform
     6200107) — 28 km due W, 69 m water, the only fully Atlantic-exposed
     sensor in range, directly up-swell of the cliff: effectively the
     deep-water incident wave field. Hs/Tz/SST only (no direction/Tp),
     hourly. (Snapshot that decided it: Sevenstones Hs 1.9 m while
     Penzance read 0.64 m in the same hour.)
   - **SECONDARY (directional): SW Isles of Scilly WaveNet**
     (SWSCILLYWN/INT, 6201054) — 67 km SW, 96 m, full directional suite
     (Hs/Tp/Tz/peak-dir/spread), Cefas-owned (genuinely open feed).
     Lag-match the 2–4 h swell-arrival offset when verifying.
   - **CROSS-CHECK: Penzance Waverider** (200/EXT, 6201000) — 15 km E but
     inside south-facing Mount's Bay, BLIND to W/NW groundswell; kept for
     southerly swell + windsea only. The representativeness caveat goes on
     the site like the EA-gauge one.
   - Feeds: realtime = Cefas WaveNet JSON API (open, no auth, non-QC'd,
     ≤10–90 min latency; collected every cycle). Archive = EMODnet Physics
     ERDDAP (CC-BY-SA, QC'd, floor 2018-01-01, mirror lags days–weeks;
     `backfill --source buoys`). The merge-dedup writer lets archive
     re-pulls upgrade non-QC realtime rows in place. NDBC has NO
     Sevenstones historical archive (verified 404s) — don't plan one.
     Rejected: Porthleven (Mount's Bay, same blindness as Penzance),
     St Mary's Sound (Scilly-sheltered), Perranporth (10 m inshore,
     shoaling-modified), Looe Bay / E1 (wrong wave climate). CCO's keyed
     API (free registration) is a future option for deeper Penzance
     directional history (2007+) if Phase 1 wants it.
4. **Tides**: `sea_level_height_msl` collected from day one (see #1).
   Newlyn harmonic constituents as the closed-form cross-check / fallback;
   NTSLF observed data if verification is ever wanted.

## Phases

- **Phase 0 — collect + backfill (IN PROGRESS, gates everything):**
  - `data/marine/` forecast tree: live per-model pulls + per-model
    offset-day (previous_day1..7) pulls each collect cycle, mirroring the
    weather forecasts tree layout (`location=/model=/date=/...`).
  - Marine hindcast backfill per model 2022-01→present (monthly chunks;
    all-null chunks = 0 rows, expected for late-start models).
  - `data/truth/waves/` tree: era5_ocean backfill 2021-01→present, daily
    refresh; buoy archive once the source decision lands.
  - The project's repeated lesson: start the archive accumulating before
    deciding how fancy the model is — and the per-lead labelled rows
    CANNOT be backfilled at all (see #1), so every day collect isn't
    running is per-lead training data lost forever.

- **Phase 1 — baseline bake-off:** raw marine models vs buoy truth at
  Sennen (MAE on Hs, direction-binned), and ERA5-ocean vs buoy (the
  truth-of-truths check, compare_ea_chard.py-style). Which wave model is
  locally best? Is the buoy honest truth for the cliff base?

- **Phase 2 — wave blender:** target = significant wave height (hourly,
  leads 24/48/72 to start). Reuse the wind_speed_lgb recipe wholesale:
  quantile-LGB q05/q50/q95 + cross-conformal CQR → point + calibrated 90%
  band (swell decisions want bands more than points). Likely python
  (WeatherProbabilistic) given the recipe lives there. Period + direction:
  pass through from the Phase-1-winning raw model as forecast OUTPUTS
  (needed by the Phase 3 combiner), with swell components as blender
  features; promote to blended targets only if pass-through is noisy.
  Until enough offset_day rows accrue, initial training can lean on the
  lead-unlabelled hindcast (same caveat as the historical-forecast API:
  best-available per valid-time) — revisit once real per-lead rows exist.

- **Phase 3 — the site product:** a "Sea state" section on Sennen's
  overview + its own tab: Hs + band chart, swell period/direction, tide
  curve, and the headline chip — "lower pitches: washed / spray zone /
  clean" — from the wetting-elevation combiner (tide + √Hs·T run-up term ×
  direction window + onshore-wind spray term). THE THRESHOLDS ARE HARRY'S
  DOMAIN KNOWLEDGE — start crude, calibrate against real sessions, exactly
  like the rock-greasiness tiers.

- **Phase 4 (later) —** wind-against-tide chop flag, spray-greasiness
  coupling into a Sennen rock-temp story, verify page (rolling Hs MAE vs
  buoy + band coverage), Newlyn harmonics if sea_level_height_msl proves
  dishonest.

## Wiring checklist (the honest touch-list)

Phase 0 (this session): MarineClient + BuoyClient + MarineForecastRow +
WaveTruthRow; `marine:` block on sennen_cove in config.yaml (pinned sea
point + wave model list + buoys) + `variables.marine` /
`variables.marineSite`; ParquetWriter methods (marine live / previous-runs
/ hindcast + wave truth merge-dedup); collect cycle marine + buoy step
(non-fatal, Met-Office-style — nothing downstream consumes it yet);
backfill sources `era5-waves` (in "all") + `marine` & `buoys` (NOT in
"all" — one-off gap-fills like hist-forecast); era5-waves into
era5-refresh.yml's daily window; R2 push is the existing whole-tree
`rclone copy ./data` (additive); collect.yml pull step gains the waves
tree (buoy realtime is read-modify-write; the marine FORECAST tree needs
no pull — its writers regenerate complete files from the API each cycle).

Phase 2+: new target (suggest `wave_height`) in phases.yaml WITH its
`display:` block; sync_train_data case (both repos if the trainer is
python); retrain step (retrain-python.yml if python); predict workflow or
fold into the existing Hop C chain; PhaseWiringConsistencyTests will
enforce the rest; sync-render-inputs entry when render starts reading
waves (Phase 3); windowed predict-mode pull from day one (don't re-create
the over-pull).

## Open questions

1. ~~Buoy choice + licence~~ RESOLVED 2026-06-11 — see Data sources #3.
2. Is a buoy 20+ km away honest enough truth for Sennen's cliff base, or
   is ERA5-ocean's wave cell closer in spirit? (Probably: train ERA5,
   verify buoy, display the caveat — but check correlation in Phase 1.)
3. ~~sea_level_height_msl vs Newlyn~~ RESOLVED 2026-06-11
   (scripts/tide_sanity_check.py, 28 days vs the EA Newlyn gauge E72239):
   correlation 0.9989, RMSE 8.6 cm after removing the constant offset,
   amplitude ratio 1.049, daily high-water error 8 cm. Phase: OM runs
   ~30 min EARLIER than Newlyn — plausibly real geography (the tide
   reaches Land's End before Mount's Bay; our point is the Sennen cell,
   the gauge is Newlyn), not an error. VERDICT: the OM tide curve is
   good enough to drive the chip — no harmonics module needed.
   ONE FLAG: the mean datum offset is −0.69 m (OM below Newlyn mAOD),
   bigger than the ~−0.2 m that MSL-vs-OD bookkeeping predicts. It is
   CONSTANT over the window, and the chip's thresholds are calibrated
   empirically against real sessions anyway, so it's absorbed — but pin
   down its origin (likely the EA feed's local datum vs true mAOD)
   before quoting absolute heights anywhere.
4. Free-tier rate limits at our cadence: collect adds ~12 marine calls per
   cycle (6 sources × live + offset) — trivial vs the weather pull. The
   backfill (~390 calls) runs once, throttled like previous-runs.
5. Run-up constant k and the direction window: Harry's calibration, Phase 3.

## Effort sketch

Phase 0 ≈ this evening (collector variants + backfills are well-trodden).
Phase 1 ≈ an evening once buoy data lands. Phase 2 ≈ 2–3 evenings (recipe
reuse). Phase 3 ≈ 1–2 evenings + threshold iteration. Tide module (if
Newlyn harmonics needed) ≈ an evening, independent of everything else.
