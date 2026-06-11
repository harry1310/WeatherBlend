# Sennen sea-state — "can you actually climb here today?"

Status: PLANNED 2026-06-11 (not started). Owner intent: Harry, fresh session.
Origin: end-of-refactor "fun next step" discussion — picked as the #1 idea.

## Why

Sennen is a sea cliff, and rain is not what gates a Sennen day — **swell**
is: wave wash on the lower pitches, spray keeping the granite greasy well
above the waterline, wind-against-tide chop. No general weather product
answers "are the bottom pitches climbable this afternoon?". We can.

A bonus that matters beyond fun: Sennen is our truth-sparse location (EA
gauges 11–21 km inland, METAR at Culdrose). Sea-state comes with REAL
nearby observations — wave buoys — giving Sennen its first genuinely local
verification truth for anything.

## Data sources (verify all of these early in the session)

1. **Forecasts — Open-Meteo Marine API** (`marine-api.open-meteo.com/v1/marine`):
   significant wave height, swell height / period / direction, wind-wave
   components. Free tier like the weather API, same JSON shape → the
   collector is a thin `OpenMeteoClient` variant (different base URL +
   variable list). Multiple underlying wave models (MFWAM, ECMWF WAM,
   GWAM, …) → blendable inputs, same champion/challenger story as ever.
   CAUTION: the cliff coordinate (50.0786, -5.7044) may resolve to a LAND
   cell — probe for the nearest valid marine grid point first and pin it
   in config (it is a different coordinate from the site's weather point;
   keep both explicitly).

2. **Training truth — ERA5 ocean waves**: ERA5 carries significant wave
   height (+ mean period / direction) — a natural extension of
   `Era5Client`, mirroring the established pattern exactly: ERA5 as
   gapless training truth, real obs as verification truth.

3. **Verification truth — wave buoys**: Channel Coastal Observatory
   (Penzance / Porthleven Datawell buoys, free archive + realtime) and/or
   CEFAS WaveNet. Confirm terms + exact endpoints in-session; pick the
   buoy with the cleanest record and note its distance/exposure vs Sennen
   (a south-coast buoy sees different swell windows than Land's End — the
   representativeness caveat goes on the site like the EA-gauge one).

4. **Tides (Phase 4, optional)**: Newlyn is ~8 km away and is the UK's
   reference tide station. Harmonic prediction from published Newlyn
   constituents is closed-form (no API); NTSLF publishes observed data if
   verification is ever wanted. Skip entirely until the swell product works.

## Phases

- **Phase 0 — collect + backfill (do FIRST, it gates everything):**
  marine forecast collector for the pinned sea point (wired into
  collect.yml's cycle); ERA5 wave-variable backfill (Era5Client variable
  list + `data/truth/era5` schema extension — additive columns,
  union_by_name tolerates); buoy archive backfill into a new
  `data/truth/waves/` tree. The project's repeated lesson: start the
  archive accumulating before deciding how fancy the model is.

- **Phase 1 — baseline bake-off:** raw marine models vs buoy truth at
  Sennen (MAE on Hs, direction-binned). Which wave model is locally best?
  Also: how well does ERA5 SWH track the buoy (the truth-of-truths check
  compare_ea_chard.py-style)?

- **Phase 2 — wave blender:** target = significant wave height (hourly,
  leads 24/48/72 to start). Reuse the wind_speed_lgb recipe wholesale:
  quantile-LGB q05/q50/q95 + cross-conformal CQR → point + calibrated 90%
  band (swell decisions want bands more than points). Likely python
  (WeatherProbabilistic) given the recipe lives there. Swell period +
  direction as features first; as separate targets only if Phase 3 needs
  them forecast rather than passed through.

- **Phase 3 — the site product:** a "Sea state" section on Sennen's
  overview + its own tab: Hs + band chart, swell period/direction,
  and the headline chip — "lower pitches: washed / spray zone / clean" —
  thresholded on (swell height × direction window × period). THE
  THRESHOLDS ARE HARRY'S DOMAIN KNOWLEDGE (which swell direction wraps
  into the cove, what height washes the first belay) — start crude,
  calibrate against real sessions, exactly like the rock-greasiness tiers.

- **Phase 4 (later) —** tide curve under the chip (harmonics), wind-
  against-tide flag, spray-greasiness coupling into a Sennen rock-temp
  story, verify page (rolling Hs MAE vs buoy + band coverage).

## Wiring checklist (the honest touch-list)

New target (suggest `wave_height`) in phases.yaml WITH its `display:`
block; sync_train_data case (both repos if the trainer is python);
retrain step (retrain-python.yml if python); predict workflow or fold
into the existing Hop C chain; PhaseWiringConsistencyTests will enforce
the rest. New truth tree + collector need: collect.yml step, R2 push
includes, sync-render-inputs entry IF render reads it (Phase 3), and a
windowed predict-mode pull from day one (don't re-create the over-pull).

## Open questions for the session

1. Marine API: archive depth for training (historical marine API vs
   ERA5-only training)? Free-tier rate limits with our cadence?
2. Buoy choice + licence; realtime latency (verification lag setting).
3. Is Hs at a buoy 20+ km away honest enough truth for Sennen's cliff
   base, or is ERA5's wave cell actually closer in spirit? (Probably:
   train ERA5, verify buoy, display the caveat — but check correlation.)
4. One blender or per-component (Hs / period / direction)? Start Hs-only.
5. Where does the product live for non-climbers — does Membury/Bonehill
   ever want it? (No. Keep it Sennen-scoped via phases.yaml locations.)

## Effort sketch

Phase 0 ≈ an evening (collector variants + backfills are well-trodden).
Phase 1 ≈ an evening once data lands. Phase 2 ≈ 2–3 evenings (recipe
reuse). Phase 3 ≈ 1–2 evenings + iteration on thresholds. Start Phase 0
soon even if the rest waits — archive depth is the only thing that can't
be backfilled later... except where historical APIs exist; verify that
before rushing.
