# Sennen rock surface temperature — sea-cliff extension plan

Status: **S1 + S2 implemented 2026-06-12 (local); S3/S4 cliff physics next; Sennen
output stays OFF the site (`enabled: false`) until S5 flips it.**
Drafted 2026-06-12. Companion to docs/ROCK_SURFACE_TEMP_PLAN.md (the base
Force-Restore module, live for Bonehill since 2026-06-05) and to
docs/SENNEN_SEA_STATE_PLAN.md (which owns the salt-spray / wave-wetting side).

Scope decisions (Harry, 2026-06-12):

- **Target rock = the main W/NW-facing vertical cliff faces**, not the clifftop
  boulders. This forces the cliff-aware physics — the base model's
  horizontal-surface assumptions are wrong for a vertical wall.
- **Cliff-aware before site display.** No interim Bonehill-physics output for
  Sennen; the first thing shown on the site should be trustworthy. Hence the
  per-location `enabled` gate (S2).
- **Calibration = IR-gun spot checks on visits** (same as the Bonehill
  2026-06-07 field validation). A contact logger is a possible later upgrade.
- **Salt spray and wave wetting are out of scope here** — the sea-state
  wetting-elevation product owns them. A combined "rock wet" badge that merges
  condensation + spray is a natural later step once both exist.

## 1. What transfers from Bonehill unchanged

- **The pipeline and rendering are already location-generic.** The predict step
  loops every configured location and `RockSurfacePredictPipeline` skips any
  location missing one of its four blended inputs (temp / wind /
  shortwave_radiation / cloud_cover champions). Site render filters the
  predictions tree by location. No structural work.
- **Dew point** comes straight from the NWP mean (`DewPoint2m`, 0% null) — no
  new blender needed, works at Sennen today.
- **The longwave cloud calibration** (`lwCloudK = 0.54`, GFS-DLWRF-fitted
  2026-06-04) is not site-specific — carry it over.
- **Granite material constants** (ρ, c, λ, ε, α) — same granite; only the
  geometry/exposure knobs differ.

## 2. What the sea cliff breaks

Three physical differences, in increasing order of implementation cost:

1. **Maritime air → permanently tight condensation margin.** Coastal dew point
   sits much closer to air temperature than on Dartmoor, so with Bonehill's
   `greasyMarginC = 3.0` the amber "potentially greasy" flag could be lit
   almost all the time. Either that reflects reality (sea cliffs do sweat) or
   the threshold needs a Sennen-specific value — decide from data + the gun
   readings in S5. Needs per-location parameters either way (S2).
2. **A vertical face sees half sky, half sea.** Night-time condensation is
   driven by radiating to the cold sky. A vertical wall's view hemisphere is
   ~50% sky; the other ~50% is the sea, a warm radiator (SST ~10–17°C) that
   returns longwave the open sky does not. Net effect: much weaker radiative
   cooling than the Bonehill model would predict. Crude v1 = `fSky ≈ 0.5`
   (the existing knob, which treats the non-sky fraction as exchange-neutral);
   honest version = an explicit sea longwave exchange term (S4).
3. **Sunshine on a vertical W/NW face is a different calculation.** The model
   consumes `shortwave_radiation`, which is flux on a HORIZONTAL surface. A
   W/NW wall gets almost no direct sun until afternoon, then near-perpendicular
   sun in the evening — the diurnal heating curve has a different shape and
   phase entirely. Needs a solar-position projection of the direct beam onto
   the face plane (S3).

## 3. Phases

### S1 — enable the missing element blenders (DONE 2026-06-12, config-only)

Added `sennen_cove` to the `locations:` lists for `shortwave_radiation`,
`cloud_cover`, and `wind_gust` in `Config/phases.yaml`. The `wind` champion was
already queued for Sennen (2026-06-11, sea-state). All four are the same
location-generic ERA5-truth element harness that worked for Sennen wind;
Sennen's ERA5 + previous-runs archives go back to early/mid-2024, comparable to
what the Bonehill elements trained on. First training = the next Sunday retrain
(2026-06-14). Until then rock-surface predict keeps soft-skipping Sennen on the
missing-champion gate, exactly as for Membury.

### S2 — per-location physics parameters + enable gate (DONE 2026-06-12)

The `rockSurface:` block in config.yaml was one global block shared by every
location. Added `LocationConfig.RockSurface` (`RockSurfaceOverrideConfig`, all
fields optional) merged over the global defaults at predict time
(`RockSurfaceConfig.ResolveFor`). Bonehill keeps the global (calibrated) values
untouched; Sennen's override sets draft sea-cliff values + `enabled: false` so
nothing emits (and therefore nothing renders) until the cliff physics lands.
`enabled: false` exits 0 — an intentional skip, not the missing-input soft-skip.

### S3 — cliff-face shortwave (the main new physics)

Project the direct beam onto the face plane:

- **Solar position** (elevation + azimuth per hour) from standard formulas —
  lat/long/time only, no new data.
- **Face geometry in config:** `faceAspectDeg` + `faceSlopeDeg` per location
  (override block; 90° slope = vertical). OPEN: need the aspect/steepness of
  the faces Harry cares about (e.g. "the main wall faces ~290°").
- **Direct/diffuse split:** keep blending total shortwave (the trained element)
  and split it by the NWP-mean direct fraction — `direct_radiation` /
  `diffuse_radiation` are already collected for Sennen. Avoids training two
  more blenders for a ratio NWPs agree on reasonably well.
- **Face-incident SW** = direct·max(0, cos θ_face)/sin(solar elev) projection
  + diffuse·Fsky. Sea-reflected shortwave is negligible (ocean albedo ~0.06
  at high sun) — note and drop.
- **Apply at forcing-assembly level** so the spin-up window (NWP-sourced SW)
  gets the same projection as the forward (blended) window.

### S4 — the sea in the longwave budget

With `fSky < 1` the base model treats the non-sky view as exchange-neutral
(no net longwave). For a wall above the Atlantic that under-states the warm
return: replace with an explicit term against sea surface temperature —
`ε·(1−Fsky)·σ·(T_sea⁴ − Ts⁴)`. SST sources, both already collected:
`sea_surface_temperature` on the marine best_match pull (forecast, hourly) and
Sevenstones buoy SST (live truth / validation). SST varies slowly — forecast
quality is not a concern.

### S5 — validate, calibrate, go live

- **Physics tests** extended with cliff scenarios: a W-face's SW peak lands in
  the evening, not noon; a clear night over a 14°C sea cools less than the
  same night at Bonehill; margin behaviour under maritime Td.
- **IR-gun spot checks** on Sennen visits → sign + rough size of the Ts−Ta
  gap, especially clear-evening and morning readings on the main faces.
- **Tune** `muScale` / `fSky` / `greasyMarginC` for Sennen from the above
  (ERA5 soil temp is a weak proxy on a part-ocean coastal cell — lean on the
  gun readings).
- **Flip `enabled: true`** in Sennen's override block; rock-surface rows start
  emitting and the existing chart + overview badge render automatically.

## 4. Risks / open questions

- **ERA5 truth on a part-ocean cell** trains the new radiation/cloud blenders
  (same accepted compromise as Sennen wind). Local cloud-shadow/sea-breeze
  effects won't be captured; watch the element verify pages once history
  accrues.
- **Face geometry is one plane.** Real crags have corners, overlaps and
  aspect spread; one (aspect, slope) pair is the same order of simplification
  as Bonehill's single `fSky`. Per-sector parameter sets are possible later
  if the gun readings justify it.
- **Greasy threshold semantics at the coast.** If maritime margins are tightly
  clustered, the 3-tier flag may need different cut points (or the margin
  itself charted more prominently than the tiers) — decide from real output.
- **Wind gust at Sennen** trains on ERA5 gusts like Bonehill's; an exposed
  clifftop's gust factor may differ from the 0.25° cell's. Acceptable v1; the
  convective term uses mean(wind, gust) so errors are halved on the way in.
