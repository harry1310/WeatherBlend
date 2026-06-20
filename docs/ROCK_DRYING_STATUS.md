# Rock-surface drying model — status (Phase A + B)

_Built overnight 2026-06-16; coefficients literature-grounded and **enabled for
Bonehill AND Sennen 2026-06-19**. Spray-as-film-wetting is still not modelled at
Sennen (no coefficient), but the rain Off-gate + dew + drying physics are the
same everywhere, and spray's effect on the verdict is carried by the separate
Spray quality factor. Behind a per-location config flag (`surfaceWaterEnabled`),
default off elsewhere._

## What it does

The rock-surface model now tracks a thin **surface-water film** (mm) on the
slab, alongside the existing temperature + condensation outputs:

- **Rain** wets the film (NWP precipitation, summed per hour at predict time).
- **Dew** wets it when the surface is below the dew point (condensation).
- On a **sloped face the film drains under gravity** (Jeffreys 1930 / Nusselt
  thin-film, `dh/dt = −ρ·g·sinθ·h³/3μL`, integrated analytically per substep) down
  to a thin **retained residual** (`RetainedFilmMm`) within ~an hour — a vertical
  wall sheds the bulk fast, a horizontal slab (sinθ=0) doesn't drain at all (it
  ponds). This is the 2026-06-20 vertical-face fix: previously the film could only
  leave by evaporation, which is the *horizontal-slab* assumption — wrong for a
  vertical climbing face, and the cause of "takes too long to dry" (façade
  literature, Blocken & Carmeliet: on a vertical surface runoff carries ~75–95% of
  the water, evaporation only 4–26%).
- The residual then **dries by a vapour-pressure-deficit flux** (warmer/drier/
  windier = faster) plus a radiation term (sun bakes a wet slab). Drainage gets the
  film to the residual in ~an hour; evaporation clears the residual over ~1–2 h.
- A **runoff cap** (`MaxSurfaceWaterMm`) is a backstop on the instantaneous film;
  the gravity-drainage term does the real shedding on a sloped face.
- The film is tracked **by source** — `RainWaterMm` vs total `SurfaceWaterMm` —
  so the climbing index can treat rain-wet and dew-wet differently.

**Phase B — latent-heat feedback:** when the film evaporates it pulls latent
heat out of the slab (and releases it when dew forms), so wet rock now runs
cooler as it dries. This feedback is applied **only when the drying model is
enabled** for the location — a gate-off location's rock temperature is
bit-for-bit identical to before.

## How it changes the climbing verdict (when enabled)

- **Rain-wet rock → a LADDER, not a snap** (Harry 2026-06-20). The original code
  snapped the verdict Off→Prime the instant the film cleared a threshold —
  "complete rubbish", because a slab comes good *gradually* as it dries. The shape
  now is:
  - **film ≥ `RainOffGateMm` (0.2 mm) → Off** — genuinely wet, you wouldn't climb it.
  - **dry threshold (0.05 mm) < film < gate → a graded friction penalty** that
    eases as the film dries (`RainWetnessFrictionMultiplier`, a smoothstep from the
    rain-wet floor at the gate up to no penalty at the dry threshold), so the
    verdict climbs **Off→Poor→Marginal→Good→Prime**.
  - **film ≤ 0.05 mm → dry**, no penalty.
  Either way the reason/friction detail names the wet event and the "climbable
  from" ETA (when the film drops back below the Off gate): "wet from rain since 04Z
  (~0.3 mm), drying — climbable ~13Z".
- **Dew/condensation → friction penalty, NOT a gate** (unchanged from the
  2026-06-16 decision) — the rock-temp/dew margin is still being field-validated,
  so an uncertain "wet" call drags the verdict down rather than nuking it.
- The wet-event time (`LastRainAtUtc`) is tracked across the hidden spin-up
  window, so the site can say *when* it last rained even when the wetting
  happened before the first reported forecast hour.

**Drying speed is unchanged (Harry 2026-06-20):** the physics coefficients were
left as-is — the presentation fix (continuous recovery + naming the wet event)
came first, to be eyeballed live before any evaporation retune.

## Near-term coverage (2026-06-20)

The rock model used to report only the blend-driven window (≥ 24h), so the verdict
for **today** was always a ≥ 24h-old run — model changes took a day to show. It now
emits the **near-term hours too** ([now, first blend hour)):

- **Air temp** stays on the temperature blend, which predicts down to **0h**.
- **Wind / shortwave / cloud** come from the **raw NWP model-mean** for those hours,
  because the element blenders don't predict inside ~24h (their earliest valid is
  ~tomorrow). Dew point + precip are NWP throughout, as before.
- The pre-report spin-up (48h before now) is still all-NWP and discarded — it only
  settles Ts.

Each row carries a `ForcingTier` (`blend` ≥24h / `nowcast` <24h). Near-term
(`nowcast`) tiles show a small **"≈ <24h estimate"** marker on the climbing verdict
+ a note in the Conditions drawer, so it's clear the wind/sun/cloud there are rawer
than the +24h blend tiles. Gravity drainage (above) is the contained physics change
that, with this near-term emit, finally makes "today" dry sensibly on the live site.

## What is NOT done (deliberately deferred)

- **Sennen wave-spray wetting** — spray adding to the film at the sea cliff.
  This needs wave height plumbed through the forcing trajectory (like SST is),
  plus a spray-deposition coefficient I have **no data to set** — I won't ship a
  guessed number. The latent-heat and rain/dew machinery is all in place; spray
  is a follow-up once there's something to calibrate it against.

## What to review / how to switch on

1. The columns are now WRITTEN on every rock predict run (`SurfaceWaterMm`,
   `RainWaterMm`) — inspect them on the next predict to sanity-check the film
   behaviour before trusting the gate.
2. **ENABLED for Bonehill 2026-06-19** (`surfaceWaterEnabled: true` in its
   `rockSurface:` block). Sennen stays off (no spray coefficient yet). To enable
   another location, set the same flag in its override block.
3. The coefficients in `RockSurfaceConfig` are now **literature-grounded, not
   guesses** (2026-06-19): `EvapVpdCoeff` (0.010) + `EvapWindSlope` (0.35) reproduce
   the bulk-aerodynamic Dalton wind-term (water-vapour transfer coeff C_E ≈ 1.3×10⁻³);
   `EvapRadCoeff` (0.0006) is anchored to the latent-heat constant (`LatentHeatWm2PerMmHr`
   ≈ 680.6) at ~40% of absorbed SW driving evaporation; `MaxSurfaceWaterMm` (0.4) sits
   below smooth-surface interception (~0.5 mm) for a steep runoff-shedding face.
   `WetThresholdMm` (0.05) is the one empirical knob (a climbing "too wet" judgement).
   Field notes ("stayed wet ~N h after rain") now VALIDATE the drying timescale
   rather than set the numbers from scratch. **Watch the first live cycle's
   `SurfaceWaterMm`/`RainWaterMm`** to confirm the film behaves before fully trusting
   the dry-by ETA.
4. The index gate threshold is `ClimbingConditions.RainWetThresholdMm` (0.05 mm),
   which shares its default with the physics `WetThresholdMm`.

## Tests

`RockSurfacePhysicsTests` (film accrues/dries, runoff cap, dew-vs-rain
attribution, latent feedback gated off = identical Ts, latent cooling when on),
`ClimbingConditionsTests` (rain gate on/off, dew not gated, sub-threshold no-op,
dry-by ETA map), `RockSurfaceConfigTests` (per-location enable + global coeff
carry-through). Fast lane green at 945 tests.
