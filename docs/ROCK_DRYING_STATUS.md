# Rock-surface drying model — status (Phase A + B)

_Built overnight 2026-06-16. Ships behind a config flag that is **OFF
everywhere by default**, so nothing on the live site changes until a location
is calibrated and the flag is flipped. This note is the morning review._

## What it does

The rock-surface model now tracks a thin **surface-water film** (mm) on the
slab, alongside the existing temperature + condensation outputs:

- **Rain** wets the film (NWP precipitation, summed per hour at predict time).
- **Dew** wets it when the surface is below the dew point (condensation).
- The film **dries** by a vapour-pressure-deficit flux (warmer/drier/windier =
  faster) plus a radiation term (sun bakes a wet slab), and **runs off** above a
  thin cap (granite holds almost nothing).
- The film is tracked **by source** — `RainWaterMm` vs total `SurfaceWaterMm` —
  so the climbing index can treat rain-wet and dew-wet differently.

**Phase B — latent-heat feedback:** when the film evaporates it pulls latent
heat out of the slab (and releases it when dew forms), so wet rock now runs
cooler as it dries. This feedback is applied **only when the drying model is
enabled** for the location — a gate-off location's rock temperature is
bit-for-bit identical to before.

## How it changes the climbing verdict (when enabled)

- **Rain-wet rock → hard Off gate**, with a dry-by ETA in the reason line
  ("Rock wet from rain — drying, climbable from ~14Z"). This is the
  "no point saying good-to-climb in hour X if X−1 had a downpour" rule.
- **Dew/condensation → friction penalty, NOT a gate** (unchanged from the
  2026-06-16 decision) — the rock-temp/dew margin is still being field-validated,
  so an uncertain "wet" call drags the verdict to Poor rather than nuking it.

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
2. To enable for a location, set `surfaceWaterEnabled: true` in that location's
   `rockSurface:` override block in `config.yaml` (Bonehill first, after the
   Thursday IR-gun readings).
3. The DRAFT coefficients to calibrate live in `RockSurfaceConfig`:
   `EvapVpdCoeff`, `EvapWindBase/Slope`, `EvapRadCoeff`, `MaxSurfaceWaterMm`,
   `WetThresholdMm`, `LatentHeatWm2PerMmHr` (the last is physically fixed at
   ≈680.6; the rest are guesses awaiting "how long did it actually stay wet"
   field notes).
4. The index gate threshold is `ClimbingConditions.RainWetThresholdMm` (0.05 mm),
   which shares its default with the physics `WetThresholdMm`.

## Tests

`RockSurfacePhysicsTests` (film accrues/dries, runoff cap, dew-vs-rain
attribution, latent feedback gated off = identical Ts, latent cooling when on),
`ClimbingConditionsTests` (rain gate on/off, dew not gated, sub-threshold no-op,
dry-by ETA map), `RockSurfaceConfigTests` (per-location enable + global coeff
carry-through). Fast lane green at 945 tests.
