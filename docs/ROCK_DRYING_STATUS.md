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
