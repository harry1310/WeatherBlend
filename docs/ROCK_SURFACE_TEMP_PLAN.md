# Rock surface temperature + condensation module — design plan

Status: **design / not built.** Drafted 2026-06-02. Companion to the
deep-research survey (memory `project_rock_surface_temp_condensation`).

## 1. Why

The site exists for climbing, and the one weather question we have never
answered is **"will the rock be wet from condensation?"** ("greasy"/"sweating"
rock). The physical criterion is simple and well established:

> condensation forms on a surface when **surface temperature ≤ air dew point**.

We already predict dew point. The missing quantity is the **rock surface
temperature** `Ts`, which departs from 2 m air temp `Ta` because the surface
runs its own energy balance: sun heats it above air by day; on clear, calm
nights it radiates to the sky and cools *below* air → dew. The worst case for
climbers — warm humid airmass over rock still cold from a cold spell — is a
*thermal-lag* effect, so the model must carry memory, not just instantaneous
forcing.

Decision output: a per-(lead, valid-time) **condensation margin**
`m = Ts − Td` (Td = dew point); `m ≤ 0` ⇒ condensation likely. Optionally a
**probability** by running the model across NWP members / blend spread.

## 2. Physical model — Force-Restore

A single ODE that captures the dominant diurnal mode plus thermal memory —
the parsimonious stand-in for solving the full 1-D heat-diffusion PDE
(Deardorff 1978; Lin 1980; Hu & Islam 1995; standard in NWP land schemes).

```
dTs/dt =  G_net / μ                 "force": instantaneous surface flux per unit areal heat capacity
        − (2π/τ)·(Ts − Td_deep)     "restore": relaxation toward the deep temperature (the memory term)

dTd_deep/dt = (Ts − Td_deep) / τ_long   slow drift of the deep reservoir (multi-day lag)
```

- `Ts`        rock surface temperature (output, °C/K)
- `Td_deep`   deep/restoring temperature — seed from a trailing multi-day mean of `Ta`
              (or ERA5 deep-soil temp if we ever ingest it); evolves slowly.
- `τ`         diurnal period = 86 400 s
- `τ_long`    multi-day reservoir timescale (~5–10 days; tune)
- `μ`         areal heat capacity of the diurnally-active skin ≈ thermal inertia
              `√(ρ·c·k)` × depth scale — **the main granite knob**

`G_net` is the four-term surface budget (W/m²):

```
G_net = (1 − α)·SW↓                    absorbed shortwave        [have: shortwave_radiation]
      + ε·(LW↓ − σ·Ts⁴)                net longwave              [LW↓ PARAMETERISED — see §3]
      − ρa·cp·h(V)·(Ts − Ta)           sensible / convective     [have: temperature, wind]
      − LE                              latent ≈ 0 for dry rock   [0 unless wet]
```

- wind enters via the convective coefficient `h(V) = 5.7 + 3.8·V` (McAdams), V = 10 m wind
- cloud enters via `LW↓` (§3) — the dominant control on nighttime cooling, hence on dew
- `α` granite shortwave albedo ≈ 0.3; `ε` longwave emissivity ≈ 0.95
- **sky-view factor `Fsky` < 1** multiplies the longwave loss for a boulder field
  (neighbouring boulders block part of the cold sky) — a Bonehill-specific knob

## 3. Longwave parameterisation (no LW field in our data)

We have shortwave (`shortwave_radiation` element) but **no downwelling longwave**.
Parameterise it the way METRo does in "manual mode":

```
LW↓ = εsky · σ · Ta⁴,    εsky = εclear · (1 + k·C²)
εclear = 0.23 + 0.484·√(e_a/100)     (Brunt; e_a = vapour pressure in Pa from Td)
C = total cloud fraction (0..1),  k ≈ 0.20–0.26   (cloud longwave enhancement)
```

This is the single biggest physics dependency on the **cloud blender**, which is
our weakest element model (see §9). Options: drive `C` from the `cloud_cover`
element blend, or from a raw/ensemble NWP cloud, or (later) the UA-improved
cloud blender if the `cloud-ua-bakeoff` shows a win.

## 4. Inputs → existing fields

| Term      | Source field                                  | Notes |
|-----------|-----------------------------------------------|-------|
| SW↓       | `shortwave_radiation` element (or NWP)        | already produced |
| C (cloud) | `cloud_cover` element (or NWP)                | feeds LW↓; weakest input |
| Ta        | temperature blend (2b/2d)                     | per-station already |
| V         | wind (element `wind` / NWP 10 m)              | convective coeff |
| Td        | dew point (from RH+T, Magnus) — for the margin only | not in the energy balance |

All are exactly the streams `FeelsLikePredictPipeline` already loads
(`temperature`, `ShortwaveRadiation`, `CloudCover` keyed by `(lead, valid)`),
plus wind + dew point.

## 5. Granite parameters & calibration

Literature seeds: ρ ≈ 2650 kg/m³, c ≈ 790 J/kg·K, k ≈ 2.5–3.5 W/m·K
(thermal inertia ≈ 2200 J·m⁻²·K⁻¹·s⁻½), α ≈ 0.3, ε ≈ 0.95. `μ`, `τ_long`,
`Fsky` are the free knobs.

**Calibration (needs on-site truth — there is no gridded rock-temp truth):**
1. cheap: a handful of IR-thermometer ("temp gun") readings on visits → sign +
   rough size of the Ts−Ta gap, especially clear-night mornings.
2. proper: a contact logger (DS18B20 / iButton) taped to a boulder for a few
   weeks, logged against nearest air temp. Fit `{μ, τ_long, Fsky, α}` by
   minimising Ts error — only ~4 parameters, so weeks of hourly data suffice.

Until then: seed literature values, sanity-check against ERA5 skin temp `skt`
(a *biased* proxy — vegetated-grid average, zero heat capacity, full sky-view,
so it leans toward over-predicting nighttime cooling → false-positive
condensation; treat as a prior, not truth).

## 6. Numerical integration & where it lives

Force-Restore is a **time integration over a contiguous hourly forcing series**,
which is the key structural difference from the existing element blenders
(those emit independent per-`(lead, valid)` samples). So the module operates
per **forecast anchor**: take that run's hourly forcing trajectory
(SW, C→LW, Ta, V) across the horizon, march the ODE forward (Euler at the
hourly cadence, sub-step if needed for stability), then slice out the
`(lead, valid)` points we report.

- **Spin-up:** start `Ts = Ta` (or `skt`) ~24–48 h before the first reported
  valid time and integrate through, so the initial condition and `Td_deep`
  settle before any output is used.
- **Home in code:** a new derived predict-tail step mirroring
  `Predict/FeelsLike/FeelsLikePredictPipeline.cs` — call it
  `Predict/Surface/RockSurfaceTempPipeline.cs`. It is **derived, not trained**
  (no LightGBM bundle, no manifest entry initially) — like 4b minting / feels-like.
- **Output:** a new `RockSurfacePredictionRow` (Ts, condensation_margin,
  optionally P_condensation) written to the predictions tree, version-stamped;
  site surfaces a "rock damp?" indicator + Ts vs dew-point chart.

## 7. Uncertainty (optional, fits the house style)

Run the integration across NWP members (or blend ± spread) → distribution of
`Ts` → **P(condensation) = P(Ts ≤ Td)**. Cheap (the ODE is trivial to run N
times) and turns a binary flag into a calibrated probability.

## 8. Phased build plan

- **P0 — offline spike (no repo wiring).** Implement the ODE in a script over
  a few months of Bonehill hourly forcing; plot Ts vs `Ta` and vs `skt`; sanity
  check the diurnal swing/lag look physical. Decide `μ`/`τ_long`/`Fsky` seeds.
- **P1 — derived field.** `RockSurfaceTempPipeline` + row + predictions write,
  driven by existing temperature/SW/cloud/wind streams + the §3 LW param.
  Condensation margin on the site. Literature granite params.
- **P2 — on-site calibration.** Ingest logger/IR-gun data; fit the 3–4 knobs;
  re-validate. This is what turns "physically plausible" into "trustworthy
  absolute margin."
- **P3 — ML upgrade (optional).** Once on-site Ts truth exists, a regression
  with a lagged-Ts memory term (or the full METRo 1-D conduction) as a
  challenger, scored against the logger.

## 9. Risks / open questions

- **Cloud-blender weakness feeds the most important term.** LW↓ (hence
  nighttime cooling, hence dew) depends on cloud, our least-skilled element
  model. Improving cloud (the `cloud-ua-bakeoff`) is complementary to this
  module, not separate. Consider driving LW from raw/ensemble cloud if the
  blend underperforms.
- **LW parameterisation error.** Brunt + cloud-enhancement is approximate;
  validate the implied `LW↓` against ERA5's longwave if we ingest it.
- **Sky-view for an all-aspect boulder field.** Bonehill faces every
  direction (aspect ~averages out per the user), but boulder mutual-shading
  reduces sky-view — a single `Fsky` is a simplification.
- **Forcing trajectory plumbing.** Needs a contiguous hourly per-anchor series;
  the current predict outputs are per-(lead,valid). May read the blended hourly
  forecast directly rather than the lead-bucketed element predictions.
- **Condensation ≠ wetness.** Seepage (porous rock weeping after rain) and
  rain-wetting are separate mechanisms already covered by the precip / dry-window
  models; this module is specifically the "dry sky but rock sweats" case.
```
