# Rock surface temperature + condensation module — design plan

Status: **P0 spike validated; not yet wired into production (no P1 derived field).**
Drafted 2026-06-02. Companion to the deep-research survey (memory
`project_rock_surface_temp_condensation`). The `scripts/rock_temp_spike.py`
Force-Restore spike passes all 5 physical checks; its LW term was GFS-calibrated
2026-06-04 (§3 — warm bias removed). Next concrete step is **P1** (§8): the derived
`RockSurfaceTempPipeline` + predictions write. Absolute condensation rate still
needs on-site truth (no logger/IR yet).

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

## 3. Longwave parameterisation (GFS-calibrated 2026-06-04)

LW↓ is parameterised (there is still no *full-coverage* LW forecast — see below).
The spike implements Brutsaert clear-sky + a linear cloud enhancement:

```
LW↓   = εsky · σ · Ta⁴
εclear = 1.24 · (e_a/Ta)^(1/7)                  (Brutsaert; e_a in hPa from Td)
εsky   = clip(k_clear·εclear + k_cloud·(1−εclear)·C, 0, 1)
C      = total cloud fraction (0..1)
```

**Calibrated to GFS DLWRF (2026-06-04).** We now collect GFS downwelling longwave
(see below); a least-squares fit of LW↓ to it gave **k_clear≈1.0** (Brutsaert
clear-sky was already right) but **k_cloud≈0.54** — the original full-to-1.0 cloud
enhancement over-stated cloudy-sky LW by **~22 W/m²**, a warm bias that under-cooled
nights and under-predicted condensation. `lw_cloud_k=0.54` removes the bias
(resid −0.3 W/m²) at full coverage, lifts soil-rail directional tracking (0.771→0.795),
and roughly **halves the condensation flag's sensitivity to cloud-blend error**
(flip rate 12%→6% under a ±15% cloud error) — i.e. it *reduces* the cloud-blender
dependency that used to be the model's biggest risk (§9). Baked into the spike
PARAMS (`--lw-cloud-k` to override; 1.0 = raw Brunt).

**NWP LW now collected, but GFS-only and used to CALIBRATE, not drive.**
`DownwardLongwaveRadiation` comes from the GFS exact archive (DLWRF, leads 1–120h
at 10 points, hourly-sparse beyond 24h; OM/GEFS/ECMWF expose no LW). It is the
calibration *target* above, not a wholesale driver (50% hourly coverage,
single-model, 10km-orography caveat). A param+NWP-LW blend (use NWP where present)
is an optional future refinement.

Drive `C` from the `cloud_cover` element blend (now beats best-single ~3–4% and
mean-of-NWPs ~7–13%, +CAPE at 24h). **Layered (low/mid/high) LW is a dead end** —
the layer fields are null on the historical forecast archive (§9).

## 4. Inputs → existing fields

| Term        | Source field                              | Used for |
|-------------|-------------------------------------------|----------|
| SW↓         | `shortwave_radiation` element (or NWP)    | absorbed shortwave |
| C (cloud)   | `cloud_cover` element (or NWP)            | LW↓ cloud enhancement; weakest input |
| Ta          | temperature blend (2b/2d)                 | LW↓ (σTa⁴), convective term |
| V           | wind (element `wind` / NWP 10 m)          | convective coefficient h(V) |
| Td (+ e_a) | **NWP dew point directly** (`DewPoint2m`, per-model, 0% null) | the condensation margin `m = Ts − Td` *and* LW↓ clear-sky emissivity (Brunt, §3; e_a = vapour pressure from Td via Magnus) |

**Use NWP dew point directly — do NOT back-calculate from the RH blender.** Dew
point is exactly what we need (margin + vapour pressure) and every model supplies
it at 0% null. Deriving it from the `humidity` blender's RH + the temperature
blend would compound RH-error and temp-error through Magnus, for no gain — the
humidity blender targets RH only because that was chosen for the feels-like/UTCI
work. Dew point is smooth and well-forecast, and the humidity-blend finding
(barely beats best-single past 24h, see §9) suggests a dedicated dew-point *blend*
buys little — so **best-single or mean NWP dew point** is the simple default; a
small dew-point blend is an optional refinement, not a prerequisite. So humidity
enters the model **twice** (LW emissivity + the margin), both from Td directly.
SW↓, C, Ta are streams `FeelsLikePredictPipeline` already loads; add wind + Td.

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

- **Cloud-blender weakness feeds the most important term — LARGELY MITIGATED
  (2026-06-04).** The GFS calibration of `k_cloud` (1.0→0.54, §3) roughly halved
  the condensation flag's sensitivity to cloud-blend error (flip 12%→6% under ±15%
  cloud error), so LW↓ now leans on cloud far less. The cloud blender also improved
  (beats best-single ~3-4%, +CAPE@24h). Residual exposure is small; not a blocker.
- **LW parameterisation error — VALIDATED & DEBIASED (2026-06-04).** Validated the
  implied `LW↓` against **GFS DLWRF** (ERA5 has no LW via Open-Meteo, so GFS is the
  reference) and recalibrated to remove a ~22 W/m² warm bias (§3). Residual RMSE
  ~26 W/m² remains (a single linear `k_cloud` can't capture the nonlinear cf→LW
  relationship). A truly clean accuracy check would still want ERA5 `strd` (CDS) or
  field IR; the layered-cloud refinement that would cut the residual is BLOCKED —
  low/mid/high cloud is null on the historical forecast archive (only live OM rows
  and GFS-low carry it), so it can't be fit historically or driven in production.
- **Dew-point source.** Taken **directly from the NWPs** (`DewPoint2m`, 0% null),
  not back-calculated from the RH blender. It drives both the clear-sky LW↓
  emissivity (low-stakes — εclear is weakly sensitive to e_a) and the condensation
  margin (matters more). The humidity blender's RH skill is thin past 24h (loses
  to best-single at 48/72h), which is why a dew-point *blend* is unlikely to beat
  best-single/mean NWP dew point — start with the simple aggregate.
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
