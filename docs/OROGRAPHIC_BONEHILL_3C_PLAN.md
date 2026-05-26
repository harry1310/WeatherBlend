# Orographic features for Bonehill 3c (pooled multi-station)

Goal: better Bonehill P(wet) by training a single pooled-station precip
blender on per-station terrain features, so the model learns terrain →
rainfall-bias as a *function* — then evaluate at Bonehill's terrain to
get a Bonehill prediction.

This is the only architecture that can move the Bonehill number rather
than the gauge numbers. The earlier "per-station 3c with Bonehill's
terrain baked in" sketch was a feature/target mismatch, and per-station
3c with each station's own terrain only improves the gauges themselves.

Scope: precipitation 3c target only. If pooled-with-terrain wins, the
features and pooled-fit pattern extend naturally to 4a (BART) and
dry-window targets — each a follow-on bake-off, not in scope here.

## Why pooling

In a per-station model, every "static" terrain feature
(`site_elevation_m`, `relief_*`, `upwind_gain_*`) is a constant across
that station's training rows. LightGBM splits on it once and learns a
station-level offset — which it already gets from labels. Static
terrain features add zero discriminative signal per-station; only the
dynamic features (`orographic_uplift × q`, `Froude`, wind-sector ×
upwind_gain) carry within-station information.

Pooling stations turns those constants into variables. The model now
sees that Bellever (~400 m, E-flank moorland) tends to over-fire on
SW-wind forecasts in one way, Hexworthy in a slightly different way,
Membury (~120 m, lowland) doesn't over-fire at all. From that
cross-station variation it learns "terrain feature vector X → typical
NWP-vs-truth bias Y". At predict time, feed Bonehill's terrain vector
and read off Bonehill's predicted bias correction on top of the NWP
ensemble — even though Bonehill is ungauged.

## The pool — 6 stations (7 with Princetown)

| Station                | Loc      | Elev   | Terrain context             |
|------------------------|----------|--------|-----------------------------|
| Bellever Dartmoor      | Bonehill | ~400 m | E-flank moorland            |
| Dartmoor nr Hexworthy  | Bonehill | ~280 m | SW moorland                 |
| Bovey Tracey           | Bonehill | ~80 m  | E valley (Dartmoor foot)    |
| Chards Snowdon Hill    | Membury  | ~150 m | E Devon ridge               |
| Goren                  | Membury  | ~120 m | E Devon lowland             |
| Raymonds Hill          | Membury  | ~140 m | E Devon ridge/coast         |
| **Princetown** (opt.)  | Bonehill | ~430 m | Central/W Dartmoor (re-add) |

Bonehill itself (393 m granite tor, E Dartmoor) sits inside the
elevation range, near Bellever/Princetown in geographic and elevation
terms but with distinct upwind geometry. That's the part the pooled
model has to extrapolate from terrain features rather than identity.

**Princetown** was dropped 2026-05-04 (redundant for triangulation);
historical parquets + dry-window models are still on disk under
`data/models/dry_window/ea_princetown/`. Re-adding to `config.yaml`
rainfall list + re-enabling the EA ID is a ~half-day pre-step. Strongly
recommended for the pool because it's the only central/west Dartmoor
high-elevation gauge — adds a different upwind geometry for the
wind-sector × terrain features.

## Hypothesis

Pooled-with-terrain (`3c-oro`) beats per-station rich `3c` at the
**three Bonehill-cell gauges** (Bellever / Hexworthy / Bovey) on Brier,
and we *infer* a Bonehill improvement via terrain transfer.

Expected lift at the gauges: 2-6% Brier at short lead, ~0-2% at long
lead. Modest because we only have 6-7 anchor points along the terrain
axis. At Bonehill: unverifiable — the assumption is that a model
demonstrably learning terrain-rainfall structure at the gauges
generalises to Bonehill's terrain vector.

This unverifiability is intrinsic to the question. State it openly
rather than dressing it up.

## Features

### Static (precomputed once per station, baked into config)

Now built **per gauge station, plus once for Bonehill itself** (for
predict-time evaluation). Stored under
`data/static/orographic/{station_or_loc_slug}.json`.

| Feature | Definition |
|---|---|
| `site_elevation_m` | Station/site elevation from DEM (cross-check vs config) |
| `nwp_cell_elevation_m` | Mean DEM elevation in the relevant NWP grid cell (per-NWP-resolution variants if needed: 9 km high-res, 25 km global) |
| `elevation_vs_cell_m` | `site_elevation_m − nwp_cell_elevation_m` — the "within-cell tor bias" |
| `relief_5km_m`, `relief_10km_m`, `relief_25km_m` | Max − min elevation within each radius |
| `upwind_gain_5km[sector]`, `upwind_gain_10km[sector]` | 8-sector lookup of mean elevation gain over the upwind 5/10 km |
| `terrain_ruggedness_5km` | Mean absolute elevation difference between site and surrounding cells within 5 km |

The three Bonehill-location gauges share the same NWP cell as Bonehill,
so `nwp_cell_elevation_m` is identical for those four — that's fine;
`elevation_vs_cell_m` still varies because each gauge sits at a
different elevation. Membury gauges sit in a different cell.

### Dynamic (per (valid_time, lead) in the feature builder)

| Feature | Definition |
|---|---|
| `wind_sector_sin`, `wind_sector_cos` | sin/cos pair of NWP-mean wind direction (kept continuous, no one-hot) |
| `upwind_gain_per_wind_5km` | `upwind_gain_5km[interp(sector)]` |
| `upwind_gain_per_wind_10km` | Same at 10 km radius |
| `orographic_uplift_m_per_s` | `max(0, u·dh/dx + v·dh/dy)` at the site, using NWP-mean (u, v) and precomputed local terrain gradient |
| `uplift_x_humidity` | `orographic_uplift · q`, where `q` is specific humidity from NWP `T2m / DewPoint2m / SurfacePressure`. Physical proxy for orographic precip rate |
| `froude_proxy` | `wind_speed_10m / max(relief_5km_m, 50)`. <1 → flow around (uplift suppressed), >1 → flow over (uplift enhanced) |

### No station-identity covariate

Deliberate choice: **no `station_id` feature**. The earlier draft
added it to absorb station-specific NWP-cell biases. But at Bonehill
predict time, LightGBM's behaviour for an unseen categorical level is
to route through the default direction — in practice the most-common
training subset's calibration. That risks silently applying (say)
Bellever's calibration on top of Bonehill's terrain features, which
defeats the point of the pooled architecture.

Cost of dropping it: station-specific NWP biases and gauge calibration
quirks confound back into the terrain coefficients. With 7 stations
this is a real but bounded risk — the bake-off SHAP step will surface
it (if terrain features pick up on cell-specific NWP biases rather
than physical structure, the SHAP signs will be incoherent and we'll
know).

If pooled-no-id fails the bake-off but the failure mode looks like
"terrain coefficients ate station bias", revisit with INLA hierarchical
intercepts as the v2 — that's the principled way to separate the two
effects.

### What I'm deliberately NOT adding (v1)

- **Full Smith-Barstad linear orographic precip model** — proper but
  several hundred lines of code; `uplift × q` proxy captures the
  first-order effect. Revisit if v1 wins big.
- **Brunt-Väisälä / true Froude** — needs multi-level NWP temperature
  profile we don't collect. Collector-side change for second-order
  improvement.
- **Per-station random intercepts in a Bayesian model** — that *would*
  be the right way to do this in INLA/PyMC (cf. 5a), but the
  closest-deployed pattern is LightGBM with station-as-categorical and
  it's the right starting point for a bake-off. If the LightGBM pool
  wins decisively, an INLA hierarchical version is a natural follow-on
  (could ride on 5a's R-INLA pipeline).

## Data source

**v1: SRTM 1-arc-second (~30 m)** — free, no auth required from
CGIAR or similar open mirrors. Covers our area (SW England + East
Devon) with 2-3 tiles. Resolution is more than enough for orographic
features at 5-25 km radius against a 9 km NWP grid cell — the
30 m vs 50 m distinction is below the noise floor of what the
features encode.

**v2 (deferred): OS Terrain 50** — 50 m grid UK-specific DEM, free
under OGL v3, distributed via OS Data Hub (requires free account).
Slightly different elevation values vs SRTM in places (different
source data). Only worth the upgrade if v1 ships and SHAP suggests
elevation precision is a binding constraint, which it almost certainly
isn't at our scale.

Don't bother with OS Terrain 5 (5 m) — orders of magnitude overkill.

Sanity check at build time: site-elevation derived from SRTM at the
configured (lat, lon) should match config's `elevationMeters` within
~±10 m. Bigger discrepancy = either the config is wrong or we're
reading the wrong tile.

## Integration

### Static-features build

New script: `WB/scripts/OrographicFeatures/build_static_orographic.csx`
(or a small .NET command).

1. Read `config.yaml` rainfall stations across both locations + each
   location's `lat/lon`.
2. Open OS Terrain 50 rasters covering the combined bounding box.
3. For each station and each *location* (so Bonehill itself gets a
   record for predict-time terrain), compute the static table.
4. Write `data/static/orographic/{slug}.json`. Sync to R2.

Run manually when stations are added/removed — not part of any cron.

### Pooled feature builder

New class `PrecipPooledRichFeatureBuilder` (sibling of the existing
`PrecipRichFeatureBuilder`):

1. Load all `data/static/orographic/{slug}.json` records on
   construction.
2. SQL pivots NWP forecast + EA truth rows for **all stations in the
   pool**, joined to each station's static record.
3. For each row, compute the 6-7 dynamic orographic features from
   (NWP-mean wind, NWP-mean q, station-static lookup).
4. Append the static block + dynamic block to the feature vector.
   No station-identity feature — see the no-station-id note above for
   the reasoning and the SHAP check that detects the failure mode.

Per-station `PrecipRichFeatureBuilder` stays untouched — `3c` remains
the deployed champion until the bake-off says otherwise.

### Predict path

New `PrecipPooledPredictCommand`:

1. Load the pooled bundle.
2. For each (lead, valid_time) target at Bonehill: build the feature
   row using Bonehill's static record (`bonehill_rocks.json`) and the
   Bonehill NWP cell's forecasts. No station-id slot in the feature
   vector — Bonehill is just another terrain vector to the model.
3. Write to `data/predictions/precipitation/bonehill_rocks/...` under
   the `3c-oro` phase tag.

Also score at each pooled gauge for the bake-off comparison — same
prediction path, just substituting that gauge's static record +
`station_id`. Gauge predictions land under the existing per-station
predictions tree so verify-history picks them up automatically.

### Spec / phase entry

```yaml
# phases.yaml
precipitation:
  phases:
    - id: "3a"
      role: champion
      impl: dotnet
    - id: "3c"
      role: challenger
      impl: dotnet
    - id: "3c-oro"            # NEW
      role: challenger
      impl: dotnet
      requires_orographic: true
      # NB: trained on the pool but predicted for every location
      # whose terrain record exists. Add Membury to predict scope
      # once we trust the Bonehill answer.
      locations: ["bonehill_rocks"]
```

```yaml
# config.yaml blenders
- target: precipitation
  featureSet: rich-oro-pooled
  requiredModels: []
  optionalModels: [gfs_seamless, ecmwf_ifs025, icon_seamless,
                   meteofrance_seamless, ukmo_seamless, gem_seamless,
                   ecmwf_aifs025_single, jma_seamless]
  # Same model membership as `rich`; differs only in feature spec.
```

## Bake-off

**Hypothesis**: `3c-oro` (pooled-with-terrain) beats `3c` (per-station
rich) on Brier at the three Bonehill-cell gauges, with biggest gain at
24-48h.

**Method**:

1. Pre-step (~half day): re-enable Princetown in `config.yaml`
   rainfall list; run an EA backfill catch-up since 2026-05-04 to
   freshen its parquets. If this turns out fiddly, ship v1 without
   Princetown (6 stations) and add it in a follow-up.
2. Add `3c-oro` to `phases.yaml` as a challenger.
3. One smoke training run locally on the pool — confirm the bundle
   trains, the feature vector packs as expected, and the predictions
   for the three gauges land in a sane ballpark vs deployed `3c`. If
   pooled-gauge Brier is wildly worse than per-station `3c` Brier,
   something is wrong in the feature build before any verify-history
   accrues.
4. Let the Sunday auto-retrain produce both for ~2 weeks.
5. Read verify-history sidecars per gauge: per-lead Brier comparison
   `3c-oro` vs `3c`.
6. SHAP analysis on `3c-oro`: do the orographic features land above
   the dead-tier threshold? Particularly the dynamic ones
   (`orographic_uplift_m_per_s`, `uplift_x_humidity`,
   `upwind_gain_per_wind_*`). If they're all dead but Brier improved,
   the lift came from pooling alone (more data) and there's no terrain
   story.
7. Look at `station_id` SHAP. If it dominates, the pooled model is
   just learning station identity — terrain transfer to Bonehill is
   suspect.

**Decision criteria**:

- Beats `3c` by ≥2% Brier averaged over the three Bonehill-cell
  gauges at 24h, AND beats by ≥1% at 48h+72h.
- At least 3 of the 6 dynamic orographic features are non-dead per
  SHAP.
- Terrain-feature SHAP signs are physically coherent — `uplift_x_humidity`
  pushes P(wet) positive, `froude_proxy` < 1 (flow-around regime)
  pushes negative, etc. If signs are scrambled, the model is reading
  the terrain features as proxies for cell-specific NWP biases rather
  than physical structure (the failure mode the dropped `station_id`
  was meant to prevent) — don't promote.

If all three pass → promote `3c-oro` to champion for the Bonehill
prediction line; retire `3c` after one more week of dual-run.

If Brier doesn't improve or terrain features are SHAP-dead → write up
the negative result. Keep the static-orographic infrastructure for
future use (e.g. INLA hierarchical pooled, or per-location wind/cloud
element blenders).

## Bonehill verification — the irreducible gap

There is no Bonehill rain gauge. We cannot directly verify that the
Bonehill `3c-oro` prediction is better than the deployed `3c` Bonehill
proxy.

Three weak forms of indirect evidence we can still gather:

1. **Bellever ≈ Bonehill check**: Bellever (7.5 km W, similar
   moorland, similar elevation) is the closest natural analogue. If
   `3c-oro`'s Bellever prediction is clearly better than `3c`'s
   Bellever prediction, that's the strongest gauge-level evidence the
   Bonehill prediction has likely also improved.
2. **Terrain-feature SHAP consistency**: if the dynamic terrain
   features have stable, physically-sensible SHAP signs (uplift × q
   pushes positive, leeward-sector wind reduces), the model is
   learning the right structure.
3. **Sanity vs raw NWP at Bonehill**: spot-check on known-orographic
   days (strong SW wind, saturated airmass) that `3c-oro`'s Bonehill
   P(wet) is higher than the NWP-mean P(wet), in the direction the
   physics predicts. Doesn't prove correctness; falsifies a clearly
   wrong model.

State on the Models page that Bonehill `3c-oro` is *not* a
gauge-verified line — its skill is inferred from the gauge bake-off.

## Order of work (~2 weeks solo)

1. **Day 1** — OS Terrain 50 download for SW England + East Devon
   bounding boxes. Validate Bonehill elevation (393 m ± 5 m) and
   Bellever/Membury elevations against config.
2. **Day 2** — re-enable Princetown in `config.yaml` rainfall list;
   trigger EA backfill catch-up; verify rainfall parquets land. If
   Princetown re-add hits friction, defer to a follow-up and continue
   with 6 stations.
3. **Day 3** — build static orographic features for all 7 stations +
   Bonehill itself → `data/static/orographic/*.json`. Sync to R2.
4. **Day 4-5** — implement `PrecipPooledRichFeatureBuilder` with the
   static block + dynamic block + `station_id`; unit tests for the
   per-row pack and for graceful behaviour when a station's static
   record is missing.
5. **Day 6** — implement `PrecipPooledPredictCommand`; local smoke
   train + predict on the 7-station pool; sanity-check gauge-level
   Brier vs deployed `3c`.
6. **Day 7** — wire `3c-oro` into `phases.yaml` + retrain-blenders.yml
   so Sunday auto-retrain produces it; predict-and-render emits the
   Bonehill `3c-oro` line.
7. **Subsequent 2 weeks** — verify-history accrues; bake-off
   evaluation against the decision criteria.
8. **Decision day** — promote, hold, or retire.

## Why this scoped narrowly

- **3c only**: 3a, 4a are separate bake-offs even if this wins.
- **Bonehill prediction target**: Membury could in principle benefit
  from the same pooled model evaluated at Membury terrain, but the
  upfront ask is the Bonehill story. Add Membury predict scope as a
  follow-up if v1 wins.
- **Not dry-window**: dry-window is a windows-of-dry-hours question;
  orographic uplift is rainfall-enhancing, not dry-window-extending.
  Indirect benefit through 3a/3o marginals feeding MC dry-window
  phases, but a separate plan.

## Future expansion (out of scope for v1)

- **INLA hierarchical pooled**: per-station random intercepts + shared
  terrain slopes, fitted in R-INLA on the same 7-station pool. Natural
  follow-on if the LightGBM v1 wins and the structure looks
  hierarchical (station_id matters but terrain matters too).
- **Membury `3c-oro` line on the Membury site**: trivial once the
  pooled model is trained — feed Membury terrain at predict time, ship
  under `membury_devon` location scope.
- **4a-oro**: same terrain features into BART feature vectors. Separate
  bake-off.
- **Smith-Barstad full physical model**: justified only if
  `uplift_x_humidity` is the dominant lift source in v1 SHAP.
- **Brunt-Väisälä from multi-level T**: collector-side change to pull
  NWP temperature at multiple pressure levels. Big upfront, second
  order.

## Reference

- OS Terrain 50: https://www.ordnancesurvey.co.uk/products/os-terrain-50
- SHAP findings 2c + 3c (2026-04-24): memory
  `project_shap_findings_2c_3c.md`; analyser at `scripts/ShapAnalyze/`.
- Existing 3c per-station builder:
  `WB/src/WeatherBlend/Train/PrecipRichFeatureBuilder.cs`.
- Existing per-station predict:
  `WB/src/WeatherBlend/Commands/PrecipPredictCommand.cs`.
- Bake-off discipline: same pattern as 3a vs 3c, 3b vs 3p, etc.
- Princetown drop note: `config.yaml:48-52`; archived models under
  `data/models/dry_window/ea_princetown/`.
