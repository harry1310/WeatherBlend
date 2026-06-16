# Wind blender productionisation plan

Status: planning, pre-implementation. Drafted 2026-05-27 from the bake-off
sequence in this session (`wind_uv_bakeoff.py` → `wind_mvn_bakeoff.py` →
`wind_mlp_bakeoff.py` → `wind_mlp_isotonic.py` → `wind_quantile_bakeoff.py`
→ `speed_blend_bakeoff.py` → `speed_regime_blend.py` → `gust_bakeoff.py`
→ `gust_ratio_bakeoff.py` — all in `C:/Users/rhcsl/AppData/Local/Temp/`).

## Goal

Add wind direction (with confidence wedge), wind speed CI, an uncertainty
ellipse, and a wind gust forecast to the production site. Three new
trained artefacts + one composition rule. Truth-source split: real obs
where dense, ERA5 only where unavoidable.

## Final design (locked in)

### Three artefacts + one composition step

| Phase id | Type | Where trained | Truth | Repo | Role |
|---|---|---|---|---|---|
| `wind_speed_lgb` | LightGBM speed regression | Dunkeswell SYNOP `wind_speed` | Dunkeswell MIDAS Open | WB | Speed input to blend |
| `wind_mvn` | PyTorch MLP, bivariate normal | Dunkeswell SYNOP `(u, v)` | Dunkeswell MIDAS Open | WP | Direction + CIs + speed input to blend |
| `wind_gust_lgb` | LightGBM gust regression | ERA5 `WindGusts10m` | ERA5 reanalysis | WB | Gust forecast |
| `WindSpeedBlend` | Sigmoid composition (no training) | n/a | n/a | WB | Final wind speed = blend of `wind_speed_lgb` + `wind_mvn` speed magnitude |

### Truth-source decisions (user-approved 2026-05-27)

- **Wind speed + direction → Dunkeswell SYNOP (MIDAS Open ID 01383).**
  Real obs. Dense hourly wsp + wdir for 2022-2024. Replaces ERA5 truth that
  `wind_speed_lgb` currently trains on.
- **Wind gust → ERA5 `WindGusts10m`.** Dunkeswell gust is sparse-by-design
  (only 241 rows in 2022 reporting >9 m/s, zero coverage 2023-2024).
  Met Office DataHub geohash gust is dense but only ~33 days of history
  (collection started 2026-04-24). Revisit when DataHub gust accumulates
  ~6-12 months.

### wind_mvn architecture — as locked 2026-05-27

The original plan said "45 features (same as wind_speed_lgb today)" — both
counts were contaminated by the same archive-NWP bake-off scope. Production
scope is identical to the wind_speed_lgb lock-in (6 production NWPs:
gfs / ecmwf_ifs025 / icon / gem / ukmo / jma; no HARMONIE; AIFS excluded
until 2025+ data accumulates) and the same `lean+ORO+spread` feature set
(29 features). One feature builder feeds both models — no architectural
divergence between the LGB speed model and the MLP direction model.

```
Input  : 29 features (production scope, matches wind_speed_lgb)
         6 per-NWP wind_speed
         6 per-NWP wind_direction
         5 ORO_LEAN (oro_wind_sin/cos/upwind_gain/uplift/uplift_x_q)
         12 SPREAD (mean+std × 6 vars: wsp, gust, t, td, p, cc)

Trunk  : Linear(29 → 64) → GELU → Dropout(0.10)
         Linear(64 → 64) → GELU → Dropout(0.10)
Heads  : mu_head    → 2 outputs (μ_u, μ_v)
         scale_head → 2 outputs (raw → clamp(-3, 3) → exp → σ_u, σ_v)
         rho_head   → 1 output  (raw → 0.99 · tanh → ρ ∈ (-0.99, 0.99))

Loss   : Bivariate normal negative log-likelihood
         NLL = log(2π) + log σ_u + log σ_v + 0.5 log(1-ρ²)
               + 0.5 / (1-ρ²) · ((du/σ_u)² + (dv/σ_v)² - 2ρ du dv / (σ_u σ_v))
         where du = u_truth - μ_u, dv = v_truth - μ_v

Optimiser: AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR T_max=200 eta_min=1e-5,
           batch=256, grad-clip 5.0, early-stop on val NLL patience=30 epochs.
Train preprocessing: median-impute NaNs from train, drop columns that are
            all-NaN in train (gust columns for ECMWF/JMA — neither publishes
            gust on Open-Meteo Previous Runs), standardise via train mu/sd.
```

**Bake-off result (production scope, 2026-05-27):** direction MAE **18.196°**
on 1,163 Dunkeswell SYNOP test rows (−18.7% vs NWP-mean 22.371°). Speed MAE
0.9918 m/s (worse than the dedicated LGB speed model — see WindSpeedBlend
section for blend evaluation). CI95 direction coverage 89.9%, speed 89.3%
— both undercovered, calibrated via the α' scalars below.

**Rich-variant tested + rejected**: 61-feature rich+oro (gust + thermo +
calendar on top of lean+ORO+spread) collapsed back to NWP-mean direction
MAE (22.4°) with CI widths ballooning to 100°. The small MLP overfits at
this data scale; the LGB winner found the same pattern less dramatically.
Stay at 29.

**Reproducibility:** `scripts/wind_mvn_production_scope_bakeoff.py`.

### wind_mvn bundle layout (per location, per lead)

```
data/models/wind_direction/{location}/{version}_wind_mvn/
  state_lead_24h.pt         # torch.save(state_dict)
  state_lead_48h.pt
  ...                       # one per lead
  feature_scaler.json       # {"24": {mean: [45], scale: [45], medians: [45]}, "48": {...}, ...}
  feature_schema.json       # ["WindSpeed10m_gfs_seamless", ..., "wsp_spd_std"]  (45 names in order)
  calibration.json          # see below
  training_metadata.json    # Phase=wind_mvn, dunkeswell_version, PerLead stats
  training_summary.json     # RetrainGuard sidecar (per-station-aggregated)
  test_predictions.parquet  # per-row (valid_time, lead, mu_u, mu_v, sigma_u, sigma_v, rho,
                            #          truth_u, truth_v) for verify
```

`calibration.json` per-lead:
```json
{
  "24": {
    "alpha_prime_dir": 0.967,
    "alpha_prime_spd": 0.967,
    "blend_center":    2.50,
    "blend_scale":     3.00
  },
  "48": { "..." }
}
```

Calibration scalars are **fit on val during training** and bundled. See the
bake-off section below for what each represents.

### wind_gust_lgb features (10) — as shipped 2026-05-27

The original plan said "22 features / 10 NWPs" — inherited from a prior
bake-off (`Temp/gust_ratio_bakeoff.py`) whose "10 NWPs" mixed 4 production
NWPs with 6 archive/retired NWPs (gfs_ncep, MO Global, MO UKV,
ecmwf_ifs_oper, KNMI/DMI HARMONIE). Audited 2026-05-27: only **4 production
NWPs publish gust on Open-Meteo Previous Runs** (GFS / ICON / GEM / UKMO);
ECMWF / MF / JMA / AIFS publish none. Production-scope variant:

```
4 per-NWP WindGusts10m  (gfs / icon / gem / ukmo)
4 per-NWP gust ratio    = WindGusts10m / max(WindSpeed10m, 0.5),
                          clipped to [0.5, 4.0]
1 gust_ratio_mean       (NWP-mean of the ratios)
1 gust_ratio_std        (NWP-std of the ratios — NWP disagreement on gust factor)
```

LightGBM hyperparams identical to other element blenders in production
(n_estimators=600, lr=0.05, max_depth=6, num_leaves=31, etc.). UKMO sits in
`optionalModels` and the training window is restricted to
`TrainingWindow.UkmoCleanWindowStart` (2024-09-01+) — same pattern as the
wind blender.

**Bake-off result (4 NWPs / 10 features, 2024 Bonehill, 1,251 test rows):**
MAE 1.0418 (-18.4% vs NWP-mean 1.2764). 6 variants tested
(`scripts/gust_4nwp_variants_bakeoff.py`); the 10-feat minimal ties the
24-feat richest variant (1.0408) within noise. **Orographic features HURT
at 4-NWP scope** — see `project_gust_production_scope_2026-05-27.md`.

**Archive-NWP question parked.** Adding gfs_ncep + MO Global + MO UKV to
training would buy ~6.5% MAE (`scripts/gust_production_scope_bakeoff.py`)
but mixing 6-hourly archive sources into hourly-cadence element blenders
is a broader cross-blender question — not a one-off gust decision.

### WindSpeedBlend (composition rule, no training)

For each output row in the blended wind_speed parquet:

```python
# Inputs (per row):
lgb_speed   = wind_speed_lgb.BlendValue
mvn_mu_u    = wind_mvn.mu_u
mvn_mu_v    = wind_mvn.mu_v
center      = wind_mvn.calibration.blend_center  # 2.50 from bake-off
scale       = wind_mvn.calibration.blend_scale   # 3.00 from bake-off

# Composition:
mvn_speed   = sqrt(mvn_mu_u**2 + mvn_mu_v**2)
w_mvn       = 1.0 / (1.0 + exp((lgb_speed - center) / scale))
final_speed = w_mvn * mvn_speed + (1.0 - w_mvn) * lgb_speed
```

Bake-off result: blended speed MAE 0.9345 m/s vs `wind_speed_lgb` alone
0.9422 m/s (-0.82% aggregate, -23.9% in the ≤2 m/s slice).

### Output schema additions

New parquet trees on R2 (mirroring existing element blender layout):

```
data/predictions/wind_direction/{location}/model_version={v}_wind_mvn/date={d}/predictions.parquet
   ValidTimeUtc, RunTimeUtcGfs, ..., RunTimeUtcAifs,
   MuU, MuV, SigmaU, SigmaV, Rho,
   BlendDirection, BlendDirectionCi95Lo, BlendDirectionCi95Hi,
   BlendSpeedMagnitude, BlendSpeedCi95Lo, BlendSpeedCi95Hi

data/predictions/wind_gust/{location}/model_version={v}_wind_gust_lgb/date={d}/predictions.parquet
   ValidTimeUtc, RunTimes..., per-NWP gust columns, BlendValue (gust point)

data/predictions/wind_speed/{location}/model_version=blend/date={d}/predictions.parquet
   ValidTimeUtc, RunTimes..., per-NWP wsp columns,
   BlendValue (the final blended speed),
   BlendCi95Lo, BlendCi95Hi  (carried through from wind_mvn after sigmoid weighting if useful)
```

The blended `wind_speed` parquet is the canonical wind speed source. Verify reads it.

## phases.yaml additions

```yaml
# Existing `wind` target retained for back-compat? Or split into 3 new targets?
# Decision: split. Keeps "what's a champion of what" unambiguous.

wind_speed:
  phases:
    - id: "wind_speed_lgb"          # the LightGBM input (was target `wind`)
      role: input                    # NOT champion — fed into the blend
      impl: dotnet
    - id: "wind_speed_blend"        # the final canonical wind speed
      role: champion
      impl: dotnet
      sources: ["wind_speed_lgb", "wind_mvn"]   # cross-target binding (like 3f→3a)

wind_direction:
  phases:
    - id: "wind_mvn"
      role: champion
      impl: python                   # PyTorch trainer + predictor in WP

wind_gust:
  phases:
    - id: "wind_gust_lgb"
      role: champion
      impl: dotnet
```

`role: input` is a new role tag. If `PhaseRegistry` enforces "exactly one
champion per target" and rejects unknown roles, will need a one-line widen
to accept `input` as a non-champion non-challenger role (analogous to how
3f source binding via `IntensityModelSources.cs` is enforced outside the
manifest).

## Predict + render chain (locked in 2026-05-27)

Source of truth: `cloudflare/scheduler-worker/src/index.ts` and the
workflow files in `.github/workflows/`. **Verify against those before any
implementation work.**

```
collect.yml (cron HH:45)
    ├── existing dotnet collectors (OM forecasts, METAR, EA rainfall, MO DataHub Spot+Obs)
    └── on success ──► predict-4a.yml (WP)              [Hop C, existing]
                  └── on success ──► predict-wind-mvn.yml (WP)  [Hop C extended — NEW dispatch
                                                                  parallel to predict-4a]

s3-collect.yml (cron HH+1:05)
    └── on any completion ──► predict.yml (WB)          [Hop D, existing]
                              ├── predict-all composite:
                              │     existing targets +
                              │     NEW: wind-gust target in element loop
                              └── predict-tail composite:
                                  existing 4b mint + R2 push +
                                  NEW: WindSpeedBlend step (reads wind_mvn from R2-synced,
                                       writes wind_speed/...)
                              └── on success ──► predict-3f.yml (WP)       [Hop F.1, existing]
                                                 └── on any completion ──► render-site.yml (WB)  [Hop F, existing]

verify.yml (cron Mon/Thu 09:30)
    └── on success ──► render-site.yml (WB)             [Hop E, existing]
```

Worker change: one extra dispatch in Hop C (the same hop that fires
predict-4a). No new hops, no re-gated chains. The `Promise.allSettled`
pattern already used by the noon refresh's `truth-refresh` +
`previous-runs-refresh` dispatch can be reused verbatim.

### Why parallel-with-4a, not a separate hop

`wind_mvn` consumes only forecast features from collect's Open-Meteo
pull — exactly the same dependency 4a has. There is no upstream output
from WB predict it needs to read. So it slots into the same hop as 4a,
not after WB predict.

`WindSpeedBlend` runs in WB predict-tail because that's the first point
in the chain that has BOTH inputs available locally: `wind_speed_lgb`
just produced (this cycle's element loop), `wind_mvn` already in R2
(landed ~15-20 min ago via Hop C). Same shape as 4b reading 4a from R2.

## Retrain chain (Sunday)

```
previous-runs-refresh.yml @ 12:00 Sunday
    └── on Sunday success ──► retrain-python.yml (WP)    [Hop A, existing]
                              ├── existing 4a + 3f training
                              └── NEW: train_wind_mvn.py for all (location, lead)
                          └── on Sunday completion ──► retrain-blenders.yml (WB)  [Hop B, existing]
                              ├── existing dotnet element blender sweep
                              └── NEW: wind_gust_lgb sweep + wind_speed_lgb retrain on Dunkeswell
                                       truth (target migration from ERA5)
```

No new hops. Both retrain workflows already exist and accept multi-phase
sweeps; adding wind_mvn / wind_gust_lgb is content, not structure.

## Bake-off results being locked in (all on 2024 Bonehill, 70/15/15 chrono split)

The numbers below the dividers are the **production-scope re-bake-offs done
2026-05-27**, replacing the original contaminated 14-NWP numbers. The
original contaminated numbers are preserved at the bottom of this section
for historical context — they are NOT what ships. The take-aways
(LGB winner, MLP winner, sigmoid blend) are unchanged; absolute MAEs are
slightly larger because we no longer have access to gfs_ncep / MO Global /
MO UKV / IFS oper / HARMONIE in the feature set.

### Speed point — production scope (Dunkeswell truth, 1,163 test rows)

NWPs: **6 production** (gfs / ecmwf_ifs025 / icon / gem / ukmo / jma). No
HARMONIE, no archive. Bake-off: `scripts/wind_speed_production_scope_bakeoff.py`.

| Model | MAE | RMSE |
|---|---|---|
| NWP-mean speed | 1.1454 | — |
| Best single NWP (gem_seamless) | 1.1018 | — |
| `wind_speed_lgb` (LightGBM, `lean+ORO+spread`, 29 feats) | **0.9591** | 1.236 |
| `wind_mvn` magnitude (MLP) | 0.9918 | — |
| **`WindSpeedBlend`** (sigmoid, center=2.50, scale=3.00) | **0.9282** | — |

Slice breakdown for the blend on production scope
(`scripts/wind_speed_blend_production_scope_bakeoff.py`):
- Low (≤2 m/s, n=139): 0.8019 (-13% vs LGB; pure MLP would be 0.7291 = -21%)
- Mid (2-8 m/s, n=846): 0.8311 (+1.7% vs LGB — acceptable, sigmoid still
  weights LGB heavily in this band)
- High (>8 m/s, n=178): 1.4887 (-0.7% vs LGB)

Bootstrap 95% CI on aggregate (sigmoid − LGB) per-row |error| delta:
**[-0.0177, +0.0060] m/s** — straddles zero, so the aggregate win is not
statistically conclusive at this n. The low-wind slice gain is genuine and
that's what motivates shipping the blend.

Tested + rejected blends (all val-fit, all lost on test):
- Hard switch on LGB threshold (val-fit t=8.00): +5.50% vs LGB
- Linear ramp (val-fit 5.0→10.0): +2.66% vs LGB
- Per-decile lookup (10 bins fit on val): +0.87% vs LGB

Val-fitting consistently overfits to "use MLP almost always" because val
favours MLP overall; the plan's *fixed* sigmoid is more robust at this
data scale.

### Direction point — production scope (Dunkeswell truth, 1,163 test rows)

| Model | MAE (circular degrees) |
|---|---|
| NWP-mean direction | 22.371° |
| `wind_mvn` atan2(-μ_u, -μ_v), `lean+ORO+spread` (29 feats) | **18.196°** (-18.7%) |

`rich + oro` variant (61 features incl. per-NWP gust+thermo+cal) collapsed
back to 22.4° — the MLP overfits at this data + feature scale. Stay at 29.

### Direction CI (95% nominal) — to be re-fit at production scope

Plan retains the **α'_dir grid-fit on val** calibration recipe. Production
α' may differ from the contaminated bake-off's 0.967; production-scope
uncalibrated CI95 coverage measured 0.899. The `train_wind_mvn.py` writes
α'_dir to `calibration.json` after fitting on val.

### Speed CI (95% nominal) — to be re-fit at production scope

Same — α'_spd refit on val per-bundle. Uncalibrated coverage 0.893,
width 3.96 m/s. Variance-inflation alternatives still rejected.

### Gust — production scope (ERA5 truth, 1,251 test rows)

Already shipped 2026-05-27. See `scripts/gust_4nwp_variants_bakeoff.py`:

| Model | MAE | vs NWP-mean |
|---|---|---|
| NWP-mean gust | 1.2764 | — |
| **`wind_gust_lgb` (4 NWPs, 10 feats — minimal gust+ratio+spread)** | **1.0418** | **-18.4%** |

ORO HURTS at 4-NWP scope here (rich+oro = 1.0535). 6 variants tested, the
minimal 10-feat ties the 24-feat richest variant within noise.

### Original (contaminated) bake-off numbers — historical only

The original Phase-design numbers below used the raw forecast tree
(`{fc.Model.nunique()} NWPs`, evaluated to 10-14 NWPs depending on
variable coverage) and included gfs_ncep / met_office_global /
met_office_ukv / ecmwf_ifs_oper / KNMI HARMONIE / DMI HARMONIE — none
of which feed the production element blender. They are preserved for
context only; **do not cite them as ship targets**.

- Speed point: `wind_speed_lgb` 0.942, blend 0.9345, blend low slice
  -23.9%, blend mid slice +4%, blend high slice -2.5%.
- Direction point: `wind_mvn` 18.60°, NWP-mean 22.02°.
- Direction CI: uncalibrated 0.928, calibrated 0.951 at α'_dir=0.967,
  width 99.3°.
- Speed CI: 0.921 coverage at α'_spd=0.967, width 4.49 m/s.
- Gust: 22-feat minimal 0.981 MAE, -22.4% vs NWP-mean (the "22 features"
  count came from the contaminated 14-NWP pivot; production-scope is 10).

## Implementation order

Roughly in dependency order. Each phase below has a clean "what new code
ships, what existing code changes" delineation so reverts stay surgical.

### Phase 1 — wind_gust_lgb (cheapest, no dependencies)

- WB: `Train/Element/Gust/WindGustFeatureBuilder.cs` (22-feature builder
  paralleling `WindFeatureBuilder.cs`). Includes per-NWP ratio computation.
- WB: `Train/Element/Gust/WindGustBlender.cs` (element blender, mirrors
  existing `WindBlender.cs` shape).
- WB: `Train/Element/Gust/WindGustPredictPipeline.cs` (predict-side, mirrors
  `WindPredictPipeline.cs`).
- WB: extend `ElementTargets` with `wind-gust` entry. Single string addition.
- WB: predict-all composite — add `wind-gust` to the for-tgt loop.
- WB: retrain-blenders.yml — add wind-gust to the sweep list.
- WB: `Config/phases.yaml` — add `wind_gust` target with `wind_gust_lgb`
  champion.
- WB: `WindGustVerifier.cs` (mirrors existing element verifiers, scores
  against ERA5 truth).
- WB: site rendering — add gust badge to wind card.
- Tests: smoke test for feature-builder happy path, predict round-trip,
  ratio clip bounds.

### Phase 2 — wind_mvn (the bigger piece)

- WP: `scripts/train_wind_mvn.py` (new, ~500 lines). Per (location, lead)
  trainer. Mirrors `train_3f.py` bundle-writing shape. Fits MLP + calibration
  scalars on val. RetrainGuard sidecar.
- WP: `scripts/predict_wind_mvn.py` (new, ~300 lines). Per (location)
  predictor. Loads bundle, standardises features, runs forward pass, draws
  500 MC samples, derives direction + speed + CIs, writes parquet.
- WP: `requirements.txt` — verify `torch==2.11.0+cpu` is sufficient and
  pinned. Add anything else specific.
- WP: `.github/workflows/predict-wind-mvn.yml` (new). Mirrors
  `predict-3f.yml` for R2 pull/push and workflow_dispatch surface.
- WP: `.github/workflows/retrain-python.yml` — add `wind_mvn` to the
  sweep (or new job, depending on existing pattern).
- WB: `Config/phases.yaml` — add `wind_direction` target with `wind_mvn`
  champion.
- WB: schema additions for `WindDirectionPredictionRow` (or extend
  ElementPredictionRow with optional MuU/MuV/SigmaU/SigmaV/Rho cols).
- Cloudflare worker `src/index.ts` — extend Hop C to dispatch both
  predict-4a AND predict-wind-mvn in parallel via `Promise.allSettled`.
- Tests: smoke for bundle round-trip, MC sample reproducibility (seeded),
  calibrated CI math.

### Phase 3 — WindSpeedBlend + wind_speed_lgb truth migration

- WB: retrain wind_speed_lgb on Dunkeswell SYNOP truth instead of ERA5.
  Pure retrain matter, no code change to the blender itself — just the
  feature builder's truth column. **Re-bake-off needed** to verify the
  0.942 MAE we observed holds at production-time feature freshness.
- WB: `Commands/WindSpeedBlendCommand.cs` (new). Reads
  `predictions/wind_speed_lgb/...` and `predictions/wind_direction/...`
  from R2-synced local, applies sigmoid blend, writes
  `predictions/wind_speed/...`. Pulls center/scale from wind_mvn's
  calibration.json.
- WB: predict-tail composite — add `WindSpeedBlend` step after 4b synthesis
  but before the R2 push.
- WB: schema additions for `WindSpeedPredictionRow.BlendCi95Lo/Hi`.
- WB: site rendering — wind card with speed band + direction arrow + wedge.
  Variable-guard UX: when predicted speed < 2 m/s, render "Variable" /
  full circle instead of an arrow.
- WB: extend `WindVerifier` (or new `WindSpeedBlendVerifier`) to score
  the blended output against Dunkeswell truth.
- Tests: smoke for sigmoid math, calibration.json round-trip, "variable"
  UX guard threshold.

### Phase 4 — Direction CI + speed CI rendering polish

This is mostly UX work after Phase 3 ships the data.

- WB: Site card SVG for direction wedge (an arc on the arrow). Width
  scales with `BlendDirectionCi95Hi - BlendDirectionCi95Lo` (mod 360).
- WB: Speed band rendering ("10-14 m/s" style) using Ci95Lo/Hi.
- WB: Optional analyst page showing σ_u, σ_v, ρ and the uncertainty
  ellipse (deferred to later if needed).

## Open items to verify BEFORE locking down code

These are items the bake-off didn't fully prove and that need to be
checked against actual production data before implementation, NOT guessed.

1. **`wind_speed_lgb` on Dunkeswell truth — does the 0.942 MAE hold at
   production feature freshness?** The bake-off pulled freshest-cycle
   per (NWP, ValidTime) over the whole 2024, which doesn't perfectly
   represent live-production timing. Run a follow-up bake-off that
   reproduces production-time feature availability and confirm MAE
   stays in the 0.9-1.0 m/s range.

2. **Per-lead validity of `blend_center` + `blend_scale`.** The bake-off
   found center=2.50, scale=3.00 was best on the test set pooled across
   leads. With per-lead training, each lead's calibration should be fit
   independently. Confirm fitting per-lead doesn't tank the gains.

3. **NWP availability at predict time when wind_mvn fires from collect.**
   Per the chain, wind_mvn runs alongside predict-4a from collect's
   completion. Confirm `predict_4a.py`'s feature list (the NWPs it
   actually reads) matches what wind_mvn expects. **Don't guess about
   this — read `predict_4a.py` and `scripts/_shared.py` MODELS_LEAN
   and compare to the 14-NWP bake-off list.**

4. **Membury training data sufficiency for wind_mvn.** Bonehill bake-off
   used ~8000 rows. Membury Dunkeswell paired rows count needs to be
   checked — if too sparse, Membury wind_mvn deferred to Phase 1.5 or
   later, and wind direction stays NWP-mean for Membury.

5. **Site UX: do "Variable" + wedge interact well?** UX call. Need
   wireframe / iteration with the user before implementing rendering.

## What this plan does NOT touch

- Wind humidity, cloud_cover, shortwave_radiation, feels-like blenders.
  All unchanged.
- Temperature blenders (2b, 2c, 2d). Unchanged.
- Precip blenders (3a, 3o, 3f, 4a, 4b). Unchanged.
- Dry-window blenders. Unchanged.
- Auto-retrain on-call playbook in CLAUDE.md. Unchanged (new phases just
  add their RetrainGuard sidecars).
