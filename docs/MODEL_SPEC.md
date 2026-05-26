# Model spec — what NWPs feed each model at each lead

Single reference for what's actually in production at each (target, phase, lead) cell. **Update this doc whenever a tier / picker / lead set changes** — it's the design intent; the feature schemas baked into each version's `feature_schema.json` are the deployed truth and should match.

## Headline conventions

- **Required** — feature value must be present for the row to be eligible for training/predict; rows missing any required model are dropped at build-time.
- **Optional** — feature column included; rows where the column is null still train/predict (LightGBM handles missingness natively).
- **UKV mode** — `Strict` (cycle = V, lead = target) vs `Averaging` (cycles 03Z + 15Z × leads bracketing target, effective lead averages to target across V hours). UKV is always passed as `optional`.

NWP id ↔ short:

| Short | Full id | Source |
|---|---|---|
| GFS | `gfs_seamless` (Open-Meteo) / `gfs_ncep` (S3) | NOAA |
| ECMWF | `ecmwf_ifs025` (Open-Meteo) / `ecmwf_ifs_oper` (S3) | ECMWF Open Data |
| AIFS | `ecmwf_aifs025_single` (Open-Meteo) / `ecmwf_aifs_oper` (S3) | ECMWF Open Data |
| ICON | `icon_seamless` | DWD via Open-Meteo |
| MF | `meteofrance_seamless` | Météo-France via Open-Meteo |
| UKMO | `ukmo_seamless` (Open-Meteo) / `met_office_global` + `met_office_ukv` (S3) | Met Office |
| GEM | `gem_seamless` | Canadian CMC via Open-Meteo |
| JMA | `jma_seamless` | Japan Met Agency via Open-Meteo |

## Temperature

### Phase 2b (lean) — _champion at lead 24+_

Open-Meteo `previous_runs` API (RunTimeSource = `offset_day`). Lead-bucket smear: each row is labelled `LeadHours = 24·N` for `N ∈ {1..7}` regardless of within-bucket valid-time. Hourly density across bucket.

| Lead | Required | Optional | UKV | Notes |
|---|---|---|---|---|
| 24 | GFS, ECMWF, ICON, MF, UKMO, GEM, AIFS | — | — | 7-NWP blend |
| 48 | same | — | — | |
| 72 | same | — | — | |
| 96 | same | — | — | |
| 120 | same | — | — | |

### Phase 2c (rich) — _challenger at lead 24+_

Same NWP set as 2b but adds 75 secondary features per row (per-model dew/RH/cloud/wind/pressure). 88 features total.

### Phase 2d (exact-runtime) — _champion at lead 12, challenger at 24+_

Raw S3 cycles via `s3-collect.yml`. RunTimeSource = `exact`. Tier T2 throughout.

| Lead | Required | Optional | UKV mode | Notes |
|---|---|---|---|---|
| 12 | GFS, AIFS | ECMWF, UKMO Global | **Strict** (cycles {0,6,12,18}Z × lead 12) | Champion at 12h |
| 24 | GFS, AIFS | ECMWF, UKMO Global | **Strict** (cycles {0,6,12,18}Z × lead 24) | Challenger to 2b at 24h |
| 48 | GFS, AIFS | ECMWF, UKMO Global | **none** — UKV had no measurable signal at 48h, dropped 2026-05-07 | Trained today; matches 72h decision |
| 72 | GFS, AIFS | ECMWF, UKMO Global | **none** — UKV's 0/6/12/18Z cycles cap at T+54h; out of reach | Trained today |

T2 history: bake-off at 12/24h on 2026-05-05 found GFS + AIFS as the right "always required" pair (both 4-cycle publishers, both at competitive long-range MAE), with ECMWF IFS oper + UKMO Global optional (2-cycle vs 4-cycle, useful when present, dropped without).

## Precipitation (per EA gauge station — Bellever, Bovey, Hexworthy)

### Phase 3a (lean) — _champion at lead 24+_

Open-Meteo `previous_runs`, lead-bucket smear like 2b. 8-NWP blend (adds JMA which is precip-only).

| Lead | Required | Optional | UKV | Notes |
|---|---|---|---|---|
| 24 | GFS, ECMWF, ICON, MF, UKMO, GEM, AIFS, JMA | — | — | 8-NWP blend |
| 48 | same | — | — | |
| 72 | same | — | — | |
| 96 | same | — | — | |
| 120 | same | — | — | |

### Phase 3c (rich) — _challenger at lead 24+_

Same NWP set as 3a, 55 features (per-model dew/RH/cloud/wind/pressure secondaries).

### Phase 3d (exact-runtime) — _challenger at all leads_

Raw S3. Tier P1 throughout. Per-station model — each (Bellever / Bovey / Hexworthy) has its own version trained on the EA gauge truth.

| Lead | Required | Optional | UKV mode | Notes |
|---|---|---|---|---|
| 12 | GFS, ECMWF IFS, AIFS | UKMO Global | **Averaging** (cycles {3,15}Z × leads {9, 15}) | |
| 24 | same | UKMO Global | **Averaging** (cycles {3,15}Z × leads {21, 27}) | Bellever +24h BSS +0.41 vs 3a's +0.37 |
| 48 | same | UKMO Global | **Averaging** (cycles {3,15}Z × leads {45, 51}) | Trained 2026-05-07; UKV gain 4-10% vs no-UKV |
| 72 | same | UKMO Global | **Averaging** (cycles {3,15}Z × leads {69, 75}) | Trained 2026-05-07; ties 3a at Bellever, beats at Bovey, loses at Hex |

P1 history: lifted from temp T2 with IFS *required* rather than optional — empirical bake-off at 24h showed 3-NWP-required gives a tighter blender for the precip task. JMA + GEM not yet wired into 3d (the empirical exact-runtime experiments stuck with the 4 long-range S3 sources; widening would require additional collectors).

### Phase 3b / 3p — _dry-window blenders_, see [docs/DESIGN.md](DESIGN.md)

Different target (binary "dry block ≥ N hours in 09–18 local") so feature space differs; not tabulated here.

## Bayesian P(wet) (WeatherProbabilistic — sibling repo)

Single hierarchical posterior across the three active stations (Bellever, Bovey, Hexworthy), 5 NWP precip features + hour sin/cos + lead.

| Phase | Variant | Lead support | Notes |
|---|---|---|---|
| 4 | Per-lead independent fits | 24, 48, 72 | Three posteriors, one per lead. Cycle-grid bottleneck: lead-24 chart limited to 2 valid times/day. |
| 5 | Lead-as-feature | Any (continuous lead covariate) | One posterior. Live predict scores at each cycle's actual lead. **Live as of 2026-05-07.** |

Model inputs: `precip_ecmwf_ifs025`, `precip_gem_seamless`, `precip_gfs_seamless`, `precip_icon_seamless`, `precip_meteofrance_seamless`, `hour_sin`, `hour_cos`, [`lead` only in Phase 5]. UKMO seamless dropped after Phase 4 5-model variant; AIFS not used.

## Element blenders (wind / humidity / shortwave-radiation / cloud-cover)

Per-element LightGBM blenders, lean feature set. Used by feels-like / UTCI compute.

| Element | Phase | Required | Optional | Lead set |
|---|---|---|---|---|
| Wind | 1 | GFS, ECMWF | ICON, MF, UKMO, GEM, AIFS | 24, 48, 72 |
| Humidity | 1 | GFS, ECMWF | ICON, MF, UKMO, GEM, AIFS | 24, 48, 72 |
| Cloud | 1 | GFS, ECMWF | ICON, MF, UKMO, GEM, AIFS | 24, 48, 72 |
| Shortwave radiation | 1 | GFS, ECMWF | ICON, MF, UKMO, GEM, AIFS | 24, 48, 72 |

Each emits hourly targets per lead bucket (commit 73de2f1 from 2026-05-07), so feels-like joins on (valid_time) populate every hour.

## Where the spec lives in code

When you change anything below, update this doc to match.

| What | File |
|---|---|
| Temp tier definitions (T1/T2/T3) | `src/WeatherBlend/Train/Exact12h/Exact12hFeatureBuilder.cs` `AllTiers` |
| Precip tier definitions (P1/P2) | `src/WeatherBlend/Train/PrecipExact/PrecipExactFeatureBuilder.cs` `AllTiers` |
| UKV picker tables (Strict + Averaging) | `Exact12hFeatureBuilder.UkvPicksStrict` / `UkvPicksAveraging` |
| Live UKV cycles + leads collected | `MetOfficeUkvArchiveCollector.DefaultCycles` / `DefaultLeads` |
| Active phases per target | `ActivePhasePolicy.ByTarget` |
| Champion-by-lead per phase | `data/models/<target>/MANIFEST.json` |
| Per-version deployed feature schema (deployed truth) | `data/models/<target>/<version>/feature_schema.json` |

Per-version `feature_schema.json` is the runtime truth — if this doc and a deployed version disagree, the deployed version wins (and this doc is wrong, fix it).
