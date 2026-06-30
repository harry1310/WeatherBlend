# Radar Nowcast — Productionisation Plan (Bonehill)

**Status:** PLAN (nothing built beyond the backtest). Drafted 2026-06-30.
**Scope:** a standalone 0–1h radar-advection precip nowcast, shipped as a *separate* product alongside the
main NWP blend — it never feeds the blend, so it cannot degrade it. **One engine, two outputs:**
1. **Live crag nowcast (the product)** — real-time at the **Bonehill crag** pixel, **armed on demand** (~12h on
   possible climbing days), shown on the site; dormant and zero-cost otherwise. No ground truth at the crag.
2. **Verification at the 3 Dartmoor EA gauges** (Bellever W / Bovey Tracey E / Dartmoor-nr-Hexworthy SW — the
   triangle the precip blend is verified against) — radar-nowcast vs the NWP blend vs gauge truth, run as a
   **scheduled backtest over the free 2-year ODIM archive** (unbiased, full history; reuses the `phase2_vs_nwp.py`
   harness). NOT dependent on armed-day live runs (those are sparse/biased toward climbing-decision days). The live
   predict *also* emits all 4 site pixels cheaply for a real-time cross-check; the archive backtest is the rigorous source.

**Sites (validated 2026-06-30 — distinct pixels, geometry matches reality):**
| site | lat | lon | ODIM pixel | role |
|---|---:|---:|---|---|
| Bonehill crag | 50.5831 | −3.7931 | (1472,678) | display (no truth) |
| Bellever Dartmoor | 50.582381 | −3.898151 | (1472,671) | EA gauge (verify) |
| Bovey Tracey | 50.592312 | −3.716672 | (1472,684) | EA gauge (verify) |
| Dartmoor nr Hexworthy | 50.548615 | −3.938746 | (1476,668) | EA gauge (verify) |

**Engine: pySTEPS (DECIDED 2026-06-30).**

---

## 1. Why — the validated basis

Backtest (`scripts/radar/phase2_vs_nwp.py`, cached) established, scored on the Bellever EA gauge on a
**matched** event set:

| +1h CSI (2,901 common events) | value |
|---|---:|
| exact +1h NWP (fair, leak-safe) | 0.442 |
| radar single pixel | 0.343 |
| radar both@3km | **0.455** |
| radar both@5km / @7km | 0.481 / 0.503 |

Radar advection, scored fairly (hour-accumulated, neighbourhood ≥3km), **beats our leak-safe +1h NWP**
at Bonehill — and 3km is *smaller* than the NWP's ~10km cell, so it isn't a box-size artifact. The edge
is a genuine 0–1h capability the hourly NWP blend lacks.

**Open caveat (not a blocker for a separate product):** whether radar beats the *fair* +1h NWP hinges on
`exact` being a healthy comparator. The freshness-ladder check was inconclusive (offset_day +24h scored
*above* exact +1h — likely offset_day's lead labels are unreliable per the documented "best-available, not
as-issued" behaviour, but possibly exact is weak). A standalone nowcast only needs to be useful on its own
(it beats persistence and is competitive/ahead at +1h), so this doesn't block productionisation — but worth
closing later (the `offset_day` skill-vs-lead diagnostic).

---

## 2. The live feed (the crux — SOLVED)

**Met Office UK radar on AWS Open Data** — confirmed reachable:

- Bucket `met-office-radar-obs-data` (region `eu-west-2`), **anonymous over plain HTTPS** (no AWS account):
  `https://met-office-radar-obs-data.s3.eu-west-2.amazonaws.com/?list-type=2&prefix=radar/YYYY/MM/DD/`
- Keys: `radar/YYYY/MM/DD/YYYYMMDDhhmm_ODIM_ng_radar_rainrate_composite_1km_UK.h5`
- **1km UK composite surface rain rate** (same resolution as the NIMROD backtest), **15-min frames**,
  **~20-min publish latency**, **ODIM-HDF5** (read with `h5py`), 2-year rolling archive.
- Licence **CC BY-SA** → the site must carry a "contains Met Office data © Crown copyright" attribution.

Rejected alternatives: RainViewer (lossy PNG tiles, personal-use ToS), Open-Meteo `minutely_15`
(model output, not observed radar), Met Office DataHub radar (paid tier).

**Timing reality:** ~20-min feed latency + ≤5-min polling ⇒ freshest frame ~20–25 min old at process time;
a +1h advect from it is valid ~35–40 min ahead of real-now (worst case) — still inside the 0–1h skill window.
The product message is "next ~30–60 min", not a clean 60.

---

## 3. Architecture / data flow

```
  Met Office AWS bucket (15-min ODIM frames, ~20-min latency)
        │  (anonymous HTTPS list + GET)
        ▼
  Cloudflare scheduler worker  ── cron */5 ──┐
        every tick, IF armed:                │  (at :00 also run the hourly watchdog, unchanged)
          list bucket → latest frame ts      │
          compare to R2 "last_frame" marker  │
          new frame? ── yes ──► dispatch radar-nowcast.yml (GitHub Actions)
                                      │
                                      ▼
                          fetch last ~3 frames → pySTEPS (LK motion + S-PROG/STEPS) → +1h
                          → read crag pixel (+ gauge pixels for cross-check) → P(wet) + start/stop
                                      │
                                      ▼
                          write radar/nowcast/bonehill.json to R2
                                      │
   Bonehill Overview page (static) ──┘  client-side fetch of the JSON →
        renders the "⚡ Live radar — next hour" card (only if armed + fresh)


  SEPARATE verification track (scheduled, e.g. weekly — NOT armed-gated):
    archive backtest over the rolling ODIM history → pySTEPS nowcast at the EA gauge pixels
      → score vs the NWP blend's gauge forecasts vs EA truth → rolling skill report (data/reports/)
```

Arming is a separate manual path: `radar-arm.yml` writes `control/radar_armed_until` to R2; the worker reads it.

**Key principle:** the static site is NOT re-rendered every cycle. The nowcast writes a small JSON; the
Overview card is drawn client-side from it. Live, cheap, degrades gracefully.

---

## 4. File-by-file changes

### Engine: lightweight OpenCV advection + growth/decay (v1, DECIDED 2026-06-30); pySTEPS = measured v2
**Constraint discovered 2026-06-30:** pySTEPS WON'T install on the dev box — it ships source-only on PyPI (no
wheels) and this Windows box has no MSVC compiler, no conda, no WSL. It builds fine on Linux (GitHub Actions) but
would be **CI-only**, i.e. no local iteration/debugging of the engine — unacceptable for the build-and-tune phase.

**v1 engine = our own lightweight OpenCV advection** (half-res Farneback motion + semi-Lagrangian `advect_box`,
already validated in the backtest), **plus a simple growth/decay term** to close the one real gap: a Lagrangian
intensity-**trend** (residual between the latest frame and the advected previous frame), extrapolated forward with
damping that increases with lead (trend less reliable further out). Builds everywhere, fast local iteration.
P(wet) via a light calibration of the accumulated field.

**pySTEPS = v2, decided empirically.** Once v1 is solid, run a pySTEPS variant (S-PROG / STEPS) through the
archive **verification track** in CI and compare gauge CSI head-to-head vs our engine. Upgrade only if it earns it.
DL (DGMR/MetNet) = SOTA for convective but heavy/GPU — out of scope.

- **Shared module `scripts/radar/_engine.py`** — `load_odim`, `pixel_of`, `to_u8`, `flow_between`, `advect`,
  `nowcast(frames, georef, sites, lead_min, dt_min, trend_gain, nbhd_km)`. Used by BOTH the live nowcast and the
  archive verification, so there's one production method. The backtest's `phase2_vs_nwp.py` stays as-is (its
  Farneback primitives can be re-pointed at `_engine.py` later to kill drift).

### Live-fetch recipe (PROVEN by the 2026-06-30 spike — read-only)
`fetch → parse ODIM → OSGB georef → crag pixel` works with h5py + pyproj (both installed). Frame `(2175×1725)`
float32 at `/dataset1/data1/data`; apply `gain`/`offset` (1.0/0.0), mask `nodata=-1` (`undetect=0`=dry); project
lat/lon through the file's own `where.projdef` (OSGB tmerc) and index off the UL corner (`col=(x-x_ul)/xscale`,
`row=(y_ul-y)/yscale`). Bonehill crag = pixel (1472,678); 7px W of the Bellever gauge (1472,671) — matches the
~8km real separation, georef validated. Latest frame auto-found by walking year/month/day prefixes.

### WeatherBlend repo — Python (feed + engine + calibration)
- **NEW `scripts/radar/_engine.py`** — the shared pySTEPS wrapper: `frames → LK motion → S-PROG/STEPS extrapolation
  → +1h field`, with helpers to read named pixels (crag + gauges) and accumulate over the hour. Used by BOTH the
  live nowcast and the archive-verification track, so there's one production method.
- **NEW `scripts/radar/fetch_odim.py`** — list the MO bucket prefix (today + yesterday near midnight; archive walk
  for backtest), download the latest **~3** `.h5` frames (pySTEPS wants a short history), parse ODIM → 1km rain-rate
  grid + georef (per the proven recipe above), crop to the crag+gauges+inflow window. Handle nodata/undetect.
- **NEW `scripts/radar/nowcast.py`** (live product) — fetch recent frames → `_engine` → read the **Bonehill crag**
  pixel (50.5831, −3.7931) → calibrate → emit `{p_wet_next_hr, start_eta, stop_eta, frame_valid, computed_at,
  frame_age_min}` JSON. May also emit the gauge pixels for a real-time cross-check.
- **NEW `scripts/radar/verify_archive.py`** (verification track) — over the rolling ODIM archive, run `_engine` at
  the **EA gauge** pixels and score radar-nowcast vs the NWP blend's gauge forecasts vs EA truth → rolling skill
  report (`data/reports/radar_live_verify_*.md`). This is the unbiased radar-vs-blend measurement; the existing
  `phase2_vs_nwp.py` (NIMROD/Farneback) stays as the historical backtest of record.
- **NEW `scripts/radar/calibrate.py`** — fit advected/ensemble output → P(wet next hour) (or use the STEPS ensemble
  exceedance probability directly in v2). NOTE calibration is fit at the **gauge**; applied at the **crag** (~8km,
  same radar field) assumes spatial transfer — reasonable, flag as an assumption.

### WeatherBlend repo — GitHub Actions
- **NEW `.github/workflows/radar-nowcast.yml`** — `workflow_dispatch` (called by the worker). Steps:
  pip install (**pysteps** + h5py/pyproj/numpy) → `fetch_odim.py` → `nowcast.py` → upload `bonehill.json` to R2
  (`radar/nowcast/`). Optionally re-checks freshness and no-ops if no new frame (belt-and-braces).
- **NEW `.github/workflows/radar-arm.yml`** — `workflow_dispatch`, input `hours` (default 12). Writes
  `control/radar_armed_until` (epoch) to R2 using existing R2 creds. `hours=0` ⇒ past timestamp ⇒ instant
  disarm. Optionally dispatch `radar-nowcast.yml` once so the first reading appears immediately.
- **NEW `.github/workflows/radar-verify.yml`** — scheduled (cron via the worker, e.g. weekly — NOT armed-gated):
  runs `verify_archive.py` over the rolling ODIM archive → commits/uploads the rolling radar-vs-blend skill report.

### WeatherBlend repo — site
- **EDIT `src/WeatherBlend/Site/SitePages.Index.cs`** — at the top of the per-location Overview, emit a
  hidden `<section id="radar-nowcast">` plus a small inline `<script>` that fetches `radar/nowcast/bonehill.json`,
  checks `armed_until > now` and `computed_at` freshness, and renders/hides the "⚡ Live radar — next hour"
  card. Bonehill-only (guard on location slug). Add the Met Office attribution line.
  (Dispatch point is `RenderSiteCommand.cs:529`, `loc.HasTab("overview")` — no change needed there.)

### Cloudflare scheduler worker
- **EDIT `cloudflare/scheduler-worker/wrangler.toml`**
  - Change watchdog cron `"0 * * * *"` → **`"*/5 * * * *"`** (still 5 crons total — no new cron, stays free).
  - Add an **`[[r2_buckets]]` binding** (the worker currently has none) so it can read the armed flag + the
    last-frame marker. (KV is an alternative; R2 reuses existing infra.)
- **EDIT `cloudflare/scheduler-worker/src/index.ts`**
  - Update `WATCHDOG_CRON` constant to `"*/5 * * * *"`.
  - In `scheduled()` for that cron:
    1. **Every tick:** read `control/radar_armed_until` from R2. If `now < armed_until`: list the MO bucket
       prefix, get the newest frame timestamp, compare to the R2 `radar_last_frame` marker; if newer,
       dispatch `radar-nowcast.yml` and write the marker.
    2. **If `new Date(event.scheduledTime).getUTCMinutes() === 0`:** also run `runWatchdog(..., 60, false)` —
       byte-for-byte today's hourly behaviour, so the failure-recovery is unchanged.
  - Add a tiny S3 XML-list parse helper (latest key under a date prefix). CPU is trivial (I/O dominated).

### Config
- **EDIT `config.yaml`** (optional) — a `radar:` block: crag lat/lon, window half-width, R2 paths for the
  armed flag / marker / nowcast JSON, calibration coefficients path. Keeps magic values out of code.

---

## 5. Ordered build sequence (de-risked — prove each rung before the next)

1. **Spike (read-only): DONE 2026-06-30.** Fetched today's latest ODIM frame, parsed it, read the crag pixel.
   Fetch → ODIM parse → OSGB georef → pixel proven (crag (1472,678); georef validated vs the gauge).
2. **`_engine.py` (lightweight OpenCV + growth/decay): DONE 2026-06-30.** Stood up locally (Path 2, no pySTEPS):
   `load_odim / pixel_of / to_u8 / flow_between / advect / nowcast` + `fetch_odim.py`. Validated on live frames —
   motion 14.8 km/h (eastward today), wet-centroid advects +14 col ≈ +12 expected (mechanics correct), site reads
   sane, `trend_gain` growth/decay stable. Locally iterable, builds everywhere.
3. **`nowcast.py`: DONE 2026-06-30.** Live frames → `_engine` (4 sites) → calibrate → `bonehill.json` with
   per-site P(wet), accum, max-rate, rain-from/until clock times, frame age, MO attribution. Runs end-to-end locally.
4. **`calibrate.py`: DONE 2026-06-30.** Logistic P(wet)=σ(a+b·log1p(accum/0.1)) fit from the backtest cache —
   **Brier 0.168 vs 0.231 climatology** (27% better), monotonic curve. v1 runs `trend_gain=0` to match it (growth/decay
   built but OFF until re-fit via verification). Output `data/radar/calibration_bonehill.json`.
5. **`radar-nowcast.yml`: WRITTEN 2026-06-30.** workflow_dispatch → pip → `nowcast.py` → rclone JSON to R2.
6. **`radar-arm.yml` + R2 flag: WRITTEN 2026-06-30.** `hours` input → write `control/radar_armed.json` to R2
   (`hours=0`=disarm) → fire one immediate nowcast.
7. **Worker: WRITTEN 2026-06-30.** `wrangler.toml` cron `*/5` + `[[r2_buckets]]` RADAR binding; `index.ts`
   `runRadarTick` (armed-check → MO list → marker compare → dispatch) with the watchdog gated to `minute===0`.
   **Type-checks clean** (`tsc --noEmit`).
8. **Site card: WRITTEN 2026-06-30.** `RenderRadarNowcastCard` in `SitePages.Index.cs` — Bonehill-only hidden
   section + client fetch, reveals only when the JSON is fresh (≈armed). **`dotnet build` clean.**
9. **Tests: WRITTEN 2026-06-30.** `scripts/radar/tests/` — engine unit tests + an end-to-end **smoke** on synthetic
   ODIM frames (offline) + dry-field guard. **7 passing.** `scripts/radar/requirements.txt` added.
10. **Verification track: WRITTEN 2026-06-30.** `verify_archive.py` (live `_engine` at the 3 gauges over the ODIM
    archive, radar P(wet) vs truth — CSI/Brier) + `radar-verify.yml`. Blend column stubbed (`load_blend_pwet`).

### Deployment — STATUS (DEPLOYED + VERIFIED LIVE 2026-06-30 ~21:30Z)
- **DONE — pushed to main** (commits 7e05416 radar code, 9d2598d rclone `--s3-no-check-bucket` fix, 1c82e7c worker).
- **DONE — worker deployed** (deploy-scheduler-worker run succeeded): `*/5` cron + `RADAR` R2 binding live.
- **DONE — pipeline verified in CI**: `radar-nowcast.yml` fetch→advect→calibrate→`bonehill.json`→R2 (success;
  JSON confirmed in R2, crag P(wet) ~0.20).
- **DONE — full loop verified**: armed via `radar-arm.yml` (flag in R2) → the WORKER's `*/5` tick fired the nowcast
  on its own and wrote `control/radar_last_frame=2026063021..` (proof `runRadarTick` runs). Watchdog untouched.
- **REMAINING — the card's data source (the one real gap).** The site is **Cloudflare Pages**; the JSON is in the
  **R2 bucket** = different origin, so the card's relative `{AssetPrefix}radar/nowcast/bonehill.json` won't reach it.
  Fix options: (a) a **Pages Function** that proxies `radar/nowcast/bonehill.json` from R2 (same-origin, cleanest);
  (b) enable the bucket's **public r2.dev / custom-domain URL + CORS** for the Pages origin and point the card's
  `url` there. Needs Cloudflare config + the chosen URL. Until then the card stays hidden (degrades gracefully).
- **REMAINING — `radar-verify.yml` auto-trigger** (worker CHAIN_EDGE off `verify.yml`); manual dispatch works now.
- **REMAINING — blend join** (`load_blend_pwet`) for radar-vs-blend in the verification.
- **REMAINING — v2 questions**: re-fit calibration with `trend_gain` on; A/B pySTEPS via the verification track.
- Disarm anytime: `gh workflow run radar-arm.yml -f hours=0`.

---

## 6. Open questions / risks
- **ODIM specifics** — RESOLVED in the spike (see the recipe above).
- **Crag-vs-gauge calibration transfer** — fit at Bellever, applied at the crag (~8km). Likely fine; monitor.
- **Verify scheduling vs the 5-cron cap** — `radar-verify.yml` needs a (e.g. weekly) trigger, but the worker is at
  the free-tier cron cap. Don't add a 6th cron — chain it off an existing weekly completion (e.g. after `verify.yml`
  or the Sunday retrain) via `handleWorkflowRun`, or run it from the watchdog tick on a day/time guard. Cheap either way.
- **pySTEPS in CI** — adds a heavier pip install (scipy/opencv); pin versions + cache the wheel in the workflow.
- **The A/B comparator question** (§1) — close later with the offset_day skill-vs-lead diagnostic; not a blocker.
- **Worker CPU budget (10ms free)** — the added S3 list+parse is I/O, not CPU; confirm via `wrangler tail` after deploy.
- **Frame gaps** — feed occasionally misses a scan; need ≥2 recent frames for flow, else skip this cycle gracefully.
- **Attribution** — CC BY-SA line must appear wherever the radar card shows.

## 7. Cost — all free
- **Cloudflare:** `*/5` = 288 ticks/day ≪ 100k/day request limit; per-tick = 1 S3 list + 1 R2 read (≪ 10ms CPU,
  ≪ 50 subrequests). No new cron (frequency change only), so no plan change.
- **GitHub Actions:** repo is **public** ⇒ unlimited free minutes; and the predict only fires on a genuinely new
  frame while armed (~4×/hr), zero when disarmed.
- **R2/KV:** flag + marker + small JSON are trivial against free-tier storage/ops.
