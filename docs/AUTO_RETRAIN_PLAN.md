# Auto-retrain plan

Drafted 2026-05-10 (Sunday morning) at the end of the Phase 4a train/predict
split work. Captures the agreed scope so a fresh session can pick this up
without re-deriving the design.

## Locked-in decisions

| Question | Decision |
|---|---|
| Cadence | **Weekly**, Sunday morning, lands well before Mon 09:30 UTC verify. |
| Champion-vs-challenger gating | **Out of scope.** Would create unmanageable model proliferation. Retrain replaces in place. |
| Pre-train sanity gates ("option 2") | **In scope.** Hard-aborts the bundle write if upstream data is wrong. |
| Verify-side drift alerting ("option 3") | **In scope.** Wire alerts to existing rolling-Brier drift flag. |
| Tolerance bands for sanity gates | Use defaults (rows ±30%, NaN% absolute 0.20, label-rate delta absolute 0.10). Tune over the first month if false alarms pile up. |
| Drift-alert threshold | Same as the existing on-page flag — **1.5× rolling baseline**. No need for a separate stricter threshold for alerts. |

## Models in scope (every deployed prediction line)

- **.NET LightGBM** — temp 2b / 2c / 2d, precip 3a / 3c / 3d, dry-window 3b
- **.NET Element blenders** — wind, humidity, shortwave-radiation, cloud-cover (predicted via `predict-and-render.yml`'s `predict-all` composite, trained via `dotnet run -- train --target {element_target}`)
- **.NET parameter-free** — dry-window 3g (no actual training; **skip**)
- **Python BART (rpy2)** — precip 4a (already on the new train/predict split architecture; just needs the cron added)
- **Python Bayesian (PyMC + nutpie)** — precip 5a

## Phase 1 — Pre-train sanity gates (~2-2.5 days)

Foundation. Every other phase depends on this.

### 1a. Shared `training_summary.json` sidecar

Each train script writes a small companion JSON next to `training_metadata.json`:

```json
{
  "rowsTrain": 69510,
  "rowsVal": 14895,
  "rowsTest": 14895,
  "featuresEffective": 23,
  "perFeature": {
    "precip_gfs": { "nanPct": 0.02, "mean": 0.31, "std": 0.78, "p01": 0.0, "p99": 4.2 },
    ...
  },
  "perStation": {
    "ea_bellever_dartmoor": { "labelRate": 0.31 }
  }
}
```

Captured at fit time, ~10 KB.

### 1b. Reusable guard helper

- **.NET**: `src/WeatherBlend/Train/Common/RetrainGuard.cs`
  - `CheckAgainstPrevious(currentSummary, previousSummary, tolerances) -> GuardResult`
  - Tolerance bands per-metric with sensible defaults; per-(target, phase) overrides via `data/models/retrain_tolerances.json`.
- **Python**: `WeatherProbabilistic/src/retrain_guard.py`
  - Same shape, numpy/pandas-based. Used by `train_4a.py` and `train_5a.py`.

Default tolerance bands:
- `RowsDeltaPct`: 0.30
- `NanPctAbsolute`: 0.20
- `LabelRateDeltaAbsolute`: 0.10
- `FeaturesEffectiveDelta`: 0 (any change = abort, signals a column dying or being added)

### 1c. Wire into every train path

- Each trainer: load previous version's `training_summary.json` (latest by lex sort under the composite's models dir), compare, **abort with non-zero exit + log a structured reason** if any band is breached. Bundle write happens only on pass.
- First-ever training (no previous summary on disk) skips the check and just writes its own summary as the new baseline.

### 1d. Friendly failure reporting

- Guard failure exits the workflow non-zero → existing GH App webhook auto-files `[ci-fail] retrain-{target}` issue.
- Issue body: which bands breached, current vs previous values, suggested action ("inspect upstream data; re-fire after manual investigation").

## Phase 2 — Auto-retrain workflows (~2 days)

### 2a. Workflow files

Two workflows per repo, grouped by trainer language so failures isolate cleanly.

**WeatherBlend** (`.NET LightGBM`):
- `retrain-blenders.yml` — runs every .NET train target sequentially with `continue-on-error: true` per step so one phase failing doesn't cascade. Targets in order:
  - temperature 2b, 2c, 2d
  - precipitation 3a, 3c, 3d
  - dry-window 3b
  - element wind, humidity, shortwave-radiation, cloud-cover
- Each step: `dotnet run -- train --target X` → guard check → (on pass) `rclone copy` the new bundle dir to R2.
- Estimated wall: ~30-60 min for the full sweep.

**WeatherProbabilistic** (Python):
- `retrain-python.yml` — runs `train_4a.py` + `train_5a.py` sequentially, each guarded.
- Estimated wall: ~25 min for 4a + however long 5a takes (PyMC sampling — likely 10-30 min).

### 2b. Sequencing — Cloudflare gates Sunday, refresh chains retrain

The original plan said to fire retrain on the same noon Cloudflare tick as
the daily refresh workflows. That's a race — both refreshes take 5–10 min,
retrain depends on their R2 output, and firing all three on the same tick
means retrain might start before refreshes have finished writing.

A 2026-05-10 review explored a workflow_run + co-located-gate design but
ran into the cross-repo wrinkle: `workflow_run` only triggers within a
single repo, and `previous-runs-refresh` lives in WeatherBlend while
`retrain-python` lives in WeatherProbabilistic. Plus the Cloudflare free
tier caps cron triggers at 5, so adding a 6th Sunday-only tick wasn't an
option.

**Adopted design (revised 2026-05-10 evening):** Cloudflare keeps cadence,
WeatherBlend's `previous-runs-refresh` chains retrain on its own
completion. The "is it Sunday?" decision lives in the worker; the workflow
just acts on a passed input. Three pieces:

1. **Cloudflare worker** branches the noon tick — fires
   `previous-runs-refresh.yml` with `inputs.train_after_refresh = true` on
   Sundays, false otherwise. No new cron added (still at the 5-cron
   free-tier cap).
2. **`previous-runs-refresh.yml`** gains the boolean input. Final step
   (gated on `inputs.train_after_refresh == 'true' && success()`) mints a
   GitHub App installation token via `actions/create-github-app-token`
   and fires `repository_dispatch` at WeatherProbabilistic's
   `retrain-python.yml` (and later WeatherBlend's `retrain-blenders.yml`
   for the .NET side). Payload carries `phases: 'all'` and
   `add_spread_features` once promoted.
3. **`retrain-python.yml`** adds `on: repository_dispatch: types: [retrain]`
   alongside the existing `workflow_dispatch` (manual). The day-of-week
   gate goes — Cloudflare already gated cadence; manual dispatches use
   `force=true` (default) to override.

Why this shape over the earlier proposals:

- **No race**: retrain literally cannot start until refresh completes
  successfully. The workflow_run path had the same property within one
  repo; repository_dispatch extends it cross-repo via the existing GitHub
  App auth path Cloudflare already trusts.
- **No Sunday-checking duplication**: cadence rule lives in ONE place
  (the worker). Workflows act on signals, not on calendar lookups.
- **Free-tier safe**: piggybacks on the existing `0 12 * * *` cron, no
  6th cron needed.
- **Manual override stays clean**: `gh workflow run retrain-python.yml`
  fires via `workflow_dispatch` and bypasses the chain entirely.

Setup cost: the GitHub App that Cloudflare already uses needs its
`app-id` and `private-key` exposed as repo secrets in WeatherBlend
(workflow secrets, not Cloudflare secrets — separate scope). One-time
~5 min config.

Sunday timing: noon UTC refresh → completes ~12:10 UTC → repository_dispatch
fires → retrain starts ~12:11 UTC → 5a finishes by ~15:00 UTC → fresh
model live for ~20h before Mon 09:30 UTC verify.

(For brevity the YAML / TS snippets are in the slice-3 implementation
commits — see `cloudflare/scheduler-worker/src/index.ts` for the worker
branch and `.github/workflows/previous-runs-refresh.yml` for the chain
step.)

### 2c. Retrain ordering / parallelism

- previous-runs-refresh's final step fires repository_dispatch for BOTH
  retrain workflows on the Sunday tick (slice 1 = retrain-python only,
  slice 4 adds retrain-blenders). They run in parallel as independent
  workflow runs — different repos, no shared lock, disjoint R2 prefixes.
- Within each workflow, targets run sequentially via the `phases` input
  flow so a single workflow runner does the whole sweep without runner-
  cap contention.

### 2d. Stale workflow_run / co-located-gate proposal (NOT adopted)

Earlier this thread iterated through:
- piggyback on the same Cloudflare tick (race condition)
- workflow_run trigger + co-located day-of-week gate (cross-repo limit)
- standalone Sunday cron (free-tier 5-cron cap)

The Cloudflare-gates-Sunday design (section 2b above) resolved all three
by moving the Sunday-check to the only place that knows what day it is at
dispatch time (the worker) and using repository_dispatch (cross-repo
capable) instead of workflow_run (same-repo only).

## Phase 3 — Verify-side rolling Brier drift alerting (~1 day)

Infrastructure already exists; just need the alerting wired.

### 3a. Threshold

Existing verify computes `DriftFlag` per row at **1.5× baseline rolling MAE** (per `project_verify_shipped.md`). We use the same threshold for the alert — no separate stricter rule needed (per locked-in decision).

### 3b. Verify post-step

Add a step at the end of `verify.yml`:
- Scan freshly-emitted `verify_history_*.json` for `DriftFlag=true` rows in the latest run.
- If any present, exit non-zero with a structured log line listing the (phase, station, lead) cells.
- Auto-issues fire via the existing webhook handler under `[ci-fail] verify`.

### 3c. Suppress noise

Cooldown: don't re-issue if there's already an open `[ci-fail] verify` issue for the same (phase, station, lead) cell. Verify whether the existing GH App de-dupe logic handles this; if not, add a per-cell hash check.

## Phase 4 — Models card "Δ vs previous train" badge (~1.5 hours)

Small UI add. Shows whether the latest retrain regressed vs. the prior trained version on training-time metrics.

### Files touched

- **`src/WeatherBlend/Site/SitePages.Models.cs`**
  - `RenderModels` (~lines 90-103): when picking the freshest `ModelSummary` per `(Composite, Phase)`, also stash the second-freshest. Pass both into `RenderBlenderCard`.
  - `RenderBlenderCard` (~lines 170-239): add `RenderTrainingDeltaBadge(latest, previous, metricLabel)` helper. Render in header right after the version tag.
  - Helper: average `BlendTestMae` across leads present in **both** versions, format as `Δ −0.005 vs prev train` with green/red class. Tooltip: "Training-slice delta — test window shifts each retrain; see verify history for real-world skill trend."
  - Skip badge if previous is null (first-ever) OR if lead sets are disjoint (split-version phase like 2d short+long).

### Tests

- Two ModelSummary versions, latest improved → green badge with correct delta.
- Only one version → no badge HTML.
- Disjoint lead sets → no badge (split-version guard).
- Latest regressed → red badge.

### No schema or trainer changes

The badge reads existing `training_metadata.json` files already loaded by `LoadModelSummaries`.

## Phase 5 — Documentation + memory (~0.5 day)

- Update `WeatherBlend/CLAUDE.md` with auto-retrain section: cadence, guard rails, drift alerting, on-call playbook.
- Memory entries:
  - `project_auto_retrain_shipped_<date>.md` — what's deployed, where the cron lives, how to triage guard failures and drift alerts.
  - Update `feedback_*` if any new conventions emerge during build.
  - Mark the planning doc itself as "implemented; archived" via an addendum.

## Sequencing (the order to ship in)

1. **Phase 4 badge first.** ~1.5 hours. Free win, gives immediate visibility into manual-retrain effects on data already on R2 (two `*_phase4a` versions per station from 2026-05-09's deploy). Independent of everything else.
2. **Phase 1 sanity gates.** Foundation. Nothing else ships safely without this.
3. **Phase 3 verify drift alerting.** Independent of retrain; useful immediately for catching deployed-model drift on the current manual cadence. Backstop for Phase 2.
4. **Phase 2 auto-retrain workflows.** Once 1 + 3 are in, this is just plumbing.
5. **Phase 5 docs/memory.** End of work.

Total wall: ~6 days of focused work, splittable across sessions.

## Acceptance / done criteria

- All in-scope phases (.NET LightGBM, Element blenders, BART 4a, Bayesian 5a) retrain on the Sunday cron without manual intervention.
- A guard failure on any phase produces a `[ci-fail] retrain-*` issue with the breached bands listed; the bundle is NOT written (verified by checking R2 has no fresh version dir for that phase that week).
- A drift event in the verify cycle produces a `[ci-fail] verify` issue listing the (phase, station, lead) cells; cooldown prevents re-issue while open.
- Each Models card header shows a "Δ vs previous train" badge (green/red, with tooltip) once the second weekly retrain has landed.
- `CLAUDE.md` documents the cadence + on-call playbook.

## Files to read first when picking this up

- `src/WeatherBlend/Train/Common/ModelArtifact.cs` — versioning + manifest patterns this work plugs into.
- `src/WeatherBlend/Site/SitePages.Models.cs` — the Models page render (Phase 4 badge target).
- `src/WeatherBlend/Site/SitePages.ModelSpec.cs` — same dedupe pattern, useful reference.
- `cloudflare/scheduler-worker/src/index.ts` — the worker that needs the Sunday-only branch added.
- `WeatherProbabilistic/scripts/train_4a.py` — proven train-script pattern; mirror for 5a + new Python guard.
- `WeatherProbabilistic/scripts/predict_4a.py` — load-side pattern (won't change here, but useful context for the train flow).
