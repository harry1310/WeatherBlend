# weatherblend-scheduler

Cloudflare Worker that fires GitHub Actions workflows on a cron schedule via a
GitHub App. Built to replace the unreliable GitHub-side `schedule:` trigger —
GitHub's scheduler routinely delays runs by 15+ minutes and occasionally skips
them entirely when the runner pool is busy, which means scheduled work
silently slips. Cloudflare Workers cron triggers fire on time.

## What it does

```
Cloudflare crons (4)                          Workflow dispatched
─────────────────────────                    ──────────────────────
45 2,8,14,20 * * *   (every 6h, :45)           →  collect.yml
5 3,9,15,21 * * *    (every 6h, :05)           →  s3-collect.yml
0 12 * * *           (daily 12:00 UTC)         →  era5-refresh + previous-runs-refresh
30 9 * * MON,THU     (Mon + Thu 09:30 UTC)     →  verify.yml

Chained off workflow_run completions (not cron'd — see handleWorkflowRun):
  collect    ─(success)────►  predict-4a                (WeatherProbabilistic)
  s3-collect ─(completion)─►  predict-and-render
  verify     ─(success)────►  render-site
  previous-runs-refresh ─(Sunday success)─►  retrain-python ─►  retrain-blenders
        │
        ▼
weatherblend-scheduler (this Worker)
        │  cron path:    pick workflow(s) from event.cron via WORKFLOW_FOR_CRON
        │  webhook path: workflow_run completions drive the chains above
        │  both:         mint App JWT → installation token → POST .../dispatches
        ▼
GitHub Actions runs the workflow as if a human had clicked "Run workflow"
```

Same workflows, same code paths; only the trigger source changes. **No
workflow in this repo uses GitHub-side `schedule:` triggers** — they're
unreliable and the Worker's whole reason to exist. If you're adding a
scheduled workflow, the cron lives here, not in the YAML.

## Schedule rationale

Four crons. The 6-hour collect/predict slot (HH ∈ {2, 8, 14, 20}) runs:

| Cron | Workflow | Why this time |
|------|----------|---------------|
| `45 2,8,14,20 * * *` | `collect.yml` | Every 6h, `:45` past the hour — the offset gives Open-Meteo's GEM ingest time to land (earlier ticks went out on a stale GEM cycle). Runs in ~2.5–3.5 min. |
| `5 3,9,15,21 * * *` | `s3-collect.yml` | Raw S3 exact-runtime cycle pulls (GFS / IFS / AIFS / MO Global / UKV / GEFS) for the 2d/3d blenders. The ~9h gap past the previous synoptic cycle clears every publisher (slowest is ECMWF IFS, ~T+7-8h). At `:05` not `:00` so the `collect → predict-4a` chain (which starts when collect finishes ~HH:50) has run-room before `predict-and-render` is chained off s3-collect. Typical runtime 5–8 min, 60-min timeout. |
| `0 12 * * *` | `era5-refresh.yml` + `previous-runs-refresh.yml` | ECMWF publishes ERA5T ~09–10 UTC and Open-Meteo ingests within hours; 12:00 is past both, so the refresh lands on fresh data. Both pull a 14-day rolling window. On a **Sunday**, `previous-runs-refresh`'s success chains the weekly retrain. |
| `30 9 * * MON,THU` | `verify.yml` | Twice-weekly Mon + Thu 09:30 UTC. **Use day-name aliases (`MON`, `THU`) not numbers — Cloudflare cron uses 1=Sunday (not POSIX 0=Sunday), so `* * 1,4` fires Sun+Wed.** |

`predict-4a` and `predict-and-render` are **not cron'd** — each is
dispatched when its upstream workflow completes (`handleWorkflowRun`,
Hops C and D in `src/index.ts`):

- `collect` success → `predict-4a` (consumes collect's Open-Meteo forecasts).
- `s3-collect` completion → `predict-and-render` (it consumes s3-collect's exact-runtime cycles).

Chaining runs each the moment its input is ready instead of guessing a fixed
offset, and keeps the worker to 4 crons — Cloudflare's free tier caps at 5,
leaving one spare.

## Concurrency invariant

`collect`, `s3-collect`, and `predict-and-render` all share the GitHub
Actions concurrency group `weatherblend-data` with `cancel-in-progress:
false`, so they can never run concurrently — a later one queues behind a
running one. That is the hard guarantee that `predict-and-render` never
starts mid-collect; the Hop D completion chain is the same ordering made
explicit at dispatch time (the render is dispatched only once `s3-collect`
has already finished), so the queue should now rarely even engage.

## Adding a new schedule

Two coordinated edits:

1. Add the cron expression to `wrangler.toml` `[triggers]` `crons = [ … ]`.
2. Add the same string verbatim as a key in `WORKFLOW_FOR_CRON` in
   `src/index.ts`, mapping to the workflow filename.

The worker uses `event.cron` (the literal string from `wrangler.toml`) as the
lookup key, so any drift between the two files surfaces as an "unknown cron
expression" exception in the next scheduled fire.

## Layout

```
cloudflare/scheduler-worker/
  src/index.ts        Worker code (~200 lines, no runtime deps)
  wrangler.toml       Worker config: name, cron, plain vars
  tsconfig.json
  package.json        wrangler + @cloudflare/workers-types only
  README.md           this file
```

## Required secrets

The Worker reads four secrets at fire time. Set them via **wrangler** or the
Cloudflare dashboard (**Workers & Pages → weatherblend-scheduler → Settings →
Variables and Secrets**):

| Name | Value |
|------|-------|
| `GH_APP_ID` | The App ID (top of the GitHub App's settings page; integer). |
| `GH_APP_INSTALLATION_ID` | The Installation ID (visible in the install URL `/installations/<n>`; integer). |
| `GH_APP_PRIVATE_KEY` | Full contents of the `.pem` file generated when you created the App. PKCS#1 (`BEGIN RSA PRIVATE KEY`) or PKCS#8 (`BEGIN PRIVATE KEY`) both work — the Worker handles either format. |
| `GH_WEBHOOK_SECRET` | Random secret (32+ bytes) shared between the Worker and each repo's `workflow_run` webhook. Used to HMAC-verify webhook deliveries. Generate with `openssl rand -hex 32` or similar; configure the same value in every repo's webhook config (Settings → Webhooks → Secret). One value across both repos is fine — each webhook signs its own POSTs and the Worker verifies per-request. |

Plain (non-secret) configuration lives in `wrangler.toml` under `[vars]`:

| Name | Default | Notes |
|------|---------|-------|
| `GH_REPO` | `harry1310/WeatherBlend` | `{owner}/{name}` form. |
| `GH_REF` | `main` | Branch the workflow runs against. |

The workflow file isn't a plain var — it's selected per-cron in the worker
via `WORKFLOW_FOR_CRON` in `src/index.ts` (see "Adding a new schedule" above).

## Required GitHub App permissions

The App needs **Repository permissions** for the Worker's two responsibilities
(workflow dispatch + CI-failure issue management):

| Permission | Access | Why |
|------------|--------|-----|
| Actions | Read and write | Dispatch workflow runs (`POST /actions/workflows/{file}/dispatches`). |
| Issues | Read and write | Open / comment on / close `[ci-fail]` issues from the `workflow_run` webhook handler (added 2026-05-08). |

Webhooks (App-level) unchecked. Install on every repo whose workflows the
Worker needs to act on — currently **WeatherBlend** + **WeatherProbabilistic**.

## GitHub `workflow_run` webhooks (one per monitored repo)

In addition to dispatching workflows, the Worker now consumes the GitHub
`workflow_run` webhook event to open / auto-close `[ci-fail]` issues per
repo. **Each monitored repo configures its own webhook** pointing at the
same Worker endpoint — one Worker, many sources.

Configure on each repo (Settings → Webhooks → Add webhook):

| Field | Value |
|-------|-------|
| Payload URL | `https://weatherblend-scheduler.rhcslater.workers.dev/github-webhook` |
| Content type | `application/json` |
| Secret | the `GH_WEBHOOK_SECRET` value from the Worker config (same value in every repo's webhook). |
| SSL verification | Enable. |
| Which events | "Let me select individual events" → tick **Workflow runs** only. |
| Active | ✓ |

Repos to configure today: **WeatherBlend** + **WeatherProbabilistic**. Add a
webhook to any new repo whose CI failures you want to be alerted to —
no Worker code change needed.

The handler opens `[ci-fail] <workflow-name>` issues with label `ci-failure`
on `conclusion = failure | timed_out | startup_failure`. Successive failures
of the same workflow append a comment rather than spawning a new issue. On
`conclusion = success`, the matching open issue auto-closes with a "now
passing" comment. `cancelled` and `skipped` runs are ignored.

## Deploy

```bash
cd cloudflare/scheduler-worker
npm install
wrangler login              # one-shot browser OAuth
wrangler deploy             # ships the Worker; cron schedule activates
```

Subsequent deploys are just `wrangler deploy`.

## Test before the cron fires for real

Two easy paths:

1. **Manual HTTP trigger** — the Worker exposes `POST /dispatch?workflow=NAME`
   so you can force any of the three workflows from anywhere:

   ```bash
   # Defaults to collect.yml when no workflow query param is given:
   curl -X POST https://weatherblend-scheduler.rhcslater.workers.dev/dispatch

   # Pick a specific workflow:
   curl -X POST 'https://weatherblend-scheduler.rhcslater.workers.dev/dispatch?workflow=era5-refresh.yml'
   curl -X POST 'https://weatherblend-scheduler.rhcslater.workers.dev/dispatch?workflow=verify.yml'
   ```

   200 means the dispatch went through — go check the GitHub Actions tab to
   see the workflow running. 400 means the workflow name isn't in
   `WORKFLOW_FOR_CRON`; the response lists known workflows. 500 means an
   auth or config error; the body has the diagnostic message and
   `wrangler tail` shows the full log line.

2. **Local cron simulation** —

   ```bash
   wrangler dev --test-scheduled
   # in another terminal:
   curl http://localhost:8787/__scheduled?cron=15+2,8,14,20+*+*+*
   ```

   Same code path as production; runs against the real GitHub API with the
   real secrets; same constraints (no caching, JWT mint per fire).

## Observe

```bash
wrangler tail               # live log stream
```

Each cron fire logs the cron expression + dispatch outcome. A failed dispatch
also throws (so the cron run is recorded as failed in Cloudflare's metrics)
and writes the GitHub error body to the log.

## Cutover history + remaining work

| Workflow | GitHub `schedule:` removed | Notes |
|----------|---------------------------|-------|
| `collect.yml` | ✅ 2026-05-01 (`874653f`) | Cloudflare proven, GH cron deleted same day. |
| `era5-refresh.yml` | ✅ 2026-05-01 (`bebca2c`) | Moved 06:00 → 12:00 UTC at the same time. |
| `verify.yml` | ✅ 2026-05-01 (`bebca2c`) | Same Monday 09:30 UTC slot, just different scheduler. |
| `predict-and-render.yml` | ✅ 2026-05-01 | Cron dropped to `45 2,8,14,20 * * *` (every 6h, paired with collect) on 2026-05-04 once temp predict moved to hourly per lead — see schedule rationale above. Cron itself untouched on 2026-05-07's GEM-driven shift; only the upstream collect + s3-collect timestamps moved. |

## Cost

Cloudflare Workers free plan: 100k requests/day. This Worker uses 4 of the
5 cron triggers the free tier allows, plus the webhook-driven dispatches —
well under 100 requests/day. Zero cost for the foreseeable future.
