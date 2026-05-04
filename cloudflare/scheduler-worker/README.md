# weatherblend-scheduler

Cloudflare Worker that fires GitHub Actions workflows on a cron schedule via a
GitHub App. Built to replace the unreliable GitHub-side `schedule:` trigger —
GitHub's scheduler routinely delays runs by 15+ minutes and occasionally skips
them entirely when the runner pool is busy, which means scheduled work
silently slips. Cloudflare Workers cron triggers fire on time.

## What it does

```
Cloudflare crons                              Workflow dispatched
─────────────────────────                     ──────────────────────
30 2,8,14,20 * * *  (every 6h, :30)      →   collect.yml
0 3,9,15,21 * * *   (every 6h, 30 after collect)  →   predict-and-render.yml
0 12 * * *          (daily 12:00 UTC)    →   era5-refresh.yml
30 9 * * MON,THU    (Mon + Thu 09:30 UTC) →   verify.yml
        │
        ▼
weatherblend-scheduler (this Worker)
        │  1. Pick workflow from event.cron via WORKFLOW_FOR_CRON
        │  2. Mint 10-min JWT signed with the GitHub App's private key
        │  3. Exchange JWT for a 1h installation access token
        │  4. POST /repos/.../actions/workflows/<file>/dispatches
        ▼
GitHub Actions runs the workflow as if a human had clicked "Run workflow"
```

Same workflows, same code paths; only the trigger source changes. Each
workflow's GitHub-side `schedule:` block gets removed once the Cloudflare
cron has been fired in production for at least a couple of cycles.

## Schedule rationale

| Cron | Workflow | Why this time |
|------|----------|---------------|
| `30 2,8,14,20 * * *` | `collect.yml` | Every 6h, offset `:30` past the hour. Was `:15` until 2026-05-04; shifted +15 to give Open-Meteo more ingestion headroom for late ECMWF cycles (06z sometimes finishes ingesting ~07:30 UTC, which an `08:15` collect would miss). Recent runs complete in 2.5–3.5 min so the 30-min gap to predict still has plenty of slack. |
| `45 2,8,14,20 * * *` | `predict-and-render.yml` | Every 6h, offset `:45` past each collect's `:15`. The 30-min lag past collect lets the R2 push settle before predict reads. Was `45 */2 * * *` (every 2h) until 2026-05-04 — but the on-disk forecast tree only changes when collect runs (which is every 6h), so 8 of every 12 daily predicts were reading IDENTICAL inputs to the previous run and producing IDENTICAL outputs. Per-cycle output is rich enough to justify the 6h cadence: temp + precip emit 24 hourly forecasts per lead bucket, dry-window emits per-day. See `data/reports/schedule_proposal_2026-05-04.md` for the full reasoning. |
| `0 12 * * *` | `era5-refresh.yml` | ECMWF publishes ERA5T daily around 09–10 UTC; Open-Meteo ingests within a few hours. 12:00 UTC is past both, so the daily refresh always lands on Open-Meteo's freshest data instead of catching ECMWF mid-publish (which writes null partitions). The refresh itself pulls a 14-day rolling window, so any null partitions left by older runs get backfilled as ECMWF catches up. |
| `30 9 * * MON,THU` | `verify.yml` | Twice-weekly Mon + Thu 09:30 UTC. Doubled from weekly-Mon on 2026-05-04: a freshly retrained champion was waiting up to 7 days for its first verify rows (cron) plus 5 days (ERA5 latency) before showing on the Models page; cutting cron lag in half cuts that to 3-4 days. **Use day-name aliases (`MON`, `THU`) not numbers — Cloudflare cron uses 1=Sunday (not POSIX 0=Sunday), so `* * 1,4` was firing Sun+Wed instead of Mon+Thu when first deployed 2026-05-04 (caught + fixed same day, see commit history for the bug).** |

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

The Worker reads three secrets at fire time. Set them via **wrangler** or the
Cloudflare dashboard (**Workers & Pages → weatherblend-scheduler → Settings →
Variables and Secrets**):

| Name | Value |
|------|-------|
| `GH_APP_ID` | The App ID (top of the GitHub App's settings page; integer). |
| `GH_APP_INSTALLATION_ID` | The Installation ID (visible in the install URL `/installations/<n>`; integer). |
| `GH_APP_PRIVATE_KEY` | Full contents of the `.pem` file generated when you created the App. PKCS#1 (`BEGIN RSA PRIVATE KEY`) or PKCS#8 (`BEGIN PRIVATE KEY`) both work — the Worker handles either format. |

Plain (non-secret) configuration lives in `wrangler.toml` under `[vars]`:

| Name | Default | Notes |
|------|---------|-------|
| `GH_REPO` | `harry1310/WeatherBlend` | `{owner}/{name}` form. |
| `GH_REF` | `main` | Branch the workflow runs against. |

The workflow file isn't a plain var — it's selected per-cron in the worker
via `WORKFLOW_FOR_CRON` in `src/index.ts` (see "Adding a new schedule" above).

## Required GitHub App permissions

When creating the App: **Repository permissions → Actions → Read and write**.
That's the only permission needed. Webhooks unchecked. Install on the
WeatherBlend repo only.

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
| `predict-and-render.yml` | ✅ 2026-05-01 | Cron dropped to `45 2,8,14,20 * * *` (every 6h, paired with collect) on 2026-05-04 once temp predict moved to hourly per lead — see schedule rationale above. |

## Cost

Cloudflare Workers free plan: 100k requests/day, 3 cron schedules included.
This Worker uses 1 cron, ~4 requests/day. Zero cost for the foreseeable
future.
