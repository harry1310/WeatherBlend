# weatherblend-scheduler

Cloudflare Worker that fires GitHub Actions workflows on a cron schedule via a
GitHub App. Built to replace the unreliable GitHub-side `schedule:` trigger on
`collect.yml` — GitHub's scheduler routinely delays runs by 15+ minutes and
occasionally skips them entirely when the runner pool is busy, which means
collect cycles miss the Open-Meteo `previous_day_N` refresh window. Cloudflare
Workers cron triggers fire on time.

## What it does

```
Cloudflare cron (15 2,8,14,20 * * *)
        │
        ▼
weatherblend-scheduler (this Worker)
        │  1. Mint 10-min JWT signed with the GitHub App's private key
        │  2. Exchange JWT for a 1h installation access token
        │  3. POST /repos/.../actions/workflows/collect.yml/dispatches
        ▼
GitHub Actions runs collect.yml as if a human had clicked "Run workflow"
```

Same workflow, same code path; only the trigger source changes. Once
Cloudflare has demonstrated it fires on time for a few days, the GitHub-side
`schedule:` trigger gets removed from `collect.yml` and the Cloudflare cron
becomes the sole driver.

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
| `GH_WORKFLOW_FILE` | `collect.yml` | The workflow to dispatch on each cron fire. |
| `GH_REF` | `main` | Branch the workflow runs against. |

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

1. **Manual HTTP trigger** — the Worker exposes `POST /dispatch` so you can
   force a workflow_dispatch from anywhere:

   ```bash
   curl -X POST https://weatherblend-scheduler.rhcslater.workers.dev/dispatch
   ```

   200 means the dispatch went through — go check the GitHub Actions tab to
   see `collect` running. 500 means an auth or config error; the response
   body has the diagnostic message and `wrangler tail` shows the full log
   line.

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

## Cutover plan

1. Deploy this Worker. Verify `POST /dispatch` lands a `collect` run in
   GitHub Actions.
2. Let the Cloudflare cron run alongside the GitHub cron for 48–72h. Both
   fire at `15 2,8,14,20 UTC`. `collect` is idempotent (writes by
   ValidTime, dedups), so duplicate runs cost a few extra Open-Meteo calls
   but don't pollute data.
3. Compare timing: Cloudflare should land within a minute of `:15`, GitHub
   often drifts. Check `wrangler tail` vs the GitHub Actions run start times.
4. If Cloudflare is reliably on time, remove the `schedule:` block from
   `.github/workflows/collect.yml` (keep `workflow_dispatch:`). Keep an eye
   on it for a week.
5. Repeat for `predict-and-render.yml` and `verify.yml` once you're confident
   the pattern is solid.

## Cost

Cloudflare Workers free plan: 100k requests/day, 3 cron schedules included.
This Worker uses 1 cron, ~4 requests/day. Zero cost for the foreseeable
future.
