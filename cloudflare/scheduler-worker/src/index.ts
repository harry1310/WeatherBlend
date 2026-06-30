/**
 * weatherblend-scheduler — fires GitHub Actions workflows on a cron schedule.
 *
 * Auth flow on every fire:
 *   1. Mint a 10-minute JWT signed with the GitHub App's RSA private key
 *      (RS256, RSASSA-PKCS1-v1_5 + SHA-256 via the Workers Web Crypto API).
 *   2. Exchange the JWT for an installation access token via
 *      `POST /app/installations/{id}/access_tokens`. Tokens last ~1h; we
 *      don't cache them — fire path is rare, simpler is better.
 *   3. Use the installation token to POST a workflow_dispatch event:
 *      `POST /repos/{owner}/{repo}/actions/workflows/{file}/dispatches`.
 *
 * Multiple cron schedules → multiple workflows: the worker looks up which
 * workflow file to dispatch using the cron expression that fired
 * (event.cron). WORKFLOW_FOR_CRON below is the single source of truth and
 * MUST stay in sync with [triggers] crons in wrangler.toml — same strings
 * verbatim. Adding a new schedule = add the cron to wrangler.toml AND the
 * workflow filename to this map.
 *
 * Two ways the Worker can run:
 *   - `scheduled()` for the cron trigger (production path).
 *   - `fetch()` exposes a manual trigger at `POST /dispatch?workflow=NAME`
 *     for testing. The route is unauthenticated; if you don't want randos
 *     firing workflows, gate it on a shared-secret header before flipping
 *     this on.
 */

/** A scheduled dispatch target. `repo` is the GitHub `owner/name` for the
 * workflow we want to fire; if omitted, the worker falls back to the
 * default `GH_REPO` env var (WeatherBlend). This lets a single worker
 * fire workflows across multiple repos without duplicating GH App auth —
 * the same App can be installed on each target repo and the same
 * Installation ID covers them all.
 *
 * `inputs` are workflow_dispatch inputs (string-typed in the GH API
 * regardless of the workflow's declared types). Used by the noon tick
 * to pass `train_after_refresh` to previous-runs-refresh — see the
 * Sunday branch in scheduled() below.
 */
import { lastFireInWindow } from "./cron";

type Dispatch = { workflow: string; repo?: string; inputs?: Record<string, string> };

/** Each cron can dispatch ONE OR MORE workflows. Multi-workflow tick lets
 * us stay within Cloudflare's free-tier limit of 5 cron triggers per
 * Worker while still fanning out to additional workflows on the same
 * schedule. Workflows on the same tick run in parallel (no shared
 * concurrency lock between repos), so cross-repo fan-out is safe as long
 * as the workflows don't write the same R2 prefixes.
 */
const WORKFLOW_FOR_CRON: Record<string, Dispatch[]> = {
  // collect.yml — Open-Meteo + ERA5 + METAR + EA + MO DataHub. Its
  // completion chains predict-4a (handleWorkflowRun Hop C).
  "45 2,8,14,20 * * *": [{ workflow: "collect.yml" }],
  // s3-collect.yml — raw S3 exact-runtime cycles for the 2d/3d blenders.
  // Its completion chains predict.yml (handleWorkflowRun Hop D), and
  // predict's completion chains render-site.yml (Hop F). The
  // predict + render split landed 2026-05-26 as a Phase 3 prereq so
  // predict-3f (WeatherProbabilistic) can sandwich between them and
  // ship 3f data same-cycle. predict-and-render.yml is retained as a
  // fallback (revert path).
  //
  // predict-4a, predict, and render-site are no longer cron'd — they
  // hang off the two completions above. That frees crons (Cloudflare
  // free tier caps at 5) and makes the dependency order explicit
  // instead of leaning on fixed-offset timing:
  //   collect ─(success)─► predict-4a
  //   s3-collect ─────────► predict ─────────────────► render-site
  // s3-collect sits at :05 (not :00) so the collect→4a chain — which
  // starts when collect finishes ~HH:50 — gets a clear ~15 min run-room
  // before predict is chained off s3-collect.
  "5 3,9,15,21 * * *": [{ workflow: "s3-collect.yml" }],
  // Noon UTC: ERA5 refresh + Open-Meteo previous_runs refresh. Both pull
  // rolling 14-day windows past Open-Meteo's publish window so neither
  // lands on null partitions; they share the weatherblend-data lock so
  // writes serialise. On a Sunday, previous-runs-refresh's success
  // chains the weekly retrain (handleWorkflowRun Hop A).
  "0 12 * * *":         [
    { workflow: "truth-refresh.yml" },
    { workflow: "previous-runs-refresh.yml" },
  ],
  "30 9 * * MON,THU":   [{ workflow: "verify.yml" }],
};

const WP_REPO = "harry1310/WeatherProbabilistic";

/** One webhook chain hop the watchdog backstops. Mirrors a hop in
 * handleWorkflowRun EXACTLY — same upstream, same conclusion gating,
 * same downstream (incl. F.1's success→predict-3f / otherwise→render
 * fork). The hops themselves stay the primary mechanism (they fire
 * within seconds); this registry only exists so a silently dropped
 * webhook delivery or a swallowed dispatch error is recovered on the
 * next hourly watchdog tick instead of stalling the chain until the
 * next cycle (2026-06-10: predict completed 15:26Z, the F.1 dispatch
 * never happened, and the site sat on its 09:44Z render for 10 hours).
 */
interface ChainEdge {
  /** Hop label from the handleWorkflowRun comment block ("C", "F.1", …). */
  hop: string;
  upstream: { workflow: string; repo?: string };
  /** Expected downstream when the upstream run concluded "success". */
  onSuccess: Dispatch | null;
  /** Expected downstream on any other completion (failure / cancelled —
   * the hops treat both alike). null = the hop expects nothing then. */
  otherwise: Dispatch | null;
  /** Minutes the live chain gets to fire before the watchdog calls the
   * missing downstream a drop. Normal hop latency is seconds; the grace
   * just avoids racing a webhook that's mid-flight at tick time. */
  graceMin: number;
  /** Hop only applies when the upstream run completed on a Sunday (UTC)
   * — the weekly retrain hops. */
  sundayOnly?: boolean;
}

/** Ordered UPSTREAM-FIRST within each chain: the pass below walks this
 * list top-down and suppresses any edge whose upstream is itself being
 * recovered this tick, so a multi-hop outage dispatches ONLY the first
 * broken link — its (Bot-dispatched) completion then re-chains the
 * rest through the normal hops. */
const CHAIN_EDGES: ChainEdge[] = [
  // Hop A — Sunday retrain entry. Success-gated like the live hop.
  { hop: "A", upstream: { workflow: "previous-runs-refresh.yml" },
    onSuccess: { workflow: "retrain-python.yml", repo: WP_REPO, inputs: { force: "true" } },
    otherwise: null, graceMin: 15, sundayOnly: true },
  // Hop B — python → blenders, any completion (partial retrains still sweep).
  { hop: "B", upstream: { workflow: "retrain-python.yml", repo: WP_REPO },
    onSuccess: { workflow: "retrain-blenders.yml", inputs: { force: "true" } },
    otherwise: { workflow: "retrain-blenders.yml", inputs: { force: "true" } },
    graceMin: 15, sundayOnly: true },
  // Hop C — collect success fans out to BOTH Python predicts. Two edges,
  // deliberately not suppressed against each other: parallel branches,
  // not links of one chain.
  { hop: "C", upstream: { workflow: "collect.yml" },
    onSuccess: { workflow: "predict-4a.yml", repo: WP_REPO },
    otherwise: null, graceMin: 10 },
  { hop: "C", upstream: { workflow: "collect.yml" },
    onSuccess: { workflow: "predict-wind.yml", repo: WP_REPO },
    otherwise: null, graceMin: 10 },
  // Hop D — s3-collect → predict, any completion.
  { hop: "D", upstream: { workflow: "s3-collect.yml" },
    onSuccess: { workflow: "predict.yml" },
    otherwise: { workflow: "predict.yml" }, graceMin: 10 },
  // Hop F.1 — predict success → predict-3f; failure → render-site bypass.
  { hop: "F.1", upstream: { workflow: "predict.yml" },
    onSuccess: { workflow: "predict-3f.yml", repo: WP_REPO },
    otherwise: { workflow: "render-site.yml" }, graceMin: 10 },
  // Hop F — predict-3f → render-site, any completion.
  { hop: "F", upstream: { workflow: "predict-3f.yml", repo: WP_REPO },
    onSuccess: { workflow: "render-site.yml" },
    otherwise: { workflow: "render-site.yml" }, graceMin: 10 },
  // Hop E — verify success → render-site.
  { hop: "E", upstream: { workflow: "verify.yml" },
    onSuccess: { workflow: "render-site.yml" },
    otherwise: null, graceMin: 10 },
];

/** Watchdog cron — fires hourly at HH:00. For each registered cron in
 * WORKFLOW_FOR_CRON, it computes whether a slot was expected in the
 * preceding hour and verifies the GH workflow actually ran. Recovers
 * from dropped Cloudflare cron ticks (free tier delivery is best-
 * effort, not guaranteed — single-tick misses happen occasionally;
 * see the 2026-05-27 15:05Z s3-collect miss for the motivating
 * incident). NOT included in WORKFLOW_FOR_CRON because it dispatches
 * via handleWatchdog, not via the generic dispatch path. */
// 2026-06-30: was hourly; now the every-5-min TICK. It drives the radar freshness-gate (runRadarTick)
// on EVERY fire, and the hourly failure-recovery watchdog only at minute===0 (so the watchdog's behaviour
// is unchanged). Must match the last cron in wrangler.toml. Name kept to avoid churn.
const WATCHDOG_CRON = "*/5 * * * *";

// Met Office UK radar (free, anonymous) — the worker only LISTS it (cheap) to detect a new frame; the heavy
// fetch+advect happens in radar-nowcast.yml on GitHub Actions.
const MO_RADAR_BASE = "https://met-office-radar-obs-data.s3.eu-west-2.amazonaws.com/";

export interface Env {
  GH_APP_ID: string;
  GH_APP_INSTALLATION_ID: string;
  GH_APP_PRIVATE_KEY: string;
  GH_REPO: string;
  GH_REF: string;
  /** Shared HMAC secret configured on the GitHub `workflow_run` webhook for
   * the repos we monitor (WeatherBlend + WeatherProbabilistic). Set via
   * `wrangler secret put GH_WEBHOOK_SECRET`. The same value goes into each
   * repo's webhook config (Settings → Webhooks → Add webhook → Secret).
   * One secret across both repos is fine; each webhook signs its own POSTs
   * and we verify the signature in handleWebhook below. */
  GH_WEBHOOK_SECRET: string;
  /** R2 bucket binding (weatherblend) — the radar tick reads the armed flag
   * (`control/radar_armed.json`, written by radar-arm.yml) and the last-frame
   * marker (`control/radar_last_frame`). Configured in wrangler.toml. */
  RADAR: R2Bucket;
}

/** Newest Met Office frame timestamp (YYYYMMDDhhmm) by listing today's + yesterday's prefix. */
async function latestFrameTs(): Promise<string | null> {
  const now = new Date();
  for (const off of [0, 1]) {
    const d = new Date(now.getTime() - off * 86_400_000);
    const pfx = `radar/${d.getUTCFullYear()}/${String(d.getUTCMonth() + 1).padStart(2, "0")}/` +
      `${String(d.getUTCDate()).padStart(2, "0")}/`;
    const res = await fetch(`${MO_RADAR_BASE}?list-type=2&prefix=${pfx}&max-keys=400`);
    if (!res.ok) continue;
    const ts = [...(await res.text()).matchAll(/<Key>radar\/[^<]*?\/(\d{12})_/g)].map((m) => m[1]);
    if (ts.length) return ts.sort().at(-1)!;
  }
  return null;
}

/** Radar tick: dispatch the nowcast iff armed AND a frame newer than the last-processed marker exists.
 * FULLY defensive — any error is swallowed so a radar problem can NEVER break the production watchdog
 * (collect/predict/retrain recovery) that shares this scheduled handler. */
async function runRadarTick(env: Env): Promise<void> {
  try {
    const armedObj = await env.RADAR.get("control/radar_armed.json");
    if (!armedObj) return;                                 // never armed
    const armedUntil = Number(JSON.parse(await armedObj.text()).armed_until) || 0;
    if (Date.now() / 1000 >= armedUntil) return;           // disarmed / window lapsed

    const latest = await latestFrameTs();
    if (!latest) return;
    const lastObj = await env.RADAR.get("control/radar_last_frame");
    const last = lastObj ? (await lastObj.text()).trim() : "";
    if (latest <= last) return;                            // no new frame since last dispatch

    console.log(`radar: new frame ${latest} (was ${last || "none"}) — dispatching radar-nowcast.yml`);
    await dispatchWorkflow(env, "radar-nowcast.yml", env.GH_REPO);
    await env.RADAR.put("control/radar_last_frame", latest);
  } catch (e) {
    console.error("radar tick error (non-fatal — watchdog unaffected):", e);
  }
}

export default {
  async scheduled(event: ScheduledController, env: Env, _ctx: ExecutionContext): Promise<void> {
    // Watchdog tick — separate code path because it doesn't map to a
    // single workflow dispatch. Scans every registered cron (did the
    // preceding hour's slots actually fire on GH?) and every chain
    // edge (did each completed upstream get its downstream?), and
    // dispatches anything that's missing.
    if (event.cron === WATCHDOG_CRON) {
      // Every 5 min: fire the Bonehill radar nowcast iff armed AND a new Met Office frame landed
      // (cheap — one S3 list + a couple of R2 reads). No-op on disarmed days / stale frames.
      await runRadarTick(env);
      // Hourly failure-recovery watchdog — UNCHANGED behaviour, gated to the top-of-hour tick so it
      // still runs exactly once an hour with a 60-min lookback (no duplicate re-dispatches).
      if (new Date(event.scheduledTime).getUTCMinutes() === 0) {
        await runWatchdog(env, event.scheduledTime, /*lookbackMin*/ 60, /*dryRun*/ false);
      }
      return;
    }

    const targets = WORKFLOW_FOR_CRON[event.cron];
    if (!targets || targets.length === 0) {
      // Unknown cron string means wrangler.toml and this file have drifted
      // out of sync. Throwing surfaces it as a failed Cloudflare scheduled
      // run instead of a silent miss.
      throw new Error(
        `unknown cron expression: '${event.cron}'. Update WORKFLOW_FOR_CRON in src/index.ts.`,
      );
    }
    // The Sunday retrain chain is driven entirely from handleWorkflowRun
    // (the workflow_run webhook handler), not from here. This cron just
    // dispatches the noon refresh; when previous-runs-refresh completes
    // on a Sunday the worker chains retrain-python, and on retrain-python's
    // completion it chains retrain-blenders. Keeping every hop in the
    // worker means one place owns the cadence.
    //
    // Fire all dispatches in parallel — they go to (potentially) different
    // repos and don't depend on each other. If one fails the others still
    // run; we surface aggregate failure at the end so a partial outage is
    // visible in `wrangler tail` rather than swallowed by an early throw.
    const results = await Promise.allSettled(
      targets.map((t) => {
        const repo = t.repo ?? env.GH_REPO;
        const inputsLog = t.inputs ? ` inputs=${JSON.stringify(t.inputs)}` : "";
        console.log(
          `scheduled: cron='${event.cron}' → ${repo}/${t.workflow}${inputsLog} ` +
          `(scheduledTime=${new Date(event.scheduledTime).toISOString()})`,
        );
        return dispatchWorkflow(env, t.workflow, repo, t.inputs);
      }),
    );
    const failures = results
      .map((r, i) => ({ r, t: targets[i] }))
      .filter((x) => x.r.status === "rejected");
    if (failures.length > 0) {
      const messages = failures.map(
        (f) => `${f.t.repo ?? env.GH_REPO}/${f.t.workflow}: ${(f.r as PromiseRejectedResult).reason}`,
      );
      throw new Error(`some dispatches failed: ${messages.join("; ")}`);
    }
  },

  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const knownWorkflows = new Map<string, Dispatch>();
    for (const ts of Object.values(WORKFLOW_FOR_CRON)) {
      for (const t of ts) knownWorkflows.set(t.workflow, t);
    }

    // Serve the live radar nowcast JSON from R2 with CORS. The static Pages site fetches this cross-origin
    // (Pages Functions didn't compile in the deploy); this reuses the worker's public URL + RADAR binding.
    if (url.pathname === "/radar/nowcast/bonehill.json") {
      const cors: Record<string, string> = { "access-control-allow-origin": "*", "cache-control": "no-store" };
      if (request.method === "OPTIONS") return new Response(null, { headers: cors });
      try {
        const obj = await env.RADAR.get("radar/nowcast/bonehill.json");
        if (!obj) return new Response("{}", { status: 404, headers: { "content-type": "application/json", ...cors } });
        return new Response(obj.body, { headers: { "content-type": "application/json", ...cors } });
      } catch (e: unknown) {
        const m = e instanceof Error ? e.message : String(e);
        return new Response(`{"error":${JSON.stringify(m)}}`, { status: 500, headers: { "content-type": "application/json", ...cors } });
      }
    }

    if (url.pathname === "/dispatch" && request.method === "POST") {
      const requested = url.searchParams.get("workflow") ?? "collect.yml";
      const target = knownWorkflows.get(requested);
      if (!target) {
        return new Response(
          `unknown workflow: '${requested}'. Known: ${[...knownWorkflows.keys()].join(", ")}\n`,
          { status: 400 },
        );
      }
      try {
        const repo = target.repo ?? env.GH_REPO;
        await dispatchWorkflow(env, target.workflow, repo);
        return new Response(`dispatched ${repo}/${target.workflow}\n`, { status: 200 });
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e);
        return new Response(`error: ${message}\n`, { status: 500 });
      }
    }
    // GitHub `workflow_run` webhook → open / close a CI-failure issue in
    // the repo where the workflow ran. One endpoint serves all repos that
    // configure a webhook pointing at it (WeatherBlend +
    // WeatherProbabilistic at the time of writing). See handleWorkflowRun
    // below for the open/auto-close behaviour.
    if (url.pathname === "/github-webhook" && request.method === "POST") {
      return await handleWebhook(request, env);
    }
    // Manual watchdog trigger. `?dry-run=1` logs what it WOULD dispatch
    // without firing — useful for verifying the parser + GH lookups
    // before the scheduled tick. `?lookback-minutes=N` lets the caller
    // simulate a wider missed-tick window than the default 60 min (e.g.
    // run with 360 to recover all slots from the last 6 hours after the
    // worker's been down).
    if (url.pathname === "/watchdog" && request.method === "POST") {
      const dryRun = url.searchParams.get("dry-run") === "1";
      const lookbackMin = parseInt(url.searchParams.get("lookback-minutes") ?? "60", 10);
      const now = Date.now();
      try {
        const summary = await runWatchdog(env, now, lookbackMin, dryRun);
        return new Response(summary + "\n", { status: 200 });
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e);
        return new Response(`error: ${message}\n`, { status: 500 });
      }
    }
    return new Response(
      `weatherblend-scheduler — POST /dispatch?workflow=<filename> to fire manually.\n` +
      `POST /watchdog[?dry-run=1][&lookback-minutes=N] to run the cron-miss watchdog manually.\n` +
      `POST /github-webhook for GitHub workflow_run events (HMAC-signed).\n` +
      `Known workflows: ${[...knownWorkflows.keys()].join(", ")}\n`,
      { status: 200 },
    );
  },
};

// ---- GitHub workflow_run webhook ------------------------------------------

/**
 * Handle a GitHub webhook delivery. Verifies the HMAC signature against
 * `GH_WEBHOOK_SECRET`, parses the event, and dispatches `workflow_run`
 * completions to <see cref="handleWorkflowRun"/>. Other event types are
 * ignored (returning 200) so we don't 4xx GitHub when it sends events we
 * don't care about (e.g. `ping` on initial webhook setup).
 */
async function handleWebhook(request: Request, env: Env): Promise<Response> {
  const eventName = request.headers.get("X-GitHub-Event") ?? "";
  const signature = request.headers.get("X-Hub-Signature-256") ?? "";
  // Read body as text first so we can verify the signature against the EXACT
  // bytes GitHub sent. JSON-parsing first would lose whitespace and make the
  // HMAC comparison fail.
  const body = await request.text();

  if (!signature) {
    return new Response("missing X-Hub-Signature-256\n", { status: 401 });
  }
  const ok = await verifyWebhookSignature(env.GH_WEBHOOK_SECRET, body, signature);
  if (!ok) {
    console.warn(`webhook: signature mismatch (event=${eventName})`);
    return new Response("signature mismatch\n", { status: 401 });
  }

  // GitHub fires a `ping` on webhook creation — acknowledge it so the UI
  // shows the green "Last delivery" tick. No payload action needed.
  if (eventName === "ping") {
    return new Response("pong\n", { status: 200 });
  }

  if (eventName !== "workflow_run") {
    // We only subscribe to `workflow_run` in the webhook config, but in
    // case someone widens the subscription, ignore other events
    // gracefully rather than 400.
    return new Response(`ignored: event=${eventName}\n`, { status: 200 });
  }

  let payload: WorkflowRunPayload;
  try {
    payload = JSON.parse(body) as WorkflowRunPayload;
  } catch (e) {
    console.warn(`webhook: malformed JSON: ${e}`);
    return new Response("malformed JSON\n", { status: 400 });
  }

  try {
    const summary = await handleWorkflowRun(env, payload);
    return new Response(`${summary}\n`, { status: 200 });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`webhook: handleWorkflowRun failed: ${message}`);
    return new Response(`handler error: ${message}\n`, { status: 500 });
  }
}

/**
 * Open / close a CI-failure issue based on a `workflow_run` event.
 *
 * Algorithm:
 *   - Only act on `action = completed`. In-progress runs are noise.
 *   - Compute issueTitle = "[ci-fail] {workflow_name}". Per-repo issues —
 *     same workflow name in two repos files into each repo independently
 *     because the issue API call uses payload.repository.full_name.
 *   - On `conclusion = failure | timed_out | startup_failure | cancelled`:
 *       Look up open issues with label `ci-failure` matching the title.
 *       If none → create a new issue assigned to the repo owner.
 *       If one exists → comment on it with the new failure's run URL.
 *   - On `conclusion = success`:
 *       Look up open issues with the same title.
 *       If any → close them with a "now passing" comment.
 *   - `cancelled` counts as a failure here. A job that exceeds its
 *     `timeout-minutes` (e.g. a wedged rclone transfer) completes as
 *     `cancelled`, NOT `failure` — predict-4a hung exactly this way on
 *     2026-05-18 and slipped past this alerting silently. No production
 *     workflow uses concurrency `cancel-in-progress: true`, so a
 *     `cancelled` run is never a benign supersede — it's a timeout or a
 *     manual stop. A manual stop filing a `[ci-fail]` issue is a
 *     tolerable false positive (just close it); a hung workflow going
 *     unnoticed is not.
 *   - Other conclusions (skipped, neutral) are still ignored.
 */
async function handleWorkflowRun(env: Env, payload: WorkflowRunPayload): Promise<string> {
  if (payload.action !== "completed") {
    return `ignored: action=${payload.action}`;
  }
  const repo = payload.repository.full_name;
  const wfRun = payload.workflow_run;
  const wfName = wfRun.name;
  const conclusion = wfRun.conclusion;
  const runUrl = wfRun.html_url;
  const branch = wfRun.head_branch ?? "(unknown)";
  const sha = (wfRun.head_sha ?? "").slice(0, 7);

  // ---- Chain-fire gate (2026-05-26) ---------------------------------------
  // All cross-workflow chain hops below fire ONLY when the upstream run was
  // dispatched by THIS worker (via its GH App token), never when a human
  // `gh workflow run` dispatch finishes. Rule: "manual runs do not trigger
  // downstream automation; only the scheduled cron path does."
  //
  // Signal: triggering_actor.type is "Bot" for the worker's GH App; "User"
  // for `gh workflow run` from a developer. The CI-failure issue logic
  // below the chain section is INTENTIONALLY left ungated — a manual run
  // that fails should still file an issue.
  //
  // Without this gate, every `gh workflow run retrain-python.yml -f
  // phases=3f` during weekday debugging chains a retrain-blenders run via
  // Hop B; multiplied across the 6 hops this is what produced the
  // chained-run pile-up on 2026-05-26 (5 retrain-blenders runs from 5
  // manual retrain-python dispatches, plus the cancelled-pile-up that
  // blocked GH Actions on this repo).
  const actorType = wfRun.triggering_actor?.type;
  const actorLogin = wfRun.triggering_actor?.login ?? "(unknown)";
  const isWorkerDispatched = actorType === "Bot";

  // ---- Completion-driven workflow chains ------------------------------
  // The worker chains workflows off each other's workflow_run completion
  // so a dependent job runs as soon as — and only after — its input is
  // ready, instead of racing a fixed-offset cron. Seven hops:
  //   A   previous-runs-refresh ─(success, Sunday)─► retrain-python
  //   B   retrain-python        ─(any completion)──► retrain-blenders
  //   C   collect               ─(success)─────────► predict-4a
  //   D   s3-collect            ─(any completion)──► predict
  //   E   verify                ─(success)─────────► render-site
  //   F.1 predict               ─(success)─────────► predict-3f
  //                             ─(failure)─────────► render-site (bypass)
  //   F   predict-3f            ─(any completion)──► render-site
  //
  // EVERY hop below is gated on the upstream run being worker-dispatched
  // (triggering_actor.type === "Bot"). Manual `gh workflow run` dispatches
  // (type === "User") fall straight through to the CI-failure issue logic
  // without triggering downstream chains. See the gate block above.
  //
  // A+B are the weekly retrain — serial so retrain-blenders' 4b mint
  // reads this cycle's fresh 4a rather than racing it. C+D+F.1+F are
  // the 6-hourly predict path; D+F.1+F replaced the fused
  //
  // Predict chain (C) used to fan to predict-4a + predict-5a; 5a was
  // retired 2026-05-26.
  // predict-and-render workflow with split predict + predict-3f +
  // render so 3f's stage-2 (in WeatherProbabilistic) lands fresh data
  // on R2 between predict and render. E still redraws the site after
  // the twice-weekly verify run. predict-and-render.yml is RETAINED
  // in the repo as a fallback — if the split breaks, point Hop D
  // back at predict-and-render.yml and disable F.1+F as a two-block
  // revert.
  // Every hop dispatches via workflow_dispatch — the same path the
  // cron uses, so no GitHub App permission beyond the actions:write
  // the worker already relies on.

  if (!isWorkerDispatched) {
    console.log(
      `chain hops SKIPPED: triggering_actor=${actorLogin} type=${actorType ?? "?"} — ` +
      `only worker-dispatched (type=Bot) runs trigger downstream chains. ` +
      `Falling through to ci-failure issue logic.`
    );
  } else {

  // Hop A — refresh → python. Gated on a successful refresh (no retrain
  // on stale data) AND on it being a Sunday: previous-runs-refresh runs
  // daily, so the Sunday test (at completion, ~12:15 UTC, same day it
  // started) is what scopes the retrain to once a week.
  if ((wfRun.path ?? "").endsWith("/previous-runs-refresh.yml")
      && conclusion === "success"
      && new Date().getUTCDay() === 0) {
    try {
      await dispatchWorkflow(env, "retrain-python.yml", "harry1310/WeatherProbabilistic", { force: "true" });
      console.log("retrain chain: previous-runs-refresh success (Sunday) → dispatched retrain-python");
    } catch (e) {
      console.error(`retrain chain: failed to dispatch retrain-python: ${e}`);
    }
  }

  // Hop B — python → blenders. Sunday-gated to match Hop A: the auto-
  // retrain chain (Hop A → retrain-python → here → retrain-blenders)
  // only fires once a week, and Hop B must NOT side-effect into
  // retrain-blenders when retrain-python is dispatched manually
  // (e.g. `gh workflow run retrain-python.yml -f phases=3f` during
  // weekday debugging). Pre-2026-05-26 this was any-day any-completion
  // and chained a retrain-blenders run off every manual dispatch — 5
  // wasted runs during the 3f stand-up alone.
  //
  // Any-completion still: on Sundays a partial-failure retrain-python
  // must NOT skip the .NET sweep — 4b mints from prior 4a, and the
  // non-4b blenders are independent of 4a anyway.
  const isSunday = new Date().getUTCDay() === 0;
  if ((wfRun.path ?? "").endsWith("/retrain-python.yml") && isSunday) {
    try {
      await dispatchWorkflow(env, "retrain-blenders.yml", "harry1310/WeatherBlend", { force: "true" });
      console.log(`retrain chain: retrain-python ${conclusion} → dispatched retrain-blenders (Sunday)`);
    } catch (e) {
      console.error(`retrain chain: failed to dispatch retrain-blenders: ${e}`);
    }
  } else if ((wfRun.path ?? "").endsWith("/retrain-python.yml")) {
    console.log(
      `retrain chain: retrain-python ${conclusion} on non-Sunday — skipped retrain-blenders dispatch ` +
      `(manual retrain-python runs do not chain; the Sunday auto-retrain does).`
    );
  }

  // Hop C — collect → 4a predict + Python wind predicts. Both predict-4a
  // and predict-wind (WeatherProbabilistic; renamed from
  // predict-wind-direction 2026-06-10) consume the
  // Open-Meteo forecasts collect.yml just pushed to R2; they have no
  // dependency on each other so they fire in parallel. predict-4a runs
  // BART per-cell precip; predict-wind runs BOTH Python wind
  // models — wind_mvn (MLP MVN, direction-only since 2026-06-09) and
  // wind_speed_lgb (quantile-LGB + cross-conformal CQR speed point +
  // 90% band, moved from .NET 2026-06-10) — as steps in one job, so no
  // extra dispatch is needed here. Phase 2 of WIND_BLENDER_PLAN added
  // the wind-direction dispatch (2026-05-28). Both gated on collect
  // success — a failed collect leaves stale data.
  // Path check uses the leading slash so `s3-collect.yml` (Hop D)
  // doesn't also match here.
  if ((wfRun.path ?? "").endsWith("/collect.yml") && conclusion === "success") {
    const results = await Promise.allSettled([
      dispatchWorkflow(env, "predict-4a.yml",
                        "harry1310/WeatherProbabilistic"),
      dispatchWorkflow(env, "predict-wind.yml",
                        "harry1310/WeatherProbabilistic"),
    ]);
    for (const [idx, wf] of (["predict-4a.yml", "predict-wind.yml"] as const).entries()) {
      const r = results[idx];
      if (r.status === "fulfilled") {
        console.log(`predict chain: collect success → dispatched ${wf}`);
      } else {
        console.error(`predict chain: failed to dispatch ${wf}: ${r.reason}`);
      }
    }
  }

  // Hop D — s3-collect → predict. The raw S3 exact-runtime cycles
  // s3-collect lands feed the 2d/3d predictors that run inside predict.
  // Chaining guarantees predict starts only after s3-collect finishes
  // — the weatherblend-data concurrency lock already enforced that, but
  // the chain makes the dependency explicit and frees a cron.
  //
  // Flipped 2026-05-26 from dispatching predict-and-render.yml (the
  // fused predict + render workflow) to predict.yml (predict-only).
  // The render step now lives in render-site.yml and is chained off
  // predict's completion (Hop F below). This split is the Phase 3
  // prereq — lets predict-3f sandwich between predict (Hop D) and
  // render-site (Hop F) for same-cycle 3f data on the site.
  // predict-and-render.yml is retained in the repo as a fallback.
  //
  // Fires on ANY completion: a failed s3-collect should still try to
  // predict + render off the most recent exact-runtime data rather
  // than leave the site stale.
  if ((wfRun.path ?? "").endsWith("/s3-collect.yml")) {
    try {
      await dispatchWorkflow(env, "predict.yml", "harry1310/WeatherBlend");
      console.log(`predict chain: s3-collect ${conclusion} → dispatched predict`);
    } catch (e) {
      console.error(`predict chain: failed to dispatch predict: ${e}`);
    }
  }

  // Hop E — verify → render-site. verify computes the rolling MAE / Brier
  // / drift metrics; render-site redraws the site so they surface as soon
  // as they're computed. Gated on success — a failed verify means the
  // metrics aren't fresh, so re-rendering would just redeploy the prior
  // site. (Was a GitHub-native `on: workflow_run` trigger on
  // render-site.yml; moved here 2026-05-22 so every workflow-to-workflow
  // hop lives in one place.)
  if ((wfRun.path ?? "").endsWith("/verify.yml") && conclusion === "success") {
    try {
      await dispatchWorkflow(env, "render-site.yml", "harry1310/WeatherBlend");
      console.log("render chain: verify success → dispatched render-site");
    } catch (e) {
      console.error(`render chain: failed to dispatch render-site: ${e}`);
    }
  }

  // Hop F.1 — predict → predict-3f (success) | render-site (failure).
  // Phase 3f sandwich, shipped 2026-05-26: on predict.yml success the
  // chain detours through WeatherProbabilistic's predict-3f.yml so
  // fresh π+3f data hits R2 before render reads. On predict failure
  // predict-3f has no fresh stage-1 to consume — dispatching it would
  // just burn ~3 min spinning up to skip — so we bypass and render
  // directly off whatever predictions are current (same defensive
  // posture as Hop D's any-completion gate).
  if ((wfRun.path ?? "").endsWith("/predict.yml")) {
    try {
      if (conclusion === "success") {
        await dispatchWorkflow(env, "predict-3f.yml", "harry1310/WeatherProbabilistic");
        console.log(`predict-3f chain: predict success → dispatched predict-3f`);
      } else {
        await dispatchWorkflow(env, "render-site.yml", "harry1310/WeatherBlend");
        console.log(`render chain: predict ${conclusion} → dispatched render-site (predict-3f bypassed)`);
      }
    } catch (e) {
      // Swallowed by design (a hop failure must not break the webhook
      // response) — the hourly watchdog's CHAIN_EDGES pass is the
      // backstop that re-fires a hop lost here (2026-06-10 incident:
      // this exact hop dropped after the 15:16Z predict and the site
      // sat stale for 10 hours).
      console.error(`Hop F.1: failed to dispatch downstream after predict: ${e}`);
    }
  }

  // Hop F — predict-3f → render-site. Closes the sandwich opened by
  // F.1 above. Fires on ANY completion: a failed predict-3f still
  // triggers a re-render so the site picks up the fresh non-3f
  // predictions (3a/3o/4a/4b/etc.) that predict.yml just landed. The
  // 3f card on the rainfall_amount page will simply show its prior
  // cycle's numbers until the next successful predict-3f.
  if ((wfRun.path ?? "").endsWith("/predict-3f.yml")) {
    try {
      await dispatchWorkflow(env, "render-site.yml", "harry1310/WeatherBlend");
      console.log(`render chain: predict-3f ${conclusion} → dispatched render-site`);
    } catch (e) {
      console.error(`render chain: failed to dispatch render-site after predict-3f: ${e}`);
    }
  }

  } // end of isWorkerDispatched chain-hops block (gate opened above Hop A)

  // Every hop falls through — the ci-failure issue logic below still runs
  // for the workflow_run that triggered this handler. CI-failure issues
  // ARE filed for manual-dispatch failures too (the gate above scopes
  // chain-dispatches, not failure visibility).

  const issueTitle = `[ci-fail] ${wfName}`;
  // `cancelled` is included to catch timeout-minutes HANGS: GitHub completes a run
  // that breaches `timeout-minutes` as `cancelled`, not `failure` (predict-4a hung
  // exactly this way on 2026-05-18 and slipped past alerting silently). The catch: a
  // workflow with concurrency `cancel-in-progress: true` ALSO reports a benign
  // supersede as `cancelled` — render-site (group `weatherblend-site`, added after
  // the original comment claimed "no workflow uses cancel-in-progress") is dispatched
  // off two converging paths (verify→render, predict-3f→render) seconds apart, so the
  // newer run cancels the older. So a `cancelled` is a real failure ONLY when it was
  // NOT superseded by a newer run of the same workflow — checked via
  // wasConcurrencySuperseded below before any issue is filed.
  const failureConclusions = new Set(["failure", "timed_out", "startup_failure", "cancelled"]);
  const isFailure = conclusion !== null && failureConclusions.has(conclusion);
  const isSuccess = conclusion === "success";

  if (!isFailure && !isSuccess) {
    return `ignored: conclusion=${conclusion}`;
  }

  // Mint a fresh installation token for the target repo. Same App auth
  // path used by dispatchWorkflow; works for any repo where the App is
  // installed (WeatherBlend + WeatherProbabilistic both qualify).
  const jwt = await mintAppJwt(env.GH_APP_ID, env.GH_APP_PRIVATE_KEY);
  const installationToken = await exchangeForInstallationToken(jwt, env.GH_APP_INSTALLATION_ID);

  // Search for existing OPEN issues with this exact title + label.
  // GitHub's issue search API is eventual-consistency on indexing — for
  // the open/close flow we want strict consistency, so use the per-repo
  // issues list endpoint with state=open + labels filter instead.
  const existing = await listOpenCiFailureIssues(installationToken, repo);
  const matching = existing.filter((i) => i.title === issueTitle);

  // Benign-supersede guard: a `cancelled` run that a NEWER run of the same workflow
  // superseded (concurrency cancel-in-progress) is not a hang — don't file. A real
  // timeout-minutes hang has no newer run during its life, so it still files.
  if (isFailure && conclusion === "cancelled"
      && await wasConcurrencySuperseded(installationToken, repo, wfRun)) {
    return `ignored: ${wfName} cancelled but superseded by a newer run (benign concurrency supersede)`;
  }

  if (isFailure) {
    if (matching.length === 0) {
      const body = renderFailureIssueBody(wfName, branch, sha, runUrl, payload.workflow_run.run_number);
      await createIssue(installationToken, repo, issueTitle, body, ["ci-failure"]);
      return `opened issue [${repo}] ${issueTitle}`;
    } else {
      // Already an open issue — append a comment so consecutive failures
      // accumulate context without spamming new issues.
      const body = renderFailureCommentBody(branch, sha, runUrl, payload.workflow_run.run_number);
      await commentOnIssue(installationToken, repo, matching[0].number, body);
      return `commented on issue [${repo}] #${matching[0].number}`;
    }
  }

  // Success path — close any open ci-failure issue for this workflow.
  if (matching.length === 0) {
    return `success but no open issue to close [${repo}] ${issueTitle}`;
  }
  const body = `Workflow now passing on \`${branch}\` @ \`${sha}\` — closing.\n\nRun: ${runUrl}`;
  await commentOnIssue(installationToken, repo, matching[0].number, body);
  await closeIssue(installationToken, repo, matching[0].number);
  return `closed issue [${repo}] #${matching[0].number}`;
}

/** GitHub Issue (subset of fields we use). */
interface GitHubIssue {
  number: number;
  title: string;
  labels: { name: string }[];
}

/** GitHub `workflow_run` webhook payload (subset). Full schema:
 * https://docs.github.com/en/webhooks/webhook-events-and-payloads#workflow_run */
interface WorkflowRunPayload {
  action: "requested" | "in_progress" | "completed" | string;
  workflow_run: {
    id: number;
    name: string;
    /** Run created + last-updated (≈ completion) timestamps, ISO 8601. Used to
     * detect a benign concurrency supersede: a newer run of the same workflow that
     * STARTED while this run was still in-progress is what cancelled it. */
    created_at?: string;
    updated_at?: string;
    /** Workflow file path, e.g. ".github/workflows/retrain-python.yml". */
    path?: string;
    /** Event that triggered the run, e.g. "repository_dispatch", "schedule". */
    event?: string;
    run_number: number;
    head_branch: string | null;
    head_sha: string | null;
    conclusion: "success" | "failure" | "cancelled" | "skipped" | "timed_out" | "neutral" | "startup_failure" | null;
    html_url: string;
    /** Who dispatched the run. The GH App that this worker authenticates
     * as has `type: "Bot"`; `gh workflow run` (human) dispatches have
     * `type: "User"`. handleWorkflowRun uses this to gate cross-workflow
     * chains — only worker-dispatched runs trigger downstream hops. */
    triggering_actor?: {
      login: string;
      type: "Bot" | "User" | string;
    } | null;
  };
  repository: {
    full_name: string;
    name: string;
    owner: { login: string };
  };
}

/** True when a `cancelled` run was superseded by a NEWER run of the same workflow
 * that started during its lifetime — a benign `concurrency: cancel-in-progress`
 * supersede, NOT a `timeout-minutes` hang. render-site (group `weatherblend-site`) is
 * the one such workflow today: the scheduler dispatches it off two converging paths
 * (verify→render, predict-3f→render) seconds apart, so the newer run cancels the older.
 * A timeout-hang has no newer run during its life (the next cron run is hours away), so
 * it still files. The check is workflow-agnostic — any future cancel-in-progress
 * workflow is covered. On any API error we return false → treat it as a real failure
 * (fail safe: a hang must never go unnoticed). */
async function wasConcurrencySuperseded(
  token: string,
  repo: string,
  wfRun: WorkflowRunPayload["workflow_run"],
): Promise<boolean> {
  const file = (wfRun.path ?? "").split("/").pop();
  if (!file || !wfRun.created_at || !wfRun.updated_at) return false;
  const startedMs = Date.parse(wfRun.created_at);
  const endedMs = Date.parse(wfRun.updated_at);
  const branchQ = wfRun.head_branch ? `&branch=${encodeURIComponent(wfRun.head_branch)}` : "";
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${file}/runs?per_page=20${branchQ}`;
  const response = await fetch(url, {
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
    },
  });
  if (!response.ok) return false;
  const data = (await response.json()) as { workflow_runs?: { id: number; created_at: string }[] };
  // A different run that STARTED within this run's [created, completed] window is what
  // cancelled it. The 15s buffer absorbs webhook-timestamp granularity; a timeout-hang's
  // next run is hours later, so the buffer can't false-positive it as a supersede.
  return (data.workflow_runs ?? []).some(
    (r) => r.id !== wfRun.id
      && Date.parse(r.created_at) > startedMs
      && Date.parse(r.created_at) <= endedMs + 15_000,
  );
}

async function listOpenCiFailureIssues(token: string, repo: string): Promise<GitHubIssue[]> {
  const url = `https://api.github.com/repos/${repo}/issues?state=open&labels=ci-failure&per_page=100`;
  const response = await fetch(url, {
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`list issues failed: ${response.status} ${response.statusText} — ${body}`);
  }
  return (await response.json()) as GitHubIssue[];
}

async function createIssue(
  token: string, repo: string, title: string, body: string, labels: string[],
  assignees: string[] = ["harry1310"],
): Promise<void> {
  // Assign to the repo owner by default — GitHub fires a personal
  // notification on `assigned` events (delivered to email + the
  // notifications inbox) regardless of repo-watch settings, which is
  // what we want for `[ci-fail]` issues. Without assignees, the worker
  // creates the issue silently and the user only sees it if they're
  // watching the repo for issue events. Override `assignees` per-call
  // if other CI failures should ping someone else.
  const url = `https://api.github.com/repos/${repo}/issues`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title, body, labels, assignees }),
  });
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`create issue failed: ${response.status} ${response.statusText} — ${errText}`);
  }
}

async function commentOnIssue(
  token: string, repo: string, issueNumber: number, body: string,
): Promise<void> {
  const url = `https://api.github.com/repos/${repo}/issues/${issueNumber}/comments`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ body }),
  });
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`comment failed: ${response.status} ${response.statusText} — ${errText}`);
  }
}

async function closeIssue(token: string, repo: string, issueNumber: number): Promise<void> {
  const url = `https://api.github.com/repos/${repo}/issues/${issueNumber}`;
  const response = await fetch(url, {
    method: "PATCH",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ state: "closed", state_reason: "completed" }),
  });
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`close issue failed: ${response.status} ${response.statusText} — ${errText}`);
  }
}

function renderFailureIssueBody(
  wfName: string, branch: string, sha: string, runUrl: string, runNumber: number,
): string {
  return [
    `Workflow **${wfName}** failed.`,
    ``,
    `- Branch: \`${branch}\``,
    `- Commit: \`${sha}\``,
    `- Run #${runNumber}: ${runUrl}`,
    ``,
    `This issue auto-closes when the same workflow next succeeds. Subsequent`,
    `failures of this workflow append a comment here rather than opening a`,
    `new issue.`,
  ].join("\n");
}

function renderFailureCommentBody(
  branch: string, sha: string, runUrl: string, runNumber: number,
): string {
  return [
    `Another failure on \`${branch}\` @ \`${sha}\`.`,
    ``,
    `Run #${runNumber}: ${runUrl}`,
  ].join("\n");
}

async function verifyWebhookSignature(
  secret: string, body: string, signatureHeader: string,
): Promise<boolean> {
  // GitHub format: "sha256=<hex>"
  const expected = "sha256=";
  if (!signatureHeader.startsWith(expected)) return false;
  const providedHex = signatureHeader.slice(expected.length);

  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sigBytes = await crypto.subtle.sign("HMAC", key, enc.encode(body));
  const computedHex = bytesToHex(new Uint8Array(sigBytes));

  // Constant-time-ish compare. JS string ===  isn't strictly constant-time
  // but we're verifying a single signature per request and the timing
  // signal is masked by network noise — defensive enough for this
  // attack surface (a 256-bit secret kept off the public web).
  if (computedHex.length !== providedHex.length) return false;
  let diff = 0;
  for (let i = 0; i < computedHex.length; i++) {
    diff |= computedHex.charCodeAt(i) ^ providedHex.charCodeAt(i);
  }
  return diff === 0;
}

function bytesToHex(bytes: Uint8Array): string {
  let hex = "";
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, "0");
  }
  return hex;
}

// ---- workflow dispatch ----------------------------------------------------

// ============================================================================
// Watchdog — recover from dropped Cloudflare cron ticks AND dropped
// webhook chain hops (CHAIN_EDGES above)
// ============================================================================

/** Tolerance applied to the "did this fire?" check. Cloudflare can
 * deliver a cron tick a few seconds late; the GH workflow_dispatch
 * round-trip adds another second or two; we don't want a 5-second
 * lag flagged as a miss. 60 s = generous enough that a clean run
 * always passes, tight enough that a truly missed slot doesn't sneak
 * past on the next watchdog tick (slots are 6 hours apart at worst).
 */
const WATCHDOG_TOLERANCE_MS = 60_000;

/** One missed-slot detection. The watchdog gathers these across all
 * registered crons, then dispatches the missing workflows in parallel
 * and surfaces an aggregate error if any redispatch fails. */
interface MissedSlot {
  expr: string;
  expectedFireMs: number;
  dispatch: Dispatch;
  repo: string;
  lastRunCreatedAtMs: number | null;
  reason: "no-runs" | "latest-too-old" | "latest-in-progress-but-pre-slot";
}

/** Safety margin between the current time and the upper end of the
 * lookback window. Avoids a race against schedules that fire at the
 * same minute as the watchdog (notably `0 12 * * *` noon refresh vs
 * the watchdog at 12:00Z, and any future cron sharing the HH:00
 * minute). With 5 min, a slot at HH:00 is only checked by the
 * (HH+1):00 watchdog tick — by which time the original cron's GH
 * dispatch has had a full hour to register a run.
 *
 * Slots at HH:55 would fall in the gap between two consecutive
 * watchdog ticks' windows, but no cron in WORKFLOW_FOR_CRON fires at
 * :55 (minutes in use: :00 / :05 / :30 / :45). Revisit if a new
 * schedule lands on :50-:59.
 */
const WATCHDOG_SAFETY_MARGIN_MS = 5 * 60_000;

async function runWatchdog(
  env: Env,
  scheduledTime: number,
  lookbackMin: number,
  dryRun: boolean,
): Promise<string> {
  // Look back lookbackMin minutes ending 5 min before now. The 5-min
  // safety margin keeps us from racing against any cron that fires at
  // the same minute as the watchdog: by the time we check a slot, it's
  // had at least 5 minutes of GH-API propagation time.
  //
  // The manual /watchdog endpoint accepts a wider lookback for catch-up
  // scenarios after a worker outage (e.g. ?lookback-minutes=360 covers
  // 6 hours of slots).
  const beforeMs = scheduledTime - WATCHDOG_SAFETY_MARGIN_MS;
  const lookbackMs = beforeMs - lookbackMin * 60_000;
  const missed: MissedSlot[] = [];
  // GH App token mint is expensive (key import + JWT sign + HTTP
  // exchange); reuse one token across every workflow lookup this tick.
  // Same token also feeds the redispatch path below.
  let installationToken: string | null = null;
  const getToken = async (): Promise<string> => {
    if (installationToken) return installationToken;
    const jwt = await mintAppJwt(env.GH_APP_ID, env.GH_APP_PRIVATE_KEY);
    installationToken = await exchangeForInstallationToken(jwt, env.GH_APP_INSTALLATION_ID);
    return installationToken;
  };

  for (const [expr, dispatches] of Object.entries(WORKFLOW_FOR_CRON)) {
    const expected = lastFireInWindow(expr, lookbackMs, beforeMs);
    if (expected === null) continue;   // no slot for this cron in the window

    for (const dispatch of dispatches) {
      const repo = dispatch.repo ?? env.GH_REPO;
      const latest = await ghLatestWorkflowDispatchRun(await getToken(), repo, dispatch.workflow);

      // A run is considered "covers this slot" when its created_at is
      // at or after (expected - tolerance). Tolerance handles small
      // dispatch-latency jitter so a healthy run isn't flagged.
      if (latest !== null) {
        const latestMs = latest.createdAtMs;
        if (latestMs >= expected - WATCHDOG_TOLERANCE_MS) continue;  // healthy
      }

      missed.push({
        expr,
        expectedFireMs: expected,
        dispatch,
        repo,
        lastRunCreatedAtMs: latest?.createdAtMs ?? null,
        reason: latest === null ? "no-runs" : "latest-too-old",
      });
    }
  }

  // ---- Chain-edge pass (2026-06-10) -----------------------------------
  // The cron pass above covers dropped Cloudflare cron ticks; this pass
  // covers dropped WEBHOOK hops — a workflow_run delivery that never
  // arrived, or a dispatch the hop's catch block swallowed. For each
  // registered edge: find the newest COMPLETED worker-dispatched run of
  // the upstream inside this window, work out what the live hop should
  // have dispatched for its conclusion, and check a downstream run was
  // created after it. `recovering` is the only-first-broken-link
  // mechanism: it is seeded with this tick's cron redispatches and grows
  // as edges fire, and any edge whose upstream OR downstream is already
  // in it is skipped — the recovered run's own completion webhook
  // re-chains everything below it.
  const recovering = new Set<string>(
    missed.map((m) => `${m.repo}/${m.dispatch.workflow}`),
  );
  const missedEdges: MissedEdge[] = [];
  for (const edge of CHAIN_EDGES) {
    const upRepo = edge.upstream.repo ?? env.GH_REPO;
    const upKey = `${upRepo}/${edge.upstream.workflow}`;
    if (recovering.has(upKey)) continue;   // upstream being recovered — its completion re-chains
    const up = await ghLatestCompletedDispatchRun(await getToken(), upRepo, edge.upstream.workflow);
    if (up === null) continue;
    if (!up.botTriggered) continue;        // manual runs never chain — watchdog must not "fix" that
    if (up.completedAtMs <= lookbackMs) continue;   // older than this tick's window
    if (scheduledTime - up.completedAtMs < edge.graceMin * 60_000) continue;  // hop may be mid-flight
    if (edge.sundayOnly && new Date(up.completedAtMs).getUTCDay() !== 0) continue;
    const target = up.conclusion === "success" ? edge.onSuccess : edge.otherwise;
    if (target === null) continue;         // hop expects nothing for this conclusion
    const downRepo = target.repo ?? env.GH_REPO;
    const downKey = `${downRepo}/${target.workflow}`;
    if (recovering.has(downKey)) continue; // already dispatched this tick (e.g. render via two paths)
    const latestDown = await ghLatestWorkflowDispatchRun(await getToken(), downRepo, target.workflow);
    if (latestDown !== null && latestDown.createdAtMs >= up.completedAtMs - WATCHDOG_TOLERANCE_MS)
      continue;                            // healthy — downstream fired after the upstream completed
    recovering.add(downKey);
    missedEdges.push({
      hop: edge.hop,
      upstreamKey: upKey,
      upstreamCompletedAtMs: up.completedAtMs,
      dispatch: target,
      repo: downRepo,
      lastDownstreamMs: latestDown?.createdAtMs ?? null,
    });
  }

  const window = `(${new Date(lookbackMs).toISOString()}, ${new Date(beforeMs).toISOString()}]`;
  if (missed.length === 0 && missedEdges.length === 0) {
    const ok = `watchdog: all ${Object.keys(WORKFLOW_FOR_CRON).length} cron schedules + ` +
      `${CHAIN_EDGES.length} chain hops covered for window ${window}.`;
    console.log(ok);
    return ok;
  }

  // Log every miss before dispatching so the diagnostic survives even
  // if a redispatch itself fails. Format mirrors scheduled()'s log
  // line so tail-grep'ing on "missed tick" finds them.
  const lines: string[] = [];
  const verb = dryRun ? "would redispatch" : "redispatching";
  for (const m of missed) {
    const latest = m.lastRunCreatedAtMs === null
      ? "no prior runs"
      : `latest=${new Date(m.lastRunCreatedAtMs).toISOString()}`;
    const line =
      `watchdog: missed tick cron='${m.expr}' → ${m.repo}/${m.dispatch.workflow} ` +
      `expected=${new Date(m.expectedFireMs).toISOString()} ${latest} (${m.reason}); ${verb}`;
    console.log(line);
    lines.push(line);
  }
  for (const m of missedEdges) {
    const latest = m.lastDownstreamMs === null
      ? "no prior runs"
      : `latest=${new Date(m.lastDownstreamMs).toISOString()}`;
    const line =
      `watchdog: dropped chain hop ${m.hop} — ${m.upstreamKey} completed ` +
      `${new Date(m.upstreamCompletedAtMs).toISOString()} but ${m.repo}/${m.dispatch.workflow} ` +
      `never followed (${latest}); ${verb}`;
    console.log(line);
    lines.push(line);
  }

  const total = missed.length + missedEdges.length;
  if (dryRun) {
    return `watchdog: dry-run found ${total} miss(es) in ${window}\n` + lines.join("\n");
  }

  const toDispatch = [
    ...missed.map((m) => ({ repo: m.repo, dispatch: m.dispatch, label: `cron ${m.expr}` })),
    ...missedEdges.map((m) => ({ repo: m.repo, dispatch: m.dispatch, label: `hop ${m.hop}` })),
  ];
  const results = await Promise.allSettled(
    toDispatch.map((d) => dispatchWorkflow(env, d.dispatch.workflow, d.repo, d.dispatch.inputs)),
  );
  const failures = results
    .map((r, i) => ({ r, d: toDispatch[i] }))
    .filter((x) => x.r.status === "rejected");
  if (failures.length > 0) {
    const messages = failures.map(
      (f) => `${f.d.repo}/${f.d.dispatch.workflow} (${f.d.label}): ${(f.r as PromiseRejectedResult).reason}`,
    );
    throw new Error(`watchdog redispatches failed: ${messages.join("; ")}`);
  }
  return `watchdog: redispatched ${total} miss(es) in ${window}\n` + lines.join("\n");
}

/** One dropped-chain-hop detection — the upstream completed but the
 * downstream the live hop should have dispatched never appeared. */
interface MissedEdge {
  hop: string;
  upstreamKey: string;
  upstreamCompletedAtMs: number;
  dispatch: Dispatch;
  repo: string;
  lastDownstreamMs: number | null;
}

/** Most-recent workflow_dispatch-event run for a workflow, regardless
 * of conclusion. Returns null when the workflow has never been
 * dispatched (fresh install) — caller treats that as a miss IFF the
 * window had an expected slot.
 *
 * Only counts event=workflow_dispatch runs: push / pull_request /
 * workflow_run runs are irrelevant for the cron-coverage check and
 * including them would falsely satisfy a missed cron slot whenever
 * someone pushed a code change near the slot time. */
async function ghLatestWorkflowDispatchRun(
  installationToken: string,
  repo: string,
  workflowFile: string,
): Promise<{ createdAtMs: number; status: string; conclusion: string | null } | null> {
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflowFile}/runs` +
    `?event=workflow_dispatch&per_page=1`;
  const response = await fetch(url, {
    headers: {
      "Authorization": `Bearer ${installationToken}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`workflow runs lookup failed for ${repo}/${workflowFile}: ${response.status} — ${body}`);
  }
  const data = (await response.json()) as {
    workflow_runs?: { created_at: string; status: string; conclusion: string | null }[];
  };
  const runs = data.workflow_runs ?? [];
  if (runs.length === 0) return null;
  const r = runs[0];
  return { createdAtMs: Date.parse(r.created_at), status: r.status, conclusion: r.conclusion };
}

/** Newest COMPLETED workflow_dispatch run for a workflow, with the
 * fields the chain-edge pass gates on. Scans the 5 most recent dispatch
 * runs so a newer in-progress run doesn't hide the completed one the
 * edge should anchor on. `completedAtMs` uses `updated_at`, which GH
 * stamps at completion for completed runs. `botTriggered` mirrors
 * handleWorkflowRun's chain-fire gate (triggering_actor.type === "Bot")
 * so the watchdog never chains off a manual `gh workflow run`. */
async function ghLatestCompletedDispatchRun(
  installationToken: string,
  repo: string,
  workflowFile: string,
): Promise<{ completedAtMs: number; conclusion: string | null; botTriggered: boolean } | null> {
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflowFile}/runs` +
    `?event=workflow_dispatch&per_page=5`;
  const response = await fetch(url, {
    headers: {
      "Authorization": `Bearer ${installationToken}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`workflow runs lookup failed for ${repo}/${workflowFile}: ${response.status} — ${body}`);
  }
  const data = (await response.json()) as {
    workflow_runs?: {
      status: string;
      conclusion: string | null;
      updated_at: string;
      triggering_actor?: { login?: string; type?: string } | null;
    }[];
  };
  // Runs come newest-first; the first completed one is the edge anchor.
  const done = (data.workflow_runs ?? []).find((r) => r.status === "completed");
  if (!done) return null;
  return {
    completedAtMs: Date.parse(done.updated_at),
    conclusion: done.conclusion,
    botTriggered: done.triggering_actor?.type === "Bot",
  };
}

async function dispatchWorkflow(
  env: Env,
  workflowFile: string,
  repo: string,
  inputs?: Record<string, string>,
): Promise<void> {
  const jwt = await mintAppJwt(env.GH_APP_ID, env.GH_APP_PRIVATE_KEY);
  const installationToken = await exchangeForInstallationToken(jwt, env.GH_APP_INSTALLATION_ID);

  const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflowFile}/dispatches`;
  // GH API requires `inputs` values to be strings even when the workflow
  // declares them as boolean / number — the serialiser on the workflow
  // side coerces back. Caller passes Record<string, string> already.
  const body: { ref: string; inputs?: Record<string, string> } = { ref: env.GH_REF };
  if (inputs && Object.keys(inputs).length > 0) body.inputs = inputs;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${installationToken}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  // GitHub returns 204 No Content on a successful dispatch — anything else is
  // an error and we want the message in the Worker logs so a misconfiguration
  // shows up on `wrangler tail` rather than vanishing into a green cron run.
  if (response.status !== 204) {
    const body = await response.text();
    throw new Error(`workflow dispatch failed: ${response.status} ${response.statusText} — ${body}`);
  }
  console.log(`dispatched ${repo}/${workflowFile}@${env.GH_REF}`);
}

async function exchangeForInstallationToken(jwt: string, installationId: string): Promise<string> {
  const url = `https://api.github.com/app/installations/${installationId}/access_tokens`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${jwt}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`installation token exchange failed: ${response.status} ${response.statusText} — ${body}`);
  }
  const data = (await response.json()) as { token: string };
  return data.token;
}

// ---- JWT minting via Web Crypto ------------------------------------------

async function mintAppJwt(appId: string, privateKeyPem: string): Promise<string> {
  // Per GitHub: iat backdated 60s to tolerate clock skew between the Worker
  // edge node and GitHub's API; exp at +10 min (the maximum GitHub accepts).
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", typ: "JWT" };
  const payload = { iat: now - 60, exp: now + 10 * 60, iss: appId };

  const encoder = new TextEncoder();
  const headerB64 = base64UrlEncode(encoder.encode(JSON.stringify(header)));
  const payloadB64 = base64UrlEncode(encoder.encode(JSON.stringify(payload)));
  const signingInput = `${headerB64}.${payloadB64}`;

  const key = await importRsaPrivateKey(privateKeyPem);
  const signature = await crypto.subtle.sign(
    { name: "RSASSA-PKCS1-v1_5" },
    key,
    encoder.encode(signingInput),
  );

  return `${signingInput}.${base64UrlEncode(new Uint8Array(signature))}`;
}

/**
 * Imports an RSA private key from a PEM, accepting BOTH formats GitHub may
 * hand you depending on download path:
 *
 *   PKCS#1 (`-----BEGIN RSA PRIVATE KEY-----`) — the default `.pem` you get
 *     when you click "Generate a private key" on the GitHub App settings page.
 *   PKCS#8 (`-----BEGIN PRIVATE KEY-----`)     — what you get if you pre-
 *     converted via `openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt`.
 *
 * Workers' Web Crypto only accepts PKCS#8 directly; PKCS#1 keys get wrapped
 * in a PKCS#8 envelope at runtime so the user never has to think about
 * format compatibility when rotating the App's key.
 */
async function importRsaPrivateKey(pem: string): Promise<CryptoKey> {
  const trimmed = pem.trim();
  let cleanBase64: string;
  let isPkcs1: boolean;

  if (trimmed.includes("BEGIN RSA PRIVATE KEY")) {
    isPkcs1 = true;
    cleanBase64 = trimmed
      .replace(/-----BEGIN RSA PRIVATE KEY-----/, "")
      .replace(/-----END RSA PRIVATE KEY-----/, "")
      .replace(/\s+/g, "");
  } else if (trimmed.includes("BEGIN PRIVATE KEY")) {
    isPkcs1 = false;
    cleanBase64 = trimmed
      .replace(/-----BEGIN PRIVATE KEY-----/, "")
      .replace(/-----END PRIVATE KEY-----/, "")
      .replace(/\s+/g, "");
  } else {
    throw new Error("GH_APP_PRIVATE_KEY: unrecognised PEM. Expected 'BEGIN RSA PRIVATE KEY' (PKCS#1) or 'BEGIN PRIVATE KEY' (PKCS#8).");
  }

  const keyBytes = base64ToBytes(cleanBase64);
  const pkcs8Bytes = isPkcs1 ? wrapPkcs1AsPkcs8(keyBytes) : keyBytes;

  return crypto.subtle.importKey(
    "pkcs8",
    pkcs8Bytes.buffer.slice(pkcs8Bytes.byteOffset, pkcs8Bytes.byteOffset + pkcs8Bytes.byteLength) as ArrayBuffer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

/**
 * Wrap an RSA PKCS#1 RSAPrivateKey blob in a PKCS#8 PrivateKeyInfo envelope
 * so Workers' importKey accepts it. The envelope is fixed shape:
 *
 *   SEQUENCE {
 *     INTEGER 0                                          (version)
 *     SEQUENCE {
 *       OBJECT IDENTIFIER 1.2.840.113549.1.1.1           (rsaEncryption)
 *       NULL                                             (params)
 *     }
 *     OCTET STRING { ...PKCS#1 bytes verbatim... }       (privateKey)
 *   }
 *
 * Lengths use DER long-form because a 2048-bit RSA key is ~1190 bytes and
 * doesn't fit in the 1-byte short form.
 */
function wrapPkcs1AsPkcs8(pkcs1: Uint8Array): Uint8Array {
  const version = new Uint8Array([0x02, 0x01, 0x00]);
  const algId = new Uint8Array([
    0x30, 0x0d, // SEQUENCE, length 13
    0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, // OID rsaEncryption
    0x05, 0x00, // NULL
  ]);
  const octetStringHeader = derTagAndLength(0x04, pkcs1.length);
  const innerLength = version.length + algId.length + octetStringHeader.length + pkcs1.length;
  const outerHeader = derTagAndLength(0x30, innerLength);

  const out = new Uint8Array(outerHeader.length + innerLength);
  let off = 0;
  out.set(outerHeader, off); off += outerHeader.length;
  out.set(version, off); off += version.length;
  out.set(algId, off); off += algId.length;
  out.set(octetStringHeader, off); off += octetStringHeader.length;
  out.set(pkcs1, off);
  return out;
}

function derTagAndLength(tag: number, length: number): Uint8Array {
  if (length < 0x80) return new Uint8Array([tag, length]);
  if (length < 0x100) return new Uint8Array([tag, 0x81, length]);
  if (length < 0x10000) return new Uint8Array([tag, 0x82, length >> 8, length & 0xff]);
  throw new Error(`derTagAndLength: length ${length} too large to encode here`);
}

// ---- base64 helpers (Workers don't ship Buffer) --------------------------

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
