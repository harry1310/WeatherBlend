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
 */
type Dispatch = { workflow: string; repo?: string };

/** Each cron can dispatch ONE OR MORE workflows. Multi-workflow tick lets
 * us stay within Cloudflare's free-tier limit of 5 cron triggers per
 * Worker while still fanning out to additional workflows on the same
 * schedule. Workflows on the same tick run in parallel (no shared
 * concurrency lock between repos), so cross-repo fan-out is safe as long
 * as the workflows don't write the same R2 prefixes.
 */
const WORKFLOW_FOR_CRON: Record<string, Dispatch[]> = {
  "45 2,8,14,20 * * *": [{ workflow: "collect.yml" }],
  // s3-collect (WeatherBlend) AND predict-5a (WeatherProbabilistic) share
  // this tick. Both fire at HH+1:00; s3-collect uses the weatherblend-data
  // lock while predict-5a reads Open-Meteo forecast data already pushed
  // to R2 by collect.yml at HH:45 and writes to the standard precipitation
  // predictions tree under model_version=*phase5a*. No shared lock, no
  // race, both finish well before predict-and-render at HH+1:15. (Renamed
  // 2026-05-09 from predict-bayesian.yml when Phase 5 → 5a re-cast it as
  // a real model artefact rather than a CI-only sidecar.)
  "0 3,9,15,21 * * *": [
    { workflow: "s3-collect.yml" },
    { workflow: "predict-5a.yml", repo: "harry1310/WeatherProbabilistic" },
  ],
  "15 3,9,15,21 * * *": [{ workflow: "predict-and-render.yml" }],
  // Daily 12:00 UTC tick fires era5-refresh AND predict-4a. Phase 4a's
  // dbarts BART tree state can't survive cross-session serialize, so each
  // run is a single train+predict (~24 min wall, lead-as-feature pooled
  // across 6 leads × 3 stations). Daily cadence — predict-and-render at
  // HH+1:15 reads the day's 4a parquet from R2; staleness peaks at ~20h
  // relative to the 12:00 fit, acceptable for a 1000-draw × 500-tree
  // posterior-mean blender. Different repo, different R2 prefix
  // (data/predictions/precipitation/.../phase4a) so no shared lock.
  "0 12 * * *":         [
    { workflow: "era5-refresh.yml" },
    { workflow: "predict-4a.yml", repo: "harry1310/WeatherProbabilistic" },
  ],
  "30 9 * * MON,THU":   [{ workflow: "verify.yml" }],
};

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
}

export default {
  async scheduled(event: ScheduledController, env: Env, _ctx: ExecutionContext): Promise<void> {
    const targets = WORKFLOW_FOR_CRON[event.cron];
    if (!targets || targets.length === 0) {
      // Unknown cron string means wrangler.toml and this file have drifted
      // out of sync. Throwing surfaces it as a failed Cloudflare scheduled
      // run instead of a silent miss.
      throw new Error(
        `unknown cron expression: '${event.cron}'. Update WORKFLOW_FOR_CRON in src/index.ts.`,
      );
    }
    // Fire all dispatches in parallel — they go to (potentially) different
    // repos and don't depend on each other. If one fails the others still
    // run; we surface aggregate failure at the end so a partial outage is
    // visible in `wrangler tail` rather than swallowed by an early throw.
    const results = await Promise.allSettled(
      targets.map((t) => {
        const repo = t.repo ?? env.GH_REPO;
        console.log(
          `scheduled: cron='${event.cron}' → ${repo}/${t.workflow} ` +
          `(scheduledTime=${new Date(event.scheduledTime).toISOString()})`,
        );
        return dispatchWorkflow(env, t.workflow, repo);
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
    return new Response(
      `weatherblend-scheduler — POST /dispatch?workflow=<filename> to fire manually.\n` +
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
 *   - On `conclusion = failure | timed_out | startup_failure`:
 *       Look up open issues with label `ci-failure` matching the title.
 *       If none → create a new issue assigned to the repo owner.
 *       If one exists → comment on it with the new failure's run URL.
 *   - On `conclusion = success`:
 *       Look up open issues with the same title.
 *       If any → close them with a "now passing" comment.
 *   - Other conclusions (cancelled, skipped, neutral) are ignored — they
 *     usually mean someone deliberately stopped the run, not a real fault.
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

  const issueTitle = `[ci-fail] ${wfName}`;
  const failureConclusions = new Set(["failure", "timed_out", "startup_failure"]);
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
    run_number: number;
    head_branch: string | null;
    head_sha: string | null;
    conclusion: "success" | "failure" | "cancelled" | "skipped" | "timed_out" | "neutral" | "startup_failure" | null;
    html_url: string;
  };
  repository: {
    full_name: string;
    name: string;
    owner: { login: string };
  };
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
): Promise<void> {
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
    body: JSON.stringify({ title, body, labels }),
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

async function dispatchWorkflow(env: Env, workflowFile: string, repo: string): Promise<void> {
  const jwt = await mintAppJwt(env.GH_APP_ID, env.GH_APP_PRIVATE_KEY);
  const installationToken = await exchangeForInstallationToken(jwt, env.GH_APP_INSTALLATION_ID);

  const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflowFile}/dispatches`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${installationToken}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "weatherblend-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.GH_REF }),
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
