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

const WORKFLOW_FOR_CRON: Record<string, string> = {
  "15 2,8,14,20 * * *": "collect.yml",
  "45 */2 * * *":       "predict-and-render.yml",
  "0 12 * * *":         "era5-refresh.yml",
  "30 9 * * 1":         "verify.yml",
};

export interface Env {
  GH_APP_ID: string;
  GH_APP_INSTALLATION_ID: string;
  GH_APP_PRIVATE_KEY: string;
  GH_REPO: string;
  GH_REF: string;
}

export default {
  async scheduled(event: ScheduledController, env: Env, _ctx: ExecutionContext): Promise<void> {
    const workflow = WORKFLOW_FOR_CRON[event.cron];
    if (!workflow) {
      // Unknown cron string means wrangler.toml and this file have drifted
      // out of sync. Throwing surfaces it as a failed Cloudflare scheduled
      // run instead of a silent miss.
      throw new Error(
        `unknown cron expression: '${event.cron}'. Update WORKFLOW_FOR_CRON in src/index.ts.`,
      );
    }
    console.log(
      `scheduled: cron='${event.cron}' → ${workflow} ` +
      `(scheduledTime=${new Date(event.scheduledTime).toISOString()})`,
    );
    await dispatchWorkflow(env, workflow);
  },

  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const knownWorkflows = new Set(Object.values(WORKFLOW_FOR_CRON));

    if (url.pathname === "/dispatch" && request.method === "POST") {
      const requested = url.searchParams.get("workflow") ?? "collect.yml";
      if (!knownWorkflows.has(requested)) {
        return new Response(
          `unknown workflow: '${requested}'. Known: ${[...knownWorkflows].join(", ")}\n`,
          { status: 400 },
        );
      }
      try {
        await dispatchWorkflow(env, requested);
        return new Response(`dispatched ${requested}\n`, { status: 200 });
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e);
        return new Response(`error: ${message}\n`, { status: 500 });
      }
    }
    return new Response(
      `weatherblend-scheduler — POST /dispatch?workflow=<filename> to fire manually.\n` +
      `Known workflows: ${[...knownWorkflows].join(", ")}\n`,
      { status: 200 },
    );
  },
};

// ---- workflow dispatch ----------------------------------------------------

async function dispatchWorkflow(env: Env, workflowFile: string): Promise<void> {
  const jwt = await mintAppJwt(env.GH_APP_ID, env.GH_APP_PRIVATE_KEY);
  const installationToken = await exchangeForInstallationToken(jwt, env.GH_APP_INSTALLATION_ID);

  const url = `https://api.github.com/repos/${env.GH_REPO}/actions/workflows/${workflowFile}/dispatches`;
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
  console.log(`dispatched ${env.GH_REPO}/${workflowFile}@${env.GH_REF}`);
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
