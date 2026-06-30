// Cloudflare Pages Function — catch-all under /radar/*. Serves live radar artifacts from R2, same-origin.
//
//   GET /radar/ping                      -> "pong"  (deploy/route diagnostic; no binding needed)
//   GET /radar/nowcast/bonehill.json     -> R2 object "radar/nowcast/bonehill.json"
//
// The static Pages site can't read the private R2 bucket; this reads just the requested object via an R2
// binding and returns it (no CORS, always-live). REQUIRES a one-time dashboard step: the weatherblend Pages
// project needs an R2 binding named `RADAR` -> the `weatherblend` bucket (Pages → Settings → Functions →
// R2 bindings, Production). Until then /radar/nowcast/* returns 503 (and the card stays hidden, gracefully).
export async function onRequest({ params, env }) {
  const path = Array.isArray(params.path) ? params.path.join("/") : params.path;

  if (path === "ping") {
    return new Response("pong", { headers: { "content-type": "text/plain", "cache-control": "no-store" } });
  }
  if (!env.RADAR) {
    return new Response('{"error":"R2 binding RADAR not configured on the Pages project"}',
      { status: 503, headers: { "content-type": "application/json", "cache-control": "no-store" } });
  }
  const obj = await env.RADAR.get("radar/" + path);
  if (!obj) {
    return new Response("{}", { status: 404, headers: { "content-type": "application/json", "cache-control": "no-store" } });
  }
  return new Response(obj.body, { headers: { "content-type": "application/json", "cache-control": "no-store, max-age=0" } });
}
