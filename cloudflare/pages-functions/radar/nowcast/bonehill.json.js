// Cloudflare Pages Function — serves the live Bonehill radar nowcast JSON from R2, SAME-ORIGIN.
//
// Route: GET /radar/nowcast/bonehill.json  (the file path under functions/ minus the .js extension).
// The static Pages site can't read the private R2 bucket directly; this tiny endpoint reads just this one
// object via an R2 binding and returns it, so the Overview card can fetch it with no CORS and always-live.
//
// REQUIRES a one-time dashboard step: the weatherblend Pages project needs an R2 binding named `RADAR`
// pointing at the `weatherblend` bucket (Pages → Settings → Functions → R2 bindings, Production).
// Without it `env.RADAR` is undefined and this returns 500 → the card stays hidden (degrades gracefully).
export async function onRequestGet({ env }) {
  const obj = await env.RADAR.get("radar/nowcast/bonehill.json");
  if (!obj) {
    return new Response("{}", {
      status: 404,
      headers: { "content-type": "application/json", "cache-control": "no-store" },
    });
  }
  return new Response(obj.body, {
    headers: { "content-type": "application/json", "cache-control": "no-store, max-age=0" },
  });
}
