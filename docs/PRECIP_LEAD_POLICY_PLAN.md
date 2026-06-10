# Per-lead precip policy (3c/3o) — design & work plan

**Status:** IMPLEMENTING (2026-06-10, local during freeze). Decisions locked with
Harry 2026-06-10: thresholds = margin 0.75% / hysteresis 0.5% / 21-day settled
holdout / quarterly Jan-Apr-Jul-Oct first Sunday; lead-12 ships for 3c+3o only;
4a gets its OWN full cross-lead study first (same 6h-band best-model/blend
methodology, Python-side — per-lead BART states fed fresher input; nothing
encoded for 4a until that study reports), and the 4b mint stays ≥24h until 4a
resolves. Phase 0 validator run + producer + predict consumption in progress.

## Goal

For each forecast lead, use the model (or 2-model blend) that actually predicts
real precipitation best at that lead, instead of the fixed per-bucket model. Fit
that policy from the freshly-trained bundles **during retrain behind an optional
flag**, persist it to R2, and consume it at predict time. The flag is set
**quarterly** from the Cloudflare scheduler; every other retrain leaves the policy
untouched (no weekly churn).

## Findings this plan encodes (Bonehill, live-OOS, no-UA, 2026-03-19→06-09)

Methodology: study bundles trained on offset_day ≤ cutoff so the live scoring
window is true OOS; every candidate model {24,48,72,96,120} scored on the **same**
live input at lead τ (bidirectional — a longer-trained model fed a
fresher-than-nominal input is allowed); equal-weight blends of every pair;
aggregated into 6h bands. Brier vs EA hourly truth (≥0.1 mm, complete hours).

### 3o → adopt a 3-zone policy (stable, interpretable)
| lead band | policy |
|---|---|
| 12–36h | **blend(m24, m48)** equal weight |
| 36–90h | **m48** (single — the durable core) |
| 90–120h | **blend(m48, m96)** equal weight |

Identical shape in both 3h and 6h views; matches every prior 3o finding (lead-48
model is durable and best across a wide span). Gains ~0.5–2% Brier vs the bucket
baseline. **Worth encoding.**

### 3c → keep production single-model buckets. NO blends. NO change.
Production already does: m24 for <48h, m48 48–72, m72 72–96, m96 96–120. The
bidirectional band test confirms this spine is at/near best in every band, and m24
cleanly owns the short end (12–42h). 3c blends gain <1% **and the winning pair is
different in nearly every band** (24+72, 24+48, 48+72, 72+96, 72+120 all appear) —
that is overfit noise, not signal. Encoding it would churn on every refit. **Do
not add blends to 3c; leave it on the production bucket spine.**

### Short-lead (the "24h-model-down-to-lead-12" plan)
Confirmed as the short-lead cell of this same policy: at 12–42h, 3c uses **m24**
and 3o uses **blend(24,48)** — both lean on m24 fed fresh (sub-nominal) input.
This plan subsumes that idea rather than competing with it.

### Cross-cutting lessons (these shape the guards below)
- **Input is ~6h-resolved.** Live input is cycle-selected (freshest run with
  lead ≥ τ); 6-hourly NWP cadence means adjacent hourly τ reuse the same forecast.
  → policy is **per 6h band**, never per hour (per-hour is false precision).
- **Equal-weight blends only.** Fitted blend weights overfit at this data scale
  (wind opt-weight lost to 50/50; 3c/3o stack weights didn't transfer; LGB-meta
  overfit). Never fit weights.
- **Margin gate.** Deviate from the default bucket policy only when the gain
  clears a margin (≈0.5–1%) on a held-out SCORE slice.
- **Same-window select=score is mildly optimistic** on marginal picks → the
  producer MUST use a SELECT/SCORE date split before emitting.

## Progress log (2026-06-10, local during freeze)

- **Phase 0 RUN** (`precip-policy-eval-split`, SELECT<05-05≤SCORE): 3c no-change
  CONFIRMED (no SELECT-picked deviation survives SCORE at any τ). 3o's m48 core
  is real (SCORE-best at 7/8 grid points τ36–90, up to +3.7%) but per-τ SELECT
  picks are unstable — band-pooled selection + gates (below) is the governance.
- **Phase 1 SHIPPED (local)**: `Train/PrecipLeadPolicy.cs` (artifact model,
  TryLoad-null-safe, atomic Save, BucketModelFor) + `precip-fit-lead-policy`
  verb (RunFitLeadPolicyAsync in PrecipCrossLeadBakeoffCommand). First run:
  truth coverage 77% (guard passed), 7 deviations emitted — 3o: blend 24+72 @
  12-18/18-24/72-78, blend 48+72 @ 36-42 (+2.2%) and 78-84 (+4.5%), blend
  72+120 @ 114-120; 3c: m72 @ 42-48 (+2.1%) — NOTE: single-model (respects the
  no-blend lock) but deviates from the strict "3c no change" reading; Harry to
  adjudicate whether 3c deviations are admissible at all.
- **Phase 2 BUILT (local)**: lead-12 bucket (today's hours) targeted for 3c+3o
  (3a filters it; 3d/4a paths untouched); per-target ResolveModelLeads (policy
  band by ACTUAL τ → equal-weight members; default = bucket model, lead-12 →
  m24); per-lead model cache; blend members reuse the composed row only under
  identical spec layout; conformal tags only on plain bucket rows; UA at lead
  12 degrades to the NaN block (no exact-pressure rows at that lead). Smoke in
  progress.
- **4a sibling study** (Harry 2026-06-10): WP `scripts/run_4a_crosslead_study.py`
  — cutoff study BART bundles (4 stations × 5 leads) + live-at-τ scoring +
  same band/gate report. Nothing encoded for 4a until it reports.

## Work plan

### Phase 0 — Validate before encoding (harness exists)
- Run `precip-policy-eval-split` (SELECT earlier slice fits the choice, SCORE later
  slice grades it) to confirm the 3o zones survive out-of-selection and 3c stays
  no-change. Lock final 3o zone boundaries.
- Decide **pooled (Bonehill) vs per-gauge** policy. Study is pooled; predict is
  per-gauge. Start pooled (more data, simpler); revisit per-gauge later.

### Phase 1 — Policy artifact + producer command
- **Artifact** `LEAD_POLICY.json` in R2 next to the precip manifest. Schema:
  `{ fittedAtUtc, window, cutoff, phase → [ {leadLo, leadHi, kind: "single"|"blend",
  leads: [L] | [Lo,Hi], baselineBrier, policyBrier, deltaPct } ] }`. Provenance +
  per-band Brier deltas baked in for audit.
- **Producer** `precip-fit-lead-policy` (extends existing `RunPolicyBandAsync` +
  `RunPolicyEvalSplitAsync`):
  1. Walk-forward: train candidate study bundles ≤ (latest settled truth − holdout);
     eval on the held-out **settled** window (truth must have landed).
  2. All-candidate band eval + SELECT/SCORE split + equal-weight top-2 blend.
  3. Apply margin gate + **hysteresis** vs the current `LEAD_POLICY.json` (don't
     flip a band on a sub-threshold change).
  4. Emit `LEAD_POLICY.json`. Default/empty policy ≡ production buckets (no-op).
  5. **Truth-latency / outage guard:** if the holdout window's EA truth isn't
     settled (e.g. the 2026-06 EA outage), skip the update and keep last-good.

### Phase 2 — Predict-path consumption
- Extend the per-lead selection the predict path already honours
  (`ChampionByLead`, proven for temp 2d) to support a **blend** entry:
  `lead → single(version)` OR `lead → blend(versionA, versionB, w=0.5)`.
- For a blend band, run both bundles on the lead-τ input and average (the
  compose-at-predict pattern already exists: 4b = 4a+3o, 3p, wind_blend).
- Confirm the feature build feeds the **τ-appropriate freshest input** regardless
  of the chosen model's nominal training lead (the freshness mechanism).
- **Fallback:** policy missing / stale / referenced bundle absent → production
  bucket policy. Predict must never break on a policy problem.

### Phase 3 — Retrain + Cloudflare quarterly flag
- Add optional `fit_lead_policy` input to `retrain-blenders.yml` (default false).
  When true: after minting production bundles, run the walk-forward study retrain
  + `precip-fit-lead-policy` + push `LEAD_POLICY.json` to R2. Adds ~30–45 min to
  that run only.
- Cloudflare `scheduler-worker`: set `fit_lead_policy=true` on a quarterly tick
  (e.g. first Sunday of Jan/Apr/Jul/Oct) in the `workflow_dispatch` payload;
  false otherwise (wrangler.toml + index.ts). All scheduling stays Cloudflare-only.

### Phase 4 — Verify + guardrails
- `verify` scores policy-driven predictions vs the bucket baseline on subsequent
  cycles; alert (and offer auto-revert to baseline policy) if the policy
  underperforms baseline over a window.
- Log the policy diff each quarter for auditability.

## Scope / decisions to settle
- Pooled vs per-gauge (start pooled).
- Phases: 3c + 3o first (4b later if it pays).
- Bands: 6h. Lead range 12–120 (extend with predict horizon).
- Margin %, hysteresis threshold, holdout length, quarterly months — pick in Phase 0/1.

## Risks & mitigations
- **Churn** → quarterly cadence + margin + hysteresis + 3c-stays-single-model.
- **Overfit** → SELECT/SCORE split, equal-weight blends only, margin gate.
- **Predict fragility** → fallback to bucket policy; reuse existing compose path.
- **Truth latency/outage** → skip-and-keep-last-good.

## Assets already built (uncommitted, local)
`src/WeatherBlend/Commands/PrecipCrossLeadBakeoffCommand.cs`:
- `RunPolicyRetrainAsync` — mints no-UA, ≤cutoff study bundles to `data/models_study/`.
- `RunPolicyBandAsync` (verb `precip-policy-band`) — all-candidate + top-2 blend per
  τ → 3h/6h band policy. Produced the findings above.
- `RunPolicyEvalSplitAsync` (verb `precip-policy-eval-split`) — full bidirectional
  matrix + SELECT/SCORE split (the Phase-0 validator).
- `RunPolicyBlendCrossoverAsync` (verb `precip-policy-blend-crossover`) — fixed
  bracketing-pair + blend per τ (the 24/48 crossover view).

Most of Phase 0–1's compute already exists; the new build is the artifact
format/producer wrapper, the predict-path blend consumption, and the CI/Cloudflare wiring.

## Rough effort
Phase 0 (validate): ~0.5 day (harness exists). Phase 1 (producer+artifact): ~1–2
days. Phase 2 (predict consume + blend): ~1–2 days. Phase 3 (CI/Cloudflare): ~0.5
day. Phase 4 (verify guard): ~0.5 day.
