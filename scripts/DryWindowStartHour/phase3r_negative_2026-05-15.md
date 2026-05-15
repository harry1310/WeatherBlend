# Phase 3r — model `longest_dry_run` directly — NEGATIVE (2026-05-15)

**Verdict: rejected.** Two independent estimators of the daytime
`longest_dry_run` response variable both lost to 3g by ~46–55% Brier at the
6-hour window. Code stripped same day. This file is the durable record.

## Idea

Every dry-window phase to date models the *answer* (binary "is there an
N-hour dry block") or the *hourly chain* (3g/3j/3n Monte-Carlo over 3a's
hourly P(wet)). 3r tried the one untried shape: model the **response
variable** — `longest_dry_run_hours`, the longest contiguous dry run inside
the 09–18 Europe/London daytime window (integer 0–9) — and derive
P(longest ≥ N) for any window N from that. Monotonicity across windows is
free, and one model serves every window.

v1 scope was a 6h-window challenger only.

## What was tried

Same 59-feature lean input as the 3b champion (`DryWindowFeatureBuilder`
`BuildSpec(phase: 3b)`), same chronological 70/15/15 split, 3 stations ×
3 leads = 9 cells. Two estimators:

### M1 — 3-bucket multiclass LightGBM

Label = bucket of `longest_dry_run`: `[0,2] [3,5] [6,9]`. ML.NET
`LightGbmMulticlassTrainer`, softmax. `P(≥6) = pmf[bucket 2]`.

(One bug found + fixed mid-run: ML.NET's `MapValueToKey` defaults to
`ByOccurrence` key ordering, which silently reversed the PMF column order
— first run scored Brier 0.69, worse than a constant predictor. Fixed with
`KeyOrdinality.ByValue`.)

**Result: aggregate Brier(P≥6) ≈ 0.20 vs 3g's ≈ 0.13.**

### M1′ — quantile-regression LightGBM

No bucketing. Python `lightgbm` with `objective='quantile'` (ML.NET's
wrapper has no quantile objective) at a 19-point α grid {0.05…0.95}.
`.NET` exported the exact 59-feature vectors + raw `longest_dry_run` so
the Python model used the production feature pipeline verbatim.
`P(≥6)` = fraction of the α-grid quantile predictions ≥ 6 (quantile grid
as equally-weighted draws), row-wise sorted to repair quantile crossing.

**Result: aggregate Brier(P≥6) = 0.2195 vs 3g 0.1497 (paired) — +46.7% worse.
Beat 3g in 1/9 cells. Beat climatology (0.2772) — so the model learns
something, just nowhere near 3g.**

Per-cell quantile bake-off:

| Station × Lead | 3r quantile | 3g | 3n | climatology |
|---|---|---|---|---|
| Bellever 24h | 0.1976 | 0.1128 | 0.1103 | 0.2776 |
| Bellever 48h | 0.2126 | 0.1064 | 0.1170 | 0.2776 |
| Bellever 72h | 0.2082 | 0.1404 | 0.1428 | 0.2774 |
| Bovey 24h | 0.2284 | 0.0982 | 0.0985 | 0.2769 |
| Bovey 48h | 0.2305 | 0.1151 | 0.1152 | 0.2767 |
| Bovey 72h | 0.2239 | 0.1369 | 0.1377 | 0.2772 |
| Hexworthy 24h | 0.2236 | 0.2091 | 0.2048 | 0.2775 |
| Hexworthy 48h | 0.2280 | 0.1924 | 0.1927 | 0.2772 |
| Hexworthy 72h | 0.2229 | **0.2355** | 0.2322 | 0.2765 |

## Why it lost (the part that generalises)

3g/3j/3n Monte-Carlo over **3a's calibrated hourly P(wet) marginals**. The
6h-dry-block question is fundamentally about the hour-by-hour wet/dry
*sequence*, and 3a's hourly probabilities are a blended, calibrated product.

3r collapses the day to one scalar and predicts it from the **59 day-level
aggregate features** — `precip_sum`, `wet_hour_count`, per-NWP
`longest_dry`, etc. Those features have already destroyed the hourly
sequencing the answer depends on, and 3r never sees 3a's calibrated hourly
layer at all (it works off raw NWP day-aggregates).

Two unrelated estimators (softmax multiclass, quantile regression), both
~46–55% worse than 3g, consistent across 8/9 cells, with one coherent
mechanism. This is not an estimator-tuning miss — **modelling the response
variable from day-aggregate features is the wrong shape for this question.**
The only "model longest-dry-run directly" variant that could plausibly
compete would need hourly-sequence features or to consume 3a's hourly
P(wet) — at which point it is just a worse-specified 3g.

## Conclusion

3g stays the dry-window champion. The dry-window experiment lineage of
negatives now reads 3f / 3h / 3i / 3k / 3l / 3m / 6a / 3o / 3p / 3q / **3r**.
Don't revisit "model the response variable" without hourly-resolution
inputs.
