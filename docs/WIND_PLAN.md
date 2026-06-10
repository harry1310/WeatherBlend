# Wind plan — MVN to direction-only, lgb as the speed product, CQR bands

**Status:** SHIPPED 2026-06-10. Items 1–3 (MVN speed retired from chart + blend,
UKMO floor dropped, wind_blend = 50/50 mean of champion + lgb) landed 2026-06-09;
item 4 (Python CQR) landed 2026-06-10 as "Option B" — the WHOLE wind_speed_lgb
model moved to Python (WeatherProbabilistic train/predict_wind_speed_pi.py:
quantile-LGB q05/q50/q95 + K=10 cross-conformal CQR; q50 matched the .NET point
within ~1%, production-regime band coverage ~88-89%), answering the first open
decision below (Python owns the full quantile model incl. the point). The .NET
WindSpeedLgb* train/predict path was removed at the cutover. Coverage levels:
90% single band (option for nested bands stays open). Drafted 2026-06-09 during
commit freeze. Supersedes the MVN-speed parts of the wind tab.

## Findings this plan rests on (this session, evidence)

- **MVN (wind_direction) after the 2026-06-07 retrain:** direction held up
  (L24/L48 MAE *improved* 45.9→45.3 / 50.7→45.6°, L72 +6.6% to 54.6°); **speed
  broke** — L72 speed MAE 4.55→18.33 m/s and the predictive distribution itself
  degraded (val NLL doubled at L48/L72, 5.1→10.0), so the **speed CI is unreliable
  too**, not just the point. Cause: MVN trains 2022–2024 on Dunkeswell with an
  index-fraction split that *slides* when the offset_day feature backfill densifies,
  feeding a high-variance MVN-NLL MLP → unstable fit. **Decision: MVN is
  direction-only.** No rollback (this retrain's *direction* is fine).
- **Cross-truth (controlled, OOS, same features/rows/split):** judged on **real
  Dunkeswell** wind, the Dunkeswell-trained lgb beats the ERA5-trained champion at
  every lead (+2.5–4.8%). The champion's headline 0.43–0.68 is the **ERA5-easy-
  target illusion** — the same NWP scores ~2× "better" vs ERA5 than vs a real
  anemometer. An **equal-weight blend(champion, lgb) beats either single on real
  wind** (−1 to −1.6%); a val-fitted weight overfit and lost to 50/50 at every lead.
- **wind_speed_lgb is data-starved by a floor:** `UkmoCleanWindowStart = 2024-09-01`
  in `WindSpeedLgbFeatureBuilder` caps it at ~4 months / 2,899 rows (autumn only).
  UKMO is *already optional* and LightGBM is NaN-tolerant, so the floor is pure cost.
  Dropping it gives Feb–Dec 2024 (~7,400 rows, 2.5×) and is **fully usable** — the
  blend still beats its best single by +12–16%, no "dual-regime" collapse. Compounds
  to ~2 years once MIDAS Open 2025 lands (~a month out).
- **Spike — ML.NET can't do quantile.** Reflection on `Microsoft.ML 4.0`
  `LightGbmRegressionTrainer.Options` (full inherited surface): tree/iteration knobs
  only; `EvaluationMetric ∈ {None,Default,MAE,RMSE,MSE}`; **no `Objective`, no
  `Alpha`, no quantile/pinball, no custom-objective passthrough.** FastTree is L2 too.
  → **CQR is not achievable in the .NET pipeline; it must be built in Python
  (Option A).**

## Plan

### 1. Stop using MVN speed everywhere; keep direction
- Remove the MVN speed series (`mvnAtLead`) **and** the MVN-derived speed CI band
  from the wind-speed chart in `SitePages.Wind.cs`.
- Remove MVN speed as a `wind_blend` input (see item 3).
- **Keep** the MVN wind-direction chart/grid and the **direction** CI — MVN remains
  a direction-only model.

### 2. Drop the UKMO clean-window floor — for wind, wind_gust AND wind_speed_lgb
**Updated 2026-06-09 after a proper floor bake-off — the "leave it on wind/gust"
assumption was wrong.** Floored (≥2024-09) vs no-floor (floor off; required-not-null
clips to ~2024-01/02) trained on a common cutoff, scored on a common OOS window
(≥2026-03-01, N=2136, ERA5 truth, same features/spec/hp):

| target | L24 | L48 | L72 |
|---|---|---|---|
| wind MAE floored→no-floor | 0.4247→0.4207 (−0.9%) | 0.5024→0.4896 (−2.5%) | 0.6598→0.6546 (−0.8%) |
| wind_gust MAE floored→no-floor | 0.9895→0.9714 (−1.8%) | 1.2583→**1.0948 (−13.0%)** | 1.3672→1.3153 (−3.8%) |

**No-floor wins at every lead for both.** The extra data is only the ~6–7 month
Jan/Feb–Aug 2024 block (no-floor ~18.1k rows vs floored 13.1k — **not** 2022; the
required GFS/ECMWF/ICON offset_day only reaches early 2024). The 2026-04-26
"floor helps" result is **superseded** — back then post-2024-09 data was thin so the
all-NaN UKMO block dominated; now there's ample clean data and LightGBM absorbs the
optional-UKMO-NaN block fine. The floored wind L24 (0.4247) ≈ the live champion's
~0.43, so dropping the floor *improves the live wind champion* too.

**Action:** remove the `ValidTimeUtc >= UkmoCleanWindowStart` floor from
`WindFeatureBuilder`, `WindGustFeatureBuilder` **and** `WindSpeedLgbFeatureBuilder`,
then retrain all three (Bonehill).

**Floor scope across element blenders (audited 2026-06-09):** only
**wind, wind_gust, cloud, wind_speed_lgb** ever carried the floor. **Radiation** drops
UKMO entirely (UKMO ~92–100% null there) and **humidity** never includes UKMO — neither
has the floor, nothing to change.

**Cloud is different — UKMO is REQUIRED there (not optional like wind/gust):**
- Dropping the floor *alone* is a **no-op** for cloud — the required-not-null on UKMO
  already clips rows to UKMO's 2024-08 start (floored 13,104 vs no-floor 13,728 rows;
  MAE flat ±0.2%). The floor is redundant.
- The real lever is **demoting UKMO required→optional, then dropping the floor**, which
  unlocks ~Feb–Aug 2024 (ecmwf then binds; +5,040 rows). Bake-off (% cloud MAE, common
  OOS ≥2026-03-01):

| cloud config | L24 | L48 | L72 |
|---|---|---|---|
| production (UKMO-req, floored) | 13.64 | 15.64 | 17.77 |
| UKMO-opt, no-floor (≥2024-02) | **13.42 (−1.6%)** | **15.23 (−2.6%)** | **17.43 (−1.9%)** |

  Demoting UKMO in the clean window alone is ~neutral (UKMO-required isn't earning its
  keep), and the earlier data does the rest. **Action for cloud:** in `config.yaml`
  `blenders.cloud`, move `ukmo_seamless` from `requiredModels` → `optionalModels`, and
  drop the floor in `CloudFeatureBuilder`. Free robustness bonus: cloud no longer dies
  when UKMO is absent (cf. the GEM-optional change).

Caveat: single recent (spring, UKMO-present) test window. Wins are consistent across
leads + both targets (gust −13% at L48 is large), so trustworthy; a second seasonal
window can confirm if wanted.

### 3. wind_blend = equal-weight mean(champion, wind_speed_lgb)
- Replace the current `wind_blend` (sigmoid of lgb + **MVN speed**) with
  `0.5·(champion ERA5 wind speed) + 0.5·(wind_speed_lgb speed)`.
- **Mean (50/50) confirmed best** — beat the fitted weight OOS at every lead.
- This also removes the MVN-speed dependency (ties to item 1).
- `wind_blend` is minted in the predict tail (`WindBlendMintCommand`) — change the
  members + the combination rule there.

### 4. Confidence bands for wind_speed_lgb via CQR — **Option A (Python)**
Spike forces this cross-runtime; mirrors the 4a (R subprocess) / precip-Python
precedent.
- **Method (CQR):** Python `lightgbm 4.6` (already in WP `.venv`) trains quantile
  heads (`objective=quantile`, e.g. α = 0.05 / 0.5 / 0.95 → 90% PI; optionally
  0.25/0.75) per lead on the wind_speed_lgb feature set + Dunkeswell truth, then a
  **conformal correction** on a calibration split restores guaranteed coverage. The
  quantile model is feature-conditioned, so heteroscedasticity (wider bands in high
  wind / high NWP spread) and the **0-bound / right-skew** come for free — which the
  dropped Gaussian MVN could not do.
- **Output:** per (lead, valid-hour) a center + PI bounds at the chosen coverage →
  shaded band(s) on the speed chart, replacing the removed MVN speed CI.
- **Calibration refresh:** re-fit **weekly** at retrain — safe, because conformal
  calibration tracks error level and has no model-selection decision to churn (unlike
  the precip per-lead *policy*, which is gated to quarterly).
- **Integration:** `train_wind_speed_pi.py` + `predict_wind_pi.py` (WP) emit a PI
  parquet to R2; the .NET render reads it. Predict reliability mirrors 4a's
  state-loader; **fallback:** PI missing → render the point line with no band (never
  break the chart).

## Local work order (post-this-doc)
1. **Item 2** (floor drop + retrain) — smallest change, and a better lgb feeds items
   3 and 4. Do first.
2. **Item 1** (strip MVN speed from chart + remove as wind_blend input).
3. **Item 3** (wind_blend = mean(champion, lgb)).
4. **Item 4** (Python CQR) — largest: train + predict + render bands + weekly
   recalibration + Cloudflare wiring.

## Open decisions (settle as we go)
- CQR: does Python produce **only** the interval (point stays the .NET lgb), or the
  **whole** quantile model incl. the median as the point? (Leaning: Python owns the
  full quantile model for coherence; the .NET lgb point can remain the blend member.)
- Coverage levels to display (80% only, or 80% + 95% nested).
- Whether `wind_blend` also gets a band (would need conformal-wrapping the blend;
  champion has no native CI). Defer.
- Bonehill-only today (single location) — keeps it simple.

## Risks & mitigations
- **Cross-runtime predict** → 4a-style state loader + point-only fallback.
- **MIDAS truth capped at 2024** until 2025 lands → lgb window limited for now;
  acceptable, recovers automatically.
- **Churn** isn't a concern here — none of these are weekly model-selection decisions
  (the quarterly-gated piece is the *precip* per-lead policy, separate doc).
