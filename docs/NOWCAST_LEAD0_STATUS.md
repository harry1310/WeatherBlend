# Lead-0 nowcast + per-location policy — status & remaining work

_Last updated 2026-06-14._

A "nowcast" = a 0h-lead blender model. The forecast NWPs give us hourly data
down to +1h, but our blenders only ever trained from lead 24h up — so this work
adds a lead-0 model, trained on the Open-Meteo **historical-forecast** archive
(≈analysis quality at lead 0), and lets the per-lead policy select it where it
wins. See also `docs/PRECIP_LEAD_POLICY_PLAN.md` for the underlying policy
machinery.

## Verdict (bake-offs, walk-forward OOS, Bonehill)

- **Temperature 2c: the nowcast wins** — at live ≤12h freshness it beats the 24h
  model by ~3–4% MAE; the policy fit selects a **0+24 equal-weight blend** at the
  0–24h bands (+4.4–6.0%), a 24+48 blend at 24–48h (+14%), baseline beyond.
- **Precipitation: it does not** — at live freshness (τ≥3, real cycles) the 0h
  model is tied/slightly worse than m24. Its only "win" is τ=0 fed the archive
  analysis, which **isn't available at live predict time** — so it's correctly
  auto-excluded by the policy (which scores on live-only inputs).

Net: keep the 0h model trained for both targets (the policy decides per-target),
but only temperature realises a gain.

## Shipped (all on `main`, validated by fast suite + smokes)

| area | commit | what |
|---|---|---|
| data | `d948a31` | hist-forecast backfill widened to surface fields (the 0h training source) |
| harness | `723d42c` | lead-0 trainable (`NowcastSource`, builder source-switch, `train --nowcast`), guard-safe, bake-off commands |
| enable | `82f10d3` | `--nowcast` for Bonehill 2c + 3o in `retrain-blenders.yml` |
| policy | `69a5007`, `0c125a8`, `112ef96` | live-only scoring; lead-0 candidate + ±24h band window; target-generic + temp producer (`temp-fit-lead-policy`); per-location artifact |
| predict | `13e144d` | temp predict consumes the policy + executes blends + adds the today bucket |
| display | `433d529` | `12h` forecast tab renamed `<24hr` |
| perf | `0b36498` | scan-once train cache in 3a/3c/3o + 2b/2c |

## 2026-06-15 — all-locations rollout (pending push, not yet on `main`)

This pass extends the nowcast from Bonehill-only to all three locations and
makes it standard. **None of it is committed yet** — it deploys via the next
Sunday retrain once pushed.

**Data correction.** The earlier claim "Bonehill is the only location with
hist_forecast data" was wrong. Membury *had* 7,073 hist_forecast files but every
**surface** field was null — they predate the 06-13 surface-field widening
(`d948a31`) and Membury's archive was stale at 05-29. Sennen had none. So neither
could train a lead-0 surface model. Fix: a chunked `hist-forecast` backfill (CI
`backfill.yml`, source `hist-forecast`, per location, yearly chunks 2024/25/26)
was dispatched for Membury + Sennen on 2026-06-15; validated as repopulating the
surface fields (24/24 non-null).

**Nowcast is now standard (no flag).** The `--nowcast` train flag is gone;
`TrainCommand` auto-appends lead 0 for exactly the champion rich phases —
temperature **2c**, precipitation **3o** AND **3c** (3c added so Membury/Sennen,
whose precip champion falls back to 3c, also carry a lead-0 candidate). A
location with no hist_forecast surface archive yet trains too few lead-0 rows and
the phase trainer **skips** that lead (warning, not abort) — so it can never
break a retrain. Applies on every location.

**2c is the temperature champion.** phases.yaml lineup reordered to 2c → 2b
(fallback), mirroring precip's 3o → 3c → 3a. The hardcoded lead-12 `ChampionByLead`
pin to 2d is now **cleared** on 2d's promotion, so the `<24hr` bucket is served by
2c's policy everywhere (incl. Bonehill — where the nowcast was proven). 2d stays a
visible challenger. **Precip keeps its 3d lead-12 pin** — the precip nowcast
doesn't win, and 3d (exact-runtime) is the best precip `<24hr` model, so the
asymmetry is deliberate: temp `<24hr` = 2c policy, precip `<24hr` = 3d.

**Per-location policy producers.** `temp-fit-lead-policy`, `precip-fit-lead-policy`
and `precip-policy-retrain` all take `--location` now (resolved like
`TrainCommand`), with the hardcoded `location=bonehill_rocks` globs swapped to the
resolved location. The precip retrain mints 3c per gauge and **skips 3o** for
locations without the Bonehill oro pool (Membury/Sennen → 3c only). The precip
fit drops the lead-0 candidate (rebuilds cands+pairs) when no lead-0 study model
exists, so the `cands.All` gate can't empty a location's policy.

**CI.** `retrain-blenders.yml`'s fit step (still behind the quarterly
`fit_lead_policy` flag) now runs **per matrix location** for both temperature and
precipitation, pushing per-location `LEAD_POLICY_<loc>.json`.

**Display.** The skill page's per-lead MAE/Brier panels relabel the lead-12 panel
to `<24hr` (matching the forecast tab) — that panel scores the policy-served
today bucket, not a fixed 12h model.

## How to activate (operational, gated)

Nothing is live until these run — by design:

1. **Finish the backfills** (Membury + Sennen `hist-forecast`, running 2026-06-15).
2. **Push this branch** + the next Sunday retrain mints `lead_0h` into the 2c/3c/3o
   bundles per location and makes 2c the Active champion.
3. **Run the fit** (`retrain-blenders.yml` dispatch with `fit_lead_policy=true`, or
   the quarterly tick) → writes `LEAD_POLICY_<loc>.json` per location to R2.
4. Predict routes the `<24hr` bucket through each location's policy; the `<24hr`
   temp tab + the relabelled skill panel show it.

## Remaining work

### 1. Per-location policy **producer** + CI  (request 1 — task #9)

The policy **artifact** (`LEAD_POLICY_<location>.json`) and **predict
consumption** are already per-location. The **producers are not** — they still
fit Bonehill only (`RunFitLeadPolicyAsync`, `RunFitLeadPolicyTempAsync`,
`RunPolicyRetrainAsync` in `PrecipCrossLeadBakeoffCommand.cs` hardcode
`location=bonehill_rocks` globs + `_cfg.Location`). So Membury/Sennen still fall
back to **Bonehill's** policy — the exact "one policy for all locations" that
request 1 set out to remove.

To finish:
- Add `--location` to `precip-fit-lead-policy`, `temp-fit-lead-policy`,
  `precip-policy-retrain`; resolve it (mirror `TrainCommand`'s `locationOverride`)
  and replace the hardcoded `location=bonehill_rocks` globs with `location.Name`.
- **No-lead-0 handling (the subtlety):** Membury/Sennen have no hist_forecast, so
  no lead-0 study bundle. Adding `0` to `cands` would make the per-station
  `cands.All(byLead.ContainsKey)` check skip every station → an empty policy
  (losing their existing 3c bands). Fix: compute an effective candidate set that
  **drops 0 when no lead-0 study bundle exists for the location**, and use it in
  the scoring loop, blend pairs, and the band candidate filter.
- CI (`retrain-blenders.yml`): run the fit **per matrix location** (temp +
  precip) and push the per-location `LEAD_POLICY_<location>.json` (the current
  step is Bonehill-only + pushes the global file).

**Validation:** gated — needs each location's data synced + an actual fit run
(multi-minute, per location); only Bonehill is synced locally. Lower risk than it
sounds: the fit is quarterly + behind a flag, and a failure writes nothing (predict
falls back). Bonehill is provably unaffected (its `location.Name` equals the old
hardcoded literal).

### 2. Smaller follow-ups

- **Confirm the persistence-latency value.** Lead-0 rainfall persistence is
  anchored at `valid − 3h` (`NowcastSource.MinPersistenceLagHours`) to avoid
  leaking the predicted hour. Confirm 3h matches the real live EA-gauge
  availability at predict run-time, and tune if needed.
- **UA-backed chart shading** lights up only after a re-predict populates the new
  `UpperAirIncluded` column (old parquets read null = no shading).
- **Watch after the first Sunday retrain (from 2026-06-15):** `lead_0h.zip` in
  Bonehill 2c/3o bundles; the 3oni tor line on the rain page.
