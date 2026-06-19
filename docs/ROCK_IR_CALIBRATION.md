# Rock-surface temperature — IR-gun field calibration

Field readings of real rock-surface temperature taken with a handheld IR
thermometer, logged against the model's prediction for the same
(location, valid hour). Purpose: calibrate the Force-Restore rock-surface
model (`RockSurfacePhysics`) and separate its error sources — direct-beam /
orientation geometry vs the shortwave-under-cloud over-read ("globals leak").

## Method / caveats

- Granite emissivity ≈ 0.95; set the gun accordingly (or note if left at 1.0).
- Record **aspect** (which way the face looks) and **tilt** (horizontal slab vs
  vertical wall) and **sun state** (direct sun vs shade) — these dominate the
  spread and the model is only as good as the orientation it represents.
- The model row is the **freshest available forecast** for that valid hour;
  note its lead, since a 24h-lead value is a forecast, not a nowcast.
- One reading is a directional anchor, not a fit target. Prefer a spread of
  conditions (clear midday, fully overcast, clear night) before retuning any
  coefficient.

## Readings

| Date | Time (UTC) | Location | Aspect | Tilt | Sun state | IR °C | Model °C | Model air / dew °C | Model SW (W/m²) | Cloud % | Model lead | Notes |
|---|---|---|---|---|---|---:|---:|---|---:|---:|---|---|
| 2026-06-18 | 12:00 | bonehill_rocks | horizontal | flat | direct sun | 25 | 23.6 | 15.8 / 14.7 | ~490 | 99 | 24h (made 06-17 21Z) | Whole-crag (Face="") model run. Matches horizontal-in-sun. |
| 2026-06-18 | 12:00 | bonehill_rocks | — | vertical | direct sun | 20 | 23.6 | 15.8 / 14.7 | ~490 | 99 | 24h | The climbing-relevant surface; model ~3.6°C warm vs this. |
| 2026-06-18 | 12:00 | bonehill_rocks | — | vertical | shade | 18 | 23.6 | 15.8 / 14.7 | ~490 | 99 | 24h | ≈ air+2°C — baseline (no beam) looks healthy; warmth is the beam/SW term. |

## What this set established (2026-06-18)

- The whole-crag horizontal model (23.6°C) tracks the **horizontal-in-sun**
  reading (25°C), but climbers are on **vertical** rock (20°C sun / 18°C shade),
  so it over-reads the climbing-relevant temperature by ~4°C. → Bonehill moved to
  a **single sunlit vertical face** (south, 90° slope, `fSky 0.5`) in `config.yaml`.
- The **vertical-shade** reading (18°C ≈ air+2°C) isolates the non-solar physics
  from the shortwave over-read: the baseline is sound, and the wall does **not**
  cool below air in shade → it sees ~half sky (fSky 0.5), not full sky.
- Note SW was ~490 W/m² at "99% cloud" yet there was direct sun on the rock — the
  NWP cloud/SW field for that hour is itself suspect (the globals-leak the
  2026-06-08 check flagged). A **fully-overcast** reading would test that term
  cleanly; a **clear-night** reading would test cooling/baseline.
