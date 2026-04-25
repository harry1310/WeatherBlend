# Lead-time backfill: scoping the "as-issued at run T" archives

Phase 1 uses Open-Meteo's historical-forecast API, which does not expose the true
issue time per row. That means lead time is unknown per row, and per-lead-time
blenders (the whole premise of phase 2+) cannot be trained properly on historical
data. Phase 3 needs real cycle-stamped archives so that every row carries
`(model, run_time, valid_time) → lead_hours = valid_time - run_time`.

This memo was generated from working knowledge; items tagged *[verify]* are
retention/licence details most likely to have drifted. A half-day spike should
confirm everything before phase 3 work kicks off.

## 1. GFS — `gfs_seamless`

1. **Archive / source.** NOAA Open Data on AWS: `s3://noaa-gfs-bdp-pds/` (public,
   no-auth). Mirrored on GCS (`gs://global-forecast-system`) and Azure.
2. **Coverage.** Rolling window on BDP-PDS — roughly last ~7–10 days live; deeper
   history on NOMADS and NCEI THREDDS (`https://www.ncei.noaa.gov/thredds/catalog/model-gfs-g4/`),
   which keeps years. For 1–3 years we want NCEI or a third-party mirror.
   *[verify BDP-PDS retention — has been extended multiple times]*.
3. **Format.** GRIB2. 0.25° (`gfs.tCCz.pgrb2.0p25.fNNN`), with `.idx` sidecar files
   mapping message offsets to `(param, level, step)` — load-bearing for point extraction.
4. **Cadence / cycles.** 4×/day at 00/06/12/18Z. Hourly steps to +120h, 3-hourly to +384h.
5. **.NET tooling.** No first-class GRIB2 decoder in .NET. Options:
   - Shell out to `wgrib2` (Windows build available) — fastest, gold standard.
   - Shell out to ECMWF `eccodes` (`grib_get_data`, `grib_ls`).
   - `NGrib` on NuGet exists but is GRIB1-biased and incomplete for GFS GRIB2
     templates; don't rely on it without a spike.
   - Realistic path: **wgrib2 subprocess**, parse stdout.
6. **Point extraction cost.** A full 0.25° GFS f000 surface-variables GRIB2 is ~400 MB.
   For one point, do NOT download full files — use HTTP Range requests keyed off the
   `.idx`. One message ≈ 0.5–1.5 MB at 0.25°. Our ~12 vars × 120 steps × 4 cycles/day
   naively ~6–10 GB/day; with `.idx`-driven ranges realistically
   **~300–500 MB/cycle**, **~1.5 GB/day**, **~550 GB/year**. *[verify with a one-day spike]*.
7. **Licence.** US federal data, public domain. Commercial use fine.
8. **Gotchas.** HRRR (`noaa-hrrr-bdp-pds`) is CONUS-only — irrelevant for Dartmoor.
   GFS pre-2015 is 0.5°. GEFS is a separate bucket; phase 6.

## 2. ECMWF IFS — `ecmwf_ifs025`

1. **Archive / source.** ECMWF Open Data: `https://data.ecmwf.int/forecasts/` (HTTPS,
   no auth), mirrored on AWS (`s3://ecmwf-forecasts/`) and Azure. Python client:
   `ecmwf-opendata`.
2. **Coverage.** Rolling ~4 days *[verify]*. ECMWF does **not** offer a free open
   archive of historic HRES runs — MARS requires a licensed account, and the
   open-data bucket does not retain old cycles. **To build a year of ECMWF lead-time
   data we must start collecting go-forward**; no backfill path.
3. **Format.** GRIB2, 0.25° open-data product (downgraded from the 0.1° operational
   HRES). AIFS also published here in GRIB2.
4. **Cadence / cycles.** 00/06/12/18Z. Steps 0–144h at 3h, 144–240h at 6h
   *[verify — schedule changes periodically]*. AIFS same cadence.
5. **.NET tooling.** wgrib2 / eccodes subprocess. ECMWF's GRIB2 templates are
   cleaner than NOAA's; eccodes handles them without surprises.
6. **Point extraction cost.** Files are small — a cycle's surface parameter file
   is ~10–30 MB. Range-request optimisation matters less. Budget
   **~100–200 MB/cycle**, **~500–800 MB/day** including AIFS. **~200–300 GB/year**.
7. **Licence.** CC-BY-4.0 since Jan 2023. Commercial use fine with attribution.
8. **Gotchas.** **No historical backfill** — biggest single constraint of the plan.
   0.25° is down-sampled, fine for a point. AIFS (ML model) is free alongside in
   the same bucket — effectively a free 7th model. URL template is straightforward:
   `YYYYMMDD/HHz/ifs/0p25/oper/YYYYMMDDHH0000-<step>h-oper-fc.grib2`.

## 3. ICON — `icon_seamless`

1. **Archive / source.** DWD Open Data: `https://opendata.dwd.de/weather/nwp/`.
   Subtrees for `icon` (global 13 km), `icon-eu` (7 km Europe nest), `icon-d2`
   (2.2 km DE/Alps nest).
2. **Coverage.** ~24–48h rolling retention *[verify]*. No deep public archive;
   go-forward only unless archive access is negotiated.
3. **Format.** GRIB2, **one file per (variable, step)** — hundreds of small files
   per cycle. Bz2-compressed.
4. **Cadence.** ICON-global: 00/06/12/18Z, hourly 0–78h then 3-hourly to +180h.
   ICON-EU: 8 cycles/day at 00/03/…/21Z, hourly 0–78h to +120h. ICON-D2: 8
   cycles/day to +48h.
5. **.NET tooling.** Same GRIB2 story. One-file-per-field actually *helps* — each
   HTTP GET lands exactly one message, no Range gymnastics. Simplest integration
   of the six.
6. **Point extraction cost.** ~0.1–1 MB per file. Our variable set × steps ≈ 10–20 MB
   per cycle global, 5–10 MB per cycle ICON-EU. **~100–200 MB/day**, **~40–70 GB/year**.
7. **Licence.** CC-BY-4.0. Commercial use fine.
8. **Gotchas.** Use the regular lat-lon grid, not the native triangular mesh.
   Bonehill Rocks is inside ICON-EU but outside ICON-D2 — D2 is irrelevant for
   Dartmoor; right pair is ICON-global + ICON-EU. Hundreds of requests per cycle —
   use HTTP/2 keep-alive or TCP handshakes dominate.

## 4. Météo-France ARPEGE / AROME — `meteofrance_seamless`

1. **Archive / source.** `https://meteo.data.gouv.fr/` and
   `https://portail-api.meteofrance.fr/`. ARPEGE (global+Europe, 0.1° / 0.25°),
   AROME (France, 0.025°).
2. **Coverage.** Rolling ~24–48h *[verify]*. No deep free archive.
3. **Format.** GRIB2.
4. **Cadence.** ARPEGE: 00/06/12/18Z to +114h. AROME: 8 cycles/day to +51h *[verify]*.
5. **.NET tooling.** wgrib2 / eccodes subprocess.
6. **Point extraction cost.** Files are bundled (domain, level-set, timestep-range);
   each package ~50–200 MB. **~300–500 MB/cycle**, **~1.5 GB/day**, **~550 GB/year**
   if taking both ARPEGE and AROME.
7. **Licence.** Etalab 2.0 (CC-BY-like). Commercial use permitted with attribution.
8. **Gotchas.** AROME is France-domain — **Dartmoor is outside AROME's native
   domain** (SW England sits on the edge). Open-Meteo's `meteofrance_seamless`
   does the ARPEGE handover for us; direct access for this site is effectively
   **ARPEGE-only**. Portal API needs a free key; meteo.data.gouv.fr bulk downloads
   don't *[verify split]*.

## 5. UK Met Office — `ukmo_seamless`

1. **Archive / source.** Met Office **AWS Open Data** bucket
   `s3://met-office-atmospheric-model-data/` (launched 2023, global ~10 km + UKV ~2 km,
   Zarr/NetCDF) *[verify product set — evolving]*. Commercial DataHub
   (`https://datahub.metoffice.gov.uk/`) is paywalled for most useful tiers.
2. **Coverage.** Rolling ~24h *[verify]*. No deep history.
3. **Format.** Zarr and NetCDF, **not GRIB** — which actually hurts us (tooling).
4. **Cadence.** Global: 00/06/12/18Z. UKV: hourly cycles *[verify]*.
5. **.NET tooling.** **The one model with no good .NET path.** Zarr has no mature
   .NET reader (`Zarr.NET` is early-stage). NetCDF .NET wrappers (SDS, HDF5.NET)
   are fiddly on Windows / abandoned. Realistic path: shell out to Python
   (`xarray` + `zarr`) or `nco`/`cdo`. Heaviest integration cost of the six.
6. **Point extraction cost.** Zarr chunking means only chunks covering our point
   are fetched — **~20–50 MB/cycle**, **~20–40 GB/year**. Efficient once tooling
   is paid for.
7. **Licence.** CC-BY-NC 4.0 on some products (**non-commercial only**) and CC-BY
   on others *[verify per product]*. PoC is non-commercial, so fine for now.
   Flag if scope ever widens.
8. **Gotchas.** UKV (2 km) is genuinely interesting for Dartmoor orography — but
   it carries the heaviest tooling lift. Global UKMO is less differentiated vs
   ECMWF/GFS. **Recommendation: drop UKMO from the direct-archive set**; keep
   Open-Meteo's `ukmo_seamless` as a supplementary (non-lead-time) feature.

## 6. Environment Canada GEM — `gem_seamless`

1. **Archive / source.** ECCC Datamart: `https://dd.weather.gc.ca/model_gem_global/`
   and `.../model_gem_regional/`. HRDPS (2.5 km Canada) at `.../model_hrdps/`.
2. **Coverage.** 24h rolling on dd.weather.gc.ca *[verify]*. HPFX
   (`https://hpfx.collab.science.gc.ca/`) keeps a few days more. No deep archive.
3. **Format.** GRIB2.
4. **Cadence.** GEM-global: **00/12Z only (2/day)**. GEM-regional: 00/06/12/18Z.
   HRDPS: 00/06/12/18Z *[verify]*.
5. **.NET tooling.** wgrib2 / eccodes subprocess.
6. **Point extraction cost.** One-file-per-field like DWD, **~50–100 MB/cycle**,
   **~30–60 GB/year**.
7. **Licence.** Open Government Licence — Canada. Commercial use fine.
8. **Gotchas.** Only 2 cycles/day on GEM-global meaningfully cuts training rows.
   HRDPS is Canada-only — GEM-global is the only Dartmoor-relevant product. GEM
   global skill over Europe trails ECMWF/GFS/ICON — it's a diversity input, not
   a standalone-accurate one.

---

## Recommended path for phase 3

**Tier 1 (worth it):**
- **GFS** via AWS Open Data + NCEI for backfill. Best documented, best tooling,
  public domain. The **only** source offering real historical lead-time data.
  ~2–3 days to build a Range-request GRIB2 fetcher + wgrib2 subprocess extractor.
- **ECMWF IFS + AIFS** via ECMWF Open Data. CC-BY, clean GRIB2, trivial URL scheme;
  two models for the price of one. ~1–2 days once the GFS plumbing is reusable.
  Go-forward only.
- **ICON-global + ICON-EU** via DWD Open Data. CC-BY, one-file-per-field = trivial
  decoding. ~1–2 days. Go-forward only.

**Tier 2 (worth it):**
- **ARPEGE** via meteo.data.gouv.fr. Independent European data-assim lineage.
  ~1–2 days. Go-forward only. (AROME dropped — Dartmoor is outside its domain.)

**Skip / defer:**
- **UKMO.** Zarr/NetCDF = ~week of .NET tooling pain for a global model outclassed
  by ECMWF. Drop from direct-archive set; keep Open-Meteo as supplementary.
- **GEM.** 2 cycles/day, weak over Europe. Defer to phase 6; keep Open-Meteo
  meanwhile.

**Effort estimate: ~6–9 engineer-days** for tier 1 + tier 2, including a shared
GRIB2 decoder abstraction (wgrib2 subprocess wrapper), per-source URL templating,
and plumbing `(model, run_time, valid_time)` through the existing hive-partitioned
Parquet writer. The writer already keys on `model/date/run=HH` so storage needs no
changes — just populate `run` honestly instead of "most recent hour".

**Biggest risk:** no historical backfill for ECMWF/ICON/ARPEGE. Day 1 of phase 3
is "turn on the collector"; useful training data exists only after 6–12 months of
accumulation. GFS via NCEI is the only way to get real historical lead-time data
before then.

## Storage / bandwidth estimate (1 year, recommended subset)

GRIB traffic pulled over the wire:
- GFS: ~550 GB/year
- ECMWF IFS + AIFS: ~200–300 GB/year
- ICON global+EU: ~40–70 GB/year
- ARPEGE: ~300–400 GB/year
- **Total: ~1.1–1.3 TB/year inbound**

Extracted point rows kept on disk:
- 4 models × ~4 cycles/day × ~120 lead hours × ~12 vars × ~16 bytes ≈ ~370 KB/day
  raw → **~150 MB/year** Parquet-compressed for all four tier-1/tier-2 models.

Storage trivial; bandwidth is the only real cost (inbound only). ~3–4 GB/day is
fine on any normal connection. **Optional GFS NCEI 3-year bootstrap: ~1.6 TB one-off**.

## Open questions for Harry

1. **Historical depth vs. go-forward.** Accept 6–12 months of go-forward wait for
   ECMWF/ICON/ARPEGE training data, or pay for ECMWF MARS / a commercial archive?
   (MARS is ~€0.01/field — likely £200–1000 for a meaningful backfill.)
2. **GFS NCEI bootstrap: worth the ~1.6 TB one-off?** Only way to get real
   lead-time training data before the new collector matures.
3. **Drop UKMO, or allocate ~5 days for Zarr?** Recommendation: drop; revisit
   only if phase 2 shows meaningful blend deficit at UK orographic sites.
4. **Commercial scope.** CLAUDE.md says non-commercial only, but Met Office AWS
   is CC-BY-NC on some products — if scope ever widens, UKMO has to go anyway
   or be negotiated via DataHub.
5. **wgrib2 vs eccodes vs NGrib half-day spike** before writing the fetcher.
   Confirm one works as a subprocess on Windows on real GFS + ECMWF samples;
   the rest of the plan assumes this.
6. **AIFS as a 7th model?** Free alongside ECMWF IFS with zero extra tooling cost
   and it's an ML model (phase 6 topic anyway). Basically free to add in phase 3 —
   worth doing?
7. **Cycle frequency in storage schema.** Current `run=HH.parquet` hive layout
   assumes ≤1 run per hour — fine for these sources, but flag for anything
   sub-hourly (historically HRDPS).
