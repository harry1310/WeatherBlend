using System.Globalization;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches ECMWF deterministic forecasts. Both the IFS oper stream and the
/// AIFS AI model are published under <c>{date}/{HH}z/{ifs|aifs}/0p25/oper/</c>.
/// One GRIB2 file per (cycle, lead) — different from GFS which packs every
/// lead into one big per-cycle file. Each GRIB2 carries a JSON-Lines
/// <c>.index</c> sidecar with byte offsets per ECMWF param code, so the
/// fetch pattern is "GET .index → range-GET every wanted message →
/// concatenate → wgrib2 extract point".
///
/// The multi-range GET (2026-05-21) replaced one Range request per variable
/// (~10 per file). A typical IFS file is ~72 MB / 184 messages and the ~10
/// surface params we want are scattered through it (a single coalesced span
/// would be ~89% waste), so one HTTP request carrying all ~10 disjoint
/// ranges — answered as <c>multipart/byteranges</c> — is both the fewest
/// requests and the fewest bytes. It cut s3-collect's ECMWF phase from
/// ~30 min to a few against the data.ecmwf.int live endpoint.
///
/// Multi-range is data.ecmwf.int-ONLY. AWS S3 doesn't honour a multi-range
/// Range header — it ignores it and returns the whole object — so the
/// <c>ecmwf-backfill</c> CLI (which uses the AWS archive endpoint) falls
/// back to one single-range GET per message. See <see cref="SupportsMultiRange"/>
/// / <see cref="FetchRangesIndividuallyAsync"/>. (Commit 417d5c2 shipped the
/// multi-range GET in the shared <c>FetchLeadAsync</c> and wrongly claimed
/// the backfill was unaffected — it broke every ECMWF backfill until this
/// per-endpoint split landed.)
///
/// Two endpoints serve the SAME dissemination stream (verified 2026-05-21 —
/// the <c>.index</c> files are byte-identical, md5-matched, between them):
///   * <see cref="LiveBaseUrl"/> — ECMWF's own server, rolling ~4-day
///     window. Used by the s3-collect refresh (3-day lookback). Chosen as
///     the live endpoint after the AWS mirror began returning persistent
///     S3 <c>503 SlowDown</c> on object GETs ~2026-05-20 (a per-prefix
///     request-rate cap on ECMWF's heavily-used bucket, affecting all
///     clients — confirmed with single cold requests from an unrelated IP).
///   * <see cref="ArchiveBaseUrl"/> — AWS Open Data mirror, deep archive
///     back to 2023-01. Used by the historical <c>ecmwf-backfill</c> CLI;
///     the live window is too short on the ECMWF server for multi-year
///     backfills.
/// Caller picks the endpoint per purpose and passes it to
/// <see cref="FetchCycleAsync"/>.
///
/// Archive scope (probed 2026-05-04):
///   * IFS oper: 2023-01-18 onwards (~2y 4m as of probe)
///   * AIFS oper: 2024-02-29 onwards (~1y 2m as of probe)
/// Cycles: 4×/day at 00/06/12/18 UTC. Leads vary by stream/cycle but
/// typically out to ~144h for IFS oper, ~120h for AIFS at 00/12 cycles
/// and shorter at off-synoptic cycles.
///
/// Variables we extract — surface-level ECMWF param codes (no visibility,
/// no cloud-ceiling, no per-altitude cloud bands in the oper stream):
///   2t / 2d        — 2m temperature + dew point (RH derived from these)
///   10u / 10v      — 10m wind components → speed + direction
///   10fg           — 10m wind gust
///   sp / msl       — surface pressure + mean sea level
///   tp             — total precipitation (cumulative since cycle start; mm)
///   tcc            — total cloud cover (0..1)
///   ssrd           — surface short-wave radiation downwards (J/m² accum)
///   mucape         — most unstable CAPE (no plain CAPE in oper stream)
///
/// Stream id: callers pass <see cref="Streams.IfsOper"/> or
/// <see cref="Streams.AifsOper"/>; that selects both the bucket subpath
/// and the model id stamped onto resulting <see cref="ForecastRow.Model"/>
/// rows (<c>ecmwf_ifs_oper</c> / <c>ecmwf_aifs_oper</c> respectively, distinct
/// from Open-Meteo's <c>ecmwf_ifs025</c> / <c>ecmwf_aifs025_single</c> ids).
/// </summary>
public sealed class EcmwfClient
{
    /// <summary>ECMWF's own dissemination server — rolling ~4-day window,
    /// not subject to the AWS mirror's S3 SlowDown throttling. The live
    /// endpoint for s3-collect's recent-window refresh.</summary>
    public const string LiveBaseUrl = "https://data.ecmwf.int/forecasts";

    /// <summary>AWS Open Data mirror — deep archive (2023-01 onwards). The
    /// endpoint for historical backfill; can return S3 503 SlowDown under
    /// load, which is why live collection uses <see cref="LiveBaseUrl"/>.</summary>
    public const string ArchiveBaseUrl = "https://ecmwf-forecasts.s3.amazonaws.com";

    public static readonly int[] CycleHours = { 0, 6, 12, 18 };

    public static class Streams
    {
        public const string IfsOper  = "ifs";
        public const string AifsOper = "aifs";
    }

    private static readonly IReadOnlyList<VarMap> Variables = new[]
    {
        new VarMap("2t",     (r, v) => r.Temperature2m       = v - 273.15),  // K → °C
        new VarMap("2d",     (r, v) => r.DewPoint2m          = v - 273.15),
        new VarMap("10u",    (r, v) => r.U10                 = v),
        new VarMap("10v",    (r, v) => r.V10                 = v),
        new VarMap("10fg",   (r, v) => r.WindGusts10m        = v),
        new VarMap("sp",     (r, v) => r.SurfacePressure     = v / 100.0),   // Pa → hPa
        new VarMap("msl",    (r, v) => r.PressureMsl         = v / 100.0),   // Pa → hPa (mean sea level)
        new VarMap("tp",     (r, v) => r.Precipitation       = v * 1000.0),  // m (depth) → mm
        new VarMap("tcc",    (r, v) => r.CloudCover          = v * 100.0),   // 0..1 → %
        new VarMap("ssrd",   (r, v) => r.ShortwaveRadiation  = v / 3600.0),  // J/m² acc → W/m² avg-over-hour
        new VarMap("mucape", (r, v) => r.Cape                = v),           // J/kg
        new VarMap("tcwv",   (r, v) => r.PrecipitableWater   = v),           // kg/m² ≈ mm (convective fuel; no CIN in the oper stream)
        // ── Added 2026-05-31: multi-level pressure fields (850/700/500 hPa) ──
        // levtype=pl entries. `gh` is geopotential height in gpm (matches OM's
        // geopotential_height); `z` (geopotential, m²/s²) is also published but
        // we want height. `r` is relative humidity (%). Wind from u/v per level.
        // AIFS publishes t/gh/u/v but NOT r — its RelativeHumidity850hPa stays
        // null (the `r` VarMap simply finds no index entry), which is fine: the
        // column is nullable and the blender's missing-value handling ignores it.
        new VarMap("t",  (r, v) => r.Temperature850hPa = v - 273.15, "pl", "850"),
        new VarMap("t",  (r, v) => r.Temperature700hPa = v - 273.15, "pl", "700"),
        new VarMap("t",  (r, v) => r.Temperature500hPa = v - 273.15, "pl", "500"),
        new VarMap("gh", (r, v) => r.GeopotentialHeight850hPa = v, "pl", "850"),  // gpm
        new VarMap("gh", (r, v) => r.GeopotentialHeight500hPa = v, "pl", "500"),
        new VarMap("r",  (r, v) => r.RelativeHumidity850hPa = v, "pl", "850"),    // % (IFS only)
        new VarMap("u",  (r, v) => r.U850 = v, "pl", "850"),
        new VarMap("v",  (r, v) => r.V850 = v, "pl", "850"),
        new VarMap("u",  (r, v) => r.U500 = v, "pl", "500"),
        new VarMap("v",  (r, v) => r.V500 = v, "pl", "500"),
        // ── Added 2026-06-21: boundary-layer / upper-air / soil bands ──
        // Verified against live IFS + AIFS-single .index files (2026-06-19 00Z).
        // The ECMWF oper stream (both IFS and AIFS) publishes:
        //   * levtype=pl: t/gh/u/v/w/q at all levels incl 925/700/500, plus r
        //     on IFS only (AIFS has q but NO r — its RH columns stay null).
        //   * levtype=sol: sot (soil temperature, K) + vsw (volumetric soil
        //     water, m³/m³) at levels 1-4 (IFS) / 1-2 (AIFS). Level 1 is the
        //     top soil layer (≈ 0-7 cm) ≈ the GFS/OM 0-7cm band.
        // It publishes NO blh (boundary-layer height), NO sshf/slhf (sensible/
        // latent heat flux), NO deg0l (freezing level) and NO et0 in this
        // stream — so BoundaryLayerHeight / SensibleHeatFlux / LatentHeatFlux /
        // FreezingLevelHeight / Et0FaoEvapotranspiration stay null for ECMWF.
        // `w` is omega in Pa/s (same convention as GFS VVEL; AIFS carries it
        // too, so AIFS gets vertical velocity even though it lacks r).
        new VarMap("t",  (r, v) => r.Temperature925hPa = v - 273.15, "pl", "925"),
        new VarMap("gh", (r, v) => r.GeopotentialHeight700hPa = v, "pl", "700"),  // gpm
        new VarMap("r",  (r, v) => r.RelativeHumidity925hPa = v, "pl", "925"),    // % (IFS only)
        new VarMap("r",  (r, v) => r.RelativeHumidity700hPa = v, "pl", "700"),    // % (IFS only)
        new VarMap("r",  (r, v) => r.RelativeHumidity500hPa = v, "pl", "500"),    // % (IFS only)
        new VarMap("u",  (r, v) => r.U700 = v, "pl", "700"),
        new VarMap("v",  (r, v) => r.V700 = v, "pl", "700"),
        new VarMap("w",  (r, v) => r.VerticalVelocity700hPa = v, "pl", "700"),    // Pa/s (omega)
        new VarMap("w",  (r, v) => r.VerticalVelocity500hPa = v, "pl", "500"),
        new VarMap("sot",(r, v) => r.SoilTemperature0to7cm = v - 273.15, "sol", "1"),  // K→°C, top layer
        new VarMap("vsw",(r, v) => r.SoilMoisture0to7cm = v, "sol", "1"),               // m³/m³, top layer
    };

    private readonly HttpClient _http;
    private readonly Wgrib2 _wgrib2;
    private readonly ILogger<EcmwfClient> _log;

    public EcmwfClient(HttpClient http, Wgrib2 wgrib2, ILogger<EcmwfClient> log)
    {
        _http = http;
        _wgrib2 = wgrib2;
        _log = log;
    }

    /// <summary>
    /// Fetch one cycle's forecasts for a given set of lead hours. Each lead
    /// is its own .grib2 + .index file under the cycle directory.
    /// <paramref name="apiBaseUrl"/> selects the endpoint — pass
    /// <see cref="LiveBaseUrl"/> for recent-window collection or
    /// <see cref="ArchiveBaseUrl"/> for historical backfill.
    /// </summary>
    public async Task<List<ForecastRow>> FetchCycleAsync(
        LocationConfig location, DateOnly cycleDate, int cycleHour, string stream,
        IEnumerable<int> leadHours, string scratchDir, string apiBaseUrl, CancellationToken ct)
    {
        Directory.CreateDirectory(scratchDir);
        var runTime = new DateTime(cycleDate.Year, cycleDate.Month, cycleDate.Day,
            cycleHour, 0, 0, DateTimeKind.Utc);
        var modelId = stream switch
        {
            Streams.IfsOper  => "ecmwf_ifs_oper",
            Streams.AifsOper => "ecmwf_aifs_oper",
            _ => throw new ArgumentException($"Unknown stream '{stream}'", nameof(stream)),
        };

        var rows = new List<ForecastRow>();
        // Deaccumulation anchors for ssrd + tp (both accumulated-since-start) —
        // the previous successfully-fetched lead's cumulative values + lead hours.
        double? prevShortwaveCumulative = null;
        double? prevPrecipCumulative = null;
        int prevLeadHours = 0;
        foreach (var lead in leadHours.OrderBy(h => h))
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var (row, swCum, precipCum) = await FetchLeadAsync(location, cycleDate, cycleHour, stream, modelId,
                    lead, runTime, scratchDir, apiBaseUrl, prevShortwaveCumulative, prevPrecipCumulative, prevLeadHours, ct);
                if (row is not null)
                {
                    rows.Add(row);
                    // Advance the anchor on a successful fetch (ssrd + tp arrive
                    // together for the oper streams); a missing lead widens the
                    // next deaccumulation window.
                    prevShortwaveCumulative = swCum;
                    prevPrecipCumulative = precipCum;
                    prevLeadHours = lead;
                }
            }
            catch (HttpRequestException e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                _log.LogWarning("Missing {Stream} {Date} {Cycle:00}z f{Lead:000}: {Msg}",
                    stream, cycleDate, cycleHour, lead, e.Message);
            }
        }
        return rows;
    }

    private async Task<(ForecastRow? Row, double? ShortwaveCumulative, double? PrecipCumulative)> FetchLeadAsync(
        LocationConfig loc, DateOnly date, int cycleHour, string stream, string modelId,
        int lead, DateTime runTime, string scratchDir, string apiBaseUrl,
        double? prevShortwaveCumulative, double? prevPrecipCumulative, int prevLeadHours, CancellationToken ct)
    {
        // Path is `{date}/{HH}z/{streamSubpath}/0p25/{streamFolder}/{date}{HH}0000-{lead}h-{streamFolder}-fc.{grib2|index}`
        // where streamSubpath ∈ {ifs, aifs-single} (or aifs legacy) and the
        // filename suffix mirrors the streamFolder.
        //
        // streamFolder for IFS 06z/18z (the "short cut-off" cycles) is
        // DATE-dependent, NOT endpoint-dependent. ECMWF renamed the 06z/18z
        // IFS folder `scda/` -> `oper/` at source ~2026-05-12 (the wave
        // stream `scwv/` -> `wave/` at the same time). The AWS bucket mirrors
        // data.ecmwf.int 1:1, so both endpoints agree for any given date —
        // verified 2026-05-21 by listing `{date}/06z/ifs/0p25/` on both:
        //   * dates >= ~2026-05-12 — 06z/18z IFS under `oper/` (…-oper-fc).
        //   * dates <  ~2026-05-12 — 06z/18z IFS under `scda/` (…-scda-fc),
        //     reachable only on the AWS archive (data.ecmwf.int keeps a
        //     rolling ~4-day window, all post-rename).
        // 06z/18z `scda` and the 00z/12z `oper` runs are the same IFS HRES
        // product (verified: identical surface-param index) — the rename is
        // cosmetic, no train/serve discontinuity.
        // So try `oper` first (correct for all live collection + any recent
        // backfill) and fall back to `scda` (old archive dates only) — one
        // code path, both layouts. 00z/12z IFS and all AIFS cycles have
        // always been `oper`.
        // (Before the rename this hard-coded `scda` for 06z/18z and worked.
        // The rename then 404'd every live 06z/18z IFS fetch — IFS exact
        // lost those cycles and 3d, which requires IFS, went dark at the
        // 06/12/18z valid hours — until this `oper`-first fallback landed
        // 2026-05-21.)
        //
        // AIFS publishes all four cycles in `oper`. The AIFS subdir was
        // renamed `aifs/` → `aifs-single/` between 2025-02-01 and 2025-03-01
        // — try modern first, fall back on 404.
        var dateStr = date.ToString("yyyyMMdd");

        var subpathCandidates = stream == Streams.AifsOper
            ? new[] { "aifs-single", "aifs" }
            : new[] { stream };
        var streamFolderCandidates =
            stream == Streams.IfsOper && (cycleHour == 6 || cycleHour == 18)
                ? new[] { "oper", "scda" }
                : new[] { "oper" };

        List<IndexEntry>? indexEntries = null;
        string? baseUrl = null;
        string stem = "";
        HttpRequestException? lastNotFound = null;
        foreach (var subpath in subpathCandidates)
        {
            foreach (var streamFolder in streamFolderCandidates)
            {
                stem = $"{dateStr}{cycleHour:00}0000-{lead}h-{streamFolder}-fc";
                baseUrl = $"{apiBaseUrl}/{dateStr}/{cycleHour:00}z/{subpath}/0p25/{streamFolder}/{stem}";
                try
                {
                    indexEntries = await FetchIndexAsync(baseUrl + ".index", ct);
                    break;  // got it
                }
                catch (HttpRequestException e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
                {
                    lastNotFound = e;  // try the next (subpath, streamFolder) candidate
                }
            }
            if (indexEntries is not null) break;
        }
        if (indexEntries is null)
        {
            // Every candidate 404'd — rethrow so FetchCycleAsync logs "Missing".
            if (lastNotFound is not null) throw lastNotFound;
            return (null, null, null);
        }
        if (indexEntries.Count == 0) return (null, null, null);

        // 2) For each variable we want, find the FIRST matching index entry
        //    by exact param match. ECMWF param codes ARE unique per (param,
        //    levtype) so a literal equals on `param` for sfc-only entries
        //    works without the substring fuzziness GFS needs.
        var picks = new List<(VarMap Var, long Offset, long Length)>();
        foreach (var v in Variables)
        {
            // FirstOrDefault on a record returns the default-constructed
            // record (Param == null, all defaults) when nothing matches —
            // not actually null on the reference itself. Filter explicitly
            // on Param being non-null so a missed variable just drops out
            // (logged separately when the picks list ends up empty). Match is
            // exact on (param, levtype, levelist) so a pressure param resolves
            // to the right level rather than the first same-param entry — and
            // AIFS's missing `r` simply finds nothing and drops out.
            var hit = indexEntries.FirstOrDefault(e =>
                string.Equals(e.Param, v.Param, StringComparison.Ordinal)
                && string.Equals(e.LevType, v.LevType, StringComparison.Ordinal)
                && (v.Level is null || string.Equals(e.Levelist, v.Level, StringComparison.Ordinal)));
            if (hit is not null && !string.IsNullOrEmpty(hit.Param))
                picks.Add((v, hit.Offset, hit.Length));
        }
        if (picks.Count == 0)
        {
            _log.LogWarning("No matching variables in .index for {Stem}", stem);
            return (null, null, null);
        }

        // Sort picks by byte offset. Step 3 requests the ranges in this
        // order and writes the multipart parts back in arrival order, so the
        // temp GRIB ends up holding the picked messages in strict offset
        // order — which is the order wgrib2 reads them out in step 4. Step 5
        // then maps vals[i] → picks[i] positionally, so picks MUST be
        // offset-sorted here for that mapping to hold. (Also makes the
        // mapping robust to a server coalescing two adjacent ranges into one
        // multipart part: coalesced or not, every message stays offset-ordered.)
        picks.Sort((a, b) => a.Offset.CompareTo(b.Offset));

        // 3) Pull every wanted message into one temp GRIB. Two fetch shapes,
        //    picked by endpoint (see SupportsMultiRange):
        //    * data.ecmwf.int — ONE multi-range GET; the server answers with
        //      a multipart/byteranges body, one part per GRIB2 message.
        //    * AWS S3 — S3 ignores a multi-range Range header and returns the
        //      whole object (200 application/octet-stream), so each message
        //      gets its own single-range GET, concatenated in offset order.
        //    Either way the temp GRIB ends up offset-ordered (picks were
        //    offset-sorted above) — what steps 4/5 rely on.
        var tmpGrib = Path.Combine(scratchDir, $"{Guid.NewGuid():N}.grib2");
        try
        {
            if (SupportsMultiRange(apiBaseUrl))
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, baseUrl + ".grib2");
                req.Headers.Range = new RangeHeaderValue();
                foreach (var (_, off, len) in picks)
                    req.Headers.Range.Ranges.Add(new RangeItemHeaderValue(off, off + len - 1));

                using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
                resp.EnsureSuccessStatusCode();
                await WriteRangeResponseAsync(resp, picks.Count, tmpGrib, stem, ct);
            }
            else
            {
                await FetchRangesIndividuallyAsync(baseUrl + ".grib2", picks, tmpGrib, ct);
            }

            // 4) Extract point values via wgrib2 (same as GFS path).
            var vals = await _wgrib2.ExtractPointAsync(tmpGrib, loc.Latitude, loc.Longitude, ct);
            if (vals.Count != picks.Count)
                _log.LogWarning("wgrib2 returned {Got} vals, expected {Want} for {Stem}",
                    vals.Count, picks.Count, stem);

            var raw = new RawEcmwfValues();
            for (int i = 0; i < Math.Min(vals.Count, picks.Count); i++)
                picks[i].Var.Apply(raw, vals[i].Value);

            // Stream-specific tp units fix-up (2026-05-05). Per ECMWF Open Data
            // docs, IFS oper publishes total precipitation in METRES (water
            // equivalent depth) while AIFS publishes in kg/m² (= mm directly).
            // Our shared VarMap multiplies tp × 1000 (m → mm) which is correct
            // for IFS but produces ~430× values for AIFS. Undo the scaling
            // for AIFS by dividing by 1000 — empirical AIFS/IFS avg ratio
            // matches the units explanation closely (~430×, not exactly 1000×
            // since AIFS and IFS predict slightly different precip totals).
            if (stream == Streams.AifsOper && raw.Precipitation is double aifsPrecipMm)
                raw.Precipitation = aifsPrecipMm / 1000.0;

            // RH at 2m derived from T + Td via Magnus — ECMWF doesn't publish
            // 2m RH directly in oper. Skip when either input is missing.
            double? rh2m = null;
            if (raw.Temperature2m is double tC && raw.DewPoint2m is double tdC)
                rh2m = MagnusRh(tC, tdC);

            var row = new ForecastRow
            {
                LocationName = loc.Name,
                Model = modelId,
                RunTimeUtc = runTime,
                ValidTimeUtc = runTime.AddHours(lead),
                LeadHours = lead,
                RunTimeSource = RunTimeSources.Exact,
                Temperature2m = raw.Temperature2m,
                DewPoint2m = raw.DewPoint2m,
                RelativeHumidity2m = rh2m,
                WindSpeed10m = raw.U10 is { } u && raw.V10 is { } v ? Math.Sqrt(u * u + v * v) : null,
                WindDirection10m = raw.U10 is { } uu && raw.V10 is { } vv ? WindDirection(uu, vv) : null,
                WindGusts10m = raw.WindGusts10m,
                SurfacePressure = raw.SurfacePressure,
                PressureMsl = raw.PressureMsl,
                // AIFS tcc arrives already as a percentage — undo the shared
                // VarMap's ×100 (IFS tcc is a fraction); clamp to [0,100].
                CloudCover = NormalizeCloudPct(raw.CloudCover, modelId),
                Cape = raw.Cape,
                PrecipitableWater = raw.PrecipitableWater,
                // tp is accumulated-since-start; deaccumulate to the per-window
                // total. 3d (exact precip) consumes ifs_oper/aifs_oper precip, so
                // this needs a coordinated backfill + 3d retrain to keep the
                // train/predict scales consistent (2026-06-15).
                Precipitation = DeaccumulateSum(
                    raw.Precipitation, prevPrecipCumulative, lead, prevLeadHours),
                // ssrd is accumulated-since-start; deaccumulate to the per-window
                // mean W/m². Unused by any blender (the SW blend uses the OM
                // *_025/_seamless variants), so safe to correct without backfill.
                ShortwaveRadiation = DeaccumulateMeanFlux(
                    raw.ShortwaveRadiation, prevShortwaveCumulative, lead, prevLeadHours),
                Temperature850hPa = raw.Temperature850hPa,
                Temperature700hPa = raw.Temperature700hPa,
                Temperature500hPa = raw.Temperature500hPa,
                GeopotentialHeight850hPa = raw.GeopotentialHeight850hPa,
                GeopotentialHeight500hPa = raw.GeopotentialHeight500hPa,
                RelativeHumidity850hPa = raw.RelativeHumidity850hPa,
                WindSpeed850hPa = raw.U850 is { } u850 && raw.V850 is { } v850 ? Math.Sqrt(u850 * u850 + v850 * v850) : null,
                WindDirection850hPa = raw.U850 is { } u850d && raw.V850 is { } v850d ? WindDirection(u850d, v850d) : null,
                WindSpeed500hPa = raw.U500 is { } u500 && raw.V500 is { } v500 ? Math.Sqrt(u500 * u500 + v500 * v500) : null,
                WindDirection500hPa = raw.U500 is { } u500d && raw.V500 is { } v500d ? WindDirection(u500d, v500d) : null,
                // BL / upper-air / soil bands (added 2026-06-21). AIFS lacks `r`
                // → its RH columns stay null; blh/flux/freezing absent → null.
                Temperature925hPa = raw.Temperature925hPa,
                GeopotentialHeight700hPa = raw.GeopotentialHeight700hPa,
                RelativeHumidity925hPa = raw.RelativeHumidity925hPa,
                RelativeHumidity700hPa = raw.RelativeHumidity700hPa,
                RelativeHumidity500hPa = raw.RelativeHumidity500hPa,
                WindSpeed700hPa = raw.U700 is { } u700 && raw.V700 is { } v700 ? Math.Sqrt(u700 * u700 + v700 * v700) : null,
                WindDirection700hPa = raw.U700 is { } u700d && raw.V700 is { } v700d ? WindDirection(u700d, v700d) : null,
                VerticalVelocity700hPa = raw.VerticalVelocity700hPa,
                VerticalVelocity500hPa = raw.VerticalVelocity500hPa,
                SoilTemperature0to7cm = raw.SoilTemperature0to7cm,
                SoilMoisture0to7cm = raw.SoilMoisture0to7cm,
            };
            // Return the row plus the still-CUMULATIVE ssrd (raw.ShortwaveRadiation,
            // = mean×leadHours) so the caller can deaccumulate the next lead.
            return (row, raw.ShortwaveRadiation, raw.Precipitation);
        }
        finally
        {
            try { File.Delete(tmpGrib); } catch { /* best-effort */ }
        }
    }

    private async Task<List<IndexEntry>> FetchIndexAsync(string url, CancellationToken ct)
    {
        using var resp = await _http.GetAsync(url, ct);
        if (resp.StatusCode == System.Net.HttpStatusCode.NotFound)
            throw new HttpRequestException($"index not found: {url}", null, resp.StatusCode);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadAsStringAsync(ct);
        var entries = new List<IndexEntry>();
        foreach (var line in body.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            entries.Add(new IndexEntry(
                Param:    root.GetProperty("param").GetString() ?? "",
                LevType:  root.TryGetProperty("levtype", out var lt) ? (lt.GetString() ?? "") : "",
                Levelist: root.TryGetProperty("levelist", out var ll) ? (ll.GetString() ?? "") : "",
                Offset:   root.GetProperty("_offset").GetInt64(),
                Length:   root.GetProperty("_length").GetInt64()));
        }
        return entries;
    }

    /// <summary>
    /// True when <paramref name="apiBaseUrl"/> is an endpoint known to honour
    /// a multi-range Range header with a <c>multipart/byteranges</c> reply.
    /// Only <see cref="LiveBaseUrl"/> (data.ecmwf.int) qualifies — AWS S3
    /// (<see cref="ArchiveBaseUrl"/>) silently ignores a multi-range header
    /// and returns the whole object, so it must use one Range GET per
    /// message. Unknown endpoints default to the safe single-range path.
    /// </summary>
    private static bool SupportsMultiRange(string apiBaseUrl)
        => apiBaseUrl.StartsWith(LiveBaseUrl, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Fetch each picked GRIB2 message with its own single-range GET and
    /// concatenate them, in iteration order, into one temp GRIB. The fetch
    /// path for endpoints that don't support multi-range requests (AWS S3).
    /// <paramref name="picks"/> is offset-sorted by the caller, so appending
    /// in order leaves the temp GRIB offset-ordered — the same invariant the
    /// multi-range path guarantees, which step 5's positional
    /// <c>vals[i] → picks[i]</c> mapping depends on.
    /// </summary>
    private async Task FetchRangesIndividuallyAsync(
        string gribUrl,
        IReadOnlyList<(VarMap Var, long Offset, long Length)> picks,
        string destPath,
        CancellationToken ct)
    {
        await using var outFs = File.Create(destPath);
        foreach (var (_, off, len) in picks)
        {
            using var req = new HttpRequestMessage(HttpMethod.Get, gribUrl);
            req.Headers.Range = new RangeHeaderValue(off, off + len - 1);
            using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            resp.EnsureSuccessStatusCode();
            await resp.Content.CopyToAsync(outFs, ct);
        }
    }

    /// <summary>
    /// Persist a Range response to <paramref name="destPath"/> as a GRIB file.
    /// A multi-range request (<paramref name="rangeCount"/> &gt; 1) is answered
    /// as <c>multipart/byteranges</c> — each part is a GRIB2 message; the part
    /// payloads are concatenated in arrival order (= the requested offset
    /// order, see the picks sort in <see cref="FetchLeadAsync"/>). A single
    /// requested range comes back as a plain 206 whose whole body is the lone
    /// message. Any other shape (e.g. a 200 full-file because the server
    /// ignored Range) throws — fail loud rather than silently mis-parse.
    /// </summary>
    private static async Task WriteRangeResponseAsync(
        HttpResponseMessage resp, int rangeCount, string destPath, string stem, CancellationToken ct)
    {
        var mediaType = resp.Content.Headers.ContentType?.MediaType;
        if (string.Equals(mediaType, "multipart/byteranges", StringComparison.OrdinalIgnoreCase))
        {
            var boundary = resp.Content.Headers.ContentType!.Parameters
                .FirstOrDefault(p => string.Equals(p.Name, "boundary", StringComparison.OrdinalIgnoreCase))
                ?.Value?.Trim('"');
            if (string.IsNullOrEmpty(boundary))
                throw new HttpRequestException($"multipart/byteranges response for {stem} has no boundary");

            var bodyBytes = await resp.Content.ReadAsByteArrayAsync(ct);
            await using var outFs = File.Create(destPath);
            var parts = 0;
            foreach (var payload in EnumerateMultipartParts(bodyBytes, boundary))
            {
                await outFs.WriteAsync(payload, ct);
                parts++;
            }
            if (parts == 0)
                throw new HttpRequestException($"multipart/byteranges response for {stem} yielded no parts");
        }
        else if (rangeCount == 1)
        {
            // One requested range → plain single-range 206; body IS the message.
            await using var outFs = File.Create(destPath);
            await resp.Content.CopyToAsync(outFs, ct);
        }
        else
        {
            throw new HttpRequestException(
                $"expected multipart/byteranges for {rangeCount}-range request ({stem}), got '{mediaType}'");
        }
    }

    /// <summary>
    /// Yield each part's raw payload from a <c>multipart/byteranges</c> body.
    /// Byte-level — the payloads are binary GRIB2, so the body must never be
    /// treated as text. Each part is delimited by <c>--{boundary}</c>; its
    /// payload runs from the part's <c>\r\n\r\n</c> header terminator to the
    /// <c>\r\n</c> immediately before the next delimiter. The closing
    /// <c>--{boundary}--</c> ends iteration.
    /// </summary>
    private static IEnumerable<ReadOnlyMemory<byte>> EnumerateMultipartParts(byte[] body, string boundary)
    {
        // byte[] (not ReadOnlySpan) — locals can't cross the yield boundary
        // in this iterator.
        var delim = Encoding.ASCII.GetBytes("--" + boundary);
        var headerEnd = "\r\n\r\n"u8.ToArray();
        int pos = 0;
        while (true)
        {
            int rel = body.AsSpan(pos).IndexOf(delim);
            if (rel < 0) yield break;
            int afterDelim = pos + rel + delim.Length;
            // Closing delimiter is "--{boundary}--".
            if (afterDelim + 1 < body.Length && body[afterDelim] == (byte)'-' && body[afterDelim + 1] == (byte)'-')
                yield break;
            int hRel = body.AsSpan(afterDelim).IndexOf(headerEnd);
            if (hRel < 0) yield break;
            int payloadStart = afterDelim + hRel + headerEnd.Length;
            int nRel = body.AsSpan(payloadStart).IndexOf(delim);
            if (nRel < 0) yield break;
            int payloadEnd = payloadStart + nRel;
            // Trim the CRLF that separates the payload from the next delimiter.
            if (payloadEnd - payloadStart >= 2 &&
                body[payloadEnd - 2] == (byte)'\r' && body[payloadEnd - 1] == (byte)'\n')
                payloadEnd -= 2;
            yield return body.AsMemory(payloadStart, payloadEnd - payloadStart);
            pos = payloadEnd;
        }
    }

    /// <summary>
    /// Deaccumulate a mean-flux field ECMWF publishes accumulated from forecast
    /// start. The <c>ssrd</c> VarMap already divides the accumulated J/m² by
    /// 3600, so the stored value equals mean-W/m² × leadHours (the 0→lead mean
    /// times elapsed hours). The per-window mean over (prevLead, lead] is
    /// therefore (cum[lead] − cum[prevLead]) / (lead − prevLead). First lead
    /// (prevLead = 0) → the 0→lead mean. Null in → null out; non-positive window
    /// → value unchanged (can't deaccumulate); clamped ≥ 0. Without this, the
    /// cumulative value /3600 grows ~linearly with lead (≈7900 W/m² at +24h).
    /// </summary>
    internal static double? DeaccumulateMeanFlux(
        double? cumThisLead, double? cumPrevLead, int leadHours, int prevLeadHours)
    {
        if (cumThisLead is not double cur) return null;
        var dLead = leadHours - prevLeadHours;
        if (dLead <= 0) return Math.Max(0.0, cur);
        return Math.Max(0.0, (cur - (cumPrevLead ?? 0.0)) / dLead);
    }

    /// <summary>
    /// Deaccumulate an accumulated-since-start TOTAL (e.g. tp precip, already
    /// converted m→mm). Unlike <see cref="DeaccumulateMeanFlux"/> this is a sum,
    /// not a rate, so the per-window value is just the difference
    /// (cum[lead] − cum[prevLead]) — the precip that fell in (prevLead, lead].
    /// First lead → the 0→lead total; non-positive window → value unchanged;
    /// clamped ≥ 0. The window width follows the fetched lead grid.
    /// </summary>
    internal static double? DeaccumulateSum(
        double? cumThisLead, double? cumPrevLead, int leadHours, int prevLeadHours)
    {
        if (cumThisLead is not double cur) return null;
        if (leadHours - prevLeadHours <= 0) return Math.Max(0.0, cur);
        return Math.Max(0.0, cur - (cumPrevLead ?? 0.0));
    }

    /// <summary>
    /// Normalise total cloud cover to percent [0,100]. The shared <c>tcc</c>
    /// VarMap multiplies by 100 — correct for IFS (tcc is a 0..1 fraction) but
    /// AIFS publishes tcc already as a percentage, so that ×100 over-scales it
    /// (e.g. 9274 for 92.74%). Undo the extra ×100 for AIFS; clamp all to [0,100].
    /// </summary>
    internal static double? NormalizeCloudPct(double? rawPct, string modelId)
    {
        if (rawPct is not double c) return null;
        if (modelId == "ecmwf_aifs_oper") c /= 100.0;
        return Math.Clamp(c, 0.0, 100.0);
    }

    /// <summary>
    /// Magnus formula RH (%) from temperature + dew-point in °C. Standard
    /// World Meteorological Org constants (Tetens 1930; AMS Glossary). Good
    /// to ±0.5% across normal atmospheric range. Returns null on extreme /
    /// non-finite inputs to avoid poisoning downstream LightGBM splits.
    /// </summary>
    private static double? MagnusRh(double tC, double tdC)
    {
        if (!double.IsFinite(tC) || !double.IsFinite(tdC)) return null;
        const double a = 17.625, b = 243.04;
        double satFrac = Math.Exp((a * tdC) / (b + tdC) - (a * tC) / (b + tC));
        return Math.Clamp(satFrac * 100.0, 0.0, 100.0);
    }

    /// <summary>Meteorological wind direction (° from which wind blows) from u/v.</summary>
    private static double WindDirection(double u, double v)
        => (Math.Atan2(-u, -v) * 180.0 / Math.PI + 360.0) % 360.0;

    private sealed record IndexEntry(string Param, string LevType, string Levelist, long Offset, long Length);

    /// <summary>
    /// One ECMWF param to extract. <paramref name="LevType"/>/<paramref name="Level"/>
    /// default to a surface field ("sfc", no level); pressure-level entries pass
    /// LevType="pl" + Level (e.g. "850"). Matching is exact on all three, so the
    /// same param code at different levels (t/u/v/gh at 850/700/500) resolves to
    /// the right index entry.
    /// </summary>
    private sealed record VarMap(
        string Param,
        Action<RawEcmwfValues, double> Apply,
        string LevType = "sfc",
        string? Level = null);

    private sealed class RawEcmwfValues
    {
        public double? Temperature2m { get; set; }
        public double? DewPoint2m { get; set; }
        public double? U10 { get; set; }
        public double? V10 { get; set; }
        public double? WindGusts10m { get; set; }
        public double? SurfacePressure { get; set; }
        public double? PressureMsl { get; set; }
        public double? CloudCover { get; set; }
        public double? Cape { get; set; }
        public double? PrecipitableWater { get; set; }
        public double? Precipitation { get; set; }
        public double? ShortwaveRadiation { get; set; }
        // Pressure-level (850/700/500 hPa); wind kept as U/V per level.
        public double? Temperature850hPa { get; set; }
        public double? Temperature700hPa { get; set; }
        public double? Temperature500hPa { get; set; }
        public double? GeopotentialHeight850hPa { get; set; }
        public double? GeopotentialHeight500hPa { get; set; }
        public double? RelativeHumidity850hPa { get; set; }
        public double? U850 { get; set; }
        public double? V850 { get; set; }
        public double? U500 { get; set; }
        public double? V500 { get; set; }
        // BL / upper-air / soil bands (added 2026-06-21).
        public double? Temperature925hPa { get; set; }
        public double? GeopotentialHeight700hPa { get; set; }
        public double? RelativeHumidity925hPa { get; set; }
        public double? RelativeHumidity700hPa { get; set; }
        public double? RelativeHumidity500hPa { get; set; }
        public double? U700 { get; set; }
        public double? V700 { get; set; }
        public double? VerticalVelocity700hPa { get; set; }
        public double? VerticalVelocity500hPa { get; set; }
        public double? SoilTemperature0to7cm { get; set; }
        public double? SoilMoisture0to7cm { get; set; }
    }
}
