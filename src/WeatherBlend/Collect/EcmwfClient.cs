using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches ECMWF deterministic forecasts from the public AWS bucket
/// <c>s3://ecmwf-forecasts/</c>. Both the IFS oper stream and the AIFS AI
/// model live in the same bucket under <c>{date}/{HH}z/{ifs|aifs}/0p25/oper/</c>.
/// One GRIB2 file per (cycle, lead) — different from GFS which packs every
/// lead into one big per-cycle file. Each GRIB2 carries a JSON-Lines
/// <c>.index</c> sidecar with byte offsets per ECMWF param code, so the
/// fetch pattern is "GET .index → range-GET each variable's bytes →
/// concatenate → wgrib2 extract point".
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
    public const string BaseUrl = "https://ecmwf-forecasts.s3.amazonaws.com";
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
        new VarMap("tp",     (r, v) => r.Precipitation       = v * 1000.0),  // m (depth) → mm
        new VarMap("tcc",    (r, v) => r.CloudCover          = v * 100.0),   // 0..1 → %
        new VarMap("ssrd",   (r, v) => r.ShortwaveRadiation  = v / 3600.0),  // J/m² acc → W/m² avg-over-hour
        new VarMap("mucape", (r, v) => r.Cape                = v),           // J/kg
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
    /// </summary>
    public async Task<List<ForecastRow>> FetchCycleAsync(
        LocationConfig location, DateOnly cycleDate, int cycleHour, string stream,
        IEnumerable<int> leadHours, string scratchDir, CancellationToken ct)
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
        foreach (var lead in leadHours.OrderBy(h => h))
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var row = await FetchLeadAsync(location, cycleDate, cycleHour, stream, modelId, lead, runTime, scratchDir, ct);
                if (row is not null) rows.Add(row);
            }
            catch (HttpRequestException e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                _log.LogWarning("Missing {Stream} {Date} {Cycle:00}z f{Lead:000}: {Msg}",
                    stream, cycleDate, cycleHour, lead, e.Message);
            }
        }
        return rows;
    }

    private async Task<ForecastRow?> FetchLeadAsync(
        LocationConfig loc, DateOnly date, int cycleHour, string stream, string modelId,
        int lead, DateTime runTime, string scratchDir, CancellationToken ct)
    {
        // ECMWF Open Data publishes IFS HRES at all 4 daily cycles, but split
        // across two streams:
        //   * 00Z + 12Z → `oper` stream, full long-range (~240h)
        //   * 06Z + 18Z → `scda` stream, short cut-off data assimilation
        //                 (~120h+ horizon, verified 2026-05-06).
        // Path is `{date}/{HH}z/{streamSubpath}/0p25/{streamFolder}/{date}{HH}0000-{lead}h-{streamFolder}-fc.{grib2|index}`
        // where streamSubpath ∈ {ifs, aifs-single} (or aifs legacy) and
        // streamFolder ∈ {oper, scda}. The filename suffix mirrors the folder.
        // AIFS publishes all four cycles in `oper` (no scda variant), so the
        // sub-cycle switch is IFS-only. AIFS subdir was renamed
        // `aifs/` → `aifs-single/` between 2025-02-01 and 2025-03-01 — try
        // modern first, fall back on 404.
        var streamFolder = stream == Streams.IfsOper && (cycleHour == 6 || cycleHour == 18)
            ? "scda"
            : "oper";
        var dateStr = date.ToString("yyyyMMdd");
        var stem = $"{dateStr}{cycleHour:00}0000-{lead}h-{streamFolder}-fc";

        var subpathCandidates = stream == Streams.AifsOper
            ? new[] { "aifs-single", "aifs" }
            : new[] { stream };

        List<IndexEntry>? indexEntries = null;
        string? baseUrl = null;
        foreach (var subpath in subpathCandidates)
        {
            baseUrl = $"{BaseUrl}/{dateStr}/{cycleHour:00}z/{subpath}/0p25/{streamFolder}/{stem}";
            try
            {
                indexEntries = await FetchIndexAsync(baseUrl + ".index", ct);
                break;  // got it
            }
            catch (HttpRequestException e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                // Try next candidate (only relevant for AIFS straddling the
                // 2025-02→03 rename); rethrow to caller after all exhausted.
                if (subpath == subpathCandidates[^1]) throw;
            }
        }
        if (indexEntries is null || indexEntries.Count == 0) return null;

        // 2) For each variable we want, find the FIRST matching index entry
        //    by exact param match. ECMWF param codes ARE unique per (param,
        //    levtype) so a literal equals on `param` for sfc-only entries
        //    works without the substring fuzziness GFS needs.
        var picks = new List<(string Param, long Offset, long Length)>();
        foreach (var v in Variables)
        {
            // FirstOrDefault on a record returns the default-constructed
            // record (Param == null, all defaults) when nothing matches —
            // not actually null on the reference itself. Filter explicitly
            // on Param being non-null so a missed variable just drops out
            // (logged separately when the picks list ends up empty).
            var hit = indexEntries.FirstOrDefault(e =>
                string.Equals(e.Param, v.Param, StringComparison.Ordinal)
                && string.Equals(e.LevType, "sfc", StringComparison.Ordinal));
            if (hit is not null && !string.IsNullOrEmpty(hit.Param))
                picks.Add((v.Param, hit.Offset, hit.Length));
        }
        if (picks.Count == 0)
        {
            _log.LogWarning("No matching variables in .index for {Stem}", stem);
            return null;
        }

        // 3) Range-download each message and concat to a temp GRIB.
        var tmpGrib = Path.Combine(scratchDir, $"{Guid.NewGuid():N}.grib2");
        try
        {
            await using (var outFs = File.Create(tmpGrib))
            {
                foreach (var (_, off, len) in picks)
                {
                    using var req = new HttpRequestMessage(HttpMethod.Get, baseUrl + ".grib2");
                    req.Headers.Range = new RangeHeaderValue(off, off + len - 1);
                    using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
                    resp.EnsureSuccessStatusCode();
                    await using var inS = await resp.Content.ReadAsStreamAsync(ct);
                    await inS.CopyToAsync(outFs, ct);
                }
            }

            // 4) Extract point values via wgrib2 (same as GFS path).
            var vals = await _wgrib2.ExtractPointAsync(tmpGrib, loc.Latitude, loc.Longitude, ct);
            if (vals.Count != picks.Count)
                _log.LogWarning("wgrib2 returned {Got} vals, expected {Want} for {Stem}",
                    vals.Count, picks.Count, stem);

            var raw = new RawEcmwfValues();
            for (int i = 0; i < Math.Min(vals.Count, picks.Count); i++)
                Variables.First(v => v.Param == picks[i].Param).Apply(raw, vals[i].Value);

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

            return new ForecastRow
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
                CloudCover = raw.CloudCover,
                Cape = raw.Cape,
                Precipitation = raw.Precipitation,
                ShortwaveRadiation = raw.ShortwaveRadiation,
            };
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
                Param:   root.GetProperty("param").GetString() ?? "",
                LevType: root.TryGetProperty("levtype", out var lt) ? (lt.GetString() ?? "") : "",
                Offset:  root.GetProperty("_offset").GetInt64(),
                Length:  root.GetProperty("_length").GetInt64()));
        }
        return entries;
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

    private sealed record IndexEntry(string Param, string LevType, long Offset, long Length);
    private sealed record VarMap(string Param, Action<RawEcmwfValues, double> Apply);

    private sealed class RawEcmwfValues
    {
        public double? Temperature2m { get; set; }
        public double? DewPoint2m { get; set; }
        public double? U10 { get; set; }
        public double? V10 { get; set; }
        public double? WindGusts10m { get; set; }
        public double? SurfacePressure { get; set; }
        public double? CloudCover { get; set; }
        public double? Cape { get; set; }
        public double? Precipitation { get; set; }
        public double? ShortwaveRadiation { get; set; }
    }
}
