using System.Globalization;
using System.Net.Http.Headers;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches NOAA GEFS (Global Ensemble Forecast System) ensemble-mean forecasts
/// from the public S3 bucket <c>s3://noaa-gefs-pds/</c>. Mirrors
/// <see cref="GfsClient"/>'s pattern (idx-driven byte-range pulls + wgrib2
/// point extraction) but reads the <c>geavg</c> member — the 31-member
/// ensemble's mean — from the <c>pgrb2ap5</c> 0.5° product.
///
/// **Why ensemble mean and not the full 31 members?** A blender feature is one
/// scalar per (model, valid_time, lead). Adding one ensemble per member would
/// be 31× the column count for marginal gain — most of GEFS's signal is in the
/// mean (smoothed deterministic trajectory). Spread (std across members) is a
/// natural follow-on confidence feature once we've validated the mean.
///
/// Path layout:
///   <c>gefs.YYYYMMDD/HH/atmos/pgrb2ap5/geavg.tHHz.pgrb2a.0p50.fLLL{,.idx}</c>
///   * Cycles: 4/day at 00/06/12/18 UTC.
///   * Members on disk alongside <c>geavg</c>: <c>gec00</c> (control),
///     <c>gep01..gep30</c> (perturbations) — we pull <c>geavg</c> only.
///   * Lead grid: 3-hourly through f240, 6-hourly to f384. Backfill is capped
///     at f120 to match the other exact-runtime sources we feed into 2d/3d.
///   * Resolution: 0.5° (<c>pgrb2ap5</c>) — the global subset is also
///     published as 0.25° in <c>pgrb2sp25</c>; we stay at 0.5° for parity
///     with GEFS literature and because Bonehill's 0.5° cell is unambiguous.
///
/// Archive depth: back to 2017-01-01 per agent probe. Comfortable margin on
/// the 1-2 year training window the blender needs.
///
/// Stamped <c>Model = "gefs_ncep_mean"</c> on every row to distinguish from
/// the deterministic <c>gfs_ncep</c> column already in the feature schemas
/// (different physics, different resolution, ensemble-derived). Distinct from
/// Open-Meteo's <c>gfs_seamless</c> too.
/// </summary>
public sealed class GefsClient
{
    private const string BaseUrl = "https://noaa-gefs-pds.s3.amazonaws.com";
    private const string ModelId = "gefs_ncep_mean";
    public static readonly int[] CycleHours = { 0, 6, 12, 18 };

    // Variable picks mirror GfsClient where the GEFS pgrb2a 0.5° product
    // publishes the same surface field. CAPE / VIS / cloud-base / boundary-
    // layer cloud are NOT in pgrb2a (only in pgrb2sp25 or pgrb2bp5); leave
    // them off rather than splitting the fetch across multiple files.
    // Accumulated APCP is published over the prior interval (3h or 6h
    // depending on lead) — same convention as GFS deterministic.
    private static readonly IReadOnlyList<VarMap> Variables = new[]
    {
        new VarMap("TMP:2 m above ground:",  (r, v) => r.Temperature2m       = v - 273.15),  // K → °C
        new VarMap("RH:2 m above ground:",   (r, v) => r.RelativeHumidity2m  = v),
        new VarMap("UGRD:10 m above ground:",(r, v) => r.U10                 = v),
        new VarMap("VGRD:10 m above ground:",(r, v) => r.V10                 = v),
        new VarMap("PRES:surface:",          (r, v) => r.SurfacePressure     = v / 100.0),   // Pa → hPa
        new VarMap("PRMSL:mean sea level:",  (r, v) => r.PressureMsl         = v / 100.0),   // Pa → hPa
        new VarMap("APCP:surface:",          (r, v) => r.Precipitation       = v),           // mm (kg/m² ≡ mm)
        new VarMap("TCDC:entire atmosphere:",(r, v) => r.CloudCover          = v),
        new VarMap("DSWRF:surface:",         (r, v) => r.ShortwaveRadiation  = v),
        // ── Added 2026-05-31: multi-level pressure fields (850/700/500 hPa) ──
        // pgrb2a 0.5° DOES carry the isobaric TMP/HGT/RH/UGRD/VGRD set (verified
        // against the live .idx), so the ensemble mean contributes the same
        // upper-air columns as the deterministic exact models. Trailing ':' is
        // doubly safe here — GEFS idx lines append ":ens mean", so the matcher
        // hits "TMP:850 mb:24 hour fcst:ens mean". See GfsClient for rationale.
        new VarMap("TMP:850 mb:",  (r, v) => r.Temperature850hPa = v - 273.15),
        new VarMap("TMP:700 mb:",  (r, v) => r.Temperature700hPa = v - 273.15),
        new VarMap("TMP:500 mb:",  (r, v) => r.Temperature500hPa = v - 273.15),
        new VarMap("HGT:850 mb:",  (r, v) => r.GeopotentialHeight850hPa = v),  // gpm
        new VarMap("HGT:500 mb:",  (r, v) => r.GeopotentialHeight500hPa = v),
        new VarMap("RH:850 mb:",   (r, v) => r.RelativeHumidity850hPa = v),
        new VarMap("UGRD:850 mb:", (r, v) => r.U850 = v),
        new VarMap("VGRD:850 mb:", (r, v) => r.V850 = v),
        new VarMap("UGRD:500 mb:", (r, v) => r.U500 = v),
        new VarMap("VGRD:500 mb:", (r, v) => r.V500 = v),
    };

    private readonly HttpClient _http;
    private readonly Wgrib2 _wgrib2;
    private readonly ILogger<GefsClient> _log;

    public GefsClient(HttpClient http, Wgrib2 wgrib2, ILogger<GefsClient> log)
    {
        _http = http;
        _wgrib2 = wgrib2;
        _log = log;
    }

    /// <summary>
    /// Fetch one cycle's ensemble-mean forecasts for a given lead set.
    /// Returns one <see cref="ForecastRow"/> per lead.
    /// </summary>
    public async Task<List<ForecastRow>> FetchCycleAsync(
        LocationConfig location,
        DateOnly cycleDate,
        int cycleHour,
        IEnumerable<int> leadHours,
        string scratchDir,
        CancellationToken ct)
    {
        Directory.CreateDirectory(scratchDir);
        var runTime = new DateTime(cycleDate.Year, cycleDate.Month, cycleDate.Day,
            cycleHour, 0, 0, DateTimeKind.Utc);

        var rows = new List<ForecastRow>();
        foreach (var lead in leadHours.OrderBy(h => h))
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var row = await FetchLeadAsync(location, cycleDate, cycleHour, lead, runTime, scratchDir, ct);
                if (row is not null) rows.Add(row);
            }
            catch (HttpRequestException e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                _log.LogWarning("GEFS cycle {Date} t{CC:00}z f{Lead:000} not available (404) — skipping",
                    cycleDate, cycleHour, lead);
            }
        }
        return rows;
    }

    private async Task<ForecastRow?> FetchLeadAsync(
        LocationConfig loc, DateOnly date, int cycleHour, int lead,
        DateTime runTime, string scratchDir, CancellationToken ct)
    {
        var dateStr = date.ToString("yyyyMMdd");
        var file = $"geavg.t{cycleHour:00}z.pgrb2a.0p50.f{lead:000}";
        var baseUrl = $"{BaseUrl}/gefs.{dateStr}/{cycleHour:00}/atmos/pgrb2ap5/{file}";

        // 1) Fetch .idx to locate message offsets. Same pattern as GfsClient
        //    but the GEFS idx tags every line with ":ens mean" — our
        //    substring matchers look for the (var, level) prefix which is
        //    unique within the file regardless of the trailing tag.
        var idxLines = await FetchIdxAsync(baseUrl + ".idx", ct);
        if (idxLines.Count == 0) return null;

        var picks = new List<(string Var, long Start, long End)>();
        for (int i = 0; i < idxLines.Count; i++)
        {
            foreach (var v in Variables)
            {
                if (picks.Any(p => p.Var == v.IdxMatch)) continue;
                if (idxLines[i].Contains(v.IdxMatch, StringComparison.Ordinal))
                {
                    var start = ParseOffset(idxLines[i]);
                    var end = (i + 1 < idxLines.Count) ? ParseOffset(idxLines[i + 1]) - 1 : -1;
                    picks.Add((v.IdxMatch, start, end));
                    break;
                }
            }
        }
        if (picks.Count == 0)
        {
            _log.LogWarning("No matching variables in idx for {File}", file);
            return null;
        }

        var tmpGrib = Path.Combine(scratchDir, $"{Guid.NewGuid():N}.grib2");
        try
        {
            await using (var outFs = File.Create(tmpGrib))
            {
                foreach (var (_, start, end) in picks)
                {
                    using var req = new HttpRequestMessage(HttpMethod.Get, baseUrl);
                    req.Headers.Range = end >= 0
                        ? new RangeHeaderValue(start, end)
                        : new RangeHeaderValue(start, null);
                    using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
                    resp.EnsureSuccessStatusCode();
                    await using var inS = await resp.Content.ReadAsStreamAsync(ct);
                    await inS.CopyToAsync(outFs, ct);
                }
            }

            var vals = await _wgrib2.ExtractPointAsync(tmpGrib, loc.Latitude, loc.Longitude, ct);
            if (vals.Count != picks.Count)
            {
                _log.LogWarning("wgrib2 returned {Got} vals, expected {Want} for {File}",
                    vals.Count, picks.Count, file);
            }

            var raw = new RawGefsValues();
            for (int i = 0; i < Math.Min(vals.Count, picks.Count); i++)
                Variables.First(v => v.IdxMatch == picks[i].Var).Apply(raw, vals[i].Value);

            var validTime = runTime.AddHours(lead);
            return new ForecastRow
            {
                LocationName = loc.Name,
                Model = ModelId,
                RunTimeUtc = runTime,
                ValidTimeUtc = validTime,
                LeadHours = lead,
                RunTimeSource = RunTimeSources.Exact,
                Temperature2m = raw.Temperature2m,
                RelativeHumidity2m = raw.RelativeHumidity2m,
                WindSpeed10m = raw.U10 is { } u && raw.V10 is { } v ? Math.Sqrt(u * u + v * v) : null,
                WindDirection10m = raw.U10 is { } uu && raw.V10 is { } vv ? WindDirection(uu, vv) : null,
                SurfacePressure = raw.SurfacePressure,
                PressureMsl = raw.PressureMsl,
                CloudCover = raw.CloudCover,
                Precipitation = raw.Precipitation,
                ShortwaveRadiation = raw.ShortwaveRadiation,
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
            };
        }
        finally
        {
            try { File.Delete(tmpGrib); } catch { /* best effort */ }
        }
    }

    private async Task<List<string>> FetchIdxAsync(string url, CancellationToken ct)
    {
        try
        {
            var idxText = await _http.GetStringAsync(url, ct);
            return idxText
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }
        catch (HttpRequestException e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return new List<string>();
        }
    }

    private static long ParseOffset(string idxLine)
    {
        // idx format: "N:OFFSET:d=YYYYMMDDHH:VAR:LEVEL:..."
        var firstColon = idxLine.IndexOf(':');
        if (firstColon < 0) return 0;
        var secondColon = idxLine.IndexOf(':', firstColon + 1);
        if (secondColon < 0) return 0;
        var offsetStr = idxLine.AsSpan(firstColon + 1, secondColon - firstColon - 1);
        return long.Parse(offsetStr, NumberStyles.Integer, CultureInfo.InvariantCulture);
    }

    private static double WindDirection(double u, double v)
    {
        // Meteorological convention: direction FROM which the wind is blowing,
        // measured clockwise from north. Same formula as GfsClient.
        var deg = Math.Atan2(-u, -v) * 180.0 / Math.PI;
        if (deg < 0) deg += 360.0;
        return deg;
    }

    private sealed record VarMap(string IdxMatch, Action<RawGefsValues, double> Apply);

    private sealed class RawGefsValues
    {
        public double? Temperature2m;
        public double? RelativeHumidity2m;
        public double? U10;
        public double? V10;
        public double? SurfacePressure;
        public double? PressureMsl;
        public double? Precipitation;
        public double? CloudCover;
        public double? ShortwaveRadiation;
        // Pressure-level (850/700/500 hPa); wind kept as U/V per level.
        public double? Temperature850hPa;
        public double? Temperature700hPa;
        public double? Temperature500hPa;
        public double? GeopotentialHeight850hPa;
        public double? GeopotentialHeight500hPa;
        public double? RelativeHumidity850hPa;
        public double? U850;
        public double? V850;
        public double? U500;
        public double? V500;
    }
}
