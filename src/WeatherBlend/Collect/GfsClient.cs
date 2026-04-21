using System.Globalization;
using System.Net.Http.Headers;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches NOAA GFS forecasts directly from the public S3 bucket
/// <c>s3://noaa-gfs-bdp-pds/</c>. Unlike the Open-Meteo collector, the files here
/// are named by real cycle (00/06/12/18Z) and lead hour, so every row carries an
/// exact <see cref="ForecastRow.RunTimeUtc"/> and <see cref="ForecastRow.LeadHours"/>
/// — tagged with <see cref="RunTimeSources.Exact"/>.
///
/// Point extraction uses <see cref="Wgrib2"/> after HTTP Range-requesting only the
/// GRIB2 messages we care about (resolved via the <c>.idx</c> sidecar). That keeps
/// a typical cycle pull to a few MB of wire traffic instead of the ~400 MB full file.
/// </summary>
public sealed class GfsClient
{
    // Bucket root: s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/CC/atmos/gfs.tCCz.pgrb2.0p25.fNNN
    private const string BaseUrl = "https://noaa-gfs-bdp-pds.s3.amazonaws.com";
    private const string ModelId = "gfs_ncep";
    public static readonly int[] CycleHours = { 0, 6, 12, 18 };

    // Variables we pull from every lead. Each tuple is (idxMatch, setter).
    // idxMatch is searched as a substring against each .idx line; the first hit
    // wins (they're all unique within a cycle because we pair variable + level).
    private static readonly IReadOnlyList<VarMap> Variables = new[]
    {
        new VarMap("TMP:2 m above ground", (r, v) => r.Temperature2m = v - 273.15),  // K→°C
        new VarMap("DPT:2 m above ground", (r, v) => r.DewPoint2m    = v - 273.15),
        new VarMap("RH:2 m above ground",  (r, v) => r.RelativeHumidity2m = v),
        new VarMap("UGRD:10 m above ground", (r, v) => r.U10 = v),
        new VarMap("VGRD:10 m above ground", (r, v) => r.V10 = v),
        new VarMap("GUST:surface",           (r, v) => r.WindGusts10m = v),
        new VarMap("PRES:surface",           (r, v) => r.SurfacePressure = v / 100.0),  // Pa→hPa
        new VarMap("TCDC:entire atmosphere", (r, v) => r.CloudCover = v),
        new VarMap("CAPE:surface",           (r, v) => r.Cape = v),
    };

    private readonly HttpClient _http;
    private readonly Wgrib2 _wgrib2;
    private readonly ILogger<GfsClient> _log;

    public GfsClient(HttpClient http, Wgrib2 wgrib2, ILogger<GfsClient> log)
    {
        _http = http;
        _wgrib2 = wgrib2;
        _log = log;
    }

    /// <summary>
    /// Fetch one cycle's forecasts for a given set of lead hours. Returns one
    /// <see cref="ForecastRow"/> per lead hour, with the full variable set.
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
                _log.LogWarning("GFS cycle {Date} t{CC:00}z f{Lead:000} not available (404) — skipping",
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
        var file = $"gfs.t{cycleHour:00}z.pgrb2.0p25.f{lead:000}";
        var baseUrl = $"{BaseUrl}/gfs.{dateStr}/{cycleHour:00}/atmos/{file}";

        // 1) Fetch .idx to locate message offsets.
        var idxLines = await FetchIdxAsync(baseUrl + ".idx", ct);
        if (idxLines.Count == 0) return null;

        // 2) For each variable we want, find its offset range in the file.
        var picks = new List<(string Var, long Start, long End)>();
        for (int i = 0; i < idxLines.Count; i++)
        {
            foreach (var v in Variables)
            {
                if (picks.Any(p => p.Var == v.IdxMatch)) continue;  // already picked
                if (idxLines[i].Contains(v.IdxMatch, StringComparison.Ordinal))
                {
                    var start = ParseOffset(idxLines[i]);
                    var end = (i + 1 < idxLines.Count) ? ParseOffset(idxLines[i + 1]) - 1 : -1;  // -1 = open
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

        // 3) Range-download each message and concatenate to a temp GRIB file.
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

            // 4) Extract point values via wgrib2 — one val per message in download order.
            var vals = await _wgrib2.ExtractPointAsync(tmpGrib, loc.Latitude, loc.Longitude, ct);
            if (vals.Count != picks.Count)
            {
                _log.LogWarning("wgrib2 returned {Got} vals, expected {Want} for {File}",
                    vals.Count, picks.Count, file);
            }

            var raw = new RawGfsValues();
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
                DewPoint2m = raw.DewPoint2m,
                RelativeHumidity2m = raw.RelativeHumidity2m,
                WindSpeed10m = raw.U10 is { } u && raw.V10 is { } v ? Math.Sqrt(u * u + v * v) : null,
                WindDirection10m = raw.U10 is { } uu && raw.V10 is { } vv ? WindDirection(uu, vv) : null,
                WindGusts10m = raw.WindGusts10m,
                SurfacePressure = raw.SurfacePressure,
                CloudCover = raw.CloudCover,
                Cape = raw.Cape,
            };
        }
        finally
        {
            try { File.Delete(tmpGrib); } catch { /* best-effort */ }
        }
    }

    private async Task<List<string>> FetchIdxAsync(string url, CancellationToken ct)
    {
        using var resp = await _http.GetAsync(url, ct);
        if (resp.StatusCode == System.Net.HttpStatusCode.NotFound)
            throw new HttpRequestException($"idx not found: {url}", null, resp.StatusCode);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadAsStringAsync(ct);
        return body.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();
    }

    private static long ParseOffset(string idxLine)
    {
        // Format "<n>:<offset>:d=<date>:<var>:<level>:<lead>:"
        var parts = idxLine.Split(':', 3);
        return long.Parse(parts[1], CultureInfo.InvariantCulture);
    }

    /// <summary>Meteorological wind direction (° from which wind blows) from u/v.</summary>
    private static double WindDirection(double u, double v)
    {
        var deg = (Math.Atan2(-u, -v) * 180.0 / Math.PI + 360.0) % 360.0;
        return deg;
    }

    private sealed record VarMap(string IdxMatch, Action<RawGfsValues, double> Apply);

    private sealed class RawGfsValues
    {
        public double? Temperature2m { get; set; }
        public double? DewPoint2m { get; set; }
        public double? RelativeHumidity2m { get; set; }
        public double? U10 { get; set; }
        public double? V10 { get; set; }
        public double? WindGusts10m { get; set; }
        public double? SurfacePressure { get; set; }
        public double? CloudCover { get; set; }
        public double? Cape { get; set; }
    }
}
