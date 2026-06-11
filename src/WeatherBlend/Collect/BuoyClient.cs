using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Wave-buoy verification truth — two complementary feeds (both probed live
/// 2026-06-11, see docs/SENNEN_SEA_STATE_PLAN.md §Data sources):
///
///   * REALTIME — Cefas WaveNet's open JSON API
///     (wavenet-api.cefas.co.uk/api/Detail/Results/{id}/{source}): the last
///     ~day of records per station, no auth. Latency ≤10 min (Sevenstones)
///     to ~90 min (CCO waveriders). NOT QC'd.
///   * ARCHIVE — EMODnet Physics ERDDAP (Copernicus INSTAC mirror,
///     er2webapps.emodnet-physics.eu/erddap): per-variable collections
///     TS_{VAR}_INSTAC, CSV, no auth, CC-BY-SA, hourly/30-min from
///     2018-01-01 (collection floor — older buoy years exist only behind
///     registration walls). QC'd; mirror lags days–weeks, so it's the
///     backfill source, never the realtime one.
///
/// Both write the same WaveTruthRow shape; the merge-dedup writer means a
/// later archive re-pull silently upgrades non-QC realtime rows in place.
/// Sevenstones (the primary, fully Atlantic-exposed buoy) reports Hs/Tz/SST
/// only — direction and peak period come from the directional waveriders
/// (SW Scilly, Penzance), whose directions are PEAK direction (VPED), not
/// mean.
/// </summary>
public sealed class BuoyClient
{
    private const string WaveNetBase = "https://wavenet-api.cefas.co.uk/api";
    private const string ErddapBase = "https://er2webapps.emodnet-physics.eu/erddap/tabledap";

    /// <summary>ERDDAP variable collections worth pulling. Missing
    /// (platform, variable) combinations 404 and are skipped quietly.</summary>
    internal static readonly string[] ArchiveVariables = { "VHM0", "VTZA", "VTPK", "VPED", "VPSP" };

    private readonly HttpClient _http;
    private readonly ILogger<BuoyClient> _log;

    public BuoyClient(HttpClient http, ILogger<BuoyClient> log)
    {
        _http = http;
        _log = log;
    }

    /// <summary>Last ~24h of records for one buoy from WaveNet.</summary>
    public async Task<List<WaveTruthRow>> FetchRealtimeAsync(
        string locationName, BuoyConfig buoy, CancellationToken ct)
    {
        var url = $"{WaveNetBase}/Detail/Results/{Uri.EscapeDataString(buoy.WavenetId)}/" +
                  $"{Uri.EscapeDataString(buoy.WavenetSource)}?showForecast=false";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return ParseRealtime(doc.RootElement, locationName, buoy.Slug);
    }

    /// <summary>QC'd archive rows for one buoy from EMODnet ERDDAP — one
    /// request per variable collection, merged on timestamp.</summary>
    public async Task<List<WaveTruthRow>> FetchArchiveAsync(
        string locationName, BuoyConfig buoy, DateOnly startDate, DateOnly endDate, CancellationToken ct)
    {
        var byVar = new Dictionary<string, Dictionary<DateTime, double>>();
        foreach (var varCode in ArchiveVariables)
        {
            ct.ThrowIfCancellationRequested();
            var url = $"{ErddapBase}/TS_{varCode}_INSTAC.csv" +
                      $"?PLATFORMCODE%2Ctime%2C{varCode}%2C{varCode}_QC" +
                      $"&PLATFORMCODE=%22{Uri.EscapeDataString(buoy.PlatformCode)}%22" +
                      $"&time%3E={startDate:yyyy-MM-dd}T00%3A00%3A00Z" +
                      $"&time%3C={endDate:yyyy-MM-dd}T23%3A59%3A59Z";
            _log.LogInformation("GET {Url}", url);

            using var resp = await _http.GetAsync(url, ct);
            if (resp.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                // ERDDAP 404s when the (platform, window, variable) slice has
                // no rows — normal for non-directional buoys / pre-deployment
                // windows, not an error.
                _log.LogInformation("  {Slug}/{Var}: no data for window (404)", buoy.Slug, varCode);
                continue;
            }
            resp.EnsureSuccessStatusCode();
            var csv = await resp.Content.ReadAsStringAsync(ct);
            byVar[varCode] = ParseErddapCsv(csv, varCode);
        }
        return MergeArchive(byVar, locationName, buoy.Slug);
    }

    internal static List<WaveTruthRow> ParseRealtime(JsonElement root, string locationName, string sourceSlug)
    {
        if (root.ValueKind != JsonValueKind.Array) return new();

        // Records can repeat a timestamp (telemetry revisions) — last wins,
        // matching the writer's dedup policy.
        var byTime = new Dictionary<DateTime, WaveTruthRow>();
        foreach (var rec in root.EnumerateArray())
        {
            if (rec.TryGetProperty("isForecast", out var fc) && fc.ValueKind == JsonValueKind.True)
                continue;
            if (!rec.TryGetProperty("timestamp", out var tsEl) || tsEl.ValueKind != JsonValueKind.String)
                continue;
            var ts = DateTime.SpecifyKind(
                DateTime.Parse(tsEl.GetString()!, CultureInfo.InvariantCulture), DateTimeKind.Utc);

            double? hm0 = null, tz = null, tp = null, pdir = null, spr = null, temp = null;
            if (rec.TryGetProperty("results", out var results) && results.ValueKind == JsonValueKind.Array)
            {
                foreach (var r in results.EnumerateArray())
                {
                    if (!r.TryGetProperty("identifier", out var idEl) ||
                        !r.TryGetProperty("value", out var valEl))
                        continue;
                    if (!double.TryParse(valEl.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                        continue;
                    switch (idEl.GetString())
                    {
                        case "Hm0": hm0 = v; break;
                        case "Tz": tz = v; break;
                        case "Tpeak": tp = v; break;
                        case "W_PDIR": pdir = v; break;
                        case "W_SPR": spr = v; break;
                        case "TEMP": temp = v; break;
                    }
                }
            }
            if (hm0 is null && tz is null && tp is null) continue;

            byTime[ts] = new WaveTruthRow
            {
                LocationName = locationName,
                Source = sourceSlug,
                ValidTimeUtc = ts,
                WaveHeight = hm0,
                WavePeriod = tz,
                PeakPeriod = tp,
                WaveDirection = pdir,
                DirectionalSpread = spr,
                SeaSurfaceTemperature = temp,
            };
        }
        return byTime.Values.OrderBy(r => r.ValidTimeUtc).ToList();
    }

    /// <summary>ERDDAP CSV: header row, units row, then data. Rows arrive
    /// duplicated per delivery mode (QC'd + raw-NaN) — keep QC flag 1 only.</summary>
    internal static Dictionary<DateTime, double> ParseErddapCsv(string csv, string varCode)
    {
        var values = new Dictionary<DateTime, double>();
        var lines = csv.Split('\n');
        for (int i = 2; i < lines.Length; i++)
        {
            var line = lines[i].TrimEnd('\r');
            if (line.Length == 0) continue;
            var parts = line.Split(',');
            if (parts.Length < 4) continue;
            if (parts[3].Trim() != "1") continue;
            if (!double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) continue;
            if (double.IsNaN(v)) continue;
            if (!DateTime.TryParse(parts[1], CultureInfo.InvariantCulture,
                    DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var ts)) continue;
            values[ts] = v;
        }
        return values;
    }

    internal static List<WaveTruthRow> MergeArchive(
        IReadOnlyDictionary<string, Dictionary<DateTime, double>> byVar,
        string locationName,
        string sourceSlug)
    {
        var allTimes = byVar.Values.SelectMany(d => d.Keys).Distinct().OrderBy(t => t);
        double? Get(string varCode, DateTime t) =>
            byVar.TryGetValue(varCode, out var d) && d.TryGetValue(t, out var v) ? v : null;

        var rows = new List<WaveTruthRow>();
        foreach (var t in allTimes)
        {
            rows.Add(new WaveTruthRow
            {
                LocationName = locationName,
                Source = sourceSlug,
                ValidTimeUtc = t,
                WaveHeight = Get("VHM0", t),
                WavePeriod = Get("VTZA", t),
                PeakPeriod = Get("VTPK", t),
                WaveDirection = Get("VPED", t),
                DirectionalSpread = Get("VPSP", t),
            });
        }
        return rows;
    }
}
