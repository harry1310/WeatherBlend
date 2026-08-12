using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Environment Agency Hydrology API: https://environment.data.gov.uk/hydrology
/// OGL v3, no auth. 15-min rainfall totals (mm), interval-end timestamped.
/// Measure URL pattern:
///   /id/measures/&lt;measureId&gt;-rainfall-t-900-mm-qualified/readings.json
/// where &lt;measureId&gt; is the station GUID, optionally suffixed with
/// "_&lt;wiskiID&gt;" for versioned stations (e.g. Bellever = ..._363307).
///
/// EA `dateTime` values are unzoned ISO-8601 strings in UTC per the API spec.
/// We interpret them as UTC and keep QC fields (`quality`, `completeness`)
/// verbatim — downstream code decides what to filter.
/// </summary>
public sealed class EaHydrologyClient
{
    private const string BaseUrl = "https://environment.data.gov.uk/hydrology";
    private const string MeasureSuffix = "-rainfall-t-900-mm-qualified";

    private readonly HttpClient _http;
    private readonly ILogger<EaHydrologyClient> _log;

    public EaHydrologyClient(HttpClient http, ILogger<EaHydrologyClient> log)
    {
        _http = http;
        _log = log;
        // EA Hydrology started returning 403 (Forbidden) to header-less requests
        // on 2026-06-14 (worked 08:46Z, blocked by 14:47Z) — gov APIs commonly
        // reject an empty/default User-Agent. Identify ourselves so the requests
        // are accepted. Guarded so a reused HttpClient isn't double-stamped.
        if (_http.DefaultRequestHeaders.UserAgent.Count == 0)
            _http.DefaultRequestHeaders.UserAgent.ParseAdd(
                "WeatherBlend/1.0 (+https://github.com/harry1310/WeatherBlend)");
    }

    /// <summary>
    /// Fetches 15-minute rainfall readings for a single station across an inclusive
    /// date range [startDate, endDate]. The EA API uses interval-end timestamps, so
    /// a range of 2024-06-01..2024-06-01 returns 96 rows (00:00:00 through 23:45:00).
    /// </summary>
    public async Task<List<RainfallRow>> FetchAsync(
        string locationName,
        RainfallStationConfig station,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
    {
        // Fail fast, loudly, on a station this client cannot address. Without this a
        // WeatherLink gauge (no EA measure id) builds ".../measures/-rainfall-t-900-mm-
        // qualified/..." — a URL that can never resolve, yet still costs three 60s
        // attempt-timeouts before the resilience handler gives up. Callers filter via
        // LocationConfig.EaRainfallStations; this is the backstop that stops a future
        // miswire from silently eating minutes of every collect cycle instead of
        // surfacing on the first run.
        if (station.Source is not RainfallTruthSource.Ea || string.IsNullOrWhiteSpace(station.Id))
            throw new ArgumentException(
                $"Station '{station.Name}' (source={station.Source}, id='{station.Id}') is not an EA " +
                "Hydrology gauge — it has no measure id. Filter with LocationConfig.EaRainfallStations.",
                nameof(station));

        var measureId = $"{station.Id}{MeasureSuffix}";
        var url =
            $"{BaseUrl}/id/measures/{measureId}/readings.json" +
            $"?mineq-date={startDate:yyyy-MM-dd}&maxeq-date={endDate:yyyy-MM-dd}&_limit=10000";

        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        return Parse(doc.RootElement, locationName, station);
    }

    internal static List<RainfallRow> Parse(
        JsonElement root,
        string locationName,
        RainfallStationConfig station)
    {
        var rows = new List<RainfallRow>();
        if (!root.TryGetProperty("items", out var items) || items.ValueKind != JsonValueKind.Array)
            return rows;

        foreach (var item in items.EnumerateArray())
        {
            if (!item.TryGetProperty("dateTime", out var dt)) continue;
            var timeStr = dt.GetString();
            if (string.IsNullOrEmpty(timeStr)) continue;

            // EA returns unzoned ISO like "2024-06-01T00:00:00". API spec says UTC.
            var observed = DateTime.SpecifyKind(
                DateTime.Parse(timeStr, System.Globalization.CultureInfo.InvariantCulture),
                DateTimeKind.Utc);

            double? value = null;
            if (item.TryGetProperty("value", out var v) && v.ValueKind == JsonValueKind.Number)
                value = v.GetDouble();

            var quality = item.TryGetProperty("quality", out var q) && q.ValueKind == JsonValueKind.String
                ? q.GetString() ?? ""
                : "";

            var completeness = item.TryGetProperty("completeness", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString() ?? ""
                : "";

            rows.Add(new RainfallRow
            {
                LocationName = locationName,
                StationId = station.Id,
                StationName = station.Name,
                ObservedTimeUtc = observed,
                Value15MinMm = value,
                Quality = quality,
                Completeness = completeness
            });
        }

        return rows;
    }
}
