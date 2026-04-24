using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Met Office Weather DataHub — Land Observations (hourly, rolling 48h window).
///
/// Two-step API per the v1.0 OpenAPI spec:
/// <list type="number">
///   <item><c>GET /nearest?lat=...&amp;lon=...</c> → list of {geohash, area, region, country, olson_time_zone}.
///     The spec asks callers to cache this — geohashes don't change.</item>
///   <item><c>GET /{geohash}</c> → flat JSON array of hourly observations.</item>
/// </list>
///
/// Auth is the <c>apikey</c> header (same as Spot). Fails soft: any non-200
/// returns an empty list and logs the response body so the cycle doesn't break.
/// </summary>
public sealed class MetOfficeObservationsClient
{
    public const string BaseUrl = "https://data.hub.api.metoffice.gov.uk/observation-land/1";

    private readonly HttpClient _http;
    private readonly ILogger<MetOfficeObservationsClient> _log;

    public MetOfficeObservationsClient(HttpClient http, ILogger<MetOfficeObservationsClient> log)
    {
        _http = http;
        _log = log;
    }

    public sealed record NearestLocation(string Geohash, string Area, string? Region, string? Country);

    public async Task<NearestLocation?> FindNearestAsync(
        double lat,
        double lon,
        string apiKey,
        CancellationToken ct)
    {
        // Spec: lat/lon must be at most 2 decimal places.
        var latStr = lat.ToString("F2", CultureInfo.InvariantCulture);
        var lonStr = lon.ToString("F2", CultureInfo.InvariantCulture);
        var url = $"{BaseUrl}/nearest?lat={latStr}&lon={lonStr}";

        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.Add("apikey", apiKey);
        req.Headers.Accept.ParseAdd("application/json");

        _log.LogInformation("GET {Url}", url);
        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            _log.LogWarning("Met Office /nearest returned {Status}: {Body}",
                (int)resp.StatusCode, Truncate(body, 400));
            return null;
        }

        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return ParseNearest(doc.RootElement);
    }

    internal static NearestLocation? ParseNearest(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Array) return null;
        foreach (var el in root.EnumerateArray())
        {
            var gh = GetString(el, "geohash");
            if (string.IsNullOrWhiteSpace(gh)) continue;
            return new NearestLocation(
                gh,
                GetString(el, "area") ?? "",
                GetString(el, "region"),
                GetString(el, "country"));
        }
        return null;
    }

    public async Task<List<MetOfficeObservationRow>> FetchAsync(
        string locationName,
        string geohash,
        string area,
        string apiKey,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(geohash))
        {
            _log.LogInformation("Met Office obs geohash not configured; skipping");
            return new();
        }

        var url = $"{BaseUrl}/{Uri.EscapeDataString(geohash)}";
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.Add("apikey", apiKey);
        req.Headers.Accept.ParseAdd("application/json");

        _log.LogInformation("GET {Url}", url);
        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            _log.LogWarning("Met Office observations returned {Status}: {Body}",
                (int)resp.StatusCode, Truncate(body, 400));
            return new();
        }

        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return Parse(doc.RootElement, locationName, geohash, area);
    }

    internal static List<MetOfficeObservationRow> Parse(
        JsonElement root,
        string locationName,
        string geohash,
        string area)
    {
        var rows = new List<MetOfficeObservationRow>();
        if (root.ValueKind != JsonValueKind.Array) return rows;

        foreach (var step in root.EnumerateArray())
        {
            var t = ParseDateTime(step, "datetime");
            if (!t.HasValue) continue;

            rows.Add(new MetOfficeObservationRow
            {
                LocationName = locationName,
                Geohash = geohash,
                Area = area,
                ObservedTimeUtc = t.Value,
                AirTemperature = GetDouble(step, "temperature"),
                Humidity = GetDouble(step, "humidity"),
                WindSpeed10m = GetDouble(step, "wind_speed"),
                WindDirection10m = CompassToDegrees(GetString(step, "wind_direction")),
                WindGusts10m = GetDouble(step, "wind_gust"),
                MeanSeaLevelPressure = GetDouble(step, "mslp"),
                Visibility = GetDouble(step, "visibility"),
                WeatherCode = GetInt(step, "weather_code"),
                PressureTendency = GetString(step, "pressure_tendency"),
            });
        }

        return rows.OrderBy(r => r.ObservedTimeUtc).ToList();
    }

    // Met Office returns wind direction in a 16-point compass string (e.g. "SWW").
    // Table values are the midpoints of each 22.5° sector, starting N=0° and
    // rotating clockwise. The API uses "N..E..S..W" letters but sometimes
    // inverts the usual ordering (e.g. "SWW" for WSW); treat both forms.
    internal static double? CompassToDegrees(string? compass)
    {
        if (string.IsNullOrWhiteSpace(compass)) return null;
        return compass.Trim().ToUpperInvariant() switch
        {
            "N" => 0,
            "NNE" => 22.5,
            "NE" => 45,
            "ENE" or "NEE" => 67.5,
            "E" => 90,
            "ESE" or "SEE" => 112.5,
            "SE" => 135,
            "SSE" => 157.5,
            "S" => 180,
            "SSW" => 202.5,
            "SW" => 225,
            "WSW" or "SWW" => 247.5,
            "W" => 270,
            "WNW" or "NWW" => 292.5,
            "NW" => 315,
            "NNW" => 337.5,
            _ => null,
        };
    }

    private static DateTime? ParseDateTime(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el) || el.ValueKind != JsonValueKind.String) return null;
        if (!DateTime.TryParse(el.GetString(), CultureInfo.InvariantCulture,
            DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var dt)) return null;
        return DateTime.SpecifyKind(dt, DateTimeKind.Utc);
    }

    private static double? GetDouble(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        if (el.ValueKind == JsonValueKind.Number) return el.GetDouble();
        return null;
    }

    private static int? GetInt(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var i)) return i;
        return null;
    }

    private static string? GetString(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        return el.ValueKind == JsonValueKind.String ? el.GetString() : null;
    }

    private static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "…";
}
