using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches METARs from the aviationweather.gov JSON API.
/// For Bonehill Rocks we use Exeter (EGTE) primary, Yeovilton (EGDY) fallback.
///
/// Caveats: METAR precipitation is coded (RA, SHRA, +RA, etc.) not quantitative.
/// This is fine as a presence/absence signal but insufficient for training an
/// intensity model. Phase 2: add Met Office DataHub for quantitative precip.
/// </summary>
public sealed class MetarClient
{
    private const string BaseUrl = "https://aviationweather.gov/api/data/metar";

    private readonly HttpClient _http;
    private readonly ILogger<MetarClient> _log;

    public MetarClient(HttpClient http, ILogger<MetarClient> log)
    {
        _http = http;
        _log = log;
    }

    public async Task<List<ObservationRow>> FetchAsync(
        string locationName,
        string station,
        int hoursBack,
        CancellationToken ct)
    {
        var url = $"{BaseUrl}?ids={Uri.EscapeDataString(station)}&format=json&hours={hoursBack}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var rows = new List<ObservationRow>();
        if (doc.RootElement.ValueKind != JsonValueKind.Array)
            return rows;

        foreach (var item in doc.RootElement.EnumerateArray())
        {
            var raw = GetString(item, "rawOb") ?? "";
            var reportTime = GetString(item, "reportTime");
            if (reportTime is null) continue;
            var observed = DateTime.SpecifyKind(
                DateTime.Parse(reportTime, CultureInfo.InvariantCulture),
                DateTimeKind.Utc);

            var wx = GetString(item, "wxString") ?? "";
            var codes = wx.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            rows.Add(new ObservationRow
            {
                LocationName = locationName,
                Station = station,
                ObservedTimeUtc = observed,
                RawMetar = raw,
                Temperature2m = GetDouble(item, "temp"),
                DewPoint2m = GetDouble(item, "dewp"),
                WindSpeed10m = KnotsToMs(GetDouble(item, "wspd")),
                WindDirection10m = GetDouble(item, "wdir"),
                WindGusts10m = KnotsToMs(GetDouble(item, "wgst")),
                SurfacePressure = InHgToHpa(GetDouble(item, "altim")),
                Visibility = MilesToMetres(GetDouble(item, "visib")),
                PresentWeather = codes,
                HasPrecipitation = HasPrecipCode(codes)
            });
        }

        _log.LogInformation("Parsed {Count} METARs from {Station}", rows.Count, station);
        return rows;
    }

    private static readonly Regex PrecipRegex =
        new(@"(RA|DZ|SN|SG|PL|GR|GS|IC|UP|SH|TS)", RegexOptions.Compiled);

    private static bool HasPrecipCode(IEnumerable<string> codes)
        => codes.Any(c => PrecipRegex.IsMatch(c));

    private static string? GetString(JsonElement el, string name)
        => el.TryGetProperty(name, out var v) && v.ValueKind != JsonValueKind.Null
           ? v.ToString() : null;

    private static double? GetDouble(JsonElement el, string name)
    {
        if (!el.TryGetProperty(name, out var v) || v.ValueKind == JsonValueKind.Null)
            return null;
        return v.ValueKind switch
        {
            JsonValueKind.Number => v.GetDouble(),
            JsonValueKind.String when double.TryParse(v.GetString(),
                NumberStyles.Any, CultureInfo.InvariantCulture, out var d) => d,
            _ => null
        };
    }

    private static double? KnotsToMs(double? kt) => kt * 0.514444;
    private static double? InHgToHpa(double? inHg) => inHg * 33.8639;
    private static double? MilesToMetres(double? mi) => mi * 1609.34;
}
