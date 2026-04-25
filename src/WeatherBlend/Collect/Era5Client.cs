using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Open-Meteo ERA5 reanalysis archive: archive-api.open-meteo.com/v1/archive
/// Gapless hourly truth back to 1940. We use this as the training target —
/// quantitative precip, no missing hours, same query pattern as the forecast APIs.
/// </summary>
public sealed class Era5Client
{
    private const string BaseUrl = "https://archive-api.open-meteo.com/v1/archive";

    private readonly HttpClient _http;
    private readonly ILogger<Era5Client> _log;
    private readonly AppConfig _cfg;

    public Era5Client(HttpClient http, ILogger<Era5Client> log, AppConfig cfg)
    {
        _http = http;
        _log = log;
        _cfg = cfg;
    }

    public async Task<List<Era5Row>> FetchAsync(
        LocationConfig location,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
    {
        var variables = _cfg.Variables.Era5;
        var hourly = string.Join(",", variables);
        var qs = new[]
        {
            $"latitude={location.Latitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"longitude={location.Longitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"elevation={location.ElevationMeters.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"hourly={Uri.EscapeDataString(hourly)}",
            "models=era5",
            "timezone=UTC",
            "windspeed_unit=ms",
            "temperature_unit=celsius",
            "precipitation_unit=mm",
            $"start_date={startDate:yyyy-MM-dd}",
            $"end_date={endDate:yyyy-MM-dd}"
        };
        var url = $"{BaseUrl}?{string.Join('&', qs)}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return Parse(doc.RootElement, location.Name);
    }

    internal static List<Era5Row> Parse(JsonElement root, string locationName)
    {
        if (!root.TryGetProperty("hourly", out var hourly)) return new();

        var times = hourly.GetProperty("time").EnumerateArray()
            .Select(e => DateTime.SpecifyKind(DateTime.Parse(e.GetString()!), DateTimeKind.Utc))
            .ToArray();

        double?[] Col(string name) =>
            hourly.TryGetProperty(name, out var arr)
                ? arr.EnumerateArray()
                     .Select(e => e.ValueKind == JsonValueKind.Null ? (double?)null : e.GetDouble())
                     .ToArray()
                : Enumerable.Repeat<double?>(null, times.Length).ToArray();

        var t2m = Col("temperature_2m"); var td = Col("dew_point_2m"); var rh = Col("relative_humidity_2m");
        var pr = Col("precipitation"); var rn = Col("rain"); var sn = Col("snowfall");
        var cc = Col("cloud_cover"); var ccl = Col("cloud_cover_low");
        var ccm = Col("cloud_cover_mid"); var cch = Col("cloud_cover_high");
        var ws = Col("wind_speed_10m"); var wd = Col("wind_direction_10m"); var wg = Col("wind_gusts_10m");
        var sp = Col("surface_pressure"); var vis = Col("visibility");
        var swr = Col("shortwave_radiation");
        var drr = Col("direct_radiation");
        var dfr = Col("diffuse_radiation");

        var rows = new List<Era5Row>(times.Length);
        for (int i = 0; i < times.Length; i++)
        {
            rows.Add(new Era5Row
            {
                LocationName = locationName,
                ValidTimeUtc = times[i],
                Temperature2m = t2m[i], DewPoint2m = td[i], RelativeHumidity2m = rh[i],
                Precipitation = pr[i], Rain = rn[i], Snowfall = sn[i],
                CloudCover = cc[i], CloudCoverLow = ccl[i], CloudCoverMid = ccm[i], CloudCoverHigh = cch[i],
                WindSpeed10m = ws[i], WindDirection10m = wd[i], WindGusts10m = wg[i],
                SurfacePressure = sp[i], Visibility = vis[i],
                ShortwaveRadiation = swr[i], DirectRadiation = drr[i], DiffuseRadiation = dfr[i]
            });
        }
        return rows;
    }
}
