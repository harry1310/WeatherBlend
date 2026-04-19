using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches forecasts from Open-Meteo.
///
/// Live endpoint:      https://api.open-meteo.com/v1/forecast
/// Historical forecast: https://historical-forecast-api.open-meteo.com/v1/forecast
///
/// The historical endpoint returns past model runs (as they were issued)
/// and is what we use for backfill. Both endpoints share a schema.
/// </summary>
public sealed class OpenMeteoClient
{
    private const string LiveBase = "https://api.open-meteo.com/v1/forecast";
    private const string HistoricalBase = "https://historical-forecast-api.open-meteo.com/v1/forecast";

    private readonly HttpClient _http;
    private readonly ILogger<OpenMeteoClient> _log;

    public OpenMeteoClient(HttpClient http, ILogger<OpenMeteoClient> log)
    {
        _http = http;
        _log = log;
    }

    public Task<List<ForecastRow>> FetchLiveAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        int forecastDays,
        CancellationToken ct)
        => FetchAsync(LiveBase, location, modelId, variables, forecastDays, null, null, ct);

    public Task<List<ForecastRow>> FetchHistoricalAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
        => FetchAsync(HistoricalBase, location, modelId, variables, null, startDate, endDate, ct);

    private async Task<List<ForecastRow>> FetchAsync(
        string baseUrl,
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        int? forecastDays,
        DateOnly? startDate,
        DateOnly? endDate,
        CancellationToken ct)
    {
        var hourly = string.Join(",", variables);
        var qs = new List<string>
        {
            $"latitude={location.Latitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"longitude={location.Longitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"elevation={location.ElevationMeters.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"hourly={Uri.EscapeDataString(hourly)}",
            $"models={Uri.EscapeDataString(modelId)}",
            "timezone=UTC",
            "windspeed_unit=ms",
            "temperature_unit=celsius",
            "precipitation_unit=mm"
        };
        if (forecastDays.HasValue) qs.Add($"forecast_days={forecastDays.Value}");
        if (startDate.HasValue) qs.Add($"start_date={startDate.Value:yyyy-MM-dd}");
        if (endDate.HasValue) qs.Add($"end_date={endDate.Value:yyyy-MM-dd}");

        var url = $"{baseUrl}?{string.Join('&', qs)}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        return Parse(doc.RootElement, location.Name, modelId);
    }

    private static List<ForecastRow> Parse(JsonElement root, string locationName, string modelId)
    {
        if (!root.TryGetProperty("hourly", out var hourly))
            return new();

        var times = hourly.GetProperty("time").EnumerateArray()
            .Select(e => DateTime.SpecifyKind(DateTime.Parse(e.GetString()!), DateTimeKind.Utc))
            .ToArray();

        double?[] Col(string name) =>
            hourly.TryGetProperty(name, out var arr)
                ? arr.EnumerateArray()
                     .Select(e => e.ValueKind == JsonValueKind.Null ? (double?)null : e.GetDouble())
                     .ToArray()
                : Enumerable.Repeat<double?>(null, times.Length).ToArray();

        var t2m   = Col("temperature_2m");
        var td2m  = Col("dew_point_2m");
        var rh2m  = Col("relative_humidity_2m");
        var pr    = Col("precipitation");
        var prp   = Col("precipitation_probability");
        var rain  = Col("rain");
        var show  = Col("showers");
        var snow  = Col("snowfall");
        var cc    = Col("cloud_cover");
        var ccl   = Col("cloud_cover_low");
        var ccm   = Col("cloud_cover_mid");
        var cch   = Col("cloud_cover_high");
        var ws10  = Col("wind_speed_10m");
        var wd10  = Col("wind_direction_10m");
        var wg10  = Col("wind_gusts_10m");
        var sp    = Col("surface_pressure");
        var cape  = Col("cape");
        var vis   = Col("visibility");

        // Open-Meteo doesn't return the run time explicitly.
        // Live endpoint: model run is recent; we approximate run_time as
        // the most recent model cycle hour prior to "now". For the historical
        // endpoint this approximation isn't quite right - the API returns the
        // best-available forecast per valid-time, not a specific run.
        // Phase 2: go direct to ECMWF/NOAA GRIB archives for rigorous run_time tracking.
        var runTime = NowUtcFloorToHour();

        var rows = new List<ForecastRow>(times.Length);
        for (int i = 0; i < times.Length; i++)
        {
            var valid = times[i];
            var lead = (int)Math.Round((valid - runTime).TotalHours);
            rows.Add(new ForecastRow
            {
                LocationName = locationName,
                Model = modelId,
                RunTimeUtc = runTime,
                ValidTimeUtc = valid,
                LeadHours = lead,
                Temperature2m = t2m[i],
                DewPoint2m = td2m[i],
                RelativeHumidity2m = rh2m[i],
                Precipitation = pr[i],
                PrecipitationProbability = prp[i],
                Rain = rain[i],
                Showers = show[i],
                Snowfall = snow[i],
                CloudCover = cc[i],
                CloudCoverLow = ccl[i],
                CloudCoverMid = ccm[i],
                CloudCoverHigh = cch[i],
                WindSpeed10m = ws10[i],
                WindDirection10m = wd10[i],
                WindGusts10m = wg10[i],
                SurfacePressure = sp[i],
                Cape = cape[i],
                Visibility = vis[i]
            });
        }
        return rows;
    }

    private static DateTime NowUtcFloorToHour()
    {
        var now = DateTime.UtcNow;
        return new DateTime(now.Year, now.Month, now.Day, now.Hour, 0, 0, DateTimeKind.Utc);
    }
}
