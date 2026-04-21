using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using RunTimeSources = WeatherBlend.Models.RunTimeSources;

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
    private const string MetadataBase = "https://api.open-meteo.com/data";

    // Our config "seamless" ids don't have metadata endpoints — they're composites.
    // Map each to the short-range / highest-res constituent whose cycle best
    // approximates the front end of the seamless forecast. Docs warn the reported
    // time "does not directly correlate with the update times in the Forecast API",
    // so this is an approximation tagged RunTimeSource=reported, not exact.
    private static readonly IReadOnlyDictionary<string, string> MetadataModelMap = new Dictionary<string, string>
    {
        { "gfs_seamless", "ncep_gfs013" },
        { "ecmwf_ifs025", "ecmwf_ifs025" },
        { "ecmwf_aifs025", "ecmwf_aifs025_single" },
        { "icon_seamless", "dwd_icon" },
        { "meteofrance_seamless", "meteofrance_arpege_europe" },
        { "ukmo_seamless", "ukmo_uk_deterministic_2km" },
        { "gem_seamless", "cmc_gem_gdps" },
    };

    private readonly HttpClient _http;
    private readonly ILogger<OpenMeteoClient> _log;

    public OpenMeteoClient(HttpClient http, ILogger<OpenMeteoClient> log)
    {
        _http = http;
        _log = log;
    }

    public async Task<List<ForecastRow>> FetchLiveAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        int forecastDays,
        CancellationToken ct)
    {
        var reported = await GetReportedCycleAsync(modelId, ct);
        return await FetchAsync(LiveBase, location, modelId, variables,
            forecastDays, null, null, isHistorical: false, reported, ct);
    }

    public Task<List<ForecastRow>> FetchHistoricalAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
        => FetchAsync(HistoricalBase, location, modelId, variables,
            null, startDate, endDate, isHistorical: true, reportedRunTime: null, ct);

    /// <summary>
    /// Hits the Open-Meteo model-metadata endpoint for the supplied config model id
    /// and returns the `last_run_initialisation_time` (unix seconds) as a UTC DateTime.
    /// Null when no mapping exists, the endpoint 404s, or the field is missing —
    /// callers should fall back to the synthesised run-time.
    /// </summary>
    public async Task<DateTime?> GetReportedCycleAsync(string modelId, CancellationToken ct)
    {
        if (!MetadataModelMap.TryGetValue(modelId, out var metaPath))
        {
            _log.LogDebug("No metadata mapping for model {Model}; run-time will be synthesised", modelId);
            return null;
        }

        var url = $"{MetadataBase}/{metaPath}/static/meta.json";
        try
        {
            using var resp = await _http.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode)
            {
                _log.LogWarning("Metadata GET {Url} returned {Status}", url, (int)resp.StatusCode);
                return null;
            }
            await using var stream = await resp.Content.ReadAsStreamAsync(ct);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
            if (!doc.RootElement.TryGetProperty("last_run_initialisation_time", out var t)
                || t.ValueKind != JsonValueKind.Number)
            {
                return null;
            }
            var unixSec = t.GetInt64();
            return DateTimeOffset.FromUnixTimeSeconds(unixSec).UtcDateTime;
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "Metadata fetch failed for {Model} via {Path}", modelId, metaPath);
            return null;
        }
    }

    private async Task<List<ForecastRow>> FetchAsync(
        string baseUrl,
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        int? forecastDays,
        DateOnly? startDate,
        DateOnly? endDate,
        bool isHistorical,
        DateTime? reportedRunTime,
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

        return Parse(doc.RootElement, location.Name, modelId, isHistorical, reportedRunTime);
    }

    internal static List<ForecastRow> Parse(
        JsonElement root,
        string locationName,
        string modelId,
        bool isHistorical,
        DateTime? reportedRunTime = null)
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

        // Run-time strategy (see Models/ForecastRow.RunTimeSource for canonical values):
        //   - Historical endpoint always synthesises: API returns best-available per
        //     valid-time, not a specific run. RunTime = midnight of valid day, so
        //     LeadHours = hour-of-day [0..23] and partitions split cleanly by valid date.
        //   - Live endpoint: if caller supplied a reportedRunTime (from the model's
        //     /data/{model}/static/meta.json `last_run_initialisation_time`), use it
        //     and tag "reported". Otherwise fall back to wall-clock-floored-to-hour
        //     and tag "synthesised" (older behaviour; rows a few hours old land with
        //     negative LeadHours, filter WHERE LeadHours>=0 for analysis).
        var liveRunTime = isHistorical
            ? default
            : (reportedRunTime ?? NowUtcFloorToHour());

        string runTimeSource;
        if (isHistorical) runTimeSource = RunTimeSources.Synthesised;
        else if (reportedRunTime.HasValue) runTimeSource = RunTimeSources.Reported;
        else runTimeSource = RunTimeSources.Synthesised;

        var rows = new List<ForecastRow>(times.Length);
        for (int i = 0; i < times.Length; i++)
        {
            var valid = times[i];
            var runTime = isHistorical ? valid.Date : liveRunTime;
            var lead = (int)Math.Round((valid - runTime).TotalHours);
            rows.Add(new ForecastRow
            {
                LocationName = locationName,
                Model = modelId,
                RunTimeUtc = runTime,
                ValidTimeUtc = valid,
                LeadHours = lead,
                RunTimeSource = runTimeSource,
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
