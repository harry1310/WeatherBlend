using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using RunTimeSources = WeatherBlend.Models.RunTimeSources;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches forecasts from Open-Meteo.
///
/// Live endpoint:        https://api.open-meteo.com/v1/forecast
/// Previous Runs:        https://previous-runs-api.open-meteo.com/v1/forecast
///
/// The Previous Runs endpoint backs all training data (proper per-lead 24..168h
/// buckets via offset_day rows). The live endpoint backs the cycle collector.
/// </summary>
public sealed class OpenMeteoClient
{
    private const string LiveBase = "https://api.open-meteo.com/v1/forecast";
    private const string PreviousRunsBase = "https://previous-runs-api.open-meteo.com/v1/forecast";
    private const string MetadataBase = "https://api.open-meteo.com/data";

    /// <summary>Offsets exposed by the Previous Runs API. Each offset N maps to
    /// the lead-hour bucket [24N..24N+23]; we tag rows with LeadHours = 24N.</summary>
    private static readonly int[] PreviousDayOffsets = { 1, 2, 3, 4, 5, 6, 7 };

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
            forecastDays, null, null, reported, ct);
    }

    /// <summary>
    /// Fetches historical forecasts from Open-Meteo's Previous Runs API. The endpoint
    /// returns one column per (variable, previous-day-offset) pair — e.g.
    /// <c>temperature_2m_previous_day1</c> through <c>_previous_day7</c>. Each offset N
    /// covers the forecast lead-hour bucket [24N..24N+23]; we emit one
    /// <see cref="ForecastRow"/> per (valid-hour, offset) with
    /// <c>LeadHours = 24·N</c> (lower edge of the bucket),
    /// <c>RunTimeUtc = ValidTime − 24·N h</c>, and
    /// <c>RunTimeSource = <see cref="RunTimeSources.OffsetDay"/></c>.
    /// </summary>
    public async Task<List<ForecastRow>> FetchPreviousRunsAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
    {
        var varList = variables.ToArray();
        // Build hourly = var1_previous_day1,var1_previous_day2,...,varK_previous_day7
        var hourlyNames = new List<string>(varList.Length * PreviousDayOffsets.Length);
        foreach (var v in varList)
            foreach (var n in PreviousDayOffsets)
                hourlyNames.Add($"{v}_previous_day{n}");
        var hourly = string.Join(",", hourlyNames);

        var qs = new List<string>
        {
            $"latitude={location.Latitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"longitude={location.Longitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"elevation={location.ElevationMeters.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"hourly={Uri.EscapeDataString(hourly)}",
            $"models={Uri.EscapeDataString(modelId)}",
            $"start_date={startDate:yyyy-MM-dd}",
            $"end_date={endDate:yyyy-MM-dd}",
            "timezone=UTC",
            "windspeed_unit=ms",
            "temperature_unit=celsius",
            "precipitation_unit=mm"
        };

        var url = $"{PreviousRunsBase}?{string.Join('&', qs)}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        return ParsePreviousRuns(doc.RootElement, location.Name, modelId, varList);
    }

    internal static List<ForecastRow> ParsePreviousRuns(
        JsonElement root,
        string locationName,
        string modelId,
        IReadOnlyList<string> variables)
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

        // Pull each (var, offset) column up front so the row loop is tight.
        // Map: variable -> offset N -> column array.
        var cols = new Dictionary<string, Dictionary<int, double?[]>>(variables.Count);
        foreach (var v in variables)
        {
            var byOffset = new Dictionary<int, double?[]>(PreviousDayOffsets.Length);
            foreach (var n in PreviousDayOffsets)
                byOffset[n] = Col($"{v}_previous_day{n}");
            cols[v] = byOffset;
        }

        double? Get(string var_, int offset, int i) => cols[var_][offset][i];

        var rows = new List<ForecastRow>(times.Length * PreviousDayOffsets.Length);
        for (int i = 0; i < times.Length; i++)
        {
            var valid = times[i];
            foreach (var n in PreviousDayOffsets)
            {
                var lead = 24 * n;
                var runTime = valid.AddHours(-lead);

                // Skip rows where every configured variable is null for this offset.
                // Open-Meteo returns all-null columns for model+date combinations
                // outside the model's Previous Runs archive (e.g. UKMO pre-launch).
                var anyValue = false;
                foreach (var v in variables)
                {
                    if (Get(v, n, i).HasValue) { anyValue = true; break; }
                }
                if (!anyValue) continue;

                rows.Add(new ForecastRow
                {
                    LocationName = locationName,
                    Model = modelId,
                    RunTimeUtc = runTime,
                    ValidTimeUtc = valid,
                    LeadHours = lead,
                    RunTimeSource = RunTimeSources.OffsetDay,
                    Temperature2m           = Maybe("temperature_2m"),
                    DewPoint2m              = Maybe("dew_point_2m"),
                    RelativeHumidity2m      = Maybe("relative_humidity_2m"),
                    Precipitation           = Maybe("precipitation"),
                    PrecipitationProbability= Maybe("precipitation_probability"),
                    Rain                    = Maybe("rain"),
                    Showers                 = Maybe("showers"),
                    Snowfall                = Maybe("snowfall"),
                    CloudCover              = Maybe("cloud_cover"),
                    CloudCoverLow           = Maybe("cloud_cover_low"),
                    CloudCoverMid           = Maybe("cloud_cover_mid"),
                    CloudCoverHigh          = Maybe("cloud_cover_high"),
                    WindSpeed10m            = Maybe("wind_speed_10m"),
                    WindDirection10m        = Maybe("wind_direction_10m"),
                    WindGusts10m            = Maybe("wind_gusts_10m"),
                    SurfacePressure         = Maybe("surface_pressure"),
                    Cape                    = Maybe("cape"),
                    Visibility              = Maybe("visibility"),
                    ShortwaveRadiation      = Maybe("shortwave_radiation"),
                    DirectRadiation         = Maybe("direct_radiation"),
                    DiffuseRadiation        = Maybe("diffuse_radiation"),
                });

                double? Maybe(string v) => cols.TryGetValue(v, out var byOff) ? byOff[n][i] : null;
            }
        }
        return rows;
    }

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

        return Parse(doc.RootElement, location.Name, modelId, reportedRunTime);
    }

    internal static List<ForecastRow> Parse(
        JsonElement root,
        string locationName,
        string modelId,
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
        var swr   = Col("shortwave_radiation");
        var drr   = Col("direct_radiation");
        var dfr   = Col("diffuse_radiation");

        // Run-time strategy (see Models/ForecastRow.RunTimeSource for canonical values):
        // Live endpoint: if the caller supplied a reportedRunTime (from the model's
        // /data/{model}/static/meta.json `last_run_initialisation_time`), use it and
        // tag "reported". Otherwise fall back to wall-clock-floored-to-hour and tag
        // "synthesised" — rows a few hours old land with negative LeadHours, so
        // filter WHERE LeadHours>=0 for analysis.
        var runTime = reportedRunTime ?? NowUtcFloorToHour();
        var runTimeSource = reportedRunTime.HasValue
            ? RunTimeSources.Reported
            : RunTimeSources.Synthesised;

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
                Visibility = vis[i],
                ShortwaveRadiation = swr[i],
                DirectRadiation = drr[i],
                DiffuseRadiation = dfr[i]
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
