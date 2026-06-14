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
    private const string HistoricalForecastBase = "https://historical-forecast-api.open-meteo.com/v1/forecast";
    private const string MetadataBase = "https://api.open-meteo.com/data";

    /// <summary>
    /// Pressure-level (…hPa) variables. The Previous Runs API rejects these with a
    /// 400 (see <see cref="FetchPreviousRunsAsync"/>), so the offset_day archive
    /// can't carry upper-air fields. The Historical Forecast API serves them as a
    /// plain (lead-unlabelled) time series back to ~2022, which we backfill into a
    /// separate hist_forecast.parquet — see <see cref="FetchHistoricalForecastAsync"/>.
    /// </summary>
    public static readonly string[] PressureLevelVariables =
    {
        "temperature_850hPa", "temperature_700hPa", "temperature_500hPa",
        "geopotential_height_850hPa", "geopotential_height_500hPa",
        "wind_speed_850hPa", "wind_speed_500hPa",
        "wind_direction_850hPa", "wind_direction_500hPa",
        "relative_humidity_850hPa",
    };

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

    /// <summary>
    /// Maximum acceptable age (hours) of a model's <c>last_run_initialisation_time</c>
    /// before the live collector skips the fetch. Every blender NWP we consume
    /// updates at most every 12 hours (GEM is the slowest); 24h covers a full
    /// cadence with margin so a single missed cycle doesn't trip this. Beyond
    /// 24h indicates the upstream (CMC, NCEP, …) or Open-Meteo's ingestion has
    /// stalled — silently overwriting the previous partition with newer
    /// ValidTimes would just hide the outage, so we skip and warn.
    ///
    /// Reason this matters: <see cref="Storage.ParquetWriter.WriteForecastsAsync"/>
    /// partitions by <c>date={RunTimeUtc.Date}</c>, so if every fresh fetch
    /// gets the SAME stale RunTime, each write clobbers the original
    /// <c>date={staleDate}/run={staleHour}.parquet</c> with rows whose
    /// ValidTimes have rolled forward — mild partition corruption + the live
    /// outage stays invisible. See the 2026-05-27 GEM incident.
    /// </summary>
    public const int MaxReportedRunAgeHours = 24;

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
        if (reported.HasValue)
        {
            var ageHours = (DateTime.UtcNow - reported.Value).TotalHours;
            if (ageHours > MaxReportedRunAgeHours)
            {
                _log.LogWarning(
                    "Live collect skipped: {Model} last_run_initialisation_time is {Age:F1}h old " +
                    "(reported {Reported:yyyy-MM-dd HH:mm}Z, threshold {Threshold}h). " +
                    "Either Open-Meteo's ingestion of the upstream source has stalled or the " +
                    "upstream centre hasn't published. Skipping the fetch avoids overwriting the " +
                    "previous date={{ReportedDate}}/run={{ReportedHour}}.parquet partition with " +
                    "rows whose ValidTimes have rolled forward. previous-runs collect is unaffected.",
                    modelId, ageHours, reported.Value, MaxReportedRunAgeHours);
                return new List<ForecastRow>();
            }
        }
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
        // Open-Meteo's Previous Runs endpoint rejects pressure-level (…hPa)
        // variables with a 400 ("Cannot initialize SurfacePressureAndHeight-
        // Variable … from invalid String value …"). Because one bad variable
        // fails the whole multi-variable request, every previous-runs pull 400s
        // and the refresh writes nothing (observed 2026-05-30, having worked the
        // day before — an OM-side validation change). The live forecast endpoint
        // still serves these; only previous_runs dropped them. Drop them here so
        // the surface variables — the bulk of the offset_day archive — keep
        // flowing; the hPa columns just come back null for offset_day rows.
        var varList = variables
            .Where(v => !v.EndsWith("hPa", StringComparison.Ordinal))
            .ToArray();
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
                    PressureMsl             = Maybe("pressure_msl"),
                    Cape                    = Maybe("cape"),
                    Visibility              = Maybe("visibility"),
                    ShortwaveRadiation      = Maybe("shortwave_radiation"),
                    DirectRadiation         = Maybe("direct_radiation"),
                    DiffuseRadiation        = Maybe("diffuse_radiation"),
                    Temperature850hPa       = Maybe("temperature_850hPa"),
                    Temperature700hPa       = Maybe("temperature_700hPa"),
                    Temperature500hPa       = Maybe("temperature_500hPa"),
                    GeopotentialHeight850hPa= Maybe("geopotential_height_850hPa"),
                    GeopotentialHeight500hPa= Maybe("geopotential_height_500hPa"),
                    WindSpeed850hPa         = Maybe("wind_speed_850hPa"),
                    WindSpeed500hPa         = Maybe("wind_speed_500hPa"),
                    WindDirection850hPa     = Maybe("wind_direction_850hPa"),
                    WindDirection500hPa     = Maybe("wind_direction_500hPa"),
                    RelativeHumidity850hPa  = Maybe("relative_humidity_850hPa"),
                });

                double? Maybe(string v) => cols.TryGetValue(v, out var byOff) ? byOff[n][i] : null;
            }
        }
        return rows;
    }

    /// <summary>
    /// Fetches pressure-level fields from Open-Meteo's Historical Forecast API
    /// (archived model forecasts, ~2022→present). Used ONLY to backfill the
    /// upper-air (…hPa) columns the Previous Runs API refuses to serve. The
    /// endpoint exposes no run/initialisation time and its per-lead mechanism
    /// rejects pressure variables, so rows are lead-unlabelled:
    /// <see cref="RunTimeSources.HistForecast"/>, RunTime = ValidTime, LeadHours = 0
    /// (placeholders, not a real cycle). Only pressure columns are populated.
    /// A model that doesn't archive pressure for the window returns all-null
    /// columns and yields zero rows (the caller logs that).
    /// </summary>
    public async Task<List<ForecastRow>> FetchHistoricalForecastAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
    {
        // The caller passes the full forecast variable list (config
        // Variables.Forecast already carries surface + pressure-level fields),
        // so one pull populates BOTH the surface columns the 0h nowcast needs
        // and the upper-air columns the original gap-fill use needs. Falling
        // back to the pressure-only set keeps legacy callers that pass nothing
        // working unchanged.
        var varList = (variables ?? Enumerable.Empty<string>()).ToArray();
        if (varList.Length == 0) varList = PressureLevelVariables;
        var hourly = string.Join(",", varList);
        var qs = new[]
        {
            $"latitude={location.Latitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"longitude={location.Longitude.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"elevation={location.ElevationMeters.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
            $"hourly={Uri.EscapeDataString(hourly)}",
            $"models={Uri.EscapeDataString(modelId)}",
            "timezone=UTC",
            "windspeed_unit=ms",
            "temperature_unit=celsius",
            "precipitation_unit=mm",
            $"start_date={startDate:yyyy-MM-dd}",
            $"end_date={endDate:yyyy-MM-dd}"
        };
        var url = $"{HistoricalForecastBase}?{string.Join('&', qs)}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        return ParseHistoricalForecast(doc.RootElement, location.Name, modelId);
    }

    internal static List<ForecastRow> ParseHistoricalForecast(
        JsonElement root, string locationName, string modelId)
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

        // Surface columns (populated when the caller requested surface vars —
        // the 0h nowcast path). A model/window that doesn't archive a field
        // returns an all-null column, which is fine.
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
        var pmsl  = Col("pressure_msl");
        var cape  = Col("cape");
        var vis   = Col("visibility");
        var swr   = Col("shortwave_radiation");
        var drr   = Col("direct_radiation");
        var dfr   = Col("diffuse_radiation");
        var t850 = Col("temperature_850hPa");
        var t700 = Col("temperature_700hPa");
        var t500 = Col("temperature_500hPa");
        var z850 = Col("geopotential_height_850hPa");
        var z500 = Col("geopotential_height_500hPa");
        var ws850 = Col("wind_speed_850hPa");
        var ws500 = Col("wind_speed_500hPa");
        var wd850 = Col("wind_direction_850hPa");
        var wd500 = Col("wind_direction_500hPa");
        var rh850 = Col("relative_humidity_850hPa");

        var rows = new List<ForecastRow>(times.Length);
        for (int i = 0; i < times.Length; i++)
        {
            // Keep any hour that carries at least one non-null payload field
            // (surface OR pressure). The original pressure-only gap-fill dropped
            // pressure-less hours; now that the same pull also serves the 0h
            // surface nowcast, a surface-only hour (e.g. UKMO/AIFS, which don't
            // archive upper-air) must survive on its surface fields alone.
            var anySurface =
                t2m[i] is not null || pr[i] is not null || rh2m[i] is not null ||
                cc[i] is not null || ws10[i] is not null || sp[i] is not null ||
                td2m[i] is not null || cape[i] is not null;
            var anyPressure =
                t850[i] is not null || t700[i] is not null || t500[i] is not null ||
                z850[i] is not null || z500[i] is not null || ws850[i] is not null ||
                ws500[i] is not null || wd850[i] is not null || wd500[i] is not null || rh850[i] is not null;
            if (!anySurface && !anyPressure)
                continue;

            rows.Add(new ForecastRow
            {
                LocationName = locationName,
                Model = modelId,
                // Lead-unlabelled: no run time exists, so these are placeholders.
                RunTimeUtc = times[i],
                ValidTimeUtc = times[i],
                LeadHours = 0,
                RunTimeSource = RunTimeSources.HistForecast,
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
                PressureMsl = pmsl[i],
                Cape = cape[i],
                Visibility = vis[i],
                ShortwaveRadiation = swr[i],
                DirectRadiation = drr[i],
                DiffuseRadiation = dfr[i],
                Temperature850hPa = t850[i],
                Temperature700hPa = t700[i],
                Temperature500hPa = t500[i],
                GeopotentialHeight850hPa = z850[i],
                GeopotentialHeight500hPa = z500[i],
                WindSpeed850hPa = ws850[i],
                WindSpeed500hPa = ws500[i],
                WindDirection850hPa = wd850[i],
                WindDirection500hPa = wd500[i],
                RelativeHumidity850hPa = rh850[i],
            });
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
        var pmsl  = Col("pressure_msl");
        var cape  = Col("cape");
        var vis   = Col("visibility");
        var swr   = Col("shortwave_radiation");
        var drr   = Col("direct_radiation");
        var dfr   = Col("diffuse_radiation");
        var t850  = Col("temperature_850hPa");
        var t700  = Col("temperature_700hPa");
        var t500  = Col("temperature_500hPa");
        var z850  = Col("geopotential_height_850hPa");
        var z500  = Col("geopotential_height_500hPa");
        var ws850 = Col("wind_speed_850hPa");
        var ws500 = Col("wind_speed_500hPa");
        var wd850 = Col("wind_direction_850hPa");
        var wd500 = Col("wind_direction_500hPa");
        var rh850 = Col("relative_humidity_850hPa");

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
                PressureMsl = pmsl[i],
                Cape = cape[i],
                Visibility = vis[i],
                ShortwaveRadiation = swr[i],
                DirectRadiation = drr[i],
                DiffuseRadiation = dfr[i],
                Temperature850hPa = t850[i],
                Temperature700hPa = t700[i],
                Temperature500hPa = t500[i],
                GeopotentialHeight850hPa = z850[i],
                GeopotentialHeight500hPa = z500[i],
                WindSpeed850hPa = ws850[i],
                WindSpeed500hPa = ws500[i],
                WindDirection850hPa = wd850[i],
                WindDirection500hPa = wd500[i],
                RelativeHumidity850hPa = rh850[i],
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
