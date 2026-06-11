using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using RunTimeSources = WeatherBlend.Models.RunTimeSources;

namespace WeatherBlend.Collect;

/// <summary>
/// Fetches wave forecasts + ERA5-ocean wave truth from Open-Meteo's Marine API
/// (marine-api.open-meteo.com/v1/marine). One endpoint serves four roles here
/// (all verified by live probes 2026-06-11):
///
///   * live per-model wave forecast (models=meteofrance_wave / ecmwf_wam025 /
///     gwam / ewam / ncep_gfswave025, plus best_match for the site extras:
///     secondary swell, sea_level_height_msl, sea_surface_temperature);
///   * per-lead offset rows via <c>{var}_previous_dayN</c> columns — these work
///     on the LIVE window only (nulls in the hindcast archive, and there is no
///     marine previous-runs API), so per-lead history accrues strictly forward
///     from the day collection starts;
///   * lead-unlabelled hindcast via start_date/end_date (best-available per
///     valid-time, RunTimeSource=hist_forecast; per-model archive starts vary
///     — all-null chunks parse to 0 rows, which is expected, not an error);
///   * gapless wave TRUTH via <c>models=era5_ocean</c> (total wave
///     height/period/direction back to at least 2020). NOTE: the weather
///     archive-api ERA5 endpoint snaps to land cells near the coast and
///     returns null waves — era5_ocean through the marine API is the one
///     honest route.
///
/// The coordinate is the location's PINNED marine cell (config marine: block),
/// not the site coordinate — the cliff's land coordinate happens to resolve to
/// a valid sea cell today, but pinning makes the choice explicit and stable.
/// </summary>
public sealed class MarineClient
{
    private const string BaseUrl = "https://marine-api.open-meteo.com/v1/marine";

    /// <summary>Model id whose pulls carry the site extras (secondary swell,
    /// sea level, SST) — those variables are "undefined" on per-model requests.</summary>
    public const string BestMatchModel = "best_match";

    /// <summary>ERA5 wave-reanalysis model id (truth source tag).</summary>
    public const string Era5OceanModel = "era5_ocean";

    /// <summary>Same 1..7 day offsets as the weather Previous Runs pipeline.</summary>
    private static readonly int[] PreviousDayOffsets = { 1, 2, 3, 4, 5, 6, 7 };

    /// <summary>Truth variables era5_ocean actually serves (swell components
    /// and the site extras come back null there).</summary>
    private static readonly string[] Era5OceanVariables = { "wave_height", "wave_period", "wave_direction" };

    private readonly HttpClient _http;
    private readonly ILogger<MarineClient> _log;

    public MarineClient(HttpClient http, ILogger<MarineClient> log)
    {
        _http = http;
        _log = log;
    }

    /// <summary>Current forecast for one wave model. Run time is wall-clock-
    /// floored ("synthesised") — the marine models have no metadata endpoint
    /// mapping, same situation the weather collector started from.</summary>
    public async Task<List<MarineForecastRow>> FetchLiveAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        int forecastDays,
        CancellationToken ct)
    {
        var root = await GetAsync(location, modelId, string.Join(",", variables),
            $"forecast_days={forecastDays}", ct);
        var runTime = NowUtcFloorToHour();
        return Parse(root, location.Name, modelId, runTime, RunTimeSources.Synthesised);
    }

    /// <summary>
    /// Per-lead offset rows from the live window: every configured variable ×
    /// previous_day1..7, over <paramref name="pastDays"/> past days + today.
    /// Emits one row per (valid-hour, offset) with LeadHours = 24·N and
    /// RunTime = ValidTime − 24·N (the weather offset_day convention). Each
    /// collect cycle re-fetches the window and the writer overwrites the
    /// per-valid-date file, converging to complete coverage a day later.
    /// </summary>
    public async Task<List<MarineForecastRow>> FetchOffsetDaysAsync(
        LocationConfig location,
        string modelId,
        IReadOnlyList<string> variables,
        int pastDays,
        CancellationToken ct)
    {
        var hourlyNames = new List<string>(variables.Count * PreviousDayOffsets.Length);
        foreach (var v in variables)
            foreach (var n in PreviousDayOffsets)
                hourlyNames.Add($"{v}_previous_day{n}");

        var root = await GetAsync(location, modelId, string.Join(",", hourlyNames),
            $"past_days={pastDays}&forecast_days=1", ct);
        return ParseOffsetDays(root, location.Name, modelId, variables);
    }

    /// <summary>Lead-unlabelled hindcast (best-available per valid-time) for
    /// one wave model. RunTime = ValidTime, LeadHours = 0, tagged
    /// hist_forecast — the same placeholder convention the weather
    /// historical-forecast backfill uses.</summary>
    public async Task<List<MarineForecastRow>> FetchHindcastAsync(
        LocationConfig location,
        string modelId,
        IEnumerable<string> variables,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
    {
        var root = await GetAsync(location, modelId, string.Join(",", variables),
            $"start_date={startDate:yyyy-MM-dd}&end_date={endDate:yyyy-MM-dd}", ct);
        return Parse(root, location.Name, modelId,
            runTime: null, RunTimeSources.HistForecast);
    }

    /// <summary>ERA5-ocean wave truth (models=era5_ocean): total wave
    /// height/period/direction, gapless back to well before any forecast
    /// archive. Rows land in data/truth/waves under source=era5_ocean.</summary>
    public async Task<List<WaveTruthRow>> FetchWaveTruthAsync(
        LocationConfig location,
        DateOnly startDate,
        DateOnly endDate,
        CancellationToken ct)
    {
        // Truth uses its own pinned cell where configured — ERA5's 0.5° wave
        // grid needs a different (more exposed) cell than the forecast point
        // at Sennen; see MarineConfig.TruthLatitude.
        var root = await GetAsync(location, Era5OceanModel, string.Join(",", Era5OceanVariables),
            $"start_date={startDate:yyyy-MM-dd}&end_date={endDate:yyyy-MM-dd}", ct, useTruthCell: true);
        return ParseWaveTruth(root, location.Name);
    }

    private async Task<JsonElement> GetAsync(
        LocationConfig location, string modelId, string hourly, string extraQuery, CancellationToken ct,
        bool useTruthCell = false)
    {
        var marine = location.Marine
            ?? throw new InvalidOperationException(
                $"Location '{location.Name}' has no marine: config block — caller should gate on Marine != null.");
        var lat = useTruthCell ? marine.TruthLatitude ?? marine.Latitude : marine.Latitude;
        var lon = useTruthCell ? marine.TruthLongitude ?? marine.Longitude : marine.Longitude;
        var qs =
            $"latitude={lat.ToString(System.Globalization.CultureInfo.InvariantCulture)}" +
            $"&longitude={lon.ToString(System.Globalization.CultureInfo.InvariantCulture)}" +
            $"&hourly={Uri.EscapeDataString(hourly)}" +
            $"&models={Uri.EscapeDataString(modelId)}" +
            "&timezone=UTC" +
            $"&{extraQuery}";
        var url = $"{BaseUrl}?{qs}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return doc.RootElement.Clone();
    }

    internal static List<MarineForecastRow> Parse(
        JsonElement root,
        string locationName,
        string modelId,
        DateTime? runTime,
        string runTimeSource)
    {
        if (!root.TryGetProperty("hourly", out var hourly)) return new();

        var times = Times(hourly);
        var col = ColumnReader(hourly, times.Length);

        var wh   = col("wave_height");          var wp   = col("wave_period");          var wd   = col("wave_direction");
        var wwh  = col("wind_wave_height");     var wwp  = col("wind_wave_period");     var wwd  = col("wind_wave_direction");
        var swh  = col("swell_wave_height");    var swp  = col("swell_wave_period");    var swd  = col("swell_wave_direction");
        var sswh = col("secondary_swell_wave_height");
        var sswp = col("secondary_swell_wave_period");
        var sswd = col("secondary_swell_wave_direction");
        var sl   = col("sea_level_height_msl");
        var sst  = col("sea_surface_temperature");

        var rows = new List<MarineForecastRow>(times.Length);
        for (int i = 0; i < times.Length; i++)
        {
            // Hindcast chunks before a model's archive start (and unpublished
            // recent days) return all-null columns — skip those hours entirely
            // so pre-archive backfill chunks write 0 rows rather than scaffold.
            if (wh[i] is null && wp[i] is null && wd[i] is null &&
                wwh[i] is null && swh[i] is null && sswh[i] is null &&
                sl[i] is null && sst[i] is null)
                continue;

            var valid = times[i];
            var run = runTime ?? valid;   // hindcast: lead-unlabelled placeholder
            rows.Add(new MarineForecastRow
            {
                LocationName = locationName,
                Model = modelId,
                RunTimeUtc = run,
                ValidTimeUtc = valid,
                LeadHours = runTime.HasValue ? (int)Math.Round((valid - run).TotalHours) : 0,
                RunTimeSource = runTimeSource,
                WaveHeight = wh[i], WavePeriod = wp[i], WaveDirection = wd[i],
                WindWaveHeight = wwh[i], WindWavePeriod = wwp[i], WindWaveDirection = wwd[i],
                SwellWaveHeight = swh[i], SwellWavePeriod = swp[i], SwellWaveDirection = swd[i],
                SecondarySwellWaveHeight = sswh[i], SecondarySwellWavePeriod = sswp[i], SecondarySwellWaveDirection = sswd[i],
                SeaLevelHeightMsl = sl[i], SeaSurfaceTemperature = sst[i],
            });
        }
        return rows;
    }

    internal static List<MarineForecastRow> ParseOffsetDays(
        JsonElement root,
        string locationName,
        string modelId,
        IReadOnlyList<string> variables)
    {
        if (!root.TryGetProperty("hourly", out var hourly)) return new();

        var times = Times(hourly);
        var col = ColumnReader(hourly, times.Length);

        // variable -> offset N -> column
        var cols = new Dictionary<string, Dictionary<int, double?[]>>(variables.Count);
        foreach (var v in variables)
        {
            var byOffset = new Dictionary<int, double?[]>(PreviousDayOffsets.Length);
            foreach (var n in PreviousDayOffsets)
                byOffset[n] = col($"{v}_previous_day{n}");
            cols[v] = byOffset;
        }

        var rows = new List<MarineForecastRow>(times.Length * PreviousDayOffsets.Length);
        for (int i = 0; i < times.Length; i++)
        {
            var valid = times[i];
            foreach (var n in PreviousDayOffsets)
            {
                double? Get(string v) => cols.TryGetValue(v, out var byOff) ? byOff[n][i] : null;

                var anyValue = false;
                foreach (var v in variables)
                {
                    if (Get(v).HasValue) { anyValue = true; break; }
                }
                if (!anyValue) continue;

                var lead = 24 * n;
                rows.Add(new MarineForecastRow
                {
                    LocationName = locationName,
                    Model = modelId,
                    RunTimeUtc = valid.AddHours(-lead),
                    ValidTimeUtc = valid,
                    LeadHours = lead,
                    RunTimeSource = RunTimeSources.OffsetDay,
                    WaveHeight = Get("wave_height"), WavePeriod = Get("wave_period"), WaveDirection = Get("wave_direction"),
                    WindWaveHeight = Get("wind_wave_height"), WindWavePeriod = Get("wind_wave_period"), WindWaveDirection = Get("wind_wave_direction"),
                    SwellWaveHeight = Get("swell_wave_height"), SwellWavePeriod = Get("swell_wave_period"), SwellWaveDirection = Get("swell_wave_direction"),
                    SecondarySwellWaveHeight = Get("secondary_swell_wave_height"),
                    SecondarySwellWavePeriod = Get("secondary_swell_wave_period"),
                    SecondarySwellWaveDirection = Get("secondary_swell_wave_direction"),
                    SeaLevelHeightMsl = Get("sea_level_height_msl"),
                    SeaSurfaceTemperature = Get("sea_surface_temperature"),
                });
            }
        }
        return rows;
    }

    internal static List<WaveTruthRow> ParseWaveTruth(JsonElement root, string locationName)
    {
        if (!root.TryGetProperty("hourly", out var hourly)) return new();

        var times = Times(hourly);
        var col = ColumnReader(hourly, times.Length);
        var wh = col("wave_height");
        var wp = col("wave_period");
        var wd = col("wave_direction");

        var rows = new List<WaveTruthRow>(times.Length);
        for (int i = 0; i < times.Length; i++)
        {
            if (wh[i] is null && wp[i] is null && wd[i] is null) continue;
            rows.Add(new WaveTruthRow
            {
                LocationName = locationName,
                Source = Era5OceanModel,
                ValidTimeUtc = times[i],
                WaveHeight = wh[i],
                WavePeriod = wp[i],
                WaveDirection = wd[i],
            });
        }
        return rows;
    }

    private static DateTime[] Times(JsonElement hourly) =>
        hourly.GetProperty("time").EnumerateArray()
            .Select(e => DateTime.SpecifyKind(DateTime.Parse(e.GetString()!), DateTimeKind.Utc))
            .ToArray();

    private static Func<string, double?[]> ColumnReader(JsonElement hourly, int length) =>
        name => hourly.TryGetProperty(name, out var arr)
            ? arr.EnumerateArray()
                 .Select(e => e.ValueKind == JsonValueKind.Null ? (double?)null : e.GetDouble())
                 .ToArray()
            : Enumerable.Repeat<double?>(null, length).ToArray();

    private static DateTime NowUtcFloorToHour()
    {
        var now = DateTime.UtcNow;
        return new DateTime(now.Year, now.Month, now.Day, now.Hour, 0, 0, DateTimeKind.Utc);
    }
}
