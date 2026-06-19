using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// Davis WeatherLink v2 API client (api.weatherlink.com/v2) — pulls real-
/// instrument observations as a close, truth source. Built for NCI Gwennap
/// Head (station_id 115072) near Sennen, but station-generic.
///
/// Auth: <c>api-key</c> is a QUERY parameter; <c>X-Api-Secret</c> is a request
/// HEADER. The weather data lives in the sensor whose <c>sensor_type == 43</c>
/// (the Davis ISS) — we iterate <c>response.sensors</c>, pick that sensor, and
/// read its <c>data</c> array of 15-minute archive records.
///
/// Endpoints used:
/// <list type="bullet">
///   <item><c>GET /historic/{station-id}?start-timestamp=&amp;end-timestamp=&amp;api-key=</c>
///     — archive, capped to a 24-HOUR window per request (96 records/day).</item>
/// </list>
///
/// Fails soft: any non-200 logs the body + returns an empty list so a collect
/// cycle never breaks on a WeatherLink hiccup (it's a supplemental truth source,
/// like Met Office obs).
/// </summary>
public sealed class WeatherLinkClient
{
    public const string BaseUrl = "https://api.weatherlink.com/v2";

    /// <summary>The Davis ISS (Integrated Sensor Suite) sensor type that carries
    /// the temp / humidity / wind / rain archive fields.</summary>
    public const int IssSensorType = 43;

    /// <summary>WeatherLink /historic requests are capped to a 24h window.</summary>
    public static readonly TimeSpan MaxHistoricWindow = TimeSpan.FromHours(24);

    private readonly HttpClient _http;
    private readonly ILogger<WeatherLinkClient> _log;

    public WeatherLinkClient(HttpClient http, ILogger<WeatherLinkClient> log)
    {
        _http = http;
        _log = log;
    }

    /// <summary>
    /// Fetch one historic window (≤ 24h) for a station and aggregate the
    /// 15-minute archive records to hourly truth rows. Soft-fails to an empty
    /// list on any non-200 or parse problem.
    /// </summary>
    public async Task<List<WeatherLinkObservationRow>> FetchHistoricAsync(
        string locationName,
        string stationId,
        DateTimeOffset start,
        DateTimeOffset end,
        string apiKey,
        string apiSecret,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(stationId))
        {
            _log.LogInformation("WeatherLink station id not configured; skipping");
            return new();
        }

        var startEpoch = start.ToUnixTimeSeconds();
        var endEpoch = end.ToUnixTimeSeconds();
        var url = $"{BaseUrl}/historic/{Uri.EscapeDataString(stationId)}"
                + $"?api-key={Uri.EscapeDataString(apiKey)}"
                + $"&start-timestamp={startEpoch}&end-timestamp={endEpoch}";

        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.Add("X-Api-Secret", apiSecret);
        req.Headers.Accept.ParseAdd("application/json");

        // Log without the api-key query value to keep the secret out of logs.
        _log.LogInformation("GET {Base}/historic/{Station}?start={Start}&end={End}",
            BaseUrl, stationId, startEpoch, endEpoch);
        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            _log.LogWarning("WeatherLink /historic returned {Status}: {Body}",
                (int)resp.StatusCode, Truncate(body, 400));
            return new();
        }

        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return Parse(doc.RootElement, locationName, stationId);
    }

    /// <summary>
    /// Parse a WeatherLink /historic (or /current) response into hourly rows:
    /// extract the sensor_type==43 records, then aggregate by UTC hour.
    /// </summary>
    internal static List<WeatherLinkObservationRow> Parse(
        JsonElement root,
        string locationName,
        string stationId)
    {
        var records = ExtractIssRecords(root);
        return AggregateHourly(records, locationName, stationId);
    }

    /// <summary>
    /// Pull the raw 15-minute records out of the ISS sensor's <c>data</c> array.
    /// Returns the parsed records (already in SI units) in API order.
    /// </summary>
    internal static List<IssRecord> ExtractIssRecords(JsonElement root)
    {
        var records = new List<IssRecord>();
        if (root.ValueKind != JsonValueKind.Object) return records;
        if (!root.TryGetProperty("sensors", out var sensors) || sensors.ValueKind != JsonValueKind.Array)
            return records;

        foreach (var sensor in sensors.EnumerateArray())
        {
            if (GetInt(sensor, "sensor_type") != IssSensorType) continue;
            if (!sensor.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.Array) continue;

            foreach (var rec in data.EnumerateArray())
            {
                var ts = GetLong(rec, "ts");
                if (ts is null) continue;

                records.Add(new IssRecord(
                    Ts: DateTimeOffset.FromUnixTimeSeconds(ts.Value).UtcDateTime,
                    TempAvgC: FahrenheitToCelsius(GetDouble(rec, "temp_avg")),
                    TempHiC: FahrenheitToCelsius(GetDouble(rec, "temp_hi")),
                    TempLoC: FahrenheitToCelsius(GetDouble(rec, "temp_lo")),
                    HumidityPct: GetDouble(rec, "hum_last"),
                    DewPointC: FahrenheitToCelsius(GetDouble(rec, "dew_point_last")),
                    RainfallMm: GetDouble(rec, "rainfall_mm"),
                    RainRateMmHr: GetDouble(rec, "rain_rate_hi_mm"),
                    WindSpeedMs: MphToMs(GetDouble(rec, "wind_speed_avg")),
                    WindGustMs: MphToMs(GetDouble(rec, "wind_speed_hi")),
                    WindDirDeg: GetDouble(rec, "wind_dir_of_prevail")));
            }
        }

        return records;
    }

    /// <summary>
    /// Aggregate 15-minute ISS records into one row per UTC hour. An hour is
    /// only emitted if it has at least one valid temp record. Rows are ordered
    /// by ObservedTimeUtc. All inputs are already in SI units.
    /// </summary>
    internal static List<WeatherLinkObservationRow> AggregateHourly(
        IEnumerable<IssRecord> records,
        string locationName,
        string stationId)
    {
        var rows = new List<WeatherLinkObservationRow>();

        foreach (var group in records
                     .GroupBy(r => new DateTime(r.Ts.Year, r.Ts.Month, r.Ts.Day, r.Ts.Hour, 0, 0, DateTimeKind.Utc))
                     .OrderBy(g => g.Key))
        {
            var hourRecords = group.ToList();

            // Only emit an hour with at least one valid temp record.
            var temps = hourRecords.Where(r => r.TempAvgC.HasValue).Select(r => r.TempAvgC!.Value).ToList();
            if (temps.Count == 0) continue;

            rows.Add(new WeatherLinkObservationRow
            {
                LocationName = locationName,
                StationId = stationId,
                ObservedTimeUtc = group.Key,
                Temperature2m = temps.Average(),
                TemperatureHigh = Max(hourRecords.Select(r => r.TempHiC)),
                TemperatureLow = Min(hourRecords.Select(r => r.TempLoC)),
                Humidity = Mean(hourRecords.Select(r => r.HumidityPct)),
                DewPoint = Mean(hourRecords.Select(r => r.DewPointC)),
                RainfallMm = Sum(hourRecords.Select(r => r.RainfallMm)),
                RainRateMmHr = Max(hourRecords.Select(r => r.RainRateMmHr)),
                WindSpeed10m = Mean(hourRecords.Select(r => r.WindSpeedMs)),
                WindGust10m = Max(hourRecords.Select(r => r.WindGustMs)),
                WindDirection10m = CircularMeanDegrees(hourRecords.Select(r => r.WindDirDeg)),
            });
        }

        return rows;
    }

    /// <summary>One parsed 15-minute ISS archive record, already in SI units.</summary>
    internal readonly record struct IssRecord(
        DateTime Ts,
        double? TempAvgC,
        double? TempHiC,
        double? TempLoC,
        double? HumidityPct,
        double? DewPointC,
        double? RainfallMm,
        double? RainRateMmHr,
        double? WindSpeedMs,
        double? WindGustMs,
        double? WindDirDeg);

    // ---- aggregation helpers (null-safe; null if no non-null values) ----------

    private static double? Mean(IEnumerable<double?> values)
    {
        var vals = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        return vals.Count == 0 ? null : vals.Average();
    }

    private static double? Sum(IEnumerable<double?> values)
    {
        var vals = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        return vals.Count == 0 ? null : vals.Sum();
    }

    private static double? Max(IEnumerable<double?> values)
    {
        var vals = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        return vals.Count == 0 ? null : vals.Max();
    }

    private static double? Min(IEnumerable<double?> values)
    {
        var vals = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        return vals.Count == 0 ? null : vals.Min();
    }

    /// <summary>
    /// Circular mean of compass bearings: atan2(mean sin, mean cos), normalised
    /// to [0, 360). Null directions are skipped; null if none remain.
    /// </summary>
    internal static double? CircularMeanDegrees(IEnumerable<double?> degrees)
    {
        double sumSin = 0, sumCos = 0;
        var n = 0;
        foreach (var d in degrees)
        {
            if (!d.HasValue) continue;
            var rad = d.Value * Math.PI / 180.0;
            sumSin += Math.Sin(rad);
            sumCos += Math.Cos(rad);
            n++;
        }
        if (n == 0) return null;

        var mean = Math.Atan2(sumSin / n, sumCos / n) * 180.0 / Math.PI;
        if (mean < 0) mean += 360.0;
        // Guard the [0,360) upper edge against floating-point landing exactly on 360.
        if (mean >= 360.0) mean -= 360.0;
        return mean;
    }

    private static double? FahrenheitToCelsius(double? f) => f.HasValue ? (f.Value - 32.0) / 1.8 : null;

    private static double? MphToMs(double? mph) => mph.HasValue ? mph.Value * 0.44704 : null;

    // ---- null-safe JSON getters ----------------------------------------------

    private static double? GetDouble(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        return el.ValueKind == JsonValueKind.Number ? el.GetDouble() : null;
    }

    private static long? GetLong(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        if (el.ValueKind == JsonValueKind.Number && el.TryGetInt64(out var l)) return l;
        return null;
    }

    private static int? GetInt(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var i)) return i;
        return null;
    }

    private static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "…";
}
