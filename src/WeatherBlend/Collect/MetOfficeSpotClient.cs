using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using RunTimeSources = WeatherBlend.Models.RunTimeSources;

namespace WeatherBlend.Collect;

/// <summary>
/// Met Office Weather DataHub — Global Spot (site-specific) deterministic forecast.
/// Free tier: 360 calls/day, apikey header, GeoJSON response.
///
/// We pull the hourly endpoint (48h horizon) once per collect cycle and also the
/// three-hourly endpoint (168h horizon) to cover the longer lead-hour buckets.
/// When both endpoints return a row for the same ValidTime the hourly one wins
/// (it's the native resolution).
///
/// Rows are written into the standard forecasts tree under model=met_office_spot
/// so the existing compare/verify/blender tooling treats this as one more model.
///
/// Pressure note: Met Office reports <c>mslp</c> (mean-sea-level pressure) in Pa.
/// Our SurfacePressure field is station-level in hPa; at Bonehill (393m) the two
/// diverge by ~50 hPa, so we leave SurfacePressure null for Met Office rows rather
/// than feed a biased value into the blender.
/// </summary>
public sealed class MetOfficeSpotClient
{
    private const string Base = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point";

    private readonly HttpClient _http;
    private readonly ILogger<MetOfficeSpotClient> _log;

    public MetOfficeSpotClient(HttpClient http, ILogger<MetOfficeSpotClient> log)
    {
        _http = http;
        _log = log;
    }

    public async Task<List<ForecastRow>> FetchAsync(
        LocationConfig location,
        string modelTag,
        string apiKey,
        CancellationToken ct)
    {
        var hourly = await FetchFrequencyAsync("hourly", location, modelTag, apiKey, ct);
        var threeHourly = await FetchFrequencyAsync("three-hourly", location, modelTag, apiKey, ct);

        var seen = new HashSet<DateTime>(hourly.Select(r => r.ValidTimeUtc));
        var merged = new List<ForecastRow>(hourly.Count + threeHourly.Count);
        merged.AddRange(hourly);
        foreach (var r in threeHourly)
        {
            if (seen.Add(r.ValidTimeUtc)) merged.Add(r);
        }
        return merged;
    }

    private async Task<List<ForecastRow>> FetchFrequencyAsync(
        string frequency,
        LocationConfig location,
        string modelTag,
        string apiKey,
        CancellationToken ct)
    {
        var ci = CultureInfo.InvariantCulture;
        var url = $"{Base}/{frequency}?latitude={location.Latitude.ToString(ci)}&longitude={location.Longitude.ToString(ci)}&excludeParameterMetadata=true";

        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.Add("apikey", apiKey);
        req.Headers.Accept.ParseAdd("application/json");

        _log.LogInformation("GET {Url}", url);
        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            _log.LogWarning("Met Office {Freq} returned {Status}: {Body}",
                frequency, (int)resp.StatusCode, Truncate(body, 300));
            return new();
        }

        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return Parse(doc.RootElement, location.Name, modelTag, _log);
    }

    internal static List<ForecastRow> Parse(
        JsonElement root,
        string locationName,
        string modelTag,
        ILogger? log = null)
    {
        if (!root.TryGetProperty("features", out var features)
            || features.ValueKind != JsonValueKind.Array
            || features.GetArrayLength() == 0)
        {
            return new();
        }

        var feature = features[0];
        if (!feature.TryGetProperty("properties", out var props)) return new();

        var modelRunDate = props.TryGetProperty("modelRunDate", out var mrd) && mrd.ValueKind == JsonValueKind.String
            ? ParseZuluToUtc(mrd.GetString())
            : (DateTime?)null;

        if (!modelRunDate.HasValue)
        {
            log?.LogWarning("Met Office response missing modelRunDate; skipping");
            return new();
        }

        if (props.TryGetProperty("requestPointDistance", out var dist) && dist.ValueKind == JsonValueKind.Number)
        {
            log?.LogInformation("Met Office grid-point distance: {Metres:F0} m", dist.GetDouble());
        }

        if (!props.TryGetProperty("timeSeries", out var series) || series.ValueKind != JsonValueKind.Array)
            return new();

        var runTime = modelRunDate.Value;
        var rows = new List<ForecastRow>(series.GetArrayLength());
        foreach (var step in series.EnumerateArray())
        {
            if (!step.TryGetProperty("time", out var timeEl) || timeEl.ValueKind != JsonValueKind.String)
                continue;
            var valid = ParseZuluToUtc(timeEl.GetString());
            if (!valid.HasValue) continue;

            var lead = (int)Math.Round((valid.Value - runTime).TotalHours);

            rows.Add(new ForecastRow
            {
                LocationName = locationName,
                Model = modelTag,
                RunTimeUtc = runTime,
                ValidTimeUtc = valid.Value,
                LeadHours = lead,
                RunTimeSource = RunTimeSources.Reported,
                // Hourly product: instantaneous screenTemperature. The
                // three-hourly product (leads ~54-168h) has NO such field —
                // only the period's max/min screen air temps — so temp was
                // null at long leads, which capped the temp skill page's
                // Spot MAE line at +48h (2026-06-12). Fall back to the
                // max/min midpoint: an approximation, but a fair long-lead
                // comparison line vs hourly ERA5 truth.
                Temperature2m = GetDouble(step, "screenTemperature")
                    ?? Midpoint(GetDouble(step, "maxScreenAirTemp"), GetDouble(step, "minScreenAirTemp")),
                DewPoint2m = GetDouble(step, "screenDewPointTemperature"),
                RelativeHumidity2m = GetDouble(step, "screenRelativeHumidity"),
                Precipitation = GetDouble(step, "precipitationRate"),
                PrecipitationProbability = GetDouble(step, "probOfPrecipitation"),
                Rain = null,
                Showers = null,
                Snowfall = GetDouble(step, "totalSnowAmount"),
                CloudCover = null,
                CloudCoverLow = null,
                CloudCoverMid = null,
                CloudCoverHigh = null,
                WindSpeed10m = GetDouble(step, "windSpeed10m"),
                WindDirection10m = GetDouble(step, "windDirectionFrom10m"),
                WindGusts10m = GetDouble(step, "windGustSpeed10m"),
                SurfacePressure = null, // mslp in Pa ≠ surface pressure in hPa; see class doc
                Cape = null,
                Visibility = GetDouble(step, "visibility"),
            });
        }
        return rows;
    }

    private static double? Midpoint(double? a, double? b)
        => a.HasValue && b.HasValue ? (a.Value + b.Value) / 2.0 : null;

    private static double? GetDouble(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out var el)) return null;
        return el.ValueKind switch
        {
            JsonValueKind.Number => el.GetDouble(),
            JsonValueKind.Null => null,
            _ => null,
        };
    }

    private static DateTime? ParseZuluToUtc(string? s)
    {
        if (string.IsNullOrEmpty(s)) return null;
        if (DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var dt))
        {
            return DateTime.SpecifyKind(dt, DateTimeKind.Utc);
        }
        return null;
    }

    private static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "…";
}
