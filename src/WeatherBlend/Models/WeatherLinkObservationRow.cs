using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One hourly observation row from a Davis WeatherLink (api.weatherlink.com v2)
/// station — real-instrument truth, the close-by sibling of the ERA5 / METAR /
/// EA-rainfall truth sources. Built for NCI Gwennap Head (station_id 115072) as
/// a near-site truth for Sennen, but the schema + collector are station-generic
/// (any location with a <c>weatherLinkStationId</c> can collect one).
///
/// The WeatherLink archive serves 15-minute records (96/day); these are
/// aggregated to HOURLY rows here to match ERA5/METAR truth granularity and the
/// hourly forecast valid-times. Units are converted to the same SI conventions
/// the rest of the truth schema uses (NOT WeatherLink's native °F / mph):
/// <list type="bullet">
///   <item>Temperature2m / TemperatureHigh / TemperatureLow / DewPoint — °C
///     (WeatherLink delivers °F)</item>
///   <item>Humidity — relative humidity percent</item>
///   <item>RainfallMm — mm, SUMMED over the hour (interval accumulation)</item>
///   <item>RainRateMmHr — mm/hr, the hour's peak rate</item>
///   <item>WindSpeed10m / WindGust10m — m/s (WeatherLink delivers mph)</item>
///   <item>WindDirection10m — degrees [0,360), circular mean over the hour</item>
///   <item>SolarRadiation — W/m² (hour mean). NOTE: only populated if the
///     station has a calibrated solar sensor; many citizen stations report a raw
///     voltage but no <c>solar_rad_avg</c>, so this lands null for them.</item>
/// </list>
/// </summary>
public sealed class WeatherLinkObservationRow
{
    [SetsRequiredMembers]
    public WeatherLinkObservationRow()
    {
        LocationName = "";
        StationId = "";
    }

    public required string LocationName { get; init; }
    public required string StationId { get; init; }
    public required DateTime ObservedTimeUtc { get; init; }

    public double? Temperature2m { get; init; }
    public double? TemperatureHigh { get; init; }
    public double? TemperatureLow { get; init; }
    public double? Humidity { get; init; }
    public double? DewPoint { get; init; }
    public double? RainfallMm { get; init; }
    public double? RainRateMmHr { get; init; }
    public double? WindSpeed10m { get; init; }
    public double? WindGust10m { get; init; }
    public double? WindDirection10m { get; init; }
    public double? SolarRadiation { get; init; }
}
