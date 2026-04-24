using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One Met Office Land Observations row for one UK land location, keyed by the
/// DataHub's geohash addressing (not a station code). Obs are delivered from
/// the nearest physical synoptic station to the supplied geohash.
///
/// Schema mirrors the Land Observations public API response (v1.0): nine numeric
/// fields plus a pressure-tendency category. Units are kept as delivered:
/// <list type="bullet">
///   <item>AirTemperature — °C</item>
///   <item>Humidity — integer percent, stored as double</item>
///   <item>WindSpeed10m / WindGusts10m — m/s (NB: METAR obs use knots)</item>
///   <item>WindDirection10m — degrees, decoded from the API's 16-point compass string</item>
///   <item>MeanSeaLevelPressure — hPa (NB: Spot forecast mslp is in Pa — not the same units)</item>
///   <item>Visibility — metres</item>
///   <item>WeatherCode — Met Office significantWeatherCode integer</item>
///   <item>PressureTendency — "R" | "F" | "S" (rising/falling/steady)</item>
/// </list>
///
/// The API does not return dewpoint or accumulated precipitation, so those
/// columns are deliberately absent from the schema.
/// </summary>
public sealed class MetOfficeObservationRow
{
    [SetsRequiredMembers]
    public MetOfficeObservationRow()
    {
        LocationName = "";
        Geohash = "";
        Area = "";
    }

    public required string LocationName { get; init; }
    public required string Geohash { get; init; }
    public required string Area { get; init; }
    public required DateTime ObservedTimeUtc { get; init; }

    public double? AirTemperature { get; init; }
    public double? Humidity { get; init; }
    public double? WindSpeed10m { get; init; }
    public double? WindDirection10m { get; init; }
    public double? WindGusts10m { get; init; }
    public double? MeanSeaLevelPressure { get; init; }
    public double? Visibility { get; init; }
    public int? WeatherCode { get; init; }
    public string? PressureTendency { get; init; }
}
