using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One METAR observation, normalised.
/// Units: temperature/dewpoint in °C, wind speed in m/s,
/// pressure in hPa, visibility in metres.
/// </summary>
public sealed class ObservationRow
{
    // Parquet.Net's DeserializeAsync needs a parameterless ctor; SetsRequiredMembers
    // satisfies the required-member contract for that path.
    [SetsRequiredMembers]
    public ObservationRow()
    {
        LocationName = "";
        Station = "";
        RawMetar = "";
    }

    public required string LocationName { get; init; }
    public required string Station { get; init; }
    public required DateTime ObservedTimeUtc { get; init; }
    public required string RawMetar { get; init; }

    public double? Temperature2m { get; init; }
    public double? DewPoint2m { get; init; }
    public double? WindSpeed10m { get; init; }
    public double? WindDirection10m { get; init; }
    public double? WindGusts10m { get; init; }
    public double? SurfacePressure { get; init; }
    public double? Visibility { get; init; }

    /// <summary>Present-weather codes from the METAR, e.g. ["RA","-RA"].</summary>
    public string[] PresentWeather { get; init; } = Array.Empty<string>();
    public bool HasPrecipitation { get; init; }
}
