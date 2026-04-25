using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One hourly ERA5 reanalysis row. Treated as gapless training truth.
/// Same variable set as ForecastRow so the blender can join on (location, valid_time).
/// </summary>
public sealed class Era5Row
{
    [SetsRequiredMembers]
    public Era5Row()
    {
        LocationName = "";
    }

    public required string LocationName { get; init; }
    public required DateTime ValidTimeUtc { get; init; }

    public double? Temperature2m { get; init; }
    public double? DewPoint2m { get; init; }
    public double? RelativeHumidity2m { get; init; }
    public double? Precipitation { get; init; }
    public double? Rain { get; init; }
    public double? Snowfall { get; init; }
    public double? CloudCover { get; init; }
    public double? CloudCoverLow { get; init; }
    public double? CloudCoverMid { get; init; }
    public double? CloudCoverHigh { get; init; }
    public double? WindSpeed10m { get; init; }
    public double? WindDirection10m { get; init; }
    public double? WindGusts10m { get; init; }
    public double? SurfacePressure { get; init; }
    public double? Visibility { get; init; }
    public double? ShortwaveRadiation { get; init; }
    public double? DirectRadiation { get; init; }
    public double? DiffuseRadiation { get; init; }
}
