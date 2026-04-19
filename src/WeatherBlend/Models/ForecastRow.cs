namespace WeatherBlend.Models;

/// <summary>
/// One hourly forecast row for one model, one run, one valid-time.
/// Nullable numeric fields so the schema is stable even if a model
/// doesn't expose a given variable.
/// </summary>
public sealed class ForecastRow
{
    public required string LocationName { get; init; }
    public required string Model { get; init; }
    public required DateTime RunTimeUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    public double? Temperature2m { get; init; }
    public double? DewPoint2m { get; init; }
    public double? RelativeHumidity2m { get; init; }
    public double? Precipitation { get; init; }
    public double? PrecipitationProbability { get; init; }
    public double? Rain { get; init; }
    public double? Showers { get; init; }
    public double? Snowfall { get; init; }
    public double? CloudCover { get; init; }
    public double? CloudCoverLow { get; init; }
    public double? CloudCoverMid { get; init; }
    public double? CloudCoverHigh { get; init; }
    public double? WindSpeed10m { get; init; }
    public double? WindDirection10m { get; init; }
    public double? WindGusts10m { get; init; }
    public double? SurfacePressure { get; init; }
    public double? Cape { get; init; }
    public double? Visibility { get; init; }
}
