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
    /// <summary>Mean sea level pressure (hPa). Added 2026-06-03; mirrors
    /// ForecastRow.PressureMsl so train-truth aligns to forecast inputs.
    /// Open-Meteo ERA5 exposes pressure_msl (unlike longwave, which is null).</summary>
    public double? PressureMsl { get; init; }
    public double? Visibility { get; init; }
    public double? ShortwaveRadiation { get; init; }
    public double? DirectRadiation { get; init; }
    public double? DiffuseRadiation { get; init; }
    // Top-soil (0-7cm) layer temperature — added 2026-06-03 as a surface-energy
    // validation rail for the rock surface-temp module (ROCK_SURFACE_TEMP_PLAN.md).
    // It is a DAMPED subsurface soil temp, NOT a skin temperature (Open-Meteo's
    // ERA5 exposes neither skin_temperature nor downwelling longwave — both null).
    // Validates the model's slow/mean component: daily-mean Ts tracks this at
    // r~0.97 with a physically-sensible +2C rock-vs-soil mean offset.
    public double? SoilTemperature0to7cm { get; init; }

    // ---- Multi-level (pressure-level) fields — added 2026-05-29 ----
    // Mirror of ForecastRow's pressure-level fields so train-truth aligns to
    // forecast inputs vertically. ERA5 exposes all standard pressure levels.
    public double? Temperature850hPa { get; init; }
    public double? Temperature700hPa { get; init; }
    public double? Temperature500hPa { get; init; }
    public double? GeopotentialHeight850hPa { get; init; }
    public double? GeopotentialHeight500hPa { get; init; }
    public double? WindSpeed850hPa { get; init; }
    public double? WindSpeed500hPa { get; init; }
    public double? WindDirection850hPa { get; init; }
    public double? WindDirection500hPa { get; init; }
    public double? RelativeHumidity850hPa { get; init; }
}
