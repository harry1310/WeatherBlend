using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One hourly wave-truth observation row (data/truth/waves/). Two source
/// families share this schema:
///   * <c>era5_ocean</c> — ERA5 wave reanalysis via the marine API
///     (gapless training truth; total-wave triple only, no swell split).
///   * wave buoys (verification truth; adds peak period / spread / SST
///     where the buoy reports them).
/// The <see cref="Source"/> column + hive partition keep them distinct.
/// </summary>
public sealed class WaveTruthRow
{
    [SetsRequiredMembers]
    public WaveTruthRow()
    {
        LocationName = "";
        Source = "";
    }

    public required string LocationName { get; init; }

    /// <summary>"era5_ocean" or a buoy slug (e.g. "sevenstones_62107").</summary>
    public required string Source { get; init; }

    public required DateTime ValidTimeUtc { get; init; }

    /// <summary>Significant wave height, m.</summary>
    public double? WaveHeight { get; init; }

    /// <summary>Mean wave period, s (buoys: Tz where that's what's reported).</summary>
    public double? WavePeriod { get; init; }

    /// <summary>Mean wave direction, ° (coming from).</summary>
    public double? WaveDirection { get; init; }

    /// <summary>Peak (spectral) period Tp, s — buoy sources only.</summary>
    public double? PeakPeriod { get; init; }

    /// <summary>Directional spread, ° — buoy sources only.</summary>
    public double? DirectionalSpread { get; init; }

    /// <summary>Sea surface temperature, °C — buoy sources only.</summary>
    public double? SeaSurfaceTemperature { get; init; }
}
