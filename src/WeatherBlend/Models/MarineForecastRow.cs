namespace WeatherBlend.Models;

/// <summary>
/// One hourly marine (wave) forecast row for one wave model, one run, one
/// valid-time — the sea-state sibling of <see cref="ForecastRow"/>, kept as
/// its own class + parquet tree (data/marine/) so the weather forecast tree
/// isn't bloated with always-null wave columns. The coordinate is the
/// location's PINNED marine grid cell (config marine: block), not the
/// land/site coordinate.
/// </summary>
public sealed class MarineForecastRow
{
    public required string LocationName { get; init; }
    public required string Model { get; init; }
    public required DateTime RunTimeUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    /// <summary>Same vocabulary as <see cref="ForecastRow.RunTimeSource"/>:
    /// "synthesised" (live cycle, wall-clock-floored), "offset_day"
    /// (previous_dayN columns; RunTime = ValidTime − 24·N), "hist_forecast"
    /// (marine hindcast archive — best-available per valid-time, lead-unlabelled).</summary>
    public string? RunTimeSource { get; init; }

    // Total sea state.
    public double? WaveHeight { get; init; }
    public double? WavePeriod { get; init; }
    public double? WaveDirection { get; init; }

    // Locally wind-driven component.
    public double? WindWaveHeight { get; init; }
    public double? WindWavePeriod { get; init; }
    public double? WindWaveDirection { get; init; }

    // Primary groundswell — the run-up driver for the wetting-elevation product.
    public double? SwellWaveHeight { get; init; }
    public double? SwellWavePeriod { get; init; }
    public double? SwellWaveDirection { get; init; }

    // Secondary swell train — best_match only (null on per-model pulls).
    public double? SecondarySwellWaveHeight { get; init; }
    public double? SecondarySwellWavePeriod { get; init; }
    public double? SecondarySwellWaveDirection { get; init; }

    /// <summary>Sea level vs MSL (m), tide + surge — best_match only. The
    /// tide-curve input of the wetting-elevation combiner; cross-check vs
    /// Newlyn harmonics before the site chip trusts it.</summary>
    public double? SeaLevelHeightMsl { get; init; }

    /// <summary>SST (°C) — best_match only. Spray/greasiness context.</summary>
    public double? SeaSurfaceTemperature { get; init; }
}
