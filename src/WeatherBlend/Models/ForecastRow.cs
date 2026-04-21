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

    /// <summary>
    /// How the RunTime/LeadHours fields were derived. Nullable for back-compat with pre-phase-3 files.
    /// "exact" — real cycle stamp from a direct-source GRIB archive (GFS S3, ECMWF Open Data, DWD, WB2).
    /// "reported" — cycle stamp reported by a model-metadata endpoint (e.g. Open-Meteo's
    ///              last_run_initialisation_time) — approximate, may not match what the forecast endpoint served.
    /// "synthesised" — fabricated for backward compatibility (e.g. midnight-of-valid-day, fetch-time-floored).
    ///                 Do not use for per-lead-time training.
    /// "offset_day"  — Open-Meteo Previous Runs API row. RunTime = ValidTime − 24·N h and
    ///                 LeadHours = 24·N is the lower edge of the [24N..24N+23] bucket.
    /// null — legacy file; treat as "synthesised" for training purposes.
    /// </summary>
    public string? RunTimeSource { get; init; }

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
