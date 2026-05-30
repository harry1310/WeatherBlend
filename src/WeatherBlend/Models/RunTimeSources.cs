namespace WeatherBlend.Models;

/// <summary>
/// Canonical values for <see cref="ForecastRow.RunTimeSource"/>. Use these constants
/// when writing rows so downstream queries (DuckDB, feature builder) can filter
/// consistently.
/// </summary>
public static class RunTimeSources
{
    /// <summary>Real cycle stamp from a direct-source GRIB archive.</summary>
    public const string Exact = "exact";

    /// <summary>Cycle stamp reported by a model-metadata endpoint; approximate.</summary>
    public const string Reported = "reported";

    /// <summary>Fabricated (midnight-of-valid-day, fetch-time-floored). Not trustworthy.</summary>
    public const string Synthesised = "synthesised";

    /// <summary>
    /// Open-Meteo Previous Runs API row. RunTime = ValidTime − 24·N h; LeadHours = 24·N
    /// is the lower edge of the [24N..24N+23] bucket the API actually covers.
    /// </summary>
    public const string OffsetDay = "offset_day";

    /// <summary>
    /// Open-Meteo Historical Forecast API row, used only to backfill pressure-level
    /// (…hPa) fields the Previous Runs API refuses to serve. The endpoint exposes no
    /// run/initialisation time and the per-lead (previous_day) mechanism rejects
    /// pressure variables, so these rows are LEAD-UNLABELLED: RunTime is set equal to
    /// ValidTime and LeadHours = 0 as placeholders, NOT a real cycle. Treat the
    /// pressure columns as a lead-invariant upper-air field joined by valid-time;
    /// never mix these rows into per-lead-time training as if they were offset_day.
    /// </summary>
    public const string HistForecast = "hist_forecast";
}
