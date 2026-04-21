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
}
