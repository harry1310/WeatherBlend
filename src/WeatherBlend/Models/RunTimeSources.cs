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
}
