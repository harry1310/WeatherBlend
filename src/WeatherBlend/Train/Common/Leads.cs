namespace WeatherBlend.Train.Common;

/// <summary>
/// Single source of truth for the per-target lead-hour sets we train,
/// predict, and render at. Two flavours:
///
///   <see cref="Full"/>  — temperature + precipitation: {24, 48, 72, 96, 120}.
///                         These targets have data out to 120h via Open-Meteo
///                         Previous Runs (day-offset-5). The site renders
///                         home cards, per-lead forecasts, rolling-MAE charts,
///                         and the Models per-lead table off this set.
///   <see cref="Short"/> — dry-window + element blenders (wind/humidity/cloud/
///                         radiation): {24, 48, 72}. Capped at 72h pending a
///                         separate scoping decision; Open-Meteo MF caps at
///                         ~72h and the dry-window day-window aggregation
///                         hasn't been validated past three days.
/// </summary>
public static class Leads
{
    public static readonly int[] Full = { 24, 48, 72, 96, 120 };
    public static readonly int[] Short = { 24, 48, 72 };

    /// <summary>
    /// Lead set for the temperature + rain forecasts pages' sub-nav and the
    /// matching <c>forecasts-{temp|rain}-{N}h.html</c> file emission. Same as
    /// <see cref="Full"/> but with +12h prepended for the exact-runtime
    /// blenders (Phase 2d / 3d) which train at {12, 24} and own the +12h
    /// champion slot. Other consumers (rolling-MAE chart, Models per-lead
    /// table) keep using <see cref="Full"/> so legacy 2b/2c/3a/3c phases
    /// don't render empty +12h cells for leads they were never trained on.
    /// </summary>
    public static readonly int[] ForecastsTempRain = { 12, 24, 48, 72, 96, 120 };
}
