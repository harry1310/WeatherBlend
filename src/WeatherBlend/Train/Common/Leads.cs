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
}
