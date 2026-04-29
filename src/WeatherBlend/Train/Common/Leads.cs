namespace WeatherBlend.Train.Common;

/// <summary>
/// Single source of truth for the per-target lead-hour sets we train and
/// predict at. Two flavours:
///
///   <see cref="Full"/>  — temperature + precipitation: {24, 48, 72, 96, 120}.
///                         These targets have data out to 120h via Open-Meteo
///                         Previous Runs (day-offset-5).
///   <see cref="Short"/> — dry-window + element blenders (wind/humidity/cloud/
///                         radiation): {24, 48, 72}. Capped at 72h pending a
///                         separate scoping decision; Open-Meteo MF caps at
///                         ~72h and the dry-window day-window aggregation
///                         hasn't been validated past three days.
///
/// Site rendering keeps its own <c>PocLeads</c> in <c>SitePages.cs</c> — that's
/// intentionally a different shape (no 96h yet) so the site lead-set can be
/// promoted independently of the trainer/predictor lead-set.
/// </summary>
public static class Leads
{
    public static readonly int[] Full = { 24, 48, 72, 96, 120 };
    public static readonly int[] Short = { 24, 48, 72 };
}
