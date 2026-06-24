namespace WeatherBlend.Train.Common;

/// <summary>
/// Guards against the "trained a bake-off on NULL-padded pre-2024 rows" footgun
/// (2026-06-24 incident). The Open-Meteo historical-forecast (<c>offset_day</c>) tree
/// only carries ~2 models before 2024-01-01, so a rich dataset built with NO
/// <c>minValidTime</c> silently sweeps in sparse rows whose ensemble features are mostly
/// NULL — which poisons feature bake-offs without any visible error. Production trainers
/// always pass the <c>phases.yaml</c> cutoff; offline harnesses historically forgot to.
/// This makes the omission throw rather than fail silently.
/// </summary>
public static class RichDatasetGuard
{
    /// <summary>Canonical safe training cutoff — mirrors the <c>minValidTime</c> in
    /// <c>Config/phases.yaml</c> for the rich phases (2b/2c/3a/3c). Offline harnesses
    /// should pass THIS (or a later date) instead of <c>null</c>, so they train on the
    /// same window production does.</summary>
    public static readonly DateTime ProductionCutoff = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    /// <summary>
    /// Throw unless the caller either supplied a cutoff or explicitly opted into full
    /// (sparse) history. Phase 3o legitimately wants full history — its terrain features
    /// stay informative on pre-2024 rows where the NWP ensemble is sparse — and passes
    /// <paramref name="allowFullHistory"/> = true on purpose.
    /// </summary>
    public static void RequireCutoff(DateTime? minValidTime, bool allowFullHistory, string what)
    {
        if (minValidTime is null && !allowFullHistory)
            throw new ArgumentException(
                $"{what}: rich dataset requested with minValidTime=null. Pre-2024 NWP rows are " +
                $"NULL-padded (only ~2 models before {ProductionCutoff:yyyy-MM-dd}) and silently poison " +
                $"feature bake-offs. Pass an explicit minValidTime (production uses the phases.yaml cutoff " +
                $"{ProductionCutoff:yyyy-MM-dd} — RichDatasetGuard.ProductionCutoff), or set " +
                $"allowFullHistory:true to opt into the full sparse history on purpose (Phase 3o does).",
                nameof(minValidTime));
    }
}
