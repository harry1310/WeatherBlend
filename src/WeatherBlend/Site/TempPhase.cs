namespace WeatherBlend.Site;

/// <summary>
/// Temperature phase → chart colour, single source of truth so the
/// rolling-MAE panel, the new phase-comparison chart, and any future
/// per-phase render path all draw 2b in the same purple, 2c in the same
/// teal, etc. Mirrors the precipitation side's <see cref="PrecipPhases"/>
/// palette (3a→2b purple, 3c→2c teal, 3d→2d red) so a reader cross-
/// referencing the temp + rain skill pages doesn't have to remap colours
/// in their head.
///
/// Temperature has no equivalent of 4a (BART) on this side yet, so the
/// amber slot is reserved for whenever a future temp blender appears as
/// a fourth active phase.
/// </summary>
public static class TempPhases
{
    private static readonly Dictionary<string, string> ColorByKey = new(StringComparer.Ordinal)
    {
        ["2b"] = "#7c4dff",  // purple — current champion (lean blender)
        ["2c"] = "#26a69a",  // teal — rich-feature challenger
        ["2d"] = "#ef5350",  // red — exact-runtime, +12h champion
    };

    /// <summary>Lookup a phase colour by key. Falls back to neutral grey
    /// for unknown / retired phases so a stale rolling-MAE row stays
    /// visible without claiming a known-phase colour.</summary>
    public static string ColorFor(string phaseKey)
        => ColorByKey.TryGetValue(phaseKey, out var c) ? c : "#9e9e9e";
}
