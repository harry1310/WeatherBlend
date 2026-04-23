namespace WeatherBlend.Site;

/// <summary>
/// One precipitation training phase ("3a", "3a_isotonic", "3c", or "other").
/// Holds the labels and chart palette colour shared by every page that buckets
/// precipitation predictions by phase, so the same string literal can't drift
/// out of sync between the three SitePages render methods and the
/// PrecipPredictCommand metadata check.
/// </summary>
/// <param name="Key">Stable identifier matching <c>training_metadata.Phase</c> (or "other" for the catch-all bucket).</param>
/// <param name="LongTitle">Heading used on the precipitation page where space allows the full feature-count gloss.</param>
/// <param name="ShortTitle">Heading used on forecast-vs-truth where the chart title also names the phase.</param>
/// <param name="Description">Skill-line paragraph rendered under the chart heading.</param>
/// <param name="ChampionVsChallengerLabel">Series label for the +24h three-way overlay. <c>null</c> means the phase is not plotted in that overlay (e.g. "other").</param>
/// <param name="Color">SVG colour for the +24h overlay. <c>null</c> when <see cref="ChampionVsChallengerLabel"/> is also null.</param>
public sealed record PrecipPhase(
    string Key,
    string LongTitle,
    string ShortTitle,
    string Description,
    string? ChampionVsChallengerLabel,
    string? Color);

/// <summary>
/// The four precipitation phase buckets and helpers for mapping a prediction
/// row's metadata phase onto one of them. Buckets are ordered for rendering:
/// callers can iterate <see cref="All"/> directly to get the canonical order.
/// </summary>
public static class PrecipPhases
{
    public static readonly PrecipPhase Phase3a = new(
        Key: "3a",
        LongTitle: "Phase 3a — lean (27 features)",
        ShortTitle: "Phase 3a (lean)",
        Description: "Per-model precip + precip-prob, spread stats, covariate means, calendar encodings. Original champion.",
        ChampionVsChallengerLabel: "Phase 3a (champion)",
        Color: "#90a4ae");

    public static readonly PrecipPhase Phase3aIsotonic = new(
        Key: "3a_isotonic",
        LongTitle: "Phase 3a_isotonic — lean + post-hoc calibration",
        ShortTitle: "Phase 3a_isotonic (lean + PAV calibration)",
        Description: "Phase 3a's model unchanged; its output is remapped through a per-lead pool-adjacent-violators isotonic regression fit on the validation slice. Tests whether calibration post-processing alone delivers what Phase 3c's extra features deliver.",
        ChampionVsChallengerLabel: "Phase 3a_isotonic (calibrated)",
        Color: "#26a69a");

    public static readonly PrecipPhase Phase3c = new(
        Key: "3c",
        LongTitle: "Phase 3c — rich (55 features)",
        ShortTitle: "Phase 3c (rich)",
        Description: "Lean + 18 per-model humidity (dew/RH/depression) + 6 per-model surface pressure + 4 EA trailing-rainfall persistence features. Challenger.",
        ChampionVsChallengerLabel: "Phase 3c (challenger)",
        Color: "#7c4dff");

    public static readonly PrecipPhase Other = new(
        Key: "other",
        LongTitle: "Other versions",
        ShortTitle: "Other versions",
        Description: "Versions with no phase tag on disk — typically pre-3a experiments left in the manifest.",
        ChampionVsChallengerLabel: null,
        Color: null);

    /// <summary>Canonical render order: 3a → 3a_isotonic → 3c → other.</summary>
    public static readonly IReadOnlyList<PrecipPhase> All = new[]
    {
        Phase3a, Phase3aIsotonic, Phase3c, Other,
    };

    /// <summary>Phases that participate in the +24h three-way overlay (those with a label and colour).</summary>
    public static readonly IReadOnlyList<PrecipPhase> Comparable = All
        .Where(p => p.ChampionVsChallengerLabel is not null)
        .ToList();

    /// <summary>
    /// Bucket a precipitation version into its phase. Returns <see cref="Other"/>
    /// when the version is missing from the lookup, has an empty/whitespace tag,
    /// or its tag isn't one of the three known phase keys.
    /// </summary>
    public static PrecipPhase Bucket(IReadOnlyDictionary<string, string> phaseByVersion, string version)
    {
        if (!phaseByVersion.TryGetValue(version, out var phase) || string.IsNullOrWhiteSpace(phase))
            return Other;
        foreach (var p in All)
        {
            if (p == Other) continue;
            if (phase.Equals(p.Key, StringComparison.OrdinalIgnoreCase)) return p;
        }
        return Other;
    }

    /// <summary>True iff <paramref name="metadataPhase"/> matches the 3a_isotonic key. Used by predict to enable PAV remapping.</summary>
    public static bool IsIsotonic(string? metadataPhase)
        => string.Equals(metadataPhase, Phase3aIsotonic.Key, StringComparison.OrdinalIgnoreCase);

    /// <summary>True iff <paramref name="metadataPhase"/> matches the 3c key. Used by predict to enable rich-feature loading.</summary>
    public static bool IsRich(string? metadataPhase)
        => string.Equals(metadataPhase, Phase3c.Key, StringComparison.OrdinalIgnoreCase);
}
