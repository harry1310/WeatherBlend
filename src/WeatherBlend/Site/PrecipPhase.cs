namespace WeatherBlend.Site;

/// <summary>
/// One precipitation training phase ("3a" or "3c"). Holds the labels and chart
/// palette colour shared by every page that buckets precipitation predictions
/// by phase, so the same string literal can't drift out of sync between the
/// SitePages render methods and the PrecipPredictCommand metadata check.
///
/// Phase 3a_isotonic was removed 2026-04-29 — PAV calibration didn't move
/// test Brier vs raw 3a, so the bucket no longer renders.
/// </summary>
/// <param name="Key">Stable identifier matching <c>training_metadata.Phase</c>.</param>
/// <param name="LongTitle">Heading used on the precipitation page where space allows the full feature-count gloss.</param>
/// <param name="ShortTitle">Heading used on forecast-vs-truth where the chart title also names the phase.</param>
/// <param name="Description">Skill-line paragraph rendered under the chart heading.</param>
/// <param name="ChampionVsChallengerLabel">Series label for the +24h three-way overlay.</param>
/// <param name="Color">SVG colour for the +24h overlay.</param>
public sealed record PrecipPhase(
    string Key,
    string LongTitle,
    string ShortTitle,
    string Description,
    string ChampionVsChallengerLabel,
    string Color);

/// <summary>
/// The three precipitation phase buckets and helpers for mapping a prediction
/// row's metadata phase onto one of them. Buckets are ordered for rendering:
/// callers can iterate <see cref="All"/> directly to get the canonical order.
/// Unknown tags bucket to <c>null</c> and are silently skipped.
/// </summary>
public static class PrecipPhases
{
    public static readonly PrecipPhase Phase3a = new(
        Key: "3a",
        LongTitle: "Phase 3a — lean (27 features)",
        ShortTitle: "Phase 3a (lean)",
        Description: "27-feature lean. Champion.",
        ChampionVsChallengerLabel: "Phase 3a (champion)",
        Color: "#90a4ae");

    public static readonly PrecipPhase Phase3c = new(
        Key: "3c",
        LongTitle: "Phase 3c — rich (55 features)",
        ShortTitle: "Phase 3c (rich)",
        Description: "55-feature rich (adds humidity, pressure, EA persistence). Challenger.",
        ChampionVsChallengerLabel: "Phase 3c (challenger)",
        Color: "#7c4dff");

    public static readonly PrecipPhase Phase3d = new(
        Key: "3d",
        LongTitle: "Phase 3d — exact-runtime (lead 12 champion)",
        ShortTitle: "Phase 3d (exact-runtime)",
        Description: "Exact-runtime blender (GFS + IFS oper + AIFS + MO Global + UKV from raw S3). Champion at +12h, challenger at +24h. Per-station EA gauge truth.",
        ChampionVsChallengerLabel: "Phase 3d (exact-runtime)",
        Color: "#26a69a");

    /// <summary>Canonical render order: 3a → 3c → 3d.</summary>
    public static readonly IReadOnlyList<PrecipPhase> All = new[]
    {
        Phase3a, Phase3c, Phase3d,
    };

    /// <summary>Phases that participate in the +24h overlay — currently both.</summary>
    public static readonly IReadOnlyList<PrecipPhase> Comparable = All;

    /// <summary>
    /// Bucket a precipitation version into its phase. Returns <c>null</c> when
    /// the version is missing from the lookup, has an empty/whitespace tag, or
    /// its tag isn't a known phase key — callers should skip.
    /// </summary>
    public static PrecipPhase? Bucket(IReadOnlyDictionary<string, string> phaseByVersion, string version)
    {
        if (!phaseByVersion.TryGetValue(version, out var phase) || string.IsNullOrWhiteSpace(phase))
            return null;
        foreach (var p in All)
        {
            if (phase.Equals(p.Key, StringComparison.OrdinalIgnoreCase)) return p;
        }
        return null;
    }

    /// <summary>True iff <paramref name="metadataPhase"/> matches the 3c key. Used by predict to enable rich-feature loading.</summary>
    public static bool IsRich(string? metadataPhase)
        => string.Equals(metadataPhase, Phase3c.Key, StringComparison.OrdinalIgnoreCase);
}
