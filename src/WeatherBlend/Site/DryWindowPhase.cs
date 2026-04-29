namespace WeatherBlend.Site;

/// <summary>
/// One dry-window training phase ("3b" or "3d-shape"). Mirrors
/// <see cref="PrecipPhase"/> for the dry-window champion/challenger group:
/// 3b is the production champion, 3d-shape adds 7 within-day shape features.
///
/// Phase 3d-calibrated was removed 2026-04-29 — PAV calibration didn't move
/// test Brier vs raw 3b, so the bucket no longer renders.
/// </summary>
/// <param name="Key">Stable identifier matching <c>training_metadata.Phase</c> exactly.</param>
/// <param name="LongTitle">Heading used on the dry-window page where space allows the full feature-count gloss.</param>
/// <param name="ShortTitle">Heading used elsewhere when the section title also names the phase.</param>
/// <param name="Description">Skill-line paragraph rendered under the section heading.</param>
/// <param name="ChampionVsChallengerLabel">Series label for any side-by-side overlay.</param>
/// <param name="Color">SVG colour for that overlay.</param>
public sealed record DryWindowPhase(
    string Key,
    string LongTitle,
    string ShortTitle,
    string Description,
    string ChampionVsChallengerLabel,
    string Color);

/// <summary>
/// The three dry-window phase buckets and helpers for mapping a prediction row's
/// metadata phase onto one of them. Buckets are ordered for rendering: callers
/// can iterate <see cref="All"/> directly to get the canonical order. Unknown
/// tags bucket to <c>null</c> and are silently skipped.
/// </summary>
public static class DryWindowPhases
{
    public static readonly DryWindowPhase Phase3b = new(
        Key: "3b",
        LongTitle: "Phase 3b — lean (53 features)",
        ShortTitle: "Phase 3b (lean)",
        Description: "Day-aggregate per-model precip totals, wet-hour counts, run-length stats, EA persistence, climatology, calendar encodings. Production champion.",
        ChampionVsChallengerLabel: "Phase 3b (champion)",
        Color: "#90a4ae");

    public static readonly DryWindowPhase Phase3dShape = new(
        Key: "3d-shape",
        LongTitle: "Phase 3d-shape — lean + 7 within-day shape features (60 features)",
        ShortTitle: "Phase 3d-shape (rich)",
        Description: "3b features plus first/last wet hour, longest forecast dry/wet run, n rain events, and morning/afternoon precip sums — derived from the ensemble-mean hourly precip vector. Lets the model condition on whether a wet day is 'wet morning, dry afternoon' vs constant drizzle.",
        ChampionVsChallengerLabel: "Phase 3d-shape (challenger)",
        Color: "#7c4dff");

    /// <summary>
    /// Phases the site renders. <see cref="Phase3dShape"/> is intentionally
    /// excluded as of 2026-04-29 — the shape-features-vs-lean bake-off on
    /// the new daytime label produced no consistent improvement (mean
    /// +1.6% Brier worse, 7 wins / 9 losses / 11 ties across 27 cells).
    /// The <c>Phase3dShape</c> constant + training pipeline are kept so a
    /// future revisit (e.g. with a larger validation slice) is one-line:
    /// re-add it here.
    /// </summary>
    public static readonly IReadOnlyList<DryWindowPhase> All = new[]
    {
        Phase3b,
    };

    /// <summary>Phases that participate in side-by-side overlay charts.</summary>
    public static readonly IReadOnlyList<DryWindowPhase> Comparable = All;

    /// <summary>
    /// Bucket a dry-window version into its phase. Returns <c>null</c> when the
    /// version is missing from the lookup, has an empty/whitespace tag, or its
    /// tag isn't one of the three known phase keys — callers should skip.
    /// </summary>
    public static DryWindowPhase? Bucket(IReadOnlyDictionary<string, string> phaseByVersion, string version)
    {
        if (!phaseByVersion.TryGetValue(version, out var phase) || string.IsNullOrWhiteSpace(phase))
            return null;
        foreach (var p in All)
        {
            if (phase.Equals(p.Key, StringComparison.OrdinalIgnoreCase)) return p;
        }
        return null;
    }
}
