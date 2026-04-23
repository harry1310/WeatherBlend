namespace WeatherBlend.Site;

/// <summary>
/// One dry-window training phase ("3b", "3d-shape", "3d-calibrated", or "other").
/// Mirrors <see cref="PrecipPhase"/> for the dry-window champion/challenger group:
/// 3b is the production champion, 3d-shape adds 7 within-day shape features, and
/// 3d-calibrated wraps 3b's output in a per-lead PAV isotonic remapping.
/// </summary>
/// <param name="Key">Stable identifier matching <c>training_metadata.Phase</c> exactly.</param>
/// <param name="LongTitle">Heading used on the dry-window page where space allows the full feature-count gloss.</param>
/// <param name="ShortTitle">Heading used elsewhere when the section title also names the phase.</param>
/// <param name="Description">Skill-line paragraph rendered under the section heading.</param>
/// <param name="ChampionVsChallengerLabel">Series label for any side-by-side overlay. <c>null</c> means the phase is not plotted there.</param>
/// <param name="Color">SVG colour for that overlay. <c>null</c> when <see cref="ChampionVsChallengerLabel"/> is also null.</param>
public sealed record DryWindowPhase(
    string Key,
    string LongTitle,
    string ShortTitle,
    string Description,
    string? ChampionVsChallengerLabel,
    string? Color);

/// <summary>
/// The four dry-window phase buckets and helpers for mapping a prediction row's
/// metadata phase onto one of them. Buckets are ordered for rendering: callers
/// can iterate <see cref="All"/> directly to get the canonical order.
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

    public static readonly DryWindowPhase Phase3dCalibrated = new(
        Key: "3d-calibrated",
        LongTitle: "Phase 3d-calibrated — 3b + post-hoc PAV calibration",
        ShortTitle: "Phase 3d-calibrated (PAV)",
        Description: "Phase 3b's model unchanged; its probabilities are re-mapped through a per-lead pool-adjacent-violators isotonic regression fit on the validation slice. Same model file, same features, same feature hash — only the mapping changes. Tests whether calibration alone moves the needle once 3b is already well-calibrated.",
        ChampionVsChallengerLabel: "Phase 3d-calibrated (challenger)",
        Color: "#26a69a");

    public static readonly DryWindowPhase Other = new(
        Key: "other",
        LongTitle: "Other versions",
        ShortTitle: "Other versions",
        Description: "Versions with no phase tag on disk — typically pre-3b experiments left in the manifest.",
        ChampionVsChallengerLabel: null,
        Color: null);

    /// <summary>Canonical render order: 3b → 3d-shape → 3d-calibrated → other.</summary>
    public static readonly IReadOnlyList<DryWindowPhase> All = new[]
    {
        Phase3b, Phase3dShape, Phase3dCalibrated, Other,
    };

    /// <summary>Phases that participate in the three-way overlay (those with a label and colour).</summary>
    public static readonly IReadOnlyList<DryWindowPhase> Comparable = All
        .Where(p => p.ChampionVsChallengerLabel is not null)
        .ToList();

    /// <summary>
    /// Bucket a dry-window version into its phase. Returns <see cref="Other"/> when
    /// the version is missing from the lookup, has an empty/whitespace tag, or its
    /// tag isn't one of the three known phase keys.
    /// </summary>
    public static DryWindowPhase Bucket(IReadOnlyDictionary<string, string> phaseByVersion, string version)
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
}
