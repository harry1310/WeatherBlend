using WeatherBlend.Models;

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

    public static readonly DryWindowPhase Phase3e = new(
        Key: "3e",
        LongTitle: "Phase 3e — conditional cascade for 3h + 4h windows",
        ShortTitle: "Phase 3e (cascade)",
        Description: "B2 decomposition. Trains a base classifier for P(3h dry block) and a conditional classifier for P(extends to 4h | has 3h block). At predict time, P(3h) = M_base, P(4h) = M_base × M_extend4. Monotonicity P(4h) ≤ P(3h) holds by construction. 6h stays on 3b because the conditional subset is too sparse to be reliable.",
        ChampionVsChallengerLabel: "Phase 3e (cascade)",
        Color: "#26a69a");

    /// <summary>
    /// Display-metadata records keyed by phase string. Source of truth for
    /// "if this phase ever ships, here's what its card / heading looks like";
    /// pure presentation, no membership claim. Membership is decided by
    /// <see cref="ActivePhasePolicy"/> — see <see cref="All"/>.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, DryWindowPhase> _byKey =
        new Dictionary<string, DryWindowPhase>(StringComparer.OrdinalIgnoreCase)
        {
            [Phase3b.Key]      = Phase3b,
            [Phase3dShape.Key] = Phase3dShape,
            [Phase3e.Key]      = Phase3e,
        };

    /// <summary>
    /// Phases the site renders, in champion-first order. Derived from
    /// <see cref="ActivePhasePolicy"/> so adding/removing a phase is a
    /// one-line change in the policy file — no risk of drift between
    /// "what the policy says is shipping" and "what the dry-window page
    /// loops over". Phases listed in the policy without a matching
    /// <see cref="DryWindowPhase"/> record are skipped (with the assumption
    /// that anyone adding a phase to the policy will also add its display
    /// metadata here in the same change).
    ///
    /// As of 2026-04-29 the policy lists only "3b"; <see cref="Phase3dShape"/>
    /// stays in <see cref="_byKey"/> as a ready-to-render record so a future
    /// re-promotion is a one-line policy edit.
    /// </summary>
    public static IReadOnlyList<DryWindowPhase> All =>
        ActivePhasePolicy.ByTarget["dry_window"]
            .Where(_byKey.ContainsKey)
            .Select(k => _byKey[k])
            .ToList();

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
