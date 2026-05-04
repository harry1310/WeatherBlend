using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// One dry-window training phase ("3b" or "3g"). Mirrors
/// <see cref="PrecipPhase"/> for the dry-window champion/challenger group:
/// 3b is the production champion, 3g is the parameter-free MC challenger.
/// 3d-shape / 3d-calibrated / 3e / 3f were retired 2026-05-04 — see
/// <see cref="Models.ActivePhasePolicy"/> for the shipping list.
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

    public static readonly DryWindowPhase Phase3g = new(
        Key: "3g",
        LongTitle: "Phase 3g — Monte Carlo over Phase 3a hourly P(wet) marginals",
        ShortTitle: "Phase 3g (MC)",
        Description: "Parameter-free. For each daytime hour, sample 10,000 Bernoullis using Phase 3a's hourly P(wet); count the fraction of samples whose longest dry run reaches the target window length. No LightGBM, no learned weights — the prediction is purely 3a's per-hour view + the structural rule that longer windows are rarer. Cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) holds by construction (single MC pass, three indicators read off the same Bernoulli sequence).",
        ChampionVsChallengerLabel: "Phase 3g (MC)",
        Color: "#43a047");

    /// <summary>
    /// Display-metadata records keyed by phase string. Source of truth for
    /// "if this phase ever ships, here's what its card / heading looks like";
    /// pure presentation, no membership claim. Membership is decided by
    /// <see cref="ActivePhasePolicy"/> — see <see cref="All"/>.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, DryWindowPhase> _byKey =
        new Dictionary<string, DryWindowPhase>(StringComparer.OrdinalIgnoreCase)
        {
            [Phase3b.Key] = Phase3b,
            [Phase3g.Key] = Phase3g,
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
    /// As of 2026-05-04 the policy lists "3b" + "3g".
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
