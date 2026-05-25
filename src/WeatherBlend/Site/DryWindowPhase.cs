using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// One dry-window training phase. Mirrors <see cref="PrecipPhase"/> for the
/// dry-window champion/challenger group: 3b is the production champion, 3g
/// is the parameter-free MC challenger, 3j (added 2026-05-13) is the
/// Gaussian copula MC sibling of 3g that captures within-day wet/dry
/// autocorrelation. 3d-shape / 3d-calibrated / 3e (legacy) / 3f (legacy)
/// were retired 2026-05-04 — see <see cref="Models.ActivePhasePolicy"/>
/// for the shipping list.
/// </summary>
/// <param name="Key">Stable identifier matching <c>training_metadata.Phase</c> exactly.</param>
/// <param name="LongTitle">Heading used on the dry-window page where space allows the full feature-count gloss.</param>
/// <param name="ShortTitle">Heading used elsewhere when the section title also names the phase.</param>
/// <param name="Description">Skill-line paragraph rendered under the section heading.</param>
/// <param name="ChampionVsChallengerLabel">Series label for any side-by-side overlay.</param>
/// <param name="Color">SVG colour for that overlay.</param>
/// <param name="StartHourCurveVersion">
/// On-disk <c>model_version</c> of this phase's start-hour curve under
/// <c>data/predictions/dry_window_start_hour/…/model_version=…/</c>, or
/// <c>null</c> when the phase produces no start-hour curve. Only the
/// iid-MC phases have one — 3g (curve <c>v2</c>, MC over 3a) and 3s
/// (curve <c>v2-3e</c>, MC over 3e). The renderer routes each phase's
/// "best start" column + start-hour chart to its own curve via this key,
/// so 3g and 3s never read each other's curve.
/// </param>
public sealed record DryWindowPhase(
    string Key,
    string LongTitle,
    string ShortTitle,
    string Description,
    string ChampionVsChallengerLabel,
    string Color,
    string? StartHourCurveVersion = null);

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
        Description: "53-feature LightGBM. Champion.",
        ChampionVsChallengerLabel: "Phase 3b (champion)",
        Color: "#90a4ae");

    // Phase 3g / 3j / 3n / 3s display records retired 2026-05-25 in
    // model-cleanup Phase 1. Cleanup Phase 2 reintroduces 3p (Gaussian
    // copula MC over 3o = 3c-oro) as the dry-window challenger.
    public static readonly DryWindowPhase Phase3p = new(
        Key: "3p",
        LongTitle: "Phase 3p — Gaussian copula MC over Phase 3o hourly P(wet)",
        ShortTitle: "Phase 3p (copula MC/3o)",
        Description: "Gaussian copula MC over the rich+orographic Phase 3o's hourly P(wet) marginals. Single empirical Σ per station, fit on train-split observed daytime wet/dry binary sequences (not per-lead — daytime-shape autocorrelation is lead-independent for truth). Bake-off Brier 0.1064 vs current 3g 0.1265 (−15.9%); composition: ~6% from source swap (3a→3o), ~11% from algorithm swap (iid→copula). Cleanup Phase 2 dry-window challenger.",
        ChampionVsChallengerLabel: "Phase 3p (copula MC/3o)",
        Color: "#7e57c2");

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
            [Phase3p.Key] = Phase3p,
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
    /// As of 2026-05-13 the policy lists "3b" + "3g" + "3j".
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
