using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// One dry-window training phase. Mirrors <see cref="PrecipPhase"/> for the
/// dry-window champion/challenger group: 3b is the production champion,
/// 3p is the Gaussian copula MC challenger. Since 2026-06-10 the per-phase
/// values live in phases.yaml's <c>display:</c> blocks;
/// <see cref="DryWindowPhases"/> materialises this record from
/// <see cref="PhaseRegistry"/>, so adding a phase is a registry edit, not a
/// code edit.
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
/// <c>null</c> when the phase produces no start-hour curve. Only MC
/// phases have one. The renderer routes each phase's "best start" column
/// + start-hour chart to its own curve via this key, so phases never
/// read each other's curve.
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
/// The dry-window phase buckets and helpers for mapping a prediction row's
/// metadata phase onto one of them. Buckets are ordered for rendering: callers
/// can iterate <see cref="All"/> directly — render order = registry order
/// (champion-first, as phases.yaml declares them). Unknown tags bucket to
/// <c>null</c> and are silently skipped.
/// </summary>
public static class DryWindowPhases
{
    private const string Target = "dry_window";
    private const string FallbackGrey = "#9e9e9e";

    /// <summary>
    /// Phases the site renders, champion-first (= phases.yaml order), with
    /// their display metadata from the same registry entry — membership and
    /// presentation can no longer drift apart. Materialised once; the
    /// registry is an immutable process-wide singleton.
    /// </summary>
    public static readonly IReadOnlyList<DryWindowPhase> All =
        PhaseRegistry.Default.AllPhases(Target).Select(ToPhase).ToList();

    /// <summary>Phases that participate in side-by-side overlay charts.</summary>
    public static readonly IReadOnlyList<DryWindowPhase> Comparable = All;

    /// <summary>The LightGBM champion's card — referenced directly by the
    /// dry-window page (the agreement column renders for 3b only).</summary>
    public static DryWindowPhase Phase3b => ByKey("3b");

    /// <summary>The copula-MC challenger's card — referenced directly for its
    /// <see cref="DryWindowPhase.StartHourCurveVersion"/> (home-tile + page
    /// start-hour routing).</summary>
    public static DryWindowPhase Phase3p => ByKey("3p");

    /// <summary>
    /// Bucket a dry-window version into its phase. Returns <c>null</c> when the
    /// version is missing from the lookup, has an empty/whitespace tag, or its
    /// tag isn't a known phase key — callers should skip.
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

    private static DryWindowPhase ByKey(string key)
    {
        foreach (var p in All)
            if (string.Equals(p.Key, key, StringComparison.Ordinal)) return p;
        // Defensive: only reachable if the phase is removed from phases.yaml
        // while a render path still references it by name — render grey /
        // generically-titled rather than crashing the whole site build.
        return Fallback(key);
    }

    private static DryWindowPhase ToPhase(PhaseEntry e)
    {
        // Display is enforced by PhaseWiringConsistencyTests for every
        // registry phase; the fallback keeps rendering alive (grey/untitled)
        // if a hand-edited yaml ever drops it.
        var d = e.Display;
        if (d is null) return Fallback(e.Id);
        return new DryWindowPhase(
            Key: e.Id,
            LongTitle: d.Title ?? $"Phase {e.Id}",
            ShortTitle: d.ShortTitle ?? $"Phase {e.Id}",
            Description: d.Description ?? "",
            ChampionVsChallengerLabel: d.Label ?? d.ShortTitle ?? $"Phase {e.Id}",
            Color: d.Color,
            StartHourCurveVersion: d.StartHourCurveVersion);
    }

    private static DryWindowPhase Fallback(string key)
        => new(key, $"Phase {key}", $"Phase {key}", "", $"Phase {key}", FallbackGrey);
}
