using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// One precipitation training phase. Holds the labels and chart palette
/// colour shared by every page that buckets precipitation predictions by
/// phase, so the same string literal can't drift out of sync between the
/// SitePages render methods and the PrecipPredictCommand metadata check.
///
/// Since 2026-06-10 the per-phase values live in phases.yaml's
/// <c>display:</c> blocks; <see cref="PrecipPhases"/> materialises this
/// record from <see cref="PhaseRegistry"/>, so adding a phase is a registry
/// edit, not a code edit.
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
/// The precipitation phase buckets and helpers for mapping a prediction
/// row's metadata phase onto one of them. Buckets are ordered for rendering:
/// callers can iterate <see cref="All"/> directly — render order = registry
/// order (champion-first, as phases.yaml declares them). Unknown tags bucket
/// to <c>null</c> and are silently skipped.
/// </summary>
public static class PrecipPhases
{
    private const string Target = "precipitation";
    private const string FallbackGrey = "#9e9e9e";

    /// <summary>Light shade of Phase 4a's amber for the q05/q95 credible
    /// interval bracket lines on the BART panel — saturation difference
    /// reads as "edges of the same uncertainty band". Material amber-200.
    /// Stays a code constant: it's per-panel styling derived from 4a's
    /// colour, not phase identity.</summary>
    public const string Phase4aBand = "#ffcc80";

    /// <summary>
    /// Canonical render order = phases.yaml order. Materialised once — the
    /// registry is an immutable process-wide singleton.
    /// </summary>
    public static readonly IReadOnlyList<PrecipPhase> All =
        PhaseRegistry.Default.AllPhases(Target).Select(ToPhase).ToList();

    /// <summary>Phases that participate in the +24h overlay — currently all.</summary>
    public static readonly IReadOnlyList<PrecipPhase> Comparable = All;

    /// <summary>Phase 4a's card — the BART panel references it directly
    /// (heading + credible-interval band styling).</summary>
    public static PrecipPhase Phase4a => ByKey("4a");

    /// <summary>
    /// Look up a phase colour by key string. Used by render paths that
    /// receive a phase string directly (e.g. rolling Brier rows from the
    /// verify pipeline) rather than a typed <see cref="PrecipPhase"/>
    /// reference. Returns a neutral grey when the phase isn't an active
    /// registry phase — silently keeps a stale row visible without
    /// misleading the eye into matching it to a known phase.
    /// </summary>
    public static string ColorFor(string phaseKey)
    {
        foreach (var p in All)
            if (string.Equals(phaseKey, p.Key, StringComparison.Ordinal)) return p.Color;
        return FallbackGrey;
    }

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
        => string.Equals(metadataPhase, "3c", StringComparison.OrdinalIgnoreCase);

    private static PrecipPhase ByKey(string key)
    {
        foreach (var p in All)
            if (string.Equals(p.Key, key, StringComparison.Ordinal)) return p;
        // Defensive: only reachable if the phase is removed from phases.yaml
        // while a render path still references it by name — render grey /
        // generically-titled rather than crashing the whole site build.
        return Fallback(key);
    }

    private static PrecipPhase ToPhase(PhaseEntry e)
    {
        // Display is enforced by PhaseWiringConsistencyTests for every
        // registry phase; the fallback keeps rendering alive (grey/untitled)
        // if a hand-edited yaml ever drops it.
        var d = e.Display;
        if (d is null) return Fallback(e.Id);
        return new PrecipPhase(
            Key: e.Id,
            LongTitle: d.Title ?? $"Phase {e.Id}",
            ShortTitle: d.ShortTitle ?? $"Phase {e.Id}",
            Description: d.Description ?? "",
            ChampionVsChallengerLabel: d.Label ?? d.ShortTitle ?? $"Phase {e.Id}",
            Color: d.Color);
    }

    private static PrecipPhase Fallback(string key)
        => new(key, $"Phase {key}", $"Phase {key}", "", $"Phase {key}", FallbackGrey);
}
