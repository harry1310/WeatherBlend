namespace WeatherBlend.Models;

/// <summary>
/// Static back-compat façade over <see cref="PhaseRegistry.Default"/>.
/// Pre-2026-05-10 this class was the source of truth (a hardcoded
/// <c>ByTarget</c> dict); the canonical list now lives in
/// <c>config/phases.yaml</c> and <see cref="PhaseRegistry"/> loads it.
/// All call sites in the codebase still use <c>ActivePhasePolicy.IsActive</c>
/// / <c>.Priority</c> / <c>.ByTarget</c>, so this façade keeps them
/// untouched while the YAML drives the contents.
///
/// The same allowlist is used to:
///   * gate which phases get a card on the Models page (and in what order),
///   * filter the rolling-MAE / rolling-Brier skill charts so a retired
///     phase like <c>3a_isotonic</c> doesn't keep appearing as a stale line,
///   * judge whether a verify-history row matches the card it's rendering
///     under (the card's phase must be in the active list).
///
/// Phases NOT in the YAML (e.g. <c>"2b_redo"</c>, <c>"3a_isotonic"</c>,
/// <c>"3d_shape"</c>, <c>"3d_calibrated"</c>, <c>"3e"</c>, <c>"3f"</c>) still
/// have parquet rows on disk because their predict trees aged into the rolling
/// window before being retired. The renderer drops them — they're reference-only.
///
/// Confidence-role phases (5a) are also excluded from this façade — they
/// render as a credible-band overlay, not a prediction line. Train workflows
/// reach them via <see cref="PhaseRegistry.AllPhases"/>.
/// </summary>
public static class ActivePhasePolicy
{
    /// <summary>
    /// Champion-first ID lists per target, from <c>phases.yaml</c>.
    /// Confidence-role phases are excluded — see class summary.
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyList<string>> ByTarget
        => PhaseRegistry.Default.ByTarget;

    /// <summary>
    /// True iff the (target, phase) pair is in the shipping lineup AS A
    /// PREDICTION LINE — champion or challenger, not confidence-role.
    /// Empty / null phase strings are never active.
    /// </summary>
    public static bool IsActive(string target, string? phase)
        => PhaseRegistry.Default.IsActive(target, phase);

    /// <summary>
    /// Champion-first phase-ID list for a (target, location) pair — like
    /// <see cref="ByTarget"/> but filtered to phases applicable to
    /// <paramref name="location"/> per each phase's optional phases.yaml
    /// <c>locations:</c> filter. A phase with no filter applies to every
    /// configured location. Phase B (commit 7).
    /// </summary>
    public static IReadOnlyList<string> ByTargetAndLocation(string target, string location)
        => PhaseRegistry.Default.ByTargetAndLocation(target, location);

    /// <summary>
    /// Index of <paramref name="phase"/> in <paramref name="target"/>'s
    /// champion-first list — low = champion, high = challenger. Returns
    /// <see cref="int.MaxValue"/> for unknown / inactive phases so callers
    /// can sort with the rest of their data without special-casing nulls.
    /// </summary>
    public static int Priority(string target, string phase)
        => PhaseRegistry.Default.Priority(target, phase);
}
