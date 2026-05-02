namespace WeatherBlend.Models;

/// <summary>
/// Single source of truth for "which model phases is the codebase actively
/// shipping right now?". Same allowlist used to:
///
///   * gate which phases get a card on the Models page (and in what order),
///   * filter the rolling-MAE / rolling-Brier skill charts so a retired
///     phase like <c>3a_isotonic</c> doesn't keep appearing as a stale line,
///   * judge whether a verify-history row matches the card it's rendering
///     under (the card's phase must be in the active list).
///
/// Phases NOT in this list (e.g. <c>"2b_redo"</c>, <c>"3a_isotonic"</c>,
/// <c>"3d_shape"</c>, <c>"3d_calibrated"</c>) still have parquet rows on disk
/// because their predict trees aged into the rolling window before being
/// retired. The renderer drops them — they're reference-only.
///
/// Keys are the values that <c>training_metadata.Phase</c> stores
/// (case-sensitive, ordinal compare). Add a phase here when shipping it;
/// remove when retiring. Anywhere in the codebase that asks "is this phase
/// live?" should call <see cref="IsActive"/> rather than re-encoding the rule.
/// </summary>
public static class ActivePhasePolicy
{
    /// <summary>
    /// Champion-first ordering per target. The first entry is the production
    /// champion; subsequent entries are challengers. Renderers use this for
    /// sort order (lean → rich) so the Models page reads top-to-bottom as
    /// "what's promoted, then what's competing". When a target has only one
    /// active phase, the per-phase header is suppressed (sites: dry-window).
    /// </summary>
    public static readonly IReadOnlyDictionary<string, IReadOnlyList<string>> ByTarget =
        new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal)
        {
            ["temperature"]   = new[] { "2b", "2c" },
            ["precipitation"] = new[] { "3a", "3c" },
            // 3e (B2 conditional decomposition for windows 3 + 4) added
            // 2026-05-03 as a challenger to 3b. 6h stays 3b-only because the
            // conditional subset for a 6|4 stage is too sparse to be reliable.
            // 3d-shape was dropped 2026-04-29 (no Brier gain on the daytime label).
            ["dry_window"]    = new[] { "3b", "3e" },
        };

    /// <summary>
    /// True iff the (target, phase) pair is in the shipping lineup. Empty /
    /// null phase strings are never active — callers needing a phase string
    /// should treat unknown phases the same way (drop, don't render).
    /// </summary>
    public static bool IsActive(string target, string? phase)
    {
        if (string.IsNullOrEmpty(phase)) return false;
        return ByTarget.TryGetValue(target, out var allowed)
               && allowed.Contains(phase, StringComparer.Ordinal);
    }

    /// <summary>
    /// Index of <paramref name="phase"/> in <paramref name="target"/>'s
    /// champion-first list — low = champion, high = challenger. Returns
    /// <see cref="int.MaxValue"/> for unknown / inactive phases so callers
    /// can sort with the rest of their data without special-casing nulls.
    /// </summary>
    public static int Priority(string target, string phase)
    {
        if (!ByTarget.TryGetValue(target, out var ordered)) return int.MaxValue;
        for (int i = 0; i < ordered.Count; i++)
            if (string.Equals(ordered[i], phase, StringComparison.Ordinal)) return i;
        return int.MaxValue;
    }
}
