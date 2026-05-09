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
/// <c>"3d_shape"</c>, <c>"3d_calibrated"</c>, <c>"3e"</c>, <c>"3f"</c>) still
/// have parquet rows on disk because their predict trees aged into the rolling
/// window before being retired. The renderer drops them — they're reference-only.
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
            // 2d ships as champion at lead 12h only (per-lead champion in
            // TemperatureManifest.ChampionByLead); 2b stays Current for 24/48/
            // 72/96/120 where 2d isn't trained. 2c keeps its challenger slot.
            ["temperature"]   = new[] { "2b", "2c", "2d" },
            // 3d ships as champion at lead 12h only (per-station ChampionByLead
            // in StationEntry); 3a stays Current for 24/48/72/96/120 where 3d
            // isn't trained. 3c keeps its challenger slot. 4a (dbarts BART,
            // trained + predicted in WeatherProbabilistic, written to the
            // same predictions tree) added 2026-05-09 as a cross-lead
            // challenger; beats deployed 3a in all 9 (station × lead) cells
            // tested (avg −5.7% Brier).
            ["precipitation"] = new[] { "3a", "3c", "3d", "4a" },
            // 3g (parameter-free MC over Phase 3a hourly P(wet) marginals)
            // is the 3b challenger; cross-window monotonicity P(N=3) ≥ P(N=4)
            // ≥ P(N=6) holds by construction. 3d-shape (no Brier gain), 3e
            // (B2 cascade — 3g supersedes structurally for all windows, not
            // just the 3h↔4h pair), and 3f (62-feature failed attempt) were
            // all retired 2026-05-04 along with their training paths.
            ["dry_window"]    = new[] { "3b", "3g" },
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
