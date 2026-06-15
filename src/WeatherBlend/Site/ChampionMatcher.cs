namespace WeatherBlend.Site;

/// <summary>
/// Champion matching for the overview pages — "does this prediction row
/// belong on the headline card?". One type so <c>RenderIndex</c>, the day
/// sub-nav's <c>CountHomeDayTiles</c>, and the P(wet) tile lookup can't
/// drift apart (they were three verbatim closures before 2026-06-10).
///
/// Champion-PHASE filter. Empty CurrentVersion = no manifest, fall back to
/// "any" so a freshly-deployed environment still renders cards (that gate
/// lives at the call sites — see <c>RenderIndex</c>).
///
/// Match by PHASE rather than strict version (2026-05-26): a retrain mints a
/// new champion version whose predictions only target the anchor day forward,
/// leaving today's window orphaned for any phase without a sub-24h lead. The
/// PREVIOUS champion-phase bundle still has predictions covering today's hours
/// — those rows live in input.Predictions thanks to the unfiltered scan in
/// PredictionsRepository — they just no longer match the strict-equality
/// champion-version check. Falling back to version-equality when the phase
/// metadata is missing keeps test fixtures (which don't always populate
/// PhaseByVersion) working unchanged.
///
/// Per-lead champion overrides (the old <c>ChampionByLead</c> dimension) were
/// removed 2026-06-15: 2d/3d no longer pin the lead-12 bucket — the per-lead
/// LEAD_POLICY owns the &lt;24hr bucket inside the 2c/3o/3c champion phases now,
/// so there is one champion version per (target, station), full stop.
/// </summary>
internal sealed class ChampionMatcher
{
    private readonly string _currentVersion;
    private readonly IReadOnlyDictionary<string, string> _phaseByVersion;

    public ChampionMatcher(SitePages.SiteInputs input)
        : this(input.CurrentVersion, input.PhaseByVersion)
    {
    }

    public ChampionMatcher(
        string currentVersion,
        IReadOnlyDictionary<string, string> phaseByVersion)
    {
        _currentVersion = currentVersion;
        _phaseByVersion = phaseByVersion;
    }

    /// <summary>Does <paramref name="rowVersion"/> share the champion's phase
    /// (or exact version)?</summary>
    public bool MatchesChampionPhase(string rowVersion)
        => SamePhaseOrVersion(rowVersion, _currentVersion);

    /// <summary>
    /// Fixed-champion-version match — same phase-or-version rule but against a
    /// caller-resolved champion (e.g. the per-station precip champion behind the
    /// P(wet) tiles). An empty / null champion never matches: "no champion
    /// resolved" means "no headline rows", not "match anything".
    /// </summary>
    public bool MatchesChampionVersion(string rowVersion, string? championVersion)
    {
        if (string.IsNullOrEmpty(championVersion)) return false;
        return SamePhaseOrVersion(rowVersion, championVersion);
    }

    private bool SamePhaseOrVersion(string rowVersion, string champVersion)
    {
        if (string.Equals(rowVersion, champVersion, StringComparison.Ordinal)) return true;
        return _phaseByVersion.TryGetValue(rowVersion, out var rowPhase)
            && _phaseByVersion.TryGetValue(champVersion, out var champPhase)
            && !string.IsNullOrEmpty(rowPhase)
            && string.Equals(rowPhase, champPhase, StringComparison.Ordinal);
    }
}
