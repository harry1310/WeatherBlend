namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Structural binding: which precipitation phase each Monte-Carlo dry-window
/// phase samples over. This is part of the DEFINITION of each model — a 3g
/// IS "MC over Phase 3a's hourly P(wet)", a 3j/3n are the copula siblings of
/// 3g over the same 3a input, and a 3s IS "MC over Phase 3e". It is NOT a
/// tunable and is deliberately NOT resolved from any mutable manifest field
/// (this replaces the old <c>StationEntry.Current</c> lookup, which silently
/// re-pointed the MC source whenever another phase promoted itself champion).
///
/// Changing an entry here changes what the model fundamentally is — so this
/// map is the single, hard-coded source of truth. The MC train path resolves
/// the source phase here, then asks
/// <see cref="WeatherBlend.Train.ModelArtifact.ResolveStationChampionVersion"/>
/// for that phase's current champion version; predict reads the resolved
/// version back from the bundle's <c>training_metadata</c> under
/// <see cref="SourceVersionKey"/>.
/// </summary>
public static class DryWindowMcSources
{
    private static readonly IReadOnlyDictionary<string, string> _sourcePhase =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [DryWindow3gPredictor.Phase3g] = "3a",
            [DryWindow3jPredictor.Phase3j] = "3a",
            [DryWindow3nPredictor.Phase3n] = "3a",
            [DryWindow3sPredictor.Phase3s] = "3e",
        };

    /// <summary>
    /// The precipitation phase <paramref name="mcPhase"/> samples over.
    /// Throws for a phase with no registered source — a new MC phase MUST
    /// declare its source here before it can train.
    /// </summary>
    public static string SourcePhaseFor(string mcPhase)
        => _sourcePhase.TryGetValue(mcPhase, out var src)
            ? src
            : throw new ArgumentException(
                $"Dry-window MC phase '{mcPhase}' has no registered precipitation "
                + "source phase. Add it to DryWindowMcSources.", nameof(mcPhase));

    /// <summary>True iff <paramref name="phase"/> is an MC phase with a
    /// registered precipitation source.</summary>
    public static bool IsMcPhase(string phase) => _sourcePhase.ContainsKey(phase);

    /// <summary>
    /// The <c>training_metadata.Hyperparameters</c> key the resolved source
    /// version is persisted under — named after the source phase
    /// (<c>precip_3a_version</c> / <c>precip_3e_version</c>) so the bundle
    /// carries honest provenance of exactly which version it sampled.
    /// </summary>
    public static string SourceVersionKey(string mcPhase)
        => $"precip_{SourcePhaseFor(mcPhase)}_version";
}
