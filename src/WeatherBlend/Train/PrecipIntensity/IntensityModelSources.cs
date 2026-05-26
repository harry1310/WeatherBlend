namespace WeatherBlend.Train.PrecipIntensity;

/// <summary>
/// Structural binding: which precipitation phase each rainfall-amount
/// (intensity) phase samples over.
///
/// This is part of the DEFINITION of each model — a Phase 3f IS
/// "NGBoost-LogNormal over Phase 3a's hourly P(wet)", gated by 3a's
/// marginal at predict time to mix the dry/wet branches of the
/// LogNormal-conditional distribution. It is NOT a tunable and is
/// deliberately NOT resolved from any mutable manifest field —
/// changing the stage-1 source changes what the model fundamentally is.
///
/// Changing an entry here changes what the model fundamentally is, so
/// this map is the single, hard-coded source of truth. The train path
/// resolves the source phase here, then asks
/// <see cref="WeatherBlend.Train.ModelArtifact.ResolveStationPhaseVersion"/>
/// for that phase's current champion version; predict reads the resolved
/// version back from the bundle's <c>training_metadata</c> under
/// <see cref="SourceVersionKey"/>.
///
/// Future siblings would be new phase IDs binding the same way:
///   * "NGBoost-LogNormal over 3c" → new phase, new entry here.
///   * "NGBoost-LogNormal over Membury 4a" → new phase, new entry here.
/// Never reconfigure 3f to point at anything other than 3a.
/// </summary>
public static class IntensityModelSources
{
    private static readonly IReadOnlyDictionary<string, string> _sourcePhase =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["3f"] = "3a",
        };

    /// <summary>
    /// The precipitation phase <paramref name="intensityPhase"/> samples
    /// over. Throws for a phase with no registered source — a new
    /// rainfall-amount phase MUST declare its source here before it can
    /// train.
    /// </summary>
    public static string SourcePhaseFor(string intensityPhase)
        => _sourcePhase.TryGetValue(intensityPhase, out var src)
            ? src
            : throw new ArgumentException(
                $"Rainfall-amount phase '{intensityPhase}' has no registered " +
                "precipitation source phase. Add it to IntensityModelSources.",
                nameof(intensityPhase));

    /// <summary>True iff <paramref name="phase"/> is a rainfall-amount
    /// phase with a registered precipitation source.</summary>
    public static bool IsIntensityPhase(string phase) => _sourcePhase.ContainsKey(phase);

    /// <summary>
    /// The <c>training_metadata.Hyperparameters</c> key the resolved
    /// source version is persisted under — named after the source phase
    /// (e.g. <c>precip_3a_version</c>) so the bundle carries honest
    /// provenance of exactly which version it sampled.
    /// </summary>
    public static string SourceVersionKey(string intensityPhase)
        => $"precip_{SourcePhaseFor(intensityPhase)}_version";

    /// <summary>All rainfall-amount phase IDs known to the project.
    /// Used by predict / verify dispatchers to whitelist phases by
    /// target (mirrors the role <c>ActivePhasePolicy</c> serves for
    /// other targets).</summary>
    public static IReadOnlyCollection<string> AllPhases => _sourcePhase.Keys.ToList();
}
