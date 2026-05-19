using Microsoft.Extensions.Logging;
using WeatherBlend.Models;

namespace WeatherBlend.Train;

/// <summary>
/// Expands the model manifests into the flat list of <see cref="Cell"/>s that
/// predict / verify / render iterate over. This is the single place that knows
/// how to walk <c>{target} → station → Active version → lead</c>; the commands
/// downstream filter the result (predict by <c>--location</c>, verify by
/// truth-station, render scans all) instead of each re-deriving the walk.
///
/// Purely manifest-driven: a version is enumerated iff it sits in a station's
/// <c>Active</c> list (the predict/verify contract). phases.yaml's per-phase
/// <c>locations:</c> filter gates *training*, not prediction, so it is
/// deliberately not consulted here — the phase of each cell is read from the
/// version's own <c>training_metadata.json</c>.
/// </summary>
public static class CellEnumerator
{
    /// <summary>
    /// Every <see cref="CellVersion"/> active across <paramref name="targets"/>'
    /// manifests under <paramref name="modelsRoot"/>. A version directory that
    /// is missing on disk, or lacks <c>feature_schema.json</c> /
    /// <c>training_metadata.json</c>, is skipped with a warning rather than
    /// aborting the whole enumeration — render in particular must tolerate a
    /// stale orphan dir left by a failed train.
    /// </summary>
    public static IReadOnlyList<CellVersion> Enumerate(
        string modelsRoot,
        IEnumerable<string> targets,
        ILogger? log = null)
    {
        var cells = new List<CellVersion>();

        foreach (var target in targets)
        {
            foreach (var station in ModelArtifact.ListStations(modelsRoot, target))
            {
                var location = ModelArtifact.ResolveStationLocation(modelsRoot, target, station);
                foreach (var version in ModelArtifact.ResolveStationActive(modelsRoot, target, station))
                {
                    var versionDir = Path.Combine(modelsRoot, target, station, version);
                    if (!Directory.Exists(versionDir))
                    {
                        log?.LogWarning(
                            "CellEnumerator: {Target}/{Station} Active version '{Version}' has no directory on disk — skipping.",
                            target, station, version);
                        continue;
                    }

                    string phase;
                    IReadOnlyList<int> leads;
                    try
                    {
                        phase = ModelArtifact.LoadTrainingMetadata(versionDir).Phase;
                        leads = ModelArtifact.LoadBlenderSpecs(versionDir).Keys.OrderBy(l => l).ToList();
                    }
                    catch (Exception ex) when (ex is FileNotFoundException or InvalidOperationException)
                    {
                        log?.LogWarning(
                            "CellEnumerator: {Target}/{Station}/{Version} is unreadable ({Message}) — skipping.",
                            target, station, version, ex.Message);
                        continue;
                    }

                    foreach (var lead in leads)
                        cells.Add(new CellVersion(
                            new Cell(location, station, target, phase, lead), version));
                }
            }
        }

        return cells;
    }
}
