namespace WeatherBlend.Train.Element;

/// <summary>
/// One per-target blender — owns its row class, feature builder, SQL, and
/// per-model layout (e.g. wind excludes MétéoFrance because Open-Meteo has no
/// MF wind data). The shared training orchestration lives in
/// <see cref="Common.ElementTrainerHarness"/>; this interface is the dispatch
/// boundary used by <c>ElementTrainCommand</c>.
/// </summary>
public interface IElementBlender
{
    ElementTarget Target { get; }

    Task<int> TrainAsync(int[] leads, CancellationToken ct);
}
