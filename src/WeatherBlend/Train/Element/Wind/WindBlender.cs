using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// Wind blender — wires <see cref="WindFeatureBuilder"/> into the generic
/// <see cref="ElementTrainerHarness"/>. Discovered by DI and dispatched
/// to from <c>ElementTrainCommand</c> when <c>--target wind</c>.
/// </summary>
public sealed class WindBlender : IElementBlender
{
    private readonly ILogger<WindBlender> _log;
    private readonly AppConfig _cfg;

    public WindBlender(ILogger<WindBlender> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public ElementTarget Target => ElementTargets.Wind;

    public Task<int> TrainAsync(int[] leads, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs<WindRow>(
            Target: Target,
            Hyperparameters: new TemperatureTrainer.Hyperparameters(),
            InternalFeatureColumns: WindFeatureBuilder.InternalFeatureColumns,
            PublicFeatureNames: WindFeatureBuilder.PublicFeatureNames,
            ModelAccessors: WindFeatureBuilder.ModelAccessors,
            TimeOf: r => r.ValidTimeUtc,
            TruthOf: r => r.Era5WindSpeed,
            DeviationsFromBrief: new[]
            {
                "MétéoFrance excluded — Open-Meteo Previous Runs API ships no MF wind " +
                "(100% null on speed and direction at Bonehill, audit 2026-04-25). Wind " +
                "blender is therefore a 5-model ensemble, not 6.",
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
            },
            LoadRowsForLead: (lead, c) => WindFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name, lead, c));

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
