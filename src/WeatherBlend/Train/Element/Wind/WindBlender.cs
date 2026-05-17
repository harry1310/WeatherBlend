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

    public Task<int> TrainAsync(int[] leads, LocationConfig location, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs(
            Target: Target,
            // Wind opts OUT of bagging. The 4-way bake-off 2026-04-26 showed bagging
            // marginally HURTS wind at every lead (-0.16/-0.55/-0.42% vs no-bag), unlike
            // every other target where bagging helps by +0.5-2%. UKMO is restored for
            // wind (only target where the 6-model wins on the same fixed test set).
            Hyperparameters: new TempTrainer.Hyperparameters(
                SubsampleFraction: 1.0, SubsampleFrequency: 0, FeatureFraction: 1.0),
            ModelsRoot: _cfg.Storage.ModelsPath,
            BuildSpec: lead => WindFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
            LoadRowsForSpec: (spec, c) => WindFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, location.Name, spec, c),
            DeviationsFromBrief: new[]
            {
                "MétéoFrance excluded — Open-Meteo Previous Runs API ships no MF wind " +
                "(100% null on speed and direction at Bonehill, audit 2026-04-25). Wind " +
                "blender is therefore a 5-model ensemble, not 6. UKMO sits in optionalModels — " +
                "rows survive UKMO-silent hours but contribute when UKMO is present.",
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
                "Bagging deliberately disabled (1.0/0/1.0). 4-way bake-off showed bagging " +
                "marginally hurts wind at every lead, unlike every other target. Wind keeps " +
                "the deterministic ML.NET defaults.",
            },
            LocationName: location.Name);

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
