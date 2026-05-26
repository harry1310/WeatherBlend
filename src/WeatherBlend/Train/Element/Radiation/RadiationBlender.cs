using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Radiation;

public sealed class RadiationBlender : IElementBlender
{
    private readonly ILogger<RadiationBlender> _log;
    private readonly AppConfig _cfg;

    public RadiationBlender(ILogger<RadiationBlender> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public ElementTarget Target => ElementTargets.ShortwaveRadiation;

    public Task<int> TrainAsync(int[] leads, LocationConfig location, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs(
            Target: Target,
            Hyperparameters: TempTrainer.Hyperparameters.Default(),
            ModelsRoot: _cfg.Storage.ModelsPath,
            BuildSpec: lead => RadiationFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
            LoadRowsForSpec: (spec, c) => RadiationFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, location.Name, spec, c),
            DeviationsFromBrief: new[]
            {
                "UKMO excluded entirely — its ShortwaveRadiation field is essentially " +
                "never populated in either backfill (≥48h: 99.99% null, 24h: 25% null) " +
                "or live collects. Genuinely missing data, not a train-time-poisoning " +
                "artefact, so 5-model is the right choice.",
                "MF dropped at 48/72h — Open-Meteo live forecasts cap at ~36h.",
                "Radiation is bimodal (zero at night). MAE will be dominated by daytime " +
                "errors; report flags this rather than re-weighting samples in v1.",
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
            },
            LocationName: location.Name);

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
