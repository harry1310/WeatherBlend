using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Humidity;

public sealed class HumidityBlender : IElementBlender
{
    private readonly ILogger<HumidityBlender> _log;
    private readonly AppConfig _cfg;

    public HumidityBlender(ILogger<HumidityBlender> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public ElementTarget Target => ElementTargets.Humidity;

    public Task<int> TrainAsync(int[] leads, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs(
            Target: Target,
            Hyperparameters: new TempTrainer.Hyperparameters(),
            BuildSpec: lead => HumidityFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
            LoadRowsForSpec: (spec, c) => HumidityFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name, spec, c),
            DeviationsFromBrief: new[]
            {
                "UKMO excluded entirely (4-way bake-off 2026-04-26 — humidity behaves like " +
                "temp/precip: 6-model + restricted-window loses to 5-model + full-window).",
                "MF dropped at 48/72h — Open-Meteo live forecasts cap at ~36h.",
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
            });

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
