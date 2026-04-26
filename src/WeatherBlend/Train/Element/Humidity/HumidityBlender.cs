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
        var inputs = new ElementTrainerHarness.ElementTrainerInputs<HumidityRow>(
            Target: Target,
            Hyperparameters: new TemperatureTrainer.Hyperparameters(),
            InternalFeatureColumns: HumidityFeatureBuilder.InternalFeatureColumns,
            PublicFeatureNames: HumidityFeatureBuilder.PublicFeatureNames,
            ModelAccessors: HumidityFeatureBuilder.ModelAccessors,
            TimeOf: r => r.ValidTimeUtc,
            TruthOf: r => r.Era5Rh,
            DeviationsFromBrief: new[]
            {
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
            },
            LoadRowsForLead: (lead, c) => HumidityFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name, lead, c));

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
