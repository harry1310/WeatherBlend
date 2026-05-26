using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Cloud;

public sealed class CloudBlender : IElementBlender
{
    private readonly ILogger<CloudBlender> _log;
    private readonly AppConfig _cfg;

    public CloudBlender(ILogger<CloudBlender> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public ElementTarget Target => ElementTargets.CloudCover;

    public Task<int> TrainAsync(int[] leads, LocationConfig location, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs(
            Target: Target,
            Hyperparameters: TempTrainer.Hyperparameters.Default(),
            ModelsRoot: _cfg.Storage.ModelsPath,
            BuildSpec: lead => CloudFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
            LoadRowsForSpec: (spec, c) => CloudFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, location.Name, spec, c),
            DeviationsFromBrief: new[]
            {
                "Layered cloud (low/mid/high) features omitted — Open-Meteo Previous Runs " +
                "returns 100% null for those columns across all models and ERA5 (audit " +
                "2026-04-25). Lean shape is total cloud cover only — N per-model + 3 spread + 4 calendar.",
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
            },
            LocationName: location.Name);

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
