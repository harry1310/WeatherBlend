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

    public Task<int> TrainAsync(int[] leads, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs<RadiationRow>(
            Target: Target,
            Hyperparameters: new TemperatureTrainer.Hyperparameters(),
            InternalFeatureColumns: RadiationFeatureBuilder.InternalFeatureColumns,
            PublicFeatureNames: RadiationFeatureBuilder.PublicFeatureNames,
            ModelAccessors: RadiationFeatureBuilder.ModelAccessors,
            TimeOf: r => r.ValidTimeUtc,
            TruthOf: r => r.Era5Sw,
            DeviationsFromBrief: new[]
            {
                "UKMO radiation is included as an optional model — present at lead 24h " +
                "(~74% coverage) but always-null at lead 48/72h (audit 2026-04-25). " +
                "Rows where UKMO is missing carry NaN; LightGBM's native missing-value " +
                "handling silently down-weights it at long lead.",
                "Radiation is bimodal (zero at night). MAE will be dominated by daytime " +
                "errors; report flags this rather than re-weighting samples in v1.",
                "Objective is L2; MAE used only as early-stopping metric (Microsoft.ML.LightGbm 4.0 limit).",
                "No monotone constraints (same Microsoft.ML.LightGbm 4.0 limit).",
            },
            LoadRowsForLead: (lead, c) => RadiationFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name, lead, c));

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
