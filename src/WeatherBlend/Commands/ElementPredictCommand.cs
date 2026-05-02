using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Common;
using WeatherBlend.Train.Element.Humidity;
using WeatherBlend.Train.Element.Radiation;
using WeatherBlend.Train.Element.Wind;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended forecasts for one of the per-variable element targets
/// (wind / humidity / shortwave-radiation / cloud-cover) at the standard
/// {24, 48, 72}h horizons. Champion/challenger aware: when the user passes
/// <c>--model-version current</c> (the default) the command iterates every
/// version in <c>Manifest.Active</c> for the chosen element and emits a
/// parquet per version.
///
/// Per-element pipelines do their own SQL pivot + row composition (each
/// element pulls a different forecast column set) and share parquet writing
/// via <see cref="ElementPredictWriter"/>. Output goes to
/// <c>data/predictions/{element}/model_version=&lt;v&gt;/date=&lt;yyyy-MM-dd&gt;/predictions.parquet</c>.
/// </summary>
public sealed class ElementPredictCommand
{
    private readonly ILogger<ElementPredictCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = Leads.Short;

    public ElementPredictCommand(ILogger<ElementPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(ElementTarget target, string modelVersion, DateOnly? forDate, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versions = ResolveRequestedVersions(modelsRoot, target.ModelDirName, modelVersion);
        if (versions.Count == 0)
        {
            _log.LogError("No versions to predict for {Target} — manifest has no Active list.", target.CliName);
            return 2;
        }

        _log.LogInformation("Predicting {Target} with versions: [{Versions}]",
            target.CliName, string.Join(", ", versions));

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        _log.LogInformation("Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — leads {Leads}h",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live", string.Join("/", DefaultLeads));

        var anyOutput = false;
        var failures = 0;
        foreach (var version in versions)
        {
            ct.ThrowIfCancellationRequested();
            string versionDir;
            ModelArtifact.TrainingMetadata metadata;
            try
            {
                versionDir = ModelArtifact.ResolveVersionDir(modelsRoot, target.ModelDirName, version);
                metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
            }
            catch (Exception ex)
            {
                _log.LogError("Cannot load {Target}/{V}: {Msg}", target.CliName, version, ex.Message);
                failures++;
                continue;
            }
            if (metadata.PerLead.Count == 0)
            {
                _log.LogError("Version {V} has no per-lead blenders — skipping.", version);
                failures++;
                continue;
            }

            var phase = (metadata.Phase ?? "").ToLowerInvariant();
            _log.LogInformation("--- Version {V} (phase={Phase}) ---", version, phase);

            // Dispatch by target — each pipeline owns its own SQL + ComposeRow.
            List<ElementPredictionRow> predictions = target.CliName switch
            {
                "wind" => WindPredictPipeline.PredictForCycle(
                    _log, _cfg.Location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                "humidity" => HumidityPredictPipeline.PredictForCycle(
                    _log, _cfg.Location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                "shortwave-radiation" => RadiationPredictPipeline.PredictForCycle(
                    _log, _cfg.Location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                "cloud-cover" => CloudPredictPipeline.PredictForCycle(
                    _log, _cfg.Location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                _ => throw new InvalidOperationException($"Unknown element target {target.CliName}"),
            };

            if (predictions.Count == 0)
            {
                _log.LogWarning("Version {V} produced no predictions for {Target}.", version, target.CliName);
                failures++;
                continue;
            }

            await ElementPredictWriter.WriteAsync(
                _log, _cfg.Storage.PredictionsPath, target.ModelDirName,
                metadata.Version, anchor, predictions, ct);
            anyOutput = true;
        }

        if (!anyOutput)
        {
            _log.LogError("No versions produced predictions for {Target} — forecast tree likely missing recent data. Run `collect` first.", target.CliName);
            return 3;
        }
        return failures == 0 ? 0 : 4;
    }

    private List<string> ResolveRequestedVersions(string modelsRoot, string modelDirName, string modelVersion)
    {
        var v = modelVersion?.ToLowerInvariant() ?? "current";
        if (v is "current" or "all")
            return ModelArtifact.ResolveActive(modelsRoot, modelDirName).ToList();
        return new List<string> { modelVersion! };
    }
}
