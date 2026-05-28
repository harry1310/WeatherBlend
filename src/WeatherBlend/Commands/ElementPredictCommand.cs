using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Common;
using WeatherBlend.Train.Element.Gust;
using WeatherBlend.Train.Element.Humidity;
using WeatherBlend.Train.Element.Radiation;
using WeatherBlend.Train.Element.Wind;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended forecasts for one of the per-variable element targets
/// (wind / humidity / shortwave-radiation / cloud-cover) at the standard
/// {24, 48, 72}h horizons. Champion/challenger aware: when the user passes
/// <c>--model-version current</c> (the default) the command iterates every
/// version in the active location's <c>Active</c> list and emits a
/// parquet per version.
///
/// Per-element pipelines do their own SQL pivot + row composition (each
/// element pulls a different forecast column set) and share parquet writing
/// via <see cref="WeatherBlend.Predict.PredictionParquetWriter"/>. Output goes to
/// <c>data/predictions/{element}/model_version=&lt;v&gt;/date=&lt;yyyy-MM-dd&gt;/predictions.parquet</c>.
/// </summary>
public sealed class ElementPredictCommand
{
    private readonly ILogger<ElementPredictCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = Leads.Short;

    /// <summary>
    /// Phases this command serves. Anything else in a MANIFEST.Active list
    /// under one of the element ModelDirName trees is silently skipped —
    /// the version was promoted by some other minter (e.g.
    /// <see cref="WindBlendPredictCommand"/> for <c>wind_blend</c>) and
    /// has its own dedicated CLI route. Mirrors
    /// <c>PrecipPredictCommand.HandledPrecipPhases</c> (L2 gate, see the
    /// comment block there for the project-wide 2-layer gate rationale).
    ///
    /// Adding a new phase requires:
    ///   1. Update phases.yaml (the L1 gate)
    ///   2. Add the phase tag here
    ///   3. Add a switch case in the dispatch below pointing at the
    ///      pipeline that owns the row build.
    /// </summary>
    private static readonly HashSet<string> HandledElementPhases =
        new(StringComparer.Ordinal)
        {
            "wind", "humidity", "shortwave_radiation", "cloud_cover",
            "wind_gust_lgb", "wind_speed_lgb",
        };

    public ElementPredictCommand(ILogger<ElementPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(ElementTarget target, string modelVersion, DateOnly? forDate, CancellationToken ct)
        => RunAsync(target, modelVersion, forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(ElementTarget target, string modelVersion, DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        var (location, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (location is null) return locRc;

        var modelsRoot = _cfg.Storage.ModelsPath;
        var versions = ResolveRequestedVersions(
            modelsRoot, target.ModelDirName, location.Name, modelVersion);
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
                versionDir = ModelArtifact.ResolveStationVersionDir(
                    modelsRoot, target.ModelDirName, location.Name, version);
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

            // Phase A multi-location safety: metadata.LocationName is
            // [JsonRequired] so a missing field already threw at deserialise.
            if (!string.Equals(metadata.LocationName, location.Name, StringComparison.OrdinalIgnoreCase))
            {
                _log.LogError(
                    "{Target} bundle {V} was trained on location '{Trained}' but predict is using NWP from '{Active}' — refusing to score.",
                    target.CliName, version, metadata.LocationName, location.Name);
                failures++;
                continue;
            }

            var phase = (metadata.Phase ?? "").ToLowerInvariant();
            if (!HandledElementPhases.Contains(phase))
            {
                _log.LogInformation(
                    "Version {V} has phase={Phase}, not in HandledElementPhases — silently skipping " +
                    "(another command owns its predict path; e.g. wind_blend → WindBlendPredictCommand).",
                    version, phase);
                continue;
            }
            _log.LogInformation("--- Version {V} (phase={Phase}) ---", version, phase);

            // Dispatch by PHASE rather than target.CliName so sibling phases
            // sharing a ModelDirName (e.g. `wind` ERA5-truth + `wind_speed_lgb`
            // Dunkeswell-truth, both under data/models/wind/) route to their
            // own pipeline. Phase is stamped on training_metadata.json by
            // each trainer, so this lookup is exact: each Active version
            // identifies its own predict path. Fallback to target.CliName
            // covers legacy bundles where Phase wasn't populated.
            List<ElementPredictionRow> predictions = (phase, target.CliName) switch
            {
                ("wind_speed_lgb", _) => WindSpeedLgbPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                ("wind_gust_lgb", _) => WindGustPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                (_, "wind") => WindPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                (_, "humidity") => HumidityPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                (_, "shortwave-radiation") => RadiationPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                (_, "cloud-cover") => CloudPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                (_, "wind-gust") => WindGustPredictPipeline.PredictForCycle(
                    _log, location.Name, _cfg.Storage.ForecastsPath,
                    versionDir, metadata.Version, anchor, predictionMadeAt, DefaultLeads, ct),
                _ => throw new InvalidOperationException(
                    $"Unknown element target/phase combination: target={target.CliName} phase={phase}"),
            };

            if (predictions.Count == 0)
            {
                _log.LogWarning("Version {V} produced no predictions for {Target}.", version, target.CliName);
                failures++;
                continue;
            }

            var dateStr = anchor.ToString("yyyy-MM-dd");
            var outPath = Path.Combine(
                _cfg.Storage.PredictionsPath, target.ModelDirName,
                $"model_version={metadata.Version}", $"date={dateStr}",
                "predictions.parquet");
            var total = await PredictionParquetWriter.WriteAsync(
                outPath, predictions,
                dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
                freshness: r => r.PredictionMadeAtUtc,
                orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours),
                ct);
            _log.LogInformation("  Wrote {New} new predictions (file now holds {Total}) → {Path}",
                predictions.Count, total, outPath);
            anyOutput = true;
        }

        if (!anyOutput)
        {
            _log.LogError("No versions produced predictions for {Target} — forecast tree likely missing recent data. Run `collect` first.", target.CliName);
            return 3;
        }
        return failures == 0 ? 0 : 4;
    }

    private List<string> ResolveRequestedVersions(
        string modelsRoot, string modelDirName, string station, string modelVersion)
    {
        var v = modelVersion?.ToLowerInvariant() ?? "current";
        if (v is "current" or "all")
            return ModelArtifact.ResolveStationActive(modelsRoot, modelDirName, station).ToList();
        return new List<string> { modelVersion! };
    }
}
