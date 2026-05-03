using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// One-shot back-fit of conformal calibrators for every Active precipitation
/// (3a + 3c) version. Same shape as the dry-window equivalent — sibling.
///
/// For each (station, lead, version) cell:
///   1. Rebuild the chronological train/val/test split with the same
///      PrecipFeatureBuilder + BinaryDataset.Split rules used at training.
///   2. Score val rows via the saved LightGBM (+ isotonic if any).
///   3. Fit ConformalCalibrator on (val_pred, val_label) at α (default 0.10
///      = 90% coverage), persist as conformal_calibrator_{L}h.json next to
///      the model.zip.
///
/// Live PrecipPredictCommand picks the new calibrators up automatically the
/// next cycle. Idempotent on rerun. No retraining, no impact on the model
/// artefacts themselves.
///
/// Note: 3a is hourly (24 predictions per station per anchor day) so the
/// val slice contains thousands of rows — much more calibration data than
/// the daily dry-window blenders had. τ should be tighter as a result.
/// </summary>
public sealed class PrecipConformalFitCommand
{
    private readonly ILogger<PrecipConformalFitCommand> _log;
    private readonly AppConfig _cfg;
    private readonly ModelMetadataRepository _metadata;

    public PrecipConformalFitCommand(
        ILogger<PrecipConformalFitCommand> log, AppConfig cfg, ModelMetadataRepository metadata)
    {
        _log = log;
        _cfg = cfg;
        _metadata = metadata;
    }

    public async Task<int> RunAsync(double alpha, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var manifest = _metadata.TryGetManifest("precipitation");
        if (manifest?.Stations is null || manifest.Stations.Count == 0)
        {
            _log.LogError("No precipitation manifest at {P}", modelsRoot);
            return 2;
        }

        int fitted = 0, skipped = 0;
        foreach (var (stationSlug, entry) in manifest.Stations)
        {
            ct.ThrowIfCancellationRequested();
            var stationName = ResolveStationName(stationSlug);
            if (stationName is null)
            {
                _log.LogWarning("{S}: cannot resolve station name from slug; skipping", stationSlug);
                skipped++;
                continue;
            }

            foreach (var versionName in entry.Active)
            {
                ct.ThrowIfCancellationRequested();
                var versionDir = System.IO.Path.Combine(modelsRoot, "precipitation", stationSlug, versionName);
                if (!System.IO.Directory.Exists(versionDir))
                {
                    _log.LogWarning("{S} {V}: version dir missing; skipping", stationSlug, versionName);
                    skipped++;
                    continue;
                }
                var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
                _log.LogInformation("=== {S} / {V} (phase {P}) ===",
                    stationSlug, versionName, metadata.Phase);

                // Each lead trains its own model+calibrator. PerLead keys are
                // string-form lead-hours; iterate them rather than guessing.
                foreach (var leadKey in metadata.PerLead.Keys)
                {
                    if (!int.TryParse(leadKey, out var lead)) continue;
                    var spec = ModelArtifact.LoadBlenderSpecs(versionDir).GetValueOrDefault(lead);
                    if (spec is null)
                    {
                        _log.LogWarning("  lead {L}h: no BlenderSpec; skipping", lead);
                        continue;
                    }

                    // Same row build + chronological split as the original
                    // training. 3a uses the lean PrecipFeatureBuilder (23
                    // features); 3c uses PrecipRichFeatureBuilder (59
                    // features). Dispatch on the saved metadata.Phase so
                    // we reproduce the exact training-time row vectors —
                    // the feature builder MUST match the spec the model
                    // was fit against, otherwise the schema mismatch trips
                    // PrecipFeatureBuilder.ComposeRow's pack-length check.
                    var rows = PrecipPhases.IsRich(metadata.Phase)
                        ? PrecipRichFeatureBuilder.BuildForLead(
                            _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                            _cfg.Location.Name, stationName, spec, ct)
                        : PrecipFeatureBuilder.BuildForLead(
                            _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                            _cfg.Location.Name, stationName, spec, ct);
                    if (rows.Count < 500)
                    {
                        _log.LogWarning("  lead {L}h: only {N} rows; skipping", lead, rows.Count);
                        skipped++;
                        continue;
                    }
                    var ds = BinaryDataset.Split(rows);

                    // Predict val. Apply isotonic if present (3a_isotonic was
                    // retired so this is currently always null in production,
                    // but keep the fallback for symmetry with predict-time).
                    var ml = new MLContext(seed: 42);
                    var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
                    var raw = PrecipOccurrenceTrainer.PredictVectorProbability(ml, model, spec, ds.Val);
                    var iso = ModelArtifact.TryLoadLeadCalibrator(versionDir, lead);
                    var shipped = iso is null ? raw : iso.PredictMany(raw);

                    var probs = shipped.ToList();
                    var labels = ds.Val.Select(r => r.Label).ToList();
                    if (probs.Count < 30)
                    {
                        _log.LogWarning("  lead {L}h: only {N} val rows; skipping", lead, probs.Count);
                        skipped++;
                        continue;
                    }

                    var cal = ConformalCalibrator.Fit(probs, labels, alpha);
                    ModelArtifact.SaveLeadConformalCalibrator(cal, versionDir, lead);
                    _log.LogInformation("  lead {L}h: {Cal} ({N} val rows, α={Alpha:0.00})",
                        lead, cal, probs.Count, alpha);
                    fitted++;
                }
            }
        }

        _log.LogInformation("Precip conformal fit complete. Fitted={F} Skipped={S}", fitted, skipped);
        await Task.CompletedTask;
        return fitted == 0 ? 3 : 0;
    }

    private string? ResolveStationName(string slug)
    {
        foreach (var s in _cfg.Location.Rainfall.Stations)
            if (StationSlug.WithEaPrefix(s.Name).Equals(slug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        return null;
    }
}
