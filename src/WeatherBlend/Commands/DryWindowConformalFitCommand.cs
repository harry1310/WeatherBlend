using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// One-shot back-fit of conformal calibrators for every Active dry-window
/// version. For each (station, window, lead, version) cell:
///   1. Rebuild the chronological train/val/test split with the same
///      DryWindowFeatureBuilder + DryWindowDataset.Split rules used at
///      training time. Same val rows the model "would have seen" if
///      conformal had been part of the original training step.
///   2. Get per-row P(wet) on the val slice — 3b loads model.zip, runs
///      inference, optionally applies isotonic. 3p has no conformal fit
///      path (MC over a separate model has no val-slice predictor).
///   3. Fit ConformalCalibrator on (val_pred, val_label) at α = 0.10
///      (90% coverage), persist to versionDir/conformal_calibrator_{L}h.json.
///
/// Doesn't touch training metadata or the model artefacts themselves;
/// rerunning is idempotent (overwrites the conformal JSONs in place).
/// Live predict picks the new calibrators up automatically because
/// DryWindowPredictCommand looks up conformal_calibrator_{L}h.json next
/// to the model.
/// </summary>
public sealed class DryWindowConformalFitCommand
{
    /// <summary>Default α — 90% target coverage. Used by the train-time
    /// auto-refit hook so a calibrator is always present at the same
    /// coverage level as the back-fill command.</summary>
    public const double DefaultAlpha = 0.10;

    private readonly ILogger<DryWindowConformalFitCommand> _log;
    private readonly AppConfig _cfg;
    private readonly ModelMetadataRepository _metadata;

    public DryWindowConformalFitCommand(
        ILogger<DryWindowConformalFitCommand> log, AppConfig cfg, ModelMetadataRepository metadata)
    {
        _log = log;
        _cfg = cfg;
        _metadata = metadata;
    }

    public async Task<int> RunAsync(double alpha, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var manifest = _metadata.TryGetManifest("dry_window");
        if (manifest?.Stations is null || manifest.Stations.Count == 0)
        {
            _log.LogError("No dry-window manifest at {P}", modelsRoot);
            return 2;
        }

        int fitted = 0, skipped = 0;
        foreach (var (compositeKey, entry) in manifest.Stations)
        {
            ct.ThrowIfCancellationRequested();
            foreach (var versionName in entry.Active)
            {
                var (f, s) = await FitOneAsync(compositeKey, versionName, alpha, ct);
                fitted += f;
                skipped += s;
            }
        }

        _log.LogInformation("Conformal fit complete. Fitted={F} Skipped={S}", fitted, skipped);
        return fitted == 0 ? 3 : 0;
    }

    /// <summary>
    /// Fit conformal calibrators for every lead of one specific (composite,
    /// version) cell. Used by the train-time auto-refit hook (called from
    /// DryWindowTrainCommand right after PromoteStationVersionAs*) so a fresh
    /// champion or challenger never ships without a calibrator.
    ///
    /// Returns (fitted, skipped) counts so callers can log a single summary
    /// line without re-deriving the metadata.
    /// </summary>
    public Task<(int Fitted, int Skipped)> FitOneAsync(
        string compositeKey, string versionName, double alpha, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        int fitted = 0, skipped = 0;

        var parsed = DryWindowPredictCommand.ParseCompositeKey(compositeKey);
        if (parsed is null)
        {
            _log.LogWarning("Skipping unparsable composite key {Key}", compositeKey);
            return Task.FromResult((fitted, skipped + 1));
        }
        var (stationSlug, windowHours) = parsed.Value;
        var stationName = _cfg.ResolveStationName(stationSlug);
        if (stationName is null)
        {
            _log.LogWarning("{Key}: cannot resolve station name from slug; skipping", compositeKey);
            return Task.FromResult((fitted, skipped + 1));
        }

        var versionDir = System.IO.Path.Combine(modelsRoot, "dry_window", compositeKey, versionName);
        if (!System.IO.Directory.Exists(versionDir))
        {
            _log.LogWarning("{Key} {V}: version dir missing; skipping", compositeKey, versionName);
            return Task.FromResult((fitted, skipped + 1));
        }
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        _log.LogInformation("Conformal fit: {Key} / {V} (phase {P})",
            compositeKey, versionName, metadata.Phase);

        foreach (var lead in Leads.Short)
        {
            ct.ThrowIfCancellationRequested();
            if (!metadata.PerLead.ContainsKey(lead.ToString()))
            {
                _log.LogWarning("  lead {L}h: not in this version's metadata; skipping", lead);
                continue;
            }

            // Same row build + chronological split as training.
            var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
            var rows = DryWindowFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                _cfg.Location.Name, stationName, spec, windowHours, daytime,
                minValidTime: null, ct);
            if (rows.Count < 100)
            {
                _log.LogWarning("  lead {L}h: only {N} rows; skipping", lead, rows.Count);
                skipped++;
                continue;
            }
            var ds = DryWindowDataset.Split(rows);

            // Get per-val P(wet) for this version. Only 3b has a conformal
            // fit path (LightGBM + optional isotonic); MC challengers have
            // no val-slice predictor that can replay deterministically.
            var (probs, labels) = Score3bOnVal(versionDir, spec, lead, ds, stationName);

            if (probs.Count < 30)
            {
                _log.LogWarning("  lead {L}h: only {N} val rows after filter; skipping conformal fit",
                    lead, probs.Count);
                skipped++;
                continue;
            }

            var cal = ConformalCalibrator.Fit(probs, labels, alpha);
            ModelArtifact.SaveLeadConformalCalibrator(cal, versionDir, lead);
            _log.LogInformation("  lead {L}h: {Cal} ({N} val rows, α={Alpha:0.00})",
                lead, cal, probs.Count, alpha);
            fitted++;
        }

        return Task.FromResult((fitted, skipped));
    }

    /// 3b path: load LightGBM, predict val rows, apply isotonic if present.
    private static (List<double> Probs, List<bool> Labels) Score3bOnVal(
        string versionDir, BlenderSpec spec, int lead,
        DryWindowDataset ds, string stationName)
    {
        var ml = new MLContext(seed: 42);
        var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
        var raw = DryWindowTrainer.PredictVectorProbability(ml, model, spec, ds.Val);
        var iso = ModelArtifact.TryLoadLeadCalibrator(versionDir, lead);
        var shipped = iso is null ? raw : iso.PredictMany(raw);
        return (shipped.ToList(), ds.Val.Select(r => r.Label).ToList());
    }


}
