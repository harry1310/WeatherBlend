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
/// version (3b champions + 3g challengers). For each (station, window,
/// lead, version) cell:
///   1. Rebuild the chronological train/val/test split with the same
///      DryWindowFeatureBuilder + DryWindowDataset.Split rules used at
///      training / 3g-scoring time. Same val rows the model "would have
///      seen" if conformal had been part of the original training step.
///   2. Get per-row P(wet) on the val slice from the appropriate predictor:
///        * 3b: load model.zip, run inference, optionally apply isotonic.
///        * 3g: replay the same MC the 3g version was trained against
///              (requires the 3a champion bound at metadata.precip_3a_version).
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
    private const int DefaultMcSamples = DryWindow3gPredictor.DefaultMcSamples;

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
        var stationName = ResolveStationName(stationSlug);
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
                _cfg.Location.Name, stationName, spec, windowHours, daytime, ct);
            if (rows.Count < 100)
            {
                _log.LogWarning("  lead {L}h: only {N} rows; skipping", lead, rows.Count);
                skipped++;
                continue;
            }
            var ds = DryWindowDataset.Split(rows);

            // Get per-val P(wet) for this version. Branch on phase:
            //   3b uses the LightGBM model + optional isotonic calibrator
            //   3g re-runs MC over 3a's hourly q from the bound replay parquet
            var (probs, labels) = metadata.Phase == DryWindow3gPredictor.Phase3g
                ? Score3gOnVal(metadata, stationSlug, lead, ds, daytime, ct)
                : Score3bOnVal(versionDir, spec, lead, ds, stationName);

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

    /// 3g path: replay MC over the bound 3a champion's hourly q.
    private (List<double> Probs, List<bool> Labels) Score3gOnVal(
        ModelArtifact.TrainingMetadata metadata, string stationSlug, int lead,
        DryWindowDataset ds, DaytimeWindow daytime, CancellationToken ct)
    {
        var probs = new List<double>();
        var labels = new List<bool>();

        var v3a = metadata.Hyperparameters.HpString("precip_3a_version");
        if (string.IsNullOrEmpty(v3a))
        {
            _log.LogWarning("  3g version missing precip_3a_version metadata; skipping");
            return (probs, labels);
        }
        var hourly = DryWindow3gPredictor.LoadReplayHourly(
            _cfg.Storage.PredictionsPath, stationSlug, v3a, lead);

        var rng = new Random(42);
        foreach (var row in ds.Val)
        {
            ct.ThrowIfCancellationRequested();
            var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
            var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, row.TargetDateUtc, s, e);
            if (q is null) continue;
            // Window length is encoded in the 3g version's training metadata
            // as Hyperparameters["window_hours"]. ds came from the same window,
            // so the row-vector window naturally matches; pull from metadata
            // to be explicit.
            var window = metadata.Hyperparameters.HpInt("window_hours") ?? 3;
            probs.Add(DryWindow3gPredictor.ProbDryWindow(q, window, rng, DefaultMcSamples));
            labels.Add(row.Label);
        }
        return (probs, labels);
    }

    private string? ResolveStationName(string slug)
    {
        foreach (var s in _cfg.Location.Rainfall.Stations)
            if (StationSlug.WithEaPrefix(s.Name).Equals(slug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        return null;
    }
}
