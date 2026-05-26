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

    /// <summary>Default α matching the dry-window equivalent — 90% target
    /// coverage. Used by the train-time auto-refit hook so a calibrator is
    /// always present at the same coverage level as the back-fill command.</summary>
    public const double DefaultAlpha = 0.10;

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
            foreach (var versionName in entry.Active)
            {
                var (f, s) = await FitOneAsync(stationSlug, versionName, alpha, ct);
                fitted += f;
                skipped += s;
            }
        }

        _log.LogInformation("Precip conformal fit complete. Fitted={F} Skipped={S}", fitted, skipped);
        return fitted == 0 ? 3 : 0;
    }

    /// <summary>
    /// Fit conformal calibrators for every lead of one specific (station,
    /// version) cell. Used by the train-time auto-refit hook (called from
    /// TempTrainCommand right after PromoteStationVersionAs*) so a fresh
    /// champion or challenger never ships without a calibrator.
    ///
    /// Returns (fitted, skipped) counts per-lead so callers can log a single
    /// summary line without re-deriving the metadata.
    /// </summary>
    public Task<(int Fitted, int Skipped)> FitOneAsync(
        string stationSlug, string versionName, double alpha, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        int fitted = 0, skipped = 0;

        var stationName = ResolveStationName(stationSlug);
        if (stationName is null)
        {
            _log.LogWarning("{S}: cannot resolve station name from slug; skipping", stationSlug);
            return Task.FromResult((fitted, skipped + 1));
        }

        var versionDir = System.IO.Path.Combine(modelsRoot, "precipitation", stationSlug, versionName);
        if (!System.IO.Directory.Exists(versionDir))
        {
            _log.LogWarning("{S} {V}: version dir missing; skipping", stationSlug, versionName);
            return Task.FromResult((fitted, skipped + 1));
        }
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        _log.LogInformation("Conformal fit: {S} / {V} (phase {P})", stationSlug, versionName, metadata.Phase);

        // Phase 3d (exact-runtime) artefacts use a different feature
        // builder (PrecipExactFeatureBuilder, exact-runtime model IDs like
        // gfs_ncep). The conformal fit currently dispatches between
        // PrecipFeatureBuilder (lean) and PrecipRichFeatureBuilder (rich)
        // — neither knows the exact-runtime IDs. Skip cleanly with a
        // warning rather than crashing in TempFeatureBuilder.ShortName.
        // 3d ConformalSetTag stays null on predict rows in the meantime
        // (legacy-row behaviour, harmless). Adding 3d support is a follow-
        // up: needs PrecipExactFeatureBuilder.Build to be plumbed in here.
        if (string.Equals(metadata.Phase, "3d", StringComparison.Ordinal))
        {
            _log.LogWarning(
                "{S} {V}: phase=3d uses exact-runtime feature builder; conformal fit not yet supported. Skipping.",
                stationSlug, versionName);
            return Task.FromResult((fitted, skipped + metadata.PerLead.Count));
        }

        // Each lead trains its own model+calibrator. PerLead keys are
        // string-form lead-hours; iterate them rather than guessing.
        foreach (var leadKey in metadata.PerLead.Keys)
        {
            ct.ThrowIfCancellationRequested();
            if (!int.TryParse(leadKey, out var lead)) continue;
            var spec = ModelArtifact.LoadBlenderSpecs(versionDir).GetValueOrDefault(lead);
            if (spec is null)
            {
                _log.LogWarning("  lead {L}h: no BlenderSpec; skipping", lead);
                continue;
            }

            // Same row build + chronological split as the original
            // training. The feature builder MUST match the spec the
            // model was fit against, otherwise the schema mismatch
            // trips ComposeRow's pack-length check (caught 2026-05-26
            // when 3o's first CI retrain crashed with "Feature pack
            // mismatch: wrote 23, expected 68" — conformal-fit hadn't
            // been taught about 3o's rich+oro path).
            //   * 3a (lean)  → PrecipFeatureBuilder        (23 feat)
            //   * 3c (rich)  → PrecipRichFeatureBuilder    (59 feat)
            //   * 3o (rich+oro, pooled 4-station Bonehill)
            //                → PrecipRichOroFeatureBuilder (68 feat)
            List<BinaryTrainingRow> rows;
            if (string.Equals(metadata.Phase, "3o", StringComparison.Ordinal))
            {
                // 3o needs the per-station orographic record + the
                // train-time station index. Index is stamped on the
                // bundle's training_metadata.Hyperparameters so we
                // recover the exact (oro, index) pair the train run
                // used, not a re-derived guess.
                var oroRoot = Path.Combine("data", "static", "orographic");
                var oroBySlug = WeatherBlend.Train.Oro.OroStaticFeatures.LoadAll(oroRoot);
                if (!oroBySlug.TryGetValue(stationSlug, out var oro))
                {
                    _log.LogWarning(
                        "  {Slug} lead {L}h: orographic record missing for 3o conformal fit (expected at {OroPath}); skipping.",
                        stationSlug, lead, Path.Combine(oroRoot, $"{stationSlug}.json"));
                    skipped++;
                    continue;
                }
                // HpInt unwraps JsonElement properly — round-tripping the
                // training_metadata dict through System.Text.Json deserialises
                // every value as a JsonElement, and Convert.ToInt32 can't
                // handle that (InvalidCastException, caught 2026-05-26 in
                // retrain rerun 26437976508). HpInt returns null on missing
                // keys / wrong types so the next check still catches "no
                // index stamped" cleanly.
                var stationIndex = metadata.Hyperparameters.HpInt("phase3o_station_index");
                if (stationIndex is null)
                {
                    _log.LogWarning(
                        "  {Slug} lead {L}h: 3o bundle missing `phase3o_station_index` metadata; skipping conformal.",
                        stationSlug, lead);
                    skipped++;
                    continue;
                }
                rows = WeatherBlend.Train.Oro.PrecipRichOroFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    _cfg.Location.Name, stationName, oro, stationIndex.Value, spec, ct);
            }
            else if (PrecipPhases.IsRich(metadata.Phase))
            {
                rows = PrecipRichFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    _cfg.Location.Name, stationName, spec, ct);
            }
            else
            {
                rows = PrecipFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    _cfg.Location.Name, stationName, spec, ct);
            }
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

        return Task.FromResult((fitted, skipped));
    }

    private string? ResolveStationName(string slug)
    {
        foreach (var s in _cfg.Location.Rainfall.Stations)
            if (StationSlug.WithEaPrefix(s.Name).Equals(slug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        return null;
    }
}
