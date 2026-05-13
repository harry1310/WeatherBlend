using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Mlp;

namespace WeatherBlend.Commands;

/// <summary>
/// Replays a trained Phase 3a precipitation blender against EVERY historical
/// row the feature builder produces (full span, not just the live anchor),
/// emitting a flat parquet of (ValidTimeUtc, lead, prob_wet, label). One-off
/// research output for the dry-window start-hour PoC — gives us the dense
/// hourly P(wet) the live predict tree never persists, so we can backtest
/// curve shapes against raw rainfall truth.
///
/// Reuses <see cref="PrecipFeatureBuilder.BuildForLead"/> (same SQL pivot as
/// training), <see cref="ModelArtifact.LoadLeadModel"/> (the saved LightGBM
/// blob) and <see cref="PrecipOccurrenceTrainer.PredictVectorProbability"/>
/// (vector inference path). One process, model loaded once per lead.
///
/// Output: <c>data/predictions/precipitation_replay/{station}/{version}/lead_{L}h.parquet</c>.
/// Separate root from the live tree so a tidy <c>rm -rf</c> after the
/// experiment leaves nothing behind. Phase 3c (rich) is intentionally
/// rejected — the persistence features it needs at predict time would be
/// leaky-vs-truth here (we'd be feeding tomorrow's rain into today's
/// forecast). Lean (3a) is feature-only on the NWP side, no truth leakage.
/// </summary>
public sealed class PrecipReplayCommand
{
    public const string OutputSubdir = "precipitation_replay";

    private readonly ILogger<PrecipReplayCommand> _log;
    private readonly AppConfig _cfg;

    public PrecipReplayCommand(ILogger<PrecipReplayCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public sealed class ReplayRow
    {
        public DateTime ValidTimeUtc { get; init; }
        public int LeadHours { get; init; }
        public double ProbWet { get; init; }
        public bool Label { get; init; }
    }

    public async Task<int> RunAsync(
        string truthStationSlug, string modelVersion, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versionDir = ModelArtifact.ResolveStationVersionDir(
            modelsRoot, "precipitation", truthStationSlug, modelVersion);
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);

        if (PrecipPhases.IsRich(metadata.Phase))
        {
            _log.LogError(
                "Replay rejects Phase 3c (rich) artefacts — the EA-rainfall persistence " +
                "features used at predict time would leak truth into rows whose label is " +
                "in the same window. Re-point at a lean (3a) version.");
            return 2;
        }
        if (string.Equals(metadata.Phase, "3d", StringComparison.Ordinal))
        {
            _log.LogError(
                "Replay rejects Phase 3d (exact-runtime) artefacts — replay rebuilds rows " +
                "via PrecipFeatureBuilder which only knows offset_day model IDs. Re-point " +
                "at a 3a (lean) version, or extend replay with an exact-runtime path.");
            return 2;
        }
        // Phase 3e (TorchSharp MLP on rich features) gets its own path: rich
        // feature build + MLP predict. Same output shape (ValidTimeUtc, LeadHours,
        // ProbWet, Label) as 3a's lean replay so downstream consumers (3g, 3h GRU
        // training in dry_window/scripts/train_3h_rnn.py) read it uniformly.
        // Added 2026-05-13 to power the "GRU on 3e hourly P(wet)" arm of the
        // dry-window bake-off.
        if (string.Equals(metadata.Phase, "3e", StringComparison.Ordinal))
            return await RunReplayMlpAsync(truthStationSlug, modelVersion, leads, ct);

        var friendly = ResolveFriendlyStationName(truthStationSlug);
        if (friendly is null)
        {
            _log.LogError("Cannot resolve config station name for slug {Slug}.", truthStationSlug);
            return 2;
        }

        _log.LogInformation(
            "Replay — station={Slug} ({Friendly}), version={V}, phase={Phase}, leads=[{Leads}]",
            truthStationSlug, friendly, metadata.Version, metadata.Phase,
            string.Join(",", leads));

        var ml = new MLContext(seed: 42);
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);

        var outRoot = Path.Combine(
            _cfg.Storage.PredictionsPath, OutputSubdir, truthStationSlug, metadata.Version);
        Directory.CreateDirectory(outRoot);

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("Lead {Lead}h not present in feature_schema.json — skip.", lead);
                continue;
            }

            _log.LogInformation("--- lead {Lead}h ---", lead);
            var rows = PrecipFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                _cfg.Location.Name, friendly, spec, ct);
            _log.LogInformation("  {N} rows ({S:yyyy-MM-dd}..{E:yyyy-MM-dd})",
                rows.Count,
                rows.Count > 0 ? rows[0].ValidTimeUtc : DateTime.MinValue,
                rows.Count > 0 ? rows[^1].ValidTimeUtc : DateTime.MinValue);
            if (rows.Count == 0) continue;

            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var probs = PrecipOccurrenceTrainer.PredictVectorProbability(ml, model, spec, rows);

            var output = new List<ReplayRow>(rows.Count);
            for (int i = 0; i < rows.Count; i++)
            {
                output.Add(new ReplayRow
                {
                    ValidTimeUtc = rows[i].ValidTimeUtc,
                    LeadHours = lead,
                    ProbWet = probs[i],
                    Label = rows[i].Label,
                });
            }

            var outPath = Path.Combine(outRoot, $"lead_{lead}h.parquet");
            await ParquetSerializer.SerializeAsync(output, outPath, cancellationToken: ct);
            _log.LogInformation("  wrote {N} rows → {Path}", output.Count, outPath);
        }

        return 0;
    }

    /// <summary>
    /// Phase 3e (TorchSharp MLP on rich features) replay. Mirrors the lean
    /// path but uses <see cref="PrecipRichFeatureBuilder"/> + MLP load/predict
    /// to produce identically-shaped (ValidTimeUtc, LeadHours, ProbWet, Label)
    /// parquets. Note: rich features include EA-rainfall persistence at
    /// runTime = ValidTime - Lead, which is in the past relative to the label
    /// at ValidTime — same as 3e training-time, no leakage between the
    /// persistence window and the label window.
    /// </summary>
    private async Task<int> RunReplayMlpAsync(
        string truthStationSlug, string modelVersion, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versionDir = ModelArtifact.ResolveStationVersionDir(
            modelsRoot, "precipitation", truthStationSlug, modelVersion);
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);

        var friendly = ResolveFriendlyStationName(truthStationSlug);
        if (friendly is null)
        {
            _log.LogError("Cannot resolve config station name for slug {Slug}.", truthStationSlug);
            return 2;
        }

        _log.LogInformation(
            "Replay (MLP) — station={Slug} ({Friendly}), version={V}, phase={Phase}, leads=[{Leads}]",
            truthStationSlug, friendly, metadata.Version, metadata.Phase,
            string.Join(",", leads));

        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var outRoot = Path.Combine(
            _cfg.Storage.PredictionsPath, OutputSubdir, truthStationSlug, metadata.Version);
        Directory.CreateDirectory(outRoot);

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("Lead {Lead}h not present in feature_schema.json — skip.", lead);
                continue;
            }

            _log.LogInformation("--- lead {Lead}h ---", lead);
            var rows = PrecipRichFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                _cfg.Location.Name, friendly, spec, ct);
            _log.LogInformation("  {N} rows ({S:yyyy-MM-dd}..{E:yyyy-MM-dd})",
                rows.Count,
                rows.Count > 0 ? rows[0].ValidTimeUtc : DateTime.MinValue,
                rows.Count > 0 ? rows[^1].ValidTimeUtc : DateTime.MinValue);
            if (rows.Count == 0) continue;

            // Load MLP weights for this lead from the 3e bundle's mlp_lead_{L}h.pt
            // and per-lead preprocess (scaler mean + scale). Same path 3e predict
            // uses at live-time. Dead-column zeroing (added 2026-05-13 in
            // MlpTrainer) carries over for free.
            var (module, perLead) = MlpArtifact.LoadLeadModel(versionDir, lead);
            var trained = new MlpTrainer.TrainedMlp(
                Module: module,
                ScalerMean: perLead.ScalerMean.ToArray(),
                ScalerScale: perLead.ScalerScale.ToArray(),
                Hyperparameters: new MlpTrainer.Hyperparameters(
                    HiddenSizes: perLead.HiddenSizes.ToArray(),
                    Dropout: perLead.Dropout, LearningRate: perLead.LearningRate,
                    BatchSize: perLead.BatchSize, MaxEpochs: perLead.MaxEpochs,
                    EarlyStoppingPatience: perLead.EarlyStoppingPatience,
                    Seed: perLead.Seed),
                FeatureNames: perLead.FeatureNames,
                EpochsRun: perLead.EpochsRun,
                BestValBrier: perLead.BestValBrier);

            var probs = MlpTrainer.PredictVectorProbability(trained, rows);

            var output = new List<ReplayRow>(rows.Count);
            for (int i = 0; i < rows.Count; i++)
            {
                output.Add(new ReplayRow
                {
                    ValidTimeUtc = rows[i].ValidTimeUtc,
                    LeadHours = lead,
                    ProbWet = probs[i],
                    Label = rows[i].Label,
                });
            }

            var outPath = Path.Combine(outRoot, $"lead_{lead}h.parquet");
            await ParquetSerializer.SerializeAsync(output, outPath, cancellationToken: ct);
            _log.LogInformation("  wrote {N} rows -> {Path}", output.Count, outPath);
        }

        return 0;
    }

    private string? ResolveFriendlyStationName(string slug)
    {
        // Mirror PrecipPredictCommand's resolution: strip "ea_" and match
        // case-insensitively against the configured rainfall station names.
        var bare = slug.StartsWith("ea_", StringComparison.Ordinal) ? slug[3..] : slug;
        var slugified = bare.ToLowerInvariant();
        foreach (var s in _cfg.Location.Rainfall.Stations)
        {
            var sSlug = StationSlug.Of(s.Name);
            if (sSlug.Equals(slugified, StringComparison.Ordinal)) return s.Name;
        }
        return null;
    }
}
