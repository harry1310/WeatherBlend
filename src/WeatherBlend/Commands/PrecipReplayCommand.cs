using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

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
                _cfg.Location.Name, friendly, spec, minValidTime: null, ct);
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
