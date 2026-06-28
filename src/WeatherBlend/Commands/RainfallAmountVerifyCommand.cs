using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.RainfallAmount;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Twice-weekly rolling verify for Phase 3f (rainfall_amount). Reads the
/// <see cref="RainfallAmountPredictionRow"/> tree + EA 15-min rainfall via
/// DuckDB, hands the join + per-row distributional metrics to
/// <see cref="RainfallAmountVerifier"/>, then writes the standard
/// markdown + JSON sidecar pair under <c>data/reports/</c>.
///
/// Drift flag fires when rolling blend CRPS &gt; threshold × the bundle's
/// training-time test CRPS (stored in <c>BlendTestMae</c> by
/// train_3f.py). Returns exit 4 on any drift so the existing
/// <c>verify.yml</c> step that propagates exit codes auto-files
/// <c>[ci-fail] verify</c> issues without target-specific glue.
///
/// Defaults mirror <see cref="PrecipVerifyCommand"/> — 30-day window,
/// 5-day latency cutoff against EA's provisional readings, 1.5× drift —
/// because the metric scale (CRPS) sits in the same ballpark as Brier.
/// </summary>
public sealed class RainfallAmountVerifyCommand
{
    private readonly ILogger<RainfallAmountVerifyCommand> _log;
    private readonly AppConfig _cfg;
    private readonly TruthRepository _truth;
    private readonly ModelMetadataRepository _metadata;
    private readonly PredictionsRepository _predictions;

    public RainfallAmountVerifyCommand(ILogger<RainfallAmountVerifyCommand> log, AppConfig cfg,
        TruthRepository truth, ModelMetadataRepository metadata, PredictionsRepository predictions)
    {
        _log = log;
        _cfg = cfg;
        _truth = truth;
        _metadata = metadata;
        _predictions = predictions;
    }

    public async Task<int> RunAsync(
        string truthStation,
        DateOnly? asOf,
        int windowDays,
        int latencyDays,
        double driftThreshold,
        CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var allStations = ModelArtifact.ListStations(modelsRoot, "rainfall_amount");
        if (allStations.Count == 0)
        {
            _log.LogError("No rainfall_amount blender artefacts found under {Dir}. Train 3f first.",
                Path.Combine(modelsRoot, "rainfall_amount"));
            return 2;
        }

        var stations = FilterStations(allStations, truthStation);
        if (stations.Count == 0)
        {
            _log.LogError("Station filter '{F}' matched no trained stations. Known: [{Known}]",
                truthStation, string.Join(", ", allStations));
            return 2;
        }

        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-latencyDays);

        _log.LogInformation(
            "RainfallAmountVerify: as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift {Drift:0.00}×, stations=[{Stations}]",
            asOfUtc, windowStart, windowEnd, latencyDays, driftThreshold, string.Join(", ", stations));

        var predictions = _predictions.GetRainfallAmountPredictions(stations, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);
        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — skipping report.");
            return 0;
        }

        var truth = _truth.GetHourlyRainfallByStation(stations, windowStart, windowEnd, ct);
        var truthPoints = truth.Sum(kv => kv.Value.Count);
        _log.LogInformation("Loaded {N} hourly rainfall truth points across {S} stations.",
            truthPoints, truth.Count);

        // 3f bundles live under data/models/rainfall_amount/{station}/{version}/.
        // The repository doesn't (yet) expose a rainfall_amount-aware bulk
        // loader — inline the per-key read here so verify doesn't ship a new
        // public surface on ModelMetadataRepository just for one caller.
        // Surface lifts later if Bonehill 3f or future distributional phases
        // make a second caller.
        var metadata = ResolveRainfallAmountMetadata(predictions);
        _log.LogInformation("Loaded metadata for {N} (station, version) keys.", metadata.Count);

        var rows = RainfallAmountVerifier.Compute(new RainfallAmountVerifier.Inputs
        {
            Predictions = predictions,
            TruthByStationTime = truth,
            MetadataByKey = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = latencyDays,
            DriftThreshold = driftThreshold,
            MinDriftN = TempVerifyCommand.MinDriftNDefault,
        });

        var md = RainfallAmountVerifyReporter.BuildMarkdown(asOfUtc, windowDays, latencyDays, driftThreshold, rows);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"verify_rainfall_amount_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        // JSON sidecar for the Models page's Verify-history table. CRPS becomes
        // the new BlendMetric (lower=better, same direction as Brier so the
        // chart's y-axis sign convention is unchanged); the distributional
        // companions (MaeWet, Coverage80, PitMean, PitBins, ExceedanceBriers)
        // are new nullable fields on VerifyHistoryRow that only the
        // rainfall_amount widget consumes.
        var history = new VerifyHistoryFile
        {
            Target = "rainfall_amount",
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = latencyDays,
            MetricLabel = "CRPS",
            Rows = rows.Select(r => new VerifyHistoryRow
            {
                Station = r.TruthStation,
                ModelVersion = r.ModelVersion,
                Phase = metadata.TryGetValue((r.TruthStation, r.ModelVersion), out var meta) ? meta.Phase : null,
                LeadHours = r.LeadHours,
                WindowHours = null,
                N = r.N,
                BlendMetric = r.BlendCrps,
                ReferenceTrainingMetric = r.ReferenceTestCrps,
                DriftFlag = r.DriftFlag,
                MaeWet = double.IsNaN(r.MaeWet) ? null : r.MaeWet,
                Coverage80 = r.Coverage80,
                PitMean = r.PitMean,
                PitBins = r.PitBins.ToList(),
                ExceedanceBriers = new Dictionary<string, double>(r.ExceedanceBriers, StringComparer.Ordinal),
            }).ToList(),
        };
        await Evaluate.VerifyHistoryWriter.WriteAsync(_cfg.Storage.ReportsPath, history, ct);

        foreach (var r in rows)
        {
            _log.LogInformation(
                "{Station} / {Version} lead {Lead}h — n={N} (wet={Wet}), CRPS {Crps:0.0000} (cov80 {Cov:0.000}, PIT mean {Pit:0.000}){Drift}",
                r.TruthStation, r.ModelVersion, r.LeadHours, r.N, r.WetN,
                r.BlendCrps, r.Coverage80, r.PitMean,
                r.DriftFlag ? "  [DRIFT]" : "");
        }

        var drifting = rows.Count(r => r.DriftFlag);
        return drifting > 0 ? 4 : 0;
    }

    /// <summary>
    /// Load 3f bundles' training_metadata.json from
    /// <c>data/models/rainfall_amount/{station}/{version}/</c>. Skips silently on
    /// missing dir / malformed JSON — keeps verify resilient to half-synced R2
    /// state at run start.
    /// </summary>
    private IReadOnlyDictionary<(string Station, string Version), ModelArtifact.TrainingMetadata>
        ResolveRainfallAmountMetadata(IReadOnlyList<RainfallAmountPredictionRow> predictions)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var keys = predictions.Select(p => (p.TruthStation, p.ModelVersion)).Distinct();
        var dict = new Dictionary<(string Station, string Version), ModelArtifact.TrainingMetadata>();
        foreach (var (station, version) in keys)
        {
            var versionDir = Path.Combine(modelsRoot, "rainfall_amount", station, version);
            if (!Directory.Exists(versionDir)) continue;
            try
            {
                dict[(station, version)] = ModelArtifact.LoadTrainingMetadata(versionDir);
            }
            catch (Exception ex)
            {
                _log.LogWarning("Failed to load 3f training_metadata for {Station}/{Version}: {Msg}",
                    station, version, ex.Message);
            }
        }
        return dict;
    }

    private IReadOnlyList<string> FilterStations(IReadOnlyList<string> all, string filter)
    {
        if (string.Equals(filter, "all", StringComparison.OrdinalIgnoreCase))
            return all;

        var exact = all.FirstOrDefault(s => string.Equals(s, filter, StringComparison.OrdinalIgnoreCase));
        if (exact is not null) return new[] { exact };

        var derivedSlug = _cfg.ResolveStationSlug(filter);
        var byDerived = all.FirstOrDefault(s => string.Equals(s, derivedSlug, StringComparison.OrdinalIgnoreCase));
        return byDerived is null ? Array.Empty<string>() : new[] { byDerived };
    }
}
