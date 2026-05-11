using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Weekly rolling precipitation verification. Reads <see cref="PrecipPredictionRow"/>
/// parquet + EA 15-min rainfall parquet via DuckDB, aggregates truth to hourly (four
/// readings per hour required, matching training), loads training metadata for every
/// (station, version) that appears, then hands the pre-joined data to
/// <see cref="PrecipVerifier"/> for the stratification + drift logic.
///
/// Defaults tuned for wet-hour sparsity vs temperature's always-dense signal:
///   --window-days   30   (longer than temperature's 14 — wet hours are sparse)
///   --latency-days  5    (EA provisional-reading buffer)
///   --drift         1.5  (flag when rolling blend Brier &gt; 1.5× training test Brier)
/// </summary>
public sealed class PrecipVerifyCommand
{
    private readonly ILogger<PrecipVerifyCommand> _log;
    private readonly AppConfig _cfg;
    private readonly TruthRepository _truth;
    private readonly ModelMetadataRepository _metadata;
    private readonly PredictionsRepository _predictions;

    public PrecipVerifyCommand(ILogger<PrecipVerifyCommand> log, AppConfig cfg,
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
        var allStations = ModelArtifact.ListStations(modelsRoot, "precipitation");
        if (allStations.Count == 0)
        {
            _log.LogError("No precipitation blender artefacts found under {Dir}. Train first.",
                Path.Combine(modelsRoot, "precipitation"));
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
            "PrecipVerify: as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift {Drift:0.00}×, stations=[{Stations}]",
            asOfUtc, windowStart, windowEnd, latencyDays, driftThreshold, string.Join(", ", stations));

        var predictions = _predictions.GetPrecipitationPredictions(stations, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);
        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — skipping report.");
            return 0;
        }

        // Truth lookup needs to cover the window + persistence lookback (up to 72h before windowStart).
        var truth = _truth.GetEaHourlyRainfallByStation(stations, windowStart.AddHours(-72), windowEnd, ct);
        var truthPoints = truth.Sum(kv => kv.Value.Count);
        _log.LogInformation("Loaded {N} hourly rainfall truth points across {S} stations.",
            truthPoints, truth.Count);

        var metadata = _metadata.GetTrainingMetadataForPrecipitationKeys(
            predictions.Select(p => (p.TruthStation, p.ModelVersion)));
        _log.LogInformation("Loaded metadata for {N} (station, version) keys.", metadata.Count);

        var rows = PrecipVerifier.Compute(new PrecipVerifier.Inputs
        {
            Predictions = predictions,
            TruthByStationTime = truth,
            MetadataByKey = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = latencyDays,
            DriftThreshold = driftThreshold,
            DriftThresholdSlopePer24h = TempVerifyCommand.DriftThresholdSlopeDefault,
            MinDriftN = TempVerifyCommand.MinDriftNDefault,
        });

        var md = PrecipVerifyReporter.BuildMarkdown(asOfUtc, windowDays, latencyDays, driftThreshold, rows);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"verify_precipitation_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        // Structured sidecar for the Models-page "Verify history" table.
        var history = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = "precipitation",
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = latencyDays,
            MetricLabel = "Brier",
            Rows = rows.Select(r => new WeatherBlend.Models.VerifyHistoryRow
            {
                Station = r.TruthStation,
                ModelVersion = r.ModelVersion,
                Phase = metadata.TryGetValue((r.TruthStation, r.ModelVersion), out var meta) ? meta.Phase : null,
                LeadHours = r.LeadHours,
                WindowHours = null,
                N = r.N,
                BlendMetric = r.BlendBrier,
                ClimMetric = r.ClimBrier,
                MeanOfModelsMetric = r.MeanOfModelsBrier,
                BestSingleName = r.BestSingleName,
                BestSingleMetric = r.BestSingleBrier,
                ReferenceTrainingMetric = r.ReferenceTestBrier,
                DriftFlag = r.DriftFlag,
            }).ToList(),
        };
        await Evaluate.VerifyHistoryWriter.WriteAsync(_cfg.Storage.ReportsPath, history, ct);

        foreach (var r in rows)
        {
            _log.LogInformation(
                "{Station} / {Version} lead {Lead}h — n={N} (wet={Wet}), Brier {Brier:0.0000} (clim {Clim:0.0000}, BSS {Bss:+0.000;-0.000;0.000}){Drift}",
                r.TruthStation, r.ModelVersion, r.LeadHours, r.N, r.WetN,
                r.BlendBrier, r.ClimBrier, r.Bss,
                r.DriftFlag ? "  [DRIFT]" : "");
        }

        var drifting = rows.Count(r => r.DriftFlag);
        return drifting > 0 ? 4 : 0;
    }

    private IReadOnlyList<string> FilterStations(IReadOnlyList<string> all, string filter)
    {
        if (string.Equals(filter, "all", StringComparison.OrdinalIgnoreCase))
            return all;

        var exact = all.FirstOrDefault(s => string.Equals(s, filter, StringComparison.OrdinalIgnoreCase));
        if (exact is not null) return new[] { exact };

        var derivedSlug = StationSlug.WithEaPrefix(filter);
        var byDerived = all.FirstOrDefault(s => string.Equals(s, derivedSlug, StringComparison.OrdinalIgnoreCase));
        return byDerived is null ? Array.Empty<string>() : new[] { byDerived };
    }

}
