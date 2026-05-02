using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Temp;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Weekly rolling verification. Reads prediction parquet + ERA5 parquet via DuckDB,
/// loads training metadata for every ModelVersion that appears, then hands the
/// pre-joined data to <see cref="TempVerifier"/> for the actual stratification + drift
/// logic. Renders markdown via <see cref="TempVerifyReporter"/>.
///
/// Defaults come straight from the verify brief:
///   --window-days   14   (rolling window size)
///   --latency-days  5    (ERA5 release lag; nothing newer than AsOf − 5d is scored)
///   --drift         1.5  (flag when rolling blend MAE &gt; 1.5× training test MAE)
/// </summary>
public sealed class TempVerifyCommand
{
    private readonly ILogger<TempVerifyCommand> _log;
    private readonly AppConfig _cfg;
    private readonly TruthRepository _truth;
    private readonly ModelMetadataRepository _metadata;
    private readonly PredictionsRepository _predictions;

    public TempVerifyCommand(ILogger<TempVerifyCommand> log, AppConfig cfg,
        TruthRepository truth, ModelMetadataRepository metadata, PredictionsRepository predictions)
    {
        _log = log;
        _cfg = cfg;
        _truth = truth;
        _metadata = metadata;
        _predictions = predictions;
    }

    public async Task<int> RunAsync(
        string target,
        DateOnly? asOf,
        int windowDays,
        int era5LatencyDays,
        double driftThreshold,
        CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported (got '{Target}')", target);
            return 2;
        }

        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-era5LatencyDays);

        _log.LogInformation(
            "Verify: as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift {Drift:0.00}×",
            asOfUtc, windowStart, windowEnd, era5LatencyDays, driftThreshold);

        var predictions = _predictions.GetTemperaturePredictions(windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);

        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — skipping report.");
            return 0;
        }

        // ERA5 lookup needs to cover the whole window plus the persistence-lookback
        // (up to 72h before windowStart) so persistence MAE can resolve.
        var truth = _truth.GetEra5Hourly(windowStart.AddHours(-72), windowEnd, ct);
        _log.LogInformation("Loaded {N} ERA5 truth points.", truth.Count);

        var metadata = _metadata.GetTrainingMetadataForVersions("temperature",
            predictions.Select(p => p.ModelVersion));
        _log.LogInformation("Loaded metadata for {N} versions: {Versions}",
            metadata.Count, string.Join(", ", metadata.Keys));

        var rows = TempVerifier.Compute(new TempVerifier.Inputs
        {
            Predictions = predictions,
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            Era5LatencyDays = era5LatencyDays,
            DriftThreshold = driftThreshold,
        });

        var md = TempVerifyReporter.BuildMarkdown(asOfUtc, windowDays, era5LatencyDays, driftThreshold, rows, metadata);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        // Renamed 2026-05-02 from verify_<asof>.md to verify_temperature_<asof>.md
        // so the filename pattern matches the precip / dry-window / element
        // siblings. Pre-rename markdowns stay as historical artefacts on R2;
        // only future writes get the new name.
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"verify_temperature_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        // Structured sidecar for the Models-page "Verify history" table —
        // same data, machine-parseable. See VerifyHistoryFile docstring.
        var history = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = "temperature",
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = era5LatencyDays,
            MetricLabel = "MAE (°C)",
            Rows = rows.Select(r => new WeatherBlend.Models.VerifyHistoryRow
            {
                Station = null,
                ModelVersion = r.ModelVersion,
                Phase = metadata.TryGetValue(r.ModelVersion, out var meta) ? meta.Phase : null,
                LeadHours = r.LeadHours,
                WindowHours = null,
                N = r.N,
                BlendMetric = r.BlendMae,
                ClimMetric = null,
                MeanOfModelsMetric = r.MeanMae,
                BestSingleName = r.BestSingleName,
                BestSingleMetric = r.BestSingleMae,
                ReferenceTrainingMetric = r.ReferenceTestMae,
                DriftFlag = r.DriftFlag,
            }).ToList(),
        };
        await Evaluate.VerifyHistoryWriter.WriteAsync(_cfg.Storage.ReportsPath, history, ct);

        // One-line summary to stdout — easy to scrape from CI logs.
        foreach (var r in rows)
        {
            _log.LogInformation(
                "{Version} lead {Lead}h — n={N}, blend MAE {Mae:0.000}°C, bias {Bias:+0.000;-0.000;0.000}{Drift}",
                r.ModelVersion, r.LeadHours, r.N, r.BlendMae, r.BlendBias,
                r.DriftFlag ? "  [DRIFT]" : "");
        }

        var drifting = rows.Count(r => r.DriftFlag);
        // Non-zero exit code so CI can notice drift even if the job otherwise "succeeded".
        // Weekly workflow should treat this as a warning, not a hard failure.
        return drifting > 0 ? 4 : 0;
    }

}
