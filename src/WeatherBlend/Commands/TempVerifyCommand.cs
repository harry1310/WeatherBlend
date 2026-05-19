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
        => await RunAsync(target, asOf, windowDays, era5LatencyDays, driftThreshold, locationOverride: null, ct);

    public async Task<int> RunAsync(
        string target,
        DateOnly? asOf,
        int windowDays,
        int era5LatencyDays,
        double driftThreshold,
        string? locationOverride,
        CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported (got '{Target}')", target);
            return 2;
        }

        // Resolve which location's predictions to verify. Default = primary
        // (Bonehill back-compat). When --location is passed, scope to that
        // location's prediction partition; the per-station temperature
        // manifest (2026-05-14 layout) means each location's predictions
        // live under temperature/{location}/... so the filter is precise.
        var primaryName = _cfg.Location.Name;
        var locationName = string.IsNullOrWhiteSpace(locationOverride) ? primaryName : locationOverride!;
        var isPrimary = string.Equals(locationName, primaryName, StringComparison.OrdinalIgnoreCase);

        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-era5LatencyDays);

        _log.LogInformation(
            "Verify: target=temperature, location={Loc}, as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift {Drift:0.00}×",
            locationName, asOfUtc, windowStart, windowEnd, era5LatencyDays, driftThreshold);

        var predictions = _predictions.GetTemperaturePredictions(
            locations: new[] { locationName }, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);

        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — skipping report.");
            return 0;
        }

        // ERA5 lookup needs to cover the whole window plus the persistence-lookback
        // (up to 72h before windowStart) so persistence MAE can resolve.
        // Phase C commit 3: locationName is now required on GetEra5Hourly.
        // 3a passes primary explicitly here; 3b will loop _cfg.Locations and
        // emit per-loc verify rows. See [[project_phase_c_status]].
        var truth = _truth.GetEra5Hourly(_cfg.Location.Name, windowStart.AddHours(-72), windowEnd, ct);
        _log.LogInformation("Loaded {N} ERA5 truth points.", truth.Count);

        // Temperature is per-station (per-location) on disk since 2026-05-14;
        // derive (station, version) tuples from the prediction LocationName
        // since predictions carry the location they were scored against.
        var metadata = _metadata.GetTemperatureTrainingMetadataForVersions(
            predictions.Select(p => (Station: p.LocationName, Version: p.ModelVersion)));
        _log.LogInformation("Loaded metadata for {N} versions: {Versions}",
            metadata.Count, string.Join(", ", metadata.Keys));

        var inputs = new TempVerifier.Inputs
        {
            Predictions = predictions,
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            Era5LatencyDays = era5LatencyDays,
            DriftThreshold = driftThreshold,
            DriftThresholdSlopePer24h = DriftThresholdSlopeDefault,
            MinDriftN = MinDriftNDefault,
        };
        var rows = TempVerifier.Compute(inputs);
        // Parallel view: same predictions grouped by ACTUAL lead at prediction
        // time in 6h buckets. Distinct from the trained-lead view above —
        // post 2026-05-04 hourly predict, each trained-lead bucket spreads
        // across actual leads L .. L+23. Surfaces whether MAE varies within
        // a trained bucket. No drift flag (no per-actual-lead baseline).
        var bucketRows = TempVerifier.ComputeActualLeadBuckets(inputs);

        var md = TempVerifyReporter.BuildMarkdown(asOfUtc, windowDays, era5LatencyDays, driftThreshold, rows, metadata);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        // Renamed 2026-05-02 from verify_<asof>.md to verify_temperature_<asof>.md
        // so the filename pattern matches the precip / dry-window / element
        // siblings. Pre-rename markdowns stay as historical artefacts on R2;
        // only future writes get the new name.
        // Non-primary location adds a {location} segment so Bonehill and
        // Membury reports for the same date don't overwrite each other.
        var mdName = isPrimary
            ? $"verify_temperature_{asOfUtc:yyyy-MM-dd}.md"
            : $"verify_temperature_{locationName}_{asOfUtc:yyyy-MM-dd}.md";
        var outPath = Path.Combine(_cfg.Storage.ReportsPath, mdName);
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
                ViewKind = "trained",
            })
            // Append the actual-lead-bucket view as additional rows in the
            // same sidecar. Distinguishable by ViewKind="actual_6h" + the
            // ActualLeadBucketLowH field; ReferenceTrainingMetric / DriftFlag
            // intentionally null/false.
            .Concat(bucketRows.Select(r => new WeatherBlend.Models.VerifyHistoryRow
            {
                Station = null,
                ModelVersion = r.ModelVersion,
                Phase = metadata.TryGetValue(r.ModelVersion, out var meta2) ? meta2.Phase : null,
                LeadHours = r.LeadHours, // = bucket low for these rows
                WindowHours = null,
                N = r.N,
                BlendMetric = r.BlendMae,
                ClimMetric = null,
                MeanOfModelsMetric = r.MeanMae,
                BestSingleName = r.BestSingleName,
                BestSingleMetric = r.BestSingleMae,
                ReferenceTrainingMetric = null,
                DriftFlag = false,
                ViewKind = "actual_6h",
                ActualLeadBucketLowH = r.ActualLeadBucketLowH,
            }))
            .ToList(),
        };
        // Models-page history sidecar: skip for non-primary locations until
        // the multi-location site rework lands. Writing a second JSON with
        // the same Target="temperature" would surface Membury rows as new
        // model cards on the Bonehill Models page — confusing today, will
        // be wired properly when the per-location page split ships.
        // Markdown report is unaffected (location prefix on the filename).
        if (isPrimary)
        {
            await Evaluate.VerifyHistoryWriter.WriteAsync(_cfg.Storage.ReportsPath, history, ct);
        }
        else
        {
            _log.LogInformation(
                "Skipping verify-history JSON for non-primary location '{Loc}' — Models-page integration deferred.",
                locationName);
        }

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

    /// <summary>
    /// Minimum N for a drift flag in production. Picked to suppress single-prediction
    /// noise (smoke tests, the first day after a new model is deployed) without
    /// hiding genuine n=10+ signal. Tests bypass this by leaving Inputs.MinDriftN
    /// at its default of 1. Shared across all four verify commands.
    /// </summary>
    public const int MinDriftNDefault = 10;

    /// <summary>
    /// Per-additional-24h relaxation of the drift threshold (added on top of the
    /// CLI-supplied <c>--drift</c> base). 0.1 gives e.g. 24h:1.5×, 48h:1.6×,
    /// 72h:1.7×, 96h:1.8×, 120h:1.9× when base is 1.5. Long-lead forecasts
    /// degrade naturally — flagging a 120h prediction missing 1.5× its training
    /// MAE is far less actionable than the same at 24h. Tune later if needed.
    /// </summary>
    public const double DriftThresholdSlopeDefault = 0.1;
}
