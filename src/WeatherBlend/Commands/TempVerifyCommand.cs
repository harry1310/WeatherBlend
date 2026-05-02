using System.Data;
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

    public TempVerifyCommand(ILogger<TempVerifyCommand> log, AppConfig cfg,
        TruthRepository truth, ModelMetadataRepository metadata)
    {
        _log = log;
        _cfg = cfg;
        _truth = truth;
        _metadata = metadata;
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

        var predictions = QueryPredictions(windowStart, windowEnd, ct);
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

    private IReadOnlyList<TempPredictionRow> QueryPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Scope to the temperature subtree only — precipitation / dry-window predictions
        // live alongside and don't share the temperature schema, so a top-level glob
        // would pull rows missing TempGfs/…/BlendTemperature and fail the query.
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "temperature", "**", "*.parquet"));

        // hive_partitioning=false — the `model_version=` hive key collides with the
        // in-file ModelVersion column under case-insensitive resolution. Same rule
        // applies across the codebase (see CLAUDE.md gotcha). In-file column wins.
        var sql = $@"
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendTemperature,
       TempGfs, TempEcmwf, TempIcon, TempMf, TempUkmo, TempGem, TempAifs,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem, RunTimeAifs,
       TempMean, TempStd, TempRange,
       FeatureVectorHash
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ModelVersion, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(sql, r => new TempPredictionRow
        {
            LocationName        = r.GetString(0),
            ModelVersion        = r.GetString(1),
            PredictionMadeAtUtc = r.GetDateTime(2),
            ValidTimeUtc        = r.GetDateTime(3),
            LeadHours           = r.GetInt32(4),
            BlendTemperature    = r.GetDouble(5),
            TempGfs   = NullableDouble(r,  6),
            TempEcmwf = NullableDouble(r,  7),
            TempIcon  = NullableDouble(r,  8),
            TempMf    = NullableDouble(r,  9),
            TempUkmo  = NullableDouble(r, 10),
            TempGem   = NullableDouble(r, 11),
            TempAifs  = NullableDouble(r, 12),
            RunTimeGfs   = NullableDate(r, 13),
            RunTimeEcmwf = NullableDate(r, 14),
            RunTimeIcon  = NullableDate(r, 15),
            RunTimeMf    = NullableDate(r, 16),
            RunTimeUkmo  = NullableDate(r, 17),
            RunTimeGem   = NullableDate(r, 18),
            RunTimeAifs  = NullableDate(r, 19),
            TempMean  = NullableDouble(r, 20),
            TempStd   = NullableDouble(r, 21),
            TempRange = NullableDouble(r, 22),
            FeatureVectorHash = r.IsDBNull(23) ? "" : r.GetString(23),
        }, _log, "Predictions tree empty — nothing to verify.", ct);
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static DateTime? NullableDate(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
