using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate;
using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Weekly rolling verification. Reads prediction parquet + ERA5 parquet via DuckDB,
/// loads training metadata for every ModelVersion that appears, then hands the
/// pre-joined data to <see cref="Verifier"/> for the actual stratification + drift
/// logic. Renders markdown via <see cref="VerifyReporter"/>.
///
/// Defaults come straight from the verify brief:
///   --window-days   14   (rolling window size)
///   --latency-days  5    (ERA5 release lag; nothing newer than AsOf − 5d is scored)
///   --drift         1.5  (flag when rolling blend MAE &gt; 1.5× training test MAE)
/// </summary>
public sealed class VerifyCommand
{
    private readonly ILogger<VerifyCommand> _log;
    private readonly AppConfig _cfg;

    public VerifyCommand(ILogger<VerifyCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
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
        var truth = QueryTruth(windowStart.AddHours(-72), windowEnd, ct);
        _log.LogInformation("Loaded {N} ERA5 truth points.", truth.Count);

        var metadata = LoadMetadataForVersions(predictions.Select(p => p.ModelVersion).Distinct());
        _log.LogInformation("Loaded metadata for {N} versions: {Versions}",
            metadata.Count, string.Join(", ", metadata.Keys));

        var rows = Verifier.Compute(new Verifier.Inputs
        {
            Predictions = predictions,
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            Era5LatencyDays = era5LatencyDays,
            DriftThreshold = driftThreshold,
        });

        var md = VerifyReporter.BuildMarkdown(asOfUtc, windowDays, era5LatencyDays, driftThreshold, rows);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"verify_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

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

    private IReadOnlyList<PredictionRow> QueryPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = Path.Combine(_cfg.Storage.PredictionsPath, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        // hive_partitioning=false — the `model_version=` hive key collides with the
        // in-file ModelVersion column under case-insensitive resolution. Same rule
        // applies across the codebase (see CLAUDE.md gotcha). In-file column wins.
        var sql = $@"
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendTemperature,
       TempGfs, TempEcmwf, TempIcon, TempMf, TempUkmo, TempGem,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem,
       TempMean, TempStd, TempRange,
       FeatureVectorHash
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ModelVersion, LeadHours, ValidTimeUtc";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<PredictionRow>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add(new PredictionRow
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
                    RunTimeGfs   = NullableDate(r, 12),
                    RunTimeEcmwf = NullableDate(r, 13),
                    RunTimeIcon  = NullableDate(r, 14),
                    RunTimeMf    = NullableDate(r, 15),
                    RunTimeUkmo  = NullableDate(r, 16),
                    RunTimeGem   = NullableDate(r, 17),
                    TempMean  = NullableDouble(r, 18),
                    TempStd   = NullableDouble(r, 19),
                    TempRange = NullableDouble(r, 20),
                    FeatureVectorHash = r.IsDBNull(21) ? "" : r.GetString(21),
                });
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Predictions tree empty — nothing to verify.");
            return Array.Empty<PredictionRow>();
        }
        return rows;
    }

    private IReadOnlyDictionary<DateTime, double> QueryTruth(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        var sql = $@"
SELECT ValidTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND Temperature2m IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var truth = new Dictionary<DateTime, double>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                truth[r.GetDateTime(0)] = r.GetDouble(1);
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("ERA5 tree empty — cannot verify.");
        }
        return truth;
    }

    private IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata> LoadMetadataForVersions(
        IEnumerable<string> versions)
    {
        var modelsRoot = Path.Combine("data", "models");
        var result = new Dictionary<string, ModelArtifact.TrainingMetadata>();
        foreach (var v in versions)
        {
            try
            {
                var dir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", v);
                if (!Directory.Exists(dir))
                {
                    _log.LogWarning("Metadata missing for version {V} — drift column will be blank.", v);
                    continue;
                }
                result[v] = ModelArtifact.LoadTrainingMetadata(dir);
            }
            catch (Exception ex)
            {
                _log.LogWarning("Failed to load metadata for version {V}: {Msg}", v, ex.Message);
            }
        }
        return result;
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static DateTime? NullableDate(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
