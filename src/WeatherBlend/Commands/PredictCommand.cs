using System.Globalization;
using System.Security.Cryptography;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended-temperature forecasts for leads {24, 48, 72} using the current
/// phase 2b per-lead blenders. Chained after <c>collect</c> in the daily workflow —
/// collect refreshes the forecast tree, predict reads it and writes prediction rows.
///
/// "Latest available" semantics: for a target valid time T, each of the six models
/// contributes the temperature from its most recent run that covers T, regardless
/// of cycle. Each model's source <c>RunTimeUtc</c> is recorded alongside the temp
/// so verification can see which cycles fed any given prediction.
///
/// Previous Runs API rows (<c>RunTimeSource = 'offset_day'</c>) are excluded — those
/// are historical training data, not a live forecast.
/// </summary>
public sealed class PredictCommand
{
    private readonly ILogger<PredictCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = { 24, 48, 72 };

    // Standard anchor time for --for-date retroactive fills. 08:00 UTC mirrors the
    // daily workflow schedule and is late enough that the major model cycles (00z,
    // 06z GFS; 00z ECMWF) will typically have landed.
    private static readonly TimeSpan BackfillAnchor = new(8, 0, 0);

    public PredictCommand(ILogger<PredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string target, string modelVersion, DateOnly? forDate, CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported (got '{Target}')", target);
            return 2;
        }

        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", modelVersion);
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Model version {V} has no per-lead blenders — predict requires a phase 2b model.",
                metadata.Version);
            return 2;
        }

        _log.LogInformation("Using blender version {V} (phase={Phase})", metadata.Version, metadata.Phase);

        // predictionMadeAt = real wall-clock UTC (provenance).
        // anchor = the hour we compute lead offsets from. For a live run, anchor is
        // now truncated to top-of-hour. For --for-date backfill, anchor is the given
        // date at 08:00 UTC.
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = forDate.HasValue
            ? forDate.Value.ToDateTime(TimeOnly.FromTimeSpan(BackfillAnchor), DateTimeKind.Utc)
            : new DateTime(predictionMadeAt.Year, predictionMadeAt.Month, predictionMadeAt.Day,
                           predictionMadeAt.Hour, 0, 0, DateTimeKind.Utc);

        var targets = DefaultLeads.Select(L => (Lead: L, Valid: anchor.AddHours(L))).ToArray();

        _log.LogInformation("Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — targets: {Targets}",
            anchor,
            forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", targets.Select(t => $"{t.Lead}h→{t.Valid:yyyy-MM-dd HH:mm}Z")));

        var perValid = QueryLatestForecastRows(
            _cfg.Storage.ForecastsPath,
            _cfg.Location.Name,
            targets.Min(t => t.Valid),
            targets.Max(t => t.Valid),
            asOfRunTime: anchor,
            ct);

        var ml = new MLContext(seed: 42);
        var predictions = new List<PredictionRow>();

        foreach (var (lead, valid) in targets)
        {
            if (!perValid.TryGetValue(valid, out var pivot))
            {
                _log.LogWarning("Lead {Lead}h: no forecast rows found for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, valid);
                continue;
            }

            var missingNames = Enumerable.Range(0, 6)
                .Where(i => !pivot.Temp[i].HasValue)
                .Select(i => FeatureBuilder.ModelColumns[i].Col)
                .ToArray();
            if (missingNames.Length > 0)
            {
                _log.LogWarning("Lead {Lead}h: missing [{Missing}] for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingNames), valid);
                continue;
            }

            var temps = pivot.Temp.Select(t => t!.Value).ToArray();
            var windDir = pivot.WindDirMean;

            // Era5Temp is ignored at predict time — ML.NET Transform only reads the
            // 13 feature columns. NaN placeholder is consistent with training schema.
            var row = FeatureBuilder.ComposeRow(valid, temps, windDir, era5Temp: double.NaN);
            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.Predict(ml, model, new[] { row });

            predictions.Add(new PredictionRow
            {
                LocationName = _cfg.Location.Name,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendTemperature = yhat[0],
                TempGfs   = temps[0], TempEcmwf = temps[1], TempIcon  = temps[2],
                TempMf    = temps[3], TempUkmo  = temps[4], TempGem   = temps[5],
                RunTimeGfs   = pivot.RunTime[0], RunTimeEcmwf = pivot.RunTime[1],
                RunTimeIcon  = pivot.RunTime[2], RunTimeMf    = pivot.RunTime[3],
                RunTimeUkmo  = pivot.RunTime[4], RunTimeGem   = pivot.RunTime[5],
                TempMean  = row.TempMean,
                TempStd   = row.TempStd,
                TempRange = row.TempRange,
                FeatureVectorHash = HashFeatures(row),
            });

            _log.LogInformation("Lead {Lead}h → blend {Blend:0.00}°C (valid {Valid:yyyy-MM-dd HH:mm}Z, mean-of-models {Mean:0.00}°C, runs: {Runs})",
                lead, yhat[0], valid, row.TempMean,
                FormatRunTimes(pivot.RunTime));
        }

        if (predictions.Count == 0)
        {
            _log.LogError("No predictions produced — forecast tree likely missing recent data. Run `collect` first.");
            return 3;
        }

        await WritePredictionsAsync(predictions, anchor, metadata.Version, ct);
        return 0;
    }

    private async Task WritePredictionsAsync(
        IReadOnlyList<PredictionRow> predictions,
        DateTime anchor,
        string modelVersion,
        CancellationToken ct)
    {
        // Layout: data/predictions/temperature/model_version=<v>/date=<yyyy-MM-dd>/predictions.parquet.
        // Date is the anchor date so re-runs across the same UTC day land in the same file.
        // Dedupe on (PredictionMadeAtUtc, LeadHours): the latest write for a given key wins —
        // matches ObservationsAsync's append-and-dedupe semantics.
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(_cfg.Storage.PredictionsPath,
            "temperature",
            $"model_version={modelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<PredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<PredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<PredictionRow>();

        var merged = existing.Concat(predictions)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.Last())
            .OrderBy(r => r.ValidTimeUtc)
            .ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("Wrote {New} new predictions (file now holds {Total}) → {Path}",
            predictions.Count, merged.Count, outPath);
    }

    // Per-valid-time pivot: nullable temp + nullable RunTime per model slot, index-aligned
    // with FeatureBuilder.ModelColumns.
    private sealed record PivotedRow(double?[] Temp, DateTime?[] RunTime, double WindDirMean);

    private Dictionary<DateTime, PivotedRow> QueryLatestForecastRows(
        string forecastsPath,
        string locationName,
        DateTime earliestValid,
        DateTime latestValid,
        DateTime asOfRunTime,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");

        // Keep per-model rows — do NOT pivot in SQL. Pivoting in .NET lets us carry
        // each model's RunTimeUtc through to the output (provenance). "Latest" per
        // (ValidTime, Model) uses the same window function the trainer uses.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, Temperature2m, WindDirection10m, RunTimeUtc,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND Temperature2m IS NOT NULL
      AND (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
      AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
)
SELECT ValidTimeUtc, Model, RunTimeUtc, Temperature2m, WindDirection10m
FROM latest
WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        // Model-id → slot index (stable per FeatureBuilder.ModelColumns ordering).
        var modelSlot = FeatureBuilder.ModelColumns
            .Select((m, i) => (m.ModelId, Index: i))
            .ToDictionary(x => x.ModelId, x => x.Index);

        var rows = new Dictionary<DateTime, (double?[] Temp, DateTime?[] RunTime, List<double> Winds)>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = reader.GetDateTime(0);
            var model = reader.GetString(1);
            var runTime = reader.GetDateTime(2);
            var temp = reader.IsDBNull(3) ? (double?)null : reader.GetDouble(3);
            var wind = reader.IsDBNull(4) ? (double?)null : reader.GetDouble(4);

            if (!modelSlot.TryGetValue(model, out var slot))
                continue;  // unexpected model id — ignore rather than blow up

            if (!rows.TryGetValue(valid, out var pivot))
            {
                pivot = (new double?[6], new DateTime?[6], new List<double>());
                rows[valid] = pivot;
            }

            pivot.Temp[slot] = temp;
            pivot.RunTime[slot] = runTime;
            if (wind.HasValue) pivot.Winds.Add(wind.Value);
        }

        return rows.ToDictionary(
            kv => kv.Key,
            kv => new PivotedRow(
                kv.Value.Temp,
                kv.Value.RunTime,
                kv.Value.Winds.Count > 0 ? kv.Value.Winds.Average() : double.NaN));
    }

    /// <summary>SHA-256 hex of the 13 feature floats in FeatureNames order.
    /// Deterministic — re-running predict on identical inputs produces identical hashes.</summary>
    public static string HashFeatures(TrainingRow row)
    {
        Span<byte> buf = stackalloc byte[13 * sizeof(float)];
        var i = 0;
        WriteFloat(buf, ref i, row.TempGfs);
        WriteFloat(buf, ref i, row.TempEcmwf);
        WriteFloat(buf, ref i, row.TempIcon);
        WriteFloat(buf, ref i, row.TempMf);
        WriteFloat(buf, ref i, row.TempUkmo);
        WriteFloat(buf, ref i, row.TempGem);
        WriteFloat(buf, ref i, row.TempMean);
        WriteFloat(buf, ref i, row.TempStd);
        WriteFloat(buf, ref i, row.TempRange);
        WriteFloat(buf, ref i, row.HourSin);
        WriteFloat(buf, ref i, row.HourCos);
        WriteFloat(buf, ref i, row.DoySin);
        WriteFloat(buf, ref i, row.DoyCos);

        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(buf, hash);
        return Convert.ToHexString(hash).ToLowerInvariant();

        static void WriteFloat(Span<byte> dst, ref int offset, float v)
        {
            BitConverter.TryWriteBytes(dst.Slice(offset, sizeof(float)), v);
            offset += sizeof(float);
        }
    }

    private static string FormatRunTimes(DateTime?[] runs)
    {
        var parts = new List<string>();
        for (int i = 0; i < 6; i++)
        {
            var label = FeatureBuilder.ModelColumns[i].Col.Replace("temp_", "");
            parts.Add(runs[i].HasValue
                ? $"{label}@{runs[i]:MM-dd HH:mm}Z"
                : $"{label}@—");
        }
        return string.Join(" ", parts);
    }
}
