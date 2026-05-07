using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Models;

namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Shared parquet writer for element predictions. Mirrors
/// <c>TempPredictCommand.WritePredictionsAsync</c>: per-element subtree, dedupe on
/// (PredictionMadeAtUtc, LeadHours), latest write wins.
/// </summary>
public static class ElementPredictWriter
{
    public static async Task WriteAsync(
        ILogger log,
        string predictionsPath,
        string elementModelDirName,
        string modelVersion,
        DateTime anchor,
        IReadOnlyList<ElementPredictionRow> predictions,
        CancellationToken ct)
    {
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(
            predictionsPath,
            elementModelDirName,
            $"model_version={modelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<ElementPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<ElementPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<ElementPredictionRow>();

        // Dedupe key MUST include ValidTimeUtc — without it, the 24 hourly
        // rows-per-lead the predict pipeline emits (since 2026-05-07's hourly
        // fix) all share the same (PredictionMadeAtUtc, LeadHours) tuple and
        // get collapsed to one. Symptom: ElementPredictionRow files held
        // (anchors × leads) rows instead of (anchors × leads × 24), feels-
        // like joined to the smaller surface, home tiles showed feels-like /
        // UTCI on at most a handful of hours per day. Mirrors the same key
        // TempPredictCommand uses for its hourly writer.
        var merged = existing.Concat(predictions)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc)
            .ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        log.LogInformation("  Wrote {New} new predictions (file now holds {Total}) → {Path}",
            predictions.Count, merged.Count, outPath);
    }
}
