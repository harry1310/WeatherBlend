using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Models;

namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Shared parquet writer for element predictions. Mirrors
/// <c>PredictCommand.WritePredictionsAsync</c>: per-element subtree, dedupe on
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

        var merged = existing.Concat(predictions)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc)
            .ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        log.LogInformation("  Wrote {New} new predictions (file now holds {Total}) → {Path}",
            predictions.Count, merged.Count, outPath);
    }
}
