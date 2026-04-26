using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Predict.Utci;

namespace WeatherBlend.Commands;

/// <summary>
/// Derives Universal Thermal Climate Index (UTCI) per (valid_time, lead) by
/// joining the latest blender outputs for temperature + humidity + wind +
/// shortwave-radiation + cloud-cover at a shared anchor date.
///
/// Pre-requisite: the five input <c>predict --target {x}</c> commands must
/// have already been run for the same anchor (typically scheduled in the
/// same cycle). UTCI's output goes to
/// <c>data/predictions/utci/model_version=v1/date=&lt;anchor&gt;/predictions.parquet</c>.
/// </summary>
public sealed class UtciPredictCommand
{
    private readonly ILogger<UtciPredictCommand> _log;
    private readonly AppConfig _cfg;

    public UtciPredictCommand(ILogger<UtciPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        _log.LogInformation("UTCI predict — anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate})",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live");

        var rows = await UtciPredictPipeline.ComposeForAnchorAsync(
            _log, _cfg.Location.Name, _cfg.Storage.PredictionsPath, modelsRoot,
            anchor, predictionMadeAt, ct);

        if (rows.Count == 0)
        {
            _log.LogError("UTCI: no rows produced — see warnings above (likely missing input predictions).");
            return 3;
        }

        await WriteAsync(rows, anchor, ct);
        foreach (var r in rows)
            _log.LogInformation("  +{Lead}h @ {Valid:yyyy-MM-dd HH:mm}Z  Ta={Ta:0.0}°C  Tmrt={Tmrt:0.0}°C  va10={Va:0.0}m/s  UTCI={U:0.0}°C  ({Band})",
                r.LeadHours, r.ValidTimeUtc, r.TemperatureC, r.MeanRadiantTemperatureC, r.WindSpeed10mMs, r.UtciC, r.Band);
        return 0;
    }

    private async Task WriteAsync(List<UtciPredictionRow> rows, DateTime anchor, CancellationToken ct)
    {
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(
            _cfg.Storage.PredictionsPath, "utci",
            $"model_version={UtciPredictPipeline.OutputModelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<UtciPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<UtciPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<UtciPredictionRow>();

        var merged = existing.Concat(rows)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("Wrote {New} new UTCI rows (file now holds {Total}) → {Path}",
            rows.Count, merged.Count, outPath);
    }
}
