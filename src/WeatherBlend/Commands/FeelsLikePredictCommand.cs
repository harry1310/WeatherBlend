using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Predict.FeelsLike;

namespace WeatherBlend.Commands;

/// <summary>
/// Derives both "feels-like" indices — UTCI (Bröde 2012 biothermal) and
/// Steadman 1994 apparent temperature — per (valid_time, lead) by joining
/// the latest blender outputs for temperature + humidity + wind +
/// shortwave-radiation + cloud-cover at a shared anchor date.
///
/// Pre-requisite: the five input <c>predict --target {x}</c> commands must
/// have already been run for the same anchor (typically scheduled in the
/// same cycle). Output goes to
/// <c>data/predictions/feels_like/model_version=v1/date=&lt;anchor&gt;/predictions.parquet</c>.
/// </summary>
public sealed class FeelsLikePredictCommand
{
    /// <summary>Hive-partition root segment for the parquet tree this command writes to.</summary>
    public const string PredictionsSubdir = "feels_like";

    private readonly ILogger<FeelsLikePredictCommand> _log;
    private readonly AppConfig _cfg;

    public FeelsLikePredictCommand(ILogger<FeelsLikePredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        _log.LogInformation("Feels-like predict — anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate})",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live");

        var rows = await FeelsLikePredictPipeline.ComposeForAnchorAsync(
            _log, _cfg.Location.Name, _cfg.Storage.PredictionsPath, modelsRoot,
            anchor, predictionMadeAt, ct);

        if (rows.Count == 0)
        {
            _log.LogError("Feels-like: no rows produced — see warnings above (likely missing input predictions).");
            return 3;
        }

        await WriteAsync(rows, anchor, ct);
        foreach (var r in rows)
            _log.LogInformation("  +{Lead}h @ {Valid:yyyy-MM-dd HH:mm}Z  Ta={Ta:0.0}°C  Tmrt={Tmrt:0.0}°C  va10={Va:0.0}m/s  UTCI={U:0.0}°C  Apparent={A:0.0}°C  ({Band})",
                r.LeadHours, r.ValidTimeUtc, r.TemperatureC, r.MeanRadiantTemperatureC,
                r.WindSpeed10mMs, r.UtciC, r.ApparentTemperatureC, r.Band);
        return 0;
    }

    private async Task WriteAsync(List<FeelsLikePredictionRow> rows, DateTime anchor, CancellationToken ct)
    {
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(
            _cfg.Storage.PredictionsPath, PredictionsSubdir,
            $"model_version={FeelsLikePredictPipeline.OutputModelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<FeelsLikePredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<FeelsLikePredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<FeelsLikePredictionRow>();

        var merged = existing.Concat(rows)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("Wrote {New} new feels-like rows (file now holds {Total}) → {Path}",
            rows.Count, merged.Count, outPath);
    }
}
