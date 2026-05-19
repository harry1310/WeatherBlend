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

    // --location resolves into _activeLocation at RunAsync entry; the
    // feels-like composition reads that location's name + lat/lon.
    // Defaults to the primary location (Phase B, commit 4).
    private Config.LocationConfig _activeLocation;

    public FeelsLikePredictCommand(ILogger<FeelsLikePredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
        _activeLocation = cfg.Location;
    }

    public Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
        => RunAsync(forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        var (resolvedLocation, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (resolvedLocation is null) return locRc;
        _activeLocation = resolvedLocation;

        var modelsRoot = _cfg.Storage.ModelsPath;
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        _log.LogInformation("Feels-like predict — anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate})",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live");

        var rows = await FeelsLikePredictPipeline.ComposeForAnchorAsync(
            _log, _activeLocation.Name, _activeLocation.Latitude, _activeLocation.Longitude,
            _cfg.Storage.PredictionsPath, modelsRoot, anchor, predictionMadeAt, ct);

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
        var outPath = Path.Combine(
            _cfg.Storage.PredictionsPath, PredictionsSubdir,
            $"model_version={FeelsLikePredictPipeline.OutputModelVersion}",
            $"date={dateStr}",
            "predictions.parquet");

        var total = await PredictionParquetWriter.WriteAsync(outPath, rows, MergeRows, ct);
        _log.LogInformation("Wrote {New} new feels-like rows (file now holds {Total}) → {Path}",
            rows.Count, total, outPath);
    }

    /// <summary>
    /// Concat existing + new feels-like rows, dedup, and return in
    /// (ValidTime, Lead) order. The dedup key is
    /// <c>(PredictionMadeAtUtc, LeadHours, ValidTimeUtc)</c>, deliberately
    /// including <c>ValidTimeUtc</c> because a single feels-like run produces
    /// one row per (Lead, ValidTime) pair found in the joined inputs —
    /// typically multiple ValidTimes per Lead now that the 2h predict cadence
    /// puts several anchors on disk per UTC day. Keying on Lead alone (as the
    /// other predict writers do) collapses those siblings under the shared
    /// PredictionMadeAtUtc and silently drops every ValidTime but one — which
    /// stripped the home-card chip from every card whose anchor wasn't the
    /// surviving one.
    /// </summary>
    internal static List<FeelsLikePredictionRow> MergeRows(
        IEnumerable<FeelsLikePredictionRow> existing,
        IEnumerable<FeelsLikePredictionRow> incoming)
        => PredictionParquetWriter.Merge(existing, incoming,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours));
}
