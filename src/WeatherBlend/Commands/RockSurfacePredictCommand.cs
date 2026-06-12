using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Predict.Surface;

namespace WeatherBlend.Commands;

/// <summary>
/// Derives rock surface temperature + condensation (Phase P1) by integrating the
/// Force-Restore ODE over a contiguous hourly forcing trajectory assembled from
/// the temperature + element blends (forward) and recent NWP best-estimate
/// (dew point + spin-up). See <see cref="RockSurfacePredictPipeline"/>.
///
/// Per-location gating, two layers: a location with <c>rockSurface.enabled:
/// false</c> (Sennen until its cliff-aware physics ships — see
/// docs/SENNEN_ROCK_TEMP_PLAN.md) is intentionally dark and exits 0; a location
/// missing an element-blender champion (Membury) returns no rows and
/// soft-skips (exit 3). Output:
/// <c>data/predictions/rock_surface/model_version=v1/date=&lt;anchor&gt;/predictions.parquet</c>.
/// Run AFTER the temperature + element predicts for the same anchor.
/// </summary>
public sealed class RockSurfacePredictCommand
{
    /// <summary>Hive-partition root segment for the parquet tree this command writes to.</summary>
    public const string PredictionsSubdir = "rock_surface";

    private readonly ILogger<RockSurfacePredictCommand> _log;
    private readonly AppConfig _cfg;

    public RockSurfacePredictCommand(ILogger<RockSurfacePredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
        => RunAsync(forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        var (location, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (location is null) return locRc;

        var rockCfg = _cfg.RockSurface.ResolveFor(location.RockSurface);
        if (!rockCfg.Enabled)
        {
            _log.LogInformation("Rock-surface: disabled for {Loc} (rockSurface.enabled=false) — intentional skip.", location.Name);
            return 0;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        _log.LogInformation("Rock-surface predict — {Loc} anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate})",
            location.Name, anchor, forDate?.ToString("yyyy-MM-dd") ?? "live");

        var rows = await RockSurfacePredictPipeline.ComposeForAnchorAsync(
            _log, location, _cfg.Storage.PredictionsPath, _cfg.Storage.ModelsPath,
            _cfg.Storage.ForecastsPath, rockCfg, anchor, predictionMadeAt, ct);

        if (rows.Count == 0)
        {
            _log.LogError("Rock-surface: no rows produced for {Loc} — see warnings above (missing blend inputs / NWP forcing).", location.Name);
            return 3;
        }

        await WriteAsync(rows, anchor, ct);

        var greasy = rows.Count(r => r.GreasinessStatus != RockSurfacePhysics.StatusDry);
        _log.LogInformation("Rock-surface: wrote {N} hours for {Loc} ({Greasy} flagged greasy/condensation). Ts range {Min:0.0}..{Max:0.0}°C.",
            rows.Count, location.Name, greasy, rows.Min(r => r.RockSurfaceTempC), rows.Max(r => r.RockSurfaceTempC));
        return 0;
    }

    private async Task WriteAsync(List<RockSurfacePredictionRow> rows, DateTime anchor, CancellationToken ct)
    {
        var outPath = Path.Combine(
            _cfg.Storage.PredictionsPath, PredictionsSubdir,
            $"model_version={RockSurfacePredictPipeline.OutputModelVersion}",
            $"date={anchor:yyyy-MM-dd}",
            "predictions.parquet");

        var total = await PredictionParquetWriter.WriteAsync(outPath, rows, MergeRows, ct);
        _log.LogInformation("  Wrote {New} new rock-surface rows (file now holds {Total}) → {Path}",
            rows.Count, total, outPath);
    }

    /// <summary>Dedup key includes ValidTimeUtc (one row per (Lead, ValidTime)
    /// per cycle), mirroring the feels-like writer — keying on Lead alone would
    /// collapse the hourly trajectory to one row per lead.</summary>
    internal static List<RockSurfacePredictionRow> MergeRows(
        IEnumerable<RockSurfacePredictionRow> existing,
        IEnumerable<RockSurfacePredictionRow> incoming)
        => PredictionParquetWriter.Merge(existing, incoming,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours));
}
