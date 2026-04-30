using System.Text.Json;
using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Predict.StartHour;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Joins per-anchor predictions from the Phase-3a precipitation blender
/// (hourly P(wet) at lead L) and the Phase-3b/3d dry-window blender (daily
/// P(∃ N-hour dry block) at lead L) by (Station, WindowHours, LeadHours,
/// TargetDate), then derives the per-start-hour curve via
/// <see cref="StartHourCurveDerivation.Derive"/>. One parquet per
/// (station, window) gets written under
/// <c>data/predictions/dry_window_start_hour/{station}/window_{N}h/model_version=v1/date={anchor}/predictions.parquet</c>.
///
/// Pre-requisite: precip + dry-window predicts have already run for this
/// anchor. The combined predict-and-render workflow runs them in order
/// (precip → dry-window → start-hour) so the inputs land before this
/// command reads them. Either upstream missing → that composite skipped
/// for this anchor, exit code stays 0 unless every composite is empty.
/// </summary>
public sealed class StartHourPredictCommand
{
    public const string PredictionsSubdir = "dry_window_start_hour";
    public const string OutputModelVersion = "v1";

    private readonly ILogger<StartHourPredictCommand> _log;
    private readonly AppConfig _cfg;

    public StartHourPredictCommand(ILogger<StartHourPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var anchorDate = anchor.Date;

        _log.LogInformation("Start-hour predict — anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate})",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live");

        var precipManifest = LoadManifest(modelsRoot, "precipitation");
        var dryManifest = LoadManifest(modelsRoot, "dry_window");
        if (precipManifest is null || dryManifest is null)
        {
            _log.LogError("Missing precipitation or dry_window manifest under {Root}.", modelsRoot);
            return 2;
        }

        var daytime = _cfg.DryWindow.BuildDaytimeWindow();

        // (station, windowHours) → all curve rows for that composite.
        var rowsByComposite = new Dictionary<(string Station, int Window), List<StartHourPredictionRow>>();
        var composites = 0;
        var skipped = 0;

        foreach (var (compositeKey, dryEntry) in dryManifest.Stations)
        {
            ct.ThrowIfCancellationRequested();
            composites++;

            var (station, windowHours) = ParseDryComposite(compositeKey);
            if (station is null)
            {
                _log.LogWarning("  skipping malformed dry-window composite key {Key}", compositeKey);
                skipped++; continue;
            }

            if (!precipManifest.Stations.TryGetValue(station, out var precipEntry)
                || string.IsNullOrEmpty(precipEntry.Current))
            {
                _log.LogWarning("  {Composite}: no precipitation champion for {Station} — skip.",
                    compositeKey, station);
                skipped++; continue;
            }

            var dryVersion = dryEntry.Current;
            var precipVersion = precipEntry.Current;
            if (string.IsNullOrEmpty(dryVersion))
            {
                _log.LogWarning("  {Composite}: no dry-window champion — skip.", compositeKey);
                skipped++; continue;
            }

            var dryRows = await LoadDryWindowRowsAsync(station, windowHours, dryVersion, anchorDate, ct);
            if (dryRows.Count == 0)
            {
                _log.LogInformation("  {Composite}: no dry-window prediction rows at this anchor; skip.", compositeKey);
                skipped++; continue;
            }

            var precipRows = await LoadPrecipRowsAsync(station, precipVersion, anchorDate, ct);
            if (precipRows.Count == 0)
            {
                _log.LogInformation("  {Composite}: no precipitation prediction rows at this anchor; skip.", compositeKey);
                skipped++; continue;
            }

            var composite = (station, windowHours);
            if (!rowsByComposite.TryGetValue(composite, out var bucket))
            {
                bucket = new List<StartHourPredictionRow>();
                rowsByComposite[composite] = bucket;
            }

            foreach (var dry in dryRows)
            {
                var hourlyQ = BuildHourlyQ(precipRows, dry.LeadHours, dry.TargetDateUtc);
                var (startUtc, endUtc) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(dry.TargetDateUtc));

                var derived = StartHourCurveDerivation.Derive(
                    locationName: _cfg.Location.Name,
                    truthStation: station,
                    windowHours: windowHours,
                    startHourVersion: OutputModelVersion,
                    predictionMadeAtUtc: predictionMadeAt,
                    leadHours: dry.LeadHours,
                    targetDateUtc: DateTime.SpecifyKind(dry.TargetDateUtc, DateTimeKind.Utc),
                    daytimeStartUtc: startUtc,
                    daytimeEndUtc: endUtc,
                    hourlyPWet: hourlyQ,
                    dailyProbAnyBlock: dry.ProbHasDryWindow,
                    precipVersion: precipVersion,
                    dryWindowVersion: dryVersion);

                bucket.AddRange(derived);
            }
        }

        if (rowsByComposite.Count == 0)
        {
            _log.LogError("Start-hour: no curves produced — every composite skipped (see warnings above).");
            return 3;
        }

        var totalRows = 0;
        foreach (var ((station, windowHours), rows) in rowsByComposite)
        {
            await WriteAsync(station, windowHours, rows, anchorDate, ct);
            totalRows += rows.Count;
        }

        _log.LogInformation(
            "Start-hour: {Total} rows across {Composites} composites ({Skipped} skipped).",
            totalRows, rowsByComposite.Count, skipped);
        return 0;
    }

    private async Task WriteAsync(
        string station, int windowHours, List<StartHourPredictionRow> rows,
        DateTime anchorDate, CancellationToken ct)
    {
        var dateStr = anchorDate.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(_cfg.Storage.PredictionsPath,
            PredictionsSubdir,
            station,
            $"window_{windowHours}h",
            $"model_version={OutputModelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<StartHourPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<StartHourPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<StartHourPredictionRow>();

        var merged = MergeRows(existing, rows);

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("  wrote {New} {Station}/{Window}h rows (file now holds {Total}) → {Path}",
            rows.Count, station, windowHours, merged.Count, outPath);
    }

    /// <summary>
    /// Concat existing + new curve rows and dedup on
    /// <c>(PredictionMadeAtUtc, TargetDateUtc, LeadHours, StartHourUtc)</c> —
    /// the row's natural identity. Same shape as the feels-like / precip
    /// merge pattern; ValidTime is the start hour for this surface.
    /// </summary>
    internal static List<StartHourPredictionRow> MergeRows(
        IEnumerable<StartHourPredictionRow> existing,
        IEnumerable<StartHourPredictionRow> incoming)
        => existing.Concat(incoming)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.TargetDateUtc, r.LeadHours, r.StartHourUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.TargetDateUtc)
            .ThenBy(r => r.LeadHours)
            .ThenBy(r => r.StartHourUtc)
            .ToList();

    /// <summary>
    /// Per (lead, hour-of-day) on the target date, take the freshest precip
    /// prediction's ProbWet. Returns a hour-of-day → q map for the
    /// requested lead bucket on the requested target date. Empty when the
    /// anchor's precip parquet has no rows for that (lead, target_date).
    /// </summary>
    internal static Dictionary<int, double> BuildHourlyQ(
        IReadOnlyList<PrecipPredictionRow> precip,
        int leadHours,
        DateTime targetDateUtc)
    {
        var q = new Dictionary<int, double>();
        var freshness = new Dictionary<int, DateTime>();
        foreach (var row in precip)
        {
            if (row.LeadHours != leadHours) continue;
            if (row.ValidTimeUtc.Date != targetDateUtc.Date) continue;
            var hour = row.ValidTimeUtc.Hour;
            if (freshness.TryGetValue(hour, out var prev) && prev >= row.PredictionMadeAtUtc) continue;
            q[hour] = row.ProbWet;
            freshness[hour] = row.PredictionMadeAtUtc;
        }
        return q;
    }

    private async Task<List<DryWindowPredictionRow>> LoadDryWindowRowsAsync(
        string station, int windowHours, string version, DateTime anchorDate,
        CancellationToken ct)
    {
        var path = Path.Combine(_cfg.Storage.PredictionsPath,
            "dry_window", station, $"window_{windowHours}h",
            $"model_version={version}",
            $"date={anchorDate:yyyy-MM-dd}",
            "predictions.parquet");
        if (!File.Exists(path)) return new List<DryWindowPredictionRow>();
        var raw = (await ParquetSerializer.DeserializeAsync<DryWindowPredictionRow>(path, cancellationToken: ct)).ToList();
        // Latest PMT per (lead, target_date) — multiple cycles can land in this
        // partition during a UTC day; pick the freshest per composite cell.
        return raw
            .GroupBy(r => (r.LeadHours, r.TargetDateUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .ToList();
    }

    private async Task<List<PrecipPredictionRow>> LoadPrecipRowsAsync(
        string station, string version, DateTime anchorDate, CancellationToken ct)
    {
        var path = Path.Combine(_cfg.Storage.PredictionsPath,
            "precipitation", station,
            $"model_version={version}",
            $"date={anchorDate:yyyy-MM-dd}",
            "predictions.parquet");
        if (!File.Exists(path)) return new List<PrecipPredictionRow>();
        return (await ParquetSerializer.DeserializeAsync<PrecipPredictionRow>(path, cancellationToken: ct)).ToList();
    }

    private static ModelArtifact.Manifest? LoadManifest(string modelsRoot, string target)
    {
        var path = Path.Combine(modelsRoot, target, ModelArtifact.ManifestFileName);
        if (!File.Exists(path)) return null;
        try
        {
            return JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path));
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Parse <c>"ea_bellever_dartmoor/window_6h"</c> → (station,
    /// windowHours). Returns (null, 0) on shape mismatch so the caller can
    /// log + skip rather than crash.
    /// </summary>
    internal static (string? Station, int WindowHours) ParseDryComposite(string compositeKey)
    {
        var slash = compositeKey.IndexOf('/');
        if (slash <= 0) return (null, 0);
        var station = compositeKey[..slash];
        var rest = compositeKey[(slash + 1)..];
        if (!rest.StartsWith("window_") || !rest.EndsWith("h")) return (null, 0);
        var hoursStr = rest["window_".Length..^1];
        return int.TryParse(hoursStr, out var hours) && hours > 0
            ? (station, hours)
            : (null, 0);
    }
}
