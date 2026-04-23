using System.Globalization;
using System.Text.RegularExpressions;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended P(dry window) forecasts for each (station, window, lead ∈
/// {24, 48, 72}) blender recorded in the dry_window manifest. Parallels
/// <see cref="PrecipPredictCommand"/> but at day granularity: each prediction
/// covers one UTC target day (anchor_date + 1/2/3 days).
///
/// Feature row is built via <see cref="DryWindowFeatureBuilder.ComposeRow"/> so
/// training and inference share a single composition path. The training-time
/// SQL pulls <c>RunTimeSource='offset_day'</c>; predict uses live-cycle rows via
/// <see cref="PredictForecastFilters.LiveCycleAsOf"/>. The feature-row shape is
/// identical — this is a known train/predict distribution difference documented
/// in the phase-3b audit.
/// </summary>
public sealed class DryWindowPredictCommand
{
    private readonly ILogger<DryWindowPredictCommand> _log;
    private readonly AppConfig _cfg;

    public DryWindowPredictCommand(ILogger<DryWindowPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string stationArg, string windowArg, string modelVersion, DateOnly? forDate, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var manifestPath = Path.Combine(modelsRoot, "dry_window", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            _log.LogError("No dry_window manifest at {Path}. Train first.", manifestPath);
            return 2;
        }
        var manifest = System.Text.Json.JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(manifestPath))
            ?? throw new InvalidOperationException("Failed to parse dry_window manifest.");

        var entries = FilterEntries(manifest, stationArg, windowArg);
        if (entries.Count == 0)
        {
            _log.LogError("No manifest entries match station='{Station}' window='{Window}'.", stationArg, windowArg);
            return 2;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var anchorDate = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc);
        var targets = new[]
        {
            (Lead: 24, Date: anchorDate.AddDays(1)),
            (Lead: 48, Date: anchorDate.AddDays(2)),
            (Lead: 72, Date: anchorDate.AddDays(3)),
        };

        _log.LogInformation(
            "Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}). Targets: {T}. Entries: {N}",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", targets.Select(t => $"{t.Lead}h→{t.Date:yyyy-MM-dd}")),
            entries.Count);

        var earliest = targets.Min(t => t.Date);
        var latest = targets.Max(t => t.Date).AddHours(23);
        var perDayPerModel = QueryForecastDaysByTarget(
            _cfg.Storage.ForecastsPath, _cfg.Location.Name, earliest, latest, anchor, ct);

        var anyWritten = false;
        foreach (var (compositeKey, station) in entries)
        {
            ct.ThrowIfCancellationRequested();
            var parsed = ParseCompositeKey(compositeKey);
            if (parsed is null)
            {
                _log.LogWarning("Skipping unparsable manifest key '{Key}'.", compositeKey);
                continue;
            }
            var (stationSlug, windowHours) = parsed.Value;

            var versionDir = ModelArtifact.ResolveStationVersionDir(
                modelsRoot, "dry_window", compositeKey, modelVersion);
            var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
            var climPath = Path.Combine(versionDir, "dry_window_climatology.json");
            if (!File.Exists(climPath))
            {
                _log.LogWarning("Missing climatology at {P}. Skipping {K}.", climPath, compositeKey);
                continue;
            }
            var climatology = DryWindowClimatology.LoadFrom(climPath);

            _log.LogInformation("{Key}: model version {V}, window {W}h", compositeKey, metadata.Version, windowHours);

            var ml = new MLContext(seed: 42);
            var predictions = new List<DryWindowPredictionRow>();

            foreach (var (lead, targetDate) in targets)
            {
                ct.ThrowIfCancellationRequested();
                if (!perDayPerModel.TryGetValue(DateOnly.FromDateTime(targetDate), out var modelDayList))
                {
                    _log.LogWarning("{Key} lead {Lead}h: no forecast rows for {D:yyyy-MM-dd}; skipping.",
                        compositeKey, lead, targetDate);
                    continue;
                }
                if (!modelDayList.Any(d => d is { AnyPresent: true }))
                {
                    _log.LogWarning("{Key} lead {Lead}h: forecast rows exist for {D:yyyy-MM-dd} but no model is populated; skipping.",
                        compositeKey, lead, targetDate);
                    continue;
                }

                var row = DryWindowFeatureBuilder.ComposeRow(
                    DateOnly.FromDateTime(targetDate),
                    windowHours,
                    modelDayList,
                    label: false,
                    truthMmDay: 0.0);

                var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
                var probs = DryWindowTrainer.PredictProbability(ml, model, new[] { row });
                var climProb = climatology.Predict(targetDate);

                predictions.Add(new DryWindowPredictionRow
                {
                    LocationName = _cfg.Location.Name,
                    TruthStation = stationSlug,
                    WindowHours = windowHours,
                    ModelVersion = metadata.Version,
                    PredictionMadeAtUtc = predictionMadeAt,
                    TargetDateUtc = targetDate,
                    LeadHours = lead,
                    ProbHasDryWindow = probs[0],
                    ClimatologyProbHasDryWindow = climProb,
                    AgreementHasDryWindow = NanToNull(row.AgreementHasDryWindow),
                    PrecipSumMean = NanToNull(row.PrecipSumMean),
                    LongestDryRunMean = NanToNull(row.LongestDryRunMean),
                    WetHourCountMean = NanToNull(row.WetHourCountMean),
                    HasDryWindowGfs   = NanToNull(row.HasDryWindowGfs),
                    HasDryWindowEcmwf = NanToNull(row.HasDryWindowEcmwf),
                    HasDryWindowIcon  = NanToNull(row.HasDryWindowIcon),
                    HasDryWindowMf    = NanToNull(row.HasDryWindowMf),
                    HasDryWindowUkmo  = NanToNull(row.HasDryWindowUkmo),
                    HasDryWindowGem   = NanToNull(row.HasDryWindowGem),
                    PrecipSumGfs   = NanToNull(row.PrecipSumGfs),
                    PrecipSumEcmwf = NanToNull(row.PrecipSumEcmwf),
                    PrecipSumIcon  = NanToNull(row.PrecipSumIcon),
                    PrecipSumMf    = NanToNull(row.PrecipSumMf),
                    PrecipSumUkmo  = NanToNull(row.PrecipSumUkmo),
                    PrecipSumGem   = NanToNull(row.PrecipSumGem),
                    FeatureVectorHash = HashFeatures(row),
                });

                _log.LogInformation(
                    "  lead {Lead}h ({Date:yyyy-MM-dd}) → P(dry {W}h)={P:0.000} (clim {C:0.000}, agreement {A:0.00})",
                    lead, targetDate, windowHours, probs[0], climProb, row.AgreementHasDryWindow);
            }

            if (predictions.Count == 0)
            {
                _log.LogWarning("{Key}: no predictions produced.", compositeKey);
                continue;
            }

            await WritePredictionsAsync(predictions, stationSlug, windowHours, anchorDate, metadata.Version, ct);
            anyWritten = true;
        }

        return anyWritten ? 0 : 3;
    }

    private List<KeyValuePair<string, ModelArtifact.StationEntry>> FilterEntries(
        ModelArtifact.Manifest manifest, string stationArg, string windowArg)
    {
        var result = new List<KeyValuePair<string, ModelArtifact.StationEntry>>();
        foreach (var kv in manifest.Stations)
        {
            var parsed = ParseCompositeKey(kv.Key);
            if (parsed is null) continue;
            var (slug, window) = parsed.Value;

            if (!string.Equals(stationArg, "all", StringComparison.OrdinalIgnoreCase))
            {
                if (!SlugMatches(slug, stationArg)) continue;
            }
            if (!string.Equals(windowArg, "all", StringComparison.OrdinalIgnoreCase))
            {
                if (!int.TryParse(windowArg, out var w) || w != window) continue;
            }
            if (string.IsNullOrWhiteSpace(kv.Value.Current)) continue;
            result.Add(kv);
        }
        return result;
    }

    private static bool SlugMatches(string slug, string arg)
    {
        if (slug.Equals(arg, StringComparison.OrdinalIgnoreCase)) return true;
        var slugWithoutPrefix = slug.StartsWith("ea_") ? slug[3..] : slug;
        if (slugWithoutPrefix.Equals(arg, StringComparison.OrdinalIgnoreCase)) return true;
        var derived = Slugify(arg);
        return slug.Equals("ea_" + derived, StringComparison.OrdinalIgnoreCase)
            || slugWithoutPrefix.Equals(derived, StringComparison.OrdinalIgnoreCase);
    }

    private static (string StationSlug, int WindowHours)? ParseCompositeKey(string key)
    {
        var m = Regex.Match(key, @"^(?<slug>[^/]+)/window_(?<w>\d+)h$");
        if (!m.Success) return null;
        return (m.Groups["slug"].Value, int.Parse(m.Groups["w"].Value, CultureInfo.InvariantCulture));
    }

    private Dictionary<DateOnly, List<DryWindowFeatureBuilder.ForecastDay?>> QueryForecastDaysByTarget(
        string forecastsPath,
        string locationName,
        DateTime earliestValid,
        DateTime latestValid,
        DateTime anchorAsOfRunTime,
        CancellationToken ct)
    {
        var glob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var filter = PredictForecastFilters.LiveCycleAsOf(locationName, anchorAsOfRunTime, earliestValid, latestValid);

        // Latest live-cycle row per (valid_time, model). Mirrors PrecipPredictCommand.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model,
           Precipitation, PrecipitationProbability,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           Cape, WindSpeed10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE {filter}
)
SELECT ValidTimeUtc, Model,
       Precipitation, PrecipitationProbability,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       Cape, WindSpeed10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = DryWindowFeatureBuilder.ModelIds
            .Select((id, i) => (id, i))
            .ToDictionary(x => x.id, x => x.i);

        // Per-target-date buckets; each carries 6 ForecastDay slots aligned with ModelIds.
        var byDate = new Dictionary<DateOnly, List<DryWindowFeatureBuilder.ForecastDay?>>();

        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = DateTime.SpecifyKind(r.GetDateTime(0), DateTimeKind.Utc);
            var model = r.GetString(1);
            if (!slotByModel.TryGetValue(model, out var slot)) continue;

            var date = DateOnly.FromDateTime(valid);
            if (!byDate.TryGetValue(date, out var list))
            {
                list = new List<DryWindowFeatureBuilder.ForecastDay?>(6) { null, null, null, null, null, null };
                byDate[date] = list;
            }
            if (list[slot] is null) list[slot] = new DryWindowFeatureBuilder.ForecastDay();

            var fr = new DryWindowFeatureBuilder.ForecastRow(
                valid, model,
                r.IsDBNull(2) ? null : r.GetDouble(2),
                r.IsDBNull(3) ? null : r.GetDouble(3),
                r.IsDBNull(4) ? null : r.GetDouble(4),
                r.IsDBNull(5) ? null : r.GetDouble(5),
                r.IsDBNull(6) ? null : r.GetDouble(6),
                r.IsDBNull(7) ? null : r.GetDouble(7),
                r.IsDBNull(8) ? null : r.GetDouble(8),
                r.IsDBNull(9) ? null : r.GetDouble(9),
                r.IsDBNull(10) ? null : r.GetDouble(10),
                r.IsDBNull(11) ? null : r.GetDouble(11));
            list[slot]!.SetHour(valid.Hour, fr);
        }

        return byDate;
    }

    private async Task WritePredictionsAsync(
        IReadOnlyList<DryWindowPredictionRow> predictions,
        string stationSlug,
        int windowHours,
        DateTime anchorDate,
        string modelVersion,
        CancellationToken ct)
    {
        var dateStr = anchorDate.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(_cfg.Storage.PredictionsPath,
            "dry_window",
            stationSlug,
            $"window_{windowHours}h",
            $"model_version={modelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<DryWindowPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<DryWindowPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<DryWindowPredictionRow>();

        var merged = existing.Concat(predictions)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.TargetDateUtc)
            .ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("Wrote {N} new predictions (file now holds {T}) → {Path}",
            predictions.Count, merged.Count, outPath);
    }

    private static double? NanToNull(float v) => float.IsNaN(v) ? null : v;

    /// <summary>SHA-256 hex of the 53 feature floats in DryWindowFeatureBuilder order.</summary>
    public static string HashFeatures(DryWindowTrainingRow row) => FeatureHashing.HashFloats(new[]
    {
        row.PrecipSumGfs, row.PrecipSumEcmwf, row.PrecipSumIcon, row.PrecipSumMf, row.PrecipSumUkmo, row.PrecipSumGem,
        row.PrecipMaxHourGfs, row.PrecipMaxHourEcmwf, row.PrecipMaxHourIcon, row.PrecipMaxHourMf, row.PrecipMaxHourUkmo, row.PrecipMaxHourGem,
        row.WetHourCountGfs, row.WetHourCountEcmwf, row.WetHourCountIcon, row.WetHourCountMf, row.WetHourCountUkmo, row.WetHourCountGem,
        row.LongestDryRunGfs, row.LongestDryRunEcmwf, row.LongestDryRunIcon, row.LongestDryRunMf, row.LongestDryRunUkmo, row.LongestDryRunGem,
        row.HasDryWindowGfs, row.HasDryWindowEcmwf, row.HasDryWindowIcon, row.HasDryWindowMf, row.HasDryWindowUkmo, row.HasDryWindowGem,
        row.ProbMaxGfs, row.ProbMaxEcmwf, row.ProbMaxIcon, row.ProbMaxMf, row.ProbMaxUkmo, row.ProbMaxGem,
        row.PrecipSumMean, row.PrecipSumStd, row.PrecipSumMax,
        row.AgreementHasDryWindow, row.LongestDryRunMean, row.WetHourCountMean,
        row.RhMean, row.RhMin, row.DewDepressionMax,
        row.CloudLowMean, row.CloudMidMean, row.CloudHighMean,
        row.CapeMax, row.WindMean, row.WindMax,
        row.DoySin, row.DoyCos,
    });

    private static string Slugify(string s) => s.ToLowerInvariant()
        .Replace(' ', '_').Replace('-', '_').Replace(',', '_');
}
