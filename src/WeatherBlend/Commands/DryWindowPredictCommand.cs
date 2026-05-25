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
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

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

    public Task<int> RunAsync(string stationArg, string windowArg, string modelVersion, DateOnly? forDate, CancellationToken ct)
        => RunAsync(stationArg, windowArg, modelVersion, forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(string stationArg, string windowArg, string modelVersion, DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        var (location, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (location is null) return locRc;

        var modelsRoot = _cfg.Storage.ModelsPath;
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
            _cfg.Storage.ForecastsPath, location.Name, earliest, latest, anchor, ct);

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

            // Active versions for this composite — 3b champion plus 3g challenger.
            // When the user pins --model-version we honour it and skip the manifest list.
            var activeVersions = string.Equals(modelVersion, "current", StringComparison.OrdinalIgnoreCase)
                ? ModelArtifact.ResolveStationActive(modelsRoot, "dry_window", compositeKey)
                : new[] { modelVersion };

            if (activeVersions.Count == 0)
            {
                _log.LogWarning("{Key}: no active versions in manifest; skipping.", compositeKey);
                continue;
            }

            foreach (var versionName in activeVersions)
            {
                ct.ThrowIfCancellationRequested();
                if (await RunCompositeVersionAsync(
                    modelsRoot, compositeKey, stationSlug, windowHours, versionName,
                    targets, perDayPerModel, anchorDate, predictionMadeAt, location, ct))
                {
                    anyWritten = true;
                }
            }
        }

        return anyWritten ? 0 : 3;
    }

    private async Task<bool> RunCompositeVersionAsync(
        string modelsRoot, string compositeKey, string stationSlug, int windowHours,
        string versionName,
        IReadOnlyList<(int Lead, DateTime Date)> targets,
        IReadOnlyDictionary<DateOnly, List<DryWindowFeatureBuilder.ForecastDay?>> perDayPerModel,
        DateTime anchorDate, DateTime predictionMadeAt,
        LocationConfig location, CancellationToken ct)
    {
        var versionDir = Path.Combine(modelsRoot, "dry_window", compositeKey, versionName);
        if (!Directory.Exists(versionDir))
        {
            _log.LogWarning("{Key}: version dir missing → {Dir}; skipping.", compositeKey, versionDir);
            return false;
        }
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        // Phase A multi-location safety: metadata.LocationName is
        // [JsonRequired] so a missing field already threw at deserialise.
        if (!string.Equals(metadata.LocationName, location.Name, StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError(
                "{Key} bundle {V} was trained on location '{Trained}' but predict is using NWP from '{Active}' — refusing to score.",
                compositeKey, versionName, metadata.LocationName, location.Name);
            return false;
        }
        var climPath = Path.Combine(versionDir, "dry_window_climatology.json");
        if (!File.Exists(climPath))
        {
            _log.LogWarning("{Key} {V}: missing climatology at {P}; skipping.", compositeKey, versionName, climPath);
            return false;
        }
        var climatology = DryWindowClimatology.LoadFrom(climPath);

        // Phase 3d-calibrated handling removed 2026-04-29 — PAV calibration on
        // dry-window didn't move test Brier vs raw 3b. Old 3d-calibrated
        // artefacts on R2 are inert; if any persist in a manifest's Active
        // list they should be dropped.

        _log.LogInformation("{Key}: version {V} (phase {P}), window {W}h",
            compositeKey, metadata.Version, metadata.Phase, windowHours);

        // Phase 3f — local bake-off experiment only, NOT a site-rendered
        // phase. No C# predict path exists for 3f (its bundle artefact is
        // a TorchSharp MLP .pt rather than a LightGBM .zip). If a stale
        // Phases 3g / 3j / 3n / 3s retired 2026-05-25 in model-cleanup
        // Phase 1 — predict dispatch removed. Surviving R2 bundles with
        // those phase tags are filtered out by phases.yaml gating before
        // we reach this point.

        // Per-lead BlenderSpec lives in feature_schema.json.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = WeatherBlend.Train.TempFeatureBuilder.CanonicalModelOrder.ToList();

        var ml = new MLContext(seed: 42);
        var predictions = new List<DryWindowPredictionRow>();

        foreach (var (lead, targetDate) in targets)
        {
            ct.ThrowIfCancellationRequested();
            if (!perDayPerModel.TryGetValue(DateOnly.FromDateTime(targetDate), out var modelDayList))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h: no forecast rows for {D:yyyy-MM-dd}; skipping.",
                    compositeKey, versionName, lead, targetDate);
                continue;
            }
            if (!modelDayList.Any(d => d is { AnyPresent: true }))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h: forecast rows exist for {D:yyyy-MM-dd} but no model is populated; skipping.",
                    compositeKey, versionName, lead, targetDate);
                continue;
            }
            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h: no BlenderSpec in feature_schema.json; skipping.",
                    compositeKey, versionName, lead);
                continue;
            }

            // Filter the canonical 6-slot model-day list down to spec.Models, in spec order.
            var specModelDays = new List<DryWindowFeatureBuilder.ForecastDay?>(spec.Models.Count);
            foreach (var modelId in spec.Models)
            {
                var ci = canonOrder.IndexOf(modelId);
                specModelDays.Add(modelDayList[ci]);
            }

            var (startHour, endHour) = _cfg.DryWindow.BuildDaytimeWindow()
                .UtcHourRangeFor(DateOnly.FromDateTime(targetDate));
            var row = DryWindowFeatureBuilder.ComposeRow(
                spec,
                DateOnly.FromDateTime(targetDate),
                windowHours,
                specModelDays,
                label: false,
                truthMmDay: 0.0,
                startHour: startHour,
                endHour: endHour);

            double rawProb;
            {
                var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
                var probs = DryWindowTrainer.PredictVectorProbability(ml, loadedModel, spec, new[] { row });
                rawProb = probs[0];
            }

            // Apply isotonic calibration if the artefact carries one — older
            // pre-calibration models simply return raw probs unchanged. For
            // 3e/4h the calibrator was fitted to the PRODUCT against 4h truth
            // at training time, so applying it here is the right end-to-end
            // calibration step.
            var calibrator = ModelArtifact.TryLoadLeadCalibrator(versionDir, lead);
            var prob = calibrator is null ? rawProb : calibrator.Predict(rawProb);
            var climProb = climatology.Predict(targetDate);

            // Build per-model output fields: populate only spec.Models, null elsewhere.
            // Sized from DryWindowPredictionRow.PerModelFieldCount (8: Gfs..Gem + Aifs + Jma).
            var perModelHasDry  = new double?[DryWindowPredictionRow.PerModelFieldCount];
            var perModelSum     = new double?[DryWindowPredictionRow.PerModelFieldCount];
            for (int i = 0; i < spec.Models.Count; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= DryWindowPredictionRow.PerModelFieldCount) continue;
                var hasDry = row.Features[spec.IndexOf($"has_dry_window_{WeatherBlend.Train.TempFeatureBuilder.ShortName(spec.Models[i])}")];
                var sum    = row.Features[spec.IndexOf($"precip_sum_{WeatherBlend.Train.TempFeatureBuilder.ShortName(spec.Models[i])}")];
                perModelHasDry[ci] = NanToNullDouble(hasDry);
                perModelSum[ci]    = NanToNullDouble(sum);
            }

            predictions.Add(new DryWindowPredictionRow
            {
                LocationName = location.Name,
                TruthStation = stationSlug,
                WindowHours = windowHours,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                TargetDateUtc = targetDate,
                LeadHours = lead,
                ProbHasDryWindow = prob,
                ClimatologyProbHasDryWindow = climProb,
                AgreementHasDryWindow = NanToNullDouble(row.Features[spec.IndexOf("agreement_has_dry_window")]),
                PrecipSumMean = NanToNullDouble(row.Features[spec.IndexOf("precip_sum_mean")]),
                LongestDryRunMean = NanToNullDouble(row.Features[spec.IndexOf("longest_dry_run_mean")]),
                WetHourCountMean = NanToNullDouble(row.Features[spec.IndexOf("wet_hour_count_mean")]),
                HasDryWindowGfs   = perModelHasDry[0], HasDryWindowEcmwf = perModelHasDry[1],
                HasDryWindowIcon  = perModelHasDry[2], HasDryWindowMf    = perModelHasDry[3],
                HasDryWindowUkmo  = perModelHasDry[4], HasDryWindowGem   = perModelHasDry[5],
                HasDryWindowAifs  = perModelHasDry[6], HasDryWindowJma   = perModelHasDry[7],
                PrecipSumGfs   = perModelSum[0], PrecipSumEcmwf = perModelSum[1],
                PrecipSumIcon  = perModelSum[2], PrecipSumMf    = perModelSum[3],
                PrecipSumUkmo  = perModelSum[4], PrecipSumGem   = perModelSum[5],
                PrecipSumAifs  = perModelSum[6], PrecipSumJma   = perModelSum[7],
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
                ConformalSetTag = ModelArtifact.PredictConformalIfPresent(versionDir, lead, prob),
            });

            _log.LogInformation(
                "  lead {Lead}h ({Date:yyyy-MM-dd}) → P(dry {W}h)={P:0.000} (clim {C:0.000}, agreement {A:0.00})",
                lead, targetDate, windowHours, prob, climProb,
                row.Features[spec.IndexOf("agreement_has_dry_window")]);
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("{Key} {V}: no predictions produced.", compositeKey, versionName);
            return false;
        }

        await WritePredictionsAsync(predictions, stationSlug, windowHours, anchorDate, metadata.Version, ct);
        return true;
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
            if (kv.Value.Active.Count == 0) continue;
            result.Add(kv);
        }
        return result;
    }

    internal static bool SlugMatches(string slug, string arg)
    {
        if (slug.Equals(arg, StringComparison.OrdinalIgnoreCase)) return true;
        var slugWithoutPrefix = slug.StartsWith("ea_") ? slug[3..] : slug;
        if (slugWithoutPrefix.Equals(arg, StringComparison.OrdinalIgnoreCase)) return true;
        var derived = StationSlug.Of(arg);
        return slug.Equals("ea_" + derived, StringComparison.OrdinalIgnoreCase)
            || slugWithoutPrefix.Equals(derived, StringComparison.OrdinalIgnoreCase);
    }

    internal static (string StationSlug, int WindowHours)? ParseCompositeKey(string key)
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

        // Slot index by canonical model order so the indices line up with
        // TempFeatureBuilder.CanonicalModelOrder used at predict time below — when AIFS
        // (slot 6) is in the trained spec, modelDayList[6] needs to exist. The legacy
        // DryWindowFeatureBuilder.ModelIds is only 6 wide; using it here was the bug.
        var canonOrder = WeatherBlend.Train.TempFeatureBuilder.CanonicalModelOrder;
        var slotByModel = canonOrder
            .Select((id, i) => (id, i))
            .ToDictionary(x => x.id, x => x.i);
        var slotCount = canonOrder.Count;

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
                list = new List<DryWindowFeatureBuilder.ForecastDay?>(slotCount);
                for (int s = 0; s < slotCount; s++) list.Add(null);
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
        var outPath = Path.Combine(_cfg.Storage.PredictionsPath,
            "dry_window",
            stationSlug,
            $"window_{windowHours}h",
            $"model_version={modelVersion}",
            $"date={dateStr}",
            "predictions.parquet");

        // Dry-window is day-granular: one row per (PMT, lead), so the dedup
        // key omits ValidTimeUtc (unlike the hourly temp/precip/element writers).
        var total = await PredictionParquetWriter.WriteAsync(
            outPath, predictions,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.TargetDateUtc).ThenBy(r => r.LeadHours),
            ct);
        _log.LogInformation("Wrote {N} new predictions (file now holds {T}) → {Path}",
            predictions.Count, total, outPath);
    }

    private static double? NanToNull(float v) => float.IsNaN(v) ? null : v;
    private static double? NanToNullDouble(float v) => float.IsNaN(v) ? null : (double)v;
}
