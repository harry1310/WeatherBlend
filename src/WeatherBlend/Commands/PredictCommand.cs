using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended-temperature forecasts for leads {24, 48, 72}. Champion/challenger
/// aware: when the user passes <c>--model-version current</c> (the default) the command
/// iterates every version in <c>Manifest.Active</c> and emits a parquet per version, so
/// Phase 2b and Phase 2c models predict side-by-side every cycle.
///
/// Per-version dispatch: <c>training_metadata.json::Phase</c> selects the feature builder.
///   "2b" → 13-feature lean (FeatureBuilder + TrainingRow)
///   "2c" → 88-feature rich (RichFeatureBuilder + RichTrainingRow)
///
/// Previous Runs API rows (<c>RunTimeSource = 'offset_day'</c>) are excluded — those
/// are historical training data, not a live forecast.
/// </summary>
public sealed class PredictCommand
{
    private readonly ILogger<PredictCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = { 24, 48, 72, 120 };

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
        var versions = ResolveRequestedVersions(modelsRoot, modelVersion);
        if (versions.Count == 0)
        {
            _log.LogError("No versions to predict — manifest has no Active list and no Current pointer.");
            return 2;
        }

        _log.LogInformation("Predicting with versions: [{Versions}]", string.Join(", ", versions));

        // predictionMadeAt = real wall-clock UTC (provenance).
        // anchor = the hour we compute lead offsets from (see PredictAnchor).
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        var targets = DefaultLeads.Select(L => (Lead: L, Valid: anchor.AddHours(L))).ToArray();

        _log.LogInformation("Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — targets: {Targets}",
            anchor,
            forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", targets.Select(t => $"{t.Lead}h→{t.Valid:yyyy-MM-dd HH:mm}Z")));

        // One forecast pull covers every version — they all read from the same tree.
        var perValid = QueryLatestForecastRows(
            _cfg.Storage.ForecastsPath,
            _cfg.Location.Name,
            targets.Min(t => t.Valid),
            targets.Max(t => t.Valid),
            asOfRunTime: anchor,
            ct);

        var anyOutput = false;
        var failures = 0;
        foreach (var version in versions)
        {
            ct.ThrowIfCancellationRequested();
            var ok = await PredictForVersionAsync(modelsRoot, version, perValid, targets, predictionMadeAt, anchor, ct);
            if (ok) anyOutput = true; else failures++;
        }

        if (!anyOutput)
        {
            _log.LogError("No versions produced predictions — forecast tree likely missing recent data. Run `collect` first.");
            return 3;
        }
        return failures == 0 ? 0 : 4;
    }

    private List<string> ResolveRequestedVersions(string modelsRoot, string modelVersion)
    {
        // "current" / "all" → iterate every active version. Anything else is an explicit
        // version dir name and we run only that one (back-compat with single-version CLI).
        var v = modelVersion?.ToLowerInvariant() ?? "current";
        if (v is "current" or "all")
            return ModelArtifact.ResolveActive(modelsRoot, "temperature").ToList();
        return new List<string> { modelVersion! };
    }

    private async Task<bool> PredictForVersionAsync(
        string modelsRoot,
        string version,
        IReadOnlyDictionary<DateTime, PivotedRow> perValid,
        (int Lead, DateTime Valid)[] targets,
        DateTime predictionMadeAt,
        DateTime anchor,
        CancellationToken ct)
    {
        string versionDir;
        ModelArtifact.TrainingMetadata metadata;
        try
        {
            versionDir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", version);
            metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        }
        catch (Exception ex)
        {
            _log.LogError("Cannot load version {V}: {Msg}", version, ex.Message);
            return false;
        }
        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Version {V} has no per-lead blenders — skipping.", version);
            return false;
        }

        var phase = (metadata.Phase ?? "").ToLowerInvariant();
        var isRich = phase == "2c";
        _log.LogInformation("--- Version {V} (phase={Phase}, {Mode}) ---",
            version, metadata.Phase, isRich ? "rich" : "lean");

        var ml = new MLContext(seed: 42);
        var predictions = new List<PredictionRow>();

        // Both lean (2b) and rich (2c) artefacts now use the per-lead BlenderSpec
        // layout — the rich variant just has a longer feature vector. Schema is
        // loaded from feature_schema.json so config can change without breaking
        // already-trained artefacts.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);

        foreach (var (lead, valid) in targets)
        {
            if (!perValid.TryGetValue(valid, out var pivot))
            {
                _log.LogWarning("  Lead {Lead}h: no forecast rows for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, valid);
                continue;
            }

            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("  Lead {Lead}h: no BlenderSpec in feature_schema.json for this lead; skipping.", lead);
                continue;
            }

            // Pull spec.Models values from the canonical 6-slot pivot, in spec order.
            var canonOrder = FeatureBuilder.CanonicalModelOrder.ToList();
            var specTemps = new double[spec.Models.Count];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < spec.Models.Count; i++)
            {
                var canonicalIdx = canonOrder.IndexOf(spec.Models[i]);
                if (canonicalIdx < 0)
                    throw new InvalidOperationException($"Spec model '{spec.Models[i]}' not in canonical order list.");
                var v = pivot.Temp[canonicalIdx];
                specTemps[i] = v ?? double.NaN;
                if (!v.HasValue && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }

            // Required-model gate: skip the row if any RequiredModels slot is null
            // (matches training's post-pivot WHERE every required NOT NULL).
            // Optional slots that are NaN survive — that's exactly the policy
            // training was built against.
            if (missingRequired.Count > 0)
            {
                _log.LogWarning("  Lead {Lead}h: missing required per-model temps [{Missing}] for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }
            // Universal at-least-one safety check (mirrors training).
            if (specTemps.All(double.IsNaN))
            {
                _log.LogWarning("  Lead {Lead}h: every model in spec is NaN at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, valid);
                continue;
            }

            // Compose feature row — lean or rich depending on the spec's feature-set.
            // Rich path also pulls per-model secondaries from the live pivot; lean path
            // ignores them. Both produce RegressionTrainingRow whose Features.Length
            // matches spec.FeatureCount.
            RegressionTrainingRow row;
            if (isRich)
            {
                // Pull per-model secondaries in spec.Models order (canonical-index lookup).
                int N = spec.Models.Count;
                var dew    = new double[N];
                var rh     = new double[N];
                var cloud  = new double[N];
                var cloudL = new double[N];
                var cloudM = new double[N];
                var cloudH = new double[N];
                var ws     = new double[N];
                var wd     = new double[N];
                var wg     = new double[N];
                var pres   = new double[N];
                for (int i = 0; i < N; i++)
                {
                    var ci = canonOrder.IndexOf(spec.Models[i]);
                    dew[i]    = pivot.DewPoints[ci];
                    rh[i]     = pivot.Rhs[ci];
                    cloud[i]  = pivot.Clouds[ci];
                    cloudL[i] = pivot.CloudLows[ci];
                    cloudM[i] = pivot.CloudMids[ci];
                    cloudH[i] = pivot.CloudHighs[ci];
                    ws[i]     = pivot.WindSpeeds[ci];
                    wd[i]     = pivot.WindDirsDeg[ci];
                    wg[i]     = pivot.WindGusts[ci];
                    pres[i]   = pivot.Pressures[ci];
                }
                row = RichFeatureBuilder.ComposeRow(spec, valid, specTemps,
                    dewPoints: dew, rhs: rh, clouds: cloud,
                    cloudLows: cloudL, cloudMids: cloudM, cloudHighs: cloudH,
                    windSpeeds: ws, windDirsDeg: wd, windGusts: wg, pressures: pres,
                    windDirMeanDeg: pivot.WindDirMean, era5Temp: double.NaN);
            }
            else
            {
                row = FeatureBuilder.ComposeRow(spec, valid, specTemps, pivot.WindDirMean, era5Temp: double.NaN);
            }

            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];
            var featureHash = FeatureHashing.HashFloats(row.Features);

            // Build PredictionRow per-model fields: populate only spec.Models, null elsewhere.
            // Output PredictionRow has 6 named per-model fields (TempGfs..TempGem); AIFS
            // (canonical index 6) contributes implicitly via BlendTemperature but isn't
            // exposed as a named field yet — schema-add follow-up.
            var perModelTemp = new double?[6];
            var perModelRun  = new DateTime?[6];
            for (int i = 0; i < spec.Models.Count; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= 6) continue;     // AIFS slot — not in 6-wide output yet
                perModelTemp[ci] = specTemps[i];
                perModelRun[ci]  = pivot.RunTime[ci];
            }

            // Spread features live at offset spec.Models.Count in the Features vector
            // (lean: N temps then mean/std/range; rich starts with the same N+3 lean block).
            var spreadStart = spec.Models.Count;
            predictions.Add(new PredictionRow
            {
                LocationName = _cfg.Location.Name,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendTemperature = yhat,
                TempGfs   = perModelTemp[0], TempEcmwf = perModelTemp[1], TempIcon  = perModelTemp[2],
                TempMf    = perModelTemp[3], TempUkmo  = perModelTemp[4], TempGem   = perModelTemp[5],
                RunTimeGfs   = perModelRun[0], RunTimeEcmwf = perModelRun[1],
                RunTimeIcon  = perModelRun[2], RunTimeMf    = perModelRun[3],
                RunTimeUkmo  = perModelRun[4], RunTimeGem   = perModelRun[5],
                TempMean  = row.Features[spreadStart + 0],
                TempStd   = row.Features[spreadStart + 1],
                TempRange = row.Features[spreadStart + 2],
                FeatureVectorHash = featureHash,
            });

            _log.LogInformation("  Lead {Lead}h → blend {Blend:0.00}°C (valid {Valid:yyyy-MM-dd HH:mm}Z, mean-of-{N} {Mean:0.00}°C)",
                lead, yhat, valid, spec.Models.Count, row.Features[spreadStart + 0]);
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Version {V} produced no predictions.", version);
            return false;
        }

        await WritePredictionsAsync(predictions, anchor, metadata.Version, ct);
        return true;
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

        // Dedupe on (PredictionMadeAtUtc, LeadHours) — newest row for the key wins.
        // MaxBy(PredictionMadeAtUtc) is explicit about "latest"; don't rely on concat
        // order so a retry that pulls existing-file rows in any order still converges.
        var merged = existing.Concat(predictions)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc)
            .ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("  Wrote {New} new predictions (file now holds {Total}) → {Path}",
            predictions.Count, merged.Count, outPath);
    }

    // Per-valid-time pivot: temp + wind dir mean + every secondary as a per-model array
    // with NaN sentinel for missing. Rich predict consumes the secondaries; lean predict
    // ignores them. One pivot serves both modes — cheaper than re-querying per version.
    private sealed record PivotedRow(
        double?[] Temp,
        DateTime?[] RunTime,
        double WindDirMean,
        double[] DewPoints,
        double[] Rhs,
        double[] Clouds,
        double[] CloudLows,
        double[] CloudMids,
        double[] CloudHighs,
        double[] WindSpeeds,
        double[] WindDirsDeg,
        double[] WindGusts,
        double[] Pressures);

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
        var liveCycleFilter = PredictForecastFilters.LiveCycleAsOf(
            locationName, asOfRunTime, earliestValid, latestValid);

        // Pull every variable a rich-feature build might want; lean predict simply ignores
        // the extras. Per-row latest-cycle resolution uses the same window function the
        // trainer uses so the predict-time and train-time pivots are byte-for-byte equivalent
        // for the temp + wind-dir columns that 2b cares about.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc,
           Temperature2m, DewPoint2m, RelativeHumidity2m,
           CloudCover, CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           WindSpeed10m, WindDirection10m, WindGusts10m, SurfacePressure,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {liveCycleFilter}
      AND Temperature2m IS NOT NULL
)
SELECT ValidTimeUtc, Model, RunTimeUtc,
       Temperature2m, DewPoint2m, RelativeHumidity2m,
       CloudCover, CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       WindSpeed10m, WindDirection10m, WindGusts10m, SurfacePressure
FROM latest
WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        // Model-id → slot index (canonical model order from FeatureBuilder).
        var modelSlot = FeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i))
            .ToDictionary(x => x.id, x => x.i);

        var working = new Dictionary<DateTime, WorkingPivot>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid   = reader.GetDateTime(0);
            var model   = reader.GetString(1);
            var runTime = reader.GetDateTime(2);

            if (!modelSlot.TryGetValue(model, out var slot))
                continue;

            if (!working.TryGetValue(valid, out var pivot))
            {
                pivot = WorkingPivot.New();
                working[valid] = pivot;
            }

            pivot.Temp[slot]     = NullableDouble(reader, 3);
            pivot.RunTime[slot]  = runTime;
            pivot.Dew[slot]      = NullableDoubleAsNan(reader, 4);
            pivot.Rh[slot]       = NullableDoubleAsNan(reader, 5);
            pivot.Cloud[slot]    = NullableDoubleAsNan(reader, 6);
            pivot.CloudLow[slot] = NullableDoubleAsNan(reader, 7);
            pivot.CloudMid[slot] = NullableDoubleAsNan(reader, 8);
            pivot.CloudHigh[slot]= NullableDoubleAsNan(reader, 9);
            pivot.WindSpeed[slot]= NullableDoubleAsNan(reader, 10);
            pivot.WindDir[slot]  = NullableDoubleAsNan(reader, 11);
            pivot.WindGusts[slot]= NullableDoubleAsNan(reader, 12);
            pivot.Pressure[slot] = NullableDoubleAsNan(reader, 13);
        }

        return working.ToDictionary(
            kv => kv.Key,
            kv =>
            {
                var p = kv.Value;
                var winds = p.WindDir.Where(d => !double.IsNaN(d)).ToList();
                return new PivotedRow(
                    Temp:        p.Temp,
                    RunTime:     p.RunTime,
                    WindDirMean: winds.Count > 0 ? winds.Average() : double.NaN,
                    DewPoints:   p.Dew,
                    Rhs:         p.Rh,
                    Clouds:      p.Cloud,
                    CloudLows:   p.CloudLow,
                    CloudMids:   p.CloudMid,
                    CloudHighs:  p.CloudHigh,
                    WindSpeeds:  p.WindSpeed,
                    WindDirsDeg: p.WindDir,
                    WindGusts:   p.WindGusts,
                    Pressures:   p.Pressure);
            });
    }

    // Internal mutable per-valid-time accumulator. Becomes an immutable PivotedRow
    // once we've finished slotting all six models in.
    private sealed class WorkingPivot
    {
        // Sized to FeatureBuilder.CanonicalModelOrder.Count (7 incl. AIFS).
        public double?[] Temp = new double?[7];
        public DateTime?[] RunTime = new DateTime?[7];
        public double[] Dew = NanArray();
        public double[] Rh = NanArray();
        public double[] Cloud = NanArray();
        public double[] CloudLow = NanArray();
        public double[] CloudMid = NanArray();
        public double[] CloudHigh = NanArray();
        public double[] WindSpeed = NanArray();
        public double[] WindDir = NanArray();
        public double[] WindGusts = NanArray();
        public double[] Pressure = NanArray();

        public static WorkingPivot New() => new();
        private static double[] NanArray() => Enumerable.Repeat(double.NaN, 7).ToArray();
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static double NullableDoubleAsNan(IDataReader r, int ord)
        => r.IsDBNull(ord) ? double.NaN : r.GetDouble(ord);
}
