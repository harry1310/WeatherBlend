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
using WeatherBlend.Train.Exact12h;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended-temperature forecasts for leads {24, 48, 72}. Champion/challenger
/// aware: when the user passes <c>--model-version current</c> (the default) the command
/// iterates every version in <c>Manifest.Active</c> and emits a parquet per version, so
/// Phase 2b and Phase 2c models predict side-by-side every cycle.
///
/// Per-version dispatch: <c>training_metadata.json::Phase</c> selects the feature builder.
///   "2b" → 13-feature lean (TempFeatureBuilder + TrainingRow)
///   "2c" → 88-feature rich (TempRichFeatureBuilder + RichTrainingRow)
///
/// Previous Runs API rows (<c>RunTimeSource = 'offset_day'</c>) are excluded — those
/// are historical training data, not a live forecast.
/// </summary>
public sealed class TempPredictCommand
{
    private readonly ILogger<TempPredictCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = Leads.Full;

    public TempPredictCommand(ILogger<TempPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(string target, string modelVersion, DateOnly? forDate, CancellationToken ct)
        => RunAsync(target, modelVersion, forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(string target, string modelVersion, DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported (got '{Target}')", target);
            return 2;
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        // Station-keyed (per-location) manifest layout post 2026-05-14.
        // Decide which station keys to predict for:
        //   --location given → just that location's entry
        //   default          → every manifest station entry that has Active
        //                      versions (typically just bonehill_rocks today).
        // Each station's predict cycle uses ITS OWN location's NWP via
        // _cfg.Locations lookup. Membury can be wired in by a future train
        // (Milestone 4 of the 2026-05-14 multi-location rollout) — nothing
        // about predict needs to change once a manifest entry exists.
        IReadOnlyList<string> stationKeys;
        if (!string.IsNullOrWhiteSpace(locationOverride))
        {
            stationKeys = new[] { locationOverride };
        }
        else
        {
            // ListStations returns only station entries with non-empty Current,
            // which mirrors precip's "demoted-archive" filter and is exactly the
            // set we want to predict for.
            var allStations = WeatherBlend.Train.ModelArtifact.ListStations(modelsRoot, "temperature");
            stationKeys = allStations.Count > 0
                ? allStations
                : new[] { _cfg.Location.Name };  // first-ever predict before any station-keyed bundle
        }
        _log.LogInformation("Predicting for stations: [{Stations}]", string.Join(", ", stationKeys));

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);

        var anyOutput = false;
        var failures = 0;
        foreach (var stationKey in stationKeys)
        {
            ct.ThrowIfCancellationRequested();
            var location = _cfg.Locations.FirstOrDefault(l =>
                l.Name.Equals(stationKey, StringComparison.OrdinalIgnoreCase));
            if (location is null)
            {
                _log.LogWarning("Station key '{Station}' has a temperature manifest entry but no matching location in config.yaml — skipping.",
                    stationKey);
                continue;
            }

            var versions = ResolveRequestedVersions(modelsRoot, stationKey, modelVersion);
            if (versions.Count == 0)
            {
                _log.LogWarning("Station '{Station}': no Active versions in manifest. Skipping.", stationKey);
                continue;
            }
            _log.LogInformation("[{Station}] Active versions: [{Versions}]", stationKey, string.Join(", ", versions));

            // 24 hourly targets per lead bucket — mirrors PrecipPredictCommand
            // so each predict cycle emits a full hourly trajectory (anchor+L .. anchor+L+23h)
            // for every lead, not just one snapshot value at lead L exactly.
            var anchorDayUtc = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc);
            var targets = DefaultLeads
                .SelectMany(L =>
                {
                    var dayStart = anchorDayUtc.AddDays(L / 24);
                    return Enumerable.Range(0, 24).Select(h => (Lead: L, Valid: dayStart.AddHours(h)));
                })
                .ToArray();

            _log.LogInformation("[{Station}] Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — {N} hourly targets across leads {Leads}",
                stationKey, anchor,
                forDate?.ToString("yyyy-MM-dd") ?? "live",
                targets.Length,
                string.Join(",", DefaultLeads));

            // Per-lead forecast pull — keyed to this station's location.
            var perLeadValid = new Dictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>>();
            foreach (var lead in DefaultLeads)
            {
                var leadTargets = targets.Where(t => t.Lead == lead).ToArray();
                if (leadTargets.Length == 0) continue;
                perLeadValid[lead] = QueryLatestForecastRows(
                    _cfg.Storage.ForecastsPath,
                    location.Name,
                    leadTargets.Min(t => t.Valid),
                    leadTargets.Max(t => t.Valid),
                    asOfRunTime: anchor,
                    leadHoursLowerBound: lead,
                    ct);
            }

            foreach (var version in versions)
            {
                ct.ThrowIfCancellationRequested();
                var ok = await PredictForVersionAsync(modelsRoot, stationKey, location, version, perLeadValid, targets, predictionMadeAt, anchor, ct);
                if (ok) anyOutput = true; else failures++;
            }
        }

        if (!anyOutput)
        {
            _log.LogError("No versions produced predictions — forecast tree likely missing recent data. Run `collect` first.");
            return 3;
        }
        return failures == 0 ? 0 : 4;
    }

    private List<string> ResolveRequestedVersions(string modelsRoot, string stationKey, string modelVersion)
    {
        // "current" / "all" → iterate every active version for THIS station entry.
        // Anything else is an explicit version dir name (back-compat single-version CLI).
        var v = modelVersion?.ToLowerInvariant() ?? "current";
        if (v is "current" or "all")
            return ModelArtifact.ResolveStationActive(modelsRoot, "temperature", stationKey).ToList();
        return new List<string> { modelVersion! };
    }

    private async Task<bool> PredictForVersionAsync(
        string modelsRoot,
        string stationKey,
        Config.LocationConfig activeLocation,
        string version,
        IReadOnlyDictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>> perLeadValid,
        (int Lead, DateTime Valid)[] targets,
        DateTime predictionMadeAt,
        DateTime anchor,
        CancellationToken ct)
    {
        string versionDir;
        ModelArtifact.TrainingMetadata metadata;
        try
        {
            versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "temperature", stationKey, version);
            metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        }
        catch (Exception ex)
        {
            _log.LogError("Cannot load version {V} for station {S}: {Msg}", version, stationKey, ex.Message);
            return false;
        }
        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Version {V} has no per-lead blenders — skipping.", version);
            return false;
        }

        // Phase A multi-location safety: the station key already pre-filters
        // versions to the right location; this LocationName pin is the
        // belt-and-braces check against corrupt metadata. metadata.LocationName
        // is [JsonRequired] so a missing field already threw at deserialise —
        // only the mismatch case remains to guard.
        if (!string.Equals(metadata.LocationName, activeLocation.Name, StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError(
                "Bundle {V} (under station '{Station}') was trained on location '{Trained}' but predict is using NWP from '{Active}' — refusing to score.",
                version, stationKey, metadata.LocationName, activeLocation.Name);
            return false;
        }

        var phase = (metadata.Phase ?? "").ToLowerInvariant();
        var isRich = phase == "2c";
        var isExact = phase == "2d";
        _log.LogInformation("--- Version {V} (phase={Phase}, {Mode}) ---",
            version, metadata.Phase, isExact ? "exact-runtime" : isRich ? "rich" : "lean");

        var ml = new MLContext(seed: 42);
        var predictions = new List<TempPredictionRow>();

        // Both lean (2b) and rich (2c) artefacts now use the per-lead BlenderSpec
        // layout — the rich variant just has a longer feature vector. Schema is
        // loaded from feature_schema.json so config can change without breaking
        // already-trained artefacts.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);

        // Phase 2d (exact-runtime) takes a separate predict path: different
        // forecast tree filter (RunTimeSource='exact'), different lead set
        // ({12, 24}), different ValidTime grid ({0,6,12,18}), per-V-hour UKV
        // pull. Feature vector + model are loaded the same way; only the
        // input pivot + output column population differ.
        if (isExact)
        {
            var ok = await PredictExactForVersionAsync(stationKey, activeLocation, versionDir, metadata, specs, predictionMadeAt, anchor, ct);
            return ok;
        }

        foreach (var (lead, valid) in targets)
        {
            if (!perLeadValid.TryGetValue(lead, out var perValid)
                || !perValid.TryGetValue(valid, out var pivot))
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
            var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();
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
                row = TempRichFeatureBuilder.ComposeRow(spec, valid, specTemps,
                    dewPoints: dew, rhs: rh, clouds: cloud,
                    cloudLows: cloudL, cloudMids: cloudM, cloudHighs: cloudH,
                    windSpeeds: ws, windDirsDeg: wd, windGusts: wg, pressures: pres,
                    windDirMeanDeg: pivot.WindDirMean, era5Temp: double.NaN);
            }
            else
            {
                row = TempFeatureBuilder.ComposeRow(spec, valid, specTemps, pivot.WindDirMean, era5Temp: double.NaN);
            }

            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TempTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];
            var featureHash = FeatureHashing.HashFloats(row.Features);

            // Build TempPredictionRow per-model fields: populate only spec.Models, null elsewhere.
            // Sized + skip-guarded from TempPredictionRow.PerModelFieldCount so adding a
            // new TempXxx field on the row type is the single change needed; this
            // pipeline picks it up automatically.
            var perModelTemp = new double?[TempPredictionRow.PerModelFieldCount];
            var perModelRun  = new DateTime?[TempPredictionRow.PerModelFieldCount];
            for (int i = 0; i < spec.Models.Count; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= TempPredictionRow.PerModelFieldCount) continue;
                perModelTemp[ci] = specTemps[i];
                perModelRun[ci]  = pivot.RunTime[ci];
            }

            // Spread features live at offset spec.Models.Count in the Features vector
            // (lean: N temps then mean/std/range; rich starts with the same N+3 lean block).
            var spreadStart = spec.Models.Count;
            predictions.Add(new TempPredictionRow
            {
                LocationName = activeLocation.Name,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendTemperature = yhat,
                TempGfs   = perModelTemp[0], TempEcmwf = perModelTemp[1], TempIcon  = perModelTemp[2],
                TempMf    = perModelTemp[3], TempUkmo  = perModelTemp[4], TempGem   = perModelTemp[5],
                TempAifs  = perModelTemp[6],
                RunTimeGfs   = perModelRun[0], RunTimeEcmwf = perModelRun[1],
                RunTimeIcon  = perModelRun[2], RunTimeMf    = perModelRun[3],
                RunTimeUkmo  = perModelRun[4], RunTimeGem   = perModelRun[5],
                RunTimeAifs  = perModelRun[6],
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

        await WritePredictionsAsync(stationKey, predictions, anchor, metadata.Version, ct);
        return true;
    }

    private async Task WritePredictionsAsync(
        string stationKey,
        IReadOnlyList<TempPredictionRow> predictions,
        DateTime anchor,
        string modelVersion,
        CancellationToken ct)
    {
        // Layout: data/predictions/temperature/{station}/model_version=<v>/date=<yyyy-MM-dd>/predictions.parquet.
        // Station-keyed since 2026-05-14 to mirror precip's per-station tree.
        // Date is the anchor date so re-runs across the same UTC day land in the same file.
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outPath = Path.Combine(_cfg.Storage.PredictionsPath,
            "temperature",
            stationKey,
            $"model_version={modelVersion}",
            $"date={dateStr}",
            "predictions.parquet");

        // Dedupe key includes ValidTimeUtc so the 24 hourly predictions per
        // lead bucket each keep their own row — keying on (PredictionMadeAtUtc,
        // LeadHours) alone would silently collapse them to 1-row-per-lead.
        var total = await PredictionParquetWriter.WriteAsync(
            outPath, predictions,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours),
            ct);
        _log.LogInformation("  Wrote {New} new predictions (file now holds {Total}) → {Path}",
            predictions.Count, total, outPath);
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
        int? leadHoursLowerBound,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var liveCycleFilter = PredictForecastFilters.LiveCycleAsOf(
            locationName, asOfRunTime, earliestValid, latestValid, leadHoursLowerBound);

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

        // Model-id → slot index (canonical model order from TempFeatureBuilder).
        var modelSlot = TempFeatureBuilder.CanonicalModelOrder
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
    // once we've finished slotting every model in.
    private sealed class WorkingPivot
    {
        // Sized to TempFeatureBuilder.CanonicalModelOrder.Count so a new NWP added
        // to the canonical order auto-grows these arrays. Indexed by canon-order
        // position; distinct from TempPredictionRow.PerModelFieldCount (output slots,
        // currently 7 — temp blender skips JMA).
        private static int N => TempFeatureBuilder.CanonicalModelOrder.Count;
        public double?[] Temp = new double?[N];
        public DateTime?[] RunTime = new DateTime?[N];
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
        private static double[] NanArray() => Enumerable.Repeat(double.NaN, N).ToArray();
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static double NullableDoubleAsNan(IDataReader r, int ord)
        => r.IsDBNull(ord) ? double.NaN : r.GetDouble(ord);

    // ------------------------------------------------------------------
    // Phase 2d (exact-runtime) predict path
    // ------------------------------------------------------------------
    //
    // Builds predictions for the per-lead specs persisted in 2d's
    // feature_schema.json (typically {12, 24}). For each lead, scans the
    // exact-runtime forecast tree for ValidTime ∈ {00,06,12,18} within the
    // forward window, picks the latest cycle satisfying RunTime ≤ Valid - L,
    // pivots per-model temps in the spec's canonical order, conditionally
    // pulls UKV at the per-V-hour (cycle, lead) tuple, composes the feature
    // vector via Exact12hFeatureBuilder.ComposeRow, scores, and emits a
    // TempPredictionRow with the new *Exact columns populated.

    /// <summary>The 4-cycle ValidTime grid 2d trains on. UKV-conditional and
    /// IFS/MO Global cycle structure all align here; predict mirrors training
    /// exactly so a row composed at predict time is feature-byte-equivalent
    /// to the one fed during training.</summary>
    private static readonly int[] Phase2dValidHoursUtc = { 0, 6, 12, 18 };

    private async Task<bool> PredictExactForVersionAsync(
        string stationKey,
        Config.LocationConfig activeLocation,
        string versionDir,
        ModelArtifact.TrainingMetadata metadata,
        IReadOnlyDictionary<int, BlenderSpec> specs,
        DateTime predictionMadeAt,
        DateTime anchor,
        CancellationToken ct)
    {
        var ml = new MLContext(seed: 42);
        var predictions = new List<TempPredictionRow>();

        // Forward window: anchor up to anchor + max(specLead) + 24h. Loose
        // upper bound — we filter to the spec's lead inside the loop anyway.
        var maxLead = specs.Keys.DefaultIfEmpty(0).Max();
        var windowStart = anchor;
        var windowEnd = anchor.AddHours(maxLead + 24);

        // One DuckDB query covers every (ValidTime, model, lead, cycle) row in
        // the relevant window — re-used across spec leads. Stays in-memory; the
        // resulting dictionary is keyed by (ValidTime, Lead) → per-model temps.
        var pivotByLeadValid = QueryExactForecastRows(
            _cfg.Storage.ForecastsPath,
            activeLocation.Name,
            specs: specs,
            earliestValid: windowStart,
            latestValid: windowEnd,
            asOfRunTime: anchor,
            ct);

        foreach (var (lead, spec) in specs.OrderBy(kv => kv.Key))
        {
            // 24 hourly targets per lead bucket — match the offset_day predict
            // contract so downstream parquets are uniformly shaped. Out of these
            // Generate today's target valid hours AND tomorrow's so a
            // late-evening predict still has future-of-now lead-12 slots
            // (without the +1-day extension, lead-12 at 21:15 had zero
            // future targets — all of today's {0,6,12,18} were past-of-
            // anchor → filtered out → no predictions written for the
            // entire evening). Only the four ValidTime hours {0,6,12,18}
            // get scored; the rest land outside the exact-runtime grid
            // and skip silently. Pivot lookup misses (cycle not in R2)
            // also skip silently.
            var dayStart = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc).AddDays(lead / 24);
            var leadTargets = Enumerable.Range(0, 48)
                .Select(h => dayStart.AddHours(h))
                .Where(v => Phase2dValidHoursUtc.Contains(v.Hour))
                .ToList();

            var includeUkv = spec.FeatureNames.Contains("temp_ukv");

            foreach (var valid in leadTargets)
            {
                if (!pivotByLeadValid.TryGetValue((lead, valid), out var pivot))
                {
                    _log.LogDebug("  Lead {Lead}h: no exact-runtime row for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                        lead, valid);
                    continue;
                }

                // Required-model gate matches training: row dropped when any
                // RequiredModels slot is missing at the target lead.
                var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
                var modelTemps = new double[spec.Models.Count];
                var modelRunTimes = new DateTime?[spec.Models.Count];
                bool allOptionalMissing = false;
                var missingRequired = new List<string>();
                for (int i = 0; i < spec.Models.Count; i++)
                {
                    var m = spec.Models[i];
                    if (pivot.PerModelTemp.TryGetValue(m, out var v) && v.HasValue)
                    {
                        modelTemps[i] = v.Value;
                        if (pivot.PerModelRunTime.TryGetValue(m, out var rt)) modelRunTimes[i] = rt;
                    }
                    else
                    {
                        modelTemps[i] = double.NaN;
                        if (requiredSet.Contains(m)) missingRequired.Add(m);
                    }
                }
                if (missingRequired.Count > 0)
                {
                    _log.LogDebug("  Lead {Lead}h: missing required exact-runtime models [{M}] at valid={V:yyyy-MM-dd HH:mm}Z; skipping.",
                        lead, string.Join(",", missingRequired), valid);
                    continue;
                }
                if (modelTemps.All(double.IsNaN))
                {
                    allOptionalMissing = true;
                }
                if (allOptionalMissing) continue;

                // UKV pull is part of the exact-runtime row already (LEFT
                // JOINed by the SQL). NaN when the per-V-hour conditional
                // didn't find a 12h-ahead UKV cycle — LightGBM splits on
                // missing.
                var ukvTemp = pivot.UkvTemp;

                var row = Exact12hFeatureBuilder.ComposeRow(
                    spec,
                    valid,
                    perModelLeadValues: modelTemps,
                    canonicalPerModel: modelTemps,   // single-lead, so canonical == per-model
                    windDirMeanDeg: double.NaN,      // 2d feature vector doesn't include wind dir mean
                    era5Temp: double.NaN,
                    ukvTemp: ukvTemp);

                var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
                var yhat = TempTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];
                var featureHash = FeatureHashing.HashFloats(row.Features);

                // Build TempPredictionRow with the new exact-runtime per-model
                // columns populated. spec.Models order matters — slot 0 is GFS,
                // slot 1 is IFS oper if present (otherwise AIFS), etc. We
                // dispatch by model id (not slot index) so out-of-order specs
                // populate the right column.
                double? gfs = null, ifs = null, aifs = null, mog = null;
                DateTime? gfsR = null, ifsR = null, aifsR = null, mogR = null;
                for (int i = 0; i < spec.Models.Count; i++)
                {
                    var t = double.IsNaN(modelTemps[i]) ? (double?)null : modelTemps[i];
                    switch (spec.Models[i])
                    {
                        case "gfs_ncep":          gfs = t;  gfsR  = modelRunTimes[i]; break;
                        case "ecmwf_ifs_oper":    ifs = t;  ifsR  = modelRunTimes[i]; break;
                        case "ecmwf_aifs_oper":   aifs = t; aifsR = modelRunTimes[i]; break;
                        case "met_office_global": mog = t;  mogR  = modelRunTimes[i]; break;
                    }
                }
                var ukvCol = includeUkv && !double.IsNaN(ukvTemp) ? (double?)ukvTemp : null;

                // spec.FeatureNames layout: per-model temps, [optional ukv],
                // mean, std, range, 4 calendar floats. The spread block lives
                // at index spec.Models.Count + (ukv ? 1 : 0).
                var spreadStart = spec.Models.Count + (includeUkv ? 1 : 0);
                predictions.Add(new TempPredictionRow
                {
                    LocationName = activeLocation.Name,
                    ModelVersion = metadata.Version,
                    PredictionMadeAtUtc = predictionMadeAt,
                    ValidTimeUtc = valid,
                    LeadHours = lead,
                    BlendTemperature = yhat,
                    // Offset-day per-model slots stay null on 2d rows.
                    TempGfsExact      = gfs,  RunTimeGfsExact      = gfsR,
                    TempIfsOperExact  = ifs,  RunTimeIfsOperExact  = ifsR,
                    TempAifsOperExact = aifs, RunTimeAifsOperExact = aifsR,
                    TempMoGlobalExact = mog,  RunTimeMoGlobalExact = mogR,
                    TempUkvExact      = ukvCol, RunTimeUkvExact    = pivot.UkvRunTime,
                    TempMean  = row.Features[spreadStart + 0],
                    TempStd   = row.Features[spreadStart + 1],
                    TempRange = row.Features[spreadStart + 2],
                    FeatureVectorHash = featureHash,
                });

                _log.LogInformation("  Lead {Lead}h → blend {Blend:0.00}°C (valid {Valid:yyyy-MM-dd HH:mm}Z, exact-runtime mean-of-{N} {Mean:0.00}°C)",
                    lead, yhat, valid, spec.Models.Count, row.Features[spreadStart + 0]);
            }
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Version {V} produced no predictions.", metadata.Version);
            return false;
        }

        await WritePredictionsAsync(stationKey, predictions, anchor, metadata.Version, ct);
        return true;
    }

    /// <summary>Per-(lead, ValidTime) pivot built from the exact-runtime tree.
    /// Per-model dictionaries keyed by canonical model id rather than slot
    /// index — keeps the dispatch above tolerant to spec.Models reordering
    /// across phases.</summary>
    private sealed record ExactPivotRow(
        Dictionary<string, double?> PerModelTemp,
        Dictionary<string, DateTime> PerModelRunTime,
        double UkvTemp,
        DateTime? UkvRunTime);

    private Dictionary<(int Lead, DateTime ValidTime), ExactPivotRow> QueryExactForecastRows(
        string forecastsPath,
        string locationName,
        IReadOnlyDictionary<int, BlenderSpec> specs,
        DateTime earliestValid,
        DateTime latestValid,
        DateTime asOfRunTime,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (specs.Count == 0) return new();

        var specLeads = specs.Keys.OrderBy(l => l).ToArray();
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var loc = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",",
            Exact12hFeatureBuilder.CanonicalModelOrder.Select(m => $"'{m}'")) + ")";
        var leadInClause = "(" + string.Join(",", specLeads) + ")";
        var hourInClause = "(" + string.Join(",", Phase2dValidHoursUtc) + ")";

        // Only build a UKV CTE leg for leads whose trained spec actually
        // pulls UKV — the FeatureNames-contains-`temp_ukv` test is the
        // codified train/predict contract (see Exact12hFeatureBuilder.cs:179
        // + :439). Without this filter, calling UkvPerVOrClause for a no-UKV
        // version's leads (e.g. v2026-05-08_060055_phase2d at 72/96/120)
        // throws ArgumentException because the strict picker only defines
        // tables at {12, 24, 48}. That's exactly how predict-and-render
        // crashed in CI on 2026-05-08 09:18 UTC after the no-UKV long-lead
        // 2d version got promoted. Fixed by skipping UKV legs for those
        // leads. If NO lead uses UKV the entire CTE + LEFT JOIN drops too.
        var ukvLeads = specs
            .Where(kv => kv.Value.FeatureNames.Contains("temp_ukv", StringComparer.Ordinal))
            .Select(kv => kv.Key)
            .OrderBy(l => l)
            .ToArray();

        // UKV CTE per blender lead — UKV's per-V-hour (cycle, lead) tuples
        // differ by target lead so the average UKV-effective-lead matches.
        // UNION ALL across spec leads with a BlenderLead tag column; the
        // outer LEFT JOIN keys on (ValidTime, BlenderLead = forecast row's
        // LeadHours) so lead-12 forecast rows get UKV picked under the
        // 12h-ahead rule and lead-24 rows get UKV picked under the
        // 24h-ahead rule. Fixes the lead-24 leak the buggy single-CTE
        // version had (UKV at 12h-ahead while other inputs were at 24h).
        var latestCte = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, LeadHours, RunTimeUtc, Temperature2m,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model, LeadHours
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{loc}'
      AND RunTimeSource = 'exact'
      AND LeadHours IN {leadInClause}
      AND HOUR(ValidTimeUtc) IN {hourInClause}
      AND Temperature2m IS NOT NULL
      AND Model IN {modelInClause}
      AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
)";

        string sql;
        if (ukvLeads.Length > 0)
        {
            // At least one spec lead uses UKV — build legs for those leads only.
            var ukvUnionLegs = string.Join("\n        UNION ALL\n",
                ukvLeads.Select(blenderLead => $@"        SELECT
            ValidTimeUtc, {blenderLead} AS BlenderLead,
            Temperature2m AS ukv_temp, RunTimeUtc AS ukv_run_time,
            ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn2
        FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{loc}'
          AND Model = 'met_office_ukv'
          AND RunTimeSource = 'exact'
          AND Temperature2m IS NOT NULL
          AND HOUR(ValidTimeUtc) IN {hourInClause}
          AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
          AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                               AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
          AND ({Exact12hFeatureBuilder.UkvPerVOrClause(blenderLead, Exact12hFeatureBuilder.UkvPickStrategy.Strict)})"));

            sql = $@"{latestCte},
ukv AS (
    SELECT ValidTimeUtc, BlenderLead, ukv_temp, ukv_run_time
    FROM (
{ukvUnionLegs}
    )
    WHERE rn2 = 1
)
SELECT l.ValidTimeUtc, l.Model, l.LeadHours, l.RunTimeUtc, l.Temperature2m,
       u.ukv_temp, u.ukv_run_time
FROM latest l
LEFT JOIN ukv u
       ON l.ValidTimeUtc = u.ValidTimeUtc
      AND l.LeadHours    = u.BlenderLead
WHERE l.rn = 1
  AND l.RunTimeUtc <= l.ValidTimeUtc - (l.LeadHours * INTERVAL 1 HOUR)
ORDER BY l.ValidTimeUtc, l.LeadHours, l.Model;";
        }
        else
        {
            // No spec lead uses UKV — drop the CTE + LEFT JOIN. Reader still
            // expects 7 columns (ukv_temp + ukv_run_time at indices 5/6) so
            // emit explicit NULLs of the right type to keep the schema stable.
            sql = $@"{latestCte}
SELECT l.ValidTimeUtc, l.Model, l.LeadHours, l.RunTimeUtc, l.Temperature2m,
       CAST(NULL AS DOUBLE) AS ukv_temp, CAST(NULL AS TIMESTAMP) AS ukv_run_time
FROM latest l
WHERE l.rn = 1
  AND l.RunTimeUtc <= l.ValidTimeUtc - (l.LeadHours * INTERVAL 1 HOUR)
ORDER BY l.ValidTimeUtc, l.LeadHours, l.Model;";
        }

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var reader = cmd.ExecuteReader();

        var result = new Dictionary<(int Lead, DateTime ValidTime), ExactPivotRow>();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid    = reader.GetDateTime(0);
            var model    = reader.GetString(1);
            var lead     = reader.GetInt32(2);
            var runTime  = reader.GetDateTime(3);
            var temp     = NullableDouble(reader, 4);
            var ukvTemp  = NullableDouble(reader, 5) ?? double.NaN;
            var ukvRunTime = reader.IsDBNull(6) ? (DateTime?)null : reader.GetDateTime(6);

            var key = (lead, valid);
            if (!result.TryGetValue(key, out var existing))
            {
                existing = new ExactPivotRow(
                    PerModelTemp: new Dictionary<string, double?>(StringComparer.Ordinal),
                    PerModelRunTime: new Dictionary<string, DateTime>(StringComparer.Ordinal),
                    UkvTemp: ukvTemp,
                    UkvRunTime: ukvRunTime);
                result[key] = existing;
            }
            existing.PerModelTemp[model] = temp;
            if (temp.HasValue) existing.PerModelRunTime[model] = runTime;
        }
        return result;
    }
}
