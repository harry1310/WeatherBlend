using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Site;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended P(wet) forecasts for leads {24, 48, 72} using the per-station
/// Phase 3a occurrence classifier. Runs one pass per truth station so the output
/// tree mirrors the model tree: one parquet folder per
/// <c>data/predictions/precipitation/{truth_station}/</c>.
///
/// "Latest available" forecast semantics match <see cref="PredictCommand"/> —
/// for each (valid_time, model) pick the most recent run that covers it,
/// excluding the historical-forecast (<c>RunTimeSource='offset_day'</c>) rows.
/// The six per-model covariates (RH, dew depression, clouds, CAPE, wind) are
/// averaged across whichever models are present, matching how the feature row
/// is assembled at training time.
/// </summary>
public sealed class PrecipPredictCommand
{
    private readonly ILogger<PrecipPredictCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = { 24, 48, 72, 120 };

    public PrecipPredictCommand(ILogger<PrecipPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    /// <summary>
    /// <paramref name="truthStation"/> is a slug (<c>ea_bellever_dartmoor</c>), or
    /// <c>all</c> to run every station in the manifest, or a config rainfall
    /// station name (<c>Bellever Dartmoor</c>) for ergonomic CLI use.
    /// </summary>
    public async Task<int> RunAsync(string truthStation, string modelVersion, DateOnly? forDate, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");

        var stationsToRun = ResolveStations(modelsRoot, truthStation);
        if (stationsToRun.Count == 0)
        {
            _log.LogError("No precipitation blender artefacts found under {Dir}. Train first.",
                Path.Combine(modelsRoot, "precipitation"));
            return 2;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var targets = DefaultLeads.Select(L => (Lead: L, Valid: anchor.AddHours(L))).ToArray();

        _log.LogInformation("Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — stations=[{Stations}] — targets: {Targets}",
            anchor,
            forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", stationsToRun),
            string.Join(", ", targets.Select(t => $"{t.Lead}h→{t.Valid:yyyy-MM-dd HH:mm}Z")));

        // Pivot once across all stations — feature inputs don't depend on which
        // station we're predicting for, only the blender weights do.
        var perValid = QueryLatestForecastRows(
            _cfg.Storage.ForecastsPath,
            _cfg.Location.Name,
            targets.Min(t => t.Valid),
            targets.Max(t => t.Valid),
            asOfRunTime: anchor,
            ct);

        var anyWritten = false;
        foreach (var station in stationsToRun)
        {
            ct.ThrowIfCancellationRequested();
            foreach (var version in ResolveRequestedVersions(modelsRoot, station, modelVersion))
            {
                var wrote = await RunStationAsync(
                    modelsRoot, station, version, predictionMadeAt, anchor, targets, perValid, ct);
                anyWritten |= wrote;
            }
        }

        return anyWritten ? 0 : 3;
    }

    private IReadOnlyList<string> ResolveRequestedVersions(string modelsRoot, string station, string modelVersion)
    {
        // Mirrors PredictCommand.ResolveRequestedVersions for the per-station layout:
        // "current"/"all" → iterate every active version for this station (Phase 3c
        // champion/challenger). Anything else is an explicit version dir name.
        var v = modelVersion?.ToLowerInvariant() ?? "current";
        if (v is "current" or "all")
        {
            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", station);
            return active.Count == 0 ? new[] { "current" } : active.ToList();
        }
        return new[] { modelVersion! };
    }

    private async Task<bool> RunStationAsync(
        string modelsRoot,
        string station,
        string modelVersion,
        DateTime predictionMadeAt,
        DateTime anchor,
        (int Lead, DateTime Valid)[] targets,
        IReadOnlyDictionary<DateTime, PivotedRow> perValid,
        CancellationToken ct)
    {
        var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", station, modelVersion);
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Station {Station} model version {V} has no per-lead blenders.", station, metadata.Version);
            return false;
        }

        var climPath = Path.Combine(versionDir, ModelArtifact.ClimatologyFileName);
        if (!File.Exists(climPath))
        {
            _log.LogError("Station {Station} version {V} is missing {File} — retrain to persist it.",
                station, metadata.Version, ModelArtifact.ClimatologyFileName);
            return false;
        }
        var climatology = PrecipClimatology.LoadFrom(climPath);

        _log.LogInformation("Station {Station}: using blender version {V} (phase={Phase})",
            station, metadata.Version, metadata.Phase);

        var isRich = PrecipPhases.IsRich(metadata.Phase);
        var isIsotonic = PrecipPhases.IsIsotonic(metadata.Phase);

        PrecipIsotonicCalibration? calibration = null;
        if (isIsotonic)
        {
            var calPath = Path.Combine(versionDir, PrecipIsotonicCalibration.FileName);
            if (!File.Exists(calPath))
            {
                _log.LogError("Station {Station} version {V} declares Phase='3a_isotonic' but {File} is missing.",
                    station, metadata.Version, PrecipIsotonicCalibration.FileName);
                return false;
            }
            calibration = PrecipIsotonicCalibration.LoadFrom(calPath);
            _log.LogInformation("Station {Station}: loaded isotonic calibration — source={Src}, leads=[{Leads}]",
                station, calibration.SourceVersion, string.Join(", ", calibration.ByLead.Keys));
        }

        // Phase 3c needs EA observation persistence anchored at run_time = valid - lead;
        // load the whole hourly series once (small) and reuse across the three leads.
        Dictionary<DateTime, double>? hourlyRain = null;
        if (isRich)
        {
            var friendly = ResolveFriendlyStationName(station);
            if (friendly is null)
            {
                _log.LogError("Station {Station}: cannot resolve rainfall config name for phase-3c persistence lookup. Known: [{Known}]",
                    station, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
                return false;
            }
            hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(
                _cfg.Storage.RainfallPath, _cfg.Location.Name, friendly, ct);
            _log.LogInformation("Station {Station}: loaded {N} hourly rainfall rows for persistence features (friendly='{Friendly}')",
                station, hourlyRain.Count, friendly);
        }

        var ml = new MLContext(seed: 42);
        var predictions = new List<PrecipPredictionRow>();

        // Both lean (3a) and rich (3c) artefacts use the per-lead BlenderSpec layout.
        // Rich's spec just has a longer feature vector. Schema is read from feature_schema.json.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = FeatureBuilder.CanonicalModelOrder.ToList();

        foreach (var (lead, valid) in targets)
        {
            if (!perValid.TryGetValue(valid, out var pivot))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no forecast rows for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }
            if (!pivot.Precip.Any(p => p.HasValue))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: all six per-model precip values null for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }
            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no BlenderSpec in feature_schema.json for this lead; skipping.",
                    station, lead);
                continue;
            }

            // Pull spec.Models precip from the canonical pivot, in spec order.
            // prob_* removed 2026-04-28 (zero-gain, see PrecipFeatureBuilder).
            int N = spec.Models.Count;
            var specPrecip = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                var pv = pivot.Precip[ci];
                specPrecip[i] = pv ?? double.NaN;
                if (!pv.HasValue && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                _log.LogWarning("Station {Station} lead {Lead}h: missing required per-model precip [{Missing}] for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, string.Join(",", missingRequired), valid);
                continue;
            }

            // Compose the row — lean or rich depending on spec.FeatureSet.
            BinaryTrainingRow row;
            if (isRich)
            {
                var dew    = new double[N];
                var rh     = new double[N];
                var dewDep = new double[N];
                var pres   = new double[N];
                for (int i = 0; i < N; i++)
                {
                    var ci = canonOrder.IndexOf(spec.Models[i]);
                    var t  = pivot.Temp2m[ci];
                    var d  = pivot.Dew[ci];
                    dew[i]    = d ?? double.NaN;
                    rh[i]     = pivot.Rh[ci] ?? double.NaN;
                    dewDep[i] = (t.HasValue && d.HasValue) ? t.Value - d.Value : double.NaN;
                    pres[i]   = pivot.Pressure[ci] ?? double.NaN;
                }
                var runTime = valid.AddHours(-lead);
                var persist = PrecipRichFeatureBuilder.ComputePersistence(hourlyRain!, runTime);
                row = PrecipRichFeatureBuilder.ComposeRow(
                    spec, valid, specPrecip, dew, rh, dewDep, pres,
                    rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                    cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                    cloudHighMean: pivot.CloudHighMean,
                    capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                    eaRainPrev24hMm: persist.Prev24hMm,
                    eaRainPrev72hMm: persist.Prev72hMm,
                    eaWetHoursLast24h: persist.WetHoursLast24h,
                    eaDryHoursTrailing: persist.DryHoursTrailing,
                    truthMmHour: 0.0);
            }
            else
            {
                row = PrecipFeatureBuilder.ComposeRow(
                    spec, valid, specPrecip,
                    rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                    cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                    cloudHighMean: pivot.CloudHighMean,
                    capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                    truthMmHour: 0.0);
            }

            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var pWet = PrecipOccurrenceTrainer.PredictVectorProbability(ml, loadedModel, spec, new[] { row });

            if (calibration is not null && calibration.ByLead.TryGetValue(lead.ToString(), out var leadCal))
                pWet[0] = leadCal.Apply(pWet[0]);

            var climPWet = climatology.Predict(valid);

            // Build per-model output fields: populate only spec.Models, null elsewhere.
            // Sized from PrecipPredictionRow.PerModelFieldCount.
            var perModelPrecip = new double?[PrecipPredictionRow.PerModelFieldCount];
            var perModelRun    = new DateTime?[PrecipPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= PrecipPredictionRow.PerModelFieldCount) continue;
                perModelPrecip[ci] = specPrecip[i];
                perModelRun[ci]    = pivot.RunTime[ci];
            }

            // Spread features live at offset 2N in both lean and rich layouts
            // (precip per model, then prob per model, then mean/std/max/agreement).
            // Spread features at offset N now that prob_* is gone (was 2*N pre-cleanup).
            var spreadStart = N;
            predictions.Add(new PrecipPredictionRow
            {
                LocationName = _cfg.Location.Name,
                TruthStation = station,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                ProbWet = pWet[0],
                ClimatologyPWet = climPWet,
                PrecipGfs   = perModelPrecip[0], PrecipEcmwf = perModelPrecip[1],
                PrecipIcon  = perModelPrecip[2], PrecipMf    = perModelPrecip[3],
                PrecipUkmo  = perModelPrecip[4], PrecipGem   = perModelPrecip[5],
                PrecipAifs  = perModelPrecip[6], PrecipJma   = perModelPrecip[7],
                RunTimeGfs   = perModelRun[0], RunTimeEcmwf = perModelRun[1],
                RunTimeIcon  = perModelRun[2], RunTimeMf    = perModelRun[3],
                RunTimeUkmo  = perModelRun[4], RunTimeGem   = perModelRun[5],
                RunTimeAifs  = perModelRun[6], RunTimeJma   = perModelRun[7],
                PrecipMean = NanToNull(row.Features[spreadStart + 0]),
                PrecipStd  = NanToNull(row.Features[spreadStart + 1]),
                PrecipMax  = NanToNull(row.Features[spreadStart + 2]),
                PrecipAgreementWet01 = NanToNull(row.Features[spreadStart + 3]),
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });

            _log.LogInformation(
                "Station {Station} lead {Lead}h → P(wet) {P:0.000} (clim {Clim:0.000}, valid {Valid:yyyy-MM-dd HH:mm}Z, agreement {Ag:0.00})",
                station, lead, pWet[0], climPWet, valid, row.Features[spreadStart + 3]);
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Station {Station}: no predictions produced — likely missing forecast data.", station);
            return false;
        }

        await WritePredictionsAsync(predictions, station, anchor, metadata.Version, ct);
        return true;
    }

    private IReadOnlyList<string> ResolveStations(string modelsRoot, string truthStation)
    {
        var manifestStations = ModelArtifact.ListStations(modelsRoot, "precipitation");
        if (string.Equals(truthStation, "all", StringComparison.OrdinalIgnoreCase))
            return manifestStations;

        // Accept either the slug (ea_bellever_dartmoor) or the config station name
        // (Bellever Dartmoor). The config name is the human-facing input on the CLI,
        // but the blender tree is keyed by slug.
        var explicitSlug = manifestStations.FirstOrDefault(s =>
            string.Equals(s, truthStation, StringComparison.OrdinalIgnoreCase));
        if (explicitSlug is not null)
            return new[] { explicitSlug };

        var derivedSlug = "ea_" + Slugify(truthStation);
        var match = manifestStations.FirstOrDefault(s =>
            string.Equals(s, derivedSlug, StringComparison.OrdinalIgnoreCase));
        if (match is not null)
            return new[] { match };

        _log.LogError("Unknown truth station '{Station}'. Known: [{Known}]", truthStation, string.Join(", ", manifestStations));
        return Array.Empty<string>();
    }

    private async Task WritePredictionsAsync(
        IReadOnlyList<PrecipPredictionRow> predictions,
        string station,
        DateTime anchor,
        string modelVersion,
        CancellationToken ct)
    {
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(_cfg.Storage.PredictionsPath,
            "precipitation",
            station,
            $"model_version={modelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<PrecipPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<PrecipPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<PrecipPredictionRow>();

        // Dedupe on (PredictionMadeAtUtc, LeadHours) — matches the temperature path.
        var merged = existing.Concat(predictions)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc)
            .ThenBy(r => r.LeadHours)
            .ToList();

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("Wrote {New} new {Station} predictions (file now holds {Total}) → {Path}",
            predictions.Count, station, merged.Count, outPath);
    }

    // Per-valid-time pivot mirrors PredictCommand.PivotedRow but carries the wider
    // precip feature set required by the occurrence blender. Per-model arrays
    // (Dew/Rh/Temp/Pressure) feed the Phase 3c rich composer; the *Mean fields stay
    // for Phase 3a's lean composer and keep the prediction-row covariate summary.
    private sealed record PivotedRow(
        double?[] Precip,
        DateTime?[] RunTime,
        double?[] Dew,
        double?[] Rh,
        double?[] Temp2m,
        double?[] Pressure,
        double RhMean,
        double DewDepressionMean,
        double CloudLowMean,
        double CloudMidMean,
        double CloudHighMean,
        double CapeMean,
        double WindSpeedMean);

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

        // Mirrors PrecipFeatureBuilder's SQL skeleton but with the live-cycle filter
        // in place of RunTimeSource='offset_day' + LeadHours=. Pivot in .NET so we
        // can emit each model's RunTimeUtc into the prediction row for provenance.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc,
           Precipitation,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           Cape, WindSpeed10m, SurfacePressure,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {liveCycleFilter}
)
SELECT ValidTimeUtc, Model, RunTimeUtc,
       Precipitation,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       Cape, WindSpeed10m, SurfacePressure
FROM latest
WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var modelSlot = PrecipFeatureBuilder.ModelColumns
            .Select((m, i) => (m.ModelId, Index: i))
            .ToDictionary(x => x.ModelId, x => x.Index);

        // Scratch accumulators per valid-time — the covariate means are computed
        // after the read loop so a missing model row doesn't skew the average.
        var scratch = new Dictionary<DateTime, Scratch>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = reader.GetDateTime(0);
            var model = reader.GetString(1);
            var runTime = reader.GetDateTime(2);
            var precip = reader.IsDBNull(3) ? (double?)null : reader.GetDouble(3);
            var rh     = reader.IsDBNull(4) ? (double?)null : reader.GetDouble(4);
            var t2m    = reader.IsDBNull(5) ? (double?)null : reader.GetDouble(5);
            var td     = reader.IsDBNull(6) ? (double?)null : reader.GetDouble(6);
            var cL     = reader.IsDBNull(7) ? (double?)null : reader.GetDouble(7);
            var cM     = reader.IsDBNull(8) ? (double?)null : reader.GetDouble(8);
            var cH     = reader.IsDBNull(9) ? (double?)null : reader.GetDouble(9);
            var cape   = reader.IsDBNull(10) ? (double?)null : reader.GetDouble(10);
            var wind   = reader.IsDBNull(11) ? (double?)null : reader.GetDouble(11);
            var pres   = reader.IsDBNull(12) ? (double?)null : reader.GetDouble(12);

            if (!modelSlot.TryGetValue(model, out var slot))
                continue;

            if (!scratch.TryGetValue(valid, out var s))
            {
                s = new Scratch();
                scratch[valid] = s;
            }
            s.Precip[slot]   = precip;
            s.RunTime[slot]  = runTime;
            s.Dew[slot]      = td;
            s.Rh[slot]       = rh;
            s.Temp2m[slot]   = t2m;
            s.CloudLow[slot] = cL;
            s.CloudMid[slot] = cM;
            s.CloudHigh[slot]= cH;
            s.Cape[slot]     = cape;
            s.WindSpeed[slot]= wind;
            s.Pressure[slot] = pres;
        }

        return scratch.ToDictionary(
            kv => kv.Key,
            kv => new PivotedRow(
                Precip: kv.Value.Precip,
                RunTime: kv.Value.RunTime,
                Dew: kv.Value.Dew,
                Rh: kv.Value.Rh,
                Temp2m: kv.Value.Temp2m,
                Pressure: kv.Value.Pressure,
                RhMean:            MeanOfSlots(kv.Value.Rh),
                DewDepressionMean: MeanOfDepressions(kv.Value.Temp2m, kv.Value.Dew),
                CloudLowMean:      MeanOfSlots(kv.Value.CloudLow),
                CloudMidMean:      MeanOfSlots(kv.Value.CloudMid),
                CloudHighMean:     MeanOfSlots(kv.Value.CloudHigh),
                CapeMean:          MeanOfSlots(kv.Value.Cape),
                WindSpeedMean:     MeanOfSlots(kv.Value.WindSpeed)));
    }

    private sealed class Scratch
    {
        public double?[] Precip { get; } = new double?[8];
        public DateTime?[] RunTime { get; } = new DateTime?[8];
        public double?[] Dew { get; } = new double?[8];
        public double?[] Rh { get; } = new double?[8];
        public double?[] Temp2m { get; } = new double?[8];
        public double?[] CloudLow { get; } = new double?[8];
        public double?[] CloudMid { get; } = new double?[8];
        public double?[] CloudHigh { get; } = new double?[8];
        public double?[] Cape { get; } = new double?[8];
        public double?[] WindSpeed { get; } = new double?[8];
        public double?[] Pressure { get; } = new double?[8];
    }

    private static double MeanOfSlots(double?[] slots)
    {
        double sum = 0; int n = 0;
        foreach (var v in slots) if (v.HasValue) { sum += v.Value; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double MeanOfDepressions(double?[] temps, double?[] dews)
    {
        double sum = 0; int n = 0;
        for (int i = 0; i < temps.Length; i++)
        {
            if (temps[i].HasValue && dews[i].HasValue)
            {
                sum += temps[i]!.Value - dews[i]!.Value;
                n++;
            }
        }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double? NanToNull(float v) => float.IsNaN(v) ? null : v;

    /// <summary>
    /// Reverse-lookup the config rainfall-station friendly name from an "ea_..." slug
    /// so predict time can reach for the same hourly rainfall series the trainer used.
    /// </summary>
    private string? ResolveFriendlyStationName(string stationSlug)
    {
        foreach (var s in _cfg.Location.Rainfall.Stations)
        {
            var slug = "ea_" + Slugify(s.Name);
            if (string.Equals(slug, stationSlug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        }
        return null;
    }

    private static string Slugify(string name)
    {
        var chars = name.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '_')
            .ToArray();
        var slug = new string(chars);
        while (slug.Contains("__")) slug = slug.Replace("__", "_");
        return slug.Trim('_');
    }
}
