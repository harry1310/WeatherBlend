using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;

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

    private static readonly int[] DefaultLeads = { 24, 48, 72 };

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
            var wrote = await RunStationAsync(
                modelsRoot, station, modelVersion, predictionMadeAt, anchor, targets, perValid, ct);
            anyWritten |= wrote;
        }

        return anyWritten ? 0 : 3;
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

        var ml = new MLContext(seed: 42);
        var predictions = new List<PrecipPredictionRow>();

        foreach (var (lead, valid) in targets)
        {
            if (!perValid.TryGetValue(valid, out var pivot))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no forecast rows for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }

            // Require at least one non-null per-model precip — matches the training
            // filter. If every model is silent there's nothing to blend.
            if (!pivot.Precip.Any(p => p.HasValue))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: all six per-model precip values null for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }

            var precip = pivot.Precip.Select(p => p ?? double.NaN).ToArray();
            var prob   = pivot.Prob.Select(p => p ?? double.NaN).ToArray();

            // truthMmHour is ignored at predict time (the trainer only reads features);
            // 0 keeps WetBinary=false in the composed row.
            var featureRow = PrecipFeatureBuilder.ComposeRow(
                valid, precip, prob,
                rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                cloudHighMean: pivot.CloudHighMean,
                capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                truthMmHour: 0.0);

            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var pWet = PrecipOccurrenceTrainer.PredictProbability(ml, model, new[] { featureRow });
            var climPWet = climatology.Predict(valid);

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
                PrecipGfs   = pivot.Precip[0], PrecipEcmwf = pivot.Precip[1],
                PrecipIcon  = pivot.Precip[2], PrecipMf    = pivot.Precip[3],
                PrecipUkmo  = pivot.Precip[4], PrecipGem   = pivot.Precip[5],
                ProbGfsModel   = pivot.Prob[0], ProbEcmwfModel = pivot.Prob[1],
                ProbIconModel  = pivot.Prob[2], ProbMfModel    = pivot.Prob[3],
                ProbUkmoModel  = pivot.Prob[4], ProbGemModel   = pivot.Prob[5],
                RunTimeGfs   = pivot.RunTime[0], RunTimeEcmwf = pivot.RunTime[1],
                RunTimeIcon  = pivot.RunTime[2], RunTimeMf    = pivot.RunTime[3],
                RunTimeUkmo  = pivot.RunTime[4], RunTimeGem   = pivot.RunTime[5],
                PrecipMean = NanToNull(featureRow.PrecipMean),
                PrecipStd  = NanToNull(featureRow.PrecipStd),
                PrecipMax  = NanToNull(featureRow.PrecipMax),
                PrecipAgreementWet01 = NanToNull(featureRow.PrecipAgreementWet01),
                FeatureVectorHash = HashFeatures(featureRow),
            });

            _log.LogInformation(
                "Station {Station} lead {Lead}h → P(wet) {P:0.000} (clim {Clim:0.000}, valid {Valid:yyyy-MM-dd HH:mm}Z, agreement {Ag:0.00})",
                station, lead, pWet[0], climPWet, valid, featureRow.PrecipAgreementWet01);
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
    // precip feature set required by the occurrence blender.
    private sealed record PivotedRow(
        double?[] Precip,
        double?[] Prob,
        DateTime?[] RunTime,
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
           Precipitation, PrecipitationProbability,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           Cape, WindSpeed10m,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {liveCycleFilter}
)
SELECT ValidTimeUtc, Model, RunTimeUtc,
       Precipitation, PrecipitationProbability,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       Cape, WindSpeed10m
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
            var prob   = reader.IsDBNull(4) ? (double?)null : reader.GetDouble(4);
            var rh     = reader.IsDBNull(5) ? (double?)null : reader.GetDouble(5);
            var t2m    = reader.IsDBNull(6) ? (double?)null : reader.GetDouble(6);
            var td     = reader.IsDBNull(7) ? (double?)null : reader.GetDouble(7);
            var cL     = reader.IsDBNull(8) ? (double?)null : reader.GetDouble(8);
            var cM     = reader.IsDBNull(9) ? (double?)null : reader.GetDouble(9);
            var cH     = reader.IsDBNull(10) ? (double?)null : reader.GetDouble(10);
            var cape   = reader.IsDBNull(11) ? (double?)null : reader.GetDouble(11);
            var wind   = reader.IsDBNull(12) ? (double?)null : reader.GetDouble(12);

            if (!modelSlot.TryGetValue(model, out var slot))
                continue;

            if (!scratch.TryGetValue(valid, out var s))
            {
                s = new Scratch();
                scratch[valid] = s;
            }
            s.Precip[slot] = precip;
            s.Prob[slot]   = prob;
            s.RunTime[slot] = runTime;

            if (rh.HasValue)   s.RhList.Add(rh.Value);
            if (t2m.HasValue && td.HasValue) s.DewDepList.Add(t2m.Value - td.Value);
            if (cL.HasValue)   s.CLList.Add(cL.Value);
            if (cM.HasValue)   s.CMList.Add(cM.Value);
            if (cH.HasValue)   s.CHList.Add(cH.Value);
            if (cape.HasValue) s.CapeList.Add(cape.Value);
            if (wind.HasValue) s.WindList.Add(wind.Value);
        }

        return scratch.ToDictionary(
            kv => kv.Key,
            kv => new PivotedRow(
                Precip: kv.Value.Precip,
                Prob: kv.Value.Prob,
                RunTime: kv.Value.RunTime,
                RhMean:             Mean(kv.Value.RhList),
                DewDepressionMean:  Mean(kv.Value.DewDepList),
                CloudLowMean:       Mean(kv.Value.CLList),
                CloudMidMean:       Mean(kv.Value.CMList),
                CloudHighMean:      Mean(kv.Value.CHList),
                CapeMean:           Mean(kv.Value.CapeList),
                WindSpeedMean:      Mean(kv.Value.WindList)));
    }

    private sealed class Scratch
    {
        public double?[] Precip { get; } = new double?[6];
        public double?[] Prob { get; } = new double?[6];
        public DateTime?[] RunTime { get; } = new DateTime?[6];
        public List<double> RhList { get; } = new();
        public List<double> DewDepList { get; } = new();
        public List<double> CLList { get; } = new();
        public List<double> CMList { get; } = new();
        public List<double> CHList { get; } = new();
        public List<double> CapeList { get; } = new();
        public List<double> WindList { get; } = new();
    }

    private static double Mean(List<double> xs) => xs.Count == 0 ? double.NaN : xs.Average();

    private static double? NanToNull(float v) => float.IsNaN(v) ? null : v;

    /// <summary>SHA-256 hex of the 27 feature floats in OccurrenceFeatureNames order.</summary>
    public static string HashFeatures(PrecipTrainingRow row) => FeatureHashing.HashFloats(new[]
    {
        row.PrecipGfs, row.PrecipEcmwf, row.PrecipIcon, row.PrecipMf, row.PrecipUkmo, row.PrecipGem,
        row.ProbGfs,   row.ProbEcmwf,   row.ProbIcon,   row.ProbMf,   row.ProbUkmo,   row.ProbGem,
        row.PrecipMean, row.PrecipStd, row.PrecipMax, row.PrecipAgreementWet01,
        row.RhMean, row.DewDepressionMean,
        row.CloudLowMean, row.CloudMidMean, row.CloudHighMean,
        row.CapeMean, row.WindSpeedMean,
        row.HourSin, row.HourCos, row.DoySin, row.DoyCos,
    });

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
