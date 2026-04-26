using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Config;
using WeatherBlend.Evaluate;
using WeatherBlend.Train;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Common;
using WeatherBlend.Train.Element.Humidity;

namespace WeatherBlend.Commands;

/// <summary>
/// One-off apples-to-apples bake-off between two saved Element model versions, on a
/// shared "UKMO-present" test set (chronologically last 15% of valid-times where
/// UKMO is non-null in the offset_day forecast tree).
///
/// Background: pattern 1 (drop UKMO entirely) and pattern 2 (require UKMO non-null)
/// train on different row sets and therefore evaluate against different test sets in
/// their own training reports. To compare them fairly we score both saved models
/// against the SAME rows.
///
/// CLI: <c>bakeoff --target humidity --version-1 v...x --version-2 v...y</c>
/// </summary>
public sealed class ElementBakeoffCommand
{
    private readonly ILogger<ElementBakeoffCommand> _log;
    private readonly AppConfig _cfg;

    public ElementBakeoffCommand(ILogger<ElementBakeoffCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(string element, string v1, string v2, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        switch (element)
        {
            case "humidity":
                return Task.FromResult(RunHumidity(modelsRoot, v1, v2, ct));
            case "cloud-cover":
                return Task.FromResult(RunCloud(modelsRoot, v1, v2, ct));
            case "temperature":
                return Task.FromResult(RunTemperature(modelsRoot, v1, v2, ct));
            case "temperature-2c":
                return Task.FromResult(RunTemperature2c(modelsRoot, v1, v2, ct));
            case "precipitation-3a-bellever":
                return Task.FromResult(RunPrecip3a(modelsRoot, "ea_bellever_dartmoor", "Bellever Dartmoor", v1, v2, ct));
            case "precipitation-3a-princetown":
                return Task.FromResult(RunPrecip3a(modelsRoot, "ea_princetown", "Princetown", v1, v2, ct));
            case "precipitation-3a-hexworthy":
                return Task.FromResult(RunPrecip3a(modelsRoot, "ea_dartmoor_nr_hexworthy", "Dartmoor nr Hexworthy", v1, v2, ct));
            case "precipitation-3c-bellever":
                return Task.FromResult(RunPrecip3c(modelsRoot, "ea_bellever_dartmoor", "Bellever Dartmoor", v1, v2, ct));
            case "precipitation-3c-princetown":
                return Task.FromResult(RunPrecip3c(modelsRoot, "ea_princetown", "Princetown", v1, v2, ct));
            case "precipitation-3c-hexworthy":
                return Task.FromResult(RunPrecip3c(modelsRoot, "ea_dartmoor_nr_hexworthy", "Dartmoor nr Hexworthy", v1, v2, ct));
            case "dry-window-3b-bellever-3h":
                return Task.FromResult(RunDryWindow3b(modelsRoot, "ea_bellever_dartmoor", "Bellever Dartmoor", 3, v1, v2, ct));
            case "dry-window-3b-bellever-6h":
                return Task.FromResult(RunDryWindow3b(modelsRoot, "ea_bellever_dartmoor", "Bellever Dartmoor", 6, v1, v2, ct));
            default:
                _log.LogError("Bakeoff currently supports humidity | cloud-cover | temperature | temperature-2c | precipitation-3a-bellever | precipitation-3c-bellever (got '{Target}')", element);
                return Task.FromResult(2);
        }
    }

    private int RunTemperature(string modelsRoot, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        var v1Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", v1);
        var v2Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", v2);

        _log.LogInformation("Temperature 2b bake-off: {V1} (pattern 1) vs {V2} (pattern 2)", v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            var rows = QueryTemperatureTestRows(lead, ct);
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            // Pattern 1 model: trained without UKMO. Score with UKMO field NaN'd
            // (matches its training distribution).
            var p1Rows = rows.Select(r =>
            {
                var x = CloneTemperature(r);
                x.TempUkmo = float.NaN;
                return x;
            }).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => (double)r.Era5Temp).ToArray();
            var p1Pred = TemperatureTrainer.Predict(ml1, m1, p1Rows);
            var p2Pred = TemperatureTrainer.Predict(ml2, m2, rows);

            var p1Mae = Metrics.Compute(p1Pred, truth).Mae;
            var p2Mae = Metrics.Compute(p2Pred, truth).Mae;

            var winner = p2Mae < p1Mae ? "Pattern 2" : (p1Mae < p2Mae ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Mae - p2Mae) / p1Mae;
            _log.LogInformation(
                "Temperature lead {Lead}h on {N} shared rows — Pattern 1 MAE {P1:0.000}°C  Pattern 2 MAE {P2:0.000}°C  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                lead, rows.Count, p1Mae, p2Mae, winner, deltaPct);
        }
        return 0;
    }

    private List<TrainingRow> QueryTemperatureTestRows(int leadHours, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var fcGlob = NormGlob(_cfg.Storage.ForecastsPath);
        var eraGlob = NormGlob(_cfg.Storage.Era5Path);

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, Temperature2m, WindDirection10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{_cfg.Location.Name}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {leadHours}
      AND Temperature2m IS NOT NULL
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN Temperature2m END) AS temp_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN Temperature2m END) AS temp_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN Temperature2m END) AS temp_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN Temperature2m END) AS temp_mf,
        MAX(CASE WHEN Model = 'ukmo_seamless'        THEN Temperature2m END) AS temp_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN Temperature2m END) AS temp_gem,
        AVG(WindDirection10m) AS wind_dir_mean
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, Temperature2m AS era5_temp
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{_cfg.Location.Name}' AND Temperature2m IS NOT NULL
)
SELECT
    p.ValidTimeUtc,
    p.temp_gfs, p.temp_ecmwf, p.temp_icon, p.temp_mf, p.temp_ukmo, p.temp_gem,
    p.wind_dir_mean, e.era5_temp
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE p.temp_gfs IS NOT NULL AND p.temp_ecmwf IS NOT NULL AND p.temp_icon IS NOT NULL
  AND p.temp_mf  IS NOT NULL AND p.temp_ukmo IS NOT NULL AND p.temp_gem IS NOT NULL
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var all = new List<TrainingRow>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var t = new double[6];
            for (int i = 0; i < 6; i++) t[i] = r.GetDouble(1 + i);
            var windDirMean = r.IsDBNull(7) ? double.NaN : r.GetDouble(7);
            var era5Temp = r.GetDouble(8);
            all.Add(FeatureBuilder.ComposeRow(valid, t, windDirMean, era5Temp));
        }
        var testStart = (int)Math.Floor(all.Count * 0.85);
        return all.Skip(testStart).ToList();
    }

    private int RunPrecip3a(string modelsRoot, string stationSlug, string stationName, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        // Precip artefacts live under a per-station subtree; helper takes the slug as the "target".
        var v1Dir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", stationSlug, v1);
        var v2Dir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", stationSlug, v2);

        _log.LogInformation("Precip 3a [{Station}] bake-off: {V1} (pattern 1) vs {V2} (pattern 2)", stationSlug, v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            // Reuse the live PrecipFeatureBuilder (pattern 3 — permissive). It hands us
            // every row that has the rainfall truth + at least one model — we'll filter
            // to UKMO-present below for apples-to-apples.
            var allRows = PrecipFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                _cfg.Location.Name, stationName, lead, ct);

            // Apples-to-apples test set: rows where UKMO precip is present (matches
            // pattern 2's training distribution). Take chronological tail 15%.
            var ukmoPresent = allRows.Where(r => !float.IsNaN(r.PrecipUkmo)).ToList();
            var testStart = (int)Math.Floor(ukmoPresent.Count * 0.85);
            var rows = ukmoPresent.Skip(testStart).ToList();
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            // Pattern 1 saw UKMO as NaN in training — score with UKMO fields force-NaN.
            var p1Rows = rows.Select(r => ClonePrecipWithUkmoNanned(r)).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => r.WetBinary ? 1.0 : 0.0).ToArray();
            var p1Pred = PrecipOccurrenceTrainer.PredictProbability(ml1, m1, p1Rows);
            var p2Pred = PrecipOccurrenceTrainer.PredictProbability(ml2, m2, rows);

            double Brier(double[] p) {
                double s = 0; for (int i = 0; i < p.Length; i++) { var d = p[i] - truth[i]; s += d * d; }
                return s / p.Length;
            }
            var p1Brier = Brier(p1Pred);
            var p2Brier = Brier(p2Pred);
            // BSS = 1 - Brier / Brier_climatology.
            var clim = truth.Average();
            var brierClim = truth.Sum(t => (clim - t) * (clim - t)) / truth.Length;
            var p1Bss = 1.0 - p1Brier / brierClim;
            var p2Bss = 1.0 - p2Brier / brierClim;

            var winner = p2Brier < p1Brier ? "Pattern 2" : (p1Brier < p2Brier ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Brier - p2Brier) / p1Brier;
            _log.LogInformation(
                "Precip 3a {Station} lead {Lead}h on {N} shared rows — Pattern 1 Brier {P1:0.0000} (BSS {B1:+0.000;-0.000;0.000}), Pattern 2 Brier {P2:0.0000} (BSS {B2:+0.000;-0.000;0.000})  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                stationSlug, lead, rows.Count, p1Brier, p1Bss, p2Brier, p2Bss, winner, deltaPct);
        }
        return 0;
    }

    private int RunDryWindow3b(string modelsRoot, string stationSlug, string stationName, int windowHours, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        // Dry-window artefacts: data/models/dry_window/{station}/window_{N}h/{version}/
        var key = $"{stationSlug}/window_{windowHours}h";
        var v1Dir = Path.Combine(modelsRoot, "dry_window", stationSlug, $"window_{windowHours}h", v1).Replace('\\', '/');
        var v2Dir = Path.Combine(modelsRoot, "dry_window", stationSlug, $"window_{windowHours}h", v2).Replace('\\', '/');

        _log.LogInformation("Dry-window 3b [{Key}] bake-off: {V1} (pattern 1) vs {V2} (production)", key, v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            var allRows = WeatherBlend.Train.DryWindow.DryWindowFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                _cfg.Location.Name, stationName, lead, windowHours, ct);

            // Apples-to-apples test set: rows where UKMO supplied the day (PrecipSumUkmo
            // not NaN means UKMO was complete for that day). Chronological tail 15%.
            var ukmoPresent = allRows.Where(r => !float.IsNaN(r.PrecipSumUkmo)).ToList();
            var testStart = (int)Math.Floor(ukmoPresent.Count * 0.85);
            var rows = ukmoPresent.Skip(testStart).ToList();
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            var p1Rows = rows.Select(CloneDryWindowWithUkmoNanned).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => r.HasDryWindow ? 1.0 : 0.0).ToArray();
            var p1Pred = WeatherBlend.Train.DryWindow.DryWindowTrainer.PredictProbability(ml1, m1, p1Rows);
            var p2Pred = WeatherBlend.Train.DryWindow.DryWindowTrainer.PredictProbability(ml2, m2, rows);

            double Brier(double[] p) {
                double s = 0; for (int i = 0; i < p.Length; i++) { var d = p[i] - truth[i]; s += d * d; }
                return s / p.Length;
            }
            var p1Brier = Brier(p1Pred);
            var p2Brier = Brier(p2Pred);
            var clim = truth.Average();
            var brierClim = truth.Sum(t => (clim - t) * (clim - t)) / truth.Length;
            var p1Bss = brierClim > 0 ? 1.0 - p1Brier / brierClim : double.NaN;
            var p2Bss = brierClim > 0 ? 1.0 - p2Brier / brierClim : double.NaN;

            var winner = p2Brier < p1Brier ? "Pattern 2" : (p1Brier < p2Brier ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Brier - p2Brier) / p1Brier;
            _log.LogInformation(
                "DryWindow 3b {Key} lead {Lead}h on {N} shared rows — Pattern 1 Brier {P1:0.0000} (BSS {B1:+0.000;-0.000;0.000}), Production Brier {P2:0.0000} (BSS {B2:+0.000;-0.000;0.000})  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                key, lead, rows.Count, p1Brier, p1Bss, p2Brier, p2Bss, winner, deltaPct);
        }
        return 0;
    }

    private static WeatherBlend.Train.DryWindow.DryWindowTrainingRow CloneDryWindowWithUkmoNanned(
        WeatherBlend.Train.DryWindow.DryWindowTrainingRow s) => new()
    {
        TargetDateUtc = s.TargetDateUtc, WindowHours = s.WindowHours,
        PrecipSumGfs = s.PrecipSumGfs, PrecipSumEcmwf = s.PrecipSumEcmwf, PrecipSumIcon = s.PrecipSumIcon,
        PrecipSumMf = s.PrecipSumMf, PrecipSumUkmo = float.NaN, PrecipSumGem = s.PrecipSumGem,
        PrecipMaxHourGfs = s.PrecipMaxHourGfs, PrecipMaxHourEcmwf = s.PrecipMaxHourEcmwf, PrecipMaxHourIcon = s.PrecipMaxHourIcon,
        PrecipMaxHourMf = s.PrecipMaxHourMf, PrecipMaxHourUkmo = float.NaN, PrecipMaxHourGem = s.PrecipMaxHourGem,
        WetHourCountGfs = s.WetHourCountGfs, WetHourCountEcmwf = s.WetHourCountEcmwf, WetHourCountIcon = s.WetHourCountIcon,
        WetHourCountMf = s.WetHourCountMf, WetHourCountUkmo = float.NaN, WetHourCountGem = s.WetHourCountGem,
        LongestDryRunGfs = s.LongestDryRunGfs, LongestDryRunEcmwf = s.LongestDryRunEcmwf, LongestDryRunIcon = s.LongestDryRunIcon,
        LongestDryRunMf = s.LongestDryRunMf, LongestDryRunUkmo = float.NaN, LongestDryRunGem = s.LongestDryRunGem,
        HasDryWindowGfs = s.HasDryWindowGfs, HasDryWindowEcmwf = s.HasDryWindowEcmwf, HasDryWindowIcon = s.HasDryWindowIcon,
        HasDryWindowMf = s.HasDryWindowMf, HasDryWindowUkmo = float.NaN, HasDryWindowGem = s.HasDryWindowGem,
        ProbMaxGfs = s.ProbMaxGfs, ProbMaxEcmwf = s.ProbMaxEcmwf, ProbMaxIcon = s.ProbMaxIcon,
        ProbMaxMf = s.ProbMaxMf, ProbMaxUkmo = float.NaN, ProbMaxGem = s.ProbMaxGem,
        PrecipSumMean = s.PrecipSumMean, PrecipSumStd = s.PrecipSumStd, PrecipSumMax = s.PrecipSumMax,
        AgreementHasDryWindow = s.AgreementHasDryWindow, LongestDryRunMean = s.LongestDryRunMean, WetHourCountMean = s.WetHourCountMean,
        RhMean = s.RhMean, RhMin = s.RhMin, DewDepressionMax = s.DewDepressionMax,
        CloudLowMean = s.CloudLowMean, CloudMidMean = s.CloudMidMean, CloudHighMean = s.CloudHighMean,
        CapeMax = s.CapeMax, WindMean = s.WindMean, WindMax = s.WindMax,
        DoySin = s.DoySin, DoyCos = s.DoyCos,
        FirstWetHour = s.FirstWetHour, LastWetHour = s.LastWetHour,
        LongestForecastDryRunHours = s.LongestForecastDryRunHours, LongestForecastWetRunHours = s.LongestForecastWetRunHours,
        NRainEvents = s.NRainEvents,
        MorningPrecipSum = s.MorningPrecipSum, AfternoonPrecipSum = s.AfternoonPrecipSum,
        HasDryWindow = s.HasDryWindow, PrecipMmDay = s.PrecipMmDay,
    };

    private int RunPrecip3c(string modelsRoot, string stationSlug, string stationName, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        var v1Dir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", stationSlug, v1);
        var v2Dir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", stationSlug, v2);

        _log.LogInformation("Precip 3c [{Station}] (rich) bake-off: {V1} (pattern 1) vs {V2} (pattern 2)", stationSlug, v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            var allRows = PrecipRichFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                _cfg.Location.Name, stationName, lead, ct);

            var ukmoPresent = allRows.Where(r => !float.IsNaN(r.PrecipUkmo)).ToList();
            var testStart = (int)Math.Floor(ukmoPresent.Count * 0.85);
            var rows = ukmoPresent.Skip(testStart).ToList();
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            var p1Rows = rows.Select(r => CloneRichPrecipWithUkmoNanned(r)).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => r.WetBinary ? 1.0 : 0.0).ToArray();
            var p1Pred = PrecipOccurrenceTrainer.PredictProbability(ml1, m1, p1Rows);
            var p2Pred = PrecipOccurrenceTrainer.PredictProbability(ml2, m2, rows);

            double Brier(double[] p) {
                double s = 0; for (int i = 0; i < p.Length; i++) { var d = p[i] - truth[i]; s += d * d; }
                return s / p.Length;
            }
            var p1Brier = Brier(p1Pred);
            var p2Brier = Brier(p2Pred);
            var clim = truth.Average();
            var brierClim = truth.Sum(t => (clim - t) * (clim - t)) / truth.Length;
            var p1Bss = 1.0 - p1Brier / brierClim;
            var p2Bss = 1.0 - p2Brier / brierClim;

            var winner = p2Brier < p1Brier ? "Pattern 2" : (p1Brier < p2Brier ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Brier - p2Brier) / p1Brier;
            _log.LogInformation(
                "Precip 3c {Station} lead {Lead}h on {N} shared rows — Pattern 1 Brier {P1:0.0000} (BSS {B1:+0.000;-0.000;0.000}), Pattern 2 Brier {P2:0.0000} (BSS {B2:+0.000;-0.000;0.000})  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                stationSlug, lead, rows.Count, p1Brier, p1Bss, p2Brier, p2Bss, winner, deltaPct);
        }
        return 0;
    }

    private static RichPrecipTrainingRow CloneRichPrecipWithUkmoNanned(RichPrecipTrainingRow s) => new()
    {
        ValidTimeUtc = s.ValidTimeUtc,
        PrecipGfs = s.PrecipGfs, PrecipEcmwf = s.PrecipEcmwf, PrecipIcon = s.PrecipIcon,
        PrecipMf = s.PrecipMf, PrecipUkmo = float.NaN, PrecipGem = s.PrecipGem,
        ProbGfs = s.ProbGfs, ProbEcmwf = s.ProbEcmwf, ProbIcon = s.ProbIcon,
        ProbMf = s.ProbMf, ProbUkmo = float.NaN, ProbGem = s.ProbGem,
        PrecipMean = s.PrecipMean, PrecipStd = s.PrecipStd, PrecipMax = s.PrecipMax,
        PrecipAgreementWet01 = s.PrecipAgreementWet01,
        RhMean = s.RhMean, DewDepressionMean = s.DewDepressionMean,
        CloudLowMean = s.CloudLowMean, CloudMidMean = s.CloudMidMean, CloudHighMean = s.CloudHighMean,
        CapeMean = s.CapeMean, WindSpeedMean = s.WindSpeedMean,
        HourSin = s.HourSin, HourCos = s.HourCos, DoySin = s.DoySin, DoyCos = s.DoyCos,
        WetBinary = s.WetBinary, PrecipMmHour = s.PrecipMmHour,
        DewGfs = s.DewGfs, DewEcmwf = s.DewEcmwf, DewIcon = s.DewIcon,
        DewMf = s.DewMf, DewUkmo = float.NaN, DewGem = s.DewGem,
        RhGfs = s.RhGfs, RhEcmwf = s.RhEcmwf, RhIcon = s.RhIcon,
        RhMf = s.RhMf, RhUkmo = float.NaN, RhGem = s.RhGem,
        DewDepressionGfs = s.DewDepressionGfs, DewDepressionEcmwf = s.DewDepressionEcmwf, DewDepressionIcon = s.DewDepressionIcon,
        DewDepressionMf = s.DewDepressionMf, DewDepressionUkmo = float.NaN, DewDepressionGem = s.DewDepressionGem,
        PressureGfs = s.PressureGfs, PressureEcmwf = s.PressureEcmwf, PressureIcon = s.PressureIcon,
        PressureMf = s.PressureMf, PressureUkmo = float.NaN, PressureGem = s.PressureGem,
        EaRainPrev24hMm = s.EaRainPrev24hMm, EaRainPrev72hMm = s.EaRainPrev72hMm,
        EaWetHoursLast24h = s.EaWetHoursLast24h, EaDryHoursTrailing = s.EaDryHoursTrailing,
    };

    private static PrecipTrainingRow ClonePrecipWithUkmoNanned(PrecipTrainingRow s) => new()
    {
        ValidTimeUtc = s.ValidTimeUtc,
        PrecipGfs = s.PrecipGfs, PrecipEcmwf = s.PrecipEcmwf, PrecipIcon = s.PrecipIcon,
        PrecipMf = s.PrecipMf, PrecipUkmo = float.NaN, PrecipGem = s.PrecipGem,
        ProbGfs = s.ProbGfs, ProbEcmwf = s.ProbEcmwf, ProbIcon = s.ProbIcon,
        ProbMf = s.ProbMf, ProbUkmo = float.NaN, ProbGem = s.ProbGem,
        PrecipMean = s.PrecipMean, PrecipStd = s.PrecipStd, PrecipMax = s.PrecipMax,
        PrecipAgreementWet01 = s.PrecipAgreementWet01,
        RhMean = s.RhMean, DewDepressionMean = s.DewDepressionMean,
        CloudLowMean = s.CloudLowMean, CloudMidMean = s.CloudMidMean, CloudHighMean = s.CloudHighMean,
        CapeMean = s.CapeMean, WindSpeedMean = s.WindSpeedMean,
        HourSin = s.HourSin, HourCos = s.HourCos, DoySin = s.DoySin, DoyCos = s.DoyCos,
        WetBinary = s.WetBinary, PrecipMmHour = s.PrecipMmHour,
    };

    private int RunTemperature2c(string modelsRoot, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        var v1Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", v1);
        var v2Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", v2);

        _log.LogInformation("Temperature 2c (rich, 88 features) bake-off: {V1} (pattern 1) vs {V2} (pattern 2)", v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            var rows = QueryRichTestRows(lead, ct);
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            // Pattern 1 model: trained without UKMO. Score with all UKMO fields NaN'd
            // (matches its training distribution).
            var p1Rows = rows.Select(r => CloneRichWithUkmoNanned(r)).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => (double)r.Era5Temp).ToArray();
            var p1Pred = TemperatureTrainer.Predict(ml1, m1, p1Rows);
            var p2Pred = TemperatureTrainer.Predict(ml2, m2, rows);

            var p1Mae = Metrics.Compute(p1Pred, truth).Mae;
            var p2Mae = Metrics.Compute(p2Pred, truth).Mae;

            var winner = p2Mae < p1Mae ? "Pattern 2" : (p1Mae < p2Mae ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Mae - p2Mae) / p1Mae;
            _log.LogInformation(
                "Temperature-2c lead {Lead}h on {N} shared rows — Pattern 1 MAE {P1:0.000}°C  Pattern 2 MAE {P2:0.000}°C  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                lead, rows.Count, p1Mae, p2Mae, winner, deltaPct);
        }
        return 0;
    }

    private List<RichTrainingRow> QueryRichTestRows(int leadHours, CancellationToken ct)
    {
        // Reuse the pattern-2 RichFeatureBuilder.BuildForLead — it already requires
        // UKMO non-null (the production WHERE clause), so what it returns is exactly
        // the apples-to-apples test set we want.
        var all = RichFeatureBuilder.BuildForLead(
            _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name, leadHours, ct);
        var testStart = (int)Math.Floor(all.Count * 0.85);
        return all.Skip(testStart).ToList();
    }

    /// <summary>
    /// Clone a RichTrainingRow with every UKMO field forced to NaN — matches what
    /// pattern 1 saw in training (the SQL pivots NULL for every UKMO column).
    /// </summary>
    private static RichTrainingRow CloneRichWithUkmoNanned(RichTrainingRow s) => new()
    {
        ValidTimeUtc = s.ValidTimeUtc,
        TempGfs = s.TempGfs, TempEcmwf = s.TempEcmwf, TempIcon = s.TempIcon,
        TempMf = s.TempMf, TempUkmo = float.NaN, TempGem = s.TempGem,
        TempMean = s.TempMean, TempStd = s.TempStd, TempRange = s.TempRange,
        HourSin = s.HourSin, HourCos = s.HourCos, DoySin = s.DoySin, DoyCos = s.DoyCos,
        WindDirMean = s.WindDirMean,
        Era5Temp = s.Era5Temp,
        DewGfs = s.DewGfs, DewEcmwf = s.DewEcmwf, DewIcon = s.DewIcon,
        DewMf = s.DewMf, DewUkmo = float.NaN, DewGem = s.DewGem,
        RhGfs = s.RhGfs, RhEcmwf = s.RhEcmwf, RhIcon = s.RhIcon,
        RhMf = s.RhMf, RhUkmo = float.NaN, RhGem = s.RhGem,
        CloudGfs = s.CloudGfs, CloudEcmwf = s.CloudEcmwf, CloudIcon = s.CloudIcon,
        CloudMf = s.CloudMf, CloudUkmo = float.NaN, CloudGem = s.CloudGem,
        CloudLowGfs = s.CloudLowGfs, CloudLowEcmwf = s.CloudLowEcmwf, CloudLowIcon = s.CloudLowIcon,
        CloudLowMf = s.CloudLowMf, CloudLowUkmo = float.NaN, CloudLowGem = s.CloudLowGem,
        CloudMidGfs = s.CloudMidGfs, CloudMidEcmwf = s.CloudMidEcmwf, CloudMidIcon = s.CloudMidIcon,
        CloudMidMf = s.CloudMidMf, CloudMidUkmo = float.NaN, CloudMidGem = s.CloudMidGem,
        CloudHighGfs = s.CloudHighGfs, CloudHighEcmwf = s.CloudHighEcmwf, CloudHighIcon = s.CloudHighIcon,
        CloudHighMf = s.CloudHighMf, CloudHighUkmo = float.NaN, CloudHighGem = s.CloudHighGem,
        WindSpeedGfs = s.WindSpeedGfs, WindSpeedEcmwf = s.WindSpeedEcmwf, WindSpeedIcon = s.WindSpeedIcon,
        WindSpeedMf = s.WindSpeedMf, WindSpeedUkmo = float.NaN, WindSpeedGem = s.WindSpeedGem,
        WindDirSinGfs = s.WindDirSinGfs, WindDirSinEcmwf = s.WindDirSinEcmwf, WindDirSinIcon = s.WindDirSinIcon,
        WindDirSinMf = s.WindDirSinMf, WindDirSinUkmo = float.NaN, WindDirSinGem = s.WindDirSinGem,
        WindDirCosGfs = s.WindDirCosGfs, WindDirCosEcmwf = s.WindDirCosEcmwf, WindDirCosIcon = s.WindDirCosIcon,
        WindDirCosMf = s.WindDirCosMf, WindDirCosUkmo = float.NaN, WindDirCosGem = s.WindDirCosGem,
        WindGustsGfs = s.WindGustsGfs, WindGustsEcmwf = s.WindGustsEcmwf, WindGustsIcon = s.WindGustsIcon,
        WindGustsMf = s.WindGustsMf, WindGustsUkmo = float.NaN, WindGustsGem = s.WindGustsGem,
        PressureGfs = s.PressureGfs, PressureEcmwf = s.PressureEcmwf, PressureIcon = s.PressureIcon,
        PressureMf = s.PressureMf, PressureUkmo = float.NaN, PressureGem = s.PressureGem,
        DewMean = s.DewMean, DewStd = s.DewStd,
        RhMean = s.RhMean, RhStd = s.RhStd,
        CloudMean = s.CloudMean,
        WindSpeedMean = s.WindSpeedMean, WindSpeedStd = s.WindSpeedStd,
        PressureMean = s.PressureMean, PressureStd = s.PressureStd,
    };

    private static TrainingRow CloneTemperature(TrainingRow r) => new()
    {
        ValidTimeUtc = r.ValidTimeUtc,
        TempGfs = r.TempGfs, TempEcmwf = r.TempEcmwf, TempIcon = r.TempIcon,
        TempMf = r.TempMf, TempUkmo = r.TempUkmo, TempGem = r.TempGem,
        TempMean = r.TempMean, TempStd = r.TempStd, TempRange = r.TempRange,
        HourSin = r.HourSin, HourCos = r.HourCos, DoySin = r.DoySin, DoyCos = r.DoyCos,
        WindDirMean = r.WindDirMean,
        Era5Temp = r.Era5Temp,
    };

    private int RunHumidity(string modelsRoot, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        var v1Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "humidity", v1);
        var v2Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "humidity", v2);

        _log.LogInformation("Humidity bake-off: {V1} (pattern 1) vs {V2} (pattern 2)", v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            var rows = QueryHumidityTestRows(lead, ct);
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            // Pattern 1 model: trained without UKMO. Score with UKMO fields force-NaN
            // (matches its training distribution).
            var p1Rows = rows.Select(r =>
            {
                var x = CloneHumidity(r);
                x.RhUkmo = float.NaN; x.DpUkmo = float.NaN;
                return x;
            }).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => (double)r.Era5Rh).ToArray();
            var p1Pred = TemperatureTrainer.Predict(ml1, m1, p1Rows);
            var p2Pred = TemperatureTrainer.Predict(ml2, m2, rows);

            var p1Mae = Metrics.Compute(p1Pred, truth).Mae;
            var p2Mae = Metrics.Compute(p2Pred, truth).Mae;

            var winner = p2Mae < p1Mae ? "Pattern 2" : (p1Mae < p2Mae ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Mae - p2Mae) / p1Mae;
            _log.LogInformation(
                "Humidity lead {Lead}h on {N} shared rows — Pattern 1 MAE {P1:0.000}%  Pattern 2 MAE {P2:0.000}%  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                lead, rows.Count, p1Mae, p2Mae, winner, deltaPct);
        }
        return 0;
    }

    private int RunCloud(string modelsRoot, string v1, string v2, CancellationToken ct)
    {
        var ml1 = new MLContext(seed: 42);
        var ml2 = new MLContext(seed: 42);
        var v1Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "cloud_cover", v1);
        var v2Dir = ModelArtifact.ResolveVersionDir(modelsRoot, "cloud_cover", v2);

        _log.LogInformation("Cloud bake-off: {V1} (pattern 1) vs {V2} (pattern 2)", v1, v2);

        foreach (var lead in new[] { 24, 48, 72 })
        {
            ct.ThrowIfCancellationRequested();
            var rows = QueryCloudTestRows(lead, ct);
            if (rows.Count == 0) { _log.LogWarning("Lead {Lead}h: no test rows.", lead); continue; }

            var p1Rows = rows.Select(r =>
            {
                var x = CloneCloud(r);
                x.CcUkmo = float.NaN;
                return x;
            }).ToList();

            var m1 = ModelArtifact.LoadLeadModel(ml1, v1Dir, lead, out _);
            var m2 = ModelArtifact.LoadLeadModel(ml2, v2Dir, lead, out _);

            var truth = rows.Select(r => (double)r.Era5Cc).ToArray();
            var p1Pred = TemperatureTrainer.Predict(ml1, m1, p1Rows);
            var p2Pred = TemperatureTrainer.Predict(ml2, m2, rows);

            var p1Mae = Metrics.Compute(p1Pred, truth).Mae;
            var p2Mae = Metrics.Compute(p2Pred, truth).Mae;

            var winner = p2Mae < p1Mae ? "Pattern 2" : (p1Mae < p2Mae ? "Pattern 1" : "tie");
            var deltaPct = 100.0 * (p1Mae - p2Mae) / p1Mae;
            _log.LogInformation(
                "Cloud lead {Lead}h on {N} shared rows — Pattern 1 MAE {P1:0.000}%  Pattern 2 MAE {P2:0.000}%  → {Winner} (Δ={Delta:+0.0;-0.0;0.0}% relative)",
                lead, rows.Count, p1Mae, p2Mae, winner, deltaPct);
        }
        return 0;
    }

    /// <summary>
    /// Build the shared humidity test set: latest 15% of valid times where ALL 6 models
    /// (including UKMO) have RH+DP non-null. Mirrors HumidityFeatureBuilder's pattern-2
    /// SQL exactly so it matches what pattern 2's model was trained on.
    /// </summary>
    private List<HumidityRow> QueryHumidityTestRows(int leadHours, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var fcGlob = NormGlob(_cfg.Storage.ForecastsPath);
        var eraGlob = NormGlob(_cfg.Storage.Era5Path);

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RelativeHumidity2m AS rh, DewPoint2m AS dp,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{_cfg.Location.Name}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {leadHours}
      AND RelativeHumidity2m IS NOT NULL
      AND DewPoint2m IS NOT NULL
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN rh END) AS rh_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN rh END) AS rh_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN rh END) AS rh_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN rh END) AS rh_mf,
        MAX(CASE WHEN Model = 'ukmo_seamless'        THEN rh END) AS rh_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN rh END) AS rh_gem,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN dp END) AS dp_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN dp END) AS dp_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN dp END) AS dp_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN dp END) AS dp_mf,
        MAX(CASE WHEN Model = 'ukmo_seamless'        THEN dp END) AS dp_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN dp END) AS dp_gem
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, RelativeHumidity2m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{_cfg.Location.Name}' AND RelativeHumidity2m IS NOT NULL
)
SELECT
    p.ValidTimeUtc,
    p.rh_gfs, p.rh_ecmwf, p.rh_icon, p.rh_mf, p.rh_ukmo, p.rh_gem,
    p.dp_gfs, p.dp_ecmwf, p.dp_icon, p.dp_mf, p.dp_ukmo, p.dp_gem,
    e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE p.rh_gfs IS NOT NULL AND p.rh_ecmwf IS NOT NULL AND p.rh_icon IS NOT NULL
  AND p.rh_mf IS NOT NULL AND p.rh_ukmo IS NOT NULL AND p.rh_gem IS NOT NULL
  AND p.dp_gfs IS NOT NULL AND p.dp_ecmwf IS NOT NULL AND p.dp_icon IS NOT NULL
  AND p.dp_mf IS NOT NULL AND p.dp_ukmo IS NOT NULL AND p.dp_gem IS NOT NULL
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var all = new List<HumidityRow>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var rh = new float[6];
            for (int i = 0; i < 6; i++) rh[i] = (float)r.GetDouble(1 + i);
            var dp = new float[6];
            for (int i = 0; i < 6; i++) dp[i] = (float)r.GetDouble(7 + i);
            var truth = (float)r.GetDouble(13);
            all.Add(HumidityFeatureBuilder.ComposeRow(valid, rh, dp, truth));
        }
        // Take chronological tail 15% — matches the train/val/test split convention.
        var testStart = (int)Math.Floor(all.Count * 0.85);
        return all.Skip(testStart).ToList();
    }

    private List<CloudRow> QueryCloudTestRows(int leadHours, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var fcGlob = NormGlob(_cfg.Storage.ForecastsPath);
        var eraGlob = NormGlob(_cfg.Storage.Era5Path);

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, CloudCover AS cc,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{_cfg.Location.Name}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {leadHours}
      AND CloudCover IS NOT NULL
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        MAX(CASE WHEN Model = 'gfs_seamless'         THEN cc END) AS cc_gfs,
        MAX(CASE WHEN Model = 'ecmwf_ifs025'         THEN cc END) AS cc_ecmwf,
        MAX(CASE WHEN Model = 'icon_seamless'        THEN cc END) AS cc_icon,
        MAX(CASE WHEN Model = 'meteofrance_seamless' THEN cc END) AS cc_mf,
        MAX(CASE WHEN Model = 'ukmo_seamless'        THEN cc END) AS cc_ukmo,
        MAX(CASE WHEN Model = 'gem_seamless'         THEN cc END) AS cc_gem
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, CloudCover AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{_cfg.Location.Name}' AND CloudCover IS NOT NULL
)
SELECT
    p.ValidTimeUtc,
    p.cc_gfs, p.cc_ecmwf, p.cc_icon, p.cc_mf, p.cc_ukmo, p.cc_gem,
    e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE p.cc_gfs IS NOT NULL AND p.cc_ecmwf IS NOT NULL AND p.cc_icon IS NOT NULL
  AND p.cc_mf IS NOT NULL AND p.cc_ukmo IS NOT NULL AND p.cc_gem IS NOT NULL
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var all = new List<CloudRow>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var cc = new float[6];
            for (int i = 0; i < 6; i++) cc[i] = (float)r.GetDouble(1 + i);
            var truth = (float)r.GetDouble(7);
            all.Add(CloudFeatureBuilder.ComposeRow(valid, cc, truth));
        }
        var testStart = (int)Math.Floor(all.Count * 0.85);
        return all.Skip(testStart).ToList();
    }

    private static string NormGlob(string root) =>
        Path.Combine(root, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");

    private static HumidityRow CloneHumidity(HumidityRow r) => new()
    {
        ValidTimeUtc = r.ValidTimeUtc,
        RhGfs = r.RhGfs, RhEcmwf = r.RhEcmwf, RhIcon = r.RhIcon, RhMf = r.RhMf, RhUkmo = r.RhUkmo, RhGem = r.RhGem,
        DpGfs = r.DpGfs, DpEcmwf = r.DpEcmwf, DpIcon = r.DpIcon, DpMf = r.DpMf, DpUkmo = r.DpUkmo, DpGem = r.DpGem,
        RhMean = r.RhMean, RhStd = r.RhStd, RhRange = r.RhRange,
        HourSin = r.HourSin, HourCos = r.HourCos, DoySin = r.DoySin, DoyCos = r.DoyCos,
        Era5Rh = r.Era5Rh,
    };

    private static CloudRow CloneCloud(CloudRow r) => new()
    {
        ValidTimeUtc = r.ValidTimeUtc,
        CcGfs = r.CcGfs, CcEcmwf = r.CcEcmwf, CcIcon = r.CcIcon, CcMf = r.CcMf, CcUkmo = r.CcUkmo, CcGem = r.CcGem,
        CcMean = r.CcMean, CcStd = r.CcStd, CcRange = r.CcRange,
        HourSin = r.HourSin, HourCos = r.HourCos, DoySin = r.DoySin, DoyCos = r.DoyCos,
        Era5Cc = r.Era5Cc,
    };
}
