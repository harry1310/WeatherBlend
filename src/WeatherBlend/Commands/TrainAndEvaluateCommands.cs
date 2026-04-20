using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 2a: train a bucket-free temperature blender (see docs/LEAD_TIME_BACKFILL.md
/// for why no lead-time buckets). Per-lead-time training is deferred to phase 3.
/// </summary>
public sealed class TrainCommand
{
    private readonly ILogger<TrainCommand> _log;
    private readonly AppConfig _cfg;

    public TrainCommand(ILogger<TrainCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string target, CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported in phase 2a (got '{Target}')", target);
            return 2;
        }

        _log.LogInformation("Loading feature dataset from local parquet...");
        var rows = FeatureBuilder.Build(
            _cfg.Storage.ForecastsPath,
            _cfg.Storage.Era5Path,
            _cfg.Location.Name,
            ct);
        _log.LogInformation("Loaded {N} joined rows spanning {Start:yyyy-MM-dd} → {End:yyyy-MM-dd}",
            rows.Count,
            rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
            rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

        if (rows.Count < 100)
        {
            _log.LogError("Only {N} joined rows available — too few to train. Run backfill first.", rows.Count);
            return 3;
        }

        var ds = TrainingDataset.Split(rows);
        _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                            "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                            "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
            ds.Train.Count, ds.TrainStart, ds.TrainEnd,
            ds.Val.Count,   ds.ValStart,   ds.ValEnd,
            ds.Test.Count,  ds.TestStart,  ds.TestEnd);

        var hp = new TemperatureTrainer.Hyperparameters();
        _log.LogInformation("Training LightGBM... iter={Iter} lr={Lr} leaves={Leaves} esr={Esr}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound);
        var trained = TemperatureTrainer.Train(ds, hp);

        // Test MAE (headline — no tuning decisions use this number).
        var testPred = TemperatureTrainer.Predict(trained.Ml, trained.Model, ds.Test);
        var testActual = ds.Test.Select(x => (double)x.Era5Temp).ToArray();
        var testStats = Metrics.Compute(testPred, testActual);

        // Baselines for quick-look in the training log; full report lives under `evaluate`.
        var best = Baselines.BestSingle(ds.Val);
        var bestPred = Baselines.SingleModel(ds.Test, best);
        var bestStats = Metrics.Compute(bestPred, testActual);
        var mean = Baselines.MeanOfModels(ds.Test);
        var meanStats = Metrics.Compute(mean, testActual);

        _log.LogInformation("Headline test MAE — blend={Blend:0.000}°C, best_single({Best})={BestMae:0.000}°C, mean_of_models={MeanMae:0.000}°C",
            testStats.Mae, best, bestStats.Mae, meanStats.Mae);

        // Persist artefacts.
        var now = DateTime.UtcNow;
        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now);
        var versionName = Path.GetFileName(versionDir);

        ModelArtifact.SaveModel(trained.Ml, trained.Model, trained.InputSchema, versionDir);
        ModelArtifact.SaveFeatureSchema(versionDir, trained.FeatureNames);
        ModelArtifact.SaveFeatureImportance(versionDir, trained.FeatureImportance);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2a",
            TrainedAtUtc = now,
            DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd HH:mm}Z → {ds.TrainEnd:yyyy-MM-dd HH:mm}Z",
            DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd HH:mm}Z → {ds.ValEnd:yyyy-MM-dd HH:mm}Z",
            DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd HH:mm}Z → {ds.TestEnd:yyyy-MM-dd HH:mm}Z",
            TrainRows = ds.Train.Count,
            ValRows   = ds.Val.Count,
            TestRows  = ds.Test.Count,
            Hyperparameters = new Dictionary<string, object>
            {
                ["numberOfIterations"]           = hp.NumberOfIterations,
                ["learningRate"]                 = hp.LearningRate,
                ["numberOfLeaves"]               = hp.NumberOfLeaves,
                ["minimumExampleCountPerLeaf"]   = hp.MinimumExampleCountPerLeaf,
                ["l1Regularization"]             = hp.L1Regularization,
                ["l2Regularization"]             = hp.L2Regularization,
                ["earlyStoppingRound"]           = hp.EarlyStoppingRound,
                ["seed"]                         = hp.Seed,
                ["evaluationMetric"]             = "MeanAbsoluteError",
                ["objective"]                    = "regression (L2) — Microsoft.ML.LightGbm 4.0 does not expose regression_l1",
            },
            TestMae = new Dictionary<string, double>
            {
                ["blend"]          = testStats.Mae,
                [$"best_single[{best}]"] = bestStats.Mae,
                ["mean_of_models"] = meanStats.Mae,
            },
            DeviationsFromBrief = new List<string>
            {
                "Objective is L2 (squared error); MAE used only as early-stopping metric. Microsoft.ML.LightGbm 4.0 does not expose regression_l1.",
                "No monotone constraints on per-model temperature inputs. Microsoft.ML.LightGbm 4.0 does not expose monotone_constraints.",
                "No lead-time bucketing. Historical-forecast API lacks real issue times; phase 3 will address this (see docs/LEAD_TIME_BACKFILL.md).",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        ModelArtifact.UpdateManifest(modelsRoot, "temperature", versionName);

        _log.LogInformation("Saved model → {Dir}", versionDir);

        // Suppress unused-warning for async caller signature.
        await Task.CompletedTask;
        return 0;
    }
}

public sealed class EvaluateCommand
{
    private readonly ILogger<EvaluateCommand> _log;
    private readonly AppConfig _cfg;

    public EvaluateCommand(ILogger<EvaluateCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string target, string modelVersion, CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported in phase 2a (got '{Target}')", target);
            return 2;
        }

        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", modelVersion);
        _log.LogInformation("Loading model from {Dir}", versionDir);

        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        var ml = new Microsoft.ML.MLContext(seed: 42);
        var model = ModelArtifact.LoadModel(ml, versionDir, out _);

        _log.LogInformation("Rebuilding features from local parquet...");
        var rows = FeatureBuilder.Build(
            _cfg.Storage.ForecastsPath,
            _cfg.Storage.Era5Path,
            _cfg.Location.Name,
            ct);
        var ds = TrainingDataset.Split(rows);

        var testActual = ds.Test.Select(x => (double)x.Era5Temp).ToArray();
        var blendPred = TemperatureTrainer.Predict(ml, model, ds.Test);

        var best = Baselines.BestSingle(ds.Val);
        var baselineRows = new List<Reporter.ModelPrediction>
        {
            new("temp_gfs",          Baselines.SingleModel(ds.Test, "temp_gfs")),
            new("temp_ecmwf",        Baselines.SingleModel(ds.Test, "temp_ecmwf")),
            new("temp_icon",         Baselines.SingleModel(ds.Test, "temp_icon")),
            new("temp_mf",           Baselines.SingleModel(ds.Test, "temp_mf")),
            new("temp_ukmo",         Baselines.SingleModel(ds.Test, "temp_ukmo")),
            new("temp_gem",          Baselines.SingleModel(ds.Test, "temp_gem")),
            new("mean_of_models",    Baselines.MeanOfModels(ds.Test)),
        };

        // Persistence + climatology use ERA5 truth lookups.
        var truthByTime = rows.ToDictionary(x => x.ValidTimeUtc, x => (double)x.Era5Temp);
        baselineRows.Add(new Reporter.ModelPrediction(
            "persistence_-24h", Baselines.Persistence(ds.Test, truthByTime, 24)));
        baselineRows.Add(new Reporter.ModelPrediction(
            "climatology(month,hour)", Baselines.Climatology(ds.Train, ds.Test)));

        var featImportance = ModelArtifact.LoadFeatureImportance(versionDir);

        // METAR secondary check over the test window.
        var (metarBlend, metarBest, metarActual, metarCount) = BuildMetarCheck(
            ds.Test, blendPred, Baselines.SingleModel(ds.Test, best), best);

        var input = new Reporter.ReportInput
        {
            GeneratedAtUtc = DateTime.UtcNow,
            ModelVersion = metadata.Version,
            Phase = metadata.Phase,
            Metadata = metadata,
            TestRows = ds.Test,
            BlendTest = new Reporter.ModelPrediction("blend", blendPred),
            BaselinesTest = baselineRows,
            BestSingleName = best,
            FeatureImportance = featImportance,
            BlendMetar = metarBlend,
            BestSingleMetar = metarBest,
            ActualMetar = metarActual,
            MetarTestRowsAvailable = metarCount,
            LeadTimeBackfillMemoPath = "docs/LEAD_TIME_BACKFILL.md",
        };

        var md = Reporter.BuildMarkdown(input);
        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var localPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"phase{metadata.Phase}_{DateTime.UtcNow:yyyy-MM-dd_HHmmss}.md");
        await File.WriteAllTextAsync(localPath, md, ct);
        _log.LogInformation("Report written → {Path}", localPath);

        // Quick blend-vs-best headline for the terminal.
        var blendMae = Metrics.Compute(blendPred, testActual).Mae;
        var bestMae  = Metrics.Compute(Baselines.SingleModel(ds.Test, best), testActual).Mae;
        _log.LogInformation("Summary — blend MAE={Blend:0.000}°C, best_single({Best}) MAE={BestMae:0.000}°C, delta={Delta:+0.000;-0.000;0.000}°C",
            blendMae, best, bestMae, blendMae - bestMae);

        return 0;
    }

    private (Reporter.ModelPrediction? blend,
             Reporter.ModelPrediction? bestSingle,
             IReadOnlyList<double>? actual,
             int count)
        BuildMetarCheck(
            IReadOnlyList<TrainingRow> testRows,
            double[] blendPred,
            double[] bestSinglePred,
            string bestSingleName)
    {
        var metarPath = _cfg.Storage.ObservationsPath;
        var station = _cfg.Location.Metar.Primary;
        if (string.IsNullOrWhiteSpace(station) || !Directory.Exists(metarPath))
            return (null, null, null, 0);

        // Pull METAR rows for the test window, match each test ValidTime to the
        // nearest observation within ±30 minutes.
        var start = testRows[0].ValidTimeUtc.AddHours(-2);
        var end   = testRows[^1].ValidTimeUtc.AddHours(2);
        var glob  = Path.Combine(metarPath, "**", "*.parquet").Replace('\\', '/');

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
SELECT ObservedTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND Station = '{station}'
  AND Temperature2m IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ObservedTimeUtc";

        var obs = new List<(DateTime T, double V)>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
                obs.Add((r.GetDateTime(0), r.GetDouble(1)));
        }
        catch (Exception ex)
        {
            _log.LogWarning("METAR query failed, skipping secondary check: {Msg}", ex.Message);
            return (null, null, null, 0);
        }

        if (obs.Count == 0)
            return (null, null, null, 0);

        // Simple two-pointer match: for each test row, binary-search nearest obs.
        var obsTimes = obs.Select(o => o.T).ToArray();
        var matchedActual = new List<double>();
        var matchedBlend  = new List<double>();
        var matchedBest   = new List<double>();

        for (int i = 0; i < testRows.Count; i++)
        {
            var target = testRows[i].ValidTimeUtc;
            var idx = Array.BinarySearch(obsTimes, target);
            if (idx < 0) idx = ~idx;

            DateTime? nearestT = null;
            double nearestV = double.NaN;
            var best = TimeSpan.FromMinutes(30);
            for (int d = -1; d <= 0; d++)
            {
                var j = idx + d;
                if (j < 0 || j >= obs.Count) continue;
                var diff = (obs[j].T - target).Duration();
                if (diff <= best)
                {
                    best = diff;
                    nearestT = obs[j].T;
                    nearestV = obs[j].V;
                }
            }

            if (nearestT.HasValue)
            {
                matchedActual.Add(nearestV);
                matchedBlend.Add(blendPred[i]);
                matchedBest.Add(bestSinglePred[i]);
            }
        }

        if (matchedActual.Count == 0)
            return (null, null, null, 0);

        return (
            blend: new Reporter.ModelPrediction("blend", matchedBlend.ToArray()),
            bestSingle: new Reporter.ModelPrediction(bestSingleName, matchedBest.ToArray()),
            actual: matchedActual,
            count: matchedActual.Count);
    }

}
