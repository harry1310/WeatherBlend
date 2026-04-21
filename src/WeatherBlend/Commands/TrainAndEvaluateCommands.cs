using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Trains the temperature blender. Default mode is phase 2b "redo" — one LightGBM
/// regressor per lead ∈ {24,48,72}, each fed only Open-Meteo Previous Runs rows
/// (RunTimeSource = 'offset_day') filtered to that exact lead. Phase 2a (bucket-free)
/// is preserved as a callable path for reproducing the v2026-04-20_221013 artefact.
/// </summary>
public sealed class TrainCommand
{
    private readonly ILogger<TrainCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = { 24, 48, 72 };

    public TrainCommand(ILogger<TrainCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string target, string lead, CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported (got '{Target}')", target);
            return 2;
        }

        // --lead phase2a is the escape hatch for reproducing the bucket-free 2a model.
        if (string.Equals(lead, "phase2a", StringComparison.OrdinalIgnoreCase))
            return await RunPhase2aAsync(ct);

        var leads = ParseLeads(lead);
        if (leads is null)
        {
            _log.LogError("Invalid --lead value '{Lead}'. Expected 24, 48, 72, all, or phase2a.", lead);
            return 2;
        }

        return await RunPhase2bAsync(leads, ct);
    }

    private static int[]? ParseLeads(string lead) => lead.ToLowerInvariant() switch
    {
        "all" => DefaultLeads,
        "24"  => new[] { 24 },
        "48"  => new[] { 48 },
        "72"  => new[] { 72 },
        _     => null,
    };

    private async Task<int> RunPhase2bAsync(int[] leads, CancellationToken ct)
    {
        var now = DateTime.UtcNow;
        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now, suffix: "phase2redo");
        var versionName = Path.GetFileName(versionDir);

        var hp = new TemperatureTrainer.Hyperparameters();
        _log.LogInformation("Phase 2b (redo) — training per-lead blenders for leads [{Leads}]",
            string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var rows = FeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.Era5Path,
                _cfg.Location.Name,
                lead,
                ct);
            _log.LogInformation("Loaded {N} rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = TrainingDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);

            var trained = TemperatureTrainer.Train(ds, hp);

            var testActual = ds.Test.Select(x => (double)x.Era5Temp).ToArray();
            var testPred   = TemperatureTrainer.Predict(trained.Ml, trained.Model, ds.Test);
            var blendStats = Metrics.Compute(testPred, testActual);

            // Best-single per-lead, selected on validation MAE.
            var best = Baselines.BestSingle(ds.Val);
            var bestValMae = Metrics.Compute(Baselines.SingleModel(ds.Val, best),
                                             ds.Val.Select(x => (double)x.Era5Temp).ToArray()).Mae;

            ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
            importanceByLead[lead] = trained.FeatureImportance;

            var testMonths = ds.Test.Select(r => new DateTime(r.ValidTimeUtc.Year, r.ValidTimeUtc.Month, 1))
                                    .Distinct().Count();

            perLead[lead.ToString()] = new ModelArtifact.PerLeadStats
            {
                LeadHours = lead,
                DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd HH:mm}Z → {ds.TrainEnd:yyyy-MM-dd HH:mm}Z",
                DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd HH:mm}Z → {ds.ValEnd:yyyy-MM-dd HH:mm}Z",
                DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd HH:mm}Z → {ds.TestEnd:yyyy-MM-dd HH:mm}Z",
                TrainRows = ds.Train.Count,
                ValRows   = ds.Val.Count,
                TestRows  = ds.Test.Count,
                TestCalendarMonths = testMonths,
                BestSingle = best,
                BestSingleValMae = bestValMae,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            _log.LogInformation("Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best_single[{Best}] val MAE={BestMae:0.000}°C, test months={M}",
                lead, blendStats.Mae, best, bestValMae, testMonths);
        }

        // Shared artefacts.
        ModelArtifact.SaveFeatureSchema(versionDir, FeatureBuilder.FeatureNames);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2b_redo",
            DataSource = "previous_runs_api",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Objective is L2 (squared error); MAE used only as early-stopping metric. Microsoft.ML.LightGbm 4.0 does not expose regression_l1.",
                "No monotone constraints on per-model temperature inputs. Microsoft.ML.LightGbm 4.0 does not expose monotone_constraints.",
                "Per-lead model artefacts named lead_{N}h.zip (ML.NET ITransformer pipeline) rather than .lgb — the zip contains the LightGBM booster plus the feature-concat transform.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        ModelArtifact.UpdateManifest(modelsRoot, "temperature", versionName);

        _log.LogInformation("Phase 2b artefacts → {Dir}", versionDir);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend MAE {kv.Value.BlendTestMae:0.000}°C vs {kv.Value.BestSingle} val MAE {kv.Value.BestSingleValMae:0.000}°C")));

        await Task.CompletedTask;
        return 0;
    }

    private async Task<int> RunPhase2aAsync(CancellationToken ct)
    {
        _log.LogInformation("Phase 2a (bucket-free) — reproduction path.");

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
        var hp = new TemperatureTrainer.Hyperparameters();
        var trained = TemperatureTrainer.Train(ds, hp);

        var testPred = TemperatureTrainer.Predict(trained.Ml, trained.Model, ds.Test);
        var testActual = ds.Test.Select(x => (double)x.Era5Temp).ToArray();
        var testStats = Metrics.Compute(testPred, testActual);
        var best = Baselines.BestSingle(ds.Val);
        var bestStats = Metrics.Compute(Baselines.SingleModel(ds.Test, best), testActual);
        var meanStats = Metrics.Compute(Baselines.MeanOfModels(ds.Test), testActual);

        _log.LogInformation("Headline test MAE — blend={Blend:0.000}°C, best_single({Best})={BestMae:0.000}°C, mean_of_models={MeanMae:0.000}°C",
            testStats.Mae, best, bestStats.Mae, meanStats.Mae);

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
            DataSource = "open_meteo_historical_forecast_api",
            TrainedAtUtc = now,
            DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd HH:mm}Z → {ds.TrainEnd:yyyy-MM-dd HH:mm}Z",
            DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd HH:mm}Z → {ds.ValEnd:yyyy-MM-dd HH:mm}Z",
            DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd HH:mm}Z → {ds.TestEnd:yyyy-MM-dd HH:mm}Z",
            TrainRows = ds.Train.Count,
            ValRows   = ds.Val.Count,
            TestRows  = ds.Test.Count,
            Hyperparameters = BuildHpDict(hp),
            TestMae = new Dictionary<string, double>
            {
                ["blend"]                = testStats.Mae,
                [$"best_single[{best}]"] = bestStats.Mae,
                ["mean_of_models"]       = meanStats.Mae,
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
        await Task.CompletedTask;
        return 0;
    }

    private static Dictionary<string, object> BuildHpDict(TemperatureTrainer.Hyperparameters hp) => new()
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
    };
}

public sealed class EvaluateCommand
{
    private readonly ILogger<EvaluateCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] Phase2bLeads = { 24, 48, 72 };

    public EvaluateCommand(ILogger<EvaluateCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string target, string modelVersion, CancellationToken ct)
    {
        if (!string.Equals(target, "temperature", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Only target=temperature is supported (got '{Target}')", target);
            return 2;
        }

        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", modelVersion);
        _log.LogInformation("Loading model from {Dir}", versionDir);

        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);

        // Route by metadata phase. Anything with PerLead populated → phase 2b report.
        if (metadata.PerLead.Count > 0)
            return await RunPhase2bAsync(versionDir, metadata, ct);

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

    private async Task<int> RunPhase2bAsync(
        string versionDir,
        ModelArtifact.TrainingMetadata metadata,
        CancellationToken ct)
    {
        _log.LogInformation("Phase 2b evaluation — per-lead blenders from {Dir}", versionDir);

        var ml = new Microsoft.ML.MLContext(seed: 42);
        var perLeadImportance = ModelArtifact.LoadPerLeadFeatureImportance(versionDir);

        // Phase 2a headline numbers, for the before/after section. Scan version dirs
        // for the most recent one with phase "2a" and non-empty TestMae.
        double? p2aBlend = null, p2aBest = null;
        string? p2aBestLabel = null, p2aVersion = null;
        TryLoadPhase2aStats(Path.GetDirectoryName(versionDir)!, out p2aBlend, out p2aBest, out p2aBestLabel, out p2aVersion);

        var bundles = new List<Reporter.LeadBundle>();
        foreach (var lead in Phase2bLeads)
        {
            ct.ThrowIfCancellationRequested();
            if (!metadata.PerLead.TryGetValue(lead.ToString(), out var stats))
            {
                _log.LogWarning("No per-lead stats for lead {Lead}h; skipping.", lead);
                continue;
            }

            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var rows = FeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name, lead, ct);
            if (rows.Count < 10)
            {
                _log.LogError("Lead {Lead}h: only {N} rows; skipping.", lead, rows.Count);
                continue;
            }
            var ds = TrainingDataset.Split(rows);

            var leadModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var blendPred = TemperatureTrainer.Predict(ml, leadModel, ds.Test);

            var actual = ds.Test.Select(x => (double)x.Era5Temp).ToArray();
            var bestSingle = Baselines.BestSingle(ds.Val);

            // Full baseline set for headline + stratification.
            var baselines = new List<Reporter.ModelPrediction>
            {
                new("temp_gfs",       Baselines.SingleModel(ds.Test, "temp_gfs")),
                new("temp_ecmwf",     Baselines.SingleModel(ds.Test, "temp_ecmwf")),
                new("temp_icon",      Baselines.SingleModel(ds.Test, "temp_icon")),
                new("temp_mf",        Baselines.SingleModel(ds.Test, "temp_mf")),
                new("temp_ukmo",      Baselines.SingleModel(ds.Test, "temp_ukmo")),
                new("temp_gem",       Baselines.SingleModel(ds.Test, "temp_gem")),
                new("mean_of_models", Baselines.MeanOfModels(ds.Test)),
            };

            // Persistence at this lead. Build truth map across train+val+test.
            var truthByTime = rows.ToDictionary(x => x.ValidTimeUtc, x => (double)x.Era5Temp);
            var persistence = Baselines.Persistence(ds.Test, truthByTime, lagHours: lead);
            var persistenceDropped = persistence.Count(double.IsNaN);
            baselines.Add(new Reporter.ModelPrediction($"persistence_-{lead}h", persistence));

            // Climatology from training rows only.
            baselines.Add(new Reporter.ModelPrediction(
                "climatology(month,hour)", Baselines.Climatology(ds.Train, ds.Test)));

            // METAR check against Exeter for this lead's test window.
            var bestPred = Baselines.SingleModel(ds.Test, bestSingle);
            var (metarBlend, metarBest, metarActual, metarCount) =
                BuildMetarCheck(ds.Test, blendPred, bestPred, bestSingle);

            perLeadImportance.TryGetValue(lead, out var importance);

            bundles.Add(new Reporter.LeadBundle
            {
                LeadHours = lead,
                Stats = stats,
                TestRows = ds.Test,
                BlendTest = new Reporter.ModelPrediction("blend", blendPred),
                BaselinesTest = baselines,
                BestSingleName = bestSingle,
                FeatureImportance = importance ?? Array.Empty<(string, double)>(),
                PersistenceDropped = persistenceDropped,
                BlendMetar = metarBlend,
                BestSingleMetar = metarBest,
                ActualMetar = metarActual,
                MetarRowsAvailable = metarCount,
            });

            _log.LogInformation("Lead {Lead}h — blend MAE {Mae:0.000}°C (best single {Best} {BestMae:0.000}°C)",
                lead, Metrics.Compute(blendPred, actual).Mae,
                bestSingle, Metrics.Compute(bestPred, actual).Mae);
        }

        if (bundles.Count == 0)
        {
            _log.LogError("No lead bundles built — aborting report.");
            return 3;
        }

        var report = Reporter.BuildPhase2bMarkdown(new Reporter.Phase2bReportInput
        {
            GeneratedAtUtc = DateTime.UtcNow,
            ModelVersion = metadata.Version,
            Metadata = metadata,
            Leads = bundles,
            Phase2aBlendMae = p2aBlend,
            Phase2aBestSingleMae = p2aBest,
            Phase2aBestSingleLabel = p2aBestLabel,
            Phase2aVersion = p2aVersion,
        });

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var localPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"phase2_redo_{DateTime.UtcNow:yyyy-MM-dd_HHmmss}.md");
        await File.WriteAllTextAsync(localPath, report, ct);
        _log.LogInformation("Report written → {Path}", localPath);

        // Terminal summary: one line per lead.
        foreach (var lb in bundles)
        {
            var actual = lb.TestRows.Select(x => (double)x.Era5Temp).ToArray();
            var blendMae = Metrics.Compute(lb.BlendTest.Predicted, actual).Mae;
            var bestPred = Pick(lb, lb.BestSingleName);
            var bestMae = Metrics.Compute(bestPred, actual).Mae;
            var delta = blendMae - bestMae;
            _log.LogInformation("Lead {Lead}h summary — blend {Blend:0.000}°C vs {Best} {BestMae:0.000}°C, delta {Delta:+0.000;-0.000;0.000}°C",
                lb.LeadHours, blendMae, lb.BestSingleName, bestMae, delta);
        }

        return 0;
    }

    private static double[] Pick(Reporter.LeadBundle lb, string name)
    {
        if (name.Equals(lb.BlendTest.Name, StringComparison.OrdinalIgnoreCase))
            return lb.BlendTest.Predicted;
        foreach (var b in lb.BaselinesTest)
            if (b.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                return b.Predicted;
        throw new KeyNotFoundException($"No baseline named '{name}' for lead {lb.LeadHours}h");
    }

    private static void TryLoadPhase2aStats(
        string targetRoot,
        out double? blendMae,
        out double? bestSingleMae,
        out string? bestSingleLabel,
        out string? version)
    {
        blendMae = null; bestSingleMae = null; bestSingleLabel = null; version = null;
        if (!Directory.Exists(targetRoot)) return;

        foreach (var dir in Directory.EnumerateDirectories(targetRoot).OrderByDescending(d => d))
        {
            var mdPath = Path.Combine(dir, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(mdPath)) continue;
            try
            {
                var md = ModelArtifact.LoadTrainingMetadata(dir);
                if (md.Phase == "2a" && md.TestMae.TryGetValue("blend", out var b))
                {
                    blendMae = b;
                    version = md.Version;
                    var bestKey = md.TestMae.Keys.FirstOrDefault(k => k.StartsWith("best_single"));
                    if (bestKey is not null && md.TestMae.TryGetValue(bestKey, out var bs))
                    {
                        bestSingleMae = bs;
                        var open = bestKey.IndexOf('[');
                        var close = bestKey.IndexOf(']');
                        bestSingleLabel = (open > 0 && close > open)
                            ? bestKey.Substring(open + 1, close - open - 1)
                            : bestKey;
                    }
                    return;
                }
            }
            catch { /* skip broken dirs */ }
        }
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
