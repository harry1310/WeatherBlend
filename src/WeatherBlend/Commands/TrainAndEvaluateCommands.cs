using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Trains the blender — one LightGBM model per lead ∈ {24,48,72}.
///
/// target=temperature: regressor against ERA5 reanalysis (Phase 2b).
/// target=precipitation: binary classifier for P(hour has >= 0.1 mm) against
/// EA Bellever rainfall truth (Phase 3a occurrence model).
/// </summary>
public sealed class TrainCommand
{
    private readonly ILogger<TrainCommand> _log;
    private readonly AppConfig _cfg;
    private readonly DryWindowTrainCommand _dryWindow;

    private static readonly int[] DefaultLeads = { 24, 48, 72 };

    public TrainCommand(ILogger<TrainCommand> log, AppConfig cfg, DryWindowTrainCommand dryWindow)
    {
        _log = log;
        _cfg = cfg;
        _dryWindow = dryWindow;
    }

    public async Task<int> RunAsync(string target, string lead, string? station, string? window, CancellationToken ct)
    {
        var t = target.ToLowerInvariant();
        if (t is not ("temperature" or "precipitation" or "dry-window"))
        {
            _log.LogError("target must be 'temperature' | 'precipitation' | 'dry-window' (got '{Target}')", target);
            return 2;
        }

        var leads = ParseLeads(lead);
        if (leads is null)
        {
            _log.LogError("Invalid --lead value '{Lead}'. Expected 24, 48, 72, or all.", lead);
            return 2;
        }

        return t switch
        {
            "temperature"   => await RunPhase2bAsync(leads, ct),
            "precipitation" => await RunPhase3aAsync(leads, station, ct),
            "dry-window"    => await _dryWindow.RunAsync(station ?? "all", window ?? "all", leads, ct),
            _ => 2,
        };
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
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now);
        var versionName = Path.GetFileName(versionDir);

        var hp = new TemperatureTrainer.Hyperparameters();
        _log.LogInformation("Phase 2b — training per-lead blenders for leads [{Leads}]",
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
            Phase = "2b",
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

    // ---- Phase 3a: precipitation occurrence classifier ----------------------------

    private async Task<int> RunPhase3aAsync(int[] leads, string? stationOverride, CancellationToken ct)
    {
        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured — cannot train precipitation blender.");
            return 2;
        }

        string primaryStation;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            primaryStation = _cfg.Location.Rainfall.Stations[0].Name;
        }
        else
        {
            var match = _cfg.Location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found in config. Available: {Available}",
                    stationOverride, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
                return 2;
            }
            primaryStation = match.Name;
        }

        // Each station gets its own subtree under data/models/precipitation/{station}/v{ts}/
        // so the manifest can track a separate "current" pointer per truth source.
        // Slug is prefixed with the provider ("ea_") so a future Met Office Princetown
        // collector can live alongside the EA one without colliding.
        var stationSlug = "ea_" + Slugify(primaryStation);
        var now = DateTime.UtcNow;
        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now);
        var versionName = Path.GetFileName(versionDir);

        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        _log.LogInformation("Phase 3a — precipitation occurrence classifier, station='{Station}', leads=[{Leads}]",
            primaryStation, string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        PrecipClimatology? climatology = null;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var rows = PrecipFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                _cfg.Location.Name,
                primaryStation,
                lead,
                ct);
            _log.LogInformation("Loaded {N} rows (wet={Wet} / {Pct:P1}) spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count(r => r.WetBinary),
                rows.Count == 0 ? 0 : (double)rows.Count(r => r.WetBinary) / rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = PrecipDataset.Split(rows);
            // Climatology is a base-rate lookup over (month, hour) and so is essentially
            // identical across leads — one copy per version is enough. Build from the
            // first lead's train partition; predict/verify will reuse it.
            climatology ??= PrecipClimatology.BuildFromTraining(ds.Train);
            _log.LogInformation("Split → train {TN} (wet {TW}), val {VN} (wet {VW}), test {EN} (wet {EW})",
                ds.Train.Count, ds.TrainWet,
                ds.Val.Count,   ds.ValWet,
                ds.Test.Count,  ds.TestWet);
            _log.LogInformation("Time ranges — train {T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}, " +
                                "val {V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}, test {E0:yyyy-MM-dd}..{E1:yyyy-MM-dd}",
                ds.TrainStart, ds.TrainEnd, ds.ValStart, ds.ValEnd, ds.TestStart, ds.TestEnd);

            var trained = PrecipOccurrenceTrainer.Train(ds, hp);

            // Test-set headline metrics.
            var truthTest = ds.Test.Select(r => r.WetBinary ? 1.0 : 0.0).ToArray();
            var blendProb = PrecipOccurrenceTrainer.PredictProbability(trained.Ml, trained.Model, ds.Test);
            var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

            // Baselines for comparison.
            var best = PrecipBaselines.BestSingle(ds.Val);
            var bestValBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(ds.Val, best),
                ds.Val.Select(r => r.WetBinary ? 1.0 : 0.0).ToArray());
            var climPred = PrecipBaselines.Climatology(ds.Train, ds.Test);
            var climBrier = PrecipMetrics.Brier(climPred, truthTest);
            var bss = PrecipMetrics.BrierSkillScore(blendBrier, climBrier);

            // Frequency bias at 0.5 decision threshold — quick sanity check that
            // the classifier isn't collapsing to all-wet or all-dry.
            var blendBinary = blendProb.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
            var fbias = PrecipMetrics.FrequencyBias(blendBinary, truthTest);

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
                BestSingleValMae = bestValBrier,   // reusing the column for Brier — documented in metadata
                BlendTestMae  = blendBrier,
                BlendTestRmse = climBrier,         // reuse — climatology Brier for BSS reference
                BlendTestBias = fbias,
            };

            _log.LogInformation(
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, climatology Brier={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, freq_bias={Fb:0.00}, best_single[{Best}] val Brier={BestBrier:0.000}",
                lead, blendBrier, climBrier, bss, fbias, best, bestValBrier);
        }

        ModelArtifact.SaveFeatureSchema(versionDir, PrecipFeatureBuilder.OccurrenceFeatureNames);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);
        // Persist climatology alongside the model zips so predict/verify can reach
        // for a P(wet) baseline without re-scanning the training rainfall tree.
        if (climatology is not null)
            climatology.SaveTo(Path.Combine(versionDir, ModelArtifact.ClimatologyFileName));

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "precipitation",
            Phase = "3a",
            DataSource = "previous_runs_api+ea_rainfall",
            TrainedAtUtc = now,
            Hyperparameters = BuildPrecipHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Intensity regressor not trained in this artefact — occurrence classifier only. Two-stage precip blender (Brier-tuned classifier × E[mm|wet] regressor) is tracked as Phase 3b follow-up.",
                "Threshold classifiers at 1/5/10 mm not trained — only 0.1 mm (WetBinary). Higher-threshold classifiers need more positive samples than Bellever's 2.3 years provides for robust val/test at 10mm (7 events).",
                "Princetown and Dartmoor-nr-Hexworthy stations not trained. Bellever is the brief's primary truth; the two secondaries remain available for cross-station verification in the evaluation report.",
                "PerLeadStats fields repurposed: BlendTestMae=blend Brier, BlendTestRmse=climatology Brier, BlendTestBias=frequency bias at p=0.5, BestSingleValMae=best-single Brier on val. Artefact schema unchanged to avoid breaking Phase 2b loaders.",
                "Microsoft.ML.LightGbm 4.0 constraints: no explicit class-weight option beyond UnbalancedSets=true; no monotone constraints. Recorded per Phase 2b pattern.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        ModelArtifact.UpdateStationManifest(modelsRoot, "precipitation", stationSlug, versionName);

        _log.LogInformation("Phase 3a artefacts → {Dir}", versionDir);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        await Task.CompletedTask;
        return 0;
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

    private static Dictionary<string, object> BuildPrecipHpDict(PrecipOccurrenceTrainer.Hyperparameters hp) => new()
    {
        ["numberOfIterations"]         = hp.NumberOfIterations,
        ["learningRate"]               = hp.LearningRate,
        ["numberOfLeaves"]             = hp.NumberOfLeaves,
        ["minimumExampleCountPerLeaf"] = hp.MinimumExampleCountPerLeaf,
        ["l1Regularization"]           = hp.L1Regularization,
        ["l2Regularization"]           = hp.L2Regularization,
        ["earlyStoppingRound"]         = hp.EarlyStoppingRound,
        ["seed"]                       = hp.Seed,
        ["unbalancedSets"]             = true,
        ["objective"]                  = "binary (LightGBM)",
        ["evaluationMetric"]           = "AUC (LightGBM default for binary)",
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

        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Model version {V} has no per-lead blenders — evaluate requires a per-lead artefact.",
                metadata.Version);
            return 2;
        }

        return await RunPhase2bAsync(versionDir, metadata, ct);
    }

    private async Task<int> RunPhase2bAsync(
        string versionDir,
        ModelArtifact.TrainingMetadata metadata,
        CancellationToken ct)
    {
        _log.LogInformation("Phase 2b evaluation — per-lead blenders from {Dir}", versionDir);

        var ml = new Microsoft.ML.MLContext(seed: 42);
        var perLeadImportance = ModelArtifact.LoadPerLeadFeatureImportance(versionDir);

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
        });

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var localPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"phase2b_{DateTime.UtcNow:yyyy-MM-dd_HHmmss}.md");
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
