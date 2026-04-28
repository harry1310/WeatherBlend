using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;

namespace WeatherBlend.Commands;

/// <summary>
/// Trains the blender — one LightGBM model per lead.
///
/// target=temperature: regressor against ERA5 reanalysis (Phase 2b/2c). Leads {24,48,72,120}.
/// target=precipitation: binary classifier for P(hour has >= 0.1 mm) against
/// EA Bellever rainfall truth (Phase 3a/3c). Leads {24,48,72,120}.
/// target=dry-window: P(at least one N-hour dry block in target day) (Phase 3b/3d). Leads {24,48,72}.
/// target=wind | humidity | shortwave-radiation | cloud-cover: per-lead regressor
/// against ERA5 truth — the lean per-element blenders. Leads {24,48,72}.
///
/// Lead 120h applies to temperature + precipitation only — dry-window and Element
/// blenders stay capped at 72h pending a separate scoping decision.
/// </summary>
public sealed class TrainCommand
{
    private readonly ILogger<TrainCommand> _log;
    private readonly AppConfig _cfg;
    private readonly DryWindowTrainCommand _dryWindow;
    private readonly ElementTrainCommand _element;

    // Default leads for temperature + precipitation. Dry-window and Element
    // train commands keep their own narrower {24,48,72} arrays internally.
    private static readonly int[] DefaultLeads = { 24, 48, 72, 120 };

    public TrainCommand(
        ILogger<TrainCommand> log,
        AppConfig cfg,
        DryWindowTrainCommand dryWindow,
        ElementTrainCommand element)
    {
        _log = log;
        _cfg = cfg;
        _dryWindow = dryWindow;
        _element = element;
    }

    public async Task<int> RunAsync(string target, string lead, string? station, string? window, string featureSet, CancellationToken ct)
    {
        var t = target.ToLowerInvariant();
        var elementTarget = ElementTargets.TryFromCli(t);
        var validTargets = new[]
        {
            "temperature", "precipitation", "dry-window",
            "wind", "humidity", "shortwave-radiation", "cloud-cover",
        };
        if (!validTargets.Contains(t))
        {
            _log.LogError("target must be one of [{Targets}] (got '{Target}')",
                string.Join(", ", validTargets), target);
            return 2;
        }

        var leads = ParseLeads(lead);
        if (leads is null)
        {
            _log.LogError("Invalid --lead value '{Lead}'. Expected 24, 48, 72, 120, or all.", lead);
            return 2;
        }

        var fs = (featureSet ?? "lean").ToLowerInvariant();
        if (fs is not ("lean" or "rich"))
        {
            _log.LogError("Invalid --feature-set value '{Fs}'. Expected lean | rich.", featureSet);
            return 2;
        }
        if (elementTarget is not null && fs != "lean")
        {
            _log.LogError(
                "Element targets currently only support --feature-set lean (got '{Fs}'). " +
                "Rich variants are not defined for the per-variable blenders yet.", fs);
            return 2;
        }

        // Lead 120h is currently scoped to temperature + precipitation only.
        // Dry-window and Element targets stay capped at 72h until separately scoped.
        if ((t == "dry-window" || elementTarget is not null) && leads.Contains(120))
        {
            if (lead.Equals("120", StringComparison.OrdinalIgnoreCase))
            {
                _log.LogError("--lead 120 is not supported for target '{T}' yet (temperature + precipitation only).", t);
                return 2;
            }
            // "all" implicitly includes 120 — silently narrow rather than error.
            leads = leads.Where(l => l != 120).ToArray();
        }

        return t switch
        {
            "temperature"   => fs == "rich"
                                  ? await RunPhase2cAsync(leads, ct)
                                  : await RunPhase2bAsync(leads, ct),
            "precipitation" => fs == "rich"
                                  ? await RunPhase3cAsync(leads, station, ct)
                                  : await RunPhase3aAsync(leads, station, ct),
            // dry-window: lean → Phase 3b (53 features), rich → Phase 3d-shape
            // (3b + 7 ensemble-mean within-day rain-structure columns).
            "dry-window"    => await _dryWindow.RunAsync(
                                   station ?? "all", window ?? "all", leads,
                                   fs == "rich" ? Train.DryWindow.DryWindowFeatureBuilder.Phase3dShape
                                                : Train.DryWindow.DryWindowFeatureBuilder.Phase3b,
                                   ct),
            // Per-variable element blenders: one dispatcher routes wind / humidity /
            // shortwave-radiation / cloud-cover to its dedicated IElementBlender.
            _ when elementTarget is not null
                   => await _element.RunAsync(elementTarget, leads, ct),
            _ => 2,
        };
    }

    private static int[]? ParseLeads(string lead) => lead.ToLowerInvariant() switch
    {
        "all" => DefaultLeads,
        "24"  => new[] { 24 },
        "48"  => new[] { 48 },
        "72"  => new[] { 72 },
        "120" => new[] { 120 },
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
        var specsPerLead = new Dictionary<int, BlenderSpec>();

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = FeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = FeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.Era5Path,
                _cfg.Location.Name,
                spec,
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

            var ds = RegressionDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);

            var trained = TemperatureTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TemperatureTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = Metrics.Compute(testPred, testActual);

            // Best-single per-lead, selected on validation MAE.
            // NOTE 2026-04-28: BestSingleTestMae deliberately NOT computed here
            // until the TIMESTAMPTZ-vs-TIMESTAMP JOIN bug between forecast and
            // ERA5 parquets is fixed — it would publish numbers off by 1–2h
            // during BST, making the comparison even more misleading than the
            // current val-vs-test mismatch on the Models page.
            var best = Baselines.BestSingle(spec, ds.Val);
            var bestValMae = Metrics.Compute(
                Baselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;

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

        // Shared artefacts — per-lead BlenderSpec is now the schema source of truth.
        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
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
                "Per-lead model artefacts named lead_{N}h.zip; vector-native LightGBM trainer (Features column = float[]).",
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

    // ---- Phase 2c: rich-feature temperature blender (champion/challenger) ----------

    private async Task<int> RunPhase2cAsync(int[] leads, CancellationToken ct)
    {
        var now = DateTime.UtcNow;
        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now, suffix: "phase2c");
        var versionName = Path.GetFileName(versionDir);

        var hp = new TemperatureTrainer.Hyperparameters();
        _log.LogInformation("Phase 2c — rich-feature blender (88 features), leads [{Leads}]",
            string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (identical to Phase 2b — feature richness is the only variable)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = RichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = RichFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.Era5Path,
                _cfg.Location.Name,
                spec,
                ct);
            _log.LogInformation("Loaded {N} rich rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = RegressionDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);

            var trained = TemperatureTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TemperatureTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = Metrics.Compute(testPred, testActual);

            var best = Baselines.BestSingle(spec, ds.Val);
            var bestValMae = Metrics.Compute(
                Baselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;

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

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2c",
            DataSource = "previous_runs_api",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Objective is L2 (squared error); MAE used only as early-stopping metric. Microsoft.ML.LightGbm 4.0 does not expose regression_l1.",
                "No monotone constraints on per-model temperature inputs. Microsoft.ML.LightGbm 4.0 does not expose monotone_constraints.",
                "Persistence features (era5_temp_24h_ago, era5_temp_168h_ago) deliberately omitted: ERA5 has a ~5d release lag, so neither lookback exists at predict time for the 24/48/72h leads. Including them would teach the model to lean on data unavailable in production.",
                "Per-lead model artefacts named lead_{N}h.zip (ML.NET ITransformer pipeline). Same hyperparameters as Phase 2b — feature richness is the isolated variable.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

        // Champion/challenger: 2c is added to the active set without unseating 2b as
        // the default. Predict + verify will iterate both versions every cycle.
        ModelArtifact.AppendVersion(modelsRoot, "temperature", versionName);
        var existingActive = ModelArtifact.ResolveActive(modelsRoot, "temperature");
        var newActive = existingActive.Concat(new[] { versionName }).Distinct().ToList();
        ModelArtifact.SetActive(modelsRoot, "temperature", newActive);

        _log.LogInformation("Phase 2c artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions now: [{Active}]", string.Join(", ", newActive));
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
        ["subsampleFraction"]            = hp.SubsampleFraction,
        ["subsampleFrequency"]           = hp.SubsampleFrequency,
        ["featureFraction"]              = hp.FeatureFraction,
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
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        PrecipClimatology? climatology = null;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = PrecipFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = PrecipFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                _cfg.Location.Name,
                primaryStation,
                spec,
                ct);
            _log.LogInformation("Loaded {N} rows (wet={Wet} / {Pct:P1}) spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count(r => r.Label),
                rows.Count == 0 ? 0 : (double)rows.Count(r => r.Label) / rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = BinaryDataset.Split(rows);
            // Climatology — base-rate lookup over (month, hour). Build a row of the
            // legacy PrecipClimatology shape from BinaryTrainingRow's labels so the
            // predict-time loader stays unchanged.
            climatology ??= PrecipClimatology.BuildFromTraining(ds.Train);
            _log.LogInformation("Split → train {TN} (wet {TW}), val {VN} (wet {VW}), test {EN} (wet {EW})",
                ds.Train.Count, ds.TrainWet,
                ds.Val.Count,   ds.ValWet,
                ds.Test.Count,  ds.TestWet);
            _log.LogInformation("Time ranges — train {T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}, " +
                                "val {V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}, test {E0:yyyy-MM-dd}..{E1:yyyy-MM-dd}",
                ds.TrainStart, ds.TrainEnd, ds.ValStart, ds.ValEnd, ds.TestStart, ds.TestEnd);

            var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var blendProb = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, ds.Test);
            var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

            var best = PrecipBaselines.BestSingle(spec, ds.Val);
            var bestValBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Val, best),
                ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
            var climPred = PrecipBaselines.Climatology(ds.Train, ds.Test);
            var climBrier = PrecipMetrics.Brier(climPred, truthTest);
            var bss = PrecipMetrics.BrierSkillScore(blendBrier, climBrier);

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
                BestSingleValMae = bestValBrier,   // reused column — Brier
                BlendTestMae  = blendBrier,
                BlendTestRmse = climBrier,         // reused column — climatology Brier
                BlendTestBias = fbias,
            };

            _log.LogInformation(
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, climatology Brier={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, freq_bias={Fb:0.00}, best_single[{Best}] val Brier={BestBrier:0.000}",
                lead, blendBrier, climBrier, bss, fbias, best, bestValBrier);
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
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

    // ---- Phase 3c: rich-feature precip occurrence blender (champion/challenger) ----

    private async Task<int> RunPhase3cAsync(int[] leads, string? stationOverride, CancellationToken ct)
    {
        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured — cannot train precipitation blender.");
            return 2;
        }

        // When no station override is given, 3c iterates every configured rainfall
        // station so one command run produces a challenger for each 3a champion.
        // A specific station still trains that one alone.
        IReadOnlyList<string> stationsToTrain;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stationsToTrain = _cfg.Location.Rainfall.Stations.Select(s => s.Name).ToList();
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
            stationsToTrain = new[] { match.Name };
        }

        var modelsRoot = Path.Combine("data", "models");
        var hp = new PrecipOccurrenceTrainer.Hyperparameters();

        _log.LogInformation("Phase 3c — rich-feature precip blender, stations=[{Stations}], leads=[{Leads}]",
            string.Join(", ", stationsToTrain), string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (identical to Phase 3a — feature richness is the only variable)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var anyFail = false;
        foreach (var station in stationsToTrain)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await TrainPhase3cStationAsync(station, leads, modelsRoot, hp, ct);
            if (rc != 0) anyFail = true;
        }
        return anyFail ? 3 : 0;
    }

    private async Task<int> TrainPhase3cStationAsync(
        string primaryStation,
        int[] leads,
        string modelsRoot,
        PrecipOccurrenceTrainer.Hyperparameters hp,
        CancellationToken ct)
    {
        var stationSlug = "ea_" + Slugify(primaryStation);
        var now = DateTime.UtcNow;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now, suffix: "phase3c");
        var versionName = Path.GetFileName(versionDir);

        _log.LogInformation("=== Station '{Station}' (slug={Slug}) ===", primaryStation, stationSlug);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        PrecipClimatology? climatology = null;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = PrecipRichFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                _cfg.Location.Name,
                primaryStation,
                spec,
                ct);
            _log.LogInformation("Loaded {N} rich rows (wet={Wet} / {Pct:P1}) spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count(r => r.Label),
                rows.Count == 0 ? 0 : (double)rows.Count(r => r.Label) / rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = BinaryDataset.Split(rows);
            climatology ??= PrecipClimatology.BuildFromTraining(ds.Train);

            _log.LogInformation("Split → train {TN} (wet {TW}), val {VN} (wet {VW}), test {EN} (wet {EW})",
                ds.Train.Count, ds.TrainWet,
                ds.Val.Count,   ds.ValWet,
                ds.Test.Count,  ds.TestWet);
            _log.LogInformation("Time ranges — train {T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}, " +
                                "val {V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}, test {E0:yyyy-MM-dd}..{E1:yyyy-MM-dd}",
                ds.TrainStart, ds.TrainEnd, ds.ValStart, ds.ValEnd, ds.TestStart, ds.TestEnd);

            var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var blendProb = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, ds.Test);
            var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

            var best = PrecipBaselines.BestSingle(spec, ds.Val);
            var bestValBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Val, best),
                ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
            var climPred = PrecipBaselines.Climatology(ds.Train, ds.Test);
            var climBrier = PrecipMetrics.Brier(climPred, truthTest);
            var bss = PrecipMetrics.BrierSkillScore(blendBrier, climBrier);

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
                BestSingleValMae = bestValBrier,
                BlendTestMae  = blendBrier,
                BlendTestRmse = climBrier,
                BlendTestBias = fbias,
            };

            _log.LogInformation(
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, climatology Brier={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, freq_bias={Fb:0.00}, best_single[{Best}] val Brier={BestBrier:0.000}",
                lead, blendBrier, climBrier, bss, fbias, best, bestValBrier);
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);
        if (climatology is not null)
            climatology.SaveTo(Path.Combine(versionDir, ModelArtifact.ClimatologyFileName));

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "precipitation",
            Phase = "3c",
            DataSource = "previous_runs_api+ea_rainfall",
            TrainedAtUtc = DateTime.UtcNow,
            Hyperparameters = BuildPrecipHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Same hyperparameters as Phase 3a — feature richness is the isolated variable. Sample weighting and class weights untouched.",
                "Forecast-time trailing persistence (H-1/H-2/H-3 of the same run) and pressure tendency not included: Phase 1 training parquet persists only leads {24, 48, 72} per RunTimeSource='offset_day' run, so the intermediate-lead cells those features need don't exist in training data. Tier is deferred until live-cycle training data is available.",
                "EA observation persistence features anchored at run_time = valid_time - leadHours, filling NaN when the 24h/72h trailing window isn't fully present in the rainfall parquet.",
                "Per-station layout: artefact folder is data/models/precipitation/{station}/v{ts}_phase3c/. Manifest appends 3c to StationEntry.Active alongside 3a so both versions produce predictions every cycle.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

        // Champion/challenger: add 3c to Active without unseating 3a as Current.
        ModelArtifact.AppendStationVersion(modelsRoot, "precipitation", stationSlug, versionName);
        var existing = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);
        var newActive = existing.Concat(new[] { versionName }).Distinct().ToList();
        ModelArtifact.SetStationActive(modelsRoot, "precipitation", stationSlug, newActive);

        _log.LogInformation("Phase 3c artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));
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
        ["subsampleFraction"]          = hp.SubsampleFraction,
        ["subsampleFrequency"]         = hp.SubsampleFrequency,
        ["featureFraction"]            = hp.FeatureFraction,
        ["unbalancedSets"]             = true,
        ["objective"]                  = "binary (LightGBM)",
        ["evaluationMetric"]           = "AUC (LightGBM default for binary)",
    };
}
