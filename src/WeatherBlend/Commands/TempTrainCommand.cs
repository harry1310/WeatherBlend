using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Temp;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Exact12h;
using WeatherBlend.Train.PrecipExact;

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
public sealed class TempTrainCommand
{
    private readonly ILogger<TempTrainCommand> _log;
    private readonly AppConfig _cfg;
    private readonly DryWindowTrainCommand _dryWindow;
    private readonly ElementTrainCommand _element;
    // Auto-refit conformal calibrators after every promote-to-(champion|challenger).
    // Without this hook a fresh version ships with no calibrator; live predict
    // would degrade to the raw model probability and the dry-window page's
    // confidence tags would default to "ambiguous" until the next manual
    // `precip-conformal-fit` / `dry-window-conformal-fit` ran.
    private readonly PrecipConformalFitCommand _precipConformal;

    // Default leads for temperature + precipitation. Dry-window and Element
    // train commands keep their own narrower {24,48,72} arrays internally.
    // Sourced from Train.Common.Leads.Full so temp + precip + predict all share
    // a single definition. Dry-window + Element blenders use Leads.Short (their
    // own train commands set DefaultLeads = Leads.Short).
    private static readonly int[] DefaultLeads = Leads.Full;

    public TempTrainCommand(
        ILogger<TempTrainCommand> log,
        AppConfig cfg,
        DryWindowTrainCommand dryWindow,
        ElementTrainCommand element,
        PrecipConformalFitCommand precipConformal)
    {
        _log = log;
        _cfg = cfg;
        _dryWindow = dryWindow;
        _element = element;
        _precipConformal = precipConformal;
    }

    public Task<int> RunAsync(string target, string lead, string? station, string? window, string featureSet, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier: null, includeUkv: null, exactLeads: null, cycles: null, ct);

    public Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier, includeUkv, exactLeads: null, cycles: null, ct);

    public Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier, includeUkv, exactLeads, cycles: null, ct);

    public async Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, int[]? cycles, CancellationToken ct)
    {
        // tier + includeUkv + exactLeads are exact-runtime levers (Phase 2d /
        // 3d). Defaults (null, null, null) preserve historical behaviour: 2d
        // picks T2 + UKV + leads {12,24}; 3d picks P1 + UKV + leads {12,24}.
        // Bake-off variants pass non-default values to swap tiers (P1 vs P2),
        // toggle UKV, or extend leads to {48,72,96,120} for the long-range
        // 2b/3a comparison without code changes.
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
            _log.LogError("Invalid --lead value '{Lead}'. Expected 24, 48, 72, 96, 120, or all.", lead);
            return 2;
        }

        var fs = (featureSet ?? "lean").ToLowerInvariant();
        if (fs is not ("lean" or "rich" or "independence-mc" or "exact"))
        {
            _log.LogError("Invalid --feature-set value '{Fs}'. Expected lean | rich | independence-mc | exact.", featureSet);
            return 2;
        }
        // "independence-mc" = Phase 3g (parameter-free MC over 3a marginals);
        // dry-window-only.
        if (fs == "independence-mc" && t != "dry-window")
        {
            _log.LogError(
                "--feature-set {Fs} is only supported for target dry-window.", fs);
            return 2;
        }
        // "exact" = Phase 2d (temperature) or Phase 3d (precipitation) — same
        // feature-set tag, dispatched per target.
        if (fs == "exact" && t != "temperature" && t != "precipitation")
        {
            _log.LogError(
                "--feature-set {Fs} is only supported for targets temperature, precipitation.", fs);
            return 2;
        }
        if (elementTarget is not null && fs != "lean")
        {
            _log.LogError(
                "Element targets currently only support --feature-set lean (got '{Fs}'). " +
                "Rich variants are not defined for the per-variable blenders yet.", fs);
            return 2;
        }

        // Leads 96h + 120h are currently scoped to temperature + precipitation only.
        // Dry-window and Element targets stay capped at 72h until separately scoped.
        if (t == "dry-window" || elementTarget is not null)
        {
            foreach (var unsupported in new[] { 96, 120 })
            {
                if (leads.Contains(unsupported) && lead.Equals(unsupported.ToString(), StringComparison.OrdinalIgnoreCase))
                {
                    _log.LogError("--lead {N} is not supported for target '{T}' yet (temperature + precipitation only).", unsupported, t);
                    return 2;
                }
            }
            // "all" implicitly includes 96 + 120 — silently narrow rather than error.
            leads = leads.Where(l => l != 96 && l != 120).ToArray();
        }

        return t switch
        {
            "temperature"   => fs switch
            {
                "rich"  => await RunPhase2cAsync(leads, ct),
                "exact" => await RunPhase2dAsync(tier, includeUkv, exactLeads, cycles, ct),
                _       => await RunPhase2bAsync(leads, ct),
            },
            "precipitation" => fs switch
            {
                "rich"  => await RunPhase3cAsync(leads, station, ct),
                "exact" => await RunPhase3dAsync(station, tier, includeUkv, exactLeads, cycles, ct),
                _       => await RunPhase3aAsync(leads, station, ct),
            },
            // dry-window: lean → Phase 3b (53 features),
            //             independence-mc → Phase 3g (parameter-free MC over 3a marginals).
            // "rich" silently maps to 3b for symmetry with the temperature/precip
            // dispatch — there's no rich dry-window variant after 3d-shape was
            // retired 2026-05-04.
            "dry-window"    => await _dryWindow.RunAsync(
                                   station ?? "all", window ?? "all", leads,
                                   fs == "independence-mc"
                                       ? Train.DryWindow.DryWindow3gPredictor.Phase3g
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
        "96"  => new[] { 96 },
        "120" => new[] { 120 },
        _     => null,
    };

    private async Task<int> RunPhase2bAsync(int[] leads, CancellationToken ct)
    {
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now);
        var versionName = Path.GetFileName(versionDir);

        var hp = new TempTrainer.Hyperparameters();
        _log.LogInformation("Phase 2b — training per-lead blenders for leads [{Leads}]",
            string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // Buffer the first-lead train slice for training_summary.json
        // (Phase 1a of AUTO_RETRAIN_PLAN.md). Single-lead avoids mixing
        // lead-axis variability into the per-feature distribution bands
        // the retrain guard reads. Row counts ARE aggregated across leads.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = TempFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = TempFeatureBuilder.BuildForLead(
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
            // Accumulate row counts across all leads (aggregate signal); buffer
            // the FIRST lead's train features for per-feature stats so the
            // distribution bands aren't blurred by per-lead variability.
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            // Best-single: pick on val (no leakage), then re-score on TEST so the
            // Models page can compare blend-vs-best on the same chronological split.
            var best = TempBaselines.BestSingle(spec, ds.Val);
            var bestValMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;
            var bestTestMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Test, best),
                testActual).Mae;

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
                BestSingleValMae  = bestValMae,
                BestSingleTestMae = bestTestMae,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPct = bestTestMae > 0 ? (blendStats.Mae - bestTestMae) / bestTestMae * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best[{Best}] test MAE={BestT:0.000}°C (val {BestV:0.000}°C), Δ {Delta:+0.0;-0.0;0.0}%, months={M}",
                lead, blendStats.Mae, best, bestTestMae, bestValMae, deltaPct, testMonths);
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
        // RetrainGuard (Phase 1b/1c of AUTO_RETRAIN_PLAN.md): build the
        // training_summary, compare against the previous 2b run's, and
        // abort BEFORE promotion if any tolerance band breached. Per-lead
        // bundles are already on disk by this point but the manifest
        // doesn't reference them until PromoteVersionAsChampion runs —
        // the orphan dir on guard-fail is invisible to predict + verify.
        var guardResult2b = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: "temperature", phase: "2b", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp0)
                ? sp0.FeatureNames.ToList() : Array.Empty<string>());
        if (!guardResult2b.Passed)
        {
            _log.LogError("Aborting Phase 2b retrain — sanity guard failed. Orphan dir {Dir} not promoted.", versionDir);
            return 4;
        }
        // Promote 2b: replaces any prior 2b entry in Active and sets Current.
        // Any active 2c challenger survives untouched.
        ModelArtifact.PromoteVersionAsChampion(modelsRoot, "temperature", versionName, newPhase: "2b");

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
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now, suffix: "phase2c");
        var versionName = Path.GetFileName(versionDir);

        var hp = new TempTrainer.Hyperparameters();
        _log.LogInformation("Phase 2c — rich-feature blender (88 features), leads [{Leads}]",
            string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (identical to Phase 2b — feature richness is the only variable)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // Buffers for training_summary.json — same pattern as 2b. See
        // TrainingSummaryBuilder.BuildAndSave call after the loop.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = TempRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = TempRichFeatureBuilder.BuildForLead(
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
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            var best = TempBaselines.BestSingle(spec, ds.Val);
            var bestValMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;
            var bestTestMaeRich = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Test, best),
                testActual).Mae;

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
                BestSingleValMae  = bestValMae,
                BestSingleTestMae = bestTestMaeRich,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPctRich = bestTestMaeRich > 0 ? (blendStats.Mae - bestTestMaeRich) / bestTestMaeRich * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best[{Best}] test MAE={BestT:0.000}°C (val {BestV:0.000}°C), Δ {Delta:+0.0;-0.0;0.0}%, months={M}",
                lead, blendStats.Mae, best, bestTestMaeRich, bestValMae, deltaPctRich, testMonths);
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
        var guardResult2c = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: "temperature", phase: "2c", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp2c)
                ? sp2c.FeatureNames.ToList() : Array.Empty<string>());
        if (!guardResult2c.Passed)
        {
            _log.LogError("Aborting Phase 2c retrain — sanity guard failed. Orphan dir {Dir} not promoted.", versionDir);
            return 4;
        }

        // Promote 2c as a challenger: replaces any prior 2c entry in Active
        // (so re-training is idempotent) and leaves Current = 2b champion.
        // Predict + verify iterate both versions every cycle.
        ModelArtifact.PromoteVersionAsChallenger(modelsRoot, "temperature", versionName, newPhase: "2c");
        var newActive = ModelArtifact.ResolveActive(modelsRoot, "temperature");

        _log.LogInformation("Phase 2c artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions now: [{Active}]", string.Join(", ", newActive));
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend MAE {kv.Value.BlendTestMae:0.000}°C vs {kv.Value.BestSingle} val MAE {kv.Value.BestSingleValMae:0.000}°C")));

        await Task.CompletedTask;
        return 0;
    }

    // ---- Phase 2d: exact-runtime temperature blender (lead-12 champion) -------------

    /// <summary>
    /// 2d's tier (P1 in the precip-exact world / T2 in the temp-exact world):
    /// GFS + AIFS required, IFS oper + MO Global optional, UKV always
    /// optional. Single-lead training at <see cref="DefaultPhase2dLeads"/>;
    /// each lead gets its own per-lead artefact saved under
    /// <c>v{ts}_phase2d/lead_{N}h.zip</c>.
    /// </summary>
    private static readonly int[] DefaultPhase2dLeads = { 12, 24 };

    private async Task<int> RunPhase2dAsync(string? tierName, bool? includeUkvOpt, int[]? exactLeads, int[]? cycleHoursFilter, CancellationToken ct)
    {
        var leads = exactLeads is { Length: > 0 } ? exactLeads : DefaultPhase2dLeads;
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, "temperature", now, suffix: "phase2d");
        var versionName = Path.GetFileName(versionDir);

        // Bake-off-tuned defaults: lr=0.05/leaves=31/min-leaf=50/feat-frac=1.0
        // is the no-UKV per-lead winner (also a reasonable middle-ground for
        // the UKV-included per-lead winners which differed at lead 12 vs 24).
        // Using one HP set across both leads keeps the artefact reproducible
        // from a single command — per-lead HP tuning can land later if the
        // delta justifies the complexity.
        var hp = new TempTrainer.Hyperparameters(
            LearningRate: 0.05,
            NumberOfLeaves: 31,
            MinimumExampleCountPerLeaf: 50,
            FeatureFraction: 1.0);

        var tier = Exact12hFeatureBuilder.AllTiers.First(t => t.Name == (tierName ?? "T2"));
        bool IncludeUkv = includeUkvOpt ?? true;

        // Cycle-filter logging — null/empty means "all cycles on disk",
        // i.e. production default. Comma-list logged when restricted so
        // bake-off runs leave a clear trail in the log of which cycles
        // contributed to the trained version.
        _log.LogInformation("Phase 2d — exact-runtime blender (tier {Tier}, UKV={Ukv}), leads [{Leads}], cycles [{Cycles}]",
            tier.Name, IncludeUkv, string.Join(",", leads),
            (cycleHoursFilter is { Length: > 0 }) ? string.Join(",", cycleHoursFilter) : "all");
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} min-leaf={Ml} esr={Esr} seed={Seed} feat-frac={Ff}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.MinimumExampleCountPerLeaf,
            hp.EarlyStoppingRound, hp.Seed, hp.FeatureFraction);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // training_summary buffers (Phase 1a). 2d uses exact-runtime cycles
        // but the dataset shape + features are otherwise the same regression
        // as 2b/2c — same buffer-and-aggregate pattern.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = Exact12hFeatureBuilder.BuildSpec(tier, targetLead: lead, includeUkv: IncludeUkv);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = Exact12hFeatureBuilder.Build(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.Era5Path,
                _cfg.Location.Name,
                tier,
                spec,
                targetLead: lead,
                includeUkv: IncludeUkv,
                runCycleHoursFilter: cycleHoursFilter,
                ct: ct);
            _log.LogInformation("Loaded {N} rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);
            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train (run s3-collect / backfill first).",
                    rows.Count, lead);
                return 3;
            }

            var ds = RegressionDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            var best = TempBaselines.BestSingle(spec, ds.Val);
            var bestValMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;
            var bestTestMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Test, best),
                testActual).Mae;

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
                BestSingleValMae  = bestValMae,
                BestSingleTestMae = bestTestMae,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPct = bestTestMae > 0 ? (blendStats.Mae - bestTestMae) / bestTestMae * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best[{Best}] test MAE={BestT:0.000}°C (val {BestV:0.000}°C), Δ {Delta:+0.0;-0.0;0.0}%, months={M}",
                lead, blendStats.Mae, best, bestTestMae, bestValMae, deltaPct, testMonths);
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2d",
            DataSource = "exact_runtime_s3_archive",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Trains on RunTimeSource='exact' rows from raw S3 archives (NOAA + ECMWF Open Data + MO AWS) — distinct from the 2b/2c offset_day path. Per-row provenance is rigorous (RunTimeUtc + ValidTimeUtc + LeadHours).",
                "Models = GFS + AIFS required, IFS oper + MO Global optional (tier T2). UKV always optional, sourced via per-V-hour conditional pull from 03Z + 15Z cycles to land at ~12h-ahead per ValidTime.",
                "Lead set restricted to {12, 24} — the leads with sufficient cycle coverage at the 4-cycle ValidTime grid {00,06,12,18}. 48/72/96/120 not trained; 2b/2c retain championship at those leads.",
                "Hyperparameters from the 2026-05-05 81-config grid search: lr=0.05, leaves=31, min-leaf=50, feat-frac=1.0 — the no-UKV per-lead winner, used as a pragmatic split between the per-lead UKV-included winners (lr=0.1/leaves=15 at L=12 vs lr=0.02/leaves=31 at L=24) which diverged.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        var firstLead2d = leads.Length > 0 ? leads[0] : 0;
        var guardResult2d = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: "temperature", phase: "2d", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(firstLead2d, out var sp2d)
                ? sp2d.FeatureNames.ToList() : Array.Empty<string>());
        if (!guardResult2d.Passed)
        {
            _log.LogError("Aborting Phase 2d retrain — sanity guard failed. Orphan dir {Dir} not promoted; ChampionByLead retains the previous 2d pin (manifest unchanged).", versionDir);
            return 4;
        }

        // Promote 2d as a challenger — 2b stays Current. ChampionByLead pins
        // 2d at lead 12 ONLY (where 2b doesn't train); 24+ falls through
        // to Current (= 2b). PromoteVersionAsChallenger is lead-set-aware
        // (post-2026-05-08) so a 2d retrain at e.g. {72,96,120} no longer
        // clobbers a sibling 2d at {12,24,48}; both stay Active when their
        // lead-sets differ.
        //
        // ChampionByLead pin is gated on whether THIS train run actually
        // covered lead 12. Pre-2026-05-08 the trainer pinned 12 →
        // versionName regardless, which left ChampionByLead.12 dangling
        // at a version with no lead 12 in its schema (silently broke the
        // home-page lead-12 tile filter).
        const int Phase2dChampionLead = 12;
        ModelArtifact.PromoteVersionAsChallenger(modelsRoot, "temperature", versionName, newPhase: "2d");
        if (leads.Contains(Phase2dChampionLead))
        {
            ModelArtifact.SetChampionForLead(modelsRoot, "temperature", Phase2dChampionLead, versionName);
        }
        else
        {
            _log.LogInformation(
                "Skipping ChampionByLead pin — this run did not train lead {Lead}h (leads: [{Leads}]).",
                Phase2dChampionLead, string.Join(",", leads));
        }
        var newActive = ModelArtifact.ResolveActive(modelsRoot, "temperature");

        _log.LogInformation("Phase 2d artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions now: [{Active}]", string.Join(", ", newActive));
        if (leads.Contains(Phase2dChampionLead))
            _log.LogInformation("Champion for lead {Lead}h: {V}", Phase2dChampionLead, versionName);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend MAE {kv.Value.BlendTestMae:0.000}°C vs {kv.Value.BestSingle} val MAE {kv.Value.BestSingleValMae:0.000}°C")));

        await Task.CompletedTask;
        return 0;
    }

    private static Dictionary<string, object> BuildHpDict(TempTrainer.Hyperparameters hp) => new()
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
        // Slug is prefixed with the provider ("ea_") so a future Met Office
        // collector for the same site can live alongside the EA one without colliding.
        var stationSlug = StationSlug.WithEaPrefix(primaryStation);
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
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
        // training_summary buffers (Phase 1a). Binary classification — also
        // track first-lead train labels for the per-station label rate.
        List<float[]>? firstLeadTrainFeatures = null;
        IReadOnlyList<bool>? firstLeadTrainLabels = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

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
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();

            var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var blendProb = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, ds.Test);
            var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

            var best = PrecipBaselines.BestSingle(spec, ds.Val);
            var bestValBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Val, best),
                ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
            var bestTestBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Test, best),
                truthTest);
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
                BestSingleValMae  = bestValBrier,   // reused column — Brier
                BestSingleTestMae = bestTestBrier,
                BlendTestMae  = blendBrier,
                BlendTestRmse = climBrier,         // reused column — climatology Brier
                BlendTestBias = fbias,
            };

            var deltaPctP = bestTestBrier > 0 ? (blendBrier - bestTestBrier) / bestTestBrier * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, clim={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, fbias={Fb:0.00}, best[{Best}] test Brier={BestT:0.000} (val {BestV:0.000}), Δ {Delta:+0.0;-0.0;0.0}%",
                lead, blendBrier, climBrier, bss, fbias, best, bestTestBrier, bestValBrier, deltaPctP);
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
                "Secondary rainfall stations not trained in this artefact. Bellever is the brief's primary truth; secondaries remain available for cross-station verification in the evaluation report.",
                "PerLeadStats fields repurposed: BlendTestMae=blend Brier, BlendTestRmse=climatology Brier, BlendTestBias=frequency bias at p=0.5, BestSingleValMae=best-single Brier on val. Artefact schema unchanged to avoid breaking Phase 2b loaders.",
                "Microsoft.ML.LightGbm 4.0 constraints: no explicit class-weight option beyond UnbalancedSets=true; no monotone constraints. Recorded per Phase 2b pattern.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        // training_summary with per-station label rate (binary classifier).
        var labelRates3a = firstLeadTrainLabels is { Count: > 0 }
            ? new Dictionary<string, double>
              {
                  [stationSlug] = firstLeadTrainLabels.Count(l => l) / (double)firstLeadTrainLabels.Count,
              }
            : null;
        var guardResult3a = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"precipitation/{stationSlug}", phase: "3a", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp3a)
                ? sp3a.FeatureNames.ToList() : Array.Empty<string>(),
            labelRates: labelRates3a);
        if (!guardResult3a.Passed)
        {
            _log.LogError("Aborting Phase 3a retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted.", stationSlug, versionDir);
            return 4;
        }
        // Promote 3a: replaces any prior 3a entry in the per-station Active
        // and sets Current. Any active 3c challenger survives untouched.
        ModelArtifact.PromoteStationVersionAsChampion(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3a");

        _log.LogInformation("Phase 3a artefacts → {Dir}", versionDir);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        var (cf, cs) = await _precipConformal.FitOneAsync(
            stationSlug, versionName, PrecipConformalFitCommand.DefaultAlpha, ct);
        _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {S2}/{V}",
            cf, cs, stationSlug, versionName);
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

        var modelsRoot = _cfg.Storage.ModelsPath;
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
        var stationSlug = StationSlug.WithEaPrefix(primaryStation);
        var now = DateTime.UtcNow;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now, suffix: "phase3c");
        var versionName = Path.GetFileName(versionDir);

        _log.LogInformation("=== Station '{Station}' (slug={Slug}) ===", primaryStation, stationSlug);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        PrecipClimatology? climatology = null;
        // training_summary buffers (Phase 1a). Same shape as 3a.
        List<float[]>? firstLeadTrainFeatures = null;
        IReadOnlyList<bool>? firstLeadTrainLabels = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

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
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();

            var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var blendProb = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, ds.Test);
            var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

            var best = PrecipBaselines.BestSingle(spec, ds.Val);
            var bestValBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Val, best),
                ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
            var bestTestBrierR = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Test, best),
                truthTest);
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
                BestSingleValMae  = bestValBrier,
                BestSingleTestMae = bestTestBrierR,
                BlendTestMae  = blendBrier,
                BlendTestRmse = climBrier,
                BlendTestBias = fbias,
            };

            var deltaPctR = bestTestBrierR > 0 ? (blendBrier - bestTestBrierR) / bestTestBrierR * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, clim={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, fbias={Fb:0.00}, best[{Best}] test Brier={BestT:0.000} (val {BestV:0.000}), Δ {Delta:+0.0;-0.0;0.0}%",
                lead, blendBrier, climBrier, bss, fbias, best, bestTestBrierR, bestValBrier, deltaPctR);
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
        var labelRates3c = firstLeadTrainLabels is { Count: > 0 }
            ? new Dictionary<string, double>
              {
                  [stationSlug] = firstLeadTrainLabels.Count(l => l) / (double)firstLeadTrainLabels.Count,
              }
            : null;
        var guardResult3c = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"precipitation/{stationSlug}", phase: "3c", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp3c)
                ? sp3c.FeatureNames.ToList() : Array.Empty<string>(),
            labelRates: labelRates3c);
        if (!guardResult3c.Passed)
        {
            _log.LogError("Aborting Phase 3c retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted.", stationSlug, versionDir);
            return 4;
        }

        // Promote 3c as a challenger: replaces any prior 3c entry in Active
        // (idempotent re-train) and leaves Current = 3a champion. Any other
        // active phases survive untouched.
        ModelArtifact.PromoteStationVersionAsChallenger(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3c");
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);

        _log.LogInformation("Phase 3c artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        var (cf, cs) = await _precipConformal.FitOneAsync(
            stationSlug, versionName, PrecipConformalFitCommand.DefaultAlpha, ct);
        _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {S2}/{V}",
            cf, cs, stationSlug, versionName);
        return 0;
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

    // ---- Phase 3d: exact-runtime precip blender (per-station, lead-12 champion) ---

    private static readonly int[] DefaultPhase3dLeads = { 12, 24 };

    private async Task<int> RunPhase3dAsync(string? stationOverride, string? tierName, bool? includeUkvOpt, int[]? exactLeads, int[]? cycleHoursFilter, CancellationToken ct)
    {
        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured — cannot train precipitation blender.");
            return 2;
        }

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

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        var tier = PrecipExactFeatureBuilder.AllTiers.First(t => t.Name == (tierName ?? "P1"));
        bool IncludeUkv = includeUkvOpt ?? true;
        var leadsToTrain = exactLeads is { Length: > 0 } ? exactLeads : DefaultPhase3dLeads;

        _log.LogInformation("Phase 3d — exact-runtime precip blender (tier {Tier}, UKV={Ukv}), stations=[{Stations}], leads=[{Leads}], cycles=[{Cycles}]",
            tier.Name, IncludeUkv, string.Join(", ", stationsToTrain), string.Join(",", leadsToTrain),
            (cycleHoursFilter is { Length: > 0 }) ? string.Join(",", cycleHoursFilter) : "all");
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (3a defaults)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var anyFail = false;
        foreach (var station in stationsToTrain)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await TrainPhase3dStationAsync(station, modelsRoot, tier, hp, IncludeUkv, leadsToTrain, cycleHoursFilter, ct);
            if (rc != 0) anyFail = true;
        }
        return anyFail ? 3 : 0;
    }

    private async Task<int> TrainPhase3dStationAsync(
        string primaryStation,
        string modelsRoot,
        PrecipExactFeatureBuilder.TierSpec tier,
        PrecipOccurrenceTrainer.Hyperparameters hp,
        bool includeUkv,
        int[] leadsToTrain,
        int[]? cycleHoursFilter,
        CancellationToken ct)
    {
        var stationSlug = StationSlug.WithEaPrefix(primaryStation);
        var now = DateTime.UtcNow;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now, suffix: "phase3d");
        var versionName = Path.GetFileName(versionDir);

        _log.LogInformation("=== Station '{Station}' (slug={Slug}) ===", primaryStation, stationSlug);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        PrecipClimatology? climatology = null;
        // training_summary buffers (Phase 1a). Same shape as 3a/3c.
        List<float[]>? firstLeadTrainFeatures = null;
        IReadOnlyList<bool>? firstLeadTrainLabels = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        foreach (var lead in leadsToTrain)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = PrecipExactFeatureBuilder.BuildSpec(tier, targetLead: lead, includeUkv: includeUkv);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            // 3d trains against the per-station EA Hydrology gauge — matches
            // 3a/3c's truth so Brier numbers are head-to-head comparable on
            // the same target. ERA5 was wrong: ~25km grid smooths out the
            // localised Dartmoor signal, making the task spuriously easy
            // (caught + fixed 2026-05-05).
            var rows = PrecipExactFeatureBuilder.Build(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                _cfg.Location.Name,
                stationName: primaryStation,
                tier, spec,
                targetLead: lead,
                includeUkv: includeUkv,
                runCycleHoursFilter: cycleHoursFilter,
                ct: ct);
            _log.LogInformation("Loaded {N} rows (wet={Wet} / {Pct:P1}) spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count(r => r.Label),
                rows.Count == 0 ? 0 : (double)rows.Count(r => r.Label) / rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train (run s3-collect / metoffice backfill first).",
                    rows.Count, lead);
                return 3;
            }

            var ds = BinaryDataset.Split(rows);
            climatology ??= PrecipClimatology.BuildFromTraining(ds.Train);

            _log.LogInformation("Split → train {TN} (wet {TW}), val {VN} (wet {VW}), test {EN} (wet {EW})",
                ds.Train.Count, ds.TrainWet,
                ds.Val.Count,   ds.ValWet,
                ds.Test.Count,  ds.TestWet);
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();

            var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var blendProb = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, ds.Test);
            var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

            var best = PrecipBaselines.BestSingle(spec, ds.Val);
            var bestValBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Val, best),
                ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
            var bestTestBrier = PrecipMetrics.Brier(
                PrecipBaselines.SingleModelWet(spec, ds.Test, best),
                truthTest);
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
                BestSingleValMae  = bestValBrier,
                BestSingleTestMae = bestTestBrier,
                BlendTestMae  = blendBrier,
                BlendTestRmse = climBrier,
                BlendTestBias = fbias,
            };

            var deltaPct = bestTestBrier > 0 ? (blendBrier - bestTestBrier) / bestTestBrier * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, clim={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, fbias={Fb:0.00}, best[{Best}] test Brier={BestT:0.000} (val {BestV:0.000}), Δ {Delta:+0.0;-0.0;0.0}%",
                lead, blendBrier, climBrier, bss, fbias, best, bestTestBrier, bestValBrier, deltaPct);
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);
        if (climatology is not null)
            climatology.SaveTo(Path.Combine(versionDir, ModelArtifact.ClimatologyFileName));

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "precipitation",
            Phase = "3d",
            DataSource = "exact_runtime_s3_archive+ea_rainfall",
            TrainedAtUtc = DateTime.UtcNow,
            Hyperparameters = BuildPrecipHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Trains on RunTimeSource='exact' rows from raw S3 archives (NOAA + ECMWF Open Data + MO AWS) — distinct from the 3a/3c offset_day path. Per-row provenance is rigorous (RunTimeUtc + ValidTimeUtc + LeadHours).",
                "Models = GFS + IFS oper + AIFS required, MO Global optional; UKV always optional via per-V-hour conditional pull from 03Z + 15Z cycles. Lead-12 reads UKV at leads {9, 15} (avg 12h-ahead); lead-24 reads at {21, 27} (avg 24h-ahead).",
                "Lead set restricted to {12, 24} — the leads with sufficient cycle coverage at the 4-cycle ValidTime grid {00, 06, 12, 18}. 48/72/96/120 not trained; 3a/3c retain championship at those leads.",
                "Truth = per-station EA Hydrology rainfall gauge — same source as 3a/3c so Brier scores are head-to-head comparable on the same per-station target.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        var labelRates3d = firstLeadTrainLabels is { Count: > 0 }
            ? new Dictionary<string, double>
              {
                  [stationSlug] = firstLeadTrainLabels.Count(l => l) / (double)firstLeadTrainLabels.Count,
              }
            : null;
        var firstLead3d = leadsToTrain.Length > 0 ? leadsToTrain[0] : 0;
        var guardResult3d = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"precipitation/{stationSlug}", phase: "3d", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(firstLead3d, out var sp3d)
                ? sp3d.FeatureNames.ToList() : Array.Empty<string>(),
            labelRates: labelRates3d);
        if (!guardResult3d.Passed)
        {
            _log.LogError("Aborting Phase 3d retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted; ChampionByLead retains the previous 3d pin (manifest unchanged).", stationSlug, versionDir);
            return 4;
        }

        // Promote 3d as a station challenger — 3a stays Current. Per-station
        // ChampionByLead pins 3d at lead 12 ONLY (where 3a competes well too,
        // but 3d's exact-runtime feature set wins on Brier). Lead-set-aware
        // promote (post-2026-05-08) lets a 3d retrain at e.g. {96,120} coexist
        // with a sibling 3d at {12,24,48,72} when their lead-sets differ.
        // ChampionByLead pin gated on whether THIS run trained lead 12.
        const int Phase3dChampionLead = 12;
        ModelArtifact.PromoteStationVersionAsChallenger(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3d");
        if (leadsToTrain.Contains(Phase3dChampionLead))
        {
            ModelArtifact.SetStationChampionForLead(
                modelsRoot, "precipitation", stationSlug, leadHours: Phase3dChampionLead, versionName);
        }
        else
        {
            _log.LogInformation(
                "Skipping ChampionByLead pin for {Station} — this run did not train lead {Lead}h (leads: [{Leads}]).",
                stationSlug, Phase3dChampionLead, string.Join(",", leadsToTrain));
        }
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);

        _log.LogInformation("Phase 3d artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));
        if (leadsToTrain.Contains(Phase3dChampionLead))
            _log.LogInformation("Champion for lead {Lead}h ({Station}): {V}", Phase3dChampionLead, stationSlug, versionName);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        // Auto-conformal skipped for 3d: PrecipConformalFitCommand re-loads
        // training rows via PrecipFeatureBuilder, which only knows the
        // offset_day model IDs (gfs_seamless etc.). 3d's exact-runtime IDs
        // (gfs_ncep etc.) trip ShortName(). Conformal cal can be added
        // later via a 3d-specific re-fit; until then, ConformalSetTag is
        // null on 3d rows (legacy-row behaviour, harmless).
        await Task.CompletedTask;
        return 0;
    }
}
