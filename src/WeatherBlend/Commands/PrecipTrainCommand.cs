using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.PrecipExact;

namespace WeatherBlend.Commands;

/// <summary>
/// Trains the precipitation blenders — one model per (station, lead), scored
/// on Brier against per-station EA Hydrology rainfall truth.
///
/// Phase 3a: LightGBM binary classifier for P(hour has >= 0.1 mm). Leads {24,48,72,120}.
/// Phase 3c: rich-feature LightGBM occurrence blender — 3a champion/challenger.
/// Phase 3e: TorchSharp MLP occurrence blender — head-to-head NN-vs-GBT on the 3c spec.
/// Phase 3d: exact-runtime occurrence blender (P1/P2 tier), lead-12 champion.
///
/// Split out of TempTrainCommand on 2026-05-19 (code-quality refactor P1) so a
/// precipitation phase is no longer navigated inside a file named "Temp". The
/// cross-target dispatch + argument validation stays in TempTrainCommand.RunAsync,
/// which calls RunAsync below once the target resolves to precipitation.
/// </summary>
public sealed class PrecipTrainCommand : TrainCommandBase
{
    // Auto-refit conformal calibrators after every promote-to-(champion|challenger).
    // Without this hook a fresh version ships with no calibrator; live predict
    // would degrade to the raw model probability and the dry-window page's
    // confidence tags would default to "ambiguous" until the next manual
    // precip-conformal-fit ran.
    private readonly PrecipConformalFitCommand _precipConformal;

    // Env var escape hatch for bake-off + research training where the
    // post-train conformal fit (~5 min/lead) is dead weight. Set
    // WB_SKIP_CONFORMAL=1 to skip the FitOneAsync call after a 3a/3c train.
    // Production retrain workflows never set this — they want the conformal
    // sidecar so live predict can emit ConformalSetTag.
    private readonly bool _skipConformal =
        string.Equals(Environment.GetEnvironmentVariable("WB_SKIP_CONFORMAL"),
            "1", StringComparison.Ordinal);

    public PrecipTrainCommand(
        ILogger<PrecipTrainCommand> log,
        AppConfig cfg,
        PrecipConformalFitCommand precipConformal)
        : base(log, cfg)
    {
        _precipConformal = precipConformal;
    }

    /// <summary>
    /// Dispatches a precipitation train run to the phase implied by
    /// <paramref name="featureSet"/>: lean -> 3a, rich -> 3c, mlp -> 3e,
    /// exact -> 3d. Feature-set validity for the precipitation target is
    /// checked by the caller (TempTrainCommand.RunAsync) before dispatch.
    /// </summary>
    public async Task<int> RunAsync(
        int[] leads, string? station, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, int[]? cycles,
        Config.LocationConfig location, CancellationToken ct)
    {
        return featureSet switch
        {
            "rich"  => await RunPhase3cAsync(leads, station, location, ct),
            "exact" => await RunPhase3dAsync(station, tier, includeUkv, exactLeads, cycles, location, ct),
            "mlp"   => await RunPhase3eAsync(leads, station, location, ct),
            _       => await RunPhase3aAsync(leads, station, location, ct),
        };
    }

    // ---- Phase 3a: precipitation occurrence classifier ----------------------------

    private async Task<int> RunPhase3aAsync(int[] leads, string? stationOverride, Config.LocationConfig location, CancellationToken ct)
    {
        if (location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}' — cannot train precipitation blender.", location.Name);
            return 2;
        }

        string primaryStation;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            primaryStation = location.Rainfall.Stations[0].Name;
        }
        else
        {
            var match = location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found in location '{Loc}' config. Available: {Available}",
                    stationOverride, location.Name, string.Join(", ", location.Rainfall.Stations.Select(s => s.Name)));
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
        // Per-row held-out test predictions for downstream bake-offs (e.g.
        // 3a + 4a linear pool). Schema mirrors 5a's test_predictions.parquet
        // so a single bake-off script can inner-join across phases. Buffered
        // across the lead loop, written after.
        var testPredictionRows = new List<TestPredictionRow>();

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
                location.Name,
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

            // Buffer per-row test predictions for the bake-off parquet.
            // ds.Test is in row-order; blendProb is index-aligned.
            for (int i = 0; i < ds.Test.Count; i++)
            {
                testPredictionRows.Add(new TestPredictionRow
                {
                    valid_time   = ds.Test[i].ValidTimeUtc,
                    station      = stationSlug,
                    lead         = lead,
                    p_wet        = blendProb[i],
                    observed_wet = (byte)(ds.Test[i].Label ? 1 : 0),
                });
            }

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
            LocationName = location.Name,
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
        // test_predictions.parquet — per-row held-out probabilities for
        // downstream bake-offs (e.g. 3a + 4a linear pool). Same schema as
        // 5a's test_predictions.parquet (valid_time, station, lead, p_wet,
        // observed_wet) so a single bake-off script can inner-join across
        // phases without per-phase schema branches.
        if (testPredictionRows.Count > 0)
        {
            await ParquetSerializer.SerializeAsync(
                testPredictionRows, Path.Combine(versionDir, "test_predictions.parquet"),
                cancellationToken: ct);
        }
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
            labelRates: labelRates3a,
            locationName: location.Name);
        if (!guardResult3a.Passed)
        {
            _log.LogError("Aborting Phase 3a retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted.", stationSlug, versionDir);
            return 4;
        }
        // Promote 3a: replaces any prior 3a entry in the per-station Active
        // and sets Current. Any active 3c challenger survives untouched.
        ModelArtifact.PromoteStationVersion(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3a");

        _log.LogInformation("Phase 3a artefacts → {Dir}", versionDir);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        if (_skipConformal)
        {
            _log.LogInformation("Auto-conformal: SKIPPED (WB_SKIP_CONFORMAL=1)");
        }
        else
        {
            var (cf, cs) = await _precipConformal.FitOneAsync(
                stationSlug, versionName, PrecipConformalFitCommand.DefaultAlpha, ct);
            _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {S2}/{V}",
                cf, cs, stationSlug, versionName);
        }
        return 0;
    }

    // ---- Phase 3c: rich-feature precip occurrence blender (champion/challenger) ----

    private async Task<int> RunPhase3cAsync(int[] leads, string? stationOverride, Config.LocationConfig location, CancellationToken ct)
    {
        if (location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}' — cannot train precipitation blender.", location.Name);
            return 2;
        }

        // When no station override is given, 3c iterates every configured rainfall
        // station so one command run produces a challenger for each 3a champion.
        // A specific station still trains that one alone.
        IReadOnlyList<string> stationsToTrain;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stationsToTrain = location.Rainfall.Stations.Select(s => s.Name).ToList();
        }
        else
        {
            var match = location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found in location '{Loc}' config. Available: {Available}",
                    stationOverride, location.Name, string.Join(", ", location.Rainfall.Stations.Select(s => s.Name)));
                return 2;
            }
            stationsToTrain = new[] { match.Name };
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = new PrecipOccurrenceTrainer.Hyperparameters();

        _log.LogInformation("Phase 3c — rich-feature precip blender, location='{Loc}', stations=[{Stations}], leads=[{Leads}]",
            location.Name, string.Join(", ", stationsToTrain), string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (identical to Phase 3a — feature richness is the only variable)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var anyFail = false;
        foreach (var station in stationsToTrain)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await TrainPhase3cStationAsync(station, leads, modelsRoot, hp, location, ct);
            if (rc != 0) anyFail = true;
        }
        return anyFail ? 3 : 0;
    }

    private async Task<int> TrainPhase3cStationAsync(
        string primaryStation,
        int[] leads,
        string modelsRoot,
        PrecipOccurrenceTrainer.Hyperparameters hp,
        Config.LocationConfig location,
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
        // Per-row held-out test predictions for the bake-off parquet —
        // same canonical schema 3a/3e/4a/5a write so a single stacking
        // analysis can inner-join across phases. Added 2026-05-11
        // (3c was previously missing test_predictions, blocking
        // 3c-vs-3e-vs-4a stacking work).
        var testPredictionRows = new List<TestPredictionRow>();

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
                location.Name,
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

            // Per-row test predictions for the bake-off parquet (2026-05-11
            // — was missing from 3c, blocking 3c/3e/4a stacking analysis).
            for (int i = 0; i < ds.Test.Count; i++)
            {
                testPredictionRows.Add(new TestPredictionRow
                {
                    valid_time   = ds.Test[i].ValidTimeUtc,
                    station      = stationSlug,
                    lead         = lead,
                    p_wet        = blendProb[i],
                    observed_wet = (byte)(ds.Test[i].Label ? 1 : 0),
                });
            }
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);
        if (climatology is not null)
            climatology.SaveTo(Path.Combine(versionDir, ModelArtifact.ClimatologyFileName));
        if (testPredictionRows.Count > 0)
        {
            var testPredPath = Path.Combine(versionDir, "test_predictions.parquet");
            await Parquet.Serialization.ParquetSerializer.SerializeAsync(
                testPredictionRows, testPredPath, cancellationToken: ct);
            _log.LogInformation("Wrote {N} test prediction rows → {Path}",
                testPredictionRows.Count, testPredPath);
        }

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "precipitation",
            Phase = "3c",
            LocationName = location.Name,
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
            locationName: location.Name,
            labelRates: labelRates3c);
        if (!guardResult3c.Passed)
        {
            _log.LogError("Aborting Phase 3c retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted.", stationSlug, versionDir);
            return 4;
        }

        // Promote 3c as a challenger: replaces any prior 3c entry in Active
        // (idempotent re-train) and leaves Current = 3a champion. Any other
        // active phases survive untouched.
        ModelArtifact.PromoteStationVersion(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3c");
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);

        _log.LogInformation("Phase 3c artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        if (_skipConformal)
        {
            _log.LogInformation("Auto-conformal: SKIPPED (WB_SKIP_CONFORMAL=1)");
        }
        else
        {
            var (cf, cs) = await _precipConformal.FitOneAsync(
                stationSlug, versionName, PrecipConformalFitCommand.DefaultAlpha, ct);
            _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {S2}/{V}",
                cf, cs, stationSlug, versionName);
        }
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

    // ---- Phase 3e: TorchSharp MLP precip blender (per-station, per-lead) ----------

    private async Task<int> RunPhase3eAsync(int[] leads, string? stationOverride, Config.LocationConfig location, CancellationToken ct)
    {
        if (location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}' — cannot train MLP precip blender.", location.Name);
            return 2;
        }

        // Same wrapper as 3c: auto-iterate stations when no --station given,
        // single-station otherwise. 3e is a challenger on the same per-station
        // axis as 3a/3c so it slots cleanly into the existing per-station
        // manifest layout.
        IReadOnlyList<string> stationsToTrain;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stationsToTrain = location.Rainfall.Stations.Select(s => s.Name).ToList();
        }
        else
        {
            var match = location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found in location '{Loc}' config. Available: {Available}",
                    stationOverride, location.Name, string.Join(", ", location.Rainfall.Stations.Select(s => s.Name)));
                return 2;
            }
            stationsToTrain = new[] { match.Name };
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = new Train.Mlp.MlpTrainer.Hyperparameters();

        _log.LogInformation("Phase 3e — TorchSharp MLP precip blender, location='{Loc}', stations=[{Stations}], leads=[{Leads}]",
            location.Name, string.Join(", ", stationsToTrain), string.Join(",", leads));
        _log.LogInformation("Hyperparameters: hidden=[{H}] dropout={D} lr={Lr} batch={B} maxEp={M} earlyStop={ES} seed={S}",
            string.Join(",", hp.HiddenSizesEffective), hp.Dropout, hp.LearningRate,
            hp.BatchSize, hp.MaxEpochs, hp.EarlyStoppingPatience, hp.Seed);

        var anyFail = false;
        foreach (var station in stationsToTrain)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await TrainPhase3eStationAsync(station, leads, modelsRoot, hp, location, ct);
            if (rc != 0) anyFail = true;
        }
        return anyFail ? 3 : 0;
    }

    private async Task<int> TrainPhase3eStationAsync(
        string primaryStation,
        int[] leads,
        string modelsRoot,
        Train.Mlp.MlpTrainer.Hyperparameters hp,
        Config.LocationConfig location,
        CancellationToken ct)
    {
        var stationSlug = StationSlug.WithEaPrefix(primaryStation);
        var now = DateTime.UtcNow;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now, suffix: "phase3e");
        var versionName = Path.GetFileName(versionDir);

        _log.LogInformation("=== Station '{Station}' (slug={Slug}) ===", primaryStation, stationSlug);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var perLeadPreprocess = new Dictionary<string, Train.Mlp.MlpArtifact.PerLeadPreprocess>(StringComparer.Ordinal);
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        PrecipClimatology? climatology = null;
        // training_summary buffers (Phase 1a) — same shape as 3c so the guard
        // can compare 3e training distributions cell-for-cell against 3a/3c.
        List<float[]>? firstLeadTrainFeatures = null;
        IReadOnlyList<bool>? firstLeadTrainLabels = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;
        var testPredictionRows = new List<TestPredictionRow>();

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            // Same 59-feat rich spec as 3c — head-to-head NN-vs-GBT bake-off
            // is only honest if the input vector matches.
            var spec = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = PrecipRichFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                location.Name,
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
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();

            var t0 = DateTime.UtcNow;
            var trained = Train.Mlp.MlpTrainer.TrainVector(ds.Train, ds.Val, spec, hp);
            var trainWall = (DateTime.UtcNow - t0).TotalSeconds;

            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var blendProb = Train.Mlp.MlpTrainer.PredictVectorProbability(trained, ds.Test);
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

            // Save MLP weights + accumulate preprocess block.
            var leadPre = Train.Mlp.MlpArtifact.SaveLeadModel(versionDir, lead, trained, spec);
            perLeadPreprocess[lead.ToString()] = leadPre;

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
                "Lead {Lead}h headline — blend Brier={Brier:0.000}, clim={Clim:0.000}, BSS={Bss:+0.000;-0.000;0.000}, fbias={Fb:0.00}, best[{Best}] test Brier={BestT:0.000} (val {BestV:0.000}), Δ {Delta:+0.0;-0.0;0.0}%, train wall {Wall:0.0}s, epochs run {EpRun}, best val Brier {ValBr:0.000}",
                lead, blendBrier, climBrier, bss, fbias, best, bestTestBrier, bestValBrier, deltaPct, trainWall, leadPre.EpochsRun, leadPre.BestValBrier);

            // Per-row test predictions for the bake-off parquet (stack with 3a/3c/4a output).
            for (int i = 0; i < ds.Test.Count; i++)
            {
                testPredictionRows.Add(new TestPredictionRow
                {
                    valid_time   = ds.Test[i].ValidTimeUtc,
                    station      = stationSlug,
                    lead         = lead,
                    p_wet        = blendProb[i],
                    observed_wet = ds.Test[i].Label ? (byte)1 : (byte)0,
                });
            }
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        // No SavePerLeadFeatureImportance — MLP doesn't expose split-gain
        // importance the way LightGBM does. Permutation importance is a
        // future v2 (run as a separate analysis script if/when wanted).
        if (climatology is not null)
            climatology.SaveTo(Path.Combine(versionDir, ModelArtifact.ClimatologyFileName));
        Train.Mlp.MlpArtifact.WritePreprocess(versionDir, new Train.Mlp.MlpArtifact.Preprocess(
            PerLead: perLeadPreprocess));

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "precipitation",
            Phase = "3e",
            LocationName = location.Name,
            DataSource = "previous_runs_api+ea_rainfall+torchsharp_mlp",
            TrainedAtUtc = DateTime.UtcNow,
            Hyperparameters = BuildMlpHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Same 59-feat rich input as Phase 3c — head-to-head NN-vs-GBT bake-off; feature richness is not the variable.",
                "Plain MLP via TorchSharp (CPU-only libtorch). 128 → 64 → 32 ReLU + dropout 0.2 + linear-1 logits + sigmoid at predict; BCEWithLogitsLoss + Adam(1e-3); standardised inputs; early-stop on val Brier.",
                "Per-station, per-lead bundle layout: data/models/precipitation/{station}/v{ts}_phase3e/. mlp_lead_NNh.pt per lead (TorchSharp state_dict); preprocess.json carries scaler params + hyperparams + per-lead BestValBrier.",
                "Per-feature importance NOT emitted: MLPs don't expose split-gain. Permutation importance is a future v2 analysis run.",
                "Manifest appends 3e to StationEntry.Active alongside 3a (champion) + 3c (challenger). Predict dispatches by metadata.Phase == '3e' to MlpPredictor.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

        if (testPredictionRows.Count > 0)
        {
            var testPredPath = Path.Combine(versionDir, "test_predictions.parquet");
            await Parquet.Serialization.ParquetSerializer.SerializeAsync(
                testPredictionRows, testPredPath, cancellationToken: ct);
            _log.LogInformation("Wrote {N} test prediction rows → {Path}", testPredictionRows.Count, testPredPath);
        }

        var labelRates3e = firstLeadTrainLabels is { Count: > 0 }
            ? new Dictionary<string, double>
              {
                  [stationSlug] = firstLeadTrainLabels.Count(l => l) / (double)firstLeadTrainLabels.Count,
              }
            : null;
        var guardResult3e = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"precipitation/{stationSlug}", phase: "3e", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp3e)
                ? sp3e.FeatureNames.ToList() : Array.Empty<string>(),
            labelRates: labelRates3e,
            locationName: location.Name);
        if (!guardResult3e.Passed)
        {
            _log.LogError("Aborting Phase 3e retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted.", stationSlug, versionDir);
            return 4;
        }

        ModelArtifact.PromoteStationVersion(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3e");
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);

        _log.LogInformation("Phase 3e artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend Brier {kv.Value.BlendTestMae:0.000} vs climatology Brier {kv.Value.BlendTestRmse:0.000}")));

        return 0;
    }

    private static Dictionary<string, object> BuildMlpHpDict(Train.Mlp.MlpTrainer.Hyperparameters hp) => new()
    {
        ["library"]               = "TorchSharp 0.105.0 (CPU-only libtorch)",
        ["architecture"]          = "MLP: Linear → (ReLU + Dropout) per hidden + Linear-1 (logits)",
        ["hiddenSizes"]           = string.Join(",", hp.HiddenSizesEffective),
        ["dropout"]               = hp.Dropout,
        ["learningRate"]          = hp.LearningRate,
        ["batchSize"]             = hp.BatchSize,
        ["maxEpochs"]             = hp.MaxEpochs,
        ["earlyStoppingPatience"] = hp.EarlyStoppingPatience,
        ["seed"]                  = hp.Seed,
        ["loss"]                  = "BCEWithLogitsLoss (sigmoid + BCE; numerically stable)",
        ["optimizer"]             = "Adam",
        ["standardisation"]       = "z-score on training split; persisted in preprocess.json",
        ["earlyStoppingMetric"]   = "val Brier (best-val weights restored before save)",
    };

    // ---- Phase 3d: exact-runtime precip blender (per-station, lead-12 champion) ---

    private static readonly int[] DefaultPhase3dLeads = { 12, 24 };

    private async Task<int> RunPhase3dAsync(string? stationOverride, string? tierName, bool? includeUkvOpt, int[]? exactLeads, int[]? cycleHoursFilter, Config.LocationConfig location, CancellationToken ct)
    {
        if (location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}' — cannot train precipitation blender.", location.Name);
            return 2;
        }

        IReadOnlyList<string> stationsToTrain;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stationsToTrain = location.Rainfall.Stations.Select(s => s.Name).ToList();
        }
        else
        {
            var match = location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found in location '{Loc}' config. Available: {Available}",
                    stationOverride, location.Name, string.Join(", ", location.Rainfall.Stations.Select(s => s.Name)));
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
            var rc = await TrainPhase3dStationAsync(station, modelsRoot, tier, hp, IncludeUkv, leadsToTrain, cycleHoursFilter, location, ct);
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
        Config.LocationConfig location,
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
                location.Name,
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
            LocationName = location.Name,
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
            locationName: location.Name,
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
        ModelArtifact.PromoteStationVersion(
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
