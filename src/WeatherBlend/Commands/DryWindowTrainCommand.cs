using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.DryWindow;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3b trainer. Produces up to 18 LightGBM classifiers:
/// {Bellever, Princetown} × {3h, 4h, 6h} × {24, 48, 72}.
///
/// Artefact layout uses composite station keys (<c>ea_bellever_dartmoor/window_3h</c>)
/// so <see cref="ModelArtifact"/> helpers apply unmodified. One version directory
/// per (station, window) pair, three lead-zips inside.
/// </summary>
public sealed class DryWindowTrainCommand
{
    private readonly ILogger<DryWindowTrainCommand> _log;
    private readonly AppConfig _cfg;
    private readonly ModelMetadataRepository _metadata;
    // Auto-refit conformal calibrators after every promote-to-(champion|challenger).
    // Without this hook a fresh version ships with no calibrator; live predict
    // would degrade to the raw model probability and the dry-window page's
    // confidence tags would default to "ambiguous" until the next manual
    // `dry-window-conformal-fit` ran.
    private readonly DryWindowConformalFitCommand _conformal;

    private static readonly int[] DefaultLeads = Leads.Short;
    private static readonly int[] DefaultWindows = { 3, 4, 6 };
    // Trained stations are now read from `_cfg.Location.Rainfall.Stations`. The
    // hardcoded {Bellever, Princetown} list was set in Phase 3b before
    // Hexworthy joined the rainfall config (2026-04-26) and never got
    // updated, leaving the dry-window family with 2 stations while precip
    // had 3. Reading from config eliminates that drift root cause.

    public DryWindowTrainCommand(
        ILogger<DryWindowTrainCommand> log,
        AppConfig cfg,
        ModelMetadataRepository metadata,
        DryWindowConformalFitCommand conformal)
    {
        _log = log;
        _cfg = cfg;
        _metadata = metadata;
        _conformal = conformal;
    }

    public Task<int> RunAsync(string stationArg, string windowArg, int[] leads, CancellationToken ct)
        => RunAsync(stationArg, windowArg, leads, DryWindowFeatureBuilder.Phase3b, ct);

    public async Task<int> RunAsync(string stationArg, string windowArg, int[] leads, string phase, CancellationToken ct)
    {
        if (phase != DryWindowFeatureBuilder.Phase3b
            && phase != DryWindow3gPredictor.Phase3g)
        {
            _log.LogError("Unsupported dry-window training phase '{Phase}'. Expected '3b' or '3g'.", phase);
            return 2;
        }

        var stations = ResolveStations(stationArg);
        if (stations is null) return 2;

        // Phase 3g — parameter-free MC over Phase 3a's hourly P(wet) marginals.
        // No LightGBM, no model artefacts: each "version" is just metadata +
        // climatology pointing at the 3a champion the predictor will read at
        // inference time. Dispatched separately because the per-lead loop
        // structure (build features → train → save model) doesn't apply.
        if (phase == DryWindow3gPredictor.Phase3g)
        {
            var windows3g = ParseWindows(windowArg);
            if (windows3g is null)
            {
                _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
                return 2;
            }
            return await RunPhase3gAsync(stations, windows3g, leads, ct);
        }

        var windows = ParseWindows(windowArg);
        if (windows is null)
        {
            _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
            return 2;
        }

        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured.");
            return 2;
        }

        _log.LogInformation("Phase {Phase} — stations=[{Stations}] windows=[{W}] leads=[{L}]",
            phase, string.Join(", ", stations), string.Join(",", windows), string.Join(",", leads));

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = new DryWindowTrainer.Hyperparameters();
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var overallStart = DateTime.UtcNow;
        int modelsTrained = 0, modelsSkipped = 0;

        foreach (var stationName in stations)
        {
            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{StationSlug.WithEaPrefix(stationName)}/window_{window}h";
                var now = DateTime.UtcNow;
                // 3b → v{ts} (champion, no suffix). 3g uses its own RunPhase3gAsync
                // which sets the _phase3g suffix; the 3d-shape / 3e suffix branches
                // were retired 2026-05-04.
                var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "dry_window", compositeKey, now);
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{Station}', window {W}h → {Key} ===",
                    stationName, window, compositeKey);

                var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
                var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
                var specsPerLead = new Dictionary<int, BlenderSpec>();
                DryWindowClimatology? climatology = null;
                bool anyLeadTrained = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {Lead}h ---", lead);

                    var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, phase);
                    specsPerLead[lead] = spec;
                    _log.LogInformation("Spec: {Spec}", spec);

                    var rows = DryWindowFeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath,
                        _cfg.Storage.RainfallPath,
                        _cfg.Location.Name,
                        stationName,
                        spec,
                        window,
                        _cfg.DryWindow.BuildDaytimeWindow(),
                        ct);

                    var pos = rows.Count(r => r.Label);
                    _log.LogInformation("Loaded {N} rows (positives={Pos} / {Pct:P1}) spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                        rows.Count, pos,
                        rows.Count == 0 ? 0 : (double)pos / rows.Count,
                        rows.Count == 0 ? DateTime.MinValue : rows[0].TargetDateUtc,
                        rows.Count == 0 ? DateTime.MinValue : rows[^1].TargetDateUtc);

                    if (rows.Count < 100)
                    {
                        _log.LogWarning("Only {N} rows for ({Station}, {W}h, lead {Lead}h) — skipping lead.",
                            rows.Count, stationName, window, lead);
                        modelsSkipped++;
                        continue;
                    }

                    var ds = WeatherBlend.Train.Common.DryWindowDataset.Split(rows);
                    // Climatology stays on the legacy day-keyed structure for predict/verify
                    // backward compat — derive from the new vector rows via month + label.
                    climatology ??= BuildClimatologyFromVectorRows(ds.Train, window);

                    _log.LogInformation("Split → train {TN} (pos {TP}), val {VN} (pos {VP}), test {EN} (pos {EP})",
                        ds.Train.Count, ds.TrainPositives,
                        ds.Val.Count,   ds.ValPositives,
                        ds.Test.Count,  ds.TestPositives);
                    _log.LogInformation("Date ranges — train {T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}, " +
                                        "val {V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}, test {E0:yyyy-MM-dd}..{E1:yyyy-MM-dd}",
                        ds.TrainStart, ds.TrainEnd, ds.ValStart, ds.ValEnd, ds.TestStart, ds.TestEnd);

                    var trained = DryWindowTrainer.TrainVector(ds.Train, ds.Val, spec, hp);
                    var calibrationEnabled = _cfg.DryWindow.ShouldCalibrate(stationName);

                    var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                    var blendProbRaw = DryWindowTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, ds.Test);
                    var blendProbCal = trained.Calibrator.PredictMany(blendProbRaw);
                    var blendBrierRaw = PrecipMetrics.Brier(blendProbRaw, truthTest);
                    var blendBrierCal = PrecipMetrics.Brier(blendProbCal, truthTest);

                    // Shipping probabilities = calibrated iff this station opted in.
                    // Both numbers logged + persisted regardless so the experiment trail
                    // stays in metadata.
                    var blendProbShipping = calibrationEnabled ? blendProbCal : blendProbRaw;
                    var blendBrierShipping = calibrationEnabled ? blendBrierCal : blendBrierRaw;

                    var best = DryWindowBaselines.BestSingle(spec, ds.Val);
                    var bestValBrier = PrecipMetrics.Brier(
                        DryWindowBaselines.FromFeature(spec, ds.Val, best),
                        ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
                    var bestTestBrier = PrecipMetrics.Brier(
                        DryWindowBaselines.FromFeature(spec, ds.Test, best),
                        truthTest);

                    var climPred = DryWindowBaselines.Climatology(ds.Train, ds.Test, window);
                    var climBrier = PrecipMetrics.Brier(climPred, truthTest);
                    var bss = PrecipMetrics.BrierSkillScore(blendBrierShipping, climBrier);

                    // Frequency bias measured on the SHIPPING probabilities — what readers
                    // see on the site is what the diagnostic should describe.
                    var blendBinary = blendProbShipping.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
                    var fbias = PrecipMetrics.FrequencyBias(blendBinary, truthTest);

                    ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
                    if (calibrationEnabled)
                        ModelArtifact.SaveLeadCalibrator(trained.Calibrator, versionDir, lead);
                    importanceByLead[lead] = trained.FeatureImportance;
                    anyLeadTrained = true;

                    var testMonths = ds.Test
                        .Select(r => new DateTime(r.TargetDateUtc.Year, r.TargetDateUtc.Month, 1))
                        .Distinct().Count();

                    perLead[lead.ToString()] = new ModelArtifact.PerLeadStats
                    {
                        LeadHours = lead,
                        DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd}Z → {ds.TrainEnd:yyyy-MM-dd}Z",
                        DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd}Z → {ds.ValEnd:yyyy-MM-dd}Z",
                        DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd}Z → {ds.TestEnd:yyyy-MM-dd}Z",
                        TrainRows = ds.Train.Count,
                        ValRows   = ds.Val.Count,
                        TestRows  = ds.Test.Count,
                        TestCalendarMonths = testMonths,
                        BestSingle = best,
                        BestSingleValMae  = bestValBrier,  // reused column — documented in DeviationsFromBrief
                        BestSingleTestMae = bestTestBrier,
                        BlendTestMae  = blendBrierShipping, // matches the deployed prediction
                        BlendTestRmse = climBrier,          // reused for climatology Brier
                        BlendTestBias = fbias,              // bias of the SHIPPING prediction
                        CalibratedBlendTestMae = blendBrierCal, // for the experiment trail
                    };

                    var calDelta = blendBrierCal - blendBrierRaw;
                    var calPct = blendBrierRaw > 0 ? calDelta / blendBrierRaw * 100.0 : 0.0;
                    var calTag = calibrationEnabled ? "calibrated SHIPPED" : "raw shipped, calibrated for reference";
                    _log.LogInformation(
                        "Lead {Lead}h — raw Brier={BrierRaw:0.0000}, calibrated={BrierCal:0.0000} ({CalPct:+0.0;-0.0;0.0}%), {Tag}, " +
                        "clim={Clim:0.0000}, BSS={Bss:+0.0000;-0.0000;0.0000}, " +
                        "freq_bias={Fb:0.00}, best_single[{Best}] test Brier={BestBrier:0.0000}",
                        lead, blendBrierRaw, blendBrierCal, calPct, calTag, climBrier, bss, fbias, best, bestTestBrier);

                    modelsTrained++;
                }

                if (!anyLeadTrained)
                {
                    _log.LogWarning("No leads trained for ({Station}, {W}h); skipping artefact save.",
                        stationName, window);
                    continue;
                }

                ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
                ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);
                if (climatology is not null)
                {
                    climatology.SaveTo(Path.Combine(versionDir, "dry_window_climatology.json"));
                }

                var metadata = new ModelArtifact.TrainingMetadata
                {
                    Version = versionName,
                    Target = "dry_window",
                    Phase = phase,
                    DataSource = "previous_runs_api+ea_rainfall",
                    TrainedAtUtc = now,
                    Hyperparameters = BuildHpDict(hp, window),
                    TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLead,
                    DeviationsFromBrief = new List<string>
                    {
                        "PerLeadStats fields reused to avoid schema divergence from 3a: BlendTestMae=blend Brier, BlendTestRmse=climatology Brier, BlendTestBias=frequency bias at p=0.5, BestSingleValMae=best-single Brier on val.",
                        "Microsoft.ML.LightGbm constraints: no monotone constraints, no class-weight tuning beyond UnbalancedSets=false. Probability calibration is LightGBM default + ML.NET Platt wrapper.",
                        "MANIFEST composite station key: <station-slug>/window_<N>h so ModelArtifact.UpdateStationManifest / ResolveStationVersionDir apply unmodified. Documented in PHASE3B_AUDIT.md.",
                        "Cross-midnight dry stretches are not credited to either UTC day; accepted target-construction limitation (see DryWindowLabelBuilder remarks).",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
                // Promote the new 3b version: replace any prior 3b entry in
                // Active with this one and set Current = newVersion. Any
                // OTHER active phases (3g challengers today) survive.
                ModelArtifact.PromoteStationVersionAsChampion(
                    modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: DryWindowFeatureBuilder.Phase3b);
                var (cf, cs) = await _conformal.FitOneAsync(
                    compositeKey, versionName, DryWindowConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {K}/{V}",
                    cf, cs, compositeKey, versionName);

                _log.LogInformation("Saved artefacts → {Dir}", versionDir);
            }
        }

        var elapsed = DateTime.UtcNow - overallStart;
        _log.LogInformation("Phase 3b training complete. Trained={T} Skipped={S} Elapsed={E}",
            modelsTrained, modelsSkipped, elapsed);

        await Task.CompletedTask;
        return modelsTrained == 0 ? 3 : 0;
    }


    /// <summary>
    /// Phase 3g training loop. There's no LightGBM training to do — 3g is
    /// purely MC sampling over Phase 3a's hourly P(wet) marginals at predict
    /// time. What this method does instead:
    ///   1. Build the same dry-window training rows as 3b (so the chronological
    ///      70/15/15 split lands on the same dates).
    ///   2. For each (station, window, lead) cell: load the 3a champion's
    ///      replay parquet for that lead, score 3g raw on the test split,
    ///      record Brier in <c>BlendTestMae</c>.
    ///   3. Write a synthetic version directory containing only:
    ///      <c>training_metadata.json</c> (Phase=3g, references the 3a
    ///      champion in Hyperparameters) and
    ///      <c>dry_window_climatology.json</c> (same as 3b's helper).
    ///   4. Promote each (station, window) entry as a CHALLENGER alongside
    ///      the existing 3b champion.
    /// 3g version dirs deliberately contain no <c>lead_*.zip</c> files —
    /// <see cref="DryWindowPredictCommand"/> detects Phase=3g in metadata and
    /// dispatches to the parameter-free MC predict path that reads the 3a
    /// prediction parquet directly.
    /// </summary>
    private async Task<int> RunPhase3gAsync(string[] stations, int[] windows, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        var rngSeed = 42;
        var mcSamples = DryWindow3gPredictor.DefaultMcSamples;

        _log.LogInformation(
            "Phase 3g — stations=[{S}] windows=[{W}] leads=[{L}], MC samples={Mc}",
            string.Join(", ", stations), string.Join(",", windows),
            string.Join(",", leads), mcSamples);

        // Resolve precip 3a champion per station once. 3g binds to ONE 3a
        // version per station for the lifetime of the artefact — re-running
        // 3g training picks up the latest 3a champion at that moment.
        var precipManifest = _metadata.TryGetManifest("precipitation");
        if (precipManifest?.Stations is null)
        {
            _log.LogError("3g needs the precipitation manifest to resolve 3a champions; not found.");
            return 2;
        }

        int trained = 0, skipped = 0;
        foreach (var stationName in stations)
        {
            var stationSlug = StationSlug.WithEaPrefix(stationName);
            if (!precipManifest.Stations.TryGetValue(stationSlug, out var precipEntry)
                || string.IsNullOrEmpty(precipEntry.Current))
            {
                _log.LogWarning(
                    "{S}: no 3a champion in precipitation manifest; skipping 3g.",
                    stationName);
                continue;
            }
            var precip3aVersion = precipEntry.Current;

            // Pre-load all needed replay parquets once per station.
            var replayByLead = new Dictionary<int, Dictionary<DateTime, double>>();
            foreach (var lead in leads)
            {
                replayByLead[lead] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3aVersion, lead);
                _log.LogInformation(
                    "  loaded {N} 3a replay hourly P(wet) for {S} lead {L}h",
                    replayByLead[lead].Count, stationName, lead);
            }

            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{stationSlug}/window_{window}h";
                var now = DateTime.UtcNow;
                var versionDir = ModelArtifact.BuildStationVersionDir(
                    modelsRoot, "dry_window", compositeKey, now, "phase3g");
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{S}', window {W}h → 3g {V} ===",
                    stationName, window, versionName);

                var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
                DryWindowClimatology? climatology = null;
                bool anyLeadScored = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {L}h ---", lead);

                    // Reuse 3b's spec + row builder so the chronological split
                    // is identical to 3b's. We don't USE the day-aggregate
                    // features for 3g — we only need the (date, label) sequence.
                    var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
                    var rows = DryWindowFeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                        _cfg.Location.Name, stationName,
                        spec, window, daytime, ct);

                    if (rows.Count < 100)
                    {
                        _log.LogWarning("  only {N} rows for ({S}, {W}h, lead {L}h); skipping lead.",
                            rows.Count, stationName, window, lead);
                        skipped++;
                        continue;
                    }

                    var ds = DryWindowDataset.Split(rows);
                    climatology ??= BuildClimatologyFromVectorRows(ds.Train, window);

                    _log.LogInformation("  Split → train {Tn}, val {Vn}, test {En}",
                        ds.Train.Count, ds.Val.Count, ds.Test.Count);

                    // Score 3g raw on the test split. PAV deliberately not
                    // applied — the 2026-05-03 bake-off found PAV hurts more
                    // than it helps (Hexworthy 6h flips 22% Brier the wrong
                    // way; raw beats PAV on 15/27 cells overall).
                    var hourly = replayByLead[lead];
                    var rng = new Random(rngSeed);
                    var probs = new List<double>(ds.Test.Count);
                    var labels = new List<bool>(ds.Test.Count);
                    foreach (var row in ds.Test)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, row.TargetDateUtc, s, e);
                        if (q is null) continue;   // replay gap — skip honest reporting
                        probs.Add(DryWindow3gPredictor.ProbDryWindow(q, window, rng, mcSamples));
                        labels.Add(row.Label);
                    }

                    if (probs.Count < 10)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: only {N} test rows after replay-gap filter; skipping.",
                            stationName, window, lead, probs.Count);
                        skipped++;
                        continue;
                    }

                    var brier = Brier(probs, labels);
                    var climPred = DryWindowBaselines.Climatology(ds.Train, ds.Test, window);
                    var climBrier = WeatherBlend.Evaluate.Precip.PrecipMetrics.Brier(
                        climPred, ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray());

                    perLead[lead.ToString()] = new ModelArtifact.PerLeadStats
                    {
                        LeadHours = lead,
                        DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd}Z → {ds.TrainEnd:yyyy-MM-dd}Z",
                        DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd}Z → {ds.ValEnd:yyyy-MM-dd}Z",
                        DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd}Z → {ds.TestEnd:yyyy-MM-dd}Z",
                        TrainRows = ds.Train.Count,
                        ValRows   = ds.Val.Count,
                        TestRows  = probs.Count,
                        TestCalendarMonths = ds.Test
                            .Select(r => new DateTime(r.TargetDateUtc.Year, r.TargetDateUtc.Month, 1))
                            .Distinct().Count(),
                        BlendTestMae = brier,
                        BlendTestRmse = climBrier,
                        BlendTestBias = 0.0,
                        CalibratedBlendTestMae = brier,  // raw == shipped for 3g
                    };
                    anyLeadScored = true;
                    trained++;
                    _log.LogInformation(
                        "  Lead {L}h Brier={B:0.0000} (clim {C:0.0000}, n_test={N})",
                        lead, brier, climBrier, probs.Count);
                }

                if (!anyLeadScored)
                {
                    _log.LogWarning("  No leads scored for ({S}, {W}h); skipping artefact save.",
                        stationName, window);
                    continue;
                }

                Directory.CreateDirectory(versionDir);
                if (climatology is not null)
                    climatology.SaveTo(Path.Combine(versionDir, "dry_window_climatology.json"));

                var metadata = new ModelArtifact.TrainingMetadata
                {
                    Version = versionName,
                    Target = "dry_window",
                    Phase = DryWindow3gPredictor.Phase3g,
                    DataSource = $"precipitation_replay@{precip3aVersion}",
                    TrainedAtUtc = now,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["mc_samples"] = mcSamples,
                        ["seed"] = rngSeed,
                        ["precip_3a_version"] = precip3aVersion,
                        ["window_hours"] = window,
                    },
                    TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLead,
                    DeviationsFromBrief = new List<string>
                    {
                        "Phase 3g — parameter-free MC over Phase 3a hourly P(wet) marginals under independence. No model.zip files; predict reads 3a's live prediction parquet at inference time and runs MC.",
                        "Cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) is guaranteed by computing all windows from a SINGLE shared Bernoulli sequence per MC sample (DryWindow3gPredictor.ProbDryWindow).",
                        $"3a champion bound at training time: {precip3aVersion}. Re-run dry-window train --feature-set independence-mc to rebind to a newer 3a champion.",
                        "PAV calibration deliberately omitted (2026-05-03 bake-off: PAV hurt 15/27 cells; Hexworthy 6h flipped 22% Brier the wrong way).",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

                ModelArtifact.PromoteStationVersionAsChallenger(
                    modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: DryWindow3gPredictor.Phase3g);

                var (cf, cs) = await _conformal.FitOneAsync(
                    compositeKey, versionName, DryWindowConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {K}/{V}",
                    cf, cs, compositeKey, versionName);

                _log.LogInformation("Saved 3g artefacts → {Dir}", versionDir);
            }
        }

        _log.LogInformation("Phase 3g training complete. Scored={T} Skipped={S}", trained, skipped);
        await Task.CompletedTask;
        return trained == 0 ? 3 : 0;
    }

    private static double Brier(IReadOnlyList<double> probs, IReadOnlyList<bool> labels)
    {
        var sum = 0.0;
        for (int i = 0; i < probs.Count; i++)
        {
            var y = labels[i] ? 1.0 : 0.0;
            sum += (probs[i] - y) * (probs[i] - y);
        }
        return sum / probs.Count;
    }

    private string[]? ResolveStations(string stationArg)
    {
        if (string.IsNullOrWhiteSpace(stationArg) || stationArg.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            var found = _cfg.Location.Rainfall.Stations.Select(s => s.Name).ToArray();
            if (found.Length == 0)
            {
                _log.LogError("No rainfall stations in config — nothing to train.");
                return null;
            }
            return found;
        }

        var match = _cfg.Location.Rainfall.Stations
            .FirstOrDefault(s => s.Name.Equals(stationArg, StringComparison.OrdinalIgnoreCase)
                              || SlugMatch(s.Name, stationArg));
        if (match is null)
        {
            _log.LogError("Station '{Arg}' not in config. Available: [{Names}]",
                stationArg, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
            return null;
        }
        return new[] { match.Name };
    }

    private static bool SlugMatch(string configName, string arg)
    {
        var slug = StationSlug.Of(configName);
        return slug.Equals(arg, StringComparison.OrdinalIgnoreCase)
            || ($"ea_{slug}").Equals(arg, StringComparison.OrdinalIgnoreCase);
    }

    private static int[]? ParseWindows(string w) => w.ToLowerInvariant() switch
    {
        "all" => DefaultWindows,
        "3"   => new[] { 3 },
        "4"   => new[] { 4 },
        "6"   => new[] { 6 },
        _     => null,
    };

    /// <summary>
    /// Build a legacy-shaped <see cref="DryWindowClimatology"/> from the new
    /// vector-row dataset. Kept on the legacy shape so predict-time loaders
    /// (which read climatology.json) don't need to change in this phase.
    /// </summary>
    private static DryWindowClimatology BuildClimatologyFromVectorRows(
        IReadOnlyList<CommonRow> trainRows, int windowHours)
    {
        // Match the legacy DryWindowClimatology layout — month-keyed P(yes-dry-window).
        var sums = new Dictionary<string, (int Pos, int N)>();
        foreach (var r in trainRows)
        {
            var k = r.TargetDateUtc.Month.ToString("D2");
            sums.TryGetValue(k, out var cur);
            sums[k] = (cur.Pos + (r.Label ? 1 : 0), cur.N + 1);
        }
        var pDry = sums.ToDictionary(kv => kv.Key, kv => (double)kv.Value.Pos / kv.Value.N);
        var globalRate = trainRows.Count == 0 ? 0.0 : trainRows.Count(r => r.Label) / (double)trainRows.Count;
        return new DryWindowClimatology
        {
            PDryByMonth = pDry,
            GlobalPositiveRate = globalRate,
            SourceRowCount = trainRows.Count,
            WindowHours = windowHours,
        };
    }

    private static Dictionary<string, object> BuildHpDict(DryWindowTrainer.Hyperparameters hp, int windowHours)
        => new()
        {
            ["number_of_iterations"] = hp.NumberOfIterations,
            ["learning_rate"] = hp.LearningRate,
            ["number_of_leaves"] = hp.NumberOfLeaves,
            ["minimum_example_count_per_leaf"] = hp.MinimumExampleCountPerLeaf,
            ["l1_regularization"] = hp.L1Regularization,
            ["l2_regularization"] = hp.L2Regularization,
            ["early_stopping_round"] = hp.EarlyStoppingRound,
            ["seed"] = hp.Seed,
            ["window_hours"] = windowHours,
        };
}
