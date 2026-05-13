using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.DryWindow;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;
using WeatherBlend.Train.Mlp;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3b trainer. Produces up to 18 LightGBM classifiers:
/// rainfall stations × {3h, 4h, 6h} × {24, 48, 72}.
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
    // Trained stations are now read from `_cfg.Location.Rainfall.Stations`. A
    // hardcoded two-station list was set in Phase 3b before Hexworthy joined
    // the rainfall config (2026-04-26) and never got updated, leaving the
    // dry-window family with 2 stations while precip had 3. Reading from
    // config eliminates that drift root cause.

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
            && phase != DryWindow3gPredictor.Phase3g
            && phase != DryWindowFeatureBuilder.Phase3f)
        {
            _log.LogError("Unsupported dry-window training phase '{Phase}'. Expected '3b', '3f' or '3g'.", phase);
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

        // Phase 3f — TorchSharp MLP on the same day-level features as 3b.
        // Bake-off challenger; never promoted to champion.
        if (phase == DryWindowFeatureBuilder.Phase3f)
        {
            var windows3f = ParseWindows(windowArg);
            if (windows3f is null)
            {
                _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
                return 2;
            }
            return await RunPhase3fAsync(stations, windows3f, leads, ct);
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
                // training_summary buffers (Phase 1a). Per-(station, window)
                // since each combo writes its own versionDir + metadata.
                List<float[]>? firstLeadTrainFeatures = null;
                IReadOnlyList<bool>? firstLeadTrainLabels = null;
                int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

                // Buffer per-row test predictions across all leads for this
                // (station, window) — written once at the bottom of the
                // window loop. Sibling of TempTrainCommand.testPredictionRows
                // but on the day-level dry-window schema. Consumed by dry-
                // window bake-offs (e.g. 3b vs 3g vs MC-over-calibrated-3a).
                var testPredictionRows = new List<DryWindowTestPredictionRow>();

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
                    totalTrainRows += ds.Train.Count;
                    totalValRows   += ds.Val.Count;
                    totalTestRows  += ds.Test.Count;
                    firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
                    firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();

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

                    // Buffer per-row held-out predictions for the bake-off
                    // parquet. ds.Test is in row-order; blendProbShipping is
                    // index-aligned. p_dry_window is the SHIPPING value so a
                    // downstream Brier on the bake-off parquet matches what
                    // the site renders.
                    var stationSlug = StationSlug.WithEaPrefix(stationName);
                    for (int i = 0; i < ds.Test.Count; i++)
                    {
                        testPredictionRows.Add(new DryWindowTestPredictionRow
                        {
                            target_date          = ds.Test[i].TargetDateUtc,
                            station              = stationSlug,
                            window               = window,
                            lead                 = lead,
                            p_dry_window         = blendProbShipping[i],
                            observed_dry_window  = (byte)(ds.Test[i].Label ? 1 : 0),
                        });
                    }

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

                // test_predictions.parquet — per-row held-out predictions
                // accumulated across the leads in this (station, window).
                // Schema matches DryWindowTestPredictionRow (target_date,
                // station, window, lead, p_dry_window, observed_dry_window).
                // Consumed by dry-window bake-off scripts that need to
                // inner-join 3b's direct prediction against MC-based
                // alternatives (3g + variants) on the same test cells.
                // p_dry_window is the SHIPPING probability — matches what
                // the site renders.
                if (testPredictionRows.Count > 0)
                {
                    var testPredPath = Path.Combine(versionDir, "test_predictions.parquet");
                    await Parquet.Serialization.ParquetSerializer.SerializeAsync(
                        testPredictionRows, testPredPath, cancellationToken: ct);
                    _log.LogInformation("Wrote {N} test_predictions rows → {Path}",
                        testPredictionRows.Count, testPredPath);
                }

                var metadata = new ModelArtifact.TrainingMetadata
                {
                    Version = versionName,
                    Target = "dry_window",
                    Phase = phase,
                    LocationName = _cfg.Location.Name,
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
                var stationSlug3b = StationSlug.WithEaPrefix(stationName);
                var labelRates3b = firstLeadTrainLabels is { Count: > 0 }
                    ? new Dictionary<string, double>
                      {
                          [stationSlug3b] = firstLeadTrainLabels.Count(l => l) / (double)firstLeadTrainLabels.Count,
                      }
                    : null;
                var firstLead3b = leads.Length > 0 ? leads[0] : 0;
                var guardResult3b = RetrainGuard.BuildCheckAndSave(_log,
                    versionDir,
                    composite: $"dry_window/{stationSlug3b}/window_{window}h",
                    phase: phase, version: versionName,
                    computedAtUtc: now,
                    rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
                    trainFeatures: firstLeadTrainFeatures,
                    featureNames: specsPerLead.TryGetValue(firstLead3b, out var sp3b)
                        ? sp3b.FeatureNames.ToList() : Array.Empty<string>(),
                    labelRates: labelRates3b,
                    locationName: _cfg.Location.Name);
                if (!guardResult3b.Passed)
                {
                    // Single-(station, window) guard fail aborts JUST this
                    // combo's promotion + conformal fit; the outer sweep
                    // continues training other (station, window) cells. Bumps
                    // modelsSkipped so the final summary log still reflects
                    // partial success.
                    _log.LogError(
                        "Aborting Phase {Phase} promotion for ({Station}, {W}h) — sanity guard failed. Orphan dir {Dir} not promoted; previous version stays Current.",
                        phase, stationName, window, versionDir);
                    modelsSkipped++;
                    continue;
                }
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
                    LocationName = _cfg.Location.Name,
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

    /// <summary>
    /// Phase 3f — TorchSharp MLP on the same day-level rich features as 3b
    /// (built via <see cref="DryWindowFeatureBuilder.BuildForLead"/> with
    /// phase=3b for feature parity). Same chronological 70/15/15 split as
    /// 3b, scored on the same test slice — so the bake-off compares
    /// like-for-like.
    ///
    /// Architecture deliberately small for the small-N tabular regime
    /// (~700 day rows per cell vs 3e's ~30k hourly): HiddenSizes=[32,16],
    /// Dropout=0.3, BatchSize=64. Anything larger overfits. The MLP
    /// trainer's NaN-aware standardiser + dead-column safety net (added
    /// 2026-05-13 to fix 3e's offset_day-vs-reported cloud_*_mean OOD
    /// collapse) carry over for free.
    ///
    /// Bundle layout mirrors 3e's precip MLP under the dry_window tree:
    /// <code>
    /// data/models/dry_window/{station}/window_{N}h/v..._phase3f/
    ///   mlp_lead_24h.pt
    ///   mlp_lead_48h.pt
    ///   mlp_lead_72h.pt
    ///   preprocess.json
    ///   feature_schema.json
    ///   training_metadata.json
    ///   test_predictions.parquet
    /// </code>
    ///
    /// 3f never promotes to champion — it's a bake-off challenger only.
    /// No conformal fit (the bake-off cares about Brier, not coverage).
    /// </summary>
    private async Task<int> RunPhase3fAsync(string[] stations, int[] windows, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();

        // Tabular-small-N MLP — see XML doc above. Defaults differ from 3e
        // because 3e fits 30k rows, 3f fits ~700.
        var mlpHp = new MlpTrainer.Hyperparameters(
            HiddenSizes: new[] { 32, 16 },
            Dropout: 0.3,
            LearningRate: 1e-3,
            BatchSize: 64,
            MaxEpochs: 200,
            EarlyStoppingPatience: 20,
            Seed: 42);

        _log.LogInformation(
            "Phase 3f (MLP) — stations=[{S}] windows=[{W}] leads=[{L}]; arch={Arch} dropout={D} bs={B}",
            string.Join(", ", stations), string.Join(",", windows), string.Join(",", leads),
            string.Join("x", mlpHp.HiddenSizesEffective), mlpHp.Dropout, mlpHp.BatchSize);

        int trained = 0, skipped = 0;

        foreach (var stationName in stations)
        {
            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var stationSlug = StationSlug.WithEaPrefix(stationName);
                var compositeKey = $"{stationSlug}/window_{window}h";
                var now = DateTime.UtcNow;
                var versionDir = ModelArtifact.BuildStationVersionDir(
                    modelsRoot, "dry_window", compositeKey, now, "phase3f");
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{S}', window {W}h → 3f {V} ===",
                    stationName, window, versionName);

                var specsPerLead = new Dictionary<int, BlenderSpec>();
                var perLeadStats = new Dictionary<string, ModelArtifact.PerLeadStats>();
                var perLeadPreprocess = new Dictionary<string, MlpArtifact.PerLeadPreprocess>(StringComparer.Ordinal);
                var testPredictionRows = new List<DryWindowTestPredictionRow>();
                DryWindowClimatology? climatology = null;
                bool anyLeadTrained = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {L}h ---", lead);

                    // Reuse 3b's spec + row builder so the chronological split
                    // matches 3b's exactly — bake-off requires identical test
                    // slices across phases or Brier numbers aren't comparable.
                    var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
                    specsPerLead[lead] = spec;
                    _log.LogInformation("Spec: {Spec}", spec);

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

                    // Convert day-level dry-window rows to the BinaryTrainingRow
                    // shape MlpTrainer expects. Only Features + Label matter —
                    // ValidTimeUtc is preserved but the trainer ignores it.
                    var trainBin = ds.Train.Select(r => new BinaryTrainingRow {
                        ValidTimeUtc = r.TargetDateUtc,
                        Features = r.Features,
                        Label = r.Label,
                    }).ToList();
                    var valBin   = ds.Val.Select(r => new BinaryTrainingRow {
                        ValidTimeUtc = r.TargetDateUtc,
                        Features = r.Features,
                        Label = r.Label,
                    }).ToList();
                    var testBin  = ds.Test.Select(r => new BinaryTrainingRow {
                        ValidTimeUtc = r.TargetDateUtc,
                        Features = r.Features,
                        Label = r.Label,
                    }).ToList();

                    var trainedMlp = MlpTrainer.TrainVector(trainBin, valBin, spec, mlpHp);
                    var testProbs = MlpTrainer.PredictVectorProbability(trainedMlp, testBin);
                    var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                    var brier = PrecipMetrics.Brier(testProbs, truthTest);
                    var climPred = DryWindowBaselines.Climatology(ds.Train, ds.Test, window);
                    var climBrier = PrecipMetrics.Brier(climPred, truthTest);
                    var bss = PrecipMetrics.BrierSkillScore(brier, climBrier);

                    _log.LogInformation(
                        "  Lead {L}h MLP test Brier={B:0.0000} (clim {C:0.0000}, BSS={Bss:+0.0000;-0.0000;0.0000}, epochs={E}, best_val={V:0.0000})",
                        lead, brier, climBrier, bss, trainedMlp.EpochsRun, trainedMlp.BestValBrier);

                    // Save the lead's .pt + accumulate the preprocess block.
                    var perLead = MlpArtifact.SaveLeadModel(versionDir, lead, trainedMlp, spec);
                    perLeadPreprocess[lead.ToString()] = perLead;

                    perLeadStats[lead.ToString()] = new ModelArtifact.PerLeadStats
                    {
                        LeadHours = lead,
                        DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd}Z → {ds.TrainEnd:yyyy-MM-dd}Z",
                        DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd}Z → {ds.ValEnd:yyyy-MM-dd}Z",
                        DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd}Z → {ds.TestEnd:yyyy-MM-dd}Z",
                        TrainRows = ds.Train.Count,
                        ValRows   = ds.Val.Count,
                        TestRows  = ds.Test.Count,
                        TestCalendarMonths = ds.Test
                            .Select(r => new DateTime(r.TargetDateUtc.Year, r.TargetDateUtc.Month, 1))
                            .Distinct().Count(),
                        BlendTestMae = brier,
                        BlendTestRmse = climBrier,
                        BlendTestBias = 0.0,
                        CalibratedBlendTestMae = brier,   // MLP doesn't ship a separate calibrator
                    };

                    for (int i = 0; i < ds.Test.Count; i++)
                    {
                        testPredictionRows.Add(new DryWindowTestPredictionRow
                        {
                            target_date         = ds.Test[i].TargetDateUtc,
                            station             = stationSlug,
                            window              = window,
                            lead                = lead,
                            p_dry_window        = testProbs[i],
                            observed_dry_window = (byte)(ds.Test[i].Label ? 1 : 0),
                        });
                    }
                    anyLeadTrained = true;
                    trained++;
                }

                if (!anyLeadTrained)
                {
                    _log.LogWarning("  No leads trained for ({S}, {W}h); skipping artefact save.",
                        stationName, window);
                    continue;
                }

                Directory.CreateDirectory(versionDir);
                ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
                MlpArtifact.WritePreprocess(versionDir,
                    new MlpArtifact.Preprocess(PerLead: perLeadPreprocess));
                if (climatology is not null)
                    climatology.SaveTo(Path.Combine(versionDir, "dry_window_climatology.json"));

                var metadata = new ModelArtifact.TrainingMetadata
                {
                    Version = versionName,
                    Target = "dry_window",
                    Phase = DryWindowFeatureBuilder.Phase3f,
                    LocationName = _cfg.Location.Name,
                    DataSource = "previous_runs_api+ea_rainfall",
                    TrainedAtUtc = now,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["hidden_sizes"] = mlpHp.HiddenSizesEffective,
                        ["dropout"] = mlpHp.Dropout,
                        ["learning_rate"] = mlpHp.LearningRate,
                        ["batch_size"] = mlpHp.BatchSize,
                        ["max_epochs"] = mlpHp.MaxEpochs,
                        ["early_stopping_patience"] = mlpHp.EarlyStoppingPatience,
                        ["seed"] = mlpHp.Seed,
                        ["window_hours"] = window,
                    },
                    TestMae = perLeadStats.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLeadStats,
                    DeviationsFromBrief = new List<string>
                    {
                        "Phase 3f — TorchSharp MLP on the same day-level rich features as 3b. Bake-off challenger; never promoted to champion.",
                        "Architecture [32, 16] / dropout 0.3 / batch 64 chosen for the small-N tabular regime (~700 day rows per cell). 3e's [128, 64, 32] would be ~18x overparameterised at this data scale.",
                        "Conformal calibration omitted — 3f is offline-only for the bake-off, not a predict-path phase.",
                        "Spec reuses 3b's BuildSpec(phase=3b) so train/val/test split + feature vector are identical, enabling like-for-like Brier comparison against 3b.",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

                if (testPredictionRows.Count > 0)
                {
                    var testPredPath = Path.Combine(versionDir, "test_predictions.parquet");
                    await Parquet.Serialization.ParquetSerializer.SerializeAsync(
                        testPredictionRows, testPredPath, cancellationToken: ct);
                    _log.LogInformation("Wrote {N} test_predictions rows → {Path}",
                        testPredictionRows.Count, testPredPath);
                }

                // 3f is a challenger by design — register in the manifest's
                // Active list so the model registry can see it, but DON'T
                // promote to champion (Current stays whatever 3b/3g set).
                ModelArtifact.PromoteStationVersionAsChallenger(
                    modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: DryWindowFeatureBuilder.Phase3f);

                _log.LogInformation("Saved 3f MLP artefacts → {Dir}", versionDir);
            }
        }

        _log.LogInformation("Phase 3f training complete. Trained={T} Skipped={S}", trained, skipped);
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
