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
            && phase != DryWindowFeatureBuilder.Phase3f
            && phase != DryWindow3jPredictor.Phase3j
            && phase != DryWindow3nPredictor.Phase3n
            && phase != DryWindow3sPredictor.Phase3s)
        {
            _log.LogError("Unsupported dry-window training phase '{Phase}'. Expected '3b', '3f', '3g', '3j', '3n' or '3s'.", phase);
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

        // Phase 3j — Gaussian copula MC over 3a's hourly P(wet). Sibling of 3g,
        // adds a 9×9 Σ per (station, lead) fit on observed daytime binary
        // sequences from 3a's replay parquet. Bake-off challenger registered
        // alongside 3g; per-window-champion routing deferred to a later
        // decision (today's 2026-05-13 bake-off: 3j beats 3g at 3h windows by
        // 4.7% but loses at 6h by 11.3%).
        if (phase == DryWindow3jPredictor.Phase3j)
        {
            var windows3j = ParseWindows(windowArg);
            if (windows3j is null)
            {
                _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
                return 2;
            }
            return await RunPhase3jAsync(stations, windows3j, leads, ct);
        }

        // Phase 3n — regime-conditioned copula MC. Two Σs per (station, lead)
        // split by NWP-consensus regime. See DryWindow3nPredictor for the
        // 2026-05-13 pre-flight diagnostic that motivated splitting Σ.
        if (phase == DryWindow3nPredictor.Phase3n)
        {
            var windows3n = ParseWindows(windowArg);
            if (windows3n is null)
            {
                _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
                return 2;
            }
            return await RunPhase3nAsync(stations, windows3n, leads, ct);
        }

        // Phase 3s — iid MC over Phase 3e's hourly P(wet). Same algorithm as
        // 3g, different marginal source. See DryWindow3sPredictor + the
        // 2026-05-15 start-hour bake-off for why (3e marginals localise the
        // dry block better; occurrence ties 3g).
        if (phase == DryWindow3sPredictor.Phase3s)
        {
            var windows3s = ParseWindows(windowArg);
            if (windows3s is null)
            {
                _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
                return 2;
            }
            return await RunPhase3sAsync(stations, windows3s, leads, ct);
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
                var testPredictionRows = new List<DryWindowTestPredictionRow>();
                DryWindowClimatology? climatology = null;
                bool anyLeadScored = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {L}h ---", lead);

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

                    var hourly = replayByLead[lead];
                    var rng = new Random(rngSeed);
                    var probs = new List<double>(ds.Test.Count);
                    var labels = new List<bool>(ds.Test.Count);
                    foreach (var row in ds.Test)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, row.TargetDateUtc, s, e);
                        if (q is null) continue;
                        var p = DryWindow3gPredictor.ProbDryWindow(q, window, rng, mcSamples);
                        probs.Add(p);
                        labels.Add(row.Label);
                        testPredictionRows.Add(new DryWindowTestPredictionRow
                        {
                            target_date = row.TargetDateUtc,
                            station = stationSlug,
                            window = window,
                            lead = lead,
                            p_dry_window = p,
                            observed_dry_window = (byte)(row.Label ? 1 : 0),
                        });
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
    /// Phase 3s — parameter-free iid MC dry-window predictor, identical
    /// algorithm to 3g but binding Phase 3e's hourly P(wet) as the marginal
    /// source instead of Phase 3a's. See <see cref="DryWindow3sPredictor"/>
    /// for why: the 2026-05-15 start-hour bake-off showed 3e marginals
    /// localise the dry block markedly better (start-hour 6h Brier −4.5%,
    /// improvement at every window) while tying 3g on the occurrence
    /// question. 3s ships as a dry-window challenger whose value is the
    /// start-hour curve.
    ///
    /// Mirrors <see cref="RunPhase3gAsync"/> exactly except: the bound
    /// version is the latest <c>_phase3e</c> entry in the precipitation
    /// manifest's per-station Active list (3e is a precip CHALLENGER, so
    /// it's never the manifest's <c>Current</c> the way 3a is); the replay
    /// parquet read is 3e's; metadata records <c>precip_3e_version</c>.
    /// </summary>
    private async Task<int> RunPhase3sAsync(string[] stations, int[] windows, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        var rngSeed = 42;
        var mcSamples = DryWindow3sPredictor.DefaultMcSamples;

        _log.LogInformation(
            "Phase 3s — stations=[{S}] windows=[{W}] leads=[{L}], MC samples={Mc}",
            string.Join(", ", stations), string.Join(",", windows),
            string.Join(",", leads), mcSamples);

        var precipManifest = _metadata.TryGetManifest("precipitation");
        if (precipManifest?.Stations is null)
        {
            _log.LogError("3s needs the precipitation manifest to resolve 3e champions; not found.");
            return 2;
        }

        int trained = 0, skipped = 0;
        foreach (var stationName in stations)
        {
            var stationSlug = StationSlug.WithEaPrefix(stationName);
            if (!precipManifest.Stations.TryGetValue(stationSlug, out var precipEntry))
            {
                _log.LogWarning("{S}: no precipitation manifest entry; skipping 3s.", stationName);
                continue;
            }
            // 3e is a precip challenger — pick the newest _phase3e version
            // from the station's Active list (Active is champion-first but
            // not sorted, so take the lexicographically-latest, which is
            // chronological for the v{yyyy-MM-dd_HHmmss} naming).
            var precip3eVersion = precipEntry.Active
                .Where(v => v.Contains("phase3e", StringComparison.Ordinal))
                .OrderBy(v => v, StringComparer.Ordinal)
                .LastOrDefault();
            if (string.IsNullOrEmpty(precip3eVersion))
            {
                _log.LogWarning(
                    "{S}: no _phase3e version in the precipitation manifest's Active list; skipping 3s.",
                    stationName);
                continue;
            }

            var replayByLead = new Dictionary<int, Dictionary<DateTime, double>>();
            foreach (var lead in leads)
            {
                replayByLead[lead] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3eVersion, lead);
                _log.LogInformation(
                    "  loaded {N} 3e replay hourly P(wet) for {S} lead {L}h",
                    replayByLead[lead].Count, stationName, lead);
            }

            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{stationSlug}/window_{window}h";
                var now = DateTime.UtcNow;
                var versionDir = ModelArtifact.BuildStationVersionDir(
                    modelsRoot, "dry_window", compositeKey, now, "phase3s");
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{S}', window {W}h → 3s {V} ===",
                    stationName, window, versionName);

                var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
                var testPredictionRows = new List<DryWindowTestPredictionRow>();
                DryWindowClimatology? climatology = null;
                bool anyLeadScored = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {L}h ---", lead);

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

                    var hourly = replayByLead[lead];
                    var rng = new Random(rngSeed);
                    var probs = new List<double>(ds.Test.Count);
                    var labels = new List<bool>(ds.Test.Count);
                    foreach (var row in ds.Test)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, row.TargetDateUtc, s, e);
                        if (q is null) continue;
                        var p = DryWindow3gPredictor.ProbDryWindow(q, window, rng, mcSamples);
                        probs.Add(p);
                        labels.Add(row.Label);
                        testPredictionRows.Add(new DryWindowTestPredictionRow
                        {
                            target_date = row.TargetDateUtc,
                            station = stationSlug,
                            window = window,
                            lead = lead,
                            p_dry_window = p,
                            observed_dry_window = (byte)(row.Label ? 1 : 0),
                        });
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
                        CalibratedBlendTestMae = brier,  // raw == shipped for 3s
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
                    Phase = DryWindow3sPredictor.Phase3s,
                    LocationName = _cfg.Location.Name,
                    DataSource = $"precipitation_replay@{precip3eVersion}",
                    TrainedAtUtc = now,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["mc_samples"] = mcSamples,
                        ["seed"] = rngSeed,
                        [DryWindow3sPredictor.Precip3eVersionKey] = precip3eVersion,
                        ["window_hours"] = window,
                    },
                    TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLead,
                    DeviationsFromBrief = new List<string>
                    {
                        "Phase 3s — parameter-free iid MC over Phase 3e hourly P(wet) marginals under independence. Identical algorithm to 3g; the marginal source is the 3e MLP occurrence blender instead of the 3a LightGBM blender. No model.zip files; predict reads 3e's live prediction parquet at inference time and runs MC.",
                        "Cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) is guaranteed by computing all windows from a SINGLE shared Bernoulli sequence per MC sample (DryWindow3gPredictor.ProbDryWindow).",
                        $"3e champion bound at training time: {precip3eVersion}. Re-run dry-window train --feature-set independence-mc-3e to rebind to a newer 3e champion.",
                        "Occurrence Brier is ≈ 3g by design (3e and 3a marginals tie on the occurrence question); 3s's value is the start-hour curve — 2026-05-15 start-hour bake-off: 6h start-hour Brier −4.5% / Top-1 +2.7pt vs 3g, improvement at every window.",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

                ModelArtifact.PromoteStationVersionAsChallenger(
                    modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: DryWindow3sPredictor.Phase3s);

                var (cf, cs) = await _conformal.FitOneAsync(
                    compositeKey, versionName, DryWindowConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {K}/{V}",
                    cf, cs, compositeKey, versionName);

                _log.LogInformation("Saved 3s artefacts → {Dir}", versionDir);
            }
        }

        _log.LogInformation("Phase 3s training complete. Scored={T} Skipped={S}", trained, skipped);
        await Task.CompletedTask;
        return trained == 0 ? 3 : 0;
    }

    /// <summary>
    /// Phase 3j — Gaussian copula MC over Phase 3a's hourly P(wet) outputs.
    /// Sibling of 3g, structurally identical except per-hour Bernoulli draws
    /// within a sample day are correlated by a fitted 9×9 Σ. Σ is fit per
    /// (station, lead) on the TRAIN slice of observed daytime binary
    /// sequences from 3a's replay parquet — observation history only, no
    /// dependency on 3a's q-values. Predict at test time uses the same
    /// daytime q-vector 3g would use plus the lead's Cholesky factor.
    ///
    /// Why this is a challenger and not a champion replacement: the
    /// 2026-05-13 15-way bake-off found 3j wins at 3h windows by 4.7%
    /// (positive within-day autocorrelation makes long-enough dry blocks
    /// rarer than iid predicts) but loses at 6h by 11.3% (long-run
    /// constraint doesn't fit the train Σ's tail behaviour). Per-window
    /// champion routing is deferred — for now 3j ships alongside 3g and
    /// the site shows both.
    ///
    /// Bundle layout under <c>data/models/dry_window/{station}/window_{N}h/v..._phase3j/</c>:
    /// <code>
    ///   correlation.json         { "ByLead": { "24": { "Sigma": [[..]] }, ... } }
    ///   dry_window_climatology.json
    ///   training_metadata.json
    /// </code>
    /// No model.zip — predict is parameter-free given Σ + 3a's q.
    /// </summary>
    private async Task<int> RunPhase3jAsync(string[] stations, int[] windows, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        var rngSeed = 42;
        var mcSamples = DryWindow3jPredictor.DefaultMcSamples;

        _log.LogInformation(
            "Phase 3j — stations=[{S}] windows=[{W}] leads=[{L}], MC samples={Mc}",
            string.Join(", ", stations), string.Join(",", windows),
            string.Join(",", leads), mcSamples);

        var precipManifest = _metadata.TryGetManifest("precipitation");
        if (precipManifest?.Stations is null)
        {
            _log.LogError("3j needs the precipitation manifest to resolve 3a champions; not found.");
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
                    "{S}: no 3a champion in precipitation manifest; skipping 3j.",
                    stationName);
                continue;
            }
            var precip3aVersion = precipEntry.Current;

            // Pre-load per-lead q (for test MC) AND observed hourly labels
            // (for Σ fit). Both come from the same replay parquet so a
            // missing label hour implies a missing q hour too.
            var replayQByLead = new Dictionary<int, Dictionary<DateTime, double>>();
            var replayLabelByLead = new Dictionary<int, Dictionary<DateTime, byte>>();
            foreach (var lead in leads)
            {
                replayQByLead[lead] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3aVersion, lead);
                replayLabelByLead[lead] = DryWindow3jPredictor.LoadReplayLabelsHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3aVersion, lead);
                _log.LogInformation(
                    "  loaded {Nq} q hours + {Nl} label hours from 3a replay for {S} lead {L}h",
                    replayQByLead[lead].Count, replayLabelByLead[lead].Count, stationName, lead);
            }

            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{stationSlug}/window_{window}h";
                var now = DateTime.UtcNow;
                var versionDir = ModelArtifact.BuildStationVersionDir(
                    modelsRoot, "dry_window", compositeKey, now, "phase3j");
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{S}', window {W}h → 3j {V} ===",
                    stationName, window, versionName);

                var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
                var sigmaByLead = new Dictionary<int, double[,]>();
                var choleskyByLead = new Dictionary<int, double[,]>();
                var testPredictionRows = new List<DryWindowTestPredictionRow>();
                DryWindowClimatology? climatology = null;
                bool anyLeadScored = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {L}h ---", lead);

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

                    // Fit Σ from TRAIN-slice observed daytime binary sequences.
                    // Skip days where any daytime hour's label is missing —
                    // matches DryWindowFeatureBuilder's gap handling.
                    var labelsHourly = replayLabelByLead[lead];
                    var trainSequences = new List<byte[]>(ds.Train.Count);
                    foreach (var row in ds.Train)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var n = e - s;
                        var seq = new byte[n];
                        bool gap = false;
                        var midnight = new DateTime(
                            row.TargetDateUtc.Year, row.TargetDateUtc.Month, row.TargetDateUtc.Day,
                            0, 0, 0, DateTimeKind.Utc);
                        for (int h = s; h < e; h++)
                        {
                            if (!labelsHourly.TryGetValue(midnight.AddHours(h), out var lbl))
                            {
                                gap = true; break;
                            }
                            seq[h - s] = lbl;
                        }
                        if (!gap) trainSequences.Add(seq);
                    }
                    if (trainSequences.Count < 50)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: only {N} train sequences after label-gap filter; skipping (need ≥50 for stable Σ).",
                            stationName, window, lead, trainSequences.Count);
                        skipped++;
                        continue;
                    }

                    var sigma = DryWindow3jPredictor.FitCorrelation(trainSequences.ToArray());
                    double[,] L;
                    try
                    {
                        L = DryWindow3jPredictor.CholeskyDecompose(sigma);
                    }
                    catch (InvalidOperationException ex)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: Cholesky failed ({Msg}); skipping lead.",
                            stationName, window, lead, ex.Message);
                        skipped++;
                        continue;
                    }
                    sigmaByLead[lead] = sigma;
                    choleskyByLead[lead] = L;

                    _log.LogInformation(
                        "  fitted Σ from {N} train sequences (mean off-diag corr ≈ {Avg:0.000})",
                        trainSequences.Count, MeanOffDiagonal(sigma));

                    var hourlyQ = replayQByLead[lead];
                    var rng = new Random(rngSeed);
                    var probs = new List<double>(ds.Test.Count);
                    var labels = new List<bool>(ds.Test.Count);
                    foreach (var row in ds.Test)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourlyQ, row.TargetDateUtc, s, e);
                        if (q is null) continue;
                        var p = DryWindow3jPredictor.ProbDryWindow(q, L, window, rng, mcSamples);
                        probs.Add(p);
                        labels.Add(row.Label);
                        testPredictionRows.Add(new DryWindowTestPredictionRow
                        {
                            target_date = row.TargetDateUtc,
                            station = stationSlug,
                            window = window,
                            lead = lead,
                            p_dry_window = p,
                            observed_dry_window = (byte)(row.Label ? 1 : 0),
                        });
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
                        CalibratedBlendTestMae = brier,
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
                DryWindow3jPredictor.WriteCorrelationJson(versionDir, sigmaByLead);

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
                    Phase = DryWindow3jPredictor.Phase3j,
                    LocationName = _cfg.Location.Name,
                    DataSource = $"precipitation_replay@{precip3aVersion}",
                    TrainedAtUtc = now,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["mc_samples"] = mcSamples,
                        ["seed"] = rngSeed,
                        ["precip_3a_version"] = precip3aVersion,
                        ["window_hours"] = window,
                        ["correlation_dim"] = sigmaByLead.Values.First().GetLength(0),
                    },
                    TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLead,
                    DeviationsFromBrief = new List<string>
                    {
                        "Phase 3j — Gaussian copula MC over Phase 3a hourly P(wet) marginals. Per-hour draws within a sample day correlated by a 9×9 Σ fit on observed daytime wet/dry binary sequences from the train slice.",
                        "Cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) preserved via single-pass-multiple-windows (DryWindow3jPredictor.ProbDryWindow).",
                        $"3a champion bound at training time: {precip3aVersion}. Σ fit on the train slice of that 3a's replay parquet (observation column, not the model's q).",
                        "Per-window scope: today's bake-off (2026-05-13) shows 3j wins 3h windows (+4.7% vs 3g) but loses 6h (-11.3%). Shipping as challenger alongside 3g; per-window-champion routing TBD.",
                        $"MC samples = {mcSamples} (2× 3g's default — copula path is slightly noisier per sample due to Φ evaluation + matrix-vector product).",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

                ModelArtifact.PromoteStationVersionAsChallenger(
                    modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: DryWindow3jPredictor.Phase3j);

                var (cf, cs) = await _conformal.FitOneAsync(
                    compositeKey, versionName, DryWindowConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {K}/{V}",
                    cf, cs, compositeKey, versionName);

                _log.LogInformation("Saved 3j artefacts → {Dir}", versionDir);
            }
        }

        _log.LogInformation("Phase 3j training complete. Scored={T} Skipped={S}", trained, skipped);
        await Task.CompletedTask;
        return trained == 0 ? 3 : 0;
    }

    /// <summary>
    /// Phase 3n — regime-conditioned copula MC. Same structure as 3j but
    /// fits TWO Σs per (station, lead): one on train days where NWPs agree
    /// strongly on the hourly wet/dry pattern (settled atmospheric regime),
    /// one on days where they disagree (unsettled/transitional). The agreement
    /// threshold is the median of the train slice's agreement scores —
    /// guarantees roughly equal data per Σ.
    ///
    /// At test/predict time the day's agreement (recomputed from whichever
    /// forecast tree applies) routes the day through Σ_settled or
    /// Σ_unsettled before the copula MC runs.
    /// </summary>
    private async Task<int> RunPhase3nAsync(string[] stations, int[] windows, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        var rngSeed = 42;
        var mcSamples = DryWindow3nPredictor.DefaultMcSamples;
        var canonicalModels = WeatherBlend.Train.TempFeatureBuilder.CanonicalModelOrder;

        _log.LogInformation(
            "Phase 3n — stations=[{S}] windows=[{W}] leads=[{L}], MC samples={Mc}",
            string.Join(", ", stations), string.Join(",", windows),
            string.Join(",", leads), mcSamples);
        _log.LogInformation(
            "Regime axis: per-day NWP consensus on hourly wet/dry (canonical {N}-model set).",
            canonicalModels.Count);

        var precipManifest = _metadata.TryGetManifest("precipitation");
        if (precipManifest?.Stations is null)
        {
            _log.LogError("3n needs the precipitation manifest to resolve 3a champions; not found.");
            return 2;
        }

        // Agreement-by-target-date is identical across stations (it's a
        // forecast-tree property of the location). Compute once per lead.
        (int Start, int EndExclusive) DaytimeFor(DateOnly d) => daytime.UtcHourRangeFor(d);
        var agreementByLead = new Dictionary<int, Dictionary<DateTime, double>>();
        foreach (var lead in leads)
        {
            var matrices = DryWindowNwpAgreement.LoadOffsetDayPerNwpDaytime(
                _cfg.Storage.ForecastsPath, _cfg.Location.Name, lead,
                canonicalModels, DaytimeFor);
            var byDate = new Dictionary<DateTime, double>(matrices.Count);
            foreach (var (date, mat) in matrices)
            {
                var a = DryWindowNwpAgreement.ComputePerDay(mat);
                if (!double.IsNaN(a)) byDate[date] = a;
            }
            agreementByLead[lead] = byDate;
            _log.LogInformation(
                "  computed agreement for {N} training days at lead {L}h", byDate.Count, lead);
        }

        int trained = 0, skipped = 0;
        foreach (var stationName in stations)
        {
            var stationSlug = StationSlug.WithEaPrefix(stationName);
            if (!precipManifest.Stations.TryGetValue(stationSlug, out var precipEntry)
                || string.IsNullOrEmpty(precipEntry.Current))
            {
                _log.LogWarning(
                    "{S}: no 3a champion in precipitation manifest; skipping 3n.",
                    stationName);
                continue;
            }
            var precip3aVersion = precipEntry.Current;

            // Same per-lead Q/label load as 3j — same replay parquets.
            var replayQByLead = new Dictionary<int, Dictionary<DateTime, double>>();
            var replayLabelByLead = new Dictionary<int, Dictionary<DateTime, byte>>();
            foreach (var lead in leads)
            {
                replayQByLead[lead] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3aVersion, lead);
                replayLabelByLead[lead] = DryWindow3jPredictor.LoadReplayLabelsHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3aVersion, lead);
                _log.LogInformation(
                    "  loaded {Nq} q hours + {Nl} label hours from 3a replay for {S} lead {L}h",
                    replayQByLead[lead].Count, replayLabelByLead[lead].Count, stationName, lead);
            }

            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{stationSlug}/window_{window}h";
                var now = DateTime.UtcNow;
                var versionDir = ModelArtifact.BuildStationVersionDir(
                    modelsRoot, "dry_window", compositeKey, now, "phase3n");
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{S}', window {W}h → 3n {V} ===",
                    stationName, window, versionName);

                var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
                var bundleByLead = new Dictionary<int, (double[,] SigmaSettled, double[,] SigmaUnsettled, double Threshold, int DaysSettled, int DaysUnsettled)>();
                var testPredictionRows = new List<DryWindowTestPredictionRow>();
                DryWindowClimatology? climatology = null;
                bool anyLeadScored = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {L}h ---", lead);

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

                    var labelsHourly = replayLabelByLead[lead];
                    var agreement = agreementByLead[lead];

                    // Per train day: build observed daytime binary sequence,
                    // attach the day's agreement. Skip days missing either.
                    var trainEntries = new List<(byte[] Seq, double Agree)>(ds.Train.Count);
                    foreach (var row in ds.Train)
                    {
                        if (!agreement.TryGetValue(row.TargetDateUtc, out var agr)) continue;
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var n = e - s;
                        var seq = new byte[n];
                        bool gap = false;
                        var midnight = new DateTime(
                            row.TargetDateUtc.Year, row.TargetDateUtc.Month, row.TargetDateUtc.Day,
                            0, 0, 0, DateTimeKind.Utc);
                        for (int h = s; h < e; h++)
                        {
                            if (!labelsHourly.TryGetValue(midnight.AddHours(h), out var lbl))
                            {
                                gap = true; break;
                            }
                            seq[h - s] = lbl;
                        }
                        if (!gap) trainEntries.Add((seq, agr));
                    }
                    if (trainEntries.Count < 100)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: only {N} train days after label-gap + agreement filter; skipping (need ≥100 for two stable Σs).",
                            stationName, window, lead, trainEntries.Count);
                        skipped++;
                        continue;
                    }

                    var trainAgrees = trainEntries.Select(t => t.Agree).OrderBy(x => x).ToArray();
                    var threshold = trainAgrees[trainAgrees.Length / 2];

                    var settledSeqs   = trainEntries.Where(t => t.Agree >= threshold).Select(t => t.Seq).ToArray();
                    var unsettledSeqs = trainEntries.Where(t => t.Agree <  threshold).Select(t => t.Seq).ToArray();
                    if (settledSeqs.Length < 30 || unsettledSeqs.Length < 30)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: regime buckets too thin (settled={Ns}, unsettled={Nu}); skipping.",
                            stationName, window, lead, settledSeqs.Length, unsettledSeqs.Length);
                        skipped++;
                        continue;
                    }

                    var sigmaSettled   = DryWindow3jPredictor.FitCorrelation(settledSeqs);
                    var sigmaUnsettled = DryWindow3jPredictor.FitCorrelation(unsettledSeqs);
                    double[,] cholSettled, cholUnsettled;
                    try
                    {
                        cholSettled   = DryWindow3jPredictor.CholeskyDecompose(sigmaSettled);
                        cholUnsettled = DryWindow3jPredictor.CholeskyDecompose(sigmaUnsettled);
                    }
                    catch (InvalidOperationException ex)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: Cholesky failed ({Msg}); skipping lead.",
                            stationName, window, lead, ex.Message);
                        skipped++;
                        continue;
                    }
                    bundleByLead[lead] = (sigmaSettled, sigmaUnsettled, threshold, settledSeqs.Length, unsettledSeqs.Length);

                    _log.LogInformation(
                        "  fitted Σ — settled (n={Ns}, mean off-diag {Ds:0.000}) | unsettled (n={Nu}, mean off-diag {Du:0.000}) | threshold={T:0.000}",
                        settledSeqs.Length, MeanOffDiagonal(sigmaSettled),
                        unsettledSeqs.Length, MeanOffDiagonal(sigmaUnsettled), threshold);

                    var hourlyQ = replayQByLead[lead];
                    var rng = new Random(rngSeed);
                    var probs = new List<double>(ds.Test.Count);
                    var labels = new List<bool>(ds.Test.Count);
                    int testSettled = 0, testUnsettled = 0;
                    foreach (var row in ds.Test)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourlyQ, row.TargetDateUtc, s, e);
                        if (q is null) continue;
                        if (!agreement.TryGetValue(row.TargetDateUtc, out var agr)) continue;
                        var L = agr >= threshold ? cholSettled : cholUnsettled;
                        if (agr >= threshold) testSettled++; else testUnsettled++;
                        var p = DryWindow3jPredictor.ProbDryWindow(q, L, window, rng, mcSamples);
                        probs.Add(p);
                        labels.Add(row.Label);
                        testPredictionRows.Add(new DryWindowTestPredictionRow
                        {
                            target_date = row.TargetDateUtc,
                            station = stationSlug,
                            window = window,
                            lead = lead,
                            p_dry_window = p,
                            observed_dry_window = (byte)(row.Label ? 1 : 0),
                        });
                    }

                    if (probs.Count < 10)
                    {
                        _log.LogWarning(
                            "  {S} {W}h lead {L}h: only {N} test rows after filters; skipping.",
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
                        CalibratedBlendTestMae = brier,
                    };
                    anyLeadScored = true;
                    trained++;
                    _log.LogInformation(
                        "  Lead {L}h Brier={B:0.0000} (clim {C:0.0000}, n_test={N}; routed {Ts} settled / {Tu} unsettled)",
                        lead, brier, climBrier, probs.Count, testSettled, testUnsettled);
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
                DryWindow3nPredictor.WriteCorrelationJson(versionDir, bundleByLead);

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
                    Phase = DryWindow3nPredictor.Phase3n,
                    LocationName = _cfg.Location.Name,
                    DataSource = $"precipitation_replay@{precip3aVersion}",
                    TrainedAtUtc = now,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["mc_samples"] = mcSamples,
                        ["seed"] = rngSeed,
                        ["precip_3a_version"] = precip3aVersion,
                        ["window_hours"] = window,
                        ["regime_axis"] = "nwp_per_hour_wet_dry_consensus",
                        ["wet_threshold_mm_h"] = DryWindowNwpAgreement.WetThresholdMmH,
                        ["min_models_per_hour"] = DryWindowNwpAgreement.MinModelsPerHour,
                    },
                    TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLead,
                    DeviationsFromBrief = new List<string>
                    {
                        "Phase 3n — regime-conditioned Gaussian copula MC. Two Σs per (station, lead) split by NWP-consensus on per-hour wet/dry. Threshold is the median of the train slice's agreement scores.",
                        $"Same MC mechanics as 3j; 3n only differs in Σ selection. Bound 3a champion: {precip3aVersion}.",
                        "Predict-time uses the SAME agreement formula on the live forecast tree (RunTimeSource='reported') so the regime label of a day stays consistent across train and predict.",
                        "Pre-flight diagnostic 2026-05-13 on Bellever lead 24: Frobenius norm of Σ_settled-Σ_unsettled = 4.04, mean off-diag 0.80 vs 0.35 — strongly suggests the regime axis is informative.",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

                ModelArtifact.PromoteStationVersionAsChallenger(
                    modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: DryWindow3nPredictor.Phase3n);

                var (cf, cs) = await _conformal.FitOneAsync(
                    compositeKey, versionName, DryWindowConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {K}/{V}",
                    cf, cs, compositeKey, versionName);

                _log.LogInformation("Saved 3n artefacts → {Dir}", versionDir);
            }
        }

        _log.LogInformation("Phase 3n training complete. Scored={T} Skipped={S}", trained, skipped);
        await Task.CompletedTask;
        return trained == 0 ? 3 : 0;
    }

    private static double MeanOffDiagonal(double[,] m)
    {
        int n = m.GetLength(0);
        if (n <= 1) return 0.0;
        double sum = 0; int count = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j) { sum += m[i, j]; count++; }
        return count == 0 ? 0.0 : sum / count;
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

                // 3f is local-only — bundles persist on disk for bake-off
                // discovery but do NOT register in the manifest's Active list.
                // No DryWindowPhases entry, no predict path in C#: registering
                // 3f as a challenger would cause DryWindowPredictCommand to
                // iterate it, hit LoadLeadModel (LightGBM .zip) on a path
                // that only has .pt MLP weights, and crash. The original
                // PromoteStationVersionAsChallenger call here was a 2026-05-13
                // mistake — 3f was never supposed to be a "phase on the site"
                // (see project_dry_window_bakeoff_2026-05-13.md). Bake-off
                // scripts glob the bundle dir directly so they still see 3f.
                _log.LogInformation("Saved 3f MLP artefacts → {Dir} (bake-off only — NOT registered in manifest)", versionDir);
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
