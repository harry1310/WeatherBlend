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
            && phase != DryWindowFeatureBuilder.Phase3dShape
            && phase != DryWindow3eFeatureBuilder.Phase3e
            && phase != DryWindow3gPredictor.Phase3g)
        {
            _log.LogError("Unsupported dry-window training phase '{Phase}'. Expected '3b', '3d-shape', '3e', or '3g'.", phase);
            return 2;
        }

        var stations = ResolveStations(stationArg);
        if (stations is null) return 2;

        // Phase 3e is a station-level cascade — windowArg is ignored because
        // 3e ALWAYS produces both 3h and 4h artefacts in one training run
        // (that's the whole point of the conditional decomposition). Dispatch
        // to the dedicated 3e codepath; the existing per-window loop below
        // doesn't apply.
        if (phase == DryWindow3eFeatureBuilder.Phase3e)
        {
            if (!string.Equals(windowArg, "all", StringComparison.OrdinalIgnoreCase))
                _log.LogInformation(
                    "Phase 3e ignores --window '{W}': cascade always trains for windows 3 + 4 jointly.",
                    windowArg);
            return await RunPhase3eAsync(stations, leads, ct);
        }

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
                // 3b → v{ts} (champion, no suffix). Challenger phases get a
                // _phaseXX suffix so MANIFEST.Active visually distinguishes them.
                var suffix = phase switch
                {
                    DryWindowFeatureBuilder.Phase3dShape => "phase3d_shape",
                    _ => null,
                };
                var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "dry_window", compositeKey, now, suffix);
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
                if (phase == DryWindowFeatureBuilder.Phase3b)
                {
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
                }
                else
                {
                    // 3d-shape can still be trained ad-hoc via --feature-set rich for
                    // experimentation, but it does not enter the Active rotation — predict
                    // and the site filter for 3b only. Append the artefact so its version dir
                    // is reachable manually if needed, but leave Active unchanged.
                    ModelArtifact.AppendStationVersion(modelsRoot, "dry_window", compositeKey, versionName);
                    _log.LogInformation(
                        "{Key}: 3d-shape artefact saved but NOT added to Active list — predict/render skip it by design.",
                        compositeKey);
                }

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
    /// Phase 3e training loop. For each station, builds one multi-label
    /// dataset per lead (rows carry both 3h and 4h labels), trains M_base
    /// against the 3h label and M_extend4 against the 4h label on the
    /// subset where 3h holds, then writes TWO artefact dirs sharing the
    /// same timestamp under <c>window_3h/v{ts}_phase3e/</c> and
    /// <c>window_4h/v{ts}_phase3e/</c>. Each is registered as a challenger
    /// to 3b in the existing per-(station, window) manifest's Active list,
    /// so verify scores both phases independently and the bake-off reads
    /// from the existing verify-history sidecars.
    /// </summary>
    private async Task<int> RunPhase3eAsync(string[] stations, int[] leads, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = new DryWindowTrainer.Hyperparameters();
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();

        _log.LogInformation(
            "Phase 3e — stations=[{S}] leads=[{L}] (cascade trains windows 3+4 jointly)",
            string.Join(", ", stations), string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={I} lr={Lr} leaves={Lv} esr={E} seed={Sd}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves,
            hp.EarlyStoppingRound, hp.Seed);

        int trained = 0, skipped = 0;

        foreach (var stationName in stations)
        {
            ct.ThrowIfCancellationRequested();
            var slug = StationSlug.WithEaPrefix(stationName);
            var compositeKey3h = $"{slug}/window_3h";
            var compositeKey4h = $"{slug}/window_4h";

            // ONE shared timestamp per training run so the (3h, 4h) artefacts
            // form a coherent pair the user can match by name.
            var now = DateTime.UtcNow;
            var versionDir3h = ModelArtifact.BuildStationVersionDir(
                modelsRoot, "dry_window", compositeKey3h, now,
                DryWindow3eCascadeArtefact.VersionSuffix);
            var versionDir4h = ModelArtifact.BuildStationVersionDir(
                modelsRoot, "dry_window", compositeKey4h, now,
                DryWindow3eCascadeArtefact.VersionSuffix);
            var versionName = Path.GetFileName(versionDir3h);   // identical for both

            _log.LogInformation("=== Station '{Station}' → 3e cascade {V} ===", stationName, versionName);

            var perLead3h = new Dictionary<string, ModelArtifact.PerLeadStats>();
            var perLead4h = new Dictionary<string, ModelArtifact.PerLeadStats>();
            var importance3h = new Dictionary<int, IEnumerable<(string, double)>>();
            var importance4h = new Dictionary<int, IEnumerable<(string, double)>>();
            var specsPerLead = new Dictionary<int, BlenderSpec>();
            DryWindowClimatology? clim3h = null;
            DryWindowClimatology? clim4h = null;
            bool anyLeadTrained = false;

            foreach (var lead in leads)
            {
                ct.ThrowIfCancellationRequested();
                _log.LogInformation("--- Lead {L}h ---", lead);

                // Spec is identical to 3b's — same 53 features, same model membership.
                // 3e is a different way to USE the same features, not a feature change.
                var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
                specsPerLead[lead] = spec;

                var multiRows = DryWindow3eFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    _cfg.Location.Name, stationName, spec, daytime, ct);
                _log.LogInformation(
                    "  loaded {N} multi-label rows (label_3h positives={P3}, label_4h positives={P4})",
                    multiRows.Count,
                    multiRows.Count(r => r.Label3h),
                    multiRows.Count(r => r.Label4h));

                if (multiRows.Count < 100)
                {
                    _log.LogWarning("  too few rows ({N}) for ({Station}, lead {L}h); skipping.",
                        multiRows.Count, stationName, lead);
                    skipped++;
                    continue;
                }

                // Project to vanilla CommonRow per stage. M_base trains on the
                // FULL set with the 3h label. M_extend4 trains on the SUBSET
                // where Label3h=true, with the 4h label.
                var baseRows = multiRows
                    .Select(r => DryWindow3eFeatureBuilder.ToCommonRow(r, r.Label3h, outputWindowHours: 3))
                    .ToList();
                var extendRows = multiRows
                    .Where(r => r.Label3h)
                    .Select(r => DryWindow3eFeatureBuilder.ToCommonRow(r, r.Label4h, outputWindowHours: 4))
                    .ToList();

                if (extendRows.Count < 30)
                {
                    _log.LogWarning(
                        "  M_extend4 subset ({N} rows where 3h-block exists) too small for ({Station}, lead {L}h); skipping.",
                        extendRows.Count, stationName, lead);
                    skipped++;
                    continue;
                }

                // Same chronological split rule as 3b — date-ordered, no shuffle.
                var dsBase = DryWindowDataset.Split(baseRows);
                var dsExt  = DryWindowDataset.Split(extendRows);
                clim3h ??= BuildClimatologyFromVectorRows(dsBase.Train, 3);
                clim4h ??= BuildClimatologyFromVectorRows(dsExt.Train, 4);

                _log.LogInformation(
                    "  M_base: train {Tn} (pos {Tp}), val {Vn} (pos {Vp}), test {En} (pos {Ep})",
                    dsBase.Train.Count, dsBase.TrainPositives,
                    dsBase.Val.Count,   dsBase.ValPositives,
                    dsBase.Test.Count,  dsBase.TestPositives);
                _log.LogInformation(
                    "  M_extend4: train {Tn} (pos {Tp}), val {Vn} (pos {Vp}), test {En} (pos {Ep})",
                    dsExt.Train.Count, dsExt.TrainPositives,
                    dsExt.Val.Count,   dsExt.ValPositives,
                    dsExt.Test.Count,  dsExt.TestPositives);

                // Stage 1: M_base.
                var trainedBase = DryWindowTrainer.TrainVector(dsBase.Train, dsBase.Val, spec, hp);
                // Stage 2: M_extend4 — same trainer, different (subset) data + label.
                var trainedExt  = DryWindowTrainer.TrainVector(dsExt.Train,  dsExt.Val,  spec, hp);

                // Per-output-window scoring on each stage's own test slice.
                // window_3h artefact emits raw M_base; window_4h emits the product.
                var truthBase = dsBase.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                var probBase  = DryWindowTrainer.PredictVectorProbability(trainedBase.Ml, trainedBase.Model, spec, dsBase.Test);
                var brierBase = PrecipMetrics.Brier(probBase, truthBase);

                // For 4h scoring we need the cascade product on the FULL multi-label
                // test slice (rows where Label3h might be false too — those rows
                // also have a 4h label, definitionally false). Build the test set
                // from multi-label rows, restricted to the same date range as
                // dsBase.Test to keep the chronological split consistent.
                var dsBaseTestStart = dsBase.Test[0].TargetDateUtc;
                var dsBaseTestEnd   = dsBase.Test[^1].TargetDateUtc;
                var fullTestMulti = multiRows
                    .Where(r => r.TargetDateUtc >= dsBaseTestStart && r.TargetDateUtc <= dsBaseTestEnd)
                    .ToList();
                var fullTestRows = fullTestMulti
                    .Select(r => DryWindow3eFeatureBuilder.ToCommonRow(r, r.Label3h, 3))
                    .ToList();
                var truth4h = fullTestMulti.Select(r => r.Label4h ? 1.0 : 0.0).ToArray();
                var prod4h  = DryWindow3eCascadeArtefact.PredictRawProductForExtend(
                    trainedBase.Ml, trainedBase.Model, trainedExt.Model, spec, fullTestRows);
                var brier4hRaw = PrecipMetrics.Brier(prod4h, truth4h);

                // PAV-on-product against 4h truth on the val slice — fits the
                // calibrator to the SHIPPED quantity rather than to the
                // intermediate factors. Validation rows analogous to fullTestMulti.
                var dsBaseValStart = dsBase.Val[0].TargetDateUtc;
                var dsBaseValEnd   = dsBase.Val[^1].TargetDateUtc;
                var fullValMulti = multiRows
                    .Where(r => r.TargetDateUtc >= dsBaseValStart && r.TargetDateUtc <= dsBaseValEnd)
                    .ToList();
                var fullValRows = fullValMulti
                    .Select(r => DryWindow3eFeatureBuilder.ToCommonRow(r, r.Label3h, 3))
                    .ToList();
                var prod4hVal = DryWindow3eCascadeArtefact.PredictRawProductForExtend(
                    trainedBase.Ml, trainedBase.Model, trainedExt.Model, spec, fullValRows);
                var truth4hVal = fullValMulti.Select(r => r.Label4h).ToArray();
                var calibrator4h = IsotonicCalibrator.Fit(prod4hVal, truth4hVal);
                var prod4hCal = calibrator4h.PredictMany(prod4h);
                var brier4hCal = PrecipMetrics.Brier(prod4hCal, truth4h);

                // Shipping calibration: Bellever PAV-cals the 3h-direct output too,
                // mirroring 3b's per-station policy. Other stations ship raw.
                var calibrationEnabled = _cfg.DryWindow.ShouldCalibrate(stationName);

                // Persist M_base under both window_3h and window_4h dirs (4h needs
                // it for the cascade). Save M_extend4 only under window_4h. Save
                // calibrators per artefact.
                ModelArtifact.SaveLeadModel(trainedBase.Ml, trainedBase.Model, trainedBase.InputSchema, versionDir3h, lead);
                ModelArtifact.SaveLeadModel(trainedBase.Ml, trainedBase.Model, trainedBase.InputSchema, versionDir4h, lead);
                // M_extend4 — distinct filename so window_4h's dir contains both.
                var extendPath = Path.Combine(versionDir4h, DryWindow3eCascadeArtefact.ExtendModelFileName(lead));
                Directory.CreateDirectory(versionDir4h);
                trainedExt.Ml.Model.Save(trainedExt.Model, trainedExt.InputSchema, extendPath);

                if (calibrationEnabled)
                {
                    // 3h artefact: PAV M_base raw against 3h truth — same as 3b.
                    ModelArtifact.SaveLeadCalibrator(trainedBase.Calibrator, versionDir3h, lead);
                    // 4h artefact: PAV the PRODUCT against 4h truth (fitted above).
                    ModelArtifact.SaveLeadCalibrator(calibrator4h, versionDir4h, lead);
                }

                importance3h[lead] = trainedBase.FeatureImportance;
                importance4h[lead] = trainedExt.FeatureImportance;

                var brierBaseShipped = calibrationEnabled
                    ? PrecipMetrics.Brier(trainedBase.Calibrator.PredictMany(probBase), truthBase)
                    : brierBase;
                var brier4hShipped = calibrationEnabled ? brier4hCal : brier4hRaw;

                var climPredBase = DryWindowBaselines.Climatology(dsBase.Train, dsBase.Test, 3);
                var climBrierBase = PrecipMetrics.Brier(climPredBase, truthBase);
                var clim4hPred = new double[truth4h.Length];
                Array.Fill(clim4hPred, clim4h?.GlobalPositiveRate ?? 0.0);
                var climBrier4h = PrecipMetrics.Brier(clim4hPred, truth4h);

                var monthsBase = dsBase.Test
                    .Select(r => new DateTime(r.TargetDateUtc.Year, r.TargetDateUtc.Month, 1))
                    .Distinct().Count();
                var months4h = fullTestMulti
                    .Select(r => new DateTime(r.TargetDateUtc.Year, r.TargetDateUtc.Month, 1))
                    .Distinct().Count();

                perLead3h[lead.ToString()] = new ModelArtifact.PerLeadStats
                {
                    LeadHours = lead,
                    DataRangeTrain = $"{dsBase.TrainStart:yyyy-MM-dd}Z → {dsBase.TrainEnd:yyyy-MM-dd}Z",
                    DataRangeVal   = $"{dsBase.ValStart:yyyy-MM-dd}Z → {dsBase.ValEnd:yyyy-MM-dd}Z",
                    DataRangeTest  = $"{dsBase.TestStart:yyyy-MM-dd}Z → {dsBase.TestEnd:yyyy-MM-dd}Z",
                    TrainRows = dsBase.Train.Count,
                    ValRows   = dsBase.Val.Count,
                    TestRows  = dsBase.Test.Count,
                    TestCalendarMonths = monthsBase,
                    BlendTestMae = brierBaseShipped,
                    BlendTestRmse = climBrierBase,
                    BlendTestBias = 0.0,
                    CalibratedBlendTestMae = calibrationEnabled
                        ? brierBaseShipped
                        : PrecipMetrics.Brier(trainedBase.Calibrator.PredictMany(probBase), truthBase),
                };

                perLead4h[lead.ToString()] = new ModelArtifact.PerLeadStats
                {
                    LeadHours = lead,
                    DataRangeTrain = $"{dsBase.TrainStart:yyyy-MM-dd}Z → {dsBase.TrainEnd:yyyy-MM-dd}Z",
                    DataRangeVal   = $"{dsBase.ValStart:yyyy-MM-dd}Z → {dsBase.ValEnd:yyyy-MM-dd}Z",
                    DataRangeTest  = $"{dsBase.TestStart:yyyy-MM-dd}Z → {dsBase.TestEnd:yyyy-MM-dd}Z",
                    TrainRows = dsBase.Train.Count,
                    ValRows   = dsBase.Val.Count,
                    TestRows  = fullTestMulti.Count,
                    TestCalendarMonths = months4h,
                    BlendTestMae = brier4hShipped,
                    BlendTestRmse = climBrier4h,
                    BlendTestBias = 0.0,
                    CalibratedBlendTestMae = brier4hCal,
                };

                _log.LogInformation(
                    "  Lead {L}h Brier — 3h={B3:0.0000} (clim {C3:0.0000}), 4h_raw={B4r:0.0000} cal={B4c:0.0000} (clim {C4:0.0000})",
                    lead, brierBaseShipped, climBrierBase, brier4hRaw, brier4hCal, climBrier4h);
                anyLeadTrained = true;
                trained++;
            }

            if (!anyLeadTrained)
            {
                _log.LogWarning("  No leads trained for {Station} — skipping artefact save.", stationName);
                continue;
            }

            // Save spec, importance, climatology, metadata for BOTH artefact dirs.
            // 3e shares the same spec across both windows (same features) so the
            // feature_schema.json is identical, but each dir gets its own copy
            // so each is self-contained and the renderer reads only one path.
            ModelArtifact.SaveBlenderSpecs(versionDir3h, specsPerLead);
            ModelArtifact.SaveBlenderSpecs(versionDir4h, specsPerLead);
            ModelArtifact.SavePerLeadFeatureImportance(versionDir3h, importance3h);
            ModelArtifact.SavePerLeadFeatureImportance(versionDir4h, importance4h);
            clim3h?.SaveTo(Path.Combine(versionDir3h, "dry_window_climatology.json"));
            clim4h?.SaveTo(Path.Combine(versionDir4h, "dry_window_climatology.json"));

            var metadata3h = new ModelArtifact.TrainingMetadata
            {
                Version = versionName,
                Target = "dry_window",
                Phase = DryWindow3eFeatureBuilder.Phase3e,
                DataSource = "previous_runs_api+ea_rainfall",
                TrainedAtUtc = now,
                Hyperparameters = BuildHpDict(hp, 3),
                TestMae = perLead3h.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                PerLead = perLead3h,
                DeviationsFromBrief = new List<string>
                {
                    "Phase 3e cascade — window_3h artefact emits M_base raw (P(3h)).",
                    "PerLeadStats fields reused (3b convention): BlendTestMae=blend Brier, BlendTestRmse=climatology Brier.",
                },
            };
            var metadata4h = new ModelArtifact.TrainingMetadata
            {
                Version = versionName,
                Target = "dry_window",
                Phase = DryWindow3eFeatureBuilder.Phase3e,
                DataSource = "previous_runs_api+ea_rainfall",
                TrainedAtUtc = now,
                Hyperparameters = BuildHpDict(hp, 4),
                TestMae = perLead4h.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                PerLead = perLead4h,
                DeviationsFromBrief = new List<string>
                {
                    "Phase 3e cascade — window_4h artefact emits the conditional product M_base × M_extend4.",
                    "M_base copy stored alongside M_extend4 in this dir for self-containment; both bytes-identical to the M_base in the sibling window_3h artefact.",
                    "PAV calibrator (when station opts in) is fitted to the PRODUCT against 4h truth, not to the intermediate factors.",
                    "Monotonicity P(4h) ≤ P(3h) holds by construction (product of two probabilities both ≤ 1).",
                },
            };
            ModelArtifact.SaveTrainingMetadata(versionDir3h, metadata3h);
            ModelArtifact.SaveTrainingMetadata(versionDir4h, metadata4h);

            // Promote 3e as a CHALLENGER alongside the 3b champion: replaces
            // any prior 3e entry in Active with this one (idempotent on
            // re-train) and leaves Current = the 3b champion untouched.
            ModelArtifact.PromoteStationVersionAsChallenger(
                modelsRoot, "dry_window", compositeKey3h, versionName,
                newPhase: DryWindow3eFeatureBuilder.Phase3e);
            ModelArtifact.PromoteStationVersionAsChallenger(
                modelsRoot, "dry_window", compositeKey4h, versionName,
                newPhase: DryWindow3eFeatureBuilder.Phase3e);

            // Auto-conformal for both legs of the cascade pair.
            foreach (var key in new[] { compositeKey3h, compositeKey4h })
            {
                var (cf, cs) = await _conformal.FitOneAsync(
                    key, versionName, DryWindowConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {K}/{V}",
                    cf, cs, key, versionName);
            }

            _log.LogInformation("Saved 3e artefacts → {D3h} + {D4h}", versionDir3h, versionDir4h);
        }

        _log.LogInformation("Phase 3e training complete. Trained={T} Skipped={S}", trained, skipped);
        return trained == 0 ? 3 : 0;
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
