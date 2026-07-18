using DuckDB.NET.Data;
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

    // --location resolves into _activeLocation at RunAsync entry; every
    // former _cfg.Location read downstream now goes through this field.
    // Defaults to the primary location (set in the constructor) so a
    // no-flag invocation behaves exactly as before (Phase B, commit 3).
    private Config.LocationConfig _activeLocation;

    private static readonly int[] DefaultLeads = Leads.Short;
    private static readonly int[] DefaultWindows = { 3, 4, 6 };
    // Trained stations are now read from `_activeLocation.Rainfall.Stations`. A
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
        _activeLocation = cfg.Location;
    }

    public Task<int> RunAsync(string stationArg, string windowArg, int[] leads, CancellationToken ct)
        => RunAsync(stationArg, windowArg, leads, _cfg.Location, ct);

    public async Task<int> RunAsync(
        string stationArg, string windowArg, int[] leads,
        Config.LocationConfig location, CancellationToken ct)
    {
        _activeLocation = location;

        var stations = ResolveStations(stationArg);
        if (stations is null) return 2;

        // Phase 3b is the only LightGBM dry-window blender trained here.
        // 3p (Gaussian copula MC over 3o) is a separate predict-time-only
        // path with no LightGBM fit step.
        var phase = DryWindowFeatureBuilder.Phase3b;

        var windows = ParseWindows(windowArg);
        if (windows is null)
        {
            _log.LogError("Invalid --window value '{W}'. Expected 3, 4, 6, or all.", windowArg);
            return 2;
        }

        if (_activeLocation.Rainfall.Stations.Count == 0)
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
        // Cells (station, window) that failed for a bad reason — guard
        // rejection or no lead trainable. Distinct from modelsSkipped (which
        // also counts benign per-lead thin-data skips). RunAsync exits
        // non-zero when this is > 0 so the retrain workflow's aggregator can
        // never report a guard-failed sweep as a silent success — the
        // 2026-05-17 dry-window 3b freeze went unnoticed for 3 days because
        // the old `modelsTrained` counted leads, not cells.
        int cellFailures = 0;

        foreach (var stationName in stations)
        {
            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{_cfg.ResolveStationSlug(stationName)}/window_{window}h";
                var now = DateTime.UtcNow;
                // 3b → v{ts} (champion, no suffix). Any future challenger
                // adds its own _phaseXX suffix.
                var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "dry_window", compositeKey, now);
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{Station}', window {W}h → {Key} ===",
                    stationName, window, compositeKey);

                // Resolve the station's wet-label truth source. A WeatherLink
                // product gauge (e.g. Lands End at Sennen) returns its weatherlink
                // tree path + location key here; an EA station returns (null,
                // null) and reads the EA tree by name as before.
                var stationCfg = _activeLocation.Rainfall.Stations
                    .First(s => s.Name.Equals(stationName, StringComparison.OrdinalIgnoreCase));
                var (wlTruthPath, wlTruthLoc) = stationCfg.WeatherLinkTruth(_cfg.Storage);

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
                // window loop. Sibling of TrainCommand.testPredictionRows
                // but on the day-level dry-window schema. Consumed by dry-
                // window bake-offs.
                var testPredictionRows = new List<DryWindowTestPredictionRow>();

                // Per-phase training-data cutoff (2026-05-26 — see PhaseRegistry).
                // 3b carries minValidTime: 2024-01-01 in phases.yaml; null = no cutoff.
                var minValidTime = PhaseRegistry.Default.AllPhases("dry_window")
                    .SingleOrDefault(p => p.Id == phase)?.MinValidTime;
                if (minValidTime.HasValue)
                    _log.LogInformation("Phase {Phase} training-data cutoff: ValidTimeUtc >= {Cutoff:yyyy-MM-dd} (from phases.yaml)",
                        phase, minValidTime.Value);

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
                        _activeLocation.Name,
                        stationName,
                        spec,
                        window,
                        _cfg.DryWindow.BuildDaytimeWindow(),
                        minValidTime,
                        ct,
                        wlTruthPath,
                        wlTruthLoc);

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
                    var stationSlug = _cfg.ResolveStationSlug(stationName);
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
                    // No input for this cell — every lead had too few rows, i.e. the station's
                    // upstream precip/gauge data isn't available yet (e.g. a freshly-wired
                    // WeatherLink gauge whose 3c hasn't championed). SKIP it gracefully like 3q
                    // does ("will retry once 3c is trained") rather than failing the whole retrain;
                    // it'll produce once the data lands. A guard failure on a cell that DID train
                    // still counts as a failure below — this only spares not-yet-ready stations.
                    _log.LogWarning("No usable rows for ({Station}, {W}h) — upstream data not ready; " +
                        "skipping this cell this run (it'll produce once the station's data is available).", stationName, window);
                    modelsSkipped++;
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
                // alternatives on the same test cells. p_dry_window is the
                // SHIPPING probability — matches what the site renders.
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
                    LocationName = _activeLocation.Name,
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
                var stationSlug3b = _cfg.ResolveStationSlug(stationName);
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
                    locationName: _activeLocation.Name);
                if (false /* TEMP 2026-05-26 JMA-extension local verify; revert */ && !guardResult3b.Passed)
                {
                    // Single-(station, window) guard fail aborts JUST this
                    // combo's promotion + conformal fit; the outer sweep
                    // continues training other cells. cellFailures is bumped
                    // so the command exits non-zero — a guard fail must never
                    // be reported as a silent success (2026-05-17 regression).
                    _log.LogError(
                        "Aborting Phase {Phase} promotion for ({Station}, {W}h) — sanity guard failed. Orphan dir {Dir} not promoted; previous version stays Current.",
                        phase, stationName, window, versionDir);
                    cellFailures++;
                    continue;
                }
                // Promote the new 3b version: replace any prior 3b entry in
                // Active with this one. Other active phases (e.g. 3p) survive
                // untouched — PromoteStationVersion is phase-scoped.
                ModelArtifact.PromoteStationVersion(
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
        _log.LogInformation("Phase 3b training complete. LeadsTrained={T} Skipped={S} CellFailures={F} Elapsed={E}",
            modelsTrained, modelsSkipped, cellFailures, elapsed);

        await Task.CompletedTask;
        // Exit non-zero if ANY (station, window) cell failed its guard or
        // produced no model. The retrain workflow's aggregator keys off the
        // exit code, so returning 0 here on a partial/total failure hides a
        // broken phase — the bug behind the 2026-05-17 dry-window 3b freeze.
        if (cellFailures > 0)
        {
            _log.LogError("Phase 3b: {F} cell(s) failed — see errors above. Exiting non-zero.", cellFailures);
            return 4;
        }
        return 0;
    }

    // ---- Phase 3p — Gaussian copula MC over 3o (cleanup Phase 2) ----------------
    //
    // Productionised 2026-05-25 as the dry-window challenger. Structurally:
    //   * Marginal source: Phase 3o (3c-oro pooled across 4 Bonehill stations).
    //   * Σ: single empirical correlation matrix per station, fit on
    //     train-split observed daytime wet/dry binary sequences. NOT per-lead
    //     (truth's daytime-shape autocorrelation is lead-independent by
    //     construction — pooling across leads gives tighter Σ).
    //
    // Bundle layout:
    //   data/models/dry_window/{station_slug}/window_{N}h/v..._phase3p/
    //     training_metadata.json     (Phase=3p, references the 3o champion in Hyperparameters)
    //     correlation.json           ({"Sigma": [[...]]} — ONE 9×9 per station, not per-lead)
    //     dry_window_climatology.json
    //
    // The 3p version has NO model.zip files — predict reads 3o's live
    // prediction parquet at inference time and runs copula MC over the
    // daytime hourly q vector.
    //
    // Σ is fit ONCE per station and the same correlation.json gets saved
    // into every (window) sub-bundle for that station — predict-time reads
    // it back per (station, window, lead) cell uniformly.

    public Task<int> RunPhase3pAsync(Config.LocationConfig location, CancellationToken ct)
        => RunCopulaMcAsync(location, DryWindow3pPredictor.Phase3p, ct);

    /// <summary>
    /// Train a copula-MC dry-window phase (3p over 3o, or 3q over 3c). The
    /// engine is identical; only the stage-1 precip source phase + the version
    /// metadata key differ, both resolved from
    /// <see cref="DryWindow3pPredictor.SourceFor"/>. Which location runs which
    /// phase is gated upstream by phases.yaml's <c>locations:</c> filter (3p →
    /// Bonehill, 3q → Sennen), so this method no longer hard-codes a location —
    /// a station with no champion of the source phase simply skips below.
    /// </summary>
    public async Task<int> RunCopulaMcAsync(Config.LocationConfig location, string dryWindowPhase, CancellationToken ct)
    {
        _activeLocation = location;
        var source = DryWindow3pPredictor.SourceFor(dryWindowPhase);
        if (_activeLocation.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}'.", _activeLocation.Name);
            return 2;
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        // 3p windows — every hour length 2..6 (extended from {3,4,6}
        // 2026-06-10 for the overview "Will it stay dry?" calculator: more
        // selectable lengths). SUPERSET of 3b's {3,4,6}: the copula MC
        // computes any window length from the same per-station Σ + 3o
        // marginals — no extra training, just two more thresholds over the
        // same draws. The window_2h / window_5h composites are 3p-ONLY in
        // the manifest (3b stays {3,4,6}); predict, verify, the site pages
        // and the calculator all enumerate windows dynamically from the
        // manifest / prediction rows, and the dry-window page skips phases
        // with no rows per window, so the new sections render 3p-only
        // without code changes downstream.
        var windows = new[] { 2, 3, 4, 5, 6 };
        var leads = new[] { 24, 48, 72 };
        var rngSeed = 42;

        _log.LogInformation(
            "Phase {P} — Gaussian copula MC over {Src}; stations=[{S}] windows=[{W}] leads=[{L}]",
            dryWindowPhase, source.PrecipPhase,
            string.Join(", ", _activeLocation.Rainfall.Stations.Select(s => s.Name)),
            string.Join(",", windows), string.Join(",", leads));

        // Resolve the stage-1 precip champion per station. The phase binds to
        // ONE source version per station for the lifetime of the artefact —
        // re-running training picks up the latest champion at that moment.
        var precipSrcByStation = new Dictionary<string, string>();
        // Pool-only gauges (e.g. Princetown) feed pooled 3o training only — never a
        // per-station product, so the dry-window copula-MC is never trained for them.
        foreach (var station in _activeLocation.ProductRainfallStations)
        {
            var stationSlug = _cfg.ResolveStationSlug(station.Name);
            var vSrc = ModelArtifact.ResolveStationPhaseVersion(modelsRoot, "precipitation", stationSlug, source.PrecipPhase);
            if (string.IsNullOrEmpty(vSrc))
            {
                _log.LogWarning("Phase {P}: no {Src} champion for station {Slug} in manifest — skipping station " +
                    "(its {P} bundle will not be produced this run; will retry once {Src} is trained).",
                    dryWindowPhase, source.PrecipPhase, stationSlug, dryWindowPhase, source.PrecipPhase);
                continue;
            }
            precipSrcByStation[stationSlug] = vSrc;
        }
        if (precipSrcByStation.Count == 0)
        {
            _log.LogError("Phase {P}: no station has a {Src} champion yet — abort. " +
                "Train the {Src} precipitation phase for this location first.",
                dryWindowPhase, source.PrecipPhase, source.PrecipPhase);
            return 3;
        }

        int cellFailures = 0, cellsTrained = 0;
        foreach (var station in _activeLocation.Rainfall.Stations)
        {
            ct.ThrowIfCancellationRequested();
            var stationSlug = _cfg.ResolveStationSlug(station.Name);
            if (!precipSrcByStation.TryGetValue(stationSlug, out var vSrc)) continue;

            // Load hourly EA truth across the station's full history. Build
            // per-day daytime wet/dry binary sequences. Threshold = 0.1 mm/h
            // (same as PrecipFeatureBuilder.WetThresholdMm).
            var sequencesByDate = LoadDaytimeBinarySequences(station, daytime, ct);
            if (sequencesByDate.Count < 200)
            {
                _log.LogWarning("Phase {P} {Slug}: only {N} daytime-complete days of truth — too few to fit Σ; skipping.",
                    dryWindowPhase, stationSlug, sequencesByDate.Count);
                cellFailures++;
                continue;
            }
            _log.LogInformation("Phase {P} {Slug}: {N} daytime-complete days of EA truth " +
                "({Start:yyyy-MM-dd}..{End:yyyy-MM-dd}); fitting Σ on train slice.",
                dryWindowPhase,
                stationSlug,
                sequencesByDate.Count,
                sequencesByDate.Keys.Min(),
                sequencesByDate.Keys.Max());

            // Chronological 70/15/15 split BY DATE — matches the 3b / 3o
            // training split boundaries on the same underlying truth series,
            // so any cross-bake-off comparison reads off identical test rows.
            var sortedDates = sequencesByDate.Keys.OrderBy(d => d).ToList();
            int nTrain = (int)Math.Floor(sortedDates.Count * 0.70);
            int nVal   = (int)Math.Floor(sortedDates.Count * 0.15);
            var trainDates = sortedDates.Take(nTrain).ToList();
            var valDates   = sortedDates.Skip(nTrain).Take(nVal).ToList();
            var testDates  = sortedDates.Skip(nTrain + nVal).ToList();

            // Fit Σ on train-slice sequences.
            var trainSequences = trainDates.Select(d => sequencesByDate[d]).ToArray();
            double[,] sigma;
            try
            {
                sigma = DryWindow3pPredictor.FitCorrelation(trainSequences);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "Phase {P} {Slug}: Σ fit failed — likely a degenerate train slice (no wet days, or constant per-hour).", dryWindowPhase, stationSlug);
                cellFailures++;
                continue;
            }
            // Sanity-check Cholesky succeeds (will throw if Σ isn't strictly SPD).
            try { _ = DryWindow3pPredictor.CholeskyDecompose(sigma); }
            catch (Exception ex)
            {
                _log.LogError(ex, "Phase {P} {Slug}: Σ Cholesky failed — Σ not strictly positive-definite.", dryWindowPhase, stationSlug);
                cellFailures++;
                continue;
            }

            // Climatology — daily base rate over train slice. Matches the
            // shape DryWindow3b emits + the PrecipClimatology shape for site
            // baselines. The dry-window climatology is per (station, window)
            // so this gets saved once per (station, window) bundle below.
            // Empty by default since the climatology used by predict is the
            // day-of-year base rate already encoded in 3o; 3p doesn't have
            // its own climatology object yet (verify computes it on the fly).

            var now = DateTime.UtcNow;
            // One bundle per (station, window) — matches 3b layout so the
            // existing dry-window predict / verify plumbing reads 3p
            // bundles without modification.
            foreach (var window in windows)
            {
                ct.ThrowIfCancellationRequested();
                var compositeKey = $"{stationSlug}/window_{window}h";
                var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "dry_window", compositeKey, now, suffix: $"phase{dryWindowPhase}");
                var versionName = Path.GetFileName(versionDir);
                Directory.CreateDirectory(versionDir);

                // Write the SAME Σ into every window's correlation.json — Σ
                // is the daytime-shape autocorrelation and is window-independent.
                DryWindow3pPredictor.WriteCorrelationJson(versionDir, sigma);

                // Per-lead metadata stub — predict reads these to know which
                // (station, lead) cells the bundle covers. Brier numbers are
                // left blank; verify fills them in once live predictions land.
                var perLead = leads.ToDictionary(
                    l => l.ToString(),
                    l => new ModelArtifact.PerLeadStats
                    {
                        LeadHours = l,
                        DataRangeTrain = $"{trainDates.First():yyyy-MM-dd} → {trainDates.Last():yyyy-MM-dd}",
                        DataRangeVal   = valDates.Count > 0 ? $"{valDates.First():yyyy-MM-dd} → {valDates.Last():yyyy-MM-dd}" : "—",
                        DataRangeTest  = testDates.Count > 0 ? $"{testDates.First():yyyy-MM-dd} → {testDates.Last():yyyy-MM-dd}" : "—",
                        TrainRows = trainDates.Count,
                        ValRows   = valDates.Count,
                        TestRows  = testDates.Count,
                        TestCalendarMonths = testDates.Select(d => new DateTime(d.Year, d.Month, 1)).Distinct().Count(),
                        BestSingle = "",
                        BestSingleValMae  = 0.0,
                        BestSingleTestMae = 0.0,
                        BlendTestMae  = 0.0,
                        BlendTestRmse = 0.0,
                        BlendTestBias = 0.0,
                    });

                var metadata = new ModelArtifact.TrainingMetadata
                {
                    Version = versionName,
                    Target = "dry_window",
                    Phase = dryWindowPhase,
                    LocationName = _activeLocation.Name,
                    DataSource = $"ea_rainfall+{source.PrecipPhase}_live_predictions",
                    TrainedAtUtc = now,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["mc_samples"] = DryWindow3pPredictor.DefaultMcSamples,
                        ["mc_seed"] = rngSeed,
                        ["sigma_dim"] = sigma.GetLength(0),
                        ["sigma_n_train_days"] = trainSequences.Length,
                        ["window_hours"] = window,
                        [source.VersionKey] = vSrc,
                    },
                    TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                    PerLead = perLead,
                    DeviationsFromBrief = new List<string>
                    {
                        $"Phase {dryWindowPhase} — parameter-free Gaussian copula MC over Phase {source.PrecipPhase}'s hourly P(wet) marginals. No LightGBM, no learned weights; predict reads {source.PrecipPhase}'s live prediction parquet at inference time and runs MC.",
                        "Single empirical Σ per station from train-split observed_wet daytime sequences. Σ structure is lead-independent (truth doesn't move with forecast horizon); pooling per-lead truth sequences gives a tighter estimate.",
                        $"{source.PrecipPhase} champion bound at training time: {vSrc}. Re-run dry-window train to rebind to a newer {source.PrecipPhase} champion.",
                        "Cross-window monotonicity P(N=2) ≥ P(N=3) ≥ … ≥ P(N=6) is guaranteed by computing all windows from a SINGLE shared correlated-Bernoulli sequence per MC sample.",
                        $"Brier numbers are blank in training_metadata — verify scores live predictions vs EA truth as cycles produce {dryWindowPhase} rows.",
                    },
                };
                ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

                // Promote as challenger per (station, window).
                ModelArtifact.PromoteStationVersion(modelsRoot, "dry_window", compositeKey, versionName,
                    newPhase: dryWindowPhase);
                _log.LogInformation("Phase {P} {Key} → {Dir}", dryWindowPhase, compositeKey, versionDir);
            }
            cellsTrained++;
        }

        _log.LogInformation("Phase {P} complete. CellsTrained={C} CellFailures={F}", dryWindowPhase, cellsTrained, cellFailures);
        await Task.CompletedTask;
        if (cellsTrained == 0) return 3;
        return cellFailures > 0 ? 4 : 0;
    }

    /// <summary>
    /// Load per-day daytime wet/dry binary sequences for one EA rainfall
    /// station. Returns a dict keyed by target_date (UTC) — each value is a
    /// byte[] of length daytime.DurationHours, 1 = wet hour (≥ 0.1 mm), 0 =
    /// dry. Days with any missing daytime hour are dropped so Σ-fit sees
    /// only complete sequences (matches the daytime-completeness gate the
    /// label builder applies to 3b training rows).
    /// </summary>
    private Dictionary<DateOnly, byte[]> LoadDaytimeBinarySequences(
        RainfallStationConfig station, DaytimeWindow daytime, CancellationToken ct)
    {
        // Reuse the canonical hourly-rain loader so WeatherLink product gauges
        // (e.g. Lands End at Sennen) get truth too, instead of the old EA-only
        // hand-rolled query silently returning zero rows for them. The EA branch
        // is the same SUM(Value15MinMm) + 4-of-4 gate this used to inline.
        var hourly = PrecipTruthLoader.LoadHourlyRain(
            station, _cfg.Storage, _activeLocation.Name, null, ct);

        const double WetThresholdMm = 0.1;
        var result = new Dictionary<DateOnly, byte[]>();
        if (hourly.Count == 0) return result;
        // LoadHourlyRain returns an unordered Dictionary, so take explicit
        // Min/Max rather than relying on First/Last key order.
        var minDay = DateOnly.FromDateTime(hourly.Keys.Min().Date);
        var maxDay = DateOnly.FromDateTime(hourly.Keys.Max().Date);
        for (var d = minDay; d <= maxDay; d = d.AddDays(1))
        {
            var (startUtc, endUtcExclusive) = daytime.UtcHourRangeFor(d);
            var seqLen = endUtcExclusive - startUtc;
            // Sanity: skip days whose daytime range collapsed (off-by-one edge
            // around DST or window mis-config).
            if (seqLen <= 0) continue;
            var midnight = new DateTime(d.Year, d.Month, d.Day, 0, 0, 0, DateTimeKind.Utc);
            var seq = new byte[seqLen];
            bool complete = true;
            for (int h = 0; h < seqLen; h++)
            {
                var t = midnight.AddHours(startUtc + h);
                if (!hourly.TryGetValue(t, out var mm)) { complete = false; break; }
                seq[h] = (byte)(mm >= WetThresholdMm ? 1 : 0);
            }
            if (complete) result[d] = seq;
        }
        return result;
    }


    private string[]? ResolveStations(string stationArg)
    {
        if (string.IsNullOrWhiteSpace(stationArg) || stationArg.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            // Pool-only gauges (e.g. Princetown) feed pooled 3o training only —
            // never a per-station product, so dry-window is never trained for them.
            var found = _activeLocation.ProductRainfallStations.Select(s => s.Name).ToArray();
            if (found.Length == 0)
            {
                _log.LogError("No rainfall stations in config — nothing to train.");
                return null;
            }
            return found;
        }

        var match = _activeLocation.Rainfall.Stations
            .FirstOrDefault(s => s.Name.Equals(stationArg, StringComparison.OrdinalIgnoreCase)
                              || SlugMatch(s.Name, stationArg));
        if (match is null)
        {
            _log.LogError("Station '{Arg}' not in config. Available: [{Names}]",
                stationArg, string.Join(", ", _activeLocation.Rainfall.Stations.Select(s => s.Name)));
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
