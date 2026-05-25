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

        // 3g / 3j / 3n / 3s and the local 3f MLP bake-off branches were
        // removed 2026-05-25 in model-cleanup Phase 1 (see
        // project_model_cleanup_plan_2026-05-25.md). Phase 3b remains the
        // dry-window champion until cleanup Phase 2 productionises 3p
        // (Gaussian copula MC over 3o).
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
                        _activeLocation.Name,
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
                    _log.LogError("No leads trained for ({Station}, {W}h) — every lead had too few " +
                        "rows; cell not produced. Counts as a failure.", stationName, window);
                    cellFailures++;
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
                    locationName: _activeLocation.Name);
                if (!guardResult3b.Passed)
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
                // Active with this one. Any OTHER active phases (3g/3j/3n/3s
                // today) survive untouched.
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



    private string[]? ResolveStations(string stationArg)
    {
        if (string.IsNullOrWhiteSpace(stationArg) || stationArg.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            var found = _activeLocation.Rainfall.Stations.Select(s => s.Name).ToArray();
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
