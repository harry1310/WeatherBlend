using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.DryWindow;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;

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

    private static readonly int[] DefaultLeads = { 24, 48, 72 };
    private static readonly int[] DefaultWindows = { 3, 4, 6 };
    private static readonly string[] Phase3bStations = { "Bellever Dartmoor", "Princetown" };

    public DryWindowTrainCommand(ILogger<DryWindowTrainCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(string stationArg, string windowArg, int[] leads, CancellationToken ct)
        => RunAsync(stationArg, windowArg, leads, DryWindowFeatureBuilder.Phase3b, ct);

    public async Task<int> RunAsync(string stationArg, string windowArg, int[] leads, string phase, CancellationToken ct)
    {
        if (phase != DryWindowFeatureBuilder.Phase3b && phase != DryWindowFeatureBuilder.Phase3dShape)
        {
            _log.LogError("Unsupported dry-window training phase '{Phase}'. Expected '3b' or '3d-shape'.", phase);
            return 2;
        }

        var stations = ResolveStations(stationArg);
        if (stations is null) return 2;

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

        var modelsRoot = Path.Combine("data", "models");
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
                var compositeKey = $"ea_{Slugify(stationName)}/window_{window}h";
                var now = DateTime.UtcNow;
                // 3b → v{ts} (champion, no suffix). 3d-shape → v{ts}_phase3d_shape so
                // the manifest's Active list visually distinguishes the variants.
                var suffix = phase == DryWindowFeatureBuilder.Phase3dShape ? "phase3d_shape" : null;
                var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "dry_window", compositeKey, now, suffix);
                var versionName = Path.GetFileName(versionDir);

                _log.LogInformation("=== Station '{Station}', window {W}h → {Key} ===",
                    stationName, window, compositeKey);

                var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
                var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
                DryWindowClimatology? climatology = null;
                bool anyLeadTrained = false;

                foreach (var lead in leads)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- Lead {Lead}h ---", lead);

                    var rows = DryWindowFeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath,
                        _cfg.Storage.RainfallPath,
                        _cfg.Location.Name,
                        stationName,
                        lead,
                        window,
                        ct);

                    var pos = rows.Count(r => r.HasDryWindow);
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

                    var ds = DryWindowDataset.Split(rows);
                    climatology ??= DryWindowClimatology.BuildFromTraining(ds.Train, window);

                    _log.LogInformation("Split → train {TN} (pos {TP}), val {VN} (pos {VP}), test {EN} (pos {EP})",
                        ds.Train.Count, ds.TrainPositives,
                        ds.Val.Count,   ds.ValPositives,
                        ds.Test.Count,  ds.TestPositives);
                    _log.LogInformation("Date ranges — train {T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}, " +
                                        "val {V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}, test {E0:yyyy-MM-dd}..{E1:yyyy-MM-dd}",
                        ds.TrainStart, ds.TrainEnd, ds.ValStart, ds.ValEnd, ds.TestStart, ds.TestEnd);

                    var trained = DryWindowTrainer.Train(ds, hp, phase);

                    var truthTest = ds.Test.Select(r => r.HasDryWindow ? 1.0 : 0.0).ToArray();
                    var blendProb = DryWindowTrainer.PredictProbability(trained.Ml, trained.Model, ds.Test);
                    var blendBrier = PrecipMetrics.Brier(blendProb, truthTest);

                    var best = DryWindowBaselines.BestSingle(ds.Val);
                    var bestValBrier = PrecipMetrics.Brier(
                        DryWindowBaselines.SingleModelHasDryWindow(ds.Val, best),
                        ds.Val.Select(r => r.HasDryWindow ? 1.0 : 0.0).ToArray());

                    var climPred = DryWindowBaselines.Climatology(ds.Train, ds.Test, window);
                    var climBrier = PrecipMetrics.Brier(climPred, truthTest);
                    var bss = PrecipMetrics.BrierSkillScore(blendBrier, climBrier);

                    var blendBinary = blendProb.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
                    var fbias = PrecipMetrics.FrequencyBias(blendBinary, truthTest);

                    ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
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
                        BestSingleValMae = bestValBrier,  // reused column — documented in DeviationsFromBrief
                        BlendTestMae  = blendBrier,
                        BlendTestRmse = climBrier,        // reused for climatology Brier
                        BlendTestBias = fbias,
                    };

                    _log.LogInformation(
                        "Lead {Lead}h — blend Brier={Brier:0.0000}, clim Brier={Clim:0.0000}, BSS={Bss:+0.0000;-0.0000;0.0000}, " +
                        "freq_bias={Fb:0.00}, best_single[{Best}] val Brier={BestBrier:0.0000}",
                        lead, blendBrier, climBrier, bss, fbias, best, bestValBrier);

                    modelsTrained++;
                }

                if (!anyLeadTrained)
                {
                    _log.LogWarning("No leads trained for ({Station}, {W}h); skipping artefact save.",
                        stationName, window);
                    continue;
                }

                ModelArtifact.SaveFeatureSchema(versionDir, DryWindowFeatureBuilder.FeatureNamesForPhase(phase));
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
                    // 3b is the champion — replace Current and reset Active to just this version.
                    ModelArtifact.UpdateStationManifest(modelsRoot, "dry_window", compositeKey, versionName);
                }
                else
                {
                    // 3d-shape (challenger) — keep 3b's Current pointer, extend Active so
                    // predict/verify produce rows for both. Stale 3d-shape entries get
                    // dropped so each train run leaves at most one of each phase active.
                    ModelArtifact.AppendStationVersion(modelsRoot, "dry_window", compositeKey, versionName);
                    var existing = ModelArtifact.ResolveStationActive(modelsRoot, "dry_window", compositeKey);
                    var nextActive = existing
                        .Where(v => !v.Contains("phase3d_shape", StringComparison.Ordinal))
                        .Append(versionName);
                    ModelArtifact.SetStationActive(modelsRoot, "dry_window", compositeKey, nextActive);
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

    private string[]? ResolveStations(string stationArg)
    {
        if (string.IsNullOrWhiteSpace(stationArg) || stationArg.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            var found = Phase3bStations
                .Where(t => _cfg.Location.Rainfall.Stations.Any(s => s.Name.Equals(t, StringComparison.OrdinalIgnoreCase)))
                .ToArray();
            if (found.Length == 0)
            {
                _log.LogError("Phase-3b stations missing from config. Expected one of: [{S}]",
                    string.Join(", ", Phase3bStations));
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
        var slug = Slugify(configName);
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

    private static string Slugify(string s) => s.ToLowerInvariant()
        .Replace(' ', '_').Replace('-', '_').Replace(',', '_');

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
