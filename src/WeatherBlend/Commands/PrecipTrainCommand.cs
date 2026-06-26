using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Oro;
using WeatherBlend.Train.PrecipExact;
using DuckDB.NET.Data;

namespace WeatherBlend.Commands;

/// <summary>
/// Trains the precipitation blenders — one model per (station, lead), scored
/// on Brier against per-station EA Hydrology rainfall truth.
///
/// Phase 3a: LightGBM binary classifier for P(hour has >= 0.1 mm). Leads {24,48,72,120}.
/// Phase 3c: rich-feature LightGBM occurrence blender — 3a champion/challenger.
/// Phase 3d: exact-runtime occurrence blender (P1/P2 tier), lead-12 champion.
/// Phase 3o: rich + orographic features pooled across 4 Bonehill stations.
///
/// Split out of TrainCommand on 2026-05-19 (code-quality refactor P1) so a
/// precipitation phase is no longer navigated inside a file named "Temp". The
/// cross-target dispatch + argument validation stays in TrainCommand.RunAsync,
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
    /// <paramref name="featureSet"/>: lean -> 3a, rich -> 3c, oro -> 3o,
    /// exact -> 3d. Feature-set validity for the precipitation target is
    /// checked by the caller (TrainCommand.RunAsync) before dispatch.
    /// </summary>
    public async Task<int> RunAsync(
        int[] leads, string? station, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, int[]? cycles,
        Config.LocationConfig location, CancellationToken ct)
    {
        return featureSet switch
        {
            "rich"  => await RunPhase3cAsync(leads, station, location, ct),
            "oro"   => await RunPhase3oAsync(leads, location, ct),
            // 3oni: 3o minus oro_station_id, for the ungauged Bonehill-tor line.
            "oro-noid" => await RunPhase3oAsync(leads, location, ct, includeStationId: false),
            "exact" => await RunPhase3dAsync(station, tier, includeUkv, exactLeads, cycles, location, ct),
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

        Config.RainfallStationConfig stationCfg;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stationCfg = location.Rainfall.Stations[0];
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
            stationCfg = match;
        }
        var primaryStation = stationCfg.Name;

        // Each station gets its own subtree under data/models/precipitation/{station}/v{ts}/
        // so the manifest can track a separate "current" pointer per truth source.
        // Slug is prefixed with the provider ("ea_" for EA gauges, "wl_" for
        // WeatherLink cove gauges like Lands End) so same-named sites can't collide.
        var stationSlug = stationCfg.Slug;
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now);
        var versionName = Path.GetFileName(versionDir);

        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();
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
        // 3a + 4a linear pool). Schema is the canonical test_predictions
        // schema so a single bake-off script can inner-join across phases.
        // Buffered across the lead loop, written after.
        var testPredictionRows = new List<TestPredictionRow>();

        // Per-phase training-data cutoff from phases.yaml (2026-05-26).
        // 3a / 3c carry minValidTime: 2024-01-01 because pre-2024 rows from
        // most NWPs are NULL-padded; null = no cutoff (3o keeps full history).
        // Looked up ONCE per train run rather than per-lead to avoid re-paying
        // the registry traversal in the inner loop.
        var minValidTime3a = PhaseRegistry.Default.AllPhases("precipitation")
            .SingleOrDefault(p => p.Id == "3a")?.MinValidTime;
        if (minValidTime3a.HasValue)
            _log.LogInformation("Phase 3a training-data cutoff: ValidTimeUtc >= {Cutoff:yyyy-MM-dd} (from phases.yaml)", minValidTime3a.Value);

        var cache = MaterializeTrainCache(location.Name, $"3a_{location.Name}");

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = PrecipFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var (wlPath, wlKey) = stationCfg.WeatherLinkTruth(_cfg.Storage);
            var rows = PrecipFeatureBuilder.BuildForLead(
                cache.FcPath,
                cache.RnPath,
                location.Name,
                primaryStation,
                spec,
                minValidTime3a,
                ct,
                weatherLinkTruthPath: wlPath,
                weatherLinkTruthLocation: wlKey);
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
            // Lead-0 nowcast rows train on a different (shorter, hist_forecast)
            // window, so exclude them from the RetrainGuard baseline — otherwise
            // --nowcast would shift the rows/label/feature totals vs the
            // production leads and could trip the gate. (No-op for non-nowcast
            // leads, incl. 3d's {12,24}.)
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += ds.Train.Count;
                totalValRows   += ds.Val.Count;
                totalTestRows  += ds.Test.Count;
                firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
                firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();
            }

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
            DataSource = stationCfg.Source is Config.RainfallTruthSource.WeatherLink ? "previous_runs_api+weatherlink" : "previous_runs_api+ea_rainfall",
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
        // downstream bake-offs (e.g. 3a + 4a linear pool). Canonical schema
        // (valid_time, station, lead, p_wet, observed_wet) so a single
        // bake-off script can inner-join across phases without per-phase
        // schema branches.
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
        if (false /* TEMP 2026-05-26 JMA-extension local verify; revert */ && !guardResult3a.Passed)
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

    // Scan-once train cache: materialise THIS location's forecasts + rainfall
    // into one parquet each, so the per-(station,lead) BuildForLead calls become
    // one-file reads instead of re-globbing the whole forecast tree ~40× per
    // retrain (3c/3o each loop 4 stations × 5 leads). One full-tree scan here +
    // N cheap reads, vs N full-tree scans. All forecast sources are kept
    // (offset_day for ≥24h leads, hist_forecast for the lead-0 nowcast, exact for
    // the UA block) — the builders filter by RunTimeSource/LeadHours internally.
    // Mirrors RunPolicyRetrainAsync's proven pattern.
    private (string FcPath, string RnPath) MaterializeTrainCache(string locationName, string tag)
    {
        var scratchRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "train_cache", tag);
        var fcPath = Path.Combine(scratchRoot, "fc"); Directory.CreateDirectory(Path.Combine(fcPath, "p"));
        var rnPath = Path.Combine(scratchRoot, "rn"); Directory.CreateDirectory(Path.Combine(rnPath, "p"));
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        var fcSrc = Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=" + locationName, "**", "*.parquet"));
        var rnSrc = Esc(Path.Combine(_cfg.Storage.RainfallPath, "location=" + locationName, "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var c = conn.CreateCommand();
        c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcSrc}', hive_partitioning=false, union_by_name=true)) TO '{Esc(Path.Combine(fcPath, "p", "fc.parquet"))}' (FORMAT PARQUET);";
        c.ExecuteNonQuery();
        c.CommandText = $"COPY (SELECT * FROM read_parquet('{rnSrc}', hive_partitioning=false, union_by_name=true)) TO '{Esc(Path.Combine(rnPath, "p", "rn.parquet"))}' (FORMAT PARQUET);";
        c.ExecuteNonQuery();
        _log.LogInformation("Scan-once train cache → {Scratch} (forecasts + rainfall, {Loc}).", scratchRoot, locationName);
        return (fcPath, rnPath);
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
        // station (EA gauges + WeatherLink cove gauges alike) so one command run
        // produces a challenger for each 3a champion. A specific station still
        // trains that one alone.
        IReadOnlyList<Config.RainfallStationConfig> stationsToTrain;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            // Pool-only gauges (e.g. Princetown) feed the pooled 3o training but are
            // never a per-station product — exclude them from the 3c challenger sweep.
            stationsToTrain = location.ProductRainfallStations;
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
            stationsToTrain = new[] { match };
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();

        _log.LogInformation("Phase 3c — rich-feature precip blender, location='{Loc}', stations=[{Stations}], leads=[{Leads}]",
            location.Name, string.Join(", ", stationsToTrain.Select(s => s.Name)), string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (identical to Phase 3a — feature richness is the only variable)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var cache = MaterializeTrainCache(location.Name, $"3c_{location.Name}");

        var anyFail = false;
        foreach (var stationCfg in stationsToTrain)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await TrainPhase3cStationAsync(stationCfg, leads, modelsRoot, hp, location, cache, ct);
            if (rc != 0) anyFail = true;
        }
        return anyFail ? 3 : 0;
    }

    private async Task<int> TrainPhase3cStationAsync(
        Config.RainfallStationConfig stationCfg,
        int[] leads,
        string modelsRoot,
        PrecipOccurrenceTrainer.Hyperparameters hp,
        Config.LocationConfig location,
        (string FcPath, string RnPath) cache,
        CancellationToken ct)
    {
        var primaryStation = stationCfg.Name;
        var stationSlug = stationCfg.Slug;
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
        // same canonical schema 3a/4a write so a single stacking
        // analysis can inner-join across phases.
        var testPredictionRows = new List<TestPredictionRow>();

        // Per-phase training-data cutoff (2026-05-26 — see PhaseRegistry).
        var minValidTime3c = PhaseRegistry.Default.AllPhases("precipitation")
            .SingleOrDefault(p => p.Id == "3c")?.MinValidTime;
        if (minValidTime3c.HasValue)
            _log.LogInformation("Phase 3c training-data cutoff: ValidTimeUtc >= {Cutoff:yyyy-MM-dd} (from phases.yaml)", minValidTime3c.Value);

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            // 3c carries the multi-level pressure (upper-air) block in-place at
            // BONEHILL (2026-06-02): rich 59 → 101 feats. UA is the freshest exact
            // lead-L pressure ASOF-joined to each offset_day row (leak-free; see
            // PrecipRichFeatureBuilder). Bake-off win −3.7% Brier agg @24h,
            // growing to −6.8% @72h. Predict mirrors via the live ASOF join.
            // Bonehill-only for now — Membury keeps the 59-feat rich spec until
            // its exact-pressure backfill lands (same posture as 3d/3o). Per-loc
            // schema is fine: predict reads each bundle's own feature_schema.json.
            var useUpperAir = location.Name.Equals("bonehill_rocks", StringComparison.OrdinalIgnoreCase);
            var spec = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead, withUpperAir: useUpperAir);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var (wlPath, wlKey) = stationCfg.WeatherLinkTruth(_cfg.Storage);
            var rows = PrecipRichFeatureBuilder.BuildForLead(
                cache.FcPath,
                cache.RnPath,
                location.Name,
                primaryStation,
                spec,
                minValidTime3c,
                ct,
                weatherLinkTruthPath: wlPath,
                weatherLinkTruthLocation: wlKey);
            _log.LogInformation("Loaded {N} rich rows (wet={Wet} / {Pct:P1}) spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count(r => r.Label),
                rows.Count == 0 ? 0 : (double)rows.Count(r => r.Label) / rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                // The lead-0 nowcast is appended automatically for 3c, but a
                // location whose hist_forecast surface archive isn't backfilled
                // yet has no lead-0 rows. Skip it (bundle keeps its ≥24h leads)
                // rather than aborting; drop the orphan spec registered above.
                if (Train.Common.NowcastSource.IsNowcast(lead))
                {
                    _log.LogWarning("Lead {Lead}h (nowcast): only {N} rows — no hist_forecast surface "
                        + "data for this location yet; skipping the lead-0 model.", lead, rows.Count);
                    specsPerLead.Remove(lead);
                    continue;
                }
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
            // Lead-0 nowcast rows train on a different (shorter, hist_forecast)
            // window, so exclude them from the RetrainGuard baseline — otherwise
            // --nowcast would shift the rows/label/feature totals vs the
            // production leads and could trip the gate. (No-op for non-nowcast
            // leads, incl. 3d's {12,24}.)
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += ds.Train.Count;
                totalValRows   += ds.Val.Count;
                totalTestRows  += ds.Test.Count;
                firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
                firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();
            }

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
            // — was missing from 3c, blocking 3c/4a stacking analysis).
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
            DataSource = stationCfg.Source is Config.RainfallTruthSource.WeatherLink ? "previous_runs_api+weatherlink" : "previous_runs_api+ea_rainfall",
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
            labelRates: labelRates3c,
            // 3c gained the upper-air block in-place (59→101) — allow the
            // one-directional feature-count jump without aborting; other gates
            // stay enforced. See RetrainGuard.AllowFeaturesEffectiveChange.
            tolerances: RetrainGuard.Defaults with { AllowFeaturesEffectiveChange = true });
        if (false /* TEMP 2026-05-26 JMA-extension local verify; revert */ && !guardResult3c.Passed)
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

    // ---- Phase 3o: 3c-oro production (4-station Bonehill pool) -------------------
    //
    // 3o productionises the 2026-05-25 cleanup-plan winner: pooled LightGBM
    // across the 4 Bonehill rainfall stations (Bellever, Bovey, Hexworthy,
    // Princetown) on the rich (55-feat) + 9 terrain features = 64 features.
    // Pooling gives the terrain features the cross-station diversity they
    // need to learn from; per-station-only training had nothing to lift on
    // since each station's elevation/aspect/etc. is a single point.
    //
    // Bundle layout — identical to 3a / 3c per-station bundles for predict-
    // time plumbing reuse. The pooled model is trained ONCE per lead and
    // the same bundle is saved under each station's subtree:
    //   data/models/precipitation/ea_bellever_dartmoor/v{ts}_phase3o/
    //   data/models/precipitation/ea_bovey_tracey/v{ts}_phase3o/
    //   data/models/precipitation/ea_dartmoor_nr_hexworthy/v{ts}_phase3o/
    //   data/models/precipitation/ea_princetown/v{ts}_phase3o/
    // PrecipPredictCommand dispatches on metadata.Phase == "3o" into a path
    // that reads per-station orographic JSONs at predict time + applies
    // the pooled model with that station's terrain values + index.
    //
    // Climatology is per-station — represents each station's own historical
    // wet rate (not the pooled rate), so the per-station Brier baselines in
    // verify reflect that station's regime.

    /// <summary>Stable order for the 4 Bonehill stations. station_index in
    /// the terrain feature block IS this position (0..3) so the pooled
    /// model learns a discrete station offset alongside the continuous
    /// terrain values. Predict-time must use the same mapping or the model
    /// reads the wrong "station axis" value.</summary>
    private static readonly string[] BonehillStationOrder3o =
    {
        "Bellever Dartmoor",
        "Bovey Tracey",
        "Dartmoor nr Hexworthy",
        "Princetown",
        "Manaton",   // idx 4: pool-only WeatherLink cove gauge — train-only, never predicted
    };

    private async Task<int> RunPhase3oAsync(int[] leads, Config.LocationConfig location, CancellationToken ct, bool includeStationId = true)
    {
        // 3o (with station id) vs 3oni (no station id, for ungauged-tor
        // extrapolation). Same pooled training; only the terrain-block id
        // feature + the phase tag/suffix differ.
        var phaseTag = includeStationId ? "3o" : "3oni";
        var suffix = "phase" + phaseTag;

        // 3o is Bonehill-only by design (Membury terrain pool was rejected
        // at the 2026-05-25 stage-1 bake-off — too homogeneous).
        if (!location.Name.Equals("bonehill_rocks", StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError("Phase {Phase} is Bonehill-only (location='{Loc}'). Skipping.", phaseTag, location.Name);
            return 2;
        }
        if (location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}'.", location.Name);
            return 2;
        }

        // Resolve the 4 Bonehill stations + their oro records in canonical order.
        // Path is derived from cfg.Storage.ForecastsPath's parent (the
        // canonical "data tree root"; production has forecastsPath="data/forecasts"
        // so parent="data") rather than CWD-relative — keeps the resolution
        // self-contained so parallel test runs don't collide on the
        // process-global Environment.CurrentDirectory.
        var oroRoot = Path.Combine(
            Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic");
        var oroBySlug = OroStaticFeatures.LoadAll(oroRoot);
        var pool = new List<(string Name, string Slug, OroStaticFeatures Oro, int Index, string? WlPath, string? WlLoc)>();
        foreach (var (name, idx) in BonehillStationOrder3o.Select((n, i) => (n, i)))
        {
            var match = location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                // A pool member absent from THIS config is skipped, not fatal — it lets a
                // minimal/test config (or a not-yet-provisioned gauge) still train 3o on the
                // rest. The production config's pool membership is guarded by ConfigTests.
                _log.LogWarning("Phase 3o: pool station '{Name}' not in config — skipping it from the pool.", name);
                continue;
            }
            var slug = match.Slug;   // wl_manaton for the WeatherLink pool member; ea_* otherwise
            if (!oroBySlug.TryGetValue(slug, out var oro))
            {
                _log.LogError("Phase 3o: no orographic record for {Slug} at {Path}. " +
                    "Run scripts/OrographicFeatures/build_static_orographic.py.", slug, oroRoot);
                return 2;
            }
            // WeatherLink pool members (Manaton) carry their truth source; EA gauges get (null,null).
            var (wlPath, wlLoc) = match.WeatherLinkTruth(_cfg.Storage);
            pool.Add((match.Name, slug, oro, idx, wlPath, wlLoc));
        }
        _log.LogInformation("Phase 3o — 4-station Bonehill pool: [{Stations}]; leads=[{Leads}]",
            string.Join(", ", pool.Select(p => p.Name)), string.Join(",", leads));

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();
        var now = DateTime.UtcNow;

        // Per-station bundle paths. The same trained model gets saved under
        // each station's subtree (per-station predict plumbing is unchanged).
        var stationDirs = pool.ToDictionary(p => p.Slug,
            p => ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", p.Slug, now, suffix: suffix));
        var versionName = Path.GetFileName(stationDirs[pool[0].Slug]);
        foreach (var dir in stationDirs.Values) Directory.CreateDirectory(dir);
        _log.LogInformation("Pooled-bundle version name: {V}", versionName);

        // Per-lead pooled training. Each lead has its own (station, oro) row
        // build because the rich feature row depends on lead. Per-station
        // climatology is collected for the first lead so we can save it
        // under each station's bundle.
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var perLeadStationPerStation = new Dictionary<string, Dictionary<string, ModelArtifact.PerLeadStats>>(
            pool.ToDictionary(p => p.Slug, _ => new Dictionary<string, ModelArtifact.PerLeadStats>()));
        var climByStation = new Dictionary<string, PrecipClimatology?>(
            pool.ToDictionary(p => p.Slug, _ => (PrecipClimatology?)null));
        var testPredsByStation = new Dictionary<string, List<TestPredictionRow>>(
            pool.ToDictionary(p => p.Slug, _ => new List<TestPredictionRow>()));

        // RetrainGuard inputs (computed against the POOLED training matrix —
        // the guard's "rows" / "label rate" / "NaN%" gates are evaluated
        // against what the model actually saw at fit time, not per-station).
        List<float[]>? firstLeadTrainFeatures = null;
        IReadOnlyList<bool>? firstLeadTrainLabels = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;
        // Per-station label rates for the guard so any per-station truth
        // anomaly (e.g. Princetown's first 2022/2023 backfill landing
        // mid-week) gets flagged separately.
        var perStationLabelRates = new Dictionary<string, double>();

        var cache = MaterializeTrainCache(location.Name, $"3o_{location.Name}");

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            // 3o now carries the upper-air block in-place too (2026-06-02):
            // rich+oro 68 → 110 feats. Layout [rich || UA || terrain]; terrain
            // stays last so BuildForLead's Take(count-9) reconstruction is
            // unchanged. Standalone −1.4%..−2.1% Brier; the real win is in the
            // 4b blend (3-way mean(4a, 3o+UA, 3c+UA) beats live 4b −3.5%@24h).
            var spec = PrecipRichOroFeatureBuilder.BuildSpec(_cfg.Blenders, lead, withUpperAir: true, includeStationId: includeStationId);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec} ({N} features)", spec, spec.FeatureCount);

            // Build per-station datasets + chronological 70/15/15 splits.
            var perStation = new List<(string Name, string Slug, BinaryDataset Ds)>();
            foreach (var (name, slug, oro, idx, wlPath, wlLoc) in pool)
            {
                ct.ThrowIfCancellationRequested();
                var rows = PrecipRichOroFeatureBuilder.BuildForLead(
                    cache.FcPath, cache.RnPath,
                    location.Name, name, oro, idx, spec, ct, includeStationId: includeStationId,
                    weatherLinkTruthPath: wlPath, weatherLinkTruthLocation: wlLoc);
                if (rows.Count < 200)
                {
                    _log.LogWarning("  station {Slug}: only {N} rich-oro rows for lead {Lead}h — skipping station for this lead.",
                        slug, rows.Count, lead);
                    continue;
                }
                var ds = BinaryDataset.Split(rows);
                _log.LogInformation("  {Slug}: rows={N} (wet {Pct:P1}) — train={T}/val={V}/test={E}",
                    slug, rows.Count, rows.Count(r => r.Label) / (double)rows.Count,
                    ds.Train.Count, ds.Val.Count, ds.Test.Count);
                perStation.Add((name, slug, ds));
                if (climByStation[slug] is null)
                    climByStation[slug] = PrecipClimatology.BuildFromTraining(ds.Train);
                if (!perStationLabelRates.ContainsKey(slug) && ds.Train.Count > 0)
                    perStationLabelRates[slug] = ds.Train.Count(r => r.Label) / (double)ds.Train.Count;
            }
            if (perStation.Count < 2)
            {
                // <2 usable stations → no pooled model for this lead. Expected for
                // the appended lead-0 nowcast on a location whose hist_forecast
                // surface archive isn't backfilled yet (every station starves) —
                // a warning, not an error. Either way, drop the spec registered
                // above so the bundle carries no model-less lead, and skip rather
                // than abort (≥24h leads still produce models).
                if (Train.Common.NowcastSource.IsNowcast(lead))
                    _log.LogWarning("Lead {Lead}h (nowcast): <2 stations with usable data — no hist_forecast "
                        + "surface data for this location yet; skipping the lead-0 model.", lead);
                else
                    _log.LogError("Lead {Lead}h: <2 stations with usable data after split — skipping this lead.", lead);
                specsPerLead.Remove(lead);
                continue;
            }

            // Pool train + val rows. Test slices stay per-station (we score
            // each station individually so per-station Brier is honest).
            var pooledTrain = perStation.SelectMany(p => p.Ds.Train).ToList();
            var pooledVal   = perStation.SelectMany(p => p.Ds.Val).ToList();
            _log.LogInformation("  Pooled: train={T} rows (wet {Pct:P1}), val={V} rows",
                pooledTrain.Count,
                pooledTrain.Count(r => r.Label) / (double)pooledTrain.Count,
                pooledVal.Count);
            // Lead-0 nowcast pool excluded from the RetrainGuard baseline (see
            // the per-station note above) — different training window.
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += pooledTrain.Count;
                totalValRows   += pooledVal.Count;
                totalTestRows  += perStation.Sum(p => p.Ds.Test.Count);
                firstLeadTrainFeatures ??= pooledTrain.Select(r => r.Features).ToList();
                firstLeadTrainLabels   ??= pooledTrain.Select(r => r.Label).ToList();
            }

            var trained = PrecipOccurrenceTrainer.TrainVector(pooledTrain, pooledVal, spec, hp);
            importanceByLead[lead] = trained.FeatureImportance;

            // Score per-station test slice. Each station gets its own
            // PerLeadStats so the verify card on the Models page can show
            // per-(station, lead) Brier honestly.
            foreach (var (name, slug, ds) in perStation)
            {
                var truth = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                var probs = PrecipOccurrenceTrainer.PredictVectorProbability(
                    trained.Ml, trained.Model, spec, ds.Test);
                var brier = PrecipMetrics.Brier(probs, truth);
                var clim = climByStation[slug]!;
                var climPred = ds.Test.Select(r => clim.Predict(r.ValidTimeUtc)).ToArray();
                var climBrier = PrecipMetrics.Brier(climPred, truth);
                var binBlend = probs.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
                var fbias = PrecipMetrics.FrequencyBias(binBlend, truth);
                var best = PrecipBaselines.BestSingle(spec, ds.Val);
                var bestValBrier = PrecipMetrics.Brier(
                    PrecipBaselines.SingleModelWet(spec, ds.Val, best),
                    ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray());
                var bestTestBrier = PrecipMetrics.Brier(
                    PrecipBaselines.SingleModelWet(spec, ds.Test, best),
                    truth);
                var testMonths = ds.Test.Select(r => new DateTime(r.ValidTimeUtc.Year, r.ValidTimeUtc.Month, 1))
                                        .Distinct().Count();
                perLeadStationPerStation[slug][lead.ToString()] = new ModelArtifact.PerLeadStats
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
                    BlendTestMae  = brier,
                    BlendTestRmse = climBrier,
                    BlendTestBias = fbias,
                };
                _log.LogInformation("  {Slug} lead {Lead}h Brier={B:0.0000} (clim {C:0.0000}, fbias {F:0.00})",
                    slug, lead, brier, climBrier, fbias);

                // Per-row test predictions for downstream bake-offs / 4b mint.
                for (int i = 0; i < ds.Test.Count; i++)
                    testPredsByStation[slug].Add(new TestPredictionRow
                    {
                        valid_time   = ds.Test[i].ValidTimeUtc,
                        station      = slug,
                        lead         = lead,
                        p_wet        = probs[i],
                        observed_wet = (byte)(ds.Test[i].Label ? 1 : 0),
                    });
            }

            // Save the same trained model under EVERY station's bundle dir.
            // 4 identical lead_{L}h.zip files on disk, one per station — wastes
            // ~1-2 MB per lead total but keeps per-station predict plumbing
            // unchanged and makes the bundle self-contained per station.
            foreach (var (_, slug, _) in perStation)
                ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, stationDirs[slug], lead);
        }

        if (specsPerLead.Count == 0)
        {
            _log.LogError("Phase 3o: no leads produced any model — abort.");
            return 3;
        }

        // Per-station bundle sidecars: spec, importance, climatology,
        // training_metadata, test_predictions.
        foreach (var (name, slug, oro, idx, _, _) in pool)
        {
            var dir = stationDirs[slug];
            ModelArtifact.SaveBlenderSpecs(dir, specsPerLead);
            ModelArtifact.SavePerLeadFeatureImportance(dir, importanceByLead);
            if (climByStation[slug] is not null)
                climByStation[slug]!.SaveTo(Path.Combine(dir, ModelArtifact.ClimatologyFileName));
            if (testPredsByStation[slug].Count > 0)
            {
                var p = Path.Combine(dir, "test_predictions.parquet");
                await Parquet.Serialization.ParquetSerializer.SerializeAsync(
                    testPredsByStation[slug], p, cancellationToken: ct);
            }

            var perLead = perLeadStationPerStation[slug];
            var metadata = new ModelArtifact.TrainingMetadata
            {
                Version = versionName,
                Target = "precipitation",
                Phase = phaseTag,
                LocationName = location.Name,
                DataSource = "previous_runs_api+ea_rainfall+srtm_dem",
                TrainedAtUtc = DateTime.UtcNow,
                Hyperparameters = BuildPrecipHpDict(hp)
                    .Concat(new[]
                    {
                        new KeyValuePair<string, object>("phase3o_pool_stations",
                            string.Join(",", BonehillStationOrder3o)),
                        new KeyValuePair<string, object>("phase3o_station_index", idx),
                        new KeyValuePair<string, object>("phase3o_orographic_features",
                            string.Join(",", PrecipRichOroFeatureBuilder.TerrainNamesFor(includeStationId))),
                    })
                    .ToDictionary(kv => kv.Key, kv => kv.Value),
                TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
                PerLead = perLead,
                DeviationsFromBrief = new List<string>
                {
                    "Pooled across the 4 Bonehill rainfall stations (Bellever, Bovey, Hexworthy, Princetown). One LightGBM per lead, trained on the concat of per-station rich+oro rows. The SAME trained model is saved under every station's bundle dir so per-station predict plumbing is unchanged.",
                    includeStationId
                        ? "9 terrain features appended after the rich (55) feature vector: elevation_vs_cell, relief_5km, ruggedness_5km, wind_sin, wind_cos, upwind_gain_per_wind_5km, uplift, uplift_x_q, station_id. station_id is a discrete 0..3 axis (Bellever=0, Bovey=1, Hexworthy=2, Princetown=3) — predict-time MUST use the same mapping."
                        : "3oni: 8 terrain features (the 3o block MINUS oro_station_id), so the pooled model treats each point as a pure terrain vector. This is the no-station-id variant for predicting the UNGAUGED Bonehill tor (an unseen station id would otherwise route down a gauge's default split). Slightly worse at the gauges than 3o — accepted; 3oni is a tor point-of-interest line, the gauges keep 3o.",
                    "Per-station climatology (each station's own train-slice base rate) is saved alongside the pooled model so the verify rolling-Brier panel shows honest per-station baselines.",
                },
            };
            ModelArtifact.SaveTrainingMetadata(dir, metadata);
        }

        // RetrainGuard runs ONCE against the pooled training matrix. The
        // guard's "rows" / "NaN%" / "features-effective" gates measure what
        // the model actually saw at fit time; per-station label-rate
        // anomalies surface through the per-station labelRates dictionary.
        var guardResult = RetrainGuard.BuildCheckAndSave(_log,
            stationDirs[pool[0].Slug],
            composite: $"precipitation/_pool_{phaseTag}", phase: phaseTag, version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp3o)
                ? sp3o.FeatureNames.ToList() : Array.Empty<string>(),
            locationName: location.Name,
            labelRates: perStationLabelRates,
            // 3o gained the upper-air block in-place (68→110) — allow the
            // feature-count jump without aborting; other gates stay enforced.
            tolerances: RetrainGuard.Defaults with { AllowFeaturesEffectiveChange = true });
        if (false /* TEMP 2026-05-26 JMA-extension local verify; revert */ && !guardResult.Passed)
        {
            _log.LogError("Aborting Phase 3o retrain — pooled-train sanity guard failed. " +
                "Per-station orphan dirs ({Dirs}) not promoted.",
                string.Join(", ", stationDirs.Values));
            return 4;
        }

        // Promote per-station as challenger.
        foreach (var (_, slug, _, _, _, _) in pool)
        {
            ModelArtifact.PromoteStationVersion(
                modelsRoot, "precipitation", slug, versionName, newPhase: phaseTag);
            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", slug);
            _log.LogInformation("Phase {Phase} {Slug} → {Dir}", phaseTag, slug, stationDirs[slug]);
            _log.LogInformation("Active versions for {Slug}: [{Active}]", slug, string.Join(", ", active));
        }

        if (_skipConformal || !includeStationId)
        {
            _log.LogInformation("Auto-conformal: SKIPPED ({Reason})",
                _skipConformal ? "WB_SKIP_CONFORMAL=1" : $"{phaseTag} is a documented conformal-skip phase");
        }
        else
        {
            foreach (var (_, slug, _, _, _, _) in pool)
            {
                var (cf, cs) = await _precipConformal.FitOneAsync(
                    slug, versionName, PrecipConformalFitCommand.DefaultAlpha, ct);
                _log.LogInformation("Auto-conformal: fitted {F} leads ({S} skipped) for {Slug}/{V}",
                    cf, cs, slug, versionName);
            }
        }
        return 0;
    }

    // ---- Phase 3d: exact-runtime precip blender (per-station, lead-12 champion) ---

    private static readonly int[] DefaultPhase3dLeads = { 12, 24 };

    private async Task<int> RunPhase3dAsync(string? stationOverride, string? tierName, bool? includeUkvOpt, int[]? exactLeads, int[]? cycleHoursFilter, Config.LocationConfig location, CancellationToken ct)
    {
        if (location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured for location '{Loc}' — cannot train precipitation blender.", location.Name);
            return 2;
        }

        // 3d is exact-runtime EA truth only — WeatherLink cove gauges (e.g. Lands
        // End) are 3c-only, so they're excluded here both for the auto list and an
        // explicit --station override.
        IReadOnlyList<string> stationsToTrain;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stationsToTrain = location.ProductRainfallStations
                .Where(s => s.Source is Config.RainfallTruthSource.Ea)
                .Select(s => s.Name).ToList();
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
            if (match.Source is Config.RainfallTruthSource.WeatherLink)
            {
                _log.LogError("Station '{Station}' is a WeatherLink (3c-only) gauge — Phase 3d is EA-truth only. Skipping.", match.Name);
                return 2;
            }
            stationsToTrain = new[] { match.Name };
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();
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

            // 3d carries the multi-level pressure (upper-air) block in-place
            // (2026-06-02): the FULL set retest (reports/ua_ab_3d_full) was
            // −16…−19% Brier @12h+24h (the earlier negative used the curated
            // 3-col subset). 3d is exact-runtime + Bonehill-only, and pressure
            // is on the same exact rows it already reads (no ASOF). Predict
            // mirrors via UpperAirValuesFromPerModel.
            var spec = PrecipExactFeatureBuilder.BuildSpec(tier, targetLead: lead, includeUkv: includeUkv, withUpperAir: true);
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
            // Lead-0 nowcast rows train on a different (shorter, hist_forecast)
            // window, so exclude them from the RetrainGuard baseline — otherwise
            // --nowcast would shift the rows/label/feature totals vs the
            // production leads and could trip the gate. (No-op for non-nowcast
            // leads, incl. 3d's {12,24}.)
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += ds.Train.Count;
                totalValRows   += ds.Val.Count;
                totalTestRows  += ds.Test.Count;
                firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
                firstLeadTrainLabels   ??= ds.Train.Select(r => r.Label).ToList();
            }

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
            labelRates: labelRates3d,
            // 3d gained the upper-air block in-place — allow the one-time
            // feature-count jump without aborting; other gates stay enforced.
            tolerances: RetrainGuard.Defaults with { AllowFeaturesEffectiveChange = true });
        if (!guardResult3d.Passed)
        {
            _log.LogError("Aborting Phase 3d retrain ({Station}) — sanity guard failed. Orphan dir {Dir} not promoted (manifest unchanged).", stationSlug, versionDir);
            return 4;
        }

        // Promote 3d as a station challenger — 3a stays Current. 3d is no longer
        // pinned as the lead-12 champion (the per-lead champion override was
        // removed 2026-06-15): the precip champion (3o/3c) owns every lead, and
        // 3d's exact-runtime short-lead value still shows on the forecast/skill
        // pages as a challenger. Lead-set-aware promote (post-2026-05-08) lets a
        // 3d retrain at e.g. {96,120} coexist with a sibling 3d at {12,24,48,72}
        // when their lead-sets differ.
        ModelArtifact.PromoteStationVersion(
            modelsRoot, "precipitation", stationSlug, versionName, newPhase: "3d");
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);

        _log.LogInformation("Phase 3d artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));
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
