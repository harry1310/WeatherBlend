using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Oro;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3c-oro bake-off — three-way comparison + SHAP.
///
/// Arms scored on identical per-station test rows:
///   1. <b>per-station 3c</b> — one LightGBM per station, rich (59) features.
///   2. <b>pooled rich</b>    — one LightGBM per lead trained on 7-station stack,
///                              rich (59) features. Decomposes "pooled helped" from
///                              "terrain helped" vs arm 3.
///   3. <b>pooled rich+oro</b> — one LightGBM per lead trained on 7-station stack,
///                              rich + 9 terrain features (= 68).
///
/// Side effects beyond the markdown report:
///   - Saves pooled bundles to <c>data/models/precipitation/_pooled_rich/{ts}/</c>
///     and <c>data/models/precipitation/_pooled_oro/{ts}/</c> via the standard
///     <see cref="ModelArtifact.SaveLeadModel"/> contract so SHAP / future bake-offs
///     can re-use them.
///   - Writes per-cell intermediate JSONL to <c>data/reports/phase3c_oro_bakeoff_running.jsonl</c>
///     line-by-line as each (lead, station) result lands. If the process is killed
///     mid-run (laptop sleep, etc.), the JSONL captures partial progress.
///   - Runs ML.NET FeatureContribution (TreeSHAP-approximate) on the pooled-oro
///     model at lead 24h, logging top-feature mean |contribution| as a SHAP
///     proxy. Helps drive the interaction-feature design for a follow-up bake-off.
/// </summary>
public sealed class Phase3cOroBakeoffCommand
{
    private static readonly int[] Leads = { 24, 48, 72 };

    private readonly AppConfig _cfg;
    private readonly ILogger<Phase3cOroBakeoffCommand> _log;

    public Phase3cOroBakeoffCommand(AppConfig cfg, ILogger<Phase3cOroBakeoffCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var oroRoot = Path.Combine(
            Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic");
        var oroBySlug = OroStaticFeatures.LoadAll(oroRoot);
        _log.LogInformation("Loaded {N} orographic static records from {Path}",
            oroBySlug.Count, oroRoot);

        // Materialise (location, station_name, station_slug, oro, station_index)
        // tuples for every configured station that has an orographic record on
        // disk. Station_index is assigned in iteration order — Bonehill location
        // first (indices 0..3), Membury second (4..6) — so an integer split at
        // station_id ≥ 4 cleanly separates the two NWP cells.
        var stations = new List<(LocationConfig Loc, string Name, string Slug, OroStaticFeatures Oro, int Index)>();
        int nextIndex = 0;
        foreach (var loc in _cfg.Locations)
        {
            foreach (var s in loc.Rainfall.Stations)
            {
                var slug = StationSlug.WithEaPrefix(s.Name);
                if (!oroBySlug.TryGetValue(slug, out var oro))
                {
                    _log.LogWarning("No orographic record for {Slug} — skipping. " +
                        "Run scripts/OrographicFeatures/build_static_orographic.py.", slug);
                    continue;
                }
                stations.Add((loc, s.Name, slug, oro, nextIndex++));
            }
        }
        if (stations.Count < 2)
        {
            _log.LogError("Need at least 2 stations with orographic records for the bake-off; got {N}.", stations.Count);
            return 2;
        }
        _log.LogInformation("Bake-off pool: {N} stations across {LocN} locations",
            stations.Count, stations.Select(s => s.Loc.Name).Distinct().Count());

        // Resilience: per-cell JSONL written line-by-line. Truncate at the start
        // of each run so the file matches THIS run's progress.
        var runningJsonl = Path.Combine(_cfg.Storage.ReportsPath,
            "phase3c_oro_bakeoff_running.jsonl");
        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        await File.WriteAllTextAsync(runningJsonl, "", ct);

        // Bundle output directories — one per pooled arm. Use timestamps so
        // multiple bake-off runs in the same day don't collide.
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var pooledRichDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", "_pooled_rich", now);
        var pooledOroDir  = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", "_pooled_oro",  now);
        Directory.CreateDirectory(pooledRichDir);
        Directory.CreateDirectory(pooledOroDir);
        _log.LogInformation("Bundle save dirs:");
        _log.LogInformation("  pooled-rich : {Path}", pooledRichDir);
        _log.LogInformation("  pooled-oro  : {Path}", pooledOroDir);

        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        var perLeadStation = new List<BakeoffResult>();
        // Per-lead SHAP rankings — only computed at lead 24h to save time.
        var shapByLead = new Dictionary<int, IReadOnlyList<(string Name, double MeanAbsContribution)>>();

        // 4th arm: pooled-oro-v2 (rich + v1 terrain + 14 v2 DEM aggregations).
        var runV2Arm = true;
        var pooledV2Dir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", "_pooled_oro_v2", now);
        if (runV2Arm)
        {
            Directory.CreateDirectory(pooledV2Dir);
            _log.LogInformation("  pooled-oro-v2 : {Path}  (+14 DEM aggregations)", pooledV2Dir);
        }

        // 5th arm: pooled-oro-v3 (v2 + 6 atmospheric climatology features).
        // Gated on the static record actually having a climatology block —
        // older bundles won't, so v3 silently degrades to a v2 run.
        var runV3Arm = stations.Any(s => s.Oro.ClimatologyBySectorMonth.Count > 0);
        var pooledV3Dir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", "_pooled_oro_v3", now);
        if (runV3Arm)
        {
            Directory.CreateDirectory(pooledV3Dir);
            _log.LogInformation("  pooled-oro-v3 : {Path}  (+6 climatology features)", pooledV3Dir);
        }

        foreach (var lead in Leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("===== Lead {Lead}h =====", lead);

            var richSpec    = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            var richOroSpec = PrecipRichOroFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            var richOroV2Spec = runV2Arm
                ? PrecipRichOroV2FeatureBuilder.BuildSpec(_cfg.Blenders, lead)
                : null;
            var richOroV3Spec = runV3Arm
                ? PrecipRichOroV3FeatureBuilder.BuildSpec(_cfg.Blenders, lead)
                : null;
            _log.LogInformation("Rich spec: {N} features. Rich-oro spec: {M} features.{Extra}",
                richSpec.FeatureCount, richOroSpec.FeatureCount,
                (runV2Arm ? $" Rich-oro-v2 spec: {richOroV2Spec!.FeatureCount} features." : "")
              + (runV3Arm ? $" Rich-oro-v3 spec: {richOroV3Spec!.FeatureCount} features." : ""));

            // Build per-station datasets.
            var perStation = new List<StationDataset>();
            foreach (var st in stations)
            {
                ct.ThrowIfCancellationRequested();
                var rich = PrecipRichFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    st.Loc.Name, st.Name, richSpec, minValidTime: null, ct);
                var richOro = PrecipRichOroFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    st.Loc.Name, st.Name, st.Oro, st.Index, richOroSpec, ct);
                var richOroV2 = runV2Arm
                    ? PrecipRichOroV2FeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                        st.Loc.Name, st.Name, st.Oro, st.Index, richOroV2Spec!, ct)
                    : null;
                var richOroV3 = runV3Arm
                    ? PrecipRichOroV3FeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                        st.Loc.Name, st.Name, st.Oro, st.Index, richOroV3Spec!, ct)
                    : null;

                if (rich.Count < 50 || richOro.Count < 50)
                {
                    _log.LogWarning("Station {Slug} lead {Lead}h: only {RR} rich / {RO} rich-oro rows — skipping.",
                        st.Slug, lead, rich.Count, richOro.Count);
                    continue;
                }

                var dsRich    = BinaryDataset.Split(rich);
                var dsRichOro = BinaryDataset.Split(richOro);
                var dsRichOroV2 = richOroV2 is not null ? BinaryDataset.Split(richOroV2) : null;
                var dsRichOroV3 = richOroV3 is not null ? BinaryDataset.Split(richOroV3) : null;
                _log.LogInformation(
                    "  {Slug}: rich rows={RR} (wet {RW:P0}) train={TR}/val={V}/test={E}; rich-oro rows={RO}{ExtraV2}{ExtraV3}",
                    st.Slug, rich.Count,
                    rich.Count == 0 ? 0 : (double)rich.Count(r => r.Label) / rich.Count,
                    dsRich.Train.Count, dsRich.Val.Count, dsRich.Test.Count,
                    richOro.Count,
                    runV2Arm ? $"; rich-oro-v2 rows={richOroV2!.Count}" : "",
                    runV3Arm ? $"; rich-oro-v3 rows={richOroV3!.Count}" : "");

                perStation.Add(new StationDataset(st.Slug, st.Loc.Name, dsRich, dsRichOro, dsRichOroV2, dsRichOroV3));
            }

            if (perStation.Count < 2)
            {
                _log.LogWarning("Lead {Lead}h: <2 stations with usable data; skipping lead.", lead);
                continue;
            }

            // ARM 1: per-station 3c baseline (rich, one LightGBM per station).
            _log.LogInformation("--- Arm 1: per-station 3c baseline ({N} models) ---", perStation.Count);
            var baselineByStation = new Dictionary<string, PrecipOccurrenceTrainer.TrainedClassifier>();
            foreach (var ps in perStation)
            {
                ct.ThrowIfCancellationRequested();
                var trained = PrecipOccurrenceTrainer.TrainVector(
                    ps.Rich.Train, ps.Rich.Val, richSpec, hp);
                baselineByStation[ps.Slug] = trained;
            }

            // ARM 2: pooled rich (no terrain) — stacked 7-station train rows, rich features only.
            _log.LogInformation("--- Arm 2: pooled rich (no terrain) on {N} stations ---", perStation.Count);
            var pooledRichTrain = perStation.SelectMany(ps => ps.Rich.Train).ToList();
            var pooledRichVal   = perStation.SelectMany(ps => ps.Rich.Val).ToList();
            _log.LogInformation("  pooled-rich train rows={N} (wet {Pct:P1}); val rows={V}",
                pooledRichTrain.Count,
                pooledRichTrain.Count(r => r.Label) / (double)pooledRichTrain.Count,
                pooledRichVal.Count);
            var pooledRich = PrecipOccurrenceTrainer.TrainVector(pooledRichTrain, pooledRichVal, richSpec, hp);
            ModelArtifact.SaveLeadModel(pooledRich.Ml, pooledRich.Model, pooledRich.InputSchema, pooledRichDir, lead);

            // ARM 3: pooled rich + 9 terrain features — stacked 7-station train rows.
            _log.LogInformation("--- Arm 3: pooled rich-oro on {N} stations ---", perStation.Count);
            var pooledOroTrain = perStation.SelectMany(ps => ps.RichOro.Train).ToList();
            var pooledOroVal   = perStation.SelectMany(ps => ps.RichOro.Val).ToList();
            _log.LogInformation("  pooled-oro train rows={N} (wet {Pct:P1}); val rows={V}",
                pooledOroTrain.Count,
                pooledOroTrain.Count(r => r.Label) / (double)pooledOroTrain.Count,
                pooledOroVal.Count);
            var pooledOro = PrecipOccurrenceTrainer.TrainVector(pooledOroTrain, pooledOroVal, richOroSpec, hp);
            ModelArtifact.SaveLeadModel(pooledOro.Ml, pooledOro.Model, pooledOro.InputSchema, pooledOroDir, lead);

            // ARM 4: pooled rich + v1 terrain + 14 v2 DEM aggregations.
            PrecipOccurrenceTrainer.TrainedClassifier? pooledV2 = null;
            if (runV2Arm)
            {
                _log.LogInformation("--- Arm 4: pooled rich-oro-v2 on {N} stations (+14 DEM aggs) ---",
                    perStation.Count);
                var pooledV2Train = perStation.SelectMany(ps => ps.RichOroV2!.Train).ToList();
                var pooledV2Val   = perStation.SelectMany(ps => ps.RichOroV2!.Val).ToList();
                pooledV2 = PrecipOccurrenceTrainer.TrainVector(pooledV2Train, pooledV2Val, richOroV2Spec!, hp);
                ModelArtifact.SaveLeadModel(pooledV2.Ml, pooledV2.Model, pooledV2.InputSchema, pooledV2Dir, lead);
            }

            // ARM 5: pooled v2 + 6 atmospheric climatology features.
            PrecipOccurrenceTrainer.TrainedClassifier? pooledV3 = null;
            if (runV3Arm)
            {
                _log.LogInformation("--- Arm 5: pooled rich-oro-v3 on {N} stations (+6 climatology) ---",
                    perStation.Count);
                var pooledV3Train = perStation.SelectMany(ps => ps.RichOroV3!.Train).ToList();
                var pooledV3Val   = perStation.SelectMany(ps => ps.RichOroV3!.Val).ToList();
                pooledV3 = PrecipOccurrenceTrainer.TrainVector(pooledV3Train, pooledV3Val, richOroV3Spec!, hp);
                ModelArtifact.SaveLeadModel(pooledV3.Ml, pooledV3.Model, pooledV3.InputSchema, pooledV3Dir, lead);
            }

            // Score per-station on test (identical test rows per station for all 3 arms).
            foreach (var ps in perStation)
            {
                ct.ThrowIfCancellationRequested();
                var truth = ps.Rich.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();

                var baseProb = PrecipOccurrenceTrainer.PredictVectorProbability(
                    baselineByStation[ps.Slug].Ml, baselineByStation[ps.Slug].Model,
                    richSpec, ps.Rich.Test);
                var baseBrier = PrecipMetrics.Brier(baseProb, truth);

                var pooledRichProb = PrecipOccurrenceTrainer.PredictVectorProbability(
                    pooledRich.Ml, pooledRich.Model, richSpec, ps.Rich.Test);
                var pooledRichBrier = PrecipMetrics.Brier(pooledRichProb, truth);

                var oroProb = PrecipOccurrenceTrainer.PredictVectorProbability(
                    pooledOro.Ml, pooledOro.Model, richOroSpec, ps.RichOro.Test);
                var oroTruth = ps.RichOro.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                var oroBrier = PrecipMetrics.Brier(oroProb, oroTruth);

                double v2Brier = double.NaN;
                if (pooledV2 is not null && ps.RichOroV2 is not null)
                {
                    var v2Prob = PrecipOccurrenceTrainer.PredictVectorProbability(
                        pooledV2.Ml, pooledV2.Model, richOroV2Spec!, ps.RichOroV2.Test);
                    var v2Truth = ps.RichOroV2.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                    v2Brier = PrecipMetrics.Brier(v2Prob, v2Truth);
                }
                double v3Brier = double.NaN;
                if (pooledV3 is not null && ps.RichOroV3 is not null)
                {
                    var v3Prob = PrecipOccurrenceTrainer.PredictVectorProbability(
                        pooledV3.Ml, pooledV3.Model, richOroV3Spec!, ps.RichOroV3.Test);
                    var v3Truth = ps.RichOroV3.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                    v3Brier = PrecipMetrics.Brier(v3Prob, v3Truth);
                }

                // Climatology per station (base rate over its train rows).
                var baseRate = ps.Rich.Train.Count(r => r.Label) / (double)ps.Rich.Train.Count;
                var climProb = new double[truth.Length];
                for (int i = 0; i < climProb.Length; i++) climProb[i] = baseRate;
                var climBrier = PrecipMetrics.Brier(climProb, truth);

                var deltaPooledPct = baseBrier > 0 ? (pooledRichBrier - baseBrier) / baseBrier * 100 : double.NaN;
                var deltaOroPct    = baseBrier > 0 ? (oroBrier        - baseBrier) / baseBrier * 100 : double.NaN;
                var deltaV2Pct     = baseBrier > 0 && !double.IsNaN(v2Brier)
                    ? (v2Brier - baseBrier) / baseBrier * 100 : double.NaN;
                var deltaV3Pct     = baseBrier > 0 && !double.IsNaN(v3Brier)
                    ? (v3Brier - baseBrier) / baseBrier * 100 : double.NaN;
                _log.LogInformation(
                    "  {Slug} lead {Lead}h: base={B:0.0000}, pool-rich={PR:0.0000} (Δ {PD:+0.0;-0.0;0.0}%), pool-oro={O:0.0000} (Δ {OD:+0.0;-0.0;0.0}%){V2Str}{V3Str}  clim={C:0.0000} n_test={N}",
                    ps.Slug, lead, baseBrier, pooledRichBrier, deltaPooledPct, oroBrier, deltaOroPct,
                    runV2Arm ? $", pool-v2={v2Brier:0.0000} (Δ {deltaV2Pct:+0.0;-0.0;0.0}%)" : "",
                    runV3Arm ? $", pool-v3={v3Brier:0.0000} (Δ {deltaV3Pct:+0.0;-0.0;0.0}%)" : "",
                    climBrier, truth.Length);

                var row = new BakeoffResult(
                    lead, ps.Slug, ps.LocationName, truth.Length,
                    BaseBrier: baseBrier, PooledRichBrier: pooledRichBrier, OroBrier: oroBrier,
                    V2Brier: v2Brier, V3Brier: v3Brier,
                    ClimBrier: climBrier, BaseRate: baseRate);
                perLeadStation.Add(row);

                // Resilient intermediate write: append one JSON line per cell.
                await File.AppendAllTextAsync(runningJsonl,
                    JsonSerializer.Serialize(row) + "\n", ct);
            }

            // Top terrain-feature importance (gain-based, fast).
            var oroImp = pooledOro.FeatureImportance
                .Where(t => t.Name.StartsWith("oro_", StringComparison.Ordinal))
                .OrderByDescending(t => t.Gain).Take(9).ToArray();
            _log.LogInformation("  Pooled-oro terrain feature importance (gain):");
            foreach (var (n, g) in oroImp)
                _log.LogInformation("    {Name}: gain={Gain:0.000}", n, g);

            // SHAP at lead 24h only (most production-relevant + computationally
            // bounded). Uses ML.NET's CalculateFeatureContribution (TreeSHAP-approx)
            // over a sampled subset of test rows for speed.
            if (lead == 24)
            {
                _log.LogInformation("  Computing SHAP (lead 24h, pooled-oro)...");
                var sampleTest = perStation.SelectMany(ps => ps.RichOro.Test)
                    .OrderBy(_ => Guid.NewGuid())
                    .Take(8000)
                    .ToList();
                var shapRanking = ComputeShapRanking(pooledOro, richOroSpec, sampleTest);
                shapByLead[lead] = shapRanking;
                _log.LogInformation("  SHAP top 20 features by mean |contribution| (n={N} test rows):", sampleTest.Count);
                foreach (var (n, c) in shapRanking.Take(20))
                    _log.LogInformation("    {Name:<40}  mean_abs_contrib={C:0.0000}", n, c);
            }
        }

        if (perLeadStation.Count == 0)
        {
            _log.LogError("No bake-off rows produced — abort.");
            return 3;
        }

        var reportPath = WriteReport(perLeadStation, stations, shapByLead, pooledRichDir, pooledOroDir,
            runV2Arm ? pooledV2Dir : null,
            runV3Arm ? pooledV3Dir : null);
        _log.LogInformation("Wrote {Path}", reportPath);
        _log.LogInformation("Saved bundles: pooled-rich={PR}, pooled-oro={PO}{V2}{V3}",
            pooledRichDir, pooledOroDir,
            runV2Arm ? $", pooled-v2={pooledV2Dir}" : "",
            runV3Arm ? $", pooled-v3={pooledV3Dir}" : "");
        return 0;
    }

    // -----------------------------------------------------------------------
    // SHAP (TreeSHAP-approximate via ML.NET FeatureContribution)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute mean |contribution| per feature over <paramref name="testRows"/>.
    /// Sorted descending — top of list = most influential features on the model's
    /// raw score. For driving interaction-feature design.
    /// </summary>
    private static IReadOnlyList<(string Name, double MeanAbsContribution)> ComputeShapRanking(
        PrecipOccurrenceTrainer.TrainedClassifier trained,
        BlenderSpec spec,
        IReadOnlyList<BinaryTrainingRow> testRows)
    {
        var predictor = ResolveBinaryPredictor(trained.Model);
        var schemaDef = SchemaDefinition.Create(typeof(BinaryTrainingRow));
        schemaDef[nameof(BinaryTrainingRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);
        var rawDv = trained.Ml.Data.LoadFromEnumerable(testRows, schemaDef);

        var contribEst = trained.Ml.Transforms.CalculateFeatureContribution(
            predictor,
            numberOfPositiveContributions: spec.FeatureCount,
            numberOfNegativeContributions: spec.FeatureCount,
            normalize: false);
        var fitted = contribEst.Fit(rawDv);
        var withContrib = fitted.Transform(rawDv);

        var sumAbs = new double[spec.FeatureCount];
        var rowCount = 0;
        var col = withContrib.Schema["FeatureContributions"];
        using (var cursor = withContrib.GetRowCursor(new[] { col }))
        {
            var getter = cursor.GetGetter<VBuffer<float>>(col);
            VBuffer<float> buf = default;
            while (cursor.MoveNext())
            {
                getter(ref buf);
                var dense = buf.DenseValues().ToArray();
                for (int i = 0; i < spec.FeatureCount && i < dense.Length; i++)
                    sumAbs[i] += Math.Abs(dense[i]);
                rowCount++;
            }
        }
        if (rowCount == 0) return Array.Empty<(string, double)>();

        var ranking = new List<(string, double)>(spec.FeatureCount);
        for (int i = 0; i < spec.FeatureCount; i++)
            ranking.Add((spec.FeatureNames[i], sumAbs[i] / rowCount));
        ranking.Sort((a, b) => b.Item2.CompareTo(a.Item2));
        return ranking;
    }

    private static BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>
        ResolveBinaryPredictor(ITransformer loaded)
    {
        if (loaded is BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>> bare)
            return bare;
        if (loaded is IEnumerable<ITransformer> chain)
        {
            var items = chain.ToArray();
            if (items.Length > 0
                && items[^1] is BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>> tail)
                return tail;
        }
        throw new InvalidOperationException(
            $"Loaded model is not a calibrated LightGbm binary predictor: {loaded.GetType().FullName}");
    }

    // -----------------------------------------------------------------------
    // Markdown report
    // -----------------------------------------------------------------------

    private string WriteReport(
        IReadOnlyList<BakeoffResult> results,
        IReadOnlyList<(LocationConfig Loc, string Name, string Slug, OroStaticFeatures Oro, int Index)> stations,
        IReadOnlyDictionary<int, IReadOnlyList<(string Name, double MeanAbsContribution)>> shapByLead,
        string pooledRichBundleDir,
        string pooledOroBundleDir,
        string? pooledV2BundleDir,
        string? pooledV3BundleDir)
    {
        var hasV2 = pooledV2BundleDir is not null
            && results.Any(r => !double.IsNaN(r.V2Brier));
        var hasV3 = pooledV3BundleDir is not null
            && results.Any(r => !double.IsNaN(r.V3Brier));
        var dir = _cfg.Storage.ReportsPath;
        Directory.CreateDirectory(dir);
        var variantTag = hasV3 ? "v6" : (hasV2 ? "v5" : "v3");
        var path = Path.Combine(dir, $"phase3c_oro_bakeoff_{variantTag}_{DateTime.UtcNow:yyyy-MM-dd}.md");

        var sb = new StringBuilder();
        sb.AppendLine($"# Phase 3c-oro bake-off {variantTag} — {(hasV2 ? "4-way" : "3-way")} decomposition");
        sb.AppendLine();
        sb.AppendLine($"Generated {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC.");
        sb.AppendLine();
        sb.AppendLine("Scope: precipitation P(wet ≥ 0.1 mm/h), leads 24/48/72 h. Pool = "
                    + $"{stations.Count} EA gauges across {stations.Select(s => s.Loc.Name).Distinct().Count()} "
                    + "locations.");
        sb.AppendLine();
        sb.AppendLine("**Arms**:");
        sb.AppendLine();
        sb.AppendLine("- **3c (per-station)** — one LightGBM per station, rich (59) features.");
        sb.AppendLine("- **pooled-rich** — one LightGBM per lead trained on stacked 7-station train rows, rich (59) features only. *Decomposes \"pooled helped\" from \"terrain helped\".*");
        sb.AppendLine("- **pooled-oro** — one LightGBM per lead trained on stacked 7-station train rows, rich + 9 v1 terrain features (68).");
        if (hasV2)
            sb.AppendLine($"- **pooled-oro-v2** — pooled-oro + {PrecipRichOroV2FeatureBuilder.V2TerrainFeatureCount} new DEM aggregations (82): TPI×4 scales, lee×8 sectors, mean_slope_5km, aspect_dominance_5km.");
        if (hasV3)
            sb.AppendLine($"- **pooled-oro-v3** — pooled-oro-v2 + {PrecipRichOroV3FeatureBuilder.ClimatologyFeatureCount} atmospheric climatology features (88): lapse/q/wind/shear/thickness looked up by (NWP wind sector × month) from 4-year GFS pressure-level history.");
        sb.AppendLine();
        sb.AppendLine("Same time-fraction split per station (70/15/15); test rows identical between arms.");
        sb.AppendLine();
        sb.AppendLine($"Saved bundles: `{pooledRichBundleDir}` / `{pooledOroBundleDir}`"
                    + (hasV2 ? $" / `{pooledV2BundleDir}`" : "")
                    + (hasV3 ? $" / `{pooledV3BundleDir}`" : "") + ".");
        sb.AppendLine();

        sb.AppendLine("## Per-station Brier comparison");
        sb.AppendLine();
        var v2Col = hasV2 ? " pooled-v2 | Δ v2 |" : "";
        var v3Col = hasV3 ? " pooled-v3 | Δ v3 |" : "";
        var v2Dash = hasV2 ? "---:|---:|" : "";
        var v3Dash = hasV3 ? "---:|---:|" : "";
        sb.AppendLine($"| Station | Lead | n_test | base 3c | pooled-rich | Δ pool | pooled-oro | Δ oro |{v2Col}{v3Col}");
        sb.AppendLine($"|---|---:|---:|---:|---:|---:|---:|---:|{v2Dash}{v3Dash}");
        foreach (var r in results.OrderBy(r => r.Slug).ThenBy(r => r.Lead))
        {
            var dPool = r.BaseBrier > 0 ? (r.PooledRichBrier - r.BaseBrier) / r.BaseBrier * 100 : double.NaN;
            var dOro  = r.BaseBrier > 0 ? (r.OroBrier        - r.BaseBrier) / r.BaseBrier * 100 : double.NaN;
            var dV2   = r.BaseBrier > 0 && !double.IsNaN(r.V2Brier)
                ? (r.V2Brier - r.BaseBrier) / r.BaseBrier * 100 : double.NaN;
            var dV3   = r.BaseBrier > 0 && !double.IsNaN(r.V3Brier)
                ? (r.V3Brier - r.BaseBrier) / r.BaseBrier * 100 : double.NaN;
            var line = string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1} | {2} | {3:0.0000} | {4:0.0000} | {5:+0.0;-0.0;0.0}% | {6:0.0000} | {7:+0.0;-0.0;0.0}% |",
                r.Slug, r.Lead, r.NTest, r.BaseBrier, r.PooledRichBrier, dPool, r.OroBrier, dOro);
            if (hasV2)
                line += string.Format(CultureInfo.InvariantCulture, " {0:0.0000} | {1:+0.0;-0.0;0.0}% |",
                    r.V2Brier, dV2);
            if (hasV3)
                line += string.Format(CultureInfo.InvariantCulture, " {0:0.0000} | {1:+0.0;-0.0;0.0}% |",
                    r.V3Brier, dV3);
            sb.AppendLine(line);
        }

        sb.AppendLine();
        sb.AppendLine("## Aggregate per lead (mean across stations)");
        sb.AppendLine();
        sb.AppendLine($"| Lead | n stations | mean 3c | mean pooled-rich | Δ pool | mean pooled-oro | Δ oro |{(hasV2 ? " mean pooled-v2 | Δ v2 |" : "")}{(hasV3 ? " mean pooled-v3 | Δ v3 |" : "")}");
        sb.AppendLine($"|---:|---:|---:|---:|---:|---:|---:|{(hasV2 ? "---:|---:|" : "")}{(hasV3 ? "---:|---:|" : "")}");
        foreach (var lead in results.Select(r => r.Lead).Distinct().OrderBy(l => l))
        {
            var slice = results.Where(r => r.Lead == lead).ToList();
            var mb = slice.Average(r => r.BaseBrier);
            var mp = slice.Average(r => r.PooledRichBrier);
            var mo = slice.Average(r => r.OroBrier);
            var dp = mb > 0 ? (mp - mb) / mb * 100 : double.NaN;
            var dr = mb > 0 ? (mo - mb) / mb * 100 : double.NaN;
            var line = string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1} | {2:0.0000} | {3:0.0000} | {4:+0.0;-0.0;0.0}% | {5:0.0000} | {6:+0.0;-0.0;0.0}% |",
                lead, slice.Count, mb, mp, dp, mo, dr);
            if (hasV2)
            {
                var v2Slice = slice.Where(r => !double.IsNaN(r.V2Brier)).ToList();
                if (v2Slice.Count > 0)
                {
                    var mi = v2Slice.Average(r => r.V2Brier);
                    var di = mb > 0 ? (mi - mb) / mb * 100 : double.NaN;
                    line += string.Format(CultureInfo.InvariantCulture, " {0:0.0000} | {1:+0.0;-0.0;0.0}% |", mi, di);
                }
                else line += " -- | -- |";
            }
            if (hasV3)
            {
                var v3Slice = slice.Where(r => !double.IsNaN(r.V3Brier)).ToList();
                if (v3Slice.Count > 0)
                {
                    var mi = v3Slice.Average(r => r.V3Brier);
                    var di = mb > 0 ? (mi - mb) / mb * 100 : double.NaN;
                    line += string.Format(CultureInfo.InvariantCulture, " {0:0.0000} | {1:+0.0;-0.0;0.0}% |", mi, di);
                }
                else line += " -- | -- |";
            }
            sb.AppendLine(line);
        }

        sb.AppendLine();
        sb.AppendLine("## Aggregate per Bonehill-cell only (Bellever / Bovey / Hexworthy / Princetown)");
        sb.AppendLine();
        var bonehillCellSlugs = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy", "ea_princetown",
        };
        sb.AppendLine($"| Lead | n stations | mean 3c | mean pooled-rich | Δ pool | mean pooled-oro | Δ oro |{(hasV2 ? " mean pooled-v2 | Δ v2 |" : "")}{(hasV3 ? " mean pooled-v3 | Δ v3 |" : "")}");
        sb.AppendLine($"|---:|---:|---:|---:|---:|---:|---:|{(hasV2 ? "---:|---:|" : "")}{(hasV3 ? "---:|---:|" : "")}");
        foreach (var lead in results.Select(r => r.Lead).Distinct().OrderBy(l => l))
        {
            var slice = results.Where(r => r.Lead == lead && bonehillCellSlugs.Contains(r.Slug)).ToList();
            if (slice.Count == 0) continue;
            var mb = slice.Average(r => r.BaseBrier);
            var mp = slice.Average(r => r.PooledRichBrier);
            var mo = slice.Average(r => r.OroBrier);
            var dp = mb > 0 ? (mp - mb) / mb * 100 : double.NaN;
            var dr = mb > 0 ? (mo - mb) / mb * 100 : double.NaN;
            var line = string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1} | {2:0.0000} | {3:0.0000} | {4:+0.0;-0.0;0.0}% | {5:0.0000} | {6:+0.0;-0.0;0.0}% |",
                lead, slice.Count, mb, mp, dp, mo, dr);
            if (hasV2)
            {
                var v2Slice = slice.Where(r => !double.IsNaN(r.V2Brier)).ToList();
                if (v2Slice.Count > 0)
                {
                    var mi = v2Slice.Average(r => r.V2Brier);
                    var di = mb > 0 ? (mi - mb) / mb * 100 : double.NaN;
                    line += string.Format(CultureInfo.InvariantCulture, " {0:0.0000} | {1:+0.0;-0.0;0.0}% |", mi, di);
                }
                else line += " -- | -- |";
            }
            if (hasV3)
            {
                var v3Slice = slice.Where(r => !double.IsNaN(r.V3Brier)).ToList();
                if (v3Slice.Count > 0)
                {
                    var mi = v3Slice.Average(r => r.V3Brier);
                    var di = mb > 0 ? (mi - mb) / mb * 100 : double.NaN;
                    line += string.Format(CultureInfo.InvariantCulture, " {0:0.0000} | {1:+0.0;-0.0;0.0}% |", mi, di);
                }
                else line += " -- | -- |";
            }
            sb.AppendLine(line);
        }

        sb.AppendLine();
        sb.AppendLine("## SHAP — pooled-oro at lead 24h (top 20 features by mean |contribution|)");
        sb.AppendLine();
        if (shapByLead.TryGetValue(24, out var shap24))
        {
            sb.AppendLine("| Rank | Feature | mean |contribution| |");
            sb.AppendLine("|---:|---|---:|");
            for (int i = 0; i < Math.Min(20, shap24.Count); i++)
            {
                var (n, c) = shap24[i];
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                    "| {0} | {1} | {2:0.0000} |", i + 1, n, c));
            }
        }
        else
        {
            sb.AppendLine("(SHAP not computed)");
        }

        sb.AppendLine();
        sb.AppendLine("## Decision criteria (see plan doc)");
        sb.AppendLine();
        sb.AppendLine("- ≥2 % Brier improvement at 24 h AND ≥1 % at 48 + 72 h, averaged across the three Bonehill-cell gauges (Bellever / Bovey Tracey / Hexworthy).");
        sb.AppendLine("- ≥3 of 6 dynamic orographic features non-dead per SHAP — table above.");
        sb.AppendLine("- Terrain-feature SHAP signs physically coherent (uplift × q positive, etc.).");

        File.WriteAllText(path, sb.ToString());
        return path;
    }

    private sealed record StationDataset(
        string Slug, string LocationName,
        BinaryDataset Rich, BinaryDataset RichOro,
        BinaryDataset? RichOroV2, BinaryDataset? RichOroV3);

    private sealed record BakeoffResult(
        int Lead, string Slug, string LocationName, int NTest,
        double BaseBrier, double PooledRichBrier, double OroBrier,
        double V2Brier, double V3Brier,
        double ClimBrier, double BaseRate);
}
