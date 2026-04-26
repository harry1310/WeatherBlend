using Microsoft.Extensions.Logging;
using WeatherBlend.Evaluate;

namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Generic per-target training pipeline used by every element blender. Owns:
///   - per-lead loop
///   - chronological 70/15/15 split
///   - LightGBM fit (via <see cref="TemperatureTrainer.Train{T}"/>)
///   - best-single-per-lead computation (val-MAE pick)
///   - artefact save (model.zip per lead, feature_schema, importance, metadata)
///   - MANIFEST update
///
/// Per-element specifics (row class, feature columns, model accessors) are
/// injected via <see cref="ElementTrainerInputs{T}"/> so the harness stays
/// agnostic of which physical variable is being blended.
/// </summary>
public static class ElementTrainerHarness
{
    /// <summary>One per-model accessor: (CLI model id, function pulling the per-row value).</summary>
    public sealed record ModelAccessor<T>(string ModelId, Func<T, float> Get);

    public sealed record ElementTrainerInputs<T>(
        ElementTarget Target,
        TemperatureTrainer.Hyperparameters Hyperparameters,
        IReadOnlyList<string> InternalFeatureColumns,
        IReadOnlyList<string> PublicFeatureNames,
        IReadOnlyList<ModelAccessor<T>> ModelAccessors,
        Func<T, DateTime> TimeOf,
        Func<T, float> TruthOf,
        IReadOnlyList<string> DeviationsFromBrief,
        Func<int, CancellationToken, IReadOnlyList<T>> LoadRowsForLead) where T : class;

    public static async Task<int> RunAsync<T>(
        ILogger log,
        ElementTrainerInputs<T> inputs,
        int[] leads,
        CancellationToken ct) where T : class
    {
        var now = DateTime.UtcNow;
        var modelsRoot = Path.Combine("data", "models");
        var versionDir = ModelArtifact.BuildVersionDir(modelsRoot, inputs.Target.ModelDirName, now);
        var versionName = Path.GetFileName(versionDir);

        log.LogInformation(
            "Element blender ({Display}) — training per-lead for leads [{Leads}]",
            inputs.Target.Display, string.Join(",", leads));
        var hp = inputs.Hyperparameters;
        log.LogInformation(
            "Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);
        log.LogInformation(
            "Feature columns ({N}): {Cols}",
            inputs.PublicFeatureNames.Count, string.Join(", ", inputs.PublicFeatureNames));

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            log.LogInformation("--- {Target} / lead {Lead}h ---", inputs.Target.CliName, lead);

            var rows = inputs.LoadRowsForLead(lead, ct);
            log.LogInformation(
                "Loaded {N} rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : inputs.TimeOf(rows[0]),
                rows.Count == 0 ? DateTime.MinValue : inputs.TimeOf(rows[^1]));

            if (rows.Count < 500)
            {
                log.LogError("Only {N} rows for {Target} lead {Lead}h — too few to train.",
                    rows.Count, inputs.Target.CliName, lead);
                return 3;
            }

            var ds = ChronologicalSplit<T>.Split(rows, inputs.TimeOf);
            log.LogInformation(
                "Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart(inputs.TimeOf), ds.TrainEnd(inputs.TimeOf),
                ds.Val.Count,   ds.ValStart(inputs.TimeOf),   ds.ValEnd(inputs.TimeOf),
                ds.Test.Count,  ds.TestStart(inputs.TimeOf),  ds.TestEnd(inputs.TimeOf));

            var trained = TemperatureTrainer.Train<T>(ds.Train, ds.Val, inputs.InternalFeatureColumns, hp);

            var testActual = ds.Test.Select(x => (double)inputs.TruthOf(x)).ToArray();
            var testPred   = TemperatureTrainer.Predict(trained.Ml, trained.Model, ds.Test);
            var blendStats = Metrics.Compute(testPred, testActual);

            var (bestModelId, bestValMae) = BestSingle(ds.Val, inputs);
            // Score that same model on test for an apples-to-apples blend-vs-best comparison.
            var bestAccessor = inputs.ModelAccessors.First(a => a.ModelId == bestModelId);
            var bestTestMae = NaNAwareMae(
                ds.Test.Select(r => (double)bestAccessor.Get(r)),
                ds.Test.Select(r => (double)inputs.TruthOf(r)).ToArray());

            ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
            importanceByLead[lead] = MapImportanceToPublicNames(trained.FeatureImportance, inputs);

            var testMonths = ds.Test
                .Select(r => { var t = inputs.TimeOf(r); return new DateTime(t.Year, t.Month, 1); })
                .Distinct().Count();

            perLead[lead.ToString()] = new ModelArtifact.PerLeadStats
            {
                LeadHours = lead,
                DataRangeTrain = $"{ds.TrainStart(inputs.TimeOf):yyyy-MM-dd HH:mm}Z → {ds.TrainEnd(inputs.TimeOf):yyyy-MM-dd HH:mm}Z",
                DataRangeVal   = $"{ds.ValStart(inputs.TimeOf):yyyy-MM-dd HH:mm}Z → {ds.ValEnd(inputs.TimeOf):yyyy-MM-dd HH:mm}Z",
                DataRangeTest  = $"{ds.TestStart(inputs.TimeOf):yyyy-MM-dd HH:mm}Z → {ds.TestEnd(inputs.TimeOf):yyyy-MM-dd HH:mm}Z",
                TrainRows = ds.Train.Count,
                ValRows   = ds.Val.Count,
                TestRows  = ds.Test.Count,
                TestCalendarMonths = testMonths,
                BestSingle = bestModelId,
                BestSingleValMae = bestValMae,
                BestSingleTestMae = bestTestMae,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPct = bestTestMae > 0
                ? 100.0 * (bestTestMae - blendStats.Mae) / bestTestMae
                : double.NaN;
            log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000} {U}, best_single[{Best}] test MAE={BestTest:0.000} {U} (val {BestVal:0.000}); blend Δ={Delta:+0.0;-0.0;0.0}%",
                lead, blendStats.Mae, inputs.Target.Units, bestModelId, bestTestMae, inputs.Target.Units, bestValMae, deltaPct);
        }

        ModelArtifact.SaveFeatureSchema(versionDir, inputs.PublicFeatureNames);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = inputs.Target.CliName,
            Phase = inputs.Target.PhaseTag,
            DataSource = "previous_runs_api",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = inputs.DeviationsFromBrief.ToList(),
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        ModelArtifact.UpdateManifest(modelsRoot, inputs.Target.ModelDirName, versionName);

        log.LogInformation("Element ({Target}) artefacts → {Dir}", inputs.Target.CliName, versionDir);
        log.LogInformation(
            "Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
            {
                var delta = kv.Value.BestSingleTestMae > 0
                    ? 100.0 * (kv.Value.BestSingleTestMae - kv.Value.BlendTestMae) / kv.Value.BestSingleTestMae
                    : double.NaN;
                return $"lead {kv.Key}h: blend {kv.Value.BlendTestMae:0.000} vs {kv.Value.BestSingle} test {kv.Value.BestSingleTestMae:0.000} {inputs.Target.Units} (Δ={delta:+0.0;-0.0;0.0}%)";
            })));

        await Task.CompletedTask;
        return 0;
    }

    private static (string ModelId, double ValMae) BestSingle<T>(
        IReadOnlyList<T> val,
        ElementTrainerInputs<T> inputs) where T : class
    {
        var actual = val.Select(r => (double)inputs.TruthOf(r)).ToArray();
        var perModel = inputs.ModelAccessors
            .Select(acc => (acc.ModelId, Mae: NaNAwareMae(val.Select(r => (double)acc.Get(r)), actual)))
            .Where(t => !double.IsNaN(t.Mae))
            .ToArray();
        if (perModel.Length == 0)
            throw new InvalidOperationException("No per-model column had any non-NaN values on the validation split.");
        var best = perModel.MinBy(x => x.Mae);
        return (best.ModelId, best.Mae);
    }

    /// <summary>Pairwise-NaN-aware MAE: skip pairs where prediction or truth is NaN.</summary>
    private static double NaNAwareMae(IEnumerable<double> predicted, IReadOnlyList<double> actual)
    {
        var pred = predicted.ToArray();
        if (pred.Length != actual.Count)
            throw new InvalidOperationException($"Length mismatch: pred={pred.Length} actual={actual.Count}");
        double sum = 0;
        int n = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            if (double.IsNaN(pred[i]) || double.IsNaN(actual[i])) continue;
            sum += Math.Abs(pred[i] - actual[i]);
            n++;
        }
        return n == 0 ? double.NaN : sum / n;
    }

    /// <summary>
    /// LightGBM names features by the C# property strings ML.NET reflects on. Map those
    /// back to the public per-target feature names persisted in feature_schema.json.
    /// Lookup is positional — internal and public lists are aligned 1:1 by construction.
    /// </summary>
    private static IEnumerable<(string Name, double Gain)> MapImportanceToPublicNames<T>(
        IReadOnlyList<(string Name, double Gain)> importance,
        ElementTrainerInputs<T> inputs) where T : class
    {
        var positionOf = new Dictionary<string, int>(inputs.InternalFeatureColumns.Count);
        for (int i = 0; i < inputs.InternalFeatureColumns.Count; i++)
            positionOf[inputs.InternalFeatureColumns[i]] = i;
        return importance
            .Where(t => positionOf.ContainsKey(t.Name))
            .Select(t => (Name: inputs.PublicFeatureNames[positionOf[t.Name]], t.Gain));
    }

    private static Dictionary<string, object> BuildHpDict(TemperatureTrainer.Hyperparameters hp)
        => new()
        {
            ["NumberOfIterations"] = hp.NumberOfIterations,
            ["LearningRate"] = hp.LearningRate,
            ["NumberOfLeaves"] = hp.NumberOfLeaves,
            ["MinimumExampleCountPerLeaf"] = hp.MinimumExampleCountPerLeaf,
            ["L1Regularization"] = hp.L1Regularization,
            ["L2Regularization"] = hp.L2Regularization,
            ["EarlyStoppingRound"] = hp.EarlyStoppingRound,
            ["Seed"] = hp.Seed,
            ["SubsampleFraction"] = hp.SubsampleFraction,
            ["SubsampleFrequency"] = hp.SubsampleFrequency,
            ["FeatureFraction"] = hp.FeatureFraction,
        };
}
