using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Temp;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Generic per-target training pipeline used by every Element blender. Owns:
///   - per-lead loop (resolving each lead's <see cref="BlenderSpec"/>)
///   - chronological 70/15/15 split via <see cref="RegressionDataset"/>
///   - vector-native LightGBM fit via <see cref="TempTrainer.TrainVector"/>
///   - best-single-per-lead computation (val-MAE pick over spec.Models)
///   - artefact save (model.zip per lead, BlenderSpec per lead in feature_schema.json,
///     importance, metadata)
///   - MANIFEST update
///
/// Per-element specifics (which spec to build, how to load rows for it) are
/// injected via <see cref="ElementTrainerInputs"/> so the harness stays
/// agnostic of which physical variable is being blended.
/// </summary>
public static class ElementTrainerHarness
{
    public sealed record ElementTrainerInputs(
        ElementTarget Target,
        TempTrainer.Hyperparameters Hyperparameters,
        string ModelsRoot,
        Func<int, BlenderSpec> BuildSpec,
        Func<BlenderSpec, CancellationToken, IReadOnlyList<RegressionTrainingRow>> LoadRowsForSpec,
        IReadOnlyList<string> DeviationsFromBrief,
        // Configured location whose NWP fed the training data. Pinned at
        // train time so predict can refuse to score the bundle against any
        // other location's NWP (Phase A multi-location safety, 2026-05-12).
        // Load-bearing since P3: it is also the element manifest's station
        // key — bundles live under data/models/{target}/{LocationName}/v{ts}/.
        // Element blenders pass _cfg.Location.Name; required, no default.
        string LocationName,
        // Optional suffix appended to the version dir name (yields
        // v{ts}_{VersionSuffix}/). Use this when multiple sibling phases
        // share the same ModelDirName (e.g. WindSpeedLgb under "wind" coexists
        // with the existing Wind phase, both stamped under data/models/wind/).
        // Defaults to the ElementTarget.PhaseTag when the PhaseTag differs
        // from the ModelDirName so sibling-phase dirs are visibly distinct;
        // single-phase targets (wind, humidity, shortwave_radiation,
        // cloud_cover where PhaseTag == ModelDirName) keep the legacy
        // unsuffixed layout. Override with null to force unsuffixed even
        // when PhaseTag differs.
        string? VersionSuffix = null);

    public static async Task<int> RunAsync(
        ILogger log,
        ElementTrainerInputs inputs,
        int[] leads,
        CancellationToken ct)
    {
        var now = DateTime.UtcNow;
        var modelsRoot = inputs.ModelsRoot;
        if (string.IsNullOrWhiteSpace(inputs.LocationName))
            throw new InvalidOperationException(
                $"Element blender ({inputs.Target.CliName}) has no LocationName — it is the " +
                "manifest station key and the per-location bundle directory; cannot train without it.");
        // Station-keyed (location-keyed) layout: every target's bundles live
        // under data/models/{target}/{location}/v{ts}[_suffix]/. The suffix
        // is auto-derived from PhaseTag when a sibling phase shares the
        // ModelDirName (wind_speed_lgb under "wind" coexists with the existing
        // Wind phase) so the two dirs are visibly distinct on disk.
        var resolvedSuffix = inputs.VersionSuffix
            ?? (string.Equals(inputs.Target.ModelDirName, inputs.Target.PhaseTag, StringComparison.OrdinalIgnoreCase)
                ? null
                : inputs.Target.PhaseTag);
        var versionDir = ModelArtifact.BuildStationVersionDir(
            modelsRoot, inputs.Target.ModelDirName, inputs.LocationName, now, resolvedSuffix);
        var versionName = Path.GetFileName(versionDir);

        log.LogInformation(
            "Element blender ({Display}) — training per-lead for leads [{Leads}]",
            inputs.Target.Display, string.Join(",", leads));
        var hp = inputs.Hyperparameters;
        log.LogInformation(
            "Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // training_summary buffers (Phase 1a). Element blenders are
        // regression — no labelRates. First lead's train slice supplies
        // per-feature stats; row counts aggregate across leads.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            log.LogInformation("--- {Target} / lead {Lead}h ---", inputs.Target.CliName, lead);

            var spec = inputs.BuildSpec(lead);
            specsPerLead[lead] = spec;
            log.LogInformation("Spec: {Spec}", spec);

            var rows = inputs.LoadRowsForSpec(spec, ct);
            log.LogInformation(
                "Loaded {N} rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                log.LogError("Only {N} rows for {Target} lead {Lead}h — too few to train.",
                    rows.Count, inputs.Target.CliName, lead);
                return 3;
            }

            var ds = RegressionDataset.Split(rows);
            log.LogInformation(
                "Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);
            totalTrainRows += ds.Train.Count;
            totalValRows   += ds.Val.Count;
            totalTestRows  += ds.Test.Count;
            firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            // Best per-model on val MAE — picks the per-model feature with the lowest
            // MAE against truth. Spec.FeatureNames[0..Models.Count-1] are per-model
            // primary-variable slots (wind_spd_X, rh_X, cloud_X, sw_X depending on target).
            var (bestModelId, bestValMae) = BestSingle(spec, ds.Val);
            // Score the SAME model on test for an apples-to-apples blend-vs-best comparison.
            var bestTestMae = NaNAwareMae(
                TempBaselines.FromFeature(spec, ds.Test, BestSingleFeatureName(spec, bestModelId)),
                ds.Test.Select(r => (double)r.Label).ToArray());

            ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
            importanceByLead[lead] = trained.FeatureImportance;

            var testMonths = ds.Test
                .Select(r => new DateTime(r.ValidTimeUtc.Year, r.ValidTimeUtc.Month, 1))
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

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = inputs.Target.CliName,
            Phase = inputs.Target.PhaseTag,
            LocationName = inputs.LocationName,
            DataSource = "previous_runs_api",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = inputs.DeviationsFromBrief.ToList(),
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        var firstLeadElement = leads.Length > 0 ? leads[0] : 0;
        var guardResultEl = RetrainGuard.BuildCheckAndSave(log,
            versionDir,
            composite: inputs.Target.CliName,
            phase: inputs.Target.PhaseTag,
            version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(firstLeadElement, out var spEl)
                ? spEl.FeatureNames.ToList() : Array.Empty<string>(),
            locationName: inputs.LocationName,
            // ONE-TIME bypass for two intentional feature-count changes productionised
            // 2026-06-04: shortwave-radiation gained cloud_mean/rh_mean (25→27 feats);
            // cloud-cover gained cape_mean at the 24h lead (13→14 feats). The guard's
            // FeaturesEffectiveDelta=0 would abort the first retrain that sees either
            // jump. Allow it for these two targets so Sunday's retrain mints the new
            // baselines; once they land (n→n passes) this should be REVERTED — it
            // currently masks any future radiation/cloud schema drift.
            tolerances: inputs.Target.CliName is "shortwave-radiation" or "cloud-cover"
                ? RetrainGuard.Defaults with { AllowFeaturesEffectiveChange = true }
                : null);
        if (!guardResultEl.Passed)
        {
            log.LogError(
                "Aborting Element ({Target}) retrain — sanity guard failed. Orphan dir {Dir} not promoted.",
                inputs.Target.CliName, versionDir);
            return 4;
        }
        // Promote: replaces any prior entry with the same Phase in this
        // location's station entry of Active. Element targets currently have a
        // single phase each (wind / humidity / shortwave_radiation /
        // cloud_cover) so this is functionally equivalent to a single-version
        // write today, but the promote helper future-proofs against a
        // challenger ever being added.
        ModelArtifact.PromoteStationVersion(
            modelsRoot, inputs.Target.ModelDirName, inputs.LocationName,
            versionName, newPhase: inputs.Target.PhaseTag);

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

    /// <summary>
    /// Pick the per-model feature with lowest MAE on <paramref name="val"/>. Returns
    /// the model id (e.g. "ecmwf_ifs025") and its val MAE so the report can name
    /// the model that gets compared against the blend.
    /// </summary>
    private static (string ModelId, double ValMae) BestSingle(
        BlenderSpec spec,
        IReadOnlyList<RegressionTrainingRow> val)
    {
        var actual = val.Select(r => (double)r.Label).ToArray();
        string bestId = spec.Models[0];
        double bestMae = double.PositiveInfinity;
        for (int i = 0; i < spec.Models.Count; i++)
        {
            // The first N feature names are the per-model primary-variable slots
            // (the TempFeatureBuilder for each Element places them at positions 0..N-1).
            var pred = TempBaselines.FromFeature(spec, val, spec.FeatureNames[i]);
            var mae = NaNAwareMae(pred, actual);
            if (!double.IsNaN(mae) && mae < bestMae) { bestMae = mae; bestId = spec.Models[i]; }
        }
        if (double.IsPositiveInfinity(bestMae))
            throw new InvalidOperationException("No per-model feature had any non-NaN values on the validation split.");
        return (bestId, bestMae);
    }

    /// <summary>
    /// Per-element primary-variable feature name for <paramref name="modelId"/>.
    /// The harness only knows that it's <c>spec.FeatureNames[i]</c> at the same
    /// index as <c>spec.Models[i]</c>. Used to score the same model on test.
    /// </summary>
    private static string BestSingleFeatureName(BlenderSpec spec, string modelId)
    {
        for (int i = 0; i < spec.Models.Count; i++)
            if (string.Equals(spec.Models[i], modelId, StringComparison.OrdinalIgnoreCase))
                return spec.FeatureNames[i];
        throw new InvalidOperationException($"Model '{modelId}' not in spec.Models for {spec}.");
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

    private static Dictionary<string, object> BuildHpDict(TempTrainer.Hyperparameters hp)
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
