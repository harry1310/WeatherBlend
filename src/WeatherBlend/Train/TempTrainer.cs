using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// LightGBM regressor for the temperature blender. Vector-native: takes
/// <see cref="RegressionTrainingRow"/> rows whose <c>Features</c> field already
/// IS the training vector — feature schema lives on the <see cref="BlenderSpec"/>.
///
/// Deviations from the original brief, forced by Microsoft.ML.LightGbm 4.0.0's public API:
///   - Objective is L2 (squared error), not regression_l1. The public Options class
///     doesn't expose objective; we use EvaluationMetric=MeanAbsoluteError so early
///     stopping is still driven by MAE on the validation set.
///   - No monotone_constraints. The public Options class doesn't expose them.
///     Consequence: the learned blend is not guaranteed monotone in each input
///     model's forecast. We'll see this in feature importance / residuals if it bites.
/// Both deviations are recorded in TrainingMetadata.DeviationsFromBrief.
/// </summary>
public sealed class TempTrainer
{
    public sealed record Hyperparameters(
        int NumberOfIterations = 500,
        double LearningRate = 0.05,
        int NumberOfLeaves = 31,
        int MinimumExampleCountPerLeaf = 20,
        double L1Regularization = 0.1,
        double L2Regularization = 0.1,
        int EarlyStoppingRound = 30,
        int Seed = 42,
        // Bagging (added 2026-04-26). Hyperparameter grid on temp 2b 6-model
        // restricted-window dataset showed feature_fraction is the bigger lever
        // (every per-lead winner has it ≤0.8); bagging_fraction 0.7–0.8 with
        // freq=1 sits within seed noise of the optimum at every lead. ~1% MAE
        // improvement over the deterministic 1.0/1.0 default at every lead;
        // also gives the ensemble the noise it needs to be a real ensemble.
        // Element trainers inherit these via ElementTrainerHarness which calls
        // TempTrainer.Train<T>.
        double SubsampleFraction = 0.8,
        int SubsampleFrequency = 1,
        double FeatureFraction = 0.8)
    {
        /// <summary>
        /// Default Hyperparameters with optional smoke-time overrides via env
        /// vars. <c>WB_SMOKE_ITER</c> caps NumberOfIterations + scales the
        /// EarlyStoppingRound proportionally; <c>WB_SMOKE_LEAVES</c> sets a
        /// smaller NumberOfLeaves. Unset env vars leave production defaults
        /// (500 / 30 / 31) untouched — every workflow / cron / manual run
        /// without those env vars gets bit-identical behaviour to a plain
        /// <c>new Hyperparameters()</c>.
        /// </summary>
        public static Hyperparameters Default()
        {
            var iter = int.TryParse(Environment.GetEnvironmentVariable("WB_SMOKE_ITER"), out var i) ? i : 500;
            var esr  = int.TryParse(Environment.GetEnvironmentVariable("WB_SMOKE_ESR"),  out var e) ? e : 30;
            var lvs  = int.TryParse(Environment.GetEnvironmentVariable("WB_SMOKE_LEAVES"), out var l) ? l : 31;
            return new Hyperparameters(
                NumberOfIterations: iter,
                EarlyStoppingRound: Math.Min(esr, iter),
                NumberOfLeaves: lvs);
        }
    }

    public sealed record TrainedBlender(
        MLContext Ml,
        ITransformer Model,
        DataViewSchema InputSchema,
        Hyperparameters Hyperparameters,
        IReadOnlyList<string> FeatureNames,
        IReadOnlyList<(string Name, double Gain)> FeatureImportance);

    /// <summary>
    /// Vector-native fit: takes <see cref="RegressionTrainingRow"/> rows whose
    /// <see cref="RegressionTrainingRow.Features"/> already IS the training
    /// vector (no Concatenate transform needed). The vector length is fixed
    /// per <paramref name="spec"/>; train/predict stay lockstep because both
    /// sides build the vector from the same <see cref="BlenderSpec"/>.
    /// </summary>
    public static TrainedBlender TrainVector(
        IReadOnlyList<RegressionTrainingRow> train,
        IReadOnlyList<RegressionTrainingRow> val,
        BlenderSpec spec,
        Hyperparameters hp)
    {
        if (train.Count == 0)
            throw new ArgumentException("No training rows", nameof(train));
        if (train[0].Features.Length != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Training row Features length {train[0].Features.Length} != spec.FeatureCount {spec.FeatureCount} for {spec}");

        var ml = new MLContext(seed: hp.Seed);

        // Load with an explicit SchemaDefinition so the Features column gets the
        // right vector size — ML.NET defaults to size-by-first-row but being
        // explicit avoids surprises if the first row's float[] length is wrong.
        var schema = SchemaDefinition.Create(typeof(RegressionTrainingRow));
        schema[nameof(RegressionTrainingRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);

        var trainDv = ml.Data.LoadFromEnumerable(train, schema);
        var valDv   = ml.Data.LoadFromEnumerable(val, schema);

        var options = new LightGbmRegressionTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = nameof(RegressionTrainingRow.Features),
            NumberOfIterations = hp.NumberOfIterations,
            LearningRate = hp.LearningRate,
            NumberOfLeaves = hp.NumberOfLeaves,
            MinimumExampleCountPerLeaf = hp.MinimumExampleCountPerLeaf,
            EarlyStoppingRound = hp.EarlyStoppingRound,
            Seed = hp.Seed,
            EvaluationMetric = LightGbmRegressionTrainer.Options.EvaluateMetricType.MeanAbsoluteError,
            Booster = new GradientBooster.Options
            {
                L1Regularization = hp.L1Regularization,
                L2Regularization = hp.L2Regularization,
                SubsampleFraction = hp.SubsampleFraction,
                SubsampleFrequency = hp.SubsampleFrequency,
                FeatureFraction = hp.FeatureFraction,
            },
        };

        var trainer = ml.Regression.Trainers.LightGbm(options);
        var predictor = trainer.Fit(trainDv, valDv);

        var weights = default(VBuffer<float>);
        predictor.Model.GetFeatureWeights(ref weights);
        var importance = weights.DenseValues()
            .Select((g, i) => (Name: spec.FeatureNames[i], Gain: (double)g))
            .OrderByDescending(t => t.Gain)
            .ToArray();

        return new TrainedBlender(
            Ml: ml,
            Model: predictor,
            InputSchema: trainDv.Schema,
            Hyperparameters: hp,
            FeatureNames: spec.FeatureNames.ToArray(),
            FeatureImportance: importance);
    }

    /// <summary>
    /// Vector-native predict over <see cref="RegressionTrainingRow"/>. Uses an
    /// explicit SchemaDefinition matching <paramref name="spec"/> so the saved
    /// model receives the same shape it was trained on.
    /// </summary>
    public static double[] PredictVector(
        MLContext ml,
        ITransformer model,
        BlenderSpec spec,
        IReadOnlyList<RegressionTrainingRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var schema = SchemaDefinition.Create(typeof(RegressionTrainingRow));
        schema[nameof(RegressionTrainingRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);
        var dv = ml.Data.LoadFromEnumerable(rows, schema);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Score").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }

}
