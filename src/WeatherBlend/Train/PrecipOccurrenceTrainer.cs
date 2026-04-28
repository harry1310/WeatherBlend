using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// LightGBM binary classifier for P(wet hour). Vector-native: takes
/// <see cref="BinaryTrainingRow"/> rows whose <c>Features</c> field already IS
/// the training vector — feature schema lives on the <see cref="BlenderSpec"/>.
///
/// Microsoft.ML.LightGbm 4.0 exposes only a subset of the raw LightGBM options.
/// Deviations from the brief's "raw LightGBM" ask:
///   - No native class-weight parameter. With ~25% wet hours the class
///     imbalance is moderate; we rely on LightGBM's native handling.
///   - EvaluationMetric is fixed to AUC in binary classification; early stopping
///     still fires on val AUC convergence.
/// Both are recorded in TrainingMetadata.DeviationsFromBrief.
/// </summary>
public sealed class PrecipOccurrenceTrainer
{
    public sealed record Hyperparameters(
        int NumberOfIterations = 600,
        double LearningRate = 0.04,
        int NumberOfLeaves = 31,
        int MinimumExampleCountPerLeaf = 40,
        double L1Regularization = 0.1,
        double L2Regularization = 0.1,
        int EarlyStoppingRound = 40,
        int Seed = 42,
        // Bagging (added 2026-04-26 alongside temp 2b — same defaults). Standard
        // LightGBM regularisation; ML.NET's wrapper defaults to 1.0/1.0 (off)
        // which leaves the booster fully deterministic and over-reliant on the
        // single-fit signal. See TemperatureTrainer.Hyperparameters for rationale.
        double SubsampleFraction = 0.8,
        int SubsampleFrequency = 1,
        double FeatureFraction = 0.8);

    public sealed record TrainedClassifier(
        MLContext Ml,
        ITransformer Model,
        DataViewSchema InputSchema,
        Hyperparameters Hyperparameters,
        IReadOnlyList<string> FeatureNames,
        IReadOnlyList<(string Name, double Gain)> FeatureImportance);

    /// <summary>
    /// Vector-native fit: takes <see cref="BinaryTrainingRow"/> rows whose
    /// <see cref="BinaryTrainingRow.Features"/> already IS the training vector
    /// (no Concatenate transform needed). The vector length is fixed per
    /// <paramref name="spec"/>; train and predict stay lockstep because both
    /// sides build the vector from the same <see cref="BlenderSpec"/>.
    /// </summary>
    public static TrainedClassifier TrainVector(
        IReadOnlyList<BinaryTrainingRow> train,
        IReadOnlyList<BinaryTrainingRow> val,
        BlenderSpec spec,
        Hyperparameters hp)
    {
        if (train.Count == 0)
            throw new ArgumentException("No training rows", nameof(train));
        if (train[0].Features.Length != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Training row Features length {train[0].Features.Length} != spec.FeatureCount {spec.FeatureCount} for {spec}");

        var ml = new MLContext(seed: hp.Seed);

        var schema = SchemaDefinition.Create(typeof(BinaryTrainingRow));
        schema[nameof(BinaryTrainingRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);

        var trainDv = ml.Data.LoadFromEnumerable(train, schema);
        var valDv   = ml.Data.LoadFromEnumerable(val, schema);

        var options = new LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = nameof(BinaryTrainingRow.Features),
            NumberOfIterations = hp.NumberOfIterations,
            LearningRate = hp.LearningRate,
            NumberOfLeaves = hp.NumberOfLeaves,
            MinimumExampleCountPerLeaf = hp.MinimumExampleCountPerLeaf,
            EarlyStoppingRound = hp.EarlyStoppingRound,
            Seed = hp.Seed,
            UnbalancedSets = false,
            Booster = new GradientBooster.Options
            {
                L1Regularization = hp.L1Regularization,
                L2Regularization = hp.L2Regularization,
                SubsampleFraction = hp.SubsampleFraction,
                SubsampleFrequency = hp.SubsampleFrequency,
                FeatureFraction = hp.FeatureFraction,
            },
        };

        var trainer = ml.BinaryClassification.Trainers.LightGbm(options);
        var predictor = trainer.Fit(trainDv, valDv);

        var weights = default(VBuffer<float>);
        predictor.Model.SubModel.GetFeatureWeights(ref weights);
        var importance = weights.DenseValues()
            .Select((g, i) => (Name: spec.FeatureNames[i], Gain: (double)g))
            .OrderByDescending(t => t.Gain)
            .ToArray();

        return new TrainedClassifier(
            Ml: ml,
            Model: predictor,
            InputSchema: trainDv.Schema,
            Hyperparameters: hp,
            FeatureNames: spec.FeatureNames.ToArray(),
            FeatureImportance: importance);
    }

    /// <summary>
    /// Vector-native predict over <see cref="BinaryTrainingRow"/>. Uses an
    /// explicit SchemaDefinition matching <paramref name="spec"/> so the saved
    /// model receives the same shape it was trained on.
    /// </summary>
    public static double[] PredictVectorProbability(
        MLContext ml,
        ITransformer model,
        BlenderSpec spec,
        IReadOnlyList<BinaryTrainingRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var schema = SchemaDefinition.Create(typeof(BinaryTrainingRow));
        schema[nameof(BinaryTrainingRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);
        var dv = ml.Data.LoadFromEnumerable(rows, schema);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Probability").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }

}
