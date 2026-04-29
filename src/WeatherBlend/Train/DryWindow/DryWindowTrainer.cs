using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using WeatherBlend.Train.Common;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// LightGBM binary classifier for P(dry-window exists) at day granularity.
/// Parallel of <see cref="PrecipOccurrenceTrainer"/> — same hyperparameter
/// defaults, same <c>UnbalancedSets=false</c> choice. Feature schema is
/// driven by the spec's <see cref="BlenderSpec.FeatureNames"/>.
/// </summary>
public sealed class DryWindowTrainer
{
    public sealed record Hyperparameters(
        int NumberOfIterations = 600,
        double LearningRate = 0.04,
        int NumberOfLeaves = 31,
        int MinimumExampleCountPerLeaf = 20, // half of 3a's 40 — day-level rows are ~24× fewer
        double L1Regularization = 0.1,
        double L2Regularization = 0.1,
        int EarlyStoppingRound = 40,
        int Seed = 42);

    public sealed record TrainedClassifier(
        MLContext Ml,
        ITransformer Model,
        DataViewSchema InputSchema,
        Hyperparameters Hyperparameters,
        IReadOnlyList<string> FeatureNames,
        IReadOnlyList<(string Name, double Gain)> FeatureImportance,
        IsotonicCalibrator Calibrator);

    /// <summary>
    /// Vector-native fit: takes generic <see cref="CommonRow"/> rows whose
    /// <see cref="CommonRow.Features"/> already IS the training vector. Vector
    /// length is fixed per <paramref name="spec"/>; train and predict stay
    /// lockstep because both sides build the vector from the same BlenderSpec.
    /// </summary>
    public static TrainedClassifier TrainVector(
        IReadOnlyList<CommonRow> train,
        IReadOnlyList<CommonRow> val,
        BlenderSpec spec,
        Hyperparameters hp)
    {
        if (train.Count == 0)
            throw new ArgumentException("No training rows", nameof(train));
        if (train[0].Features.Length != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Training row Features length {train[0].Features.Length} != spec.FeatureCount {spec.FeatureCount} for {spec}");

        var ml = new MLContext(seed: hp.Seed);

        var schema = SchemaDefinition.Create(typeof(CommonRow));
        schema[nameof(CommonRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);

        var trainDv = ml.Data.LoadFromEnumerable(train, schema);
        var valDv   = ml.Data.LoadFromEnumerable(val, schema);

        var options = new LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = nameof(CommonRow.Features),
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

        // Fit isotonic calibrator on validation predictions. PAV maps raw LightGBM
        // probability → empirical positive rate, correcting the systematic
        // over-prediction bias visible in the freq_bias diagnostic. Fit on val
        // (NOT train — that would be perfectly calibrated by definition; NOT test
        // — that would leak). Test is scored both raw and calibrated downstream.
        var valPredicted = predictor.Transform(valDv);
        var valRawProbs = valPredicted.GetColumn<float>("Probability").Select(p => (double)p).ToArray();
        var valLabels = val.Select(r => r.Label).ToArray();
        var calibrator = IsotonicCalibrator.Fit(valRawProbs, valLabels);

        return new TrainedClassifier(ml, predictor, trainDv.Schema, hp,
            spec.FeatureNames.ToArray(), importance, calibrator);
    }

    public static double[] PredictVectorProbability(
        MLContext ml,
        ITransformer model,
        BlenderSpec spec,
        IReadOnlyList<CommonRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var schema = SchemaDefinition.Create(typeof(CommonRow));
        schema[nameof(CommonRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);
        var dv = ml.Data.LoadFromEnumerable(rows, schema);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Probability").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }
}
