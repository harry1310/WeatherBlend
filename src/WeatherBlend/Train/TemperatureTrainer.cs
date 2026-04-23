using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace WeatherBlend.Train;

/// <summary>
/// LightGBM regressor for the temperature blender. Pure fit-and-predict over the
/// 13 features in <see cref="FeatureBuilder.FeatureNames"/> — callers decide what
/// training data to hand in (e.g. one dataset per lead for per-lead blenders).
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
public sealed class TemperatureTrainer
{
    public sealed record Hyperparameters(
        int NumberOfIterations = 500,
        double LearningRate = 0.05,
        int NumberOfLeaves = 31,
        int MinimumExampleCountPerLeaf = 20,
        double L1Regularization = 0.1,
        double L2Regularization = 0.1,
        int EarlyStoppingRound = 30,
        int Seed = 42);

    public sealed record TrainedBlender(
        MLContext Ml,
        ITransformer Model,
        DataViewSchema InputSchema,
        Hyperparameters Hyperparameters,
        IReadOnlyList<string> FeatureNames,
        IReadOnlyList<(string Name, double Gain)> FeatureImportance);

    /// <summary>
    /// Phase 2b call site: fit on the lean 13-feature TrainingDataset.
    /// Thin wrapper over the generic <see cref="Train{T}"/> with the lean feature list.
    /// </summary>
    public static TrainedBlender Train(TrainingDataset ds, Hyperparameters hp)
        => Train(ds.Train, ds.Val, LeanFeatureColumnInternalNames(), hp);

    /// <summary>
    /// Generic LightGBM fit over arbitrary rows + feature column list. Used by
    /// Phase 2b (lean 13 cols) and Phase 2c (rich 88 cols) without further changes.
    /// Each named column must exist as a public property of <typeparamref name="T"/>;
    /// the type must also expose <c>[ColumnName("Label")]</c> on the regression target
    /// — both the lean and rich row types inherit this from <see cref="TrainingRow"/>.
    /// </summary>
    public static TrainedBlender Train<T>(
        IReadOnlyList<T> train,
        IReadOnlyList<T> val,
        IReadOnlyList<string> featureColumnNames,
        Hyperparameters hp)
        where T : class
    {
        var ml = new MLContext(seed: hp.Seed);

        var trainDv = ml.Data.LoadFromEnumerable(train);
        var valDv   = ml.Data.LoadFromEnumerable(val);

        var featureNames = featureColumnNames.ToArray();

        // Microsoft.ML normalises to the "Features" + "Label" convention; we've
        // already set [ColumnName("Label")] on Era5Temp in TrainingRow.
        var featurize = ml.Transforms.Concatenate("Features", featureNames);
        var featModel = featurize.Fit(trainDv);
        var trainFeat = featModel.Transform(trainDv);
        var valFeat   = featModel.Transform(valDv);

        var options = new LightGbmRegressionTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = "Features",
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
            },
        };

        var trainer = ml.Regression.Trainers.LightGbm(options);
        var predictor = trainer.Fit(trainFeat, valFeat);

        // Compose feat-model + predictor so the saved .zip accepts raw row-shaped input.
        var fullModel = featModel.Append(predictor);

        // Gain-based feature importance from the LightGBM booster.
        var weights = default(VBuffer<float>);
        predictor.Model.GetFeatureWeights(ref weights);
        var importance = weights.DenseValues()
            .Select((g, i) => (Name: featureNames[i], Gain: (double)g))
            .OrderByDescending(t => t.Gain)
            .ToArray();

        return new TrainedBlender(
            Ml: ml,
            Model: fullModel,
            InputSchema: trainDv.Schema,
            Hyperparameters: hp,
            FeatureNames: featureNames,
            FeatureImportance: importance);
    }

    /// <summary>Run the trained model over rows, return predicted values as a double[].</summary>
    public static double[] Predict<T>(MLContext ml, ITransformer model, IReadOnlyList<T> rows)
        where T : class
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var dv = ml.Data.LoadFromEnumerable(rows);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Score").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }

    // Lean (Phase 2b) feature property names. Maps 1:1 to the 13-entry
    // FeatureBuilder.FeatureNames list. ValidTimeUtc, WindDirMean (diagnostic) and
    // Label are deliberately excluded.
    private static string[] LeanFeatureColumnInternalNames() => new[]
    {
        nameof(TrainingRow.TempGfs),
        nameof(TrainingRow.TempEcmwf),
        nameof(TrainingRow.TempIcon),
        nameof(TrainingRow.TempMf),
        nameof(TrainingRow.TempUkmo),
        nameof(TrainingRow.TempGem),
        nameof(TrainingRow.TempMean),
        nameof(TrainingRow.TempStd),
        nameof(TrainingRow.TempRange),
        nameof(TrainingRow.HourSin),
        nameof(TrainingRow.HourCos),
        nameof(TrainingRow.DoySin),
        nameof(TrainingRow.DoyCos),
    };

    /// <summary>
    /// Rich (Phase 2c) feature property names. Maps 1:1 to RichFeatureBuilder.FeatureNames
    /// and the 88 RichTrainingRow public properties (lean 13 + 66 secondaries + 9 aggregates).
    /// </summary>
    public static string[] RichFeatureColumnInternalNames() => new[]
    {
        // 13 lean (must match LeanFeatureColumnInternalNames order).
        nameof(TrainingRow.TempGfs), nameof(TrainingRow.TempEcmwf), nameof(TrainingRow.TempIcon),
        nameof(TrainingRow.TempMf), nameof(TrainingRow.TempUkmo), nameof(TrainingRow.TempGem),
        nameof(TrainingRow.TempMean), nameof(TrainingRow.TempStd), nameof(TrainingRow.TempRange),
        nameof(TrainingRow.HourSin), nameof(TrainingRow.HourCos),
        nameof(TrainingRow.DoySin), nameof(TrainingRow.DoyCos),

        // 6 per-model secondaries × 10 vars (wind dir is sin+cos = 11 effective per model,
        // so 60 columns for the non-direction vars + 12 for sin/cos).
        nameof(RichTrainingRow.DewGfs), nameof(RichTrainingRow.DewEcmwf), nameof(RichTrainingRow.DewIcon),
        nameof(RichTrainingRow.DewMf), nameof(RichTrainingRow.DewUkmo), nameof(RichTrainingRow.DewGem),

        nameof(RichTrainingRow.RhGfs), nameof(RichTrainingRow.RhEcmwf), nameof(RichTrainingRow.RhIcon),
        nameof(RichTrainingRow.RhMf), nameof(RichTrainingRow.RhUkmo), nameof(RichTrainingRow.RhGem),

        nameof(RichTrainingRow.CloudGfs), nameof(RichTrainingRow.CloudEcmwf), nameof(RichTrainingRow.CloudIcon),
        nameof(RichTrainingRow.CloudMf), nameof(RichTrainingRow.CloudUkmo), nameof(RichTrainingRow.CloudGem),

        nameof(RichTrainingRow.CloudLowGfs), nameof(RichTrainingRow.CloudLowEcmwf), nameof(RichTrainingRow.CloudLowIcon),
        nameof(RichTrainingRow.CloudLowMf), nameof(RichTrainingRow.CloudLowUkmo), nameof(RichTrainingRow.CloudLowGem),

        nameof(RichTrainingRow.CloudMidGfs), nameof(RichTrainingRow.CloudMidEcmwf), nameof(RichTrainingRow.CloudMidIcon),
        nameof(RichTrainingRow.CloudMidMf), nameof(RichTrainingRow.CloudMidUkmo), nameof(RichTrainingRow.CloudMidGem),

        nameof(RichTrainingRow.CloudHighGfs), nameof(RichTrainingRow.CloudHighEcmwf), nameof(RichTrainingRow.CloudHighIcon),
        nameof(RichTrainingRow.CloudHighMf), nameof(RichTrainingRow.CloudHighUkmo), nameof(RichTrainingRow.CloudHighGem),

        nameof(RichTrainingRow.WindSpeedGfs), nameof(RichTrainingRow.WindSpeedEcmwf), nameof(RichTrainingRow.WindSpeedIcon),
        nameof(RichTrainingRow.WindSpeedMf), nameof(RichTrainingRow.WindSpeedUkmo), nameof(RichTrainingRow.WindSpeedGem),

        nameof(RichTrainingRow.WindDirSinGfs), nameof(RichTrainingRow.WindDirSinEcmwf), nameof(RichTrainingRow.WindDirSinIcon),
        nameof(RichTrainingRow.WindDirSinMf), nameof(RichTrainingRow.WindDirSinUkmo), nameof(RichTrainingRow.WindDirSinGem),

        nameof(RichTrainingRow.WindDirCosGfs), nameof(RichTrainingRow.WindDirCosEcmwf), nameof(RichTrainingRow.WindDirCosIcon),
        nameof(RichTrainingRow.WindDirCosMf), nameof(RichTrainingRow.WindDirCosUkmo), nameof(RichTrainingRow.WindDirCosGem),

        nameof(RichTrainingRow.WindGustsGfs), nameof(RichTrainingRow.WindGustsEcmwf), nameof(RichTrainingRow.WindGustsIcon),
        nameof(RichTrainingRow.WindGustsMf), nameof(RichTrainingRow.WindGustsUkmo), nameof(RichTrainingRow.WindGustsGem),

        nameof(RichTrainingRow.PressureGfs), nameof(RichTrainingRow.PressureEcmwf), nameof(RichTrainingRow.PressureIcon),
        nameof(RichTrainingRow.PressureMf), nameof(RichTrainingRow.PressureUkmo), nameof(RichTrainingRow.PressureGem),

        // 9 cross-model aggregates.
        nameof(RichTrainingRow.DewMean), nameof(RichTrainingRow.DewStd),
        nameof(RichTrainingRow.RhMean), nameof(RichTrainingRow.RhStd),
        nameof(RichTrainingRow.CloudMean),
        nameof(RichTrainingRow.WindSpeedMean), nameof(RichTrainingRow.WindSpeedStd),
        nameof(RichTrainingRow.PressureMean), nameof(RichTrainingRow.PressureStd),
    };
}
