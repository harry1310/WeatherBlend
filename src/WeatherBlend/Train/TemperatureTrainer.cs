using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace WeatherBlend.Train;

/// <summary>
/// LightGBM regressor for the temperature blender. One model, no lead-time bucketing
/// (phase 2a Option A — see docs/LEAD_TIME_BACKFILL.md and session notes for why).
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
    /// Fit on the train split, use the val split for early stopping. The test split
    /// is deliberately not touched here — it's for the final report only.
    /// </summary>
    public static TrainedBlender Train(TrainingDataset ds, Hyperparameters hp)
    {
        var ml = new MLContext(seed: hp.Seed);

        var trainDv = ml.Data.LoadFromEnumerable(ds.Train);
        var valDv   = ml.Data.LoadFromEnumerable(ds.Val);

        var featureNames = FeatureBuilder.FeatureNames.ToArray();

        // Microsoft.ML normalises to the "Features" + "Label" convention; we've
        // already set [ColumnName("Label")] on Era5Temp in TrainingRow.
        var featurize = ml.Transforms.Concatenate("Features", FeatureColumnInternalNames());
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

        // Compose feat-model + predictor so the saved .zip accepts raw TrainingRow-shaped input.
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
    public static double[] Predict(MLContext ml, ITransformer model, IReadOnlyList<TrainingRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var dv = ml.Data.LoadFromEnumerable(rows);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Score").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }

    // TrainingRow property names (C# identifiers) map 1:1 to DataView columns.
    // Note we exclude ValidTimeUtc, WindDirMean (diagnostic only) and Label.
    private static string[] FeatureColumnInternalNames() => new[]
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
}
