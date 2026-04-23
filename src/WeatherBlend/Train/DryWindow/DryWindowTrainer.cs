using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// LightGBM binary classifier for P(dry-window exists) at day granularity.
/// Parallel of <see cref="PrecipOccurrenceTrainer"/> — same hyperparameter
/// defaults, same <c>UnbalancedSets=false</c> choice. Feature set is
/// <see cref="DryWindowFeatureBuilder.FeatureNames"/> (53 columns), label is
/// <see cref="DryWindowTrainingRow.HasDryWindow"/>.
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
        IReadOnlyList<(string Name, double Gain)> FeatureImportance);

    public static TrainedClassifier Train(DryWindowDataset ds, Hyperparameters hp)
        => Train(ds, hp, DryWindowFeatureBuilder.Phase3b);

    /// <summary>
    /// Train a dry-window classifier under the named phase. <c>3b</c> and
    /// <c>3d-calibrated</c> use the 53-column base feature set; <c>3d-shape</c>
    /// adds the 7 ensemble-mean shape columns.
    /// </summary>
    public static TrainedClassifier Train(DryWindowDataset ds, Hyperparameters hp, string phase)
    {
        var ml = new MLContext(seed: hp.Seed);

        var trainDv = ml.Data.LoadFromEnumerable(ds.Train);
        var valDv   = ml.Data.LoadFromEnumerable(ds.Val);

        var featureNames = DryWindowFeatureBuilder.FeatureNamesForPhase(phase).ToArray();

        var featurize = ml.Transforms.Concatenate("Features", FeatureColumnInternalNames(phase));
        var featModel = featurize.Fit(trainDv);
        var trainFeat = featModel.Transform(trainDv);
        var valFeat   = featModel.Transform(valDv);

        // UnbalancedSets=false matches 3a — we want well-calibrated probabilities
        // for Brier/reliability, not a positive-class scale factor. For the
        // day-level target with >80% positive rate the imbalance is the other
        // direction (too many 1s). Still, keep it off — the mean-of-models /
        // climatology baselines are the honest reference.
        var options = new LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = "Features",
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
        var predictor = trainer.Fit(trainFeat, valFeat);
        var fullModel = featModel.Append(predictor);

        var weights = default(VBuffer<float>);
        predictor.Model.SubModel.GetFeatureWeights(ref weights);
        var importance = weights.DenseValues()
            .Select((g, i) => (Name: featureNames[i], Gain: (double)g))
            .OrderByDescending(t => t.Gain)
            .ToArray();

        return new TrainedClassifier(ml, fullModel, trainDv.Schema, hp, featureNames, importance);
    }

    public static double[] PredictProbability(MLContext ml, ITransformer model, IReadOnlyList<DryWindowTrainingRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var dv = ml.Data.LoadFromEnumerable(rows);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Probability").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }

    /// <summary>
    /// Internal ML.NET column names in schema order for the given phase. 3b /
    /// 3d-calibrated emit the 53-column base; 3d-shape appends the 7 shape
    /// columns. Must stay aligned with
    /// <see cref="DryWindowFeatureBuilder.FeatureNamesForPhase"/>.
    /// </summary>
    private static string[] FeatureColumnInternalNames(string phase)
    {
        var baseCols = BaseFeatureColumnInternalNames();
        if (phase != DryWindowFeatureBuilder.Phase3dShape) return baseCols;
        return baseCols.Concat(new[]
        {
            nameof(DryWindowTrainingRow.FirstWetHour),
            nameof(DryWindowTrainingRow.LastWetHour),
            nameof(DryWindowTrainingRow.LongestForecastDryRunHours),
            nameof(DryWindowTrainingRow.LongestForecastWetRunHours),
            nameof(DryWindowTrainingRow.NRainEvents),
            nameof(DryWindowTrainingRow.MorningPrecipSum),
            nameof(DryWindowTrainingRow.AfternoonPrecipSum),
        }).ToArray();
    }

    private static string[] BaseFeatureColumnInternalNames() => new[]
    {
        nameof(DryWindowTrainingRow.PrecipSumGfs),
        nameof(DryWindowTrainingRow.PrecipSumEcmwf),
        nameof(DryWindowTrainingRow.PrecipSumIcon),
        nameof(DryWindowTrainingRow.PrecipSumMf),
        nameof(DryWindowTrainingRow.PrecipSumUkmo),
        nameof(DryWindowTrainingRow.PrecipSumGem),

        nameof(DryWindowTrainingRow.PrecipMaxHourGfs),
        nameof(DryWindowTrainingRow.PrecipMaxHourEcmwf),
        nameof(DryWindowTrainingRow.PrecipMaxHourIcon),
        nameof(DryWindowTrainingRow.PrecipMaxHourMf),
        nameof(DryWindowTrainingRow.PrecipMaxHourUkmo),
        nameof(DryWindowTrainingRow.PrecipMaxHourGem),

        nameof(DryWindowTrainingRow.WetHourCountGfs),
        nameof(DryWindowTrainingRow.WetHourCountEcmwf),
        nameof(DryWindowTrainingRow.WetHourCountIcon),
        nameof(DryWindowTrainingRow.WetHourCountMf),
        nameof(DryWindowTrainingRow.WetHourCountUkmo),
        nameof(DryWindowTrainingRow.WetHourCountGem),

        nameof(DryWindowTrainingRow.LongestDryRunGfs),
        nameof(DryWindowTrainingRow.LongestDryRunEcmwf),
        nameof(DryWindowTrainingRow.LongestDryRunIcon),
        nameof(DryWindowTrainingRow.LongestDryRunMf),
        nameof(DryWindowTrainingRow.LongestDryRunUkmo),
        nameof(DryWindowTrainingRow.LongestDryRunGem),

        nameof(DryWindowTrainingRow.HasDryWindowGfs),
        nameof(DryWindowTrainingRow.HasDryWindowEcmwf),
        nameof(DryWindowTrainingRow.HasDryWindowIcon),
        nameof(DryWindowTrainingRow.HasDryWindowMf),
        nameof(DryWindowTrainingRow.HasDryWindowUkmo),
        nameof(DryWindowTrainingRow.HasDryWindowGem),

        nameof(DryWindowTrainingRow.ProbMaxGfs),
        nameof(DryWindowTrainingRow.ProbMaxEcmwf),
        nameof(DryWindowTrainingRow.ProbMaxIcon),
        nameof(DryWindowTrainingRow.ProbMaxMf),
        nameof(DryWindowTrainingRow.ProbMaxUkmo),
        nameof(DryWindowTrainingRow.ProbMaxGem),

        nameof(DryWindowTrainingRow.PrecipSumMean),
        nameof(DryWindowTrainingRow.PrecipSumStd),
        nameof(DryWindowTrainingRow.PrecipSumMax),
        nameof(DryWindowTrainingRow.AgreementHasDryWindow),
        nameof(DryWindowTrainingRow.LongestDryRunMean),
        nameof(DryWindowTrainingRow.WetHourCountMean),

        nameof(DryWindowTrainingRow.RhMean),
        nameof(DryWindowTrainingRow.RhMin),
        nameof(DryWindowTrainingRow.DewDepressionMax),
        nameof(DryWindowTrainingRow.CloudLowMean),
        nameof(DryWindowTrainingRow.CloudMidMean),
        nameof(DryWindowTrainingRow.CloudHighMean),
        nameof(DryWindowTrainingRow.CapeMax),
        nameof(DryWindowTrainingRow.WindMean),
        nameof(DryWindowTrainingRow.WindMax),

        nameof(DryWindowTrainingRow.DoySin),
        nameof(DryWindowTrainingRow.DoyCos),
    };
}
