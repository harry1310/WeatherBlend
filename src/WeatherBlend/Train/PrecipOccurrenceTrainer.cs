using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// LightGBM binary classifier for P(wet hour). Feature set is caller-supplied:
///   • Phase 3a — <see cref="LeanFeatureColumnInternalNames"/> (27 columns on <see cref="PrecipTrainingRow"/>)
///   • Phase 3c — <see cref="RichFeatureColumnInternalNames"/> (55 columns on <see cref="RichPrecipTrainingRow"/>)
/// Label = WetBinary; both row types expose it through the base <see cref="PrecipTrainingRow"/>.
///
/// Microsoft.ML.LightGbm 4.0 exposes only a subset of the raw LightGBM options.
/// Deviations from the brief's "raw LightGBM" ask:
///   - No native class-weight parameter. With ~25% wet hours the class
///     imbalance is moderate; we rely on LightGBM's native handling. If calibration
///     suffers we can revisit with sigmoid/Platt scaling on the val set.
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
    /// Phase 3a call site: lean 27-feature dataset. Thin wrapper over the generic
    /// <see cref="Train{T}"/> with the lean feature-column list.
    /// Legacy code path — superseded by <see cref="TrainVector"/> after the
    /// unify-model-membership refactor (Phase 2+).
    /// </summary>
    [Obsolete("Use TrainVector with BinaryTrainingRow instead.")]
    public static TrainedClassifier Train(PrecipDataset ds, Hyperparameters hp)
        => Train(ds.Train, ds.Val, LeanFeatureColumnInternalNames(), hp);

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

    /// <summary>
    /// Generic LightGBM fit over arbitrary rows + feature column list. Used by
    /// Phase 3a (27 lean cols) and Phase 3c (55 rich cols). Each named column must
    /// exist as a public property of <typeparamref name="T"/>; the type must also
    /// expose <c>[ColumnName("Label")]</c> on the binary label — both lean and
    /// rich row types inherit this from <see cref="PrecipTrainingRow"/>.
    /// </summary>
    public static TrainedClassifier Train<T>(
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

        var featurize = ml.Transforms.Concatenate("Features", featureNames);
        var featModel = featurize.Fit(trainDv);
        var trainFeat = featModel.Transform(trainDv);
        var valFeat   = featModel.Transform(valDv);

        // UnbalancedSets deliberately off: at 27% wet-hour rate the class imbalance
        // is moderate and flipping this on pushes probabilities too high (measured:
        // freq bias 1.31 vs 1.05 off, Brier 0.148 vs 0.137 off — mean-of-models
        // baseline actually beats the blend when UnbalancedSets is on). Keep it
        // off so Brier optimisation isn't fighting a positive-class scale factor.
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
                SubsampleFraction = hp.SubsampleFraction,
                SubsampleFrequency = hp.SubsampleFrequency,
                FeatureFraction = hp.FeatureFraction,
            },
        };

        var trainer = ml.BinaryClassification.Trainers.LightGbm(options);
        var predictor = trainer.Fit(trainFeat, valFeat);
        var fullModel = featModel.Append(predictor);

        // Binary LightGBM comes wrapped in a Platt calibrator; feature weights live on
        // the inner SubModel.
        var weights = default(VBuffer<float>);
        predictor.Model.SubModel.GetFeatureWeights(ref weights);
        var importance = weights.DenseValues()
            .Select((g, i) => (Name: featureNames[i], Gain: (double)g))
            .OrderByDescending(t => t.Gain)
            .ToArray();

        return new TrainedClassifier(
            Ml: ml,
            Model: fullModel,
            InputSchema: trainDv.Schema,
            Hyperparameters: hp,
            FeatureNames: featureNames,
            FeatureImportance: importance);
    }

    /// <summary>
    /// Returns calibrated probabilities in [0,1]. LightGBM binary outputs already
    /// pass through a sigmoid via the ML.NET PlattCalibrator.
    /// </summary>
    public static double[] PredictProbability<T>(MLContext ml, ITransformer model, IReadOnlyList<T> rows)
        where T : class
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var dv = ml.Data.LoadFromEnumerable(rows);
        var predicted = model.Transform(dv);
        var scores = predicted.GetColumn<float>("Probability").ToArray();
        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++) result[i] = scores[i];
        return result;
    }

    /// <summary>Lean (Phase 3a) feature property names — 27 columns on <see cref="PrecipTrainingRow"/>.</summary>
    public static string[] LeanFeatureColumnInternalNames() => new[]
    {
        nameof(PrecipTrainingRow.PrecipGfs),
        nameof(PrecipTrainingRow.PrecipEcmwf),
        nameof(PrecipTrainingRow.PrecipIcon),
        nameof(PrecipTrainingRow.PrecipMf),
        nameof(PrecipTrainingRow.PrecipUkmo),
        nameof(PrecipTrainingRow.PrecipGem),
        nameof(PrecipTrainingRow.ProbGfs),
        nameof(PrecipTrainingRow.ProbEcmwf),
        nameof(PrecipTrainingRow.ProbIcon),
        nameof(PrecipTrainingRow.ProbMf),
        nameof(PrecipTrainingRow.ProbUkmo),
        nameof(PrecipTrainingRow.ProbGem),
        nameof(PrecipTrainingRow.PrecipMean),
        nameof(PrecipTrainingRow.PrecipStd),
        nameof(PrecipTrainingRow.PrecipMax),
        nameof(PrecipTrainingRow.PrecipAgreementWet01),
        nameof(PrecipTrainingRow.RhMean),
        nameof(PrecipTrainingRow.DewDepressionMean),
        nameof(PrecipTrainingRow.CloudLowMean),
        nameof(PrecipTrainingRow.CloudMidMean),
        nameof(PrecipTrainingRow.CloudHighMean),
        nameof(PrecipTrainingRow.CapeMean),
        nameof(PrecipTrainingRow.WindSpeedMean),
        nameof(PrecipTrainingRow.HourSin),
        nameof(PrecipTrainingRow.HourCos),
        nameof(PrecipTrainingRow.DoySin),
        nameof(PrecipTrainingRow.DoyCos),
    };

    /// <summary>
    /// Rich (Phase 3c) feature property names — 55 columns on <see cref="RichPrecipTrainingRow"/>.
    /// Order matches <see cref="PrecipRichFeatureBuilder.OccurrenceFeatureNames"/>.
    /// </summary>
    public static string[] RichFeatureColumnInternalNames()
    {
        var lean = LeanFeatureColumnInternalNames();
        var extras = new[]
        {
            // 18 per-model humidity.
            nameof(RichPrecipTrainingRow.DewGfs), nameof(RichPrecipTrainingRow.DewEcmwf), nameof(RichPrecipTrainingRow.DewIcon),
            nameof(RichPrecipTrainingRow.DewMf),  nameof(RichPrecipTrainingRow.DewUkmo),  nameof(RichPrecipTrainingRow.DewGem),

            nameof(RichPrecipTrainingRow.RhGfs), nameof(RichPrecipTrainingRow.RhEcmwf), nameof(RichPrecipTrainingRow.RhIcon),
            nameof(RichPrecipTrainingRow.RhMf),  nameof(RichPrecipTrainingRow.RhUkmo),  nameof(RichPrecipTrainingRow.RhGem),

            nameof(RichPrecipTrainingRow.DewDepressionGfs), nameof(RichPrecipTrainingRow.DewDepressionEcmwf),
            nameof(RichPrecipTrainingRow.DewDepressionIcon), nameof(RichPrecipTrainingRow.DewDepressionMf),
            nameof(RichPrecipTrainingRow.DewDepressionUkmo), nameof(RichPrecipTrainingRow.DewDepressionGem),

            // 6 per-model pressure.
            nameof(RichPrecipTrainingRow.PressureGfs), nameof(RichPrecipTrainingRow.PressureEcmwf),
            nameof(RichPrecipTrainingRow.PressureIcon), nameof(RichPrecipTrainingRow.PressureMf),
            nameof(RichPrecipTrainingRow.PressureUkmo), nameof(RichPrecipTrainingRow.PressureGem),

            // 4 EA observation persistence.
            nameof(RichPrecipTrainingRow.EaRainPrev24hMm),
            nameof(RichPrecipTrainingRow.EaRainPrev72hMm),
            nameof(RichPrecipTrainingRow.EaWetHoursLast24h),
            nameof(RichPrecipTrainingRow.EaDryHoursTrailing),
        };
        return lean.Concat(extras).ToArray();
    }
}
