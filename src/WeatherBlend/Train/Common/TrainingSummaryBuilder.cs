namespace WeatherBlend.Train.Common;

/// <summary>
/// Build a <see cref="TrainingSummary"/> from a feature matrix + names.
/// Decouples summary computation from the dataset type so binary
/// (<see cref="BinaryDataset"/>) + regression (<see cref="RegressionDataset"/>)
/// + per-lead-loop trainers (which buffer features across leads) can all
/// feed in. Computes everything on the TRAIN slice only.
///
/// NaN handling: float.NaN is treated as missing. NanPct = fraction of
/// rows where the column was NaN. Mean/std/quantiles are computed over the
/// non-NaN subset. A column that's 100% NaN gets {NanPct=1.0, Mean=0,
/// Std=0, P01=0, P99=0} — won't trip the guard's "feature appeared/
/// disappeared" tolerance because FeaturesEffective is dimensioned at the
/// caller (post-drop), but it does land an obvious zero-row sentinel in
/// PerFeature so the previous summary's stats aren't silently inherited.
/// </summary>
public static class TrainingSummaryBuilder
{
    /// <summary>
    /// Compute the per-feature block of a TrainingSummary from a matrix of
    /// train-slice feature vectors. Each <paramref name="trainFeatures"/>
    /// row must have <paramref name="featureNames"/>.Count entries — caller
    /// is responsible for shape consistency (the guard's FeaturesEffective
    /// check is the late-binding consistency tripwire).
    ///
    /// Allocates one double[] per feature column for the non-NaN values,
    /// sorts them once for quantiles, computes mean/std with Welford-style
    /// numerical stability. Memory is O(rows × features); fine at the row
    /// counts seen here (~70k train × ~25 features = 1.7M floats = 13 MB).
    /// </summary>
    public static Dictionary<string, FeatureStats> ComputeFeatureStats(
        IReadOnlyList<float[]> trainFeatures,
        IReadOnlyList<string> featureNames)
    {
        if (trainFeatures.Count == 0)
            throw new ArgumentException("Cannot compute stats over an empty train slice.", nameof(trainFeatures));
        var nFeatures = featureNames.Count;
        var nRows = trainFeatures.Count;

        // Per-column buckets of non-NaN values. Allocate at full row count
        // to avoid List<double> resizing pressure for the dense (no-NaN)
        // common case — tail unused for sparse cols.
        var values = new double[nFeatures][];
        var counts = new int[nFeatures];
        var nanCounts = new int[nFeatures];
        for (int j = 0; j < nFeatures; j++) values[j] = new double[nRows];

        for (int i = 0; i < nRows; i++)
        {
            var row = trainFeatures[i];
            if (row.Length != nFeatures)
                throw new InvalidOperationException(
                    $"Feature row {i} has {row.Length} columns but expected {nFeatures}");
            for (int j = 0; j < nFeatures; j++)
            {
                var v = row[j];
                if (float.IsNaN(v)) nanCounts[j]++;
                else values[j][counts[j]++] = v;
            }
        }

        var stats = new Dictionary<string, FeatureStats>(nFeatures, StringComparer.Ordinal);
        for (int j = 0; j < nFeatures; j++)
        {
            var nNonNan = counts[j];
            var nanPct = nanCounts[j] / (double)nRows;

            if (nNonNan == 0)
            {
                stats[featureNames[j]] = new FeatureStats
                {
                    NanPct = nanPct, Mean = 0, Std = 0, P01 = 0, P99 = 0,
                };
                continue;
            }

            // Welford for mean/std — stable for large n.
            double mean = 0, m2 = 0;
            for (int k = 0; k < nNonNan; k++)
            {
                var v = values[j][k];
                var delta = v - mean;
                mean += delta / (k + 1);
                m2 += delta * (v - mean);
            }
            var variance = nNonNan > 1 ? m2 / (nNonNan - 1) : 0;
            var std = Math.Sqrt(variance);

            // Quantiles via in-place sort of the non-NaN slice. Linear
            // interpolation (R type 7) for in-between positions.
            Array.Sort(values[j], 0, nNonNan);
            stats[featureNames[j]] = new FeatureStats
            {
                NanPct = nanPct,
                Mean = mean,
                Std = std,
                P01 = Quantile(values[j], nNonNan, 0.01),
                P99 = Quantile(values[j], nNonNan, 0.99),
            };
        }

        return stats;
    }

    /// <summary>R-7 (linear interp) quantile on a pre-sorted array slice.
    /// Matches numpy.quantile default and pandas.quantile default.</summary>
    private static double Quantile(double[] sorted, int n, double q)
    {
        if (n == 1) return sorted[0];
        var pos = q * (n - 1);
        var lo = (int)Math.Floor(pos);
        var hi = (int)Math.Ceiling(pos);
        if (lo == hi) return sorted[lo];
        var frac = pos - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
    }

    /// <summary>
    /// Compose a complete TrainingSummary. Convenience wrapper that ties
    /// the per-feature stats to the metadata fields. Caller passes the
    /// already-aggregated row counts and feature-effective count; this
    /// helper does the column-stats math + binds it all together.
    /// </summary>
    public static TrainingSummary Build(
        string composite, string phase, string version,
        DateTime computedAtUtc,
        int rowsTrain, int rowsVal, int rowsTest,
        IReadOnlyList<float[]> trainFeatures,
        IReadOnlyList<string> featureNames,
        Dictionary<string, double>? labelRates = null)
    {
        return new TrainingSummary
        {
            SchemaVersion = "1",
            Composite = composite,
            Phase = phase,
            Version = version,
            ComputedAtUtc = computedAtUtc,
            RowsTrain = rowsTrain,
            RowsVal = rowsVal,
            RowsTest = rowsTest,
            FeaturesEffective = featureNames.Count,
            PerFeature = ComputeFeatureStats(trainFeatures, featureNames),
            LabelRates = labelRates ?? new Dictionary<string, double>(),
        };
    }

    /// <summary>
    /// One-liner the trainers call right after SaveTrainingMetadata — does
    /// the Build + SaveTrainingSummary pair atomically. Silently no-ops
    /// when the buffered features are empty (e.g. a partial dataset that
    /// failed validation but still reached this point); the missing
    /// summary on disk is treated as "first-ever train" by the upcoming
    /// retrain guard, which is the correct degraded behaviour.
    ///
    /// versionDir is the same dir SaveTrainingMetadata writes into;
    /// summary lands as training_summary.json alongside.
    /// </summary>
    public static void BuildAndSave(
        string versionDir,
        string composite, string phase, string version,
        DateTime computedAtUtc,
        int rowsTrain, int rowsVal, int rowsTest,
        IReadOnlyList<float[]>? trainFeatures,
        IReadOnlyList<string> featureNames,
        Dictionary<string, double>? labelRates = null)
    {
        if (trainFeatures is null || trainFeatures.Count == 0
            || featureNames.Count == 0)
        {
            return;
        }
        var summary = Build(
            composite, phase, version, computedAtUtc,
            rowsTrain, rowsVal, rowsTest,
            trainFeatures, featureNames, labelRates);
        WeatherBlend.Train.ModelArtifact.SaveTrainingSummary(versionDir, summary);
    }
}
