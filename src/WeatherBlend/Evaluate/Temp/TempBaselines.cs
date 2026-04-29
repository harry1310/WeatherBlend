using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Evaluate.Temp;

/// <summary>
/// Non-learned baselines the blender must beat to earn its keep.
///
/// Persistence: ERA5(T - lagHours), where lagHours matches the forecast lead
/// being evaluated (24/48/72). Per-lead evaluation passes the lead directly;
/// the default lag of 24h is the "same hour yesterday" fallback.
/// </summary>
public static class TempBaselines
{
    /// <summary>
    /// Persistence baseline. Truth lookup is at <c>row.ValidTimeUtc - lagHours</c>;
    /// ERA5(T) used as the truth at T.
    /// </summary>
    public static double[] Persistence(
        IReadOnlyList<RegressionTrainingRow> rows,
        IReadOnlyDictionary<DateTime, double> truthByTime,
        int lagHours = 24)
    {
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++)
        {
            var key = rows[i].ValidTimeUtc.AddHours(-lagHours);
            p[i] = truthByTime.TryGetValue(key, out var v) ? v : double.NaN;
        }
        return p;
    }

    /// <summary>
    /// Climatology: mean ERA5 per (month, hour-of-day) over the training set.
    /// </summary>
    public static double[] Climatology(
        IReadOnlyList<RegressionTrainingRow> trainRows,
        IReadOnlyList<RegressionTrainingRow> targetRows)
    {
        var sums = new Dictionary<(int Month, int Hour), (double Sum, int N)>();
        foreach (var r in trainRows)
        {
            var k = (r.ValidTimeUtc.Month, r.ValidTimeUtc.Hour);
            sums.TryGetValue(k, out var cur);
            sums[k] = (cur.Sum + r.Label, cur.N + 1);
        }
        var means = sums.ToDictionary(kv => kv.Key, kv => kv.Value.Sum / kv.Value.N);
        double globalMean = trainRows.Count > 0 ? trainRows.Average(r => (double)r.Label) : double.NaN;

        var p = new double[targetRows.Count];
        for (int i = 0; i < targetRows.Count; i++)
        {
            var k = (targetRows[i].ValidTimeUtc.Month, targetRows[i].ValidTimeUtc.Hour);
            p[i] = means.TryGetValue(k, out var m) ? m : globalMean;
        }
        return p;
    }

    /// <summary>Pull a named feature column from each row as a baseline series (mean-of-models, single model, ...).</summary>
    public static double[] FromFeature(
        BlenderSpec spec,
        IReadOnlyList<RegressionTrainingRow> rows,
        string featureName)
    {
        var idx = spec.IndexOf(featureName);
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++) p[i] = rows[i].Features[idx];
        return p;
    }

    /// <summary>
    /// Pick the per-model feature with lowest MAE against the row labels.
    /// Iterates the per-model feature names (the first N entries of
    /// <see cref="BlenderSpec.FeatureNames"/>) and returns the best.
    /// </summary>
    public static string BestSingle(BlenderSpec spec, IReadOnlyList<RegressionTrainingRow> rows)
    {
        var actual = rows.Select(r => (double)r.Label).ToArray();
        string best = spec.FeatureNames[0];
        double bestMae = double.PositiveInfinity;
        for (int i = 0; i < spec.Models.Count; i++)
        {
            var name = spec.FeatureNames[i];
            var mae = TempMetrics.Compute(FromFeature(spec, rows, name), actual).Mae;
            if (mae < bestMae) { bestMae = mae; best = name; }
        }
        return best;
    }
}
