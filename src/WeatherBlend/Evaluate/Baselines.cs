using WeatherBlend.Train;

namespace WeatherBlend.Evaluate;

/// <summary>
/// Non-learned baselines the blender must beat to earn its keep.
///
/// Persistence: ERA5(T - lagHours), where lagHours matches the forecast lead
/// being evaluated (24/48/72). Per-lead evaluation passes the lead directly;
/// the default lag of 24h is the "same hour yesterday" fallback.
/// </summary>
public static class Baselines
{
    /// <summary>
    /// Persistence: predict ERA5(T) = ERA5(T - lagHours). Looks up by ValidTimeUtc
    /// in a full truth map built from train+val+test (lookup is deterministic and
    /// doesn't leak anything — the target is the ERA5 value at T, and we use
    /// the ERA5 value at T-24h which is already observed at train time).
    /// </summary>
    public static double[] Persistence(
        IReadOnlyList<TrainingRow> rows,
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
    /// Climatology: mean ERA5 temperature per (month, hour-of-day) across the
    /// training set. Predicts that mean for every test row. Important: use
    /// training set only — computing from test leaks.
    /// </summary>
    public static double[] Climatology(
        IReadOnlyList<TrainingRow> trainRows,
        IReadOnlyList<TrainingRow> targetRows)
    {
        var sums = new Dictionary<(int Month, int Hour), (double Sum, int N)>();
        foreach (var r in trainRows)
        {
            var k = (r.ValidTimeUtc.Month, r.ValidTimeUtc.Hour);
            sums.TryGetValue(k, out var cur);
            sums[k] = (cur.Sum + r.Era5Temp, cur.N + 1);
        }
        var means = sums.ToDictionary(kv => kv.Key, kv => kv.Value.Sum / kv.Value.N);

        // Global mean as a fallback for (month, hour) combinations unseen in training.
        double globalMean = trainRows.Count > 0 ? trainRows.Average(r => (double)r.Era5Temp) : double.NaN;

        var p = new double[targetRows.Count];
        for (int i = 0; i < targetRows.Count; i++)
        {
            var k = (targetRows[i].ValidTimeUtc.Month, targetRows[i].ValidTimeUtc.Hour);
            p[i] = means.TryGetValue(k, out var m) ? m : globalMean;
        }
        return p;
    }

    /// <summary>Mean-of-models: the arithmetic mean of the 6 per-model forecasts (already a feature).</summary>
    public static double[] MeanOfModels(IReadOnlyList<TrainingRow> rows)
        => rows.Select(r => (double)r.TempMean).ToArray();

    /// <summary>Extract one per-model forecast column as a prediction series.</summary>
    public static double[] SingleModel(IReadOnlyList<TrainingRow> rows, string col) => col switch
    {
        "temp_gfs"   => rows.Select(r => (double)r.TempGfs  ).ToArray(),
        "temp_ecmwf" => rows.Select(r => (double)r.TempEcmwf).ToArray(),
        "temp_icon"  => rows.Select(r => (double)r.TempIcon ).ToArray(),
        "temp_mf"    => rows.Select(r => (double)r.TempMf   ).ToArray(),
        "temp_ukmo"  => rows.Select(r => (double)r.TempUkmo ).ToArray(),
        "temp_gem"   => rows.Select(r => (double)r.TempGem  ).ToArray(),
        _ => throw new ArgumentException($"Unknown model column: {col}")
    };

    /// <summary>
    /// Pick the single-model column with lowest MAE on the provided rows.
    /// Called on the validation set to select "best single" for the report.
    /// </summary>
    public static string BestSingle(IReadOnlyList<TrainingRow> rows)
    {
        var actual = rows.Select(r => (double)r.Era5Temp).ToArray();
        string best = "temp_ecmwf";
        double bestMae = double.PositiveInfinity;
        foreach (var (_, col) in FeatureBuilder.ModelColumns)
        {
            var mae = Metrics.Compute(SingleModel(rows, col), actual).Mae;
            if (mae < bestMae)
            {
                bestMae = mae;
                best = col;
            }
        }
        return best;
    }
}
