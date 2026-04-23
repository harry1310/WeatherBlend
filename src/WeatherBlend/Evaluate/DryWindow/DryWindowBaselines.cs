using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Evaluate.DryWindow;

/// <summary>
/// Non-learned baselines for the dry-window blender. Probabilities live in
/// [0,1]; binary model self-predictions are mapped to {0,1} just like 3a's
/// per-model wet indicator.
/// </summary>
public static class DryWindowBaselines
{
    public static double[] SingleModelHasDryWindow(IReadOnlyList<DryWindowTrainingRow> rows, string col) => col switch
    {
        "has_dry_window_gfs"   => rows.Select(r => (double)r.HasDryWindowGfs  ).ToArray(),
        "has_dry_window_ecmwf" => rows.Select(r => (double)r.HasDryWindowEcmwf).ToArray(),
        "has_dry_window_icon"  => rows.Select(r => (double)r.HasDryWindowIcon ).ToArray(),
        "has_dry_window_mf"    => rows.Select(r => (double)r.HasDryWindowMf   ).ToArray(),
        "has_dry_window_ukmo"  => rows.Select(r => (double)r.HasDryWindowUkmo ).ToArray(),
        "has_dry_window_gem"   => rows.Select(r => (double)r.HasDryWindowGem  ).ToArray(),
        _ => throw new ArgumentException($"Unknown model column: {col}")
    };

    /// <summary>Mean-of-models: fraction of ensemble members predicting a dry window exists.</summary>
    public static double[] MeanOfModels(IReadOnlyList<DryWindowTrainingRow> rows)
        => rows.Select(r => float.IsNaN(r.AgreementHasDryWindow) ? 0.0 : (double)r.AgreementHasDryWindow).ToArray();

    /// <summary>Month-keyed climatology from <paramref name="trainRows"/> evaluated at each target row's date.</summary>
    public static double[] Climatology(
        IReadOnlyList<DryWindowTrainingRow> trainRows,
        IReadOnlyList<DryWindowTrainingRow> targetRows,
        int windowHours)
    {
        var clim = DryWindowClimatology.BuildFromTraining(trainRows, windowHours);
        var p = new double[targetRows.Count];
        for (int i = 0; i < targetRows.Count; i++)
            p[i] = clim.Predict(targetRows[i].TargetDateUtc);
        return p;
    }

    /// <summary>
    /// Persistence: predict today's label from yesterday's truth label. Looks up
    /// via <paramref name="labelByDate"/>; returns NaN when yesterday is missing.
    /// </summary>
    public static double[] Persistence(
        IReadOnlyList<DryWindowTrainingRow> rows,
        IReadOnlyDictionary<DateOnly, bool> labelByDate,
        int lagDays = 1)
    {
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++)
        {
            var key = DateOnly.FromDateTime(rows[i].TargetDateUtc).AddDays(-lagDays);
            if (labelByDate.TryGetValue(key, out var v))
                p[i] = v ? 1.0 : 0.0;
            else
                p[i] = double.NaN;
        }
        return p;
    }

    /// <summary>Column name of the single model with lowest Brier over <paramref name="rows"/>.</summary>
    public static string BestSingle(IReadOnlyList<DryWindowTrainingRow> rows)
    {
        var truth = rows.Select(r => r.HasDryWindow ? 1.0 : 0.0).ToArray();
        string best = "has_dry_window_ecmwf";
        double bestBrier = double.PositiveInfinity;
        foreach (var col in new[]
        {
            "has_dry_window_gfs", "has_dry_window_ecmwf", "has_dry_window_icon",
            "has_dry_window_mf",  "has_dry_window_ukmo",  "has_dry_window_gem",
        })
        {
            var b = WeatherBlend.Evaluate.Precip.PrecipMetrics.Brier(SingleModelHasDryWindow(rows, col), truth);
            if (!double.IsNaN(b) && b < bestBrier)
            {
                bestBrier = b;
                best = col;
            }
        }
        return best;
    }

}
