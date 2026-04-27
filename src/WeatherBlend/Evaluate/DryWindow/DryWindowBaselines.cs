using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Evaluate.DryWindow;

/// <summary>
/// Non-learned baselines for the dry-window blender. Probabilities live in
/// [0,1]; binary model self-predictions are mapped to {0,1} just like 3a's
/// per-model wet indicator.
/// </summary>
public static class DryWindowBaselines
{
    public static double[] SingleModelHasDryWindow(IReadOnlyList<WeatherBlend.Train.DryWindow.DryWindowTrainingRow> rows, string col) => col switch
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
    public static double[] MeanOfModels(IReadOnlyList<WeatherBlend.Train.DryWindow.DryWindowTrainingRow> rows)
        => rows.Select(r => float.IsNaN(r.AgreementHasDryWindow) ? 0.0 : (double)r.AgreementHasDryWindow).ToArray();

    /// <summary>Month-keyed climatology from <paramref name="trainRows"/> evaluated at each target row's date.</summary>
    public static double[] Climatology(
        IReadOnlyList<WeatherBlend.Train.DryWindow.DryWindowTrainingRow> trainRows,
        IReadOnlyList<WeatherBlend.Train.DryWindow.DryWindowTrainingRow> targetRows,
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
        IReadOnlyList<WeatherBlend.Train.DryWindow.DryWindowTrainingRow> rows,
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
    public static string BestSingle(IReadOnlyList<WeatherBlend.Train.DryWindow.DryWindowTrainingRow> rows)
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

    // -----------------------------------------------------------------------
    // Vector-row API (BlenderSpec + Common.DryWindowTrainingRow). Phase 4+.
    // -----------------------------------------------------------------------

    /// <summary>Pull a named feature column from the vector — used for per-model baselines.</summary>
    public static double[] FromFeature(BlenderSpec spec, IReadOnlyList<CommonRow> rows, string featureName)
    {
        var idx = spec.IndexOf(featureName);
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++) p[i] = rows[i].Features[idx];
        return p;
    }

    /// <summary>Mean-of-models = ensemble agreement on has-dry-window (already a feature).</summary>
    public static double[] MeanOfModels(BlenderSpec spec, IReadOnlyList<CommonRow> rows)
    {
        var idx = spec.IndexOf("agreement_has_dry_window");
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++)
        {
            var v = rows[i].Features[idx];
            p[i] = float.IsNaN(v) ? 0.0 : v;
        }
        return p;
    }

    /// <summary>Climatology baseline against the new vector-row dataset.</summary>
    public static double[] Climatology(
        IReadOnlyList<CommonRow> trainRows,
        IReadOnlyList<CommonRow> targetRows,
        int windowHours)
    {
        var sums = new Dictionary<int, (int Pos, int N)>();
        foreach (var r in trainRows)
        {
            var k = r.TargetDateUtc.Month;
            sums.TryGetValue(k, out var cur);
            sums[k] = (cur.Pos + (r.Label ? 1 : 0), cur.N + 1);
        }
        var globalRate = trainRows.Count == 0 ? 0.0 : trainRows.Count(r => r.Label) / (double)trainRows.Count;

        var p = new double[targetRows.Count];
        for (int i = 0; i < targetRows.Count; i++)
        {
            var k = targetRows[i].TargetDateUtc.Month;
            p[i] = sums.TryGetValue(k, out var cur) && cur.N > 0
                ? cur.Pos / (double)cur.N
                : globalRate;
        }
        return p;
    }

    /// <summary>
    /// Per-model has-dry-window slot with lowest Brier on the rows. Iterates
    /// the per-model "has_dry_window_X" entries in the spec — those live at
    /// index 4N..5N-1 (after sum/max_hour/wet_count/longest_dry blocks).
    /// </summary>
    public static string BestSingle(BlenderSpec spec, IReadOnlyList<CommonRow> rows)
    {
        var truth = rows.Select(r => r.Label ? 1.0 : 0.0).ToArray();
        string best = $"has_dry_window_{Train.FeatureBuilder.ShortName(spec.Models[0])}";
        double bestBrier = double.PositiveInfinity;
        foreach (var m in spec.Models)
        {
            var name = $"has_dry_window_{Train.FeatureBuilder.ShortName(m)}";
            var b = WeatherBlend.Evaluate.Precip.PrecipMetrics.Brier(FromFeature(spec, rows, name), truth);
            if (!double.IsNaN(b) && b < bestBrier) { bestBrier = b; best = name; }
        }
        return best;
    }
}
