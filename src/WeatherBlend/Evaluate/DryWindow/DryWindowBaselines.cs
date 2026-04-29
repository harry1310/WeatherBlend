using WeatherBlend.Train.Common;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Evaluate.DryWindow;

/// <summary>
/// Non-learned baselines for the dry-window blender. Probabilities live in
/// [0,1]; binary model self-predictions are mapped to {0,1} just like 3a's
/// per-model wet indicator.
/// </summary>
public static class DryWindowBaselines
{
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
        string best = $"has_dry_window_{Train.TempFeatureBuilder.ShortName(spec.Models[0])}";
        double bestBrier = double.PositiveInfinity;
        foreach (var m in spec.Models)
        {
            var name = $"has_dry_window_{Train.TempFeatureBuilder.ShortName(m)}";
            var b = WeatherBlend.Evaluate.Precip.PrecipMetrics.Brier(FromFeature(spec, rows, name), truth);
            if (!double.IsNaN(b) && b < bestBrier) { bestBrier = b; best = name; }
        }
        return best;
    }
}
