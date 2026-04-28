using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Evaluate.Precip;

/// <summary>
/// Non-learned baselines the precipitation occurrence blender must beat.
///
/// Deterministic wet/dry predictions are mapped to probabilities {0,1}; this
/// is equivalent to what each raw forecast offers (precipitation&gt;=0.1mm is
/// the model saying "I expect a wet hour"). Climatology is the probabilistic
/// reference — base rate by (month, hour-of-day) computed on training rows only.
/// </summary>
public static class PrecipBaselines
{
    public const double WetThresholdMm = PrecipFeatureBuilder.WetThresholdMm;

    /// <summary>
    /// Per-model wet-indicator probability. Pulls precip_<short> from
    /// <see cref="BinaryTrainingRow.Features"/> and maps to {0, 1, NaN} via
    /// the 0.1 mm threshold.
    /// </summary>
    public static double[] SingleModelWet(
        BlenderSpec spec,
        IReadOnlyList<BinaryTrainingRow> rows,
        string featureName)
    {
        var idx = spec.IndexOf(featureName);
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++) p[i] = Indicate(rows[i].Features[idx]);
        return p;
    }

    /// <summary>Mean-of-models = ensemble agreement on wet01 (already a feature).</summary>
    public static double[] MeanOfModels(BlenderSpec spec, IReadOnlyList<BinaryTrainingRow> rows)
    {
        var idx = spec.IndexOf("precip_agreement_wet_01");
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++)
        {
            var v = rows[i].Features[idx];
            p[i] = float.IsNaN(v) ? 0.0 : v;
        }
        return p;
    }

    /// <summary>Climatology baseline against vector-row datasets — P(wet) per (month, hour-of-day).</summary>
    public static double[] Climatology(
        IReadOnlyList<BinaryTrainingRow> trainRows,
        IReadOnlyList<BinaryTrainingRow> targetRows)
    {
        var sums = new Dictionary<(int Month, int Hour), (int Wet, int N)>();
        foreach (var r in trainRows)
        {
            var k = (r.ValidTimeUtc.Month, r.ValidTimeUtc.Hour);
            sums.TryGetValue(k, out var cur);
            sums[k] = (cur.Wet + (r.Label ? 1 : 0), cur.N + 1);
        }
        double globalRate = trainRows.Count > 0 ? trainRows.Count(r => r.Label) / (double)trainRows.Count : 0.0;
        var p = new double[targetRows.Count];
        for (int i = 0; i < targetRows.Count; i++)
        {
            var k = (targetRows[i].ValidTimeUtc.Month, targetRows[i].ValidTimeUtc.Hour);
            p[i] = sums.TryGetValue(k, out var cur) && cur.N > 0
                ? cur.Wet / (double)cur.N
                : globalRate;
        }
        return p;
    }

    /// <summary>
    /// Pick the per-model precip feature with lowest Brier on the rows.
    /// Iterates the first N feature names (per-model precip slots; spec.Models
    /// in canonical order).
    /// </summary>
    public static string BestSingle(BlenderSpec spec, IReadOnlyList<BinaryTrainingRow> rows)
    {
        var truth = rows.Select(r => r.Label ? 1.0 : 0.0).ToArray();
        string best = spec.FeatureNames[0];
        double bestBrier = double.PositiveInfinity;
        for (int i = 0; i < spec.Models.Count; i++)
        {
            var name = spec.FeatureNames[i];   // per-model precip slot
            var b = PrecipMetrics.Brier(SingleModelWet(spec, rows, name), truth);
            if (!double.IsNaN(b) && b < bestBrier) { bestBrier = b; best = name; }
        }
        return best;
    }

    private static double Indicate(float precipMm)
        => float.IsNaN(precipMm) ? double.NaN : (precipMm >= WetThresholdMm ? 1.0 : 0.0);
}
