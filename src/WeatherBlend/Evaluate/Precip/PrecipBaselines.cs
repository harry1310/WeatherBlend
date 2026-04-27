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
    /// Per-model probability proxy: 1 if predicted precip &gt;= 0.1mm else 0.
    /// Missing model forecasts become NaN; scoring routines skip them.
    /// </summary>
    public static double[] SingleModelWet(IReadOnlyList<PrecipTrainingRow> rows, string col) => col switch
    {
        "precip_gfs"   => rows.Select(r => Indicate(r.PrecipGfs  )).ToArray(),
        "precip_ecmwf" => rows.Select(r => Indicate(r.PrecipEcmwf)).ToArray(),
        "precip_icon"  => rows.Select(r => Indicate(r.PrecipIcon )).ToArray(),
        "precip_mf"    => rows.Select(r => Indicate(r.PrecipMf   )).ToArray(),
        "precip_ukmo"  => rows.Select(r => Indicate(r.PrecipUkmo )).ToArray(),
        "precip_gem"   => rows.Select(r => Indicate(r.PrecipGem  )).ToArray(),
        _ => throw new ArgumentException($"Unknown model column: {col}")
    };

    /// <summary>
    /// Mean-of-models probability: fraction of the 6 per-model forecasts that
    /// predict &gt;= 0.1mm (already computed as PrecipAgreementWet01).
    /// NaN counts as "not voting wet" — same convention as LightGBM's input.
    /// </summary>
    public static double[] MeanOfModels(IReadOnlyList<PrecipTrainingRow> rows)
        => rows.Select(r => double.IsNaN(r.PrecipAgreementWet01) ? 0.0 : (double)r.PrecipAgreementWet01).ToArray();

    /// <summary>
    /// Climatology probability: P(wet) per (month, hour-of-day) from training rows.
    /// Falls back to global wet rate when a (month, hour) bucket is unseen in training.
    /// </summary>
    public static double[] Climatology(
        IReadOnlyList<PrecipTrainingRow> trainRows,
        IReadOnlyList<PrecipTrainingRow> targetRows)
    {
        var clim = PrecipClimatology.BuildFromTraining(trainRows);
        var p = new double[targetRows.Count];
        for (int i = 0; i < targetRows.Count; i++)
            p[i] = clim.Predict(targetRows[i].ValidTimeUtc);
        return p;
    }

    /// <summary>
    /// Persistence: predict P(wet at T) = 1 if truth(T - lag) >= 0.1mm else 0.
    /// NaN when the lagged time isn't in the truth map.
    /// </summary>
    public static double[] Persistence(
        IReadOnlyList<PrecipTrainingRow> rows,
        IReadOnlyDictionary<DateTime, double> truthByTime,
        int lagHours = 24)
    {
        var p = new double[rows.Count];
        for (int i = 0; i < rows.Count; i++)
        {
            var key = rows[i].ValidTimeUtc.AddHours(-lagHours);
            if (truthByTime.TryGetValue(key, out var mm))
                p[i] = mm >= WetThresholdMm ? 1.0 : 0.0;
            else
                p[i] = double.NaN;
        }
        return p;
    }

    /// <summary>
    /// Pick the single-model column with lowest Brier on the provided rows.
    /// Selects on validation set for the headline "best single" comparison.
    /// </summary>
    public static string BestSingle(IReadOnlyList<PrecipTrainingRow> rows)
    {
        var truth = rows.Select(r => r.WetBinary ? 1.0 : 0.0).ToArray();
        string best = "precip_ecmwf";
        double bestBrier = double.PositiveInfinity;
        foreach (var (_, col, _) in PrecipFeatureBuilder.ModelColumns)
        {
            var b = PrecipMetrics.Brier(SingleModelWet(rows, col), truth);
            if (!double.IsNaN(b) && b < bestBrier)
            {
                bestBrier = b;
                best = col;
            }
        }
        return best;
    }

    private static double Indicate(float precipMm)
        => float.IsNaN(precipMm) ? double.NaN : (precipMm >= WetThresholdMm ? 1.0 : 0.0);

    // -----------------------------------------------------------------------
    // Vector-row API (BlenderSpec + BinaryTrainingRow). Phase 2+ canonical path.
    // -----------------------------------------------------------------------

    /// <summary>
    /// Per-model wet-indicator probability for the new vector-row path.
    /// Pulls precip_<short> from <see cref="BinaryTrainingRow.Features"/>
    /// and maps to {0, 1, NaN} via the 0.1 mm threshold.
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

    private static double Indicate(double precipMm)
        => double.IsNaN(precipMm) ? double.NaN : (precipMm >= WetThresholdMm ? 1.0 : 0.0);
}
