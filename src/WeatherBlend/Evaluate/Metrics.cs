namespace WeatherBlend.Evaluate;

/// <summary>
/// Error metrics for temperature forecasts. All operate on paired
/// (predicted, actual) sequences and silently ignore NaN pairs (rows
/// where persistence lookups failed etc.).
/// </summary>
public static class Metrics
{
    public readonly record struct ErrorStats(double Mae, double Rmse, double Bias, int N);

    public static ErrorStats Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        if (predicted.Count != actual.Count)
            throw new ArgumentException(
                $"Length mismatch: predicted={predicted.Count}, actual={actual.Count}");

        double sumAbs = 0, sumSq = 0, sumSigned = 0;
        int n = 0;
        for (int i = 0; i < predicted.Count; i++)
        {
            var p = predicted[i];
            var a = actual[i];
            if (double.IsNaN(p) || double.IsNaN(a)) continue;
            var err = p - a;
            sumAbs += Math.Abs(err);
            sumSq += err * err;
            sumSigned += err;
            n++;
        }

        if (n == 0) return new ErrorStats(double.NaN, double.NaN, double.NaN, 0);

        return new ErrorStats(
            Mae: sumAbs / n,
            Rmse: Math.Sqrt(sumSq / n),
            Bias: sumSigned / n,
            N: n);
    }

    /// <summary>MAE stratified by an arbitrary bucket key. Returns per-bucket (mae, n).</summary>
    public static SortedDictionary<TKey, (double Mae, int N)> StratifiedMae<TKey>(
        IReadOnlyList<double> predicted,
        IReadOnlyList<double> actual,
        IReadOnlyList<TKey> bucketKeys)
        where TKey : notnull, IComparable<TKey>
    {
        if (predicted.Count != actual.Count || predicted.Count != bucketKeys.Count)
            throw new ArgumentException(
                $"Length mismatch: predicted={predicted.Count}, actual={actual.Count}, keys={bucketKeys.Count}");

        var buckets = new Dictionary<TKey, (double Sum, int N)>();
        for (int i = 0; i < predicted.Count; i++)
        {
            var p = predicted[i];
            var a = actual[i];
            if (double.IsNaN(p) || double.IsNaN(a)) continue;
            var k = bucketKeys[i];
            buckets.TryGetValue(k, out var cur);
            buckets[k] = (cur.Sum + Math.Abs(p - a), cur.N + 1);
        }

        var result = new SortedDictionary<TKey, (double Mae, int N)>();
        foreach (var (k, v) in buckets)
            result[k] = (v.Sum / v.N, v.N);
        return result;
    }

    /// <summary>
    /// Quintile-bucket the input values and assign each row a bucket label 0..4 by value.
    /// Used for spread-bucket stratification (does the blend help most when models disagree?).
    /// </summary>
    public static int[] Quintiles(IReadOnlyList<double> values)
    {
        var sorted = values
            .Select((v, i) => (v, i))
            .Where(t => !double.IsNaN(t.v))
            .OrderBy(t => t.v)
            .ToArray();

        var labels = new int[values.Count];
        for (int i = 0; i < labels.Length; i++) labels[i] = -1;

        if (sorted.Length == 0) return labels;

        // Assign each sorted item to a quintile by position.
        for (int rank = 0; rank < sorted.Length; rank++)
        {
            var q = (int)Math.Min(4, (long)rank * 5 / sorted.Length);
            labels[sorted[rank].i] = q;
        }
        return labels;
    }

    /// <summary>
    /// Wind-direction quadrant: N [315..45), E [45..135), S [135..225), W [225..315).
    /// Returns "N"/"E"/"S"/"W" or "?" for NaN.
    /// </summary>
    public static string WindQuadrant(double directionDeg)
    {
        if (double.IsNaN(directionDeg)) return "?";
        var d = ((directionDeg % 360) + 360) % 360;
        if (d >= 315 || d < 45) return "N";
        if (d < 135) return "E";
        if (d < 225) return "S";
        return "W";
    }
}
