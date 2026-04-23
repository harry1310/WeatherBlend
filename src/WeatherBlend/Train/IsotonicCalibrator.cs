using System.Text.Json;

namespace WeatherBlend.Train;

/// <summary>
/// Monotonic isotonic regression calibrator fit via pool-adjacent-violators (PAV).
/// Given pairs of (raw score, binary truth), it learns a piecewise-constant
/// monotone-increasing map raw → calibrated-probability; we interpolate linearly
/// between knots at predict time so the output is continuous. Use one per
/// lead-time model — calibration is lead-specific because each model has its own
/// reliability curve.
/// </summary>
public sealed class IsotonicCalibrator
{
    /// <summary>Sorted ascending by <see cref="X"/>. Piecewise-constant pools from PAV; linear interpolation between midpoints at predict time.</summary>
    public List<Knot> Knots { get; set; } = new();

    public sealed class Knot
    {
        /// <summary>Lower edge of the pool (smallest raw score that maps to this calibrated value).</summary>
        public double X { get; set; }
        /// <summary>Upper edge of the pool (largest raw score that maps to this calibrated value).</summary>
        public double XMax { get; set; }
        /// <summary>Calibrated probability for this pool (pool mean of truth).</summary>
        public double Y { get; set; }
    }

    /// <summary>
    /// Fit PAV on (raw, truth) pairs. <paramref name="truth"/> values are 0/1
    /// but accepted as doubles to keep the API flexible.
    /// </summary>
    public static IsotonicCalibrator Fit(IReadOnlyList<double> rawProbs, IReadOnlyList<double> truth)
    {
        if (rawProbs.Count != truth.Count)
            throw new ArgumentException($"rawProbs ({rawProbs.Count}) and truth ({truth.Count}) must have the same length.");
        if (rawProbs.Count == 0)
            return new IsotonicCalibrator();

        var order = Enumerable.Range(0, rawProbs.Count)
            .OrderBy(i => rawProbs[i])
            .ToArray();

        // Each pool: (sum of truth, count, xMin, xMax).
        var poolSum = new List<double>(rawProbs.Count);
        var poolCount = new List<int>(rawProbs.Count);
        var poolXMin = new List<double>(rawProbs.Count);
        var poolXMax = new List<double>(rawProbs.Count);

        foreach (var i in order)
        {
            poolSum.Add(truth[i]);
            poolCount.Add(1);
            poolXMin.Add(rawProbs[i]);
            poolXMax.Add(rawProbs[i]);

            // Merge while the previous pool's mean exceeds the newest pool's mean.
            while (poolSum.Count >= 2)
            {
                var last = poolSum.Count - 1;
                var prev = last - 1;
                var prevMean = poolSum[prev] / poolCount[prev];
                var lastMean = poolSum[last] / poolCount[last];
                if (prevMean <= lastMean) break;

                poolSum[prev] += poolSum[last];
                poolCount[prev] += poolCount[last];
                poolXMax[prev] = poolXMax[last];
                poolSum.RemoveAt(last);
                poolCount.RemoveAt(last);
                poolXMin.RemoveAt(last);
                poolXMax.RemoveAt(last);
            }
        }

        var knots = new List<Knot>(poolSum.Count);
        for (var i = 0; i < poolSum.Count; i++)
        {
            knots.Add(new Knot
            {
                X = poolXMin[i],
                XMax = poolXMax[i],
                Y = poolSum[i] / poolCount[i],
            });
        }
        return new IsotonicCalibrator { Knots = knots };
    }

    /// <summary>
    /// Map a raw probability to the calibrated probability. Piecewise-constant
    /// inside each pool (so a pool covering [XMin, XMax] always returns its
    /// weighted mean Y); linear interpolation between adjacent pools on the
    /// (XMax_left, XMin_right) gap; clamped to the endpoint Y values outside
    /// the fitted range so we never extrapolate past observed behaviour.
    /// </summary>
    public double Apply(double rawProb)
    {
        if (Knots.Count == 0) return rawProb;
        if (Knots.Count == 1) return Knots[0].Y;
        if (rawProb <= Knots[0].X)   return Knots[0].Y;
        if (rawProb >= Knots[^1].XMax) return Knots[^1].Y;

        // Binary search for the rightmost knot whose XMin <= rawProb — that's
        // either the pool containing rawProb (if rawProb <= its XMax) or the
        // left side of the interpolation gap to the next pool.
        var lo = 0;
        var hi = Knots.Count - 1;
        while (lo < hi)
        {
            var mid = (lo + hi + 1) / 2;
            if (Knots[mid].X <= rawProb) lo = mid;
            else hi = mid - 1;
        }

        var left = Knots[lo];
        if (rawProb <= left.XMax) return left.Y;
        if (lo + 1 >= Knots.Count) return left.Y;

        var right = Knots[lo + 1];
        var span = right.X - left.XMax;
        if (span <= 0) return right.Y;
        var t = (rawProb - left.XMax) / span;
        return left.Y + t * (right.Y - left.Y);
    }
}

/// <summary>
/// Container for per-lead isotonic calibrators plus provenance. Lives alongside
/// the copied LightGBM model zips in an `*_isotonic` version directory so
/// predict can apply a lead-specific transform after scoring.
/// </summary>
public sealed class PrecipIsotonicCalibration
{
    /// <summary>Calibrator per lead-hour key (e.g. "24", "48", "72").</summary>
    public Dictionary<string, IsotonicCalibrator> ByLead { get; set; } = new();

    /// <summary>Version-dir name of the source 3a model the calibration was fit against.</summary>
    public string SourceVersion { get; set; } = "";

    /// <summary>Total rows used across all leads (validation slice); sanity-check provenance.</summary>
    public int SourceRowCount { get; set; }

    public DateTime FitAtUtc { get; set; }

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    public void SaveTo(string path)
        => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOpts));

    public static PrecipIsotonicCalibration LoadFrom(string path)
        => JsonSerializer.Deserialize<PrecipIsotonicCalibration>(File.ReadAllText(path))
           ?? throw new InvalidOperationException($"Could not parse isotonic calibration at {path}");

    public const string FileName = "calibration.json";
}
