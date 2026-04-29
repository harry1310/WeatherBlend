using System.Globalization;
using System.Text.Json;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Isotonic (Pool Adjacent Violators) calibrator for binary-classifier
/// probabilities. Fits a monotonic step function from raw model probability
/// to empirical positive rate using a held-out set (typically the
/// validation slice — never the training set, never the test set).
///
/// Why this exists: LightGBM's raw probabilities (after the ML.NET Platt
/// wrapper) often systematically over- or under-predict the positive
/// class, especially at the tails. The "frequency bias" diagnostic in
/// training output captures the same thing — when the bias is &gt; 1.2
/// or so, raw probs say "70% chance of dry" on days that are actually
/// dry 60% of the time, and PAV maps the prediction down accordingly.
///
/// Improves the reliability component of Brier (calibration), can't
/// help resolution (discrimination). Net Brier improvement is typically
/// 0–5%, concentrated where bias is highest. Cheap to fit, cheap to
/// apply, cheap to serialise — no good reason not to ship it as a
/// core layer once the gain is established.
/// </summary>
public sealed class IsotonicCalibrator
{
    /// <summary>Sorted ascending; <c>X[i]</c> raw probability → <c>Y[i]</c> calibrated.</summary>
    public double[] X { get; }
    public double[] Y { get; }

    private IsotonicCalibrator(double[] x, double[] y)
    {
        X = x;
        Y = y;
    }

    /// <summary>
    /// Fit a PAV-monotonic step function from (rawProb, 0/1 label) pairs.
    /// Standard pool-adjacent-violators algorithm: sort by raw prob, walk
    /// left-to-right, merge any block where the running mean of labels
    /// would violate monotonicity with the previous block. Returns a
    /// calibrator that linear-interpolates between the merged break points
    /// (so unseen raw probs at predict time get a smooth value, not a jump).
    /// </summary>
    public static IsotonicCalibrator Fit(IReadOnlyList<double> rawProbs, IReadOnlyList<bool> labels)
    {
        if (rawProbs.Count != labels.Count)
            throw new ArgumentException("rawProbs and labels must have equal length.");
        if (rawProbs.Count == 0)
            throw new ArgumentException("Need at least one observation to fit.", nameof(rawProbs));

        // Pair, sort ascending by raw prob, stable on input order to keep ties deterministic.
        var pairs = rawProbs
            .Select((p, i) => (P: p, Y: labels[i] ? 1.0 : 0.0, Idx: i))
            .OrderBy(t => t.P).ThenBy(t => t.Idx)
            .ToArray();

        // Stack-based PAV: walk left-to-right, push each pair as a new block,
        // merge with the predecessor whenever the predecessor's running mean
        // exceeds the current block's mean (would violate monotonicity).
        // Each merge produces a block whose new mean might still violate
        // against ITS predecessor, so the merge runs in a loop.
        var stack = new List<(double SumY, int Count, double MinP, double MaxP)>();
        foreach (var pair in pairs)
        {
            var block = (SumY: pair.Y, Count: 1, MinP: pair.P, MaxP: pair.P);
            while (stack.Count > 0)
            {
                var top = stack[^1];
                var topMean = top.SumY / top.Count;
                var blockMean = block.SumY / block.Count;
                // Merge equal-mean blocks too — collapses redundant breakpoints
                // (e.g. all-ones / all-zeros input → single block) and keeps the
                // serialised calibrator small.
                if (topMean < blockMean) break;
                stack.RemoveAt(stack.Count - 1);
                block = (top.SumY + block.SumY,
                         top.Count + block.Count,
                         Math.Min(top.MinP, block.MinP),
                         Math.Max(top.MaxP, block.MaxP));
            }
            stack.Add(block);
        }

        // Block midpoint (in raw-prob space) → block mean (in label space).
        var xs = stack.Select(b => (b.MinP + b.MaxP) / 2.0).ToList();
        var ys = stack.Select(b => b.SumY / b.Count).ToList();

        // Edge case: PAV collapsed to a single block (every label was identical,
        // or input is fully degenerate). Predict still works but linear-interp
        // wants ≥ 2 breakpoints. Synthesise [0, 1] endpoints with the constant
        // value so out-of-range inputs map sensibly too.
        if (xs.Count == 1)
            return new IsotonicCalibrator(new[] { 0.0, 1.0 }, new[] { ys[0], ys[0] });

        return new IsotonicCalibrator(xs.ToArray(), ys.ToArray());
    }

    /// <summary>
    /// Map a raw probability to its calibrated value via piecewise-linear
    /// interpolation between the fitted break points. Values outside the
    /// fitted range clamp to the nearest endpoint (the standard PAV-with-
    /// linear-interp behaviour — extrapolation would invent calibration
    /// signal we don't have).
    /// </summary>
    public double Predict(double rawProb)
    {
        if (rawProb <= X[0]) return Y[0];
        if (rawProb >= X[^1]) return Y[^1];

        // Binary search for the right segment.
        int lo = 0, hi = X.Length - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (X[mid] <= rawProb) lo = mid; else hi = mid;
        }

        var x0 = X[lo]; var x1 = X[hi];
        var y0 = Y[lo]; var y1 = Y[hi];
        if (x1 == x0) return (y0 + y1) / 2.0; // shouldn't happen post-PAV but defensive
        var t = (rawProb - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    public double[] PredictMany(IReadOnlyList<double> rawProbs)
    {
        var result = new double[rawProbs.Count];
        for (int i = 0; i < rawProbs.Count; i++) result[i] = Predict(rawProbs[i]);
        return result;
    }

    // ---- Serialisation -----------------------------------------------------

    private sealed class Dto
    {
        public double[] X { get; set; } = Array.Empty<double>();
        public double[] Y { get; set; } = Array.Empty<double>();
    }

    public string ToJson()
        => JsonSerializer.Serialize(new Dto { X = X, Y = Y },
            new JsonSerializerOptions { WriteIndented = false });

    public static IsotonicCalibrator FromJson(string json)
    {
        var d = JsonSerializer.Deserialize<Dto>(json)
            ?? throw new ArgumentException("Empty or malformed isotonic calibrator JSON.", nameof(json));
        if (d.X.Length != d.Y.Length || d.X.Length < 2)
            throw new ArgumentException(
                $"Calibrator X/Y mismatch or too few breakpoints: X={d.X.Length}, Y={d.Y.Length}.", nameof(json));
        return new IsotonicCalibrator(d.X, d.Y);
    }

    public override string ToString()
        => $"IsotonicCalibrator(breakpoints={X.Length}, range=[{X[0].ToString("0.00", CultureInfo.InvariantCulture)}..{X[^1].ToString("0.00", CultureInfo.InvariantCulture)}] → [{Y[0].ToString("0.00", CultureInfo.InvariantCulture)}..{Y[^1].ToString("0.00", CultureInfo.InvariantCulture)}])";
}
