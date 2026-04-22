namespace WeatherBlend.Evaluate.Precip;

/// <summary>
/// Scoring rules for probabilistic precipitation forecasts.
///
/// All methods operate on paired (forecast, truth) sequences and silently skip
/// rows where either value is NaN. Brier / reliability are for binary truth
/// (0 or 1) with probabilistic forecast in [0, 1]. Frequency bias is for
/// threshold-based binarised forecasts.
/// </summary>
public static class PrecipMetrics
{
    /// <summary>Brier score = mean squared error of probabilistic forecast vs binary outcome.</summary>
    public static double Brier(IReadOnlyList<double> prob, IReadOnlyList<double> truthBinary)
    {
        if (prob.Count != truthBinary.Count)
            throw new ArgumentException($"Length mismatch: prob={prob.Count}, truth={truthBinary.Count}");

        double sum = 0;
        int n = 0;
        for (int i = 0; i < prob.Count; i++)
        {
            var p = prob[i];
            var y = truthBinary[i];
            if (double.IsNaN(p) || double.IsNaN(y)) continue;
            var d = p - y;
            sum += d * d;
            n++;
        }
        return n == 0 ? double.NaN : sum / n;
    }

    /// <summary>
    /// Brier skill score vs a reference forecast (typically climatology).
    /// Positive = better than reference, 0 = same, negative = worse.
    /// </summary>
    public static double BrierSkillScore(double brier, double brierReference)
    {
        if (brierReference == 0.0) return double.NaN;
        return 1.0 - brier / brierReference;
    }

    public readonly record struct ReliabilityBin(
        double BinLower,
        double BinUpper,
        double MeanForecast,
        double ObservedFrequency,
        int N);

    /// <summary>
    /// Reliability table: bucket probabilistic forecasts into N equal-width bins
    /// of [0,1], report observed frequency per bin. A well-calibrated forecast
    /// has ObservedFrequency ≈ MeanForecast per bin.
    /// </summary>
    public static IReadOnlyList<ReliabilityBin> Reliability(
        IReadOnlyList<double> prob,
        IReadOnlyList<double> truthBinary,
        int nBins = 10)
    {
        if (nBins < 1) throw new ArgumentException("nBins must be >= 1", nameof(nBins));
        if (prob.Count != truthBinary.Count)
            throw new ArgumentException($"Length mismatch: prob={prob.Count}, truth={truthBinary.Count}");

        var sumF = new double[nBins];
        var sumY = new double[nBins];
        var cnt = new int[nBins];

        for (int i = 0; i < prob.Count; i++)
        {
            var p = prob[i];
            var y = truthBinary[i];
            if (double.IsNaN(p) || double.IsNaN(y)) continue;
            if (p < 0 || p > 1) continue;

            var bin = Math.Min(nBins - 1, (int)(p * nBins));
            sumF[bin] += p;
            sumY[bin] += y;
            cnt[bin]++;
        }

        var bins = new List<ReliabilityBin>(nBins);
        for (int b = 0; b < nBins; b++)
        {
            var lo = (double)b / nBins;
            var hi = (double)(b + 1) / nBins;
            if (cnt[b] == 0)
            {
                bins.Add(new ReliabilityBin(lo, hi, double.NaN, double.NaN, 0));
            }
            else
            {
                bins.Add(new ReliabilityBin(lo, hi, sumF[b] / cnt[b], sumY[b] / cnt[b], cnt[b]));
            }
        }
        return bins;
    }

    /// <summary>
    /// Frequency bias = forecast_wet_count / observed_wet_count.
    /// 1.0 = unbiased, &gt; 1.0 = over-forecasts wet, &lt; 1.0 = under-forecasts wet.
    /// </summary>
    public static double FrequencyBias(
        IReadOnlyList<double> forecastBinary,
        IReadOnlyList<double> truthBinary)
    {
        if (forecastBinary.Count != truthBinary.Count)
            throw new ArgumentException(
                $"Length mismatch: forecast={forecastBinary.Count}, truth={truthBinary.Count}");

        int fWet = 0, oWet = 0;
        for (int i = 0; i < forecastBinary.Count; i++)
        {
            var f = forecastBinary[i];
            var y = truthBinary[i];
            if (double.IsNaN(f) || double.IsNaN(y)) continue;
            if (f >= 0.5) fWet++;
            if (y >= 0.5) oWet++;
        }
        return oWet == 0 ? double.NaN : (double)fWet / oWet;
    }

    /// <summary>
    /// Continuous Ranked Probability Score for a deterministic forecast against
    /// a point truth — degenerates to |forecast - truth|. Equivalent to MAE.
    /// For ensemble/quantile forecasts use <see cref="CrpsQuantile"/>.
    /// </summary>
    public static double CrpsDeterministic(
        IReadOnlyList<double> forecast,
        IReadOnlyList<double> truth)
    {
        if (forecast.Count != truth.Count)
            throw new ArgumentException($"Length mismatch: forecast={forecast.Count}, truth={truth.Count}");

        double sum = 0;
        int n = 0;
        for (int i = 0; i < forecast.Count; i++)
        {
            var f = forecast[i];
            var y = truth[i];
            if (double.IsNaN(f) || double.IsNaN(y)) continue;
            sum += Math.Abs(f - y);
            n++;
        }
        return n == 0 ? double.NaN : sum / n;
    }

    /// <summary>
    /// CRPS for a quantile forecast: each row supplies sorted quantile values
    /// at evenly-spaced probability levels {1/(K+1), 2/(K+1), ..., K/(K+1)}.
    /// Integrates (F(x) - 1[x &gt;= y])^2 via trapezoidal sum across the K+1
    /// intervals defined by the quantile points and the truth y.
    /// </summary>
    public static double CrpsQuantile(
        IReadOnlyList<IReadOnlyList<double>> quantiles,
        IReadOnlyList<double> truth)
    {
        if (quantiles.Count != truth.Count)
            throw new ArgumentException($"Length mismatch: quantiles={quantiles.Count}, truth={truth.Count}");

        double sum = 0;
        int n = 0;
        for (int i = 0; i < quantiles.Count; i++)
        {
            var y = truth[i];
            if (double.IsNaN(y)) continue;
            var q = quantiles[i];
            if (q.Count == 0) continue;

            // Probability levels 1/(K+1), 2/(K+1), ..., K/(K+1).
            int k = q.Count;
            double crps = 0;

            // Left tail: x from -inf to q[0] -> F(x)=0 (we use q[0] - (q[1]-q[0]) as a proxy for infinity, but for
            // rainfall (non-negative), the left tail below 0 contributes nothing, so clamp at 0).
            // For simplicity and since rainfall >= 0, use a discrete approximation:
            // CRPS ≈ sum over bins ((prob_at_bin - indicator(truth<=x_right))^2 * bin_width).
            var xs = new List<double> { 0.0 }; // rainfall lower bound
            xs.AddRange(q);
            var probs = new List<double> { 0.0 };
            for (int j = 0; j < k; j++)
                probs.Add((j + 1.0) / (k + 1.0));

            for (int j = 0; j < xs.Count - 1; j++)
            {
                var width = xs[j + 1] - xs[j];
                if (width <= 0) continue;
                var f = probs[j + 1];
                var indicator = y <= xs[j + 1] ? 1.0 : 0.0;
                var diff = f - indicator;
                crps += diff * diff * width;
            }

            // Right tail beyond last quantile: F=K/(K+1) until y if y > q[k-1], then F=1.
            if (y > q[k - 1])
            {
                var tailWidth = y - q[k - 1];
                var f = (double)k / (k + 1);
                crps += (f - 0.0) * (f - 0.0) * tailWidth; // indicator is 0 until x reaches y
            }

            sum += crps;
            n++;
        }
        return n == 0 ? double.NaN : sum / n;
    }

    /// <summary>
    /// Contingency table at a wet/dry threshold. Hits/misses/false-alarms/true-negs for
    /// the 2x2 confusion matrix. Useful for POD, FAR, CSI, ETS etc.
    /// </summary>
    public readonly record struct Contingency(int Hits, int Misses, int FalseAlarms, int TrueNegatives)
    {
        public int Total => Hits + Misses + FalseAlarms + TrueNegatives;
        public double PodHitRate => Hits + Misses == 0 ? double.NaN : (double)Hits / (Hits + Misses);
        public double FarFalseAlarmRatio => Hits + FalseAlarms == 0 ? double.NaN : (double)FalseAlarms / (Hits + FalseAlarms);
        public double Csi => Hits + Misses + FalseAlarms == 0 ? double.NaN : (double)Hits / (Hits + Misses + FalseAlarms);
        public double FrequencyBias => Hits + Misses == 0 ? double.NaN : (double)(Hits + FalseAlarms) / (Hits + Misses);
    }

    public static Contingency BuildContingency(
        IReadOnlyList<double> forecastBinary,
        IReadOnlyList<double> truthBinary)
    {
        if (forecastBinary.Count != truthBinary.Count)
            throw new ArgumentException(
                $"Length mismatch: forecast={forecastBinary.Count}, truth={truthBinary.Count}");

        int hits = 0, misses = 0, fa = 0, tn = 0;
        for (int i = 0; i < forecastBinary.Count; i++)
        {
            var f = forecastBinary[i];
            var y = truthBinary[i];
            if (double.IsNaN(f) || double.IsNaN(y)) continue;
            var fBin = f >= 0.5;
            var yBin = y >= 0.5;
            if (fBin && yBin) hits++;
            else if (!fBin && yBin) misses++;
            else if (fBin && !yBin) fa++;
            else tn++;
        }
        return new Contingency(hits, misses, fa, tn);
    }
}
