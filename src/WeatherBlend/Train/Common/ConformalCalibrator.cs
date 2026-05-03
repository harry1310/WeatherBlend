using System.Globalization;
using System.Text.Json;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Distribution-free conformal prediction for binary classifiers. Fits a
/// nonconformity threshold τ from a held-out calibration set; at predict
/// time, classifies each prediction's "set" as
/// {Dry}, {Wet}, or {Ambiguous} ({Dry, Wet}). The "ambiguous" tag is the
/// useful confidence signal — it flags rows where the model can't commit
/// to a single class with the requested coverage guarantee.
///
/// Coverage guarantee: with target miscoverage rate α, the prediction set
/// contains the true class on (1 − α) of held-out rows on average. So if
/// the model is well-trained and the val/test distributions match, ~90% of
/// "Ambiguous" days will see one of the two classes confirmed and ~10%
/// will be genuinely surprising.
///
/// The math (split conformal for binary):
///   Nonconformity score s(x, y) = 1 − p̂(y | x)
///   Calibration: scores s_i = 1 − p̂(y_i | x_i) for true labels on val
///   Threshold τ = quantile_⌈(1-α)(n+1)⌉/n of {s_1, ..., s_n}
///   Prediction set = { y : p̂(y | x_new) ≥ 1 − τ }
///
/// In our binary case with p = P(wet):
///   "Wet" ∈ set ⇔ p ≥ 1 − τ
///   "Dry" ∈ set ⇔ 1 − p ≥ 1 − τ ⇔ p ≤ τ
/// So the ambiguity zone is p ∈ [1 − τ, τ]; outside, exactly one class.
///
/// Pure stateless library: no I/O, no logging. Pairs cleanly with the
/// existing IsotonicCalibrator pattern (Fit / Predict / FromJson / ToJson).
/// </summary>
public sealed class ConformalCalibrator
{
    /// <summary>The fitted threshold τ ∈ (0, 1]. Higher τ → wider ambiguity
    /// zone; tighter τ → narrower (more rows committed to a single class).</summary>
    public double Tau { get; }

    /// <summary>Target miscoverage rate the calibrator was fit at (typically
    /// 0.10 = 90% coverage). Persisted so a reader can interpret τ.</summary>
    public double Alpha { get; }

    /// <summary>Calibration set size at fit time. Quantile uses ⌈(1-α)(n+1)⌉/n
    /// for the finite-sample correction, so reproducing the fit needs n.</summary>
    public int N { get; }

    public enum SetTag
    {
        /// <summary>Confident dry: P(wet) ≤ 1 − τ. Set = {Dry}.</summary>
        Dry,
        /// <summary>Confident wet: P(wet) ≥ τ. Set = {Wet}.</summary>
        Wet,
        /// <summary>Genuinely uncertain: 1 − τ &lt; P(wet) &lt; τ. Set = {Dry, Wet}.</summary>
        Ambiguous,
    }

    private ConformalCalibrator(double tau, double alpha, int n)
    {
        Tau = tau;
        Alpha = alpha;
        N = n;
    }

    /// <summary>
    /// Fit τ from calibration-set predictions and labels. Standard split-
    /// conformal — calibration data must NOT have been used to train the
    /// underlying probability model (use val, not train).
    /// </summary>
    /// <param name="probWet">Model's P(wet) per calibration row.</param>
    /// <param name="trueWet">Ground-truth wet/dry per row.</param>
    /// <param name="alpha">Target miscoverage rate (0.10 = 90% coverage).</param>
    public static ConformalCalibrator Fit(
        IReadOnlyList<double> probWet, IReadOnlyList<bool> trueWet, double alpha = 0.10)
    {
        if (probWet.Count != trueWet.Count)
            throw new ArgumentException("probWet and trueWet must have equal length.");
        if (probWet.Count == 0)
            throw new ArgumentException("Need at least one calibration row.", nameof(probWet));
        if (alpha <= 0.0 || alpha >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(alpha), "0 < α < 1");

        // Nonconformity scores under the model's labelling. For each row,
        // score = 1 − p̂(true_label) — small when the model put high mass
        // on the right class, large when it didn't.
        var scores = new double[probWet.Count];
        for (int i = 0; i < probWet.Count; i++)
        {
            var p = Math.Clamp(probWet[i], 0.0, 1.0);
            scores[i] = trueWet[i] ? 1.0 - p : p;
        }
        Array.Sort(scores);

        // Finite-sample threshold: quantile_q where q = ⌈(1-α)(n+1)⌉/n.
        // This is the "split conformal" correction — guarantees marginal
        // coverage (1-α) under exchangeability.
        var n = scores.Length;
        var q = Math.Min(1.0, Math.Ceiling((1.0 - alpha) * (n + 1)) / n);
        // Index of the q-quantile, clamped to the array.
        var idx = Math.Min(n - 1, (int)Math.Ceiling(q * n) - 1);
        if (idx < 0) idx = 0;
        var tau = scores[idx];
        // τ outside (0,1] would mean "never confident" / "always confident";
        // clamp so downstream Predict logic stays well-behaved.
        tau = Math.Clamp(tau, 1e-9, 1.0);
        return new ConformalCalibrator(tau, alpha, n);
    }

    /// <summary>
    /// Classify one prediction's set membership. See class summary for the
    /// math; in plain terms: the more "ambiguous" results you see, the more
    /// often the model is genuinely on the fence — useful confidence signal
    /// for users deciding whether to act on a borderline forecast.
    /// </summary>
    public SetTag Predict(double probWet)
    {
        var p = Math.Clamp(probWet, 0.0, 1.0);
        // Wet ∈ set ⇔ p ≥ 1 − τ;  Dry ∈ set ⇔ 1 − p ≥ 1 − τ ⇔ p ≤ τ.
        bool wetIn = p >= 1.0 - Tau;
        bool dryIn = p <= Tau;
        if (wetIn && dryIn) return SetTag.Ambiguous;
        return wetIn ? SetTag.Wet : SetTag.Dry;
    }

    // ---- Serialisation -----------------------------------------------------

    private sealed class Dto
    {
        public double Tau { get; set; }
        public double Alpha { get; set; }
        public int N { get; set; }
    }

    public string ToJson() => JsonSerializer.Serialize(
        new Dto { Tau = Tau, Alpha = Alpha, N = N },
        new JsonSerializerOptions { WriteIndented = false });

    public static ConformalCalibrator FromJson(string json)
    {
        var d = JsonSerializer.Deserialize<Dto>(json)
            ?? throw new ArgumentException("Empty or malformed conformal calibrator JSON.", nameof(json));
        if (!(d.Tau > 0 && d.Tau <= 1))
            throw new ArgumentException($"τ out of range: {d.Tau}", nameof(json));
        if (!(d.Alpha > 0 && d.Alpha < 1))
            throw new ArgumentException($"α out of range: {d.Alpha}", nameof(json));
        return new ConformalCalibrator(d.Tau, d.Alpha, d.N);
    }

    public override string ToString() =>
        $"ConformalCalibrator(τ={Tau.ToString("0.000", CultureInfo.InvariantCulture)}, "
        + $"α={Alpha.ToString("0.00", CultureInfo.InvariantCulture)}, n={N}, "
        + $"ambiguity zone P(wet) ∈ [{(1 - Tau).ToString("0.00", CultureInfo.InvariantCulture)}, "
        + $"{Tau.ToString("0.00", CultureInfo.InvariantCulture)}])";
}
