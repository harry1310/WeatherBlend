using FluentAssertions;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins ConformalCalibrator's coverage guarantee + the binary-classifier set
/// semantics. The calibrator's only user-facing claim is "the prediction set
/// covers the true class with frequency ≈ 1 − α on held-out data" — these
/// tests check that empirically + check the SetTag dispatch logic and JSON
/// round-trip.
/// </summary>
public class ConformalCalibratorTests
{
    // ---- SetTag dispatch ----

    [Theory]
    [InlineData(0.99, ConformalCalibrator.SetTag.Wet)]   // p > τ → only "Wet"
    [InlineData(0.01, ConformalCalibrator.SetTag.Dry)]   // p < 1-τ → only "Dry"
    [InlineData(0.50, ConformalCalibrator.SetTag.Ambiguous)]  // mid → both
    public void Predict_classifies_by_tau_threshold(double p, ConformalCalibrator.SetTag expected)
    {
        // Realistic calibration data: well-calibrated but not perfect — labels
        // drawn Bernoulli(P̂) so scores spread out and τ lands in (0.5, 0.95)
        // at α=0.10. Without spread, scores collapse to a single value and
        // the ambiguity zone is degenerate (empty or all of [0,1]).
        var rng = new Random(42);
        var probs = new List<double>();
        var labels = new List<bool>();
        for (int i = 0; i < 1000; i++)
        {
            var pw = rng.NextDouble();
            probs.Add(pw);
            labels.Add(rng.NextDouble() < pw);
        }
        var cal = ConformalCalibrator.Fit(probs, labels, alpha: 0.10);
        // Sanity: τ should land in a non-degenerate range.
        cal.Tau.Should().BeGreaterThan(0.6);
        cal.Tau.Should().BeLessThan(0.99);

        cal.Predict(p).Should().Be(expected);
    }

    // ---- Coverage guarantee (the load-bearing claim) ----

    [Fact]
    public void Coverage_meets_the_1_minus_alpha_target_on_held_out_data()
    {
        // Build a synthetic well-calibrated 1D model: P(wet | x) = sigmoid(x).
        // Sample (x, P(wet)) → draw true label from Bernoulli. Split into
        // calibration (fit τ) and held-out (measure coverage).
        var rng = new Random(42);
        int Sample(int n, out List<double> p, out List<bool> y)
        {
            p = new List<double>(n);
            y = new List<bool>(n);
            for (int i = 0; i < n; i++)
            {
                var x = (rng.NextDouble() - 0.5) * 6.0;
                var pw = 1.0 / (1.0 + Math.Exp(-x));
                p.Add(pw);
                y.Add(rng.NextDouble() < pw);
            }
            return n;
        }
        Sample(2000, out var calP, out var calY);
        Sample(2000, out var heldP, out var heldY);

        const double alpha = 0.10;
        var cal = ConformalCalibrator.Fit(calP, calY, alpha);

        // For each held-out row, count whether the conformal SET covers the
        // true label. Set covers Wet ↔ tag is Wet or Ambiguous;
        // covers Dry ↔ tag is Dry or Ambiguous.
        int covered = 0;
        for (int i = 0; i < heldP.Count; i++)
        {
            var tag = cal.Predict(heldP[i]);
            bool wetCovered = tag is ConformalCalibrator.SetTag.Wet or ConformalCalibrator.SetTag.Ambiguous;
            bool dryCovered = tag is ConformalCalibrator.SetTag.Dry or ConformalCalibrator.SetTag.Ambiguous;
            if (heldY[i] ? wetCovered : dryCovered) covered++;
        }
        var actualCoverage = (double)covered / heldP.Count;

        // Target is 1 − α = 0.90. Empirical coverage should be within
        // ~3% (Monte Carlo noise at n=2000 with conformal's non-iid finite-
        // sample bound).
        actualCoverage.Should().BeGreaterThan(0.87);
        actualCoverage.Should().BeLessThan(0.99);   // shouldn't be wildly over-covered either
    }

    [Fact]
    public void Tighter_alpha_widens_ambiguity_zone()
    {
        var rng = new Random(7);
        var probs = new List<double>();
        var labels = new List<bool>();
        for (int i = 0; i < 500; i++)
        {
            var p = rng.NextDouble();
            probs.Add(p);
            labels.Add(rng.NextDouble() < p);
        }

        var calLooseCoverage = ConformalCalibrator.Fit(probs, labels, alpha: 0.30);  // 70% target
        var calTightCoverage = ConformalCalibrator.Fit(probs, labels, alpha: 0.05);  // 95% target

        // Tighter target ⇒ wider τ ⇒ more "Ambiguous" classifications.
        calTightCoverage.Tau.Should().BeGreaterThan(calLooseCoverage.Tau);
    }

    // ---- JSON round-trip ----

    [Fact]
    public void Json_round_trip_preserves_tau_alpha_n()
    {
        var probs = Enumerable.Range(0, 100).Select(i => i / 100.0).ToList();
        var labels = probs.Select(p => p > 0.5).ToList();
        var original = ConformalCalibrator.Fit(probs, labels, alpha: 0.15);

        var revived = ConformalCalibrator.FromJson(original.ToJson());
        revived.Tau.Should().Be(original.Tau);
        revived.Alpha.Should().Be(original.Alpha);
        revived.N.Should().Be(original.N);
        revived.Predict(0.5).Should().Be(original.Predict(0.5));
    }

    [Fact]
    public void Fit_throws_on_mismatched_lengths()
    {
        var probs = new[] { 0.1, 0.2, 0.3 };
        var labels = new[] { true, false };
        var act = () => ConformalCalibrator.Fit(probs, labels);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Fit_throws_on_invalid_alpha()
    {
        var probs = new[] { 0.1, 0.5, 0.9 };
        var labels = new[] { false, true, true };
        var act = () => ConformalCalibrator.Fit(probs, labels, alpha: 1.5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
