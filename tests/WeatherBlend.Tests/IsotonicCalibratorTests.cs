using FluentAssertions;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins the PAV calibrator's contract: monotone-non-decreasing fit, sane
/// behaviour at the tails, round-trip JSON, and the load-bearing
/// "calibration shifts probabilities toward empirical" property the
/// dry-window blender depends on.
/// </summary>
public class IsotonicCalibratorTests
{
    [Fact]
    public void Fit_yields_monotone_non_decreasing_calibration_curve()
    {
        // Random-ish raw probs against monotonically-related labels; the fit
        // must enforce monotonicity even if the input data has local violations.
        var rng = new Random(42);
        var raw = Enumerable.Range(0, 200).Select(_ => rng.NextDouble()).ToList();
        // True positive rate is roughly raw + noise — PAV should recover the trend.
        var labels = raw.Select(p => rng.NextDouble() < p).ToList();

        var cal = IsotonicCalibrator.Fit(raw, labels);

        // Monotonic non-decreasing on the breakpoints by construction.
        for (int i = 1; i < cal.Y.Length; i++)
            cal.Y[i].Should().BeGreaterThanOrEqualTo(cal.Y[i - 1], $"breakpoint {i} must not regress");
    }

    [Fact]
    public void Predict_clamps_to_endpoints_outside_fitted_range()
    {
        // Every label dry → calibrator says "always dry" everywhere.
        var raw = new List<double> { 0.3, 0.5, 0.7 };
        var labels = new List<bool> { true, true, true };

        var cal = IsotonicCalibrator.Fit(raw, labels);

        cal.Predict(-1.0).Should().Be(cal.Y[0]);
        cal.Predict(0.0).Should().BeApproximately(cal.Y[0], 1e-9);
        cal.Predict(2.0).Should().Be(cal.Y[^1]);
    }

    [Fact]
    public void Fit_corrects_systematic_over_prediction()
    {
        // Synth case: model predicts P=0.8 on every row, but truth is positive
        // only 50% of the time. PAV should map 0.8 → ~0.5.
        var raw = Enumerable.Repeat(0.8, 100).ToList();
        var labels = Enumerable.Range(0, 100).Select(i => i < 50).ToList();

        var cal = IsotonicCalibrator.Fit(raw, labels);
        var calibrated = cal.Predict(0.8);

        calibrated.Should().BeApproximately(0.5, 0.05,
            "PAV should compress systematic over-prediction back to the empirical positive rate");
    }

    [Fact]
    public void Fit_preserves_well_calibrated_input_within_a_margin()
    {
        // Genuinely well-calibrated synth: at each raw prob bin, the empirical
        // positive rate equals the prob. Calibrator should be close to identity
        // (within sampling noise on 1000 obs).
        var rng = new Random(7);
        var raw = new List<double>();
        var labels = new List<bool>();
        for (int bin = 0; bin < 10; bin++)
        {
            var p = (bin + 0.5) / 10.0;
            for (int i = 0; i < 100; i++)
            {
                raw.Add(p);
                labels.Add(rng.NextDouble() < p);
            }
        }

        var cal = IsotonicCalibrator.Fit(raw, labels);

        // Probe at each bin centre — calibrated value should be near the bin.
        // Loose tolerance (±0.15) to absorb sampling noise on 100-row bins
        // plus PAV's equal-mean merging which collapses adjacent flat regions.
        for (int bin = 0; bin < 10; bin++)
        {
            var p = (bin + 0.5) / 10.0;
            cal.Predict(p).Should().BeApproximately(p, 0.15,
                $"well-calibrated input at p={p:F2} should map close to itself");
        }
    }

    [Fact]
    public void Json_round_trip_preserves_breakpoints()
    {
        var raw = new List<double> { 0.1, 0.3, 0.5, 0.7, 0.9 };
        var labels = new List<bool> { false, false, true, true, true };

        var original = IsotonicCalibrator.Fit(raw, labels);
        var json = original.ToJson();
        var restored = IsotonicCalibrator.FromJson(json);

        restored.X.Should().Equal(original.X);
        restored.Y.Should().Equal(original.Y);
        restored.Predict(0.4).Should().BeApproximately(original.Predict(0.4), 1e-12);
    }

    [Fact]
    public void Fit_handles_all_labels_identical_with_synthesised_endpoints()
    {
        // If every observation has the same label, the PAV output collapses to
        // a single (x, y) which can't be linearly interpolated. The constructor
        // must synthesise [0, 1] endpoints with the constant value.
        var raw = new List<double> { 0.2, 0.4, 0.6 };
        var allTrue = new List<bool> { true, true, true };

        var cal = IsotonicCalibrator.Fit(raw, allTrue);

        cal.X.Should().HaveCount(2);
        cal.Y[0].Should().Be(1.0);
        cal.Y[1].Should().Be(1.0);
        cal.Predict(0.0).Should().Be(1.0);
        cal.Predict(0.5).Should().Be(1.0);
        cal.Predict(1.0).Should().Be(1.0);
    }

    [Fact]
    public void Fit_rejects_mismatched_input_lengths()
    {
        Action act = () => IsotonicCalibrator.Fit(new[] { 0.1, 0.2 }, new[] { true });
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void FromJson_rejects_malformed_input()
    {
        Action emptyJson = () => IsotonicCalibrator.FromJson("{}");
        emptyJson.Should().Throw<ArgumentException>();

        Action mismatched = () => IsotonicCalibrator.FromJson("""{"X":[0.1,0.2],"Y":[0.3]}""");
        mismatched.Should().Throw<ArgumentException>();
    }
}
