using FluentAssertions;
using WeatherBlend.Evaluate.StartHour;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin <see cref="StartHourMetrics"/>. The verify command aggregates these
/// row-level numbers into per-(station, window, lead) skill scores; if any
/// of these drift the aggregate becomes meaningless.
/// </summary>
public class StartHourMetricsTests
{
    private static IReadOnlyList<(int, double)> Curve(params (int s, double pi)[] items) => items;
    private static IReadOnlySet<int> Truth(params int[] starts) => new HashSet<int>(starts);

    // ---- IsInformative -----------------------------------------------------

    [Theory]
    [InlineData(0, 4, false)]   // no truth-valid start → no shape signal
    [InlineData(4, 4, false)]   // every start valid → no shape signal
    [InlineData(1, 4, true)]    // exactly one start valid → maximum signal
    [InlineData(3, 4, true)]    // some-but-not-all → still informative
    public void IsInformative_excludes_degenerate_days(int truth, int total, bool expected)
    {
        StartHourMetrics.IsInformative(truth, total).Should().Be(expected);
    }

    // ---- Top1Hit -----------------------------------------------------------

    [Fact]
    public void Top1Hit_returns_true_when_argmax_is_a_truth_valid_start()
    {
        var curve = Curve((8, 0.10), (9, 0.40), (10, 0.30), (11, 0.20));
        var truth = Truth(9, 10);
        StartHourMetrics.Top1Hit(curve, truth).Should().BeTrue();
    }

    [Fact]
    public void Top1Hit_returns_false_when_argmax_is_not_in_truth()
    {
        var curve = Curve((8, 0.40), (9, 0.10), (10, 0.20), (11, 0.30));
        var truth = Truth(10, 11);  // argmax is 8, not in truth
        StartHourMetrics.Top1Hit(curve, truth).Should().BeFalse();
    }

    [Fact]
    public void Top1Hit_picks_first_in_iteration_order_on_ties()
    {
        // Genuine ties shouldn't crash. Caller supplies the curve in the
        // order it cares about (StartHourUtc ascending in practice); first
        // occurrence of the max wins. With a uniform-25% curve and truth at
        // the FIRST start, the function returns true; truth at any non-first
        // start, false. This pins the deterministic-on-ties behaviour.
        var curve = Curve((8, 0.25), (9, 0.25), (10, 0.25), (11, 0.25));
        StartHourMetrics.Top1Hit(curve, Truth(8)).Should().BeTrue();
        StartHourMetrics.Top1Hit(curve, Truth(11)).Should().BeFalse();
    }

    [Fact]
    public void Top1Hit_returns_false_for_empty_curve()
    {
        StartHourMetrics.Top1Hit(Array.Empty<(int, double)>(), Truth(8)).Should().BeFalse();
    }

    // ---- Brier -------------------------------------------------------------

    [Fact]
    public void Brier_zero_when_curve_perfectly_matches_uniform_over_truth()
    {
        // Truth at {9, 10}, τ_s = 0.5 each. A curve that places all mass on
        // 9 and 10 at 0.5 each scores Brier = 0.
        var curve = Curve((8, 0.0), (9, 0.5), (10, 0.5), (11, 0.0));
        var truth = Truth(9, 10);
        StartHourMetrics.Brier(curve, truth).Should().BeApproximately(0.0, 1e-12);
    }

    [Fact]
    public void Brier_uniform_curve_against_two_truth_starts_matches_hand_calculation()
    {
        // Uniform π = 0.25, τ at {9, 10} = 0.5. Squared diffs:
        //   s=8:  (0.25 − 0)²   = 0.0625
        //   s=9:  (0.25 − 0.5)² = 0.0625
        //   s=10: (0.25 − 0.5)² = 0.0625
        //   s=11: (0.25 − 0)²   = 0.0625
        // Sum = 0.25.
        var curve = Curve((8, 0.25), (9, 0.25), (10, 0.25), (11, 0.25));
        var truth = Truth(9, 10);
        StartHourMetrics.Brier(curve, truth).Should().BeApproximately(0.25, 1e-12);
    }

    [Fact]
    public void Brier_zero_for_empty_truth()
    {
        var curve = Curve((8, 0.25), (9, 0.25), (10, 0.25), (11, 0.25));
        StartHourMetrics.Brier(curve, Truth()).Should().Be(0.0);
    }

    // ---- LogLoss -----------------------------------------------------------

    [Fact]
    public void LogLoss_perfect_curve_approaches_zero()
    {
        // Truth at {9}, τ_9 = 1.0. Curve places 1.0 on 9. Log-loss = 0.
        var curve = Curve((8, 0.0), (9, 1.0), (10, 0.0), (11, 0.0));
        var truth = Truth(9);
        StartHourMetrics.LogLoss(curve, truth).Should().BeApproximately(0.0, 1e-12);
    }

    [Fact]
    public void LogLoss_uniform_curve_matches_minus_log_uniform_when_one_truth_start()
    {
        // Truth at {9}, τ_9 = 1.0. Uniform curve π_9 = 0.25. ll = -1·log(0.25).
        var curve = Curve((8, 0.25), (9, 0.25), (10, 0.25), (11, 0.25));
        StartHourMetrics.LogLoss(curve, Truth(9))
            .Should().BeApproximately(-Math.Log(0.25), 1e-12);
    }

    [Fact]
    public void LogLoss_clamps_zero_pi_to_epsilon_so_one_bad_row_does_not_drown_the_aggregate()
    {
        // π_9 = 0 with truth at 9 → log(0) = -∞, would poison the rolling
        // mean. Clamp to LogLossEpsilon = 1e-6, so the worst-case ll per
        // truth start is bounded by -log(1e-6) ≈ 13.8.
        var curve = Curve((8, 0.5), (9, 0.0), (10, 0.5), (11, 0.0));
        var truth = Truth(9);
        var ll = StartHourMetrics.LogLoss(curve, truth);
        ll.Should().BeApproximately(-Math.Log(1e-6), 1e-9);
    }

    // ---- LogLossUniform ----------------------------------------------------

    [Fact]
    public void LogLossUniform_is_minus_log_one_over_n()
    {
        StartHourMetrics.LogLossUniform(totalStartCount: 4, truthStartCount: 1)
            .Should().BeApproximately(-Math.Log(0.25), 1e-12);
        StartHourMetrics.LogLossUniform(totalStartCount: 7, truthStartCount: 3)
            .Should().BeApproximately(-Math.Log(1.0 / 7), 1e-12);
    }
}
