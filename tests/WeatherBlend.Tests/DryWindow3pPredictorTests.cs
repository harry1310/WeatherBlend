using FluentAssertions;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Tests for the per-MC-sample longest-dry-run distribution path added 2026-05-26
/// to expose 3p confidence on the dry-window page (3p has no fitted conformal
/// calibrator on disk; the MC P10-P90 band is the only confidence signal that
/// ships without a retrain-dependency on 3o val-replay infrastructure).
/// </summary>
public class DryWindow3pPredictorTests
{
    private static double[,] IdentityCholesky(int n)
    {
        // Σ = I → L = I; under iid Bernoullis the MC degenerates to independent
        // hourly draws, which keeps each test's expected value easy to reason
        // about without depending on the empirical-Σ math.
        var L = new double[n, n];
        for (int i = 0; i < n; i++) L[i, i] = 1.0;
        return L;
    }

    [Fact]
    public void ProbDryWindowWithStats_returns_same_ProbWindow_as_plain_overload_for_same_seed()
    {
        // Regression: the stats overload must consume the same RNG stream the
        // plain overload does so a caller switching to the new API on the same
        // seeded RNG sees the same headline. Caught by side-by-side comparison
        // — anything that changes the per-sample draw count or order would
        // silently shift Brier numbers when 3p predict swaps APIs.
        var q = new[] { 0.1, 0.3, 0.5, 0.7, 0.2, 0.1, 0.4, 0.6, 0.2 };  // mid-range day
        var L = IdentityCholesky(q.Length);

        var rng1 = new Random(42);
        var plain = DryWindow3pPredictor.ProbDryWindow(q, L, windowLength: 3, rng1, nSamples: 5000);

        var rng2 = new Random(42);
        var stats = DryWindow3pPredictor.ProbDryWindowWithStats(q, L, windowLength: 3, rng2, nSamples: 5000);

        stats.ProbWindow.Should().Be(plain);
    }

    [Fact]
    public void ProbDryWindowWithStats_quantiles_are_monotone()
    {
        var q = new[] { 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1 };
        var L = IdentityCholesky(q.Length);

        var stats = DryWindow3pPredictor.ProbDryWindowWithStats(
            q, L, windowLength: 3, new Random(7), nSamples: 5000);

        stats.P10LongestDryRunHours.Should().BeLessThanOrEqualTo(stats.P50LongestDryRunHours);
        stats.P50LongestDryRunHours.Should().BeLessThanOrEqualTo(stats.P90LongestDryRunHours);
        stats.MeanLongestDryRunHours.Should().BeGreaterThanOrEqualTo(stats.P10LongestDryRunHours);
        stats.MeanLongestDryRunHours.Should().BeLessThanOrEqualTo(stats.P90LongestDryRunHours);
    }

    [Fact]
    public void ProbDryWindowWithStats_almost_all_dry_q_gives_full_length_run_with_high_confidence()
    {
        // Every hour P(wet)=0.01: nearly always dry. Longest run should be at
        // or near the full day for nearly every sample, P(window) ≈ 1, and
        // the P10–P90 band should be narrow (high confidence).
        var q = Enumerable.Repeat(0.01, 9).ToArray();
        var L = IdentityCholesky(q.Length);

        var stats = DryWindow3pPredictor.ProbDryWindowWithStats(
            q, L, windowLength: 3, new Random(123), nSamples: 5000);

        stats.ProbWindow.Should().BeGreaterThan(0.99);
        stats.P10LongestDryRunHours.Should().BeGreaterThanOrEqualTo(7);
        stats.P90LongestDryRunHours.Should().Be(9);  // hits the day length ceiling
        (stats.P90LongestDryRunHours - stats.P10LongestDryRunHours).Should().BeLessThanOrEqualTo(2,
            "almost-always-dry q should produce a tight longest-run distribution");
    }

    [Fact]
    public void ProbDryWindowWithStats_almost_all_wet_q_gives_short_runs_with_high_confidence()
    {
        // Opposite extreme: every hour wet. Longest dry run should be near 0
        // for nearly every sample.
        var q = Enumerable.Repeat(0.99, 9).ToArray();
        var L = IdentityCholesky(q.Length);

        var stats = DryWindow3pPredictor.ProbDryWindowWithStats(
            q, L, windowLength: 3, new Random(99), nSamples: 5000);

        stats.ProbWindow.Should().BeLessThan(0.01);
        stats.P10LongestDryRunHours.Should().Be(0);
        stats.P90LongestDryRunHours.Should().BeLessThanOrEqualTo(2);
    }

    [Fact]
    public void ProbDryWindowWithStats_throws_on_dimension_mismatch()
    {
        var q = new[] { 0.3, 0.4, 0.5 };
        var L = IdentityCholesky(4);  // wrong size on purpose

        var act = () => DryWindow3pPredictor.ProbDryWindowWithStats(
            q, L, windowLength: 3, new Random(0), nSamples: 100);

        act.Should().Throw<ArgumentException>().WithMessage("*do not match*");
    }
}
