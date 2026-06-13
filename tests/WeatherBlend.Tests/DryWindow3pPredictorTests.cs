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
    // ---- Copula-MC source binding (3p over 3o, 3q over 3c — 2026-06-13) ----

    [Fact]
    public void SourceFor_binds_3p_to_3o_and_3q_to_3c()
    {
        var p = DryWindow3pPredictor.SourceFor(DryWindow3pPredictor.Phase3p);
        p.PrecipPhase.Should().Be("3o");
        p.VersionKey.Should().Be(DryWindow3pPredictor.Precip3oVersionKey);
        p.StartHourCurveVersion.Should().Be("v3-3p");

        var q = DryWindow3pPredictor.SourceFor(DryWindow3pPredictor.Phase3q);
        q.PrecipPhase.Should().Be("3c");
        q.VersionKey.Should().Be(DryWindow3pPredictor.Precip3cVersionKey);
        q.StartHourCurveVersion.Should().Be("v3-3q");
    }

    [Fact]
    public void SourceFor_throws_for_a_non_copula_phase()
    {
        var act = () => DryWindow3pPredictor.SourceFor("3b");
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData("copula-mc", "3p")]
    [InlineData("copula-mc-3c", "3q")]
    [InlineData("rich", null)]
    [InlineData("lean", null)]
    public void PhaseForFeatureSet_maps_only_the_copula_tokens(string featureSet, string? expected)
        => DryWindow3pPredictor.PhaseForFeatureSet(featureSet).Should().Be(expected);

    [Theory]
    [InlineData("3p", true)]
    [InlineData("3q", true)]
    [InlineData("3b", false)]
    [InlineData("3a", false)]
    public void IsCopulaMcPhase_covers_3p_and_3q_only(string phase, bool expected)
        => DryWindow3pPredictor.IsCopulaMcPhase(phase).Should().Be(expected);

    [Fact]
    public void Predict_start_hour_versions_match_the_registry_for_both_copula_phases()
    {
        // The site reads StartHourCurveVersion from phases.yaml; SourceFor is
        // the predict/train side. They must agree per phase or the start-hour
        // curve a phase WRITES won't be the one the page READS.
        DryWindow3pPredictor.SourceFor(DryWindow3pPredictor.Phase3p).StartHourCurveVersion
            .Should().Be(WeatherBlend.Site.DryWindowPhases.Phase3p.StartHourCurveVersion);
        DryWindow3pPredictor.SourceFor(DryWindow3pPredictor.Phase3q).StartHourCurveVersion
            .Should().Be(WeatherBlend.Site.DryWindowPhases.Phase3q.StartHourCurveVersion);
    }

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
    public void ProbDryWindow_windows_2_to_6_are_monotone_and_all_returned()
    {
        // The production window set since 2026-06-10 ({3,4,6} → every hour
        // 2..6, feeding the overview "Will it stay dry?" calculator's length
        // menu). All five must come back from ONE call, and because every
        // window thresholds the same correlated-Bernoulli draws, the
        // monotonicity P(≥2h) ≥ P(≥3h) ≥ … ≥ P(≥6h) is EXACT (not just
        // statistical) — the property the widget's length menu relies on
        // (a longer ask can never look more likely).
        var q = new[] { 0.1, 0.3, 0.5, 0.7, 0.2, 0.1, 0.4, 0.6, 0.2 };
        var L = IdentityCholesky(q.Length);
        var windows = new[] { 2, 3, 4, 5, 6 };

        var probs = DryWindow3pPredictor.ProbDryWindow(
            q, L, windows, new Random(42), nSamples: 5000);

        probs.Keys.Should().BeEquivalentTo(windows);
        for (int i = 1; i < windows.Length; i++)
            probs[windows[i]].Should().BeLessThanOrEqualTo(probs[windows[i - 1]],
                $"P(dry ≥ {windows[i]}h) cannot exceed P(dry ≥ {windows[i - 1]}h) under shared draws");
        // Mid-range q day: the extremes shouldn't saturate, so the ordering
        // is exercised on strictly interior values.
        probs[2].Should().BeInRange(0.05, 0.999);
        probs[6].Should().BeInRange(0.0, probs[2]);
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

    // ---- ProbDryWindowWithStartHours (curve emitted as MC byproduct) -------
    //
    // The start-hour overload runs the same MC pass that produces the daily
    // stats AND tallies, per draw, which candidate start hour windows came
    // out entirely dry. The chart on the dry-window page consumes the curve,
    // so the tests below pin three invariants the renderer relies on:
    //
    //   1. ProbWindow matches the stats overload for the same seed (the chart
    //      and the headline must come from one MC pass — no drift).
    //   2. Curve shape: length = n - windowLength + 1, every value in [0, 1].
    //   3. Headline ≥ max(curve) AND headline ≤ sum(curve) under any draws —
    //      mathematical invariants that any correct implementation must
    //      satisfy (the headline is the union over candidate-start events,
    //      bounded below by any single event's marginal and above by the sum
    //      of marginals via overlap).

    [Fact]
    public void ProbDryWindowWithStartHours_returns_same_ProbWindow_as_stats_overload_for_same_seed()
    {
        // Regression: the chart's daily aggregate and the dry-window page's
        // headline must agree to the bit because they come from the same MC
        // pass. Anything that changes the per-sample draw count or order
        // would silently desynchronise the two.
        var q = new[] { 0.1, 0.3, 0.5, 0.7, 0.2, 0.1, 0.4, 0.6, 0.2 };
        var L = IdentityCholesky(q.Length);

        var rng1 = new Random(42);
        var stats = DryWindow3pPredictor.ProbDryWindowWithStats(q, L, windowLength: 3, rng1, nSamples: 5000);

        var rng2 = new Random(42);
        var (statsCurve, _) = DryWindow3pPredictor.ProbDryWindowWithStartHours(q, L, windowLength: 3, rng2, nSamples: 5000);

        statsCurve.ProbWindow.Should().Be(stats.ProbWindow);
        statsCurve.MeanLongestDryRunHours.Should().Be(stats.MeanLongestDryRunHours);
        statsCurve.P10LongestDryRunHours.Should().Be(stats.P10LongestDryRunHours);
        statsCurve.P50LongestDryRunHours.Should().Be(stats.P50LongestDryRunHours);
        statsCurve.P90LongestDryRunHours.Should().Be(stats.P90LongestDryRunHours);
    }

    [Fact]
    public void ProbDryWindowWithStartHours_curve_has_expected_shape()
    {
        var q = new[] { 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1 };
        var L = IdentityCholesky(q.Length);

        var (_, curve) = DryWindow3pPredictor.ProbDryWindowWithStartHours(
            q, L, windowLength: 3, new Random(7), nSamples: 5000);

        curve.Length.Should().Be(q.Length - 3 + 1, "one row per candidate start hour");
        curve.Should().AllSatisfy(p => p.Should().BeInRange(0.0, 1.0));
    }

    [Fact]
    public void ProbDryWindowWithStartHours_headline_satisfies_union_bounds()
    {
        // Mathematical invariants any correct implementation must satisfy:
        //   max(per-start P) ≤ P(any block exists) ≤ Σ(per-start P)
        // The headline IS the union of the candidate-start dry events on
        // each MC draw, so it must dominate any individual marginal and be
        // dominated by their sum (which double-counts overlaps).
        var q = new[] { 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1 };
        var L = IdentityCholesky(q.Length);

        var (stats, curve) = DryWindow3pPredictor.ProbDryWindowWithStartHours(
            q, L, windowLength: 3, new Random(11), nSamples: 5000);

        var maxStart = curve.Max();
        var sumStart = curve.Sum();
        stats.ProbWindow.Should().BeGreaterThanOrEqualTo(maxStart, "headline ≥ any individual start's marginal");
        stats.ProbWindow.Should().BeLessThanOrEqualTo(sumStart + 1e-9, "headline ≤ sum of marginals (overlaps double-counted in the sum)");
    }

    [Fact]
    public void ProbDryWindowWithStartHours_almost_all_dry_q_gives_high_marginals_at_every_start()
    {
        // Every hour P(wet)=0.01: every candidate window is almost always
        // dry, so the curve should be saturated near 1 at every start.
        var q = Enumerable.Repeat(0.01, 9).ToArray();
        var L = IdentityCholesky(q.Length);

        var (_, curve) = DryWindow3pPredictor.ProbDryWindowWithStartHours(
            q, L, windowLength: 3, new Random(31), nSamples: 5000);

        curve.Should().AllSatisfy(p => p.Should().BeGreaterThan(0.95,
            "every 3-hour window in an almost-always-dry day should be dry on nearly every sample"));
    }

    [Fact]
    public void ProbDryWindowWithStartHours_almost_all_wet_q_gives_near_zero_marginals_at_every_start()
    {
        var q = Enumerable.Repeat(0.95, 9).ToArray();
        var L = IdentityCholesky(q.Length);

        var (_, curve) = DryWindow3pPredictor.ProbDryWindowWithStartHours(
            q, L, windowLength: 3, new Random(53), nSamples: 5000);

        curve.Should().AllSatisfy(p => p.Should().BeLessThan(0.05,
            "every 3-hour window in an almost-always-wet day should be dry on essentially no sample"));
    }

    [Fact]
    public void ProbDryWindowWithStartHours_empty_inputs_return_empty_curve()
    {
        var L = IdentityCholesky(0);

        var (stats, curve) = DryWindow3pPredictor.ProbDryWindowWithStartHours(
            Array.Empty<double>(), L, windowLength: 3, new Random(0), nSamples: 100);

        curve.Should().BeEmpty();
        stats.ProbWindow.Should().Be(0.0);
    }

    [Fact]
    public void ProbDryWindowWithStartHours_window_longer_than_day_returns_empty_curve()
    {
        // Window=10 against a 9-hour day: no candidate start hour exists.
        // The renderer treats an empty curve as "no chart", which is what
        // we want (a chart with zero series would be visual noise).
        var q = Enumerable.Repeat(0.3, 9).ToArray();
        var L = IdentityCholesky(q.Length);

        var (stats, curve) = DryWindow3pPredictor.ProbDryWindowWithStartHours(
            q, L, windowLength: 10, new Random(0), nSamples: 100);

        curve.Should().BeEmpty();
        stats.ProbWindow.Should().Be(0.0);
    }

    [Fact]
    public void ProbDryWindowWithStartHours_throws_on_dimension_mismatch()
    {
        var q = new[] { 0.3, 0.4, 0.5 };
        var L = IdentityCholesky(4);  // wrong size on purpose

        var act = () => DryWindow3pPredictor.ProbDryWindowWithStartHours(
            q, L, windowLength: 2, new Random(0), nSamples: 100);

        act.Should().Throw<ArgumentException>().WithMessage("*do not match*");
    }
}
