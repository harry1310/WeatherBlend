using FluentAssertions;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins the structural guarantees of Phase 3g (parameter-free MC over Phase
/// 3a hourly P(wet) marginals). Most of 3g's value comes from two
/// properties:
///   (1) cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) holds exactly
///       (not just in expectation) because all windows are computed off the
///       same Bernoulli sequence per MC sample;
///   (2) determinism — same q vector + same seed → same output, so
///       train-time scoring matches predict-time output bit-for-bit.
/// Both regress silently if someone adds an extra rng.NextDouble() call in
/// the wrong place. These tests catch that.
/// </summary>
public class DryWindow3gPredictorTests
{
    // ---------- Structural monotonicity ----------

    [Fact]
    public void ProbDryWindow_multi_window_is_non_increasing_in_N()
    {
        // Mid-range hourly q with realistic spread; 9 daytime hours.
        var q = new[] { 0.2, 0.5, 0.8, 0.3, 0.4, 0.6, 0.1, 0.7, 0.2 };
        var rng = new System.Random(1);

        var p = DryWindow3gPredictor.ProbDryWindow(q, new[] { 3, 4, 6 }, rng, nSamples: 5000);

        // Monotonicity holds by construction — single MC pass shares the
        // Bernoulli sequence across all three window indicators.
        p[3].Should().BeGreaterThanOrEqualTo(p[4]);
        p[4].Should().BeGreaterThanOrEqualTo(p[6]);
    }

    [Fact]
    public void ProbDryWindow_multi_window_holds_monotonicity_under_extreme_q()
    {
        // Very wet day — should put most mass on "no long dry block exists".
        var qWet = Enumerable.Repeat(0.95, 9).ToArray();
        var pWet = DryWindow3gPredictor.ProbDryWindow(qWet, new[] { 3, 4, 6 }, new System.Random(42));
        pWet[3].Should().BeGreaterThanOrEqualTo(pWet[4]);
        pWet[4].Should().BeGreaterThanOrEqualTo(pWet[6]);
        pWet[6].Should().BeLessThan(0.05);   // 6h dry block essentially impossible

        // Very dry day — all three should approach 1.0.
        var qDry = Enumerable.Repeat(0.05, 9).ToArray();
        var pDry = DryWindow3gPredictor.ProbDryWindow(qDry, new[] { 3, 4, 6 }, new System.Random(42));
        pDry[3].Should().BeGreaterThanOrEqualTo(pDry[4]);
        pDry[4].Should().BeGreaterThanOrEqualTo(pDry[6]);
        pDry[6].Should().BeGreaterThan(0.5); // most samples will have a 6h dry run
    }

    [Fact]
    public void ProbDryWindow_returns_zero_when_window_exceeds_hours()
    {
        // Asking for a 6h dry block in a 4h day — structurally impossible.
        var q = new[] { 0.1, 0.1, 0.1, 0.1 };
        var p = DryWindow3gPredictor.ProbDryWindow(q, 6, new System.Random(42));
        p.Should().Be(0.0);
    }

    // ---------- Determinism ----------

    [Fact]
    public void ProbDryWindow_is_deterministic_with_fixed_seed()
    {
        var q = new[] { 0.3, 0.4, 0.5, 0.2, 0.7, 0.1, 0.6, 0.4, 0.3 };

        var run1 = DryWindow3gPredictor.ProbDryWindow(q, 4, new System.Random(42), nSamples: 2000);
        var run2 = DryWindow3gPredictor.ProbDryWindow(q, 4, new System.Random(42), nSamples: 2000);

        run1.Should().Be(run2);
    }

    [Fact]
    public void ProbDryWindow_multi_run_matches_single_run_for_same_window()
    {
        // The single-window overload and the multi-window overload should
        // agree to within MC noise (they consume Bernoullis differently —
        // single-window doesn't run the indicator-array loop). Use a large
        // sample count so the agreement window is tight.
        var q = new[] { 0.2, 0.5, 0.3, 0.4, 0.6, 0.1, 0.7, 0.3, 0.2 };
        const int n = 50_000;

        var single = DryWindow3gPredictor.ProbDryWindow(q, 4, new System.Random(7), nSamples: n);
        var multi  = DryWindow3gPredictor.ProbDryWindow(q, new[] { 4 }, new System.Random(7), nSamples: n)[4];

        // 1.5 / sqrt(N) ≈ 0.007 at N=50k — generous bound covers MC variance
        // from the slightly different rng-consumption order.
        multi.Should().BeApproximately(single, 0.01);
    }

    // ---------- Boundary behaviour ----------

    [Fact]
    public void ProbDryWindow_with_empty_q_returns_zero_for_every_window()
    {
        var q = System.Array.Empty<double>();
        var p = DryWindow3gPredictor.ProbDryWindow(q, new[] { 3, 4, 6 }, new System.Random(42));
        p.Should().HaveCount(3);
        p.Values.Should().AllSatisfy(v => v.Should().Be(0.0));
    }

    [Fact]
    public void ExtractDaytimeQ_returns_null_when_any_hour_missing()
    {
        var hourly = new Dictionary<DateTime, double>();
        var midnight = new DateTime(2026, 5, 4, 0, 0, 0, DateTimeKind.Utc);
        for (int h = 9; h < 18; h++) hourly[midnight.AddHours(h)] = 0.3;
        // Drop hour 12 — should make the whole day unscoreable.
        hourly.Remove(midnight.AddHours(12));

        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, midnight, 9, 18);
        q.Should().BeNull();
    }

    [Fact]
    public void ExtractDaytimeQ_returns_clamped_q_vector_when_full_coverage()
    {
        var hourly = new Dictionary<DateTime, double>();
        var midnight = new DateTime(2026, 5, 4, 0, 0, 0, DateTimeKind.Utc);
        // Mix of in-range and out-of-range values to verify the clamp.
        var inputs = new[] { 0.1, -0.05, 0.5, 0.9, 1.2, 0.0, 0.7, 0.4, 0.3 };
        for (int i = 0; i < 9; i++) hourly[midnight.AddHours(9 + i)] = inputs[i];

        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, midnight, 9, 18);
        q.Should().NotBeNull();
        q!.Should().HaveCount(9);
        q[0].Should().Be(0.1);
        q[1].Should().Be(0.0);   // -0.05 clamped to 0
        q[4].Should().Be(1.0);   // 1.2 clamped to 1
        q[5].Should().Be(0.0);
    }
}
