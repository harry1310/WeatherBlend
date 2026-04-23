using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class IsotonicCalibratorTests
{
    [Fact]
    public void Fit_empty_input_returns_identity_calibrator()
    {
        var c = IsotonicCalibrator.Fit(Array.Empty<double>(), Array.Empty<double>());
        c.Knots.Should().BeEmpty();
        c.Apply(0.3).Should().Be(0.3);
    }

    [Fact]
    public void Fit_monotone_input_preserves_pool_means_as_knots()
    {
        // Already monotone — each row stays its own pool.
        var raws   = new[] { 0.1, 0.2, 0.3, 0.4 };
        var truth  = new[] { 0.0, 0.0, 1.0, 1.0 };
        var c = IsotonicCalibrator.Fit(raws, truth);
        c.Knots.Should().HaveCount(4);
        c.Knots.Select(k => k.Y).Should().Equal(new[] { 0.0, 0.0, 1.0, 1.0 });
    }

    [Fact]
    public void Fit_pools_adjacent_violators_into_weighted_mean()
    {
        // (0.1, 1), (0.2, 0), (0.3, 1) — the first violator pair merges into a
        // pool of mean 0.5 over [0.1, 0.2]; the third sample keeps its own
        // pool (mean 1 at 0.3) because 0.5 ≤ 1 is already monotone.
        var raws  = new[] { 0.1, 0.2, 0.3 };
        var truth = new[] { 1.0, 0.0, 1.0 };
        var c = IsotonicCalibrator.Fit(raws, truth);

        c.Knots.Should().HaveCount(2);
        c.Knots[0].X.Should().Be(0.1);
        c.Knots[0].XMax.Should().Be(0.2);
        c.Knots[0].Y.Should().BeApproximately(0.5, 1e-9);
        c.Knots[1].X.Should().Be(0.3);
        c.Knots[1].XMax.Should().Be(0.3);
        c.Knots[1].Y.Should().BeApproximately(1.0, 1e-9);
    }

    [Fact]
    public void Fit_cascades_merges_when_pool_mean_itself_violates()
    {
        // (0.1, 0), (0.2, 1), (0.3, 0) — pushing the third point creates a
        // local violator with the second, merging to mean 0.5. That new pool
        // (0.5) is now below the first (0.0)... no wait, 0.0 ≤ 0.5, so the
        // first stays separate. But reverse the pattern and we see a cascade:
        //   (0.1, 1), (0.2, 1), (0.3, 0) → third vs second means 1 > 0 merge
        //   to (1+0)/2 = 0.5; then second vs first means 1 > 0.5 — cascade
        //   merge to (1+1+0)/3 = 2/3 over [0.1, 0.3].
        var raws  = new[] { 0.1, 0.2, 0.3 };
        var truth = new[] { 1.0, 1.0, 0.0 };
        var c = IsotonicCalibrator.Fit(raws, truth);

        c.Knots.Should().HaveCount(1);
        c.Knots[0].Y.Should().BeApproximately(2.0 / 3.0, 1e-9);
        c.Knots[0].X.Should().Be(0.1);
        c.Knots[0].XMax.Should().Be(0.3);
    }

    [Fact]
    public void Fit_output_is_monotone_nondecreasing()
    {
        var rng = new Random(42);
        var raws = Enumerable.Range(0, 200).Select(_ => rng.NextDouble()).ToArray();
        // Noisy monotone truth so we get several violations.
        var truth = raws.Select(r => rng.NextDouble() < r ? 1.0 : 0.0).ToArray();
        var c = IsotonicCalibrator.Fit(raws, truth);

        for (var i = 1; i < c.Knots.Count; i++)
            c.Knots[i].Y.Should().BeGreaterThanOrEqualTo(c.Knots[i - 1].Y);
    }

    [Fact]
    public void Apply_clamps_outside_knot_range()
    {
        var raws  = new[] { 0.2, 0.5, 0.8 };
        var truth = new[] { 0.0, 0.5, 1.0 };
        var c = IsotonicCalibrator.Fit(raws, truth);

        c.Apply(0.0).Should().Be(c.Knots[0].Y);
        c.Apply(1.0).Should().Be(c.Knots[^1].Y);
    }

    [Fact]
    public void Apply_linearly_interpolates_between_knots()
    {
        // Three singleton pools at (0.1→0.0), (0.5→0.4), (0.9→1.0).
        var raws  = new[] { 0.1, 0.5, 0.9 };
        var truth = new[] { 0.0, 0.4, 1.0 };
        var c = IsotonicCalibrator.Fit(raws, truth);

        // Midpoint between 0.1 and 0.5 — calibrated output should be midway.
        c.Apply(0.3).Should().BeApproximately(0.2, 1e-9);
        // Three-quarter mark between 0.5 and 0.9.
        c.Apply(0.8).Should().BeApproximately(0.85, 1e-9);
    }

    [Fact]
    public void Container_round_trips_through_json()
    {
        var a = IsotonicCalibrator.Fit(new[] { 0.1, 0.3, 0.7 }, new[] { 0.0, 0.5, 1.0 });
        var b = IsotonicCalibrator.Fit(new[] { 0.1, 0.4 }, new[] { 0.2, 0.8 });
        var container = new PrecipIsotonicCalibration
        {
            ByLead = { ["24"] = a, ["48"] = b },
            SourceVersion = "v2026-04-23_071842",
            SourceRowCount = 600,
            FitAtUtc = new DateTime(2026, 4, 23, 18, 0, 0, DateTimeKind.Utc),
        };

        var tmp = Path.GetTempFileName();
        try
        {
            container.SaveTo(tmp);
            var reloaded = PrecipIsotonicCalibration.LoadFrom(tmp);

            reloaded.SourceVersion.Should().Be("v2026-04-23_071842");
            reloaded.SourceRowCount.Should().Be(600);
            reloaded.ByLead.Keys.Should().BeEquivalentTo(new[] { "24", "48" });
            reloaded.ByLead["24"].Apply(0.3).Should().BeApproximately(a.Apply(0.3), 1e-12);
            reloaded.ByLead["48"].Apply(0.25).Should().BeApproximately(b.Apply(0.25), 1e-12);
        }
        finally
        {
            File.Delete(tmp);
        }
    }
}
