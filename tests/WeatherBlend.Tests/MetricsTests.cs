using FluentAssertions;
using WeatherBlend.Evaluate;
using Xunit;

namespace WeatherBlend.Tests;

public class MetricsTests
{
    [Fact]
    public void Compute_returns_zero_error_on_perfect_predictions()
    {
        var pred = new[] { 10.0, 11.0, 12.0 };
        var actual = new[] { 10.0, 11.0, 12.0 };
        var s = Metrics.Compute(pred, actual);
        s.Mae.Should().Be(0);
        s.Rmse.Should().Be(0);
        s.Bias.Should().Be(0);
        s.N.Should().Be(3);
    }

    [Fact]
    public void Compute_matches_hand_worked_example()
    {
        // predicted − actual: +2, -1, +3, -2 → abs {2,1,3,2}, sq {4,1,9,4}, signed {+2,-1,+3,-2}
        var pred   = new[] { 12.0, 10.0, 15.0, 10.0 };
        var actual = new[] { 10.0, 11.0, 12.0, 12.0 };
        var s = Metrics.Compute(pred, actual);

        s.Mae.Should().BeApproximately((2 + 1 + 3 + 2) / 4.0, 1e-9);
        s.Rmse.Should().BeApproximately(Math.Sqrt((4 + 1 + 9 + 4) / 4.0), 1e-9);
        s.Bias.Should().BeApproximately((2 - 1 + 3 - 2) / 4.0, 1e-9);
        s.N.Should().Be(4);
    }

    [Fact]
    public void Compute_ignores_nan_pairs()
    {
        var pred = new[] { 1.0, double.NaN, 3.0 };
        var actual = new[] { 1.0, 2.0, 4.0 };
        var s = Metrics.Compute(pred, actual);
        s.N.Should().Be(2);
        s.Mae.Should().BeApproximately(0.5, 1e-9);  // |1-1|=0, |3-4|=1 → avg 0.5
    }

    [Fact]
    public void Compute_mismatched_lengths_throws()
    {
        var act = () => Metrics.Compute(new[] { 1.0 }, new[] { 1.0, 2.0 });
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void StratifiedMae_groups_by_bucket_key()
    {
        var pred   = new[] { 10.0, 11.0, 20.0, 22.0 };
        var actual = new[] { 10.0, 10.0, 20.0, 20.0 };
        var keys   = new[] { "a", "a", "b", "b" };

        var strat = Metrics.StratifiedMae(pred, actual, keys);

        strat["a"].Mae.Should().BeApproximately(0.5, 1e-9); // |0| + |1| / 2
        strat["a"].N.Should().Be(2);
        strat["b"].Mae.Should().BeApproximately(1.0, 1e-9); // |0| + |2| / 2
        strat["b"].N.Should().Be(2);
    }

    [Fact]
    public void Quintiles_assigns_roughly_equal_bins()
    {
        var values = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
        var labels = Metrics.Quintiles(values);

        // Each quintile should have ~20 members.
        var counts = labels.GroupBy(x => x).OrderBy(g => g.Key).ToList();
        counts.Should().HaveCount(5);
        foreach (var g in counts)
            g.Count().Should().BeInRange(19, 21);

        // Monotonic — smaller value → smaller quintile label.
        labels[0].Should().Be(0);
        labels[99].Should().Be(4);
    }

    [Theory]
    [InlineData(0,   "N")]
    [InlineData(44,  "N")]
    [InlineData(45,  "E")]
    [InlineData(134, "E")]
    [InlineData(135, "S")]
    [InlineData(224, "S")]
    [InlineData(225, "W")]
    [InlineData(314, "W")]
    [InlineData(315, "N")]
    [InlineData(359, "N")]
    public void WindQuadrant_bins_correctly(double deg, string expected)
    {
        Metrics.WindQuadrant(deg).Should().Be(expected);
    }

    [Fact]
    public void WindQuadrant_nan_is_unknown()
    {
        Metrics.WindQuadrant(double.NaN).Should().Be("?");
    }
}
