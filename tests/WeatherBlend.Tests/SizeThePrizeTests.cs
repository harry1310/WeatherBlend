using FluentAssertions;
using WeatherBlend.Analysis;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins the reusable size-the-prize math (OLS / R² / residual-prize / Corr / multi-seed
/// delta) extracted from the bakeoff harnesses 2026-06-21.
/// </summary>
public class SizeThePrizeTests
{
    [Fact]
    public void Ols_recovers_a_known_linear_relationship_with_R2_one()
    {
        // y = 2 + 3x exactly; design = [intercept, x].
        var x = new[] { new[] { 1.0, 0.0 }, new[] { 1.0, 1.0 }, new[] { 1.0, 2.0 }, new[] { 1.0, 3.0 } };
        var y = new[] { 2.0, 5.0, 8.0, 11.0 };

        var beta = SizeThePrize.Ols(x, y);

        beta[0].Should().BeApproximately(2.0, 1e-6);
        beta[1].Should().BeApproximately(3.0, 1e-6);
        SizeThePrize.R2(x, y, beta).Should().BeApproximately(1.0, 1e-6);
    }

    [Fact]
    public void ResidualR2_is_one_when_a_candidate_equals_the_residuals_and_zero_when_unrelated()
    {
        var resid = new[] { 1.0, -1.0, 1.0, -1.0 };
        // Candidate column == residuals → explains them fully.
        var explains = new[] { new[] { 1.0, 1.0 }, new[] { 1.0, -1.0 }, new[] { 1.0, 1.0 }, new[] { 1.0, -1.0 } };
        SizeThePrize.ResidualR2(explains, resid, minRows: 1).Should().BeApproximately(1.0, 1e-6);

        // Candidate orthogonal to the residuals → explains nothing.
        var unrelated = new[] { new[] { 1.0, 1.0 }, new[] { 1.0, 1.0 }, new[] { 1.0, -1.0 }, new[] { 1.0, -1.0 } };
        SizeThePrize.ResidualR2(unrelated, resid, minRows: 1).Should().BeApproximately(0.0, 1e-6);
    }

    [Fact]
    public void ResidualR2_returns_zero_below_the_min_row_floor()
    {
        var x = new[] { new[] { 1.0, 1.0 }, new[] { 1.0, -1.0 } };
        var r = new[] { 1.0, -1.0 };
        SizeThePrize.ResidualR2(x, r, minRows: 10).Should().Be(0.0);
    }

    [Fact]
    public void Corr_is_plus_or_minus_one_for_perfect_lines_and_zero_for_constant()
    {
        var x = new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 } };
        SizeThePrize.Corr(x, 0, new[] { 0.0, 2.0, 4.0, 6.0 }).Should().BeApproximately(1.0, 1e-6);
        SizeThePrize.Corr(x, 0, new[] { 6.0, 4.0, 2.0, 0.0 }).Should().BeApproximately(-1.0, 1e-6);
        SizeThePrize.Corr(x, 0, new[] { 5.0, 5.0, 5.0, 5.0 }).Should().Be(0.0);
    }

    [Fact]
    public void MultiSeedDelta_reports_mean_std_wins_and_noise_flag()
    {
        var seeds = new[] { 1, 2, 3, 4, 5 };

        // Arm uniformly 10% better than a unit baseline every seed.
        var clear = SizeThePrize.MultiSeedDelta(seeds, _ => 1.0, _ => 0.9);
        clear.BaseMean.Should().BeApproximately(1.0, 1e-9);
        clear.MeanDeltaPct.Should().BeApproximately(-10.0, 1e-6);
        clear.StdPct.Should().BeApproximately(0.0, 1e-9);
        clear.Wins.Should().Be(5);
        clear.WithinNoise.Should().BeFalse();
        clear.ToString().Should().Be("-10.0±0.0% [5/5]");
    }

    [Fact]
    public void MultiSeedDelta_flags_within_noise_when_the_arm_oscillates_around_baseline()
    {
        int i = 0;
        // Arm alternates +10% / −10% → mean ~0, large std → inside noise, 2/4 wins.
        var noisy = SizeThePrize.MultiSeedDelta(new[] { 1, 2, 3, 4 }, _ => 1.0, _ => (i++ % 2 == 0) ? 1.1 : 0.9);
        noisy.MeanDeltaPct.Should().BeApproximately(0.0, 1e-6);
        noisy.Wins.Should().Be(2);
        noisy.WithinNoise.Should().BeTrue("|mean| ≤ std means it's not distinguishable from noise");
    }
}
