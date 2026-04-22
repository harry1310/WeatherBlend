using FluentAssertions;
using WeatherBlend.Evaluate.Precip;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipMetricsTests
{
    // ------ Brier --------------------------------------------------------------

    [Fact]
    public void Brier_perfect_deterministic_forecast_is_zero()
    {
        var prob   = new[] { 1.0, 0.0, 1.0, 0.0 };
        var truth  = new[] { 1.0, 0.0, 1.0, 0.0 };
        PrecipMetrics.Brier(prob, truth).Should().Be(0);
    }

    [Fact]
    public void Brier_all_wrong_deterministic_is_one()
    {
        var prob  = new[] { 1.0, 0.0 };
        var truth = new[] { 0.0, 1.0 };
        PrecipMetrics.Brier(prob, truth).Should().Be(1);
    }

    [Fact]
    public void Brier_climatology_matches_variance()
    {
        // Always forecast p = base rate 0.3; truth 3/10 wet.
        // Brier = (1-0.3)^2*3 + (0-0.3)^2*7 / 10 = (0.49*3 + 0.09*7)/10 = (1.47+0.63)/10 = 0.21
        var prob  = Enumerable.Repeat(0.3, 10).ToArray();
        var truth = new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        PrecipMetrics.Brier(prob, truth).Should().BeApproximately(0.21, 1e-9);
    }

    [Fact]
    public void BrierSkillScore_positive_when_better_than_reference()
    {
        PrecipMetrics.BrierSkillScore(brier: 0.10, brierReference: 0.20)
            .Should().BeApproximately(0.5, 1e-9);
    }

    // ------ Reliability --------------------------------------------------------

    [Fact]
    public void Reliability_groups_into_bins_with_counts()
    {
        // 4 forecasts in bin [0.0, 0.1), 2 forecasts in bin [0.9, 1.0]
        var prob  = new[] { 0.05, 0.05, 0.05, 0.05, 0.95, 0.95 };
        var truth = new[] { 0.0,  1.0,  0.0,  0.0,  1.0,  1.0  };
        var bins = PrecipMetrics.Reliability(prob, truth, nBins: 10);
        bins.Should().HaveCount(10);

        bins[0].N.Should().Be(4);
        bins[0].ObservedFrequency.Should().BeApproximately(0.25, 1e-9);  // 1 wet out of 4

        bins[9].N.Should().Be(2);
        bins[9].ObservedFrequency.Should().BeApproximately(1.0, 1e-9);   // 2 wet out of 2
    }

    [Fact]
    public void Reliability_empty_bins_return_nan_stats_but_zero_count()
    {
        var prob  = new[] { 0.5 };
        var truth = new[] { 1.0 };
        var bins = PrecipMetrics.Reliability(prob, truth, nBins: 4);
        // Bin 2 covers [0.5, 0.75) and contains the point. Others are empty.
        bins[0].N.Should().Be(0);
        bins[0].MeanForecast.Should().Be(double.NaN);
        bins[2].N.Should().Be(1);
        bins[2].ObservedFrequency.Should().Be(1);
    }

    // ------ FrequencyBias ------------------------------------------------------

    [Fact]
    public void FrequencyBias_one_when_counts_match()
    {
        var forecast = new[] { 1.0, 0.0, 1.0, 0.0 };
        var truth    = new[] { 0.0, 1.0, 1.0, 0.0 };
        PrecipMetrics.FrequencyBias(forecast, truth).Should().Be(1.0);
    }

    [Fact]
    public void FrequencyBias_over_forecasts_wet_gives_above_one()
    {
        var forecast = new[] { 1.0, 1.0, 1.0, 0.0 };
        var truth    = new[] { 1.0, 0.0, 0.0, 0.0 };
        PrecipMetrics.FrequencyBias(forecast, truth).Should().Be(3.0);
    }

    // ------ CRPS deterministic -------------------------------------------------

    [Fact]
    public void CrpsDeterministic_matches_mae()
    {
        var fc    = new[] { 1.0, 2.0, 3.0 };
        var truth = new[] { 0.5, 2.5, 4.0 };
        PrecipMetrics.CrpsDeterministic(fc, truth)
            .Should().BeApproximately((0.5 + 0.5 + 1.0) / 3.0, 1e-9);
    }

    // ------ Contingency --------------------------------------------------------

    [Fact]
    public void Contingency_counts_all_four_cells()
    {
        var forecast = new[] { 1.0, 1.0, 0.0, 0.0, 1.0 };
        var truth    = new[] { 1.0, 0.0, 1.0, 0.0, 1.0 };
        var c = PrecipMetrics.BuildContingency(forecast, truth);

        c.Hits.Should().Be(2);
        c.Misses.Should().Be(1);
        c.FalseAlarms.Should().Be(1);
        c.TrueNegatives.Should().Be(1);
        c.Total.Should().Be(5);
        c.PodHitRate.Should().BeApproximately(2.0 / 3.0, 1e-9);
        c.FarFalseAlarmRatio.Should().BeApproximately(1.0 / 3.0, 1e-9);
        c.Csi.Should().BeApproximately(2.0 / 4.0, 1e-9);
        c.FrequencyBias.Should().BeApproximately(3.0 / 3.0, 1e-9);
    }

    [Fact]
    public void Brier_throws_on_mismatched_lengths()
    {
        var a = new[] { 0.5 };
        var b = new[] { 0.0, 1.0 };
        var act = () => PrecipMetrics.Brier(a, b);
        act.Should().Throw<ArgumentException>();
    }
}
