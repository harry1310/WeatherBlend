using FluentAssertions;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Unit tests on the static helpers in <see cref="DryWindowFeatureBuilder"/>.
/// Exercises ShapeFeatures (used by both the spec-driven feature builder and
/// the 3d-shape report) directly via its public surface — the SQL loaders are
/// integration-tested implicitly by training/predict pipelines.
/// </summary>
public class DryWindowFeatureBuilderTests
{
    private static DryWindowFeatureBuilder.ForecastDay DayOf(Func<int, double> precipForHour,
                                                              Func<int, double>? probForHour = null)
    {
        var day = new DryWindowFeatureBuilder.ForecastDay();
        for (int h = 0; h < 24; h++)
        {
            day.SetHour(h, new DryWindowFeatureBuilder.ForecastRow(
                new DateTime(2026, 3, 1, h, 0, 0, DateTimeKind.Utc),
                "gfs_seamless",
                precipForHour(h),
                probForHour?.Invoke(h),
                Rh: 80, T: 10, Td: 8,
                CloudLow: 50, CloudMid: 30, CloudHigh: 20,
                Cape: 100, Wind: 5));
        }
        return day;
    }

    [Fact]
    public void ShapeFeatures_all_dry_returns_sentinel_first_and_last_wet()
    {
        var dry = DayOf(_ => 0.0);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { dry, dry, dry, dry, dry, dry };

        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        shape.FirstWetHour.Should().Be(24.0);
        shape.LastWetHour.Should().Be(-1.0);
        shape.LongestDryRun.Should().Be(24.0);
        shape.LongestWetRun.Should().Be(0.0);
        shape.NRainEvents.Should().Be(0.0);
        shape.MorningPrecipSum.Should().Be(0.0);
        shape.AfternoonPrecipSum.Should().Be(0.0);
    }

    [Fact]
    public void ShapeFeatures_all_wet_makes_one_event_spanning_the_day()
    {
        var wet = DayOf(_ => 0.5);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { wet, wet };

        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        shape.FirstWetHour.Should().Be(0.0);
        shape.LastWetHour.Should().Be(23.0);
        shape.LongestDryRun.Should().Be(0.0);
        shape.LongestWetRun.Should().Be(24.0);
        shape.NRainEvents.Should().Be(1.0);
        shape.MorningPrecipSum.Should().BeApproximately(3.0, 1e-9);
        shape.AfternoonPrecipSum.Should().BeApproximately(3.0, 1e-9);
    }

    [Fact]
    public void ShapeFeatures_two_distinct_events_counts_them_separately_and_keeps_longest_wet()
    {
        var d = DayOf(h => (h >= 3 && h <= 5) || (h >= 10 && h <= 14) ? 0.4 : 0.0);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { d };

        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        shape.NRainEvents.Should().Be(2.0);
        shape.LongestWetRun.Should().Be(5.0);
        shape.LongestDryRun.Should().BeGreaterThanOrEqualTo(9.0);
        shape.FirstWetHour.Should().Be(3.0);
        shape.LastWetHour.Should().Be(14.0);
    }

    [Fact]
    public void ShapeFeatures_morning_and_afternoon_buckets_only_count_their_hours()
    {
        var d = DayOf(h => (h == 6 || h == 12) ? 1.5 : 0.0);
        var shape = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { d });

        shape.MorningPrecipSum.Should().BeApproximately(1.5, 1e-9);
        shape.AfternoonPrecipSum.Should().BeApproximately(1.5, 1e-9);

        // Hour-2 wet sits outside both buckets — verify exclusivity.
        var d2 = DayOf(h => (h == 2) ? 1.0 : 0.0);
        var shape2 = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { d2 });
        shape2.MorningPrecipSum.Should().Be(0.0);
        shape2.AfternoonPrecipSum.Should().Be(0.0);
    }

    [Fact]
    public void ShapeFeatures_threshold_uses_inclusive_zero_point_one_mm()
    {
        var atThreshold = DayOf(h => h == 10 ? 0.1 : 0.0);
        var shapeAt = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { atThreshold });
        shapeAt.NRainEvents.Should().Be(1.0);
        shapeAt.FirstWetHour.Should().Be(10.0);

        var below = DayOf(h => h == 10 ? 0.099 : 0.0);
        var shapeBelow = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { below });
        shapeBelow.NRainEvents.Should().Be(0.0);
        shapeBelow.FirstWetHour.Should().Be(24.0);
    }

    [Fact]
    public void ShapeFeatures_no_models_present_returns_all_nan()
    {
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { null, null, null, null, null, null };
        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        double.IsNaN(shape.FirstWetHour).Should().BeTrue();
        double.IsNaN(shape.LastWetHour).Should().BeTrue();
        double.IsNaN(shape.LongestDryRun).Should().BeTrue();
        double.IsNaN(shape.LongestWetRun).Should().BeTrue();
        double.IsNaN(shape.NRainEvents).Should().BeTrue();
        double.IsNaN(shape.MorningPrecipSum).Should().BeTrue();
        double.IsNaN(shape.AfternoonPrecipSum).Should().BeTrue();
    }

    [Fact]
    public void ShapeFeatures_partial_model_coverage_uses_present_models_only_per_hour()
    {
        // Model A has hours 0–11 only; model B has hours 12–23 only. Each hour has a
        // single contributor, so the ensemble mean stays 0.5 across the full day.
        var dayA = new DryWindowFeatureBuilder.ForecastDay();
        var dayB = new DryWindowFeatureBuilder.ForecastDay();
        for (int h = 0; h < 24; h++)
        {
            var which = h < 12 ? dayA : dayB;
            which.SetHour(h, new DryWindowFeatureBuilder.ForecastRow(
                new DateTime(2026, 3, 1, h, 0, 0, DateTimeKind.Utc),
                "gfs_seamless", Precip: 0.5, Prob: 0.0,
                Rh: 80, T: 10, Td: 8,
                CloudLow: 50, CloudMid: 30, CloudHigh: 20,
                Cape: 100, Wind: 5));
        }
        var shape = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { dayA, dayB });
        shape.LongestWetRun.Should().Be(24.0);
        shape.NRainEvents.Should().Be(1.0);
    }
}
