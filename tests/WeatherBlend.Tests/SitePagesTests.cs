using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

public class SitePagesTests
{
    private const string Station = "ea_bellever_dartmoor";
    private static readonly DateTime Day = new(2026, 4, 1, 0, 0, 0, DateTimeKind.Utc);

    [Fact]
    public void ComputeObservedDryWindows_flags_day_with_long_enough_run_as_dry()
    {
        // 6 dry hours 10Z-15Z (≤ 0.1 mm), rest wet. 6h window should fire.
        var hourly = BuildHourly(Day, h => h >= 10 && h <= 15 ? 0.0 : 2.0);
        var input = MakeInput(hourly, windowHours: 6);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 6, Day)].Should().BeTrue();
    }

    [Fact]
    public void ComputeObservedDryWindows_needs_consecutive_run_not_total_dry_hours()
    {
        // 5 dry hours total, but broken: 2 + 3 with a wet hour between. 4h window shouldn't fire.
        var hourly = BuildHourly(Day, h => h == 10 || h == 11 || h == 13 || h == 14 || h == 15 ? 0.0 : 2.0);
        var input = MakeInput(hourly, windowHours: 4);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 4, Day)].Should().BeFalse();
    }

    [Fact]
    public void ComputeObservedDryWindows_treats_exactly_0_1_mm_as_dry()
    {
        // Boundary: the classifier uses ≤ 0.1 mm/h as "dry".
        var hourly = BuildHourly(Day, h => h >= 10 && h <= 12 ? 0.1 : 5.0);
        var input = MakeInput(hourly, windowHours: 3);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 3, Day)].Should().BeTrue();
    }

    [Fact]
    public void ComputeObservedDryWindows_skips_day_with_missing_hour()
    {
        // Drop hour 5 entirely — need full 24-hour coverage for a verdict.
        var hourly = BuildHourly(Day, h => 0.0);
        hourly.Remove(Day.AddHours(5));
        var input = MakeInput(hourly, windowHours: 3);

        var result = SitePages.ComputeObservedDryWindows(input);

        result.Should().NotContainKey((Station, 3, Day));
    }

    [Fact]
    public void ComputeObservedDryWindows_skips_station_with_no_rainfall_loaded()
    {
        // Prediction references a station whose rainfall dict is empty — should silently skip.
        var input = new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = Day.AddDays(1),
            WindowStartUtc = Day,
            Predictions = Array.Empty<PredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = new[]
            {
                new SitePages.DryWindowForecastPoint(Station, 3, "v1", Day, Day, 24, 0.5, 0.4, null),
            },
            RainfallTruth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
        };

        var result = SitePages.ComputeObservedDryWindows(input);

        result.Should().BeEmpty();
    }

    private static Dictionary<DateTime, double> BuildHourly(DateTime day, Func<int, double> mmForHour)
    {
        var dict = new Dictionary<DateTime, double>();
        for (int h = 0; h < 24; h++) dict[day.AddHours(h)] = mmForHour(h);
        return dict;
    }

    private static SitePages.SiteInputs MakeInput(Dictionary<DateTime, double> hourly, int windowHours)
    {
        var rainfall = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase)
        {
            [Station] = hourly,
        };
        return new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = Day.AddDays(1),
            WindowStartUtc = Day,
            Predictions = Array.Empty<PredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = new[]
            {
                new SitePages.DryWindowForecastPoint(Station, windowHours, "v1", Day, Day, 24, 0.5, 0.4, null),
            },
            RainfallTruth = rainfall,
        };
    }
}
