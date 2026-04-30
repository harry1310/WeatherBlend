using FluentAssertions;
using WeatherBlend.Evaluate.StartHour;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin <see cref="StartHourTruth.ValidStartsFor"/>. Truth derivation is the
/// load-bearing input to verify — a wrong truth set silently inflates or
/// deflates skill scores forever.
/// </summary>
public class StartHourTruthTests
{
    private static IReadOnlyDictionary<int, double> Hours(params (int Hour, double Mm)[] items)
        => items.ToDictionary(x => x.Hour, x => x.Mm);

    [Fact]
    public void ValidStartsFor_returns_every_start_when_whole_daytime_is_dry()
    {
        // 9-hour daytime, all hours below threshold → every 6-hour window is
        // dry, candidate starts {8..11} all valid.
        var hourly = Hours(
            (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (13, 0), (14, 0), (15, 0), (16, 0));

        var truth = StartHourTruth.ValidStartsFor(hourly,
            daytimeStartUtc: 8, daytimeEndUtc: 17, windowHours: 6);

        truth.Should().BeEquivalentTo(new[] { 8, 9, 10, 11 });
    }

    [Fact]
    public void ValidStartsFor_excludes_starts_whose_window_contains_a_wet_hour()
    {
        // Wet hour at 13:00 (≥ 0.1 mm). 6-hour windows starting at 8..11 each
        // contain hour 13 (s=8 covers 8..13, s=11 covers 11..16). Every start
        // is invalid; truth = {}.
        var hourly = Hours(
            (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (13, 0.5), (14, 0), (15, 0), (16, 0));

        var truth = StartHourTruth.ValidStartsFor(hourly, 8, 17, 6);
        truth.Should().BeEmpty();
    }

    [Fact]
    public void ValidStartsFor_with_smaller_window_excludes_only_starts_through_the_wet_hour()
    {
        // 4-hour window, wet at hour 12. Only starts where [s, s+4) contains
        // 12 are invalid: s=9 (9..12), s=10 (10..13), s=11 (11..14), s=12
        // (12..15). Valid: {8, 13}.
        var hourly = Hours(
            (8, 0), (9, 0), (10, 0), (11, 0), (12, 0.5),
            (13, 0), (14, 0), (15, 0), (16, 0));

        var truth = StartHourTruth.ValidStartsFor(hourly, 8, 17, 4);
        truth.Should().BeEquivalentTo(new[] { 8, 13 });
    }

    [Fact]
    public void ValidStartsFor_returns_null_when_a_daytime_hour_is_missing()
    {
        // Hour 12 unobserved → can't tell dry from missing. Drop the day.
        var hourly = Hours(
            (8, 0), (9, 0), (10, 0), (11, 0),
            /* 12 missing */ (13, 0), (14, 0), (15, 0), (16, 0));

        StartHourTruth.ValidStartsFor(hourly, 8, 17, 6).Should().BeNull();
    }

    [Fact]
    public void ValidStartsFor_returns_empty_when_window_exceeds_daytime_span()
    {
        var hourly = Hours(
            (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (13, 0), (14, 0), (15, 0), (16, 0));

        StartHourTruth.ValidStartsFor(hourly, 8, 17, windowHours: 12)
            .Should().BeEmpty();
    }

    [Fact]
    public void ValidStartsFor_uses_strict_below_zero_point_one_threshold()
    {
        // 0.1mm is the wet threshold (matches DryWindowLabelBuilder + the
        // training-time label rule). Exactly 0.1mm = wet, 0.099 = dry.
        var hourly = Hours(
            (8, 0.099),  // dry (just below)
            (9, 0), (10, 0), (11, 0), (12, 0),
            (13, 0.1),    // wet (at threshold)
            (14, 0), (15, 0), (16, 0));

        // 6-hour windows: every start [8..11] covers hour 13 → all invalid.
        StartHourTruth.ValidStartsFor(hourly, 8, 17, 6).Should().BeEmpty();

        // 5-hour windows: s=8 covers 8..12 (no hour 13), valid. Others through
        // 13 are invalid. s=14 (covers 14..18) is out of [8, 17) range — span
        // is 9, 5-hour starts ∈ [8..12] = 5 candidates.
        var t5 = StartHourTruth.ValidStartsFor(hourly, 8, 17, 5);
        t5.Should().BeEquivalentTo(new[] { 8 });
    }
}
