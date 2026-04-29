using FluentAssertions;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins per-day UTC bound resolution for the local-time daytime window.
/// The whole reason this class exists is DST — these tests cover BST and GMT
/// regular days plus the spring-forward and fall-back transitions, where a
/// naive "9–18 UTC" config would slide an hour against wall-clock walking
/// time twice a year.
/// </summary>
public class DaytimeWindowTests
{
    [Fact]
    public void Summer_BST_day_resolves_9_18_local_to_8_17_UTC()
    {
        // 2026-07-15 — middle of British Summer Time, UTC+1. 09:00 BST = 08:00 UTC,
        // 18:00 BST = 17:00 UTC.
        var window = new DaytimeWindow(9, 18, "Europe/London");
        var (start, end) = window.UtcHourRangeFor(new DateOnly(2026, 7, 15));
        start.Should().Be(8);
        end.Should().Be(17);
    }

    [Fact]
    public void Winter_GMT_day_resolves_9_18_local_to_9_18_UTC()
    {
        // 2026-01-15 — GMT, UTC+0. 09:00 GMT = 09:00 UTC, 18:00 GMT = 18:00 UTC.
        var window = new DaytimeWindow(9, 18, "Europe/London");
        var (start, end) = window.UtcHourRangeFor(new DateOnly(2026, 1, 15));
        start.Should().Be(9);
        end.Should().Be(18);
    }

    [Fact]
    public void Spring_forward_day_still_resolves_to_BST_offsets()
    {
        // 2026 spring forward: 30 March, 01:00 GMT → 02:00 BST. By the time the
        // 09–18 walking window starts, the day is already in BST. Resolves the
        // same as a regular summer day.
        var window = new DaytimeWindow(9, 18, "Europe/London");
        var (start, end) = window.UtcHourRangeFor(new DateOnly(2026, 3, 29));
        start.Should().Be(8);
        end.Should().Be(17);
    }

    [Fact]
    public void Fall_back_day_still_resolves_to_GMT_offsets()
    {
        // 2026 fall back: 25 October, 02:00 BST → 01:00 GMT. By the time the
        // 09–18 walking window starts, the day is already in GMT. Resolves the
        // same as a regular winter day.
        var window = new DaytimeWindow(9, 18, "Europe/London");
        var (start, end) = window.UtcHourRangeFor(new DateOnly(2026, 10, 25));
        start.Should().Be(9);
        end.Should().Be(18);
    }

    [Fact]
    public void Pure_UTC_window_is_identity()
    {
        // The "scan the whole UTC day" sentinel used by legacy callers and tests.
        var (start, end) = DryWindowLabelBuilder.FullUtcDay.UtcHourRangeFor(new DateOnly(2026, 6, 1));
        start.Should().Be(0);
        end.Should().Be(24);
    }

    [Fact]
    public void Duration_reports_local_hour_count()
    {
        new DaytimeWindow(9, 18, "Europe/London").DurationHours.Should().Be(9);
        new DaytimeWindow(0, 24, TimeZoneInfo.Utc).DurationHours.Should().Be(24);
        new DaytimeWindow(6, 22, "Europe/London").DurationHours.Should().Be(16);
    }

    [Theory]
    [InlineData(-1, 18)]
    [InlineData(24, 25)]
    [InlineData(18, 9)]   // end ≤ start
    [InlineData(9, 25)]
    public void Constructor_rejects_invalid_hour_pairs(int startLocal, int endLocal)
    {
        Action ctor = () => new DaytimeWindow(startLocal, endLocal, TimeZoneInfo.Utc);
        ctor.Should().Throw<ArgumentOutOfRangeException>();
    }
}
