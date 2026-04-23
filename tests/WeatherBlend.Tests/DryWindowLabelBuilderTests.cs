using FluentAssertions;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Fixture-based checks on the dry-window target constructor. The maths is
/// simple enough that these cases double as the spec: what counts as a dry
/// window, what counts as unusable, how UTC-day boundaries are treated.
/// </summary>
public class DryWindowLabelBuilderTests
{
    private static readonly int[] AllWindows = { 3, 4, 6 };

    private static DryWindowLabelBuilder.HourlyTruth H(DateTime baseDay, int hour, double mm)
        => new(DateTime.SpecifyKind(baseDay.AddHours(hour), DateTimeKind.Utc), mm);

    private static List<DryWindowLabelBuilder.HourlyTruth> FullDay(
        DateTime baseDay,
        Func<int, double> mmForHour)
    {
        var rows = new List<DryWindowLabelBuilder.HourlyTruth>(24);
        for (int h = 0; h < 24; h++)
            rows.Add(H(baseDay, h, mmForHour(h)));
        return rows;
    }

    [Fact]
    public void All_dry_day_yields_label_one_for_every_window()
    {
        var day = new DateTime(2026, 1, 15, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, _ => 0.0);

        var result = DryWindowLabelBuilder.Build(hourly, AllWindows);

        result.UsableDays.Should().Be(1);
        result.DroppedDays.Should().Be(0);
        result.Labels.Should().HaveCount(3); // one label per window
        result.Labels.Should().OnlyContain(l => l.HasDryWindow);
    }

    [Fact]
    public void All_wet_day_yields_label_zero_for_every_window()
    {
        var day = new DateTime(2026, 1, 16, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, _ => 0.5);

        var result = DryWindowLabelBuilder.Build(hourly, AllWindows);

        result.UsableDays.Should().Be(1);
        result.Labels.Should().HaveCount(3);
        result.Labels.Should().OnlyContain(l => !l.HasDryWindow);
    }

    [Fact]
    public void Exactly_four_consecutive_dry_hours_gives_one_for_3h_and_4h_but_zero_for_6h()
    {
        // Hours 10..13 dry (4 consecutive); the rest wet.
        var day = new DateTime(2026, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, h => (h >= 10 && h <= 13) ? 0.0 : 0.5);

        var result = DryWindowLabelBuilder.Build(hourly, AllWindows);
        var byWindow = result.Labels.ToDictionary(l => l.WindowHours, l => l.HasDryWindow);

        byWindow[3].Should().BeTrue();
        byWindow[4].Should().BeTrue();
        byWindow[6].Should().BeFalse();
    }

    [Fact]
    public void Wet_threshold_is_strictly_below_point_one_mm()
    {
        // 0.10 mm/h is wet per the 3a convention; 0.09 mm/h is dry.
        var day = new DateTime(2026, 2, 2, 0, 0, 0, DateTimeKind.Utc);

        var hourlyWetExactly = FullDay(day, _ => 0.1);
        DryWindowLabelBuilder.Build(hourlyWetExactly, new[] { 3 })
            .Labels.Single().HasDryWindow.Should().BeFalse();

        var hourlyJustDry = FullDay(day, _ => 0.09);
        DryWindowLabelBuilder.Build(hourlyJustDry, new[] { 3 })
            .Labels.Single().HasDryWindow.Should().BeTrue();
    }

    [Fact]
    public void Missing_hour_marks_the_whole_day_unusable_for_every_window()
    {
        var day = new DateTime(2026, 2, 3, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, _ => 0.0);
        hourly.RemoveAll(x => x.HourUtc.Hour == 15); // drop 15:00

        var result = DryWindowLabelBuilder.Build(hourly, AllWindows);

        result.UsableDays.Should().Be(0);
        result.DroppedDays.Should().Be(1);
        result.Labels.Should().BeEmpty();
    }

    [Fact]
    public void Cross_midnight_dry_stretch_is_not_credited_to_either_day()
    {
        // Monday wet 00..21, dry 22..23. Tuesday dry 00..02, wet 03..23.
        // Cross-midnight run is 5 hours (22 Mon → 02 Tue) but neither day sees
        // ≥ 4 consecutive dry hours inside its own UTC window: Monday has 2h at
        // the end, Tuesday has 3h at the start. This is the accepted phase-3b
        // limitation documented in the brief.
        var mon = new DateTime(2026, 3, 2, 0, 0, 0, DateTimeKind.Utc);
        var tue = mon.AddDays(1);

        var hourly = new List<DryWindowLabelBuilder.HourlyTruth>();
        hourly.AddRange(FullDay(mon, h => h >= 22 ? 0.0 : 0.5));
        hourly.AddRange(FullDay(tue, h => h <= 2  ? 0.0 : 0.5));

        var result = DryWindowLabelBuilder.Build(hourly, new[] { 4 });
        result.UsableDays.Should().Be(2);
        result.Labels.Should().OnlyContain(l => !l.HasDryWindow);
    }

    [Fact]
    public void Missing_whole_day_inside_span_is_counted_as_dropped()
    {
        // Day 1 + Day 3 present with 24 hours each; Day 2 absent entirely.
        var day1 = new DateTime(2026, 4, 1, 0, 0, 0, DateTimeKind.Utc);
        var day3 = day1.AddDays(2);

        var hourly = new List<DryWindowLabelBuilder.HourlyTruth>();
        hourly.AddRange(FullDay(day1, _ => 0.0));
        hourly.AddRange(FullDay(day3, _ => 0.0));

        var result = DryWindowLabelBuilder.Build(hourly, new[] { 3 });

        result.CandidateDays.Should().Be(3);
        result.UsableDays.Should().Be(2);
        result.DroppedDays.Should().Be(1);
    }

    [Fact]
    public void Dry_run_at_end_of_day_still_qualifies()
    {
        // Hours 18..23 dry (6 consecutive); earlier hours wet.
        var day = new DateTime(2026, 5, 10, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, h => h >= 18 ? 0.0 : 0.5);

        var result = DryWindowLabelBuilder.Build(hourly, new[] { 6 });
        result.Labels.Single().HasDryWindow.Should().BeTrue();
    }

    [Fact]
    public void Dry_run_at_start_of_day_still_qualifies()
    {
        // Hours 00..05 dry (6 consecutive); rest wet.
        var day = new DateTime(2026, 5, 11, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, h => h <= 5 ? 0.0 : 0.5);

        var result = DryWindowLabelBuilder.Build(hourly, new[] { 6 });
        result.Labels.Single().HasDryWindow.Should().BeTrue();
    }

    [Fact]
    public void Empty_input_yields_empty_result()
    {
        var result = DryWindowLabelBuilder.Build(Array.Empty<DryWindowLabelBuilder.HourlyTruth>(), AllWindows);
        result.Labels.Should().BeEmpty();
        result.CandidateDays.Should().Be(0);
        result.UsableDays.Should().Be(0);
        result.DroppedDays.Should().Be(0);
    }

    [Fact]
    public void Invalid_window_length_throws()
    {
        var day = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var hourly = FullDay(day, _ => 0.0);

        Action zero = () => DryWindowLabelBuilder.Build(hourly, new[] { 0 });
        zero.Should().Throw<ArgumentException>();

        Action negative = () => DryWindowLabelBuilder.Build(hourly, new[] { -1 });
        negative.Should().Throw<ArgumentException>();

        Action tooLong = () => DryWindowLabelBuilder.Build(hourly, new[] { 25 });
        tooLong.Should().Throw<ArgumentException>();

        Action empty = () => DryWindowLabelBuilder.Build(hourly, Array.Empty<int>());
        empty.Should().Throw<ArgumentException>();
    }
}
