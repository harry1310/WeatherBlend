using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

// Phase 3f village 09Z→09Z daily totals (Harry 2026-06-18). Locks the rain-day
// boundary math: an hour belongs to the 09Z→09Z window that STARTS at the most
// recent 09:00Z at/before it, and only near-complete windows are shown.
public class SitePagesVillageDailyTests
{
    // Minimal village-composite row — only ValidTimeUtc + MedianMmPerHr matter to
    // the daily table; the rest are filled with inert values.
    private static RainfallAmountPredictionRow Row(DateTime validUtc, double medianMm) => new()
    {
        LocationName = "membury_devon",
        TruthStation = "membury_devon_village",
        ModelVersion = "test",
        Precip3aVersion = "test",
        PredictionMadeAtUtc = validUtc.AddHours(-24),
        ValidTimeUtc = validUtc,
        LeadHours = 24,
        Pi = 0, MuLog = double.NaN, SigmaLog = double.NaN,
        MeanMmPerHr = medianMm, MedianMmPerHr = medianMm,
        P2_5MmPerHr = 0, P10MmPerHr = 0, P50MmPerHr = 0, P90MmPerHr = 0, P97_5MmPerHr = 0,
        PExceed0_1 = 0, PExceed1 = 0, PExceed5 = 0, PExceed10 = 0,
    };

    // A full 09Z→09Z window = the 24 hourly stamps 09Z(day) .. 08Z(day+1).
    private static IEnumerable<RainfallAmountPredictionRow> FullDay(DateTime nineZ, double perHourMm)
    {
        for (int h = 0; h < 24; h++)
            yield return Row(nineZ.AddHours(h), perHourMm);
    }

    [Fact]
    public void Village9to9_sums_a_full_window_into_one_dated_row()
    {
        var nineZ = new DateTime(2026, 6, 19, 9, 0, 0, DateTimeKind.Utc);
        var rows = FullDay(nineZ, 0.1).ToList();   // 24 × 0.1 = 2.4 mm

        var html = SitePages.RenderVillage9to9DailyTable(rows, lead: 24);

        html.Should().Contain("19 Jun");
        html.Should().Contain("2.4 mm");
        // A complete window carries no hour-count flag.
        html.Should().NotContain("(24h)");
    }

    [Fact]
    public void Village9to9_splits_windows_on_the_0900Z_boundary()
    {
        // Two back-to-back full windows; the 08Z hour must land in the EARLIER
        // day and the 09Z hour must open the NEXT — proving the (valid−9h) key.
        var jun18_09 = new DateTime(2026, 6, 18, 9, 0, 0, DateTimeKind.Utc);
        var jun19_09 = new DateTime(2026, 6, 19, 9, 0, 0, DateTimeKind.Utc);
        var rows = FullDay(jun18_09, 0.2).Concat(FullDay(jun19_09, 0.1)).ToList();

        var html = SitePages.RenderVillage9to9DailyTable(rows, lead: 24);

        html.Should().Contain("18 Jun");   // 24 × 0.2 = 4.8 mm
        html.Should().Contain("4.8 mm");
        html.Should().Contain("19 Jun");   // 24 × 0.1 = 2.4 mm
        html.Should().Contain("2.4 mm");
    }

    [Fact]
    public void Village9to9_drops_a_part_covered_window()
    {
        // Only 5 of 24 hours present — below MinHoursForDay, so the day is hidden
        // rather than reading as a falsely dry total.
        var nineZ = new DateTime(2026, 6, 20, 9, 0, 0, DateTimeKind.Utc);
        var rows = Enumerable.Range(0, 5).Select(h => Row(nineZ.AddHours(h), 0.5)).ToList();

        var html = SitePages.RenderVillage9to9DailyTable(rows, lead: 24);

        html.Should().BeEmpty();
    }

    [Fact]
    public void Village9to9_flags_a_near_complete_window_with_its_hour_count()
    {
        // 23 of 24 hours (one gap) still shows, but flagged with the count so the
        // reader knows the total is off one hour.
        var nineZ = new DateTime(2026, 6, 21, 9, 0, 0, DateTimeKind.Utc);
        var rows = FullDay(nineZ, 0.1).Take(23).ToList();

        var html = SitePages.RenderVillage9to9DailyTable(rows, lead: 24);

        html.Should().Contain("21 Jun");
        html.Should().Contain("(23h)");
    }
}
