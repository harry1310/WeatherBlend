using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using Xunit;
// BuildHourlyQ tests retired 2026-05-04 alongside the analytical-product
// derivation; the 3g-MC predict path reads through DryWindow3gPredictor's
// helpers, which have their own dedicated tests.

namespace WeatherBlend.Tests;

/// <summary>
/// Pin the small leaf helpers inside <see cref="StartHourPredictCommand"/>.
/// Heavier integration coverage (RunAsync end-to-end with parquet fixtures)
/// can come later once the curve is rendering on the live site; for now,
/// what matters is that the per-row identity dedup is right and the
/// composite-key parser doesn't silently swallow shape errors.
/// </summary>
public class StartHourPredictCommandTests
{
    // ---- ParseDryComposite -------------------------------------------------

    [Theory]
    [InlineData("ea_bellever_dartmoor/window_3h",  "ea_bellever_dartmoor", 3)]
    [InlineData("ea_bellever_dartmoor/window_6h",  "ea_bellever_dartmoor", 6)]
    [InlineData("ea_bovey_tracey/window_4h",        "ea_bovey_tracey",      4)]
    [InlineData("ea_dartmoor_nr_hexworthy/window_6h", "ea_dartmoor_nr_hexworthy", 6)]
    public void ParseDryComposite_extracts_station_and_hours(string key, string slug, int hours)
    {
        var (station, w) = StartHourPredictCommand.ParseDryComposite(key);
        station.Should().Be(slug);
        w.Should().Be(hours);
    }

    [Theory]
    [InlineData("")]                                   // empty
    [InlineData("ea_bellever_dartmoor")]                // no /window_ suffix
    [InlineData("/window_6h")]                          // empty station
    [InlineData("ea_bellever/window_3")]                // missing 'h'
    [InlineData("ea_bellever/window_h")]                // missing digits
    [InlineData("ea_bellever/window_-3h")]              // negative hours
    [InlineData("ea_bellever/wind_3h")]                 // wrong prefix
    public void ParseDryComposite_rejects_malformed(string key)
    {
        var (station, _) = StartHourPredictCommand.ParseDryComposite(key);
        station.Should().BeNull();
    }

    // ---- MergeRows ---------------------------------------------------------

    private static StartHourPredictionRow Row(DateTime made, DateTime targetDate,
        int leadHours, int startHourUtc, double cal)
        => new()
        {
            LocationName = "bonehill_rocks",
            TruthStation = "ea_bellever_dartmoor",
            WindowHours = 6,
            ModelVersion = "v1",
            PredictionMadeAtUtc = made,
            TargetDateUtc = targetDate,
            LeadHours = leadHours,
            StartHourUtc = startHourUtc,
            RawProduct = 1.0,
            ConditionalProb = 0.25,
            CalibratedProb = cal,
            DailyProbAnyBlock = 0.95,
            PrecipVersion = "vp",
            DryWindowVersion = "vd",
        };

    [Fact]
    public void MergeRows_keeps_every_StartHour_per_PMT_run()
    {
        // One cycle emits the full curve (4 starts at lead 24h on a 6-hour
        // window in BST). The dedup key must include StartHourUtc — keying on
        // (PMT, lead) alone would collapse the row vector to a single survivor,
        // exactly the bug the precip / feels-like writers had.
        var made = new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc);
        var target = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, target, 24, 8,  0.30),
            Row(made, target, 24, 9,  0.25),
            Row(made, target, 24, 10, 0.22),
            Row(made, target, 24, 11, 0.18),
        };

        var merged = StartHourPredictCommand.MergeRows(
            existing: Array.Empty<StartHourPredictionRow>(), incoming: rows);

        merged.Should().HaveCount(4);
        merged.Select(r => r.StartHourUtc).Should().Equal(8, 9, 10, 11);
    }

    [Fact]
    public void MergeRows_appends_new_run_to_existing_without_dropping_prior_PMT()
    {
        // Prior cycle's full curve + new cycle's full curve coexist; same
        // (TargetDate, Lead, StartHour) cell but different PMTs → both kept.
        var t = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var existing = new[]
        {
            Row(new DateTime(2026, 4, 30, 5, 45, 0, DateTimeKind.Utc), t, 24, 8, 0.20),
            Row(new DateTime(2026, 4, 30, 5, 45, 0, DateTimeKind.Utc), t, 24, 9, 0.20),
        };
        var incoming = new[]
        {
            Row(new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc), t, 24, 8, 0.40),
            Row(new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc), t, 24, 9, 0.30),
        };

        var merged = StartHourPredictCommand.MergeRows(existing, incoming);
        merged.Should().HaveCount(4);
        // Sort key is (TargetDate, Lead, StartHour) — PMT is *not* part of the
        // ordering, so within a (start-hour) group the older + newer rows
        // alternate as they appear under each start.
        merged.Select(r => (r.StartHourUtc, r.PredictionMadeAtUtc.Hour)).Should().Equal(
            (8, 5), (8, 10), (9, 5), (9, 10));
    }

    [Fact]
    public void MergeRows_orders_by_TargetDate_then_Lead_then_StartHour()
    {
        // Downstream consumers (site renderer, verify) iterate chronologically
        // then by lead bucket. Keep the parquet pre-ordered so they don't have
        // to re-sort.
        var made = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var d1 = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var d2 = new DateTime(2026, 5, 2, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, d2, 48, 9, 0.2),
            Row(made, d1, 24, 11, 0.1),
            Row(made, d1, 24, 8, 0.3),
            Row(made, d1, 24, 9, 0.25),
        };

        var merged = StartHourPredictCommand.MergeRows(Array.Empty<StartHourPredictionRow>(), rows);

        merged.Select(r => (r.TargetDateUtc, r.LeadHours, r.StartHourUtc)).Should().Equal(
            (d1, 24, 8),
            (d1, 24, 9),
            (d1, 24, 11),
            (d2, 48, 9));
    }
}
