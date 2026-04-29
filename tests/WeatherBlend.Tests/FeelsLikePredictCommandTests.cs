using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin <see cref="FeelsLikePredictCommand.MergeRows"/>. The dedup key has to
/// include <c>ValidTimeUtc</c> alongside <c>(PredictionMadeAtUtc, LeadHours)</c>,
/// because each feels-like run produces one row per (Lead, ValidTime) pair
/// found in the joined inputs — and now that the 2h predict cadence puts
/// multiple anchors per UTC date on disk, a single run typically emits
/// several ValidTimes per Lead under one shared PredictionMadeAtUtc. Keying
/// on Lead alone collapsed those siblings, leaving the home-card feels-like
/// chip stripped from every card whose anchor wasn't the surviving one.
/// </summary>
public class FeelsLikePredictCommandTests
{
    private static FeelsLikePredictionRow Row(DateTime madeAt, int leadHours, DateTime validTime, double tempC)
        => new()
        {
            LocationName = "Bonehill Rocks",
            ModelVersion = "v1",
            PredictionMadeAtUtc = madeAt,
            LeadHours = leadHours,
            ValidTimeUtc = validTime,
            TemperatureC = tempC,
            RelativeHumidityPct = 80,
            WindSpeed10mMs = 4,
            WindSpeed1m1Ms = 2.6,
            ShortwaveDownWm2 = 200,
            CloudCoverPct = 50,
            VapourPressureHpa = 8,
            LongwaveDownWm2 = 320,
            LongwaveUpWm2 = 380,
            MeanRadiantTemperatureC = tempC + 1,
            UtciC = tempC - 2,
            Band = "NoStress",
            ApparentTemperatureC = tempC - 1,
            TempModelVersion = "tv",
            HumidityModelVersion = "hv",
            WindModelVersion = "wv",
            RadiationModelVersion = "rv",
            CloudModelVersion = "cv",
        };

    [Fact]
    public void MergeRows_keeps_every_ValidTime_per_Lead_within_a_single_run()
    {
        // Single run @ 10:35 emits 6 rows (3 leads × 2 ValidTimes — one per
        // anchor on disk). Pre-fix dedup keyed on (PMT, Lead) would collapse
        // each Lead's pair to one survivor → 3 rows. The fix keeps all 6.
        var made = new DateTime(2026, 4, 29, 10, 35, 0, DateTimeKind.Utc);
        var v0500 = new DateTime(2026, 4, 30, 5, 0, 0, DateTimeKind.Utc);
        var v1000 = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var incoming = new[]
        {
            Row(made, 24, v0500,             tempC: 9.7),
            Row(made, 24, v1000,             tempC: 12.0),
            Row(made, 48, v0500.AddDays(1),  tempC: 8.4),
            Row(made, 48, v1000.AddDays(1),  tempC: 13.1),
            Row(made, 72, v0500.AddDays(2),  tempC: 7.9),
            Row(made, 72, v1000.AddDays(2),  tempC: 11.9),
        };

        var merged = FeelsLikePredictCommand.MergeRows(
            existing: Array.Empty<FeelsLikePredictionRow>(),
            incoming: incoming);

        merged.Should().HaveCount(6);
        merged.Select(r => (r.LeadHours, r.ValidTimeUtc))
            .Should().BeEquivalentTo(incoming.Select(r => (r.LeadHours, r.ValidTimeUtc)));
    }

    [Fact]
    public void MergeRows_appends_new_run_to_existing_without_dropping_prior_rows()
    {
        // Three runs that share LeadHours but each target a different anchor
        // (different ValidTime). All three runs' rows should survive the merge —
        // this is the on-disk shape of a healthy date= partition.
        var v = new DateTime(2026, 4, 30, 0, 0, 0, DateTimeKind.Utc);
        var existing = new[]
        {
            Row(new DateTime(2026, 4, 29, 5, 45, 0, DateTimeKind.Utc),  24, v.AddHours(5),  9.7),
            Row(new DateTime(2026, 4, 29, 10, 7, 0, DateTimeKind.Utc),  24, v.AddHours(10), 11.9),
        };
        var incoming = new[]
        {
            Row(new DateTime(2026, 4, 29, 10, 35, 0, DateTimeKind.Utc), 24, v.AddHours(10), 12.0),
        };

        var merged = FeelsLikePredictCommand.MergeRows(existing, incoming);

        merged.Should().HaveCount(3);
        merged.Select(r => r.PredictionMadeAtUtc.Hour).Should().Equal(5, 10, 10);
    }

    [Fact]
    public void MergeRows_dedupes_exact_repeats_within_one_run_keeping_freshest_PredictionMadeAt()
    {
        // A retry that re-emits the same (PMT, Lead, ValidTime) row should not
        // double-count. MaxBy(PredictionMadeAtUtc) is a tiebreaker — when PMTs
        // are identical, GroupBy collapses them and we keep one. The choice of
        // payload between identical-keyed rows is unimportant; that they
        // collapse to one is.
        var made = new DateTime(2026, 4, 29, 10, 35, 0, DateTimeKind.Utc);
        var valid = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, 24, valid, tempC: 12.0),
            Row(made, 24, valid, tempC: 12.0),
        };

        FeelsLikePredictCommand.MergeRows(Array.Empty<FeelsLikePredictionRow>(), rows)
            .Should().ContainSingle();
    }

    [Fact]
    public void MergeRows_orders_by_ValidTime_then_LeadHours()
    {
        // Site / downstream consumers iterate the parquet in chronological
        // order; ordering the merge output here saves a re-sort at every read.
        var made = new DateTime(2026, 4, 29, 10, 35, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, 72, new DateTime(2026, 5, 2, 10, 0, 0, DateTimeKind.Utc), 11.9),
            Row(made, 24, new DateTime(2026, 4, 30, 5, 0, 0, DateTimeKind.Utc),  9.7),
            Row(made, 48, new DateTime(2026, 5, 1, 10, 0, 0, DateTimeKind.Utc), 13.1),
            Row(made, 24, new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc), 12.0),
        };

        var merged = FeelsLikePredictCommand.MergeRows(Array.Empty<FeelsLikePredictionRow>(), rows);

        merged.Select(r => (r.ValidTimeUtc, r.LeadHours)).Should().Equal(
            (new DateTime(2026, 4, 30,  5, 0, 0, DateTimeKind.Utc), 24),
            (new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc), 24),
            (new DateTime(2026, 5,  1, 10, 0, 0, DateTimeKind.Utc), 48),
            (new DateTime(2026, 5,  2, 10, 0, 0, DateTimeKind.Utc), 72));
    }
}
