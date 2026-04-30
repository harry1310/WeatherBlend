using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin <see cref="PrecipPredictCommand.MergeRows"/>. The dedup key is
/// <c>(PredictionMadeAtUtc, LeadHours, ValidTimeUtc)</c>: today every cycle
/// emits one row per (PMT, lead) so adding ValidTime is a no-op for current
/// behaviour, but it's load-bearing the moment the command starts emitting
/// hourly rows per (PMT, lead) (the upcoming target-list extension). Keying
/// on (PMT, lead) alone would silently drop those siblings — same shape of
/// bug we just fixed in feels-like.
/// </summary>
public class PrecipPredictCommandTests
{
    private static PrecipPredictionRow Row(DateTime madeAt, int leadHours, DateTime validTime, double pWet)
        => new()
        {
            LocationName = "bonehill_rocks",
            TruthStation = "ea_bellever_dartmoor",
            ModelVersion = "v1",
            PredictionMadeAtUtc = madeAt,
            ValidTimeUtc = validTime,
            LeadHours = leadHours,
            ProbWet = pWet,
            ClimatologyPWet = 0.5,
            FeatureVectorHash = "abc",
        };

    [Fact]
    public void MergeRows_keeps_every_ValidTime_per_Lead_within_a_single_run()
    {
        // Future hourly extension: one cycle emits N target hours per lead, all
        // sharing one PredictionMadeAtUtc. Pre-fix dedup keyed on (PMT, lead)
        // would collapse them to one survivor; the 3-tuple key keeps them.
        var made = new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc);
        var v00 = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, 24, v00.AddHours(0),  0.10),
            Row(made, 24, v00.AddHours(6),  0.20),
            Row(made, 24, v00.AddHours(12), 0.30),
            Row(made, 24, v00.AddHours(18), 0.40),
            Row(made, 48, v00.AddDays(1).AddHours(0),  0.15),
            Row(made, 48, v00.AddDays(1).AddHours(12), 0.25),
        };

        var merged = PrecipPredictCommand.MergeRows(
            existing: Array.Empty<PrecipPredictionRow>(),
            incoming: rows);

        merged.Should().HaveCount(6);
        merged.Select(r => (r.LeadHours, r.ValidTimeUtc))
            .Should().BeEquivalentTo(rows.Select(r => (r.LeadHours, r.ValidTimeUtc)));
    }

    [Fact]
    public void MergeRows_appends_new_run_to_existing_without_dropping_prior_rows()
    {
        // Multiple cycles writing the same lead but different ValidTimes (or
        // overlapping ValidTimes from different PMTs) all survive. This is
        // already the on-disk shape today: each cycle adds a fresh PMT row at
        // each (lead, ValidTime).
        var v = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var existing = new[]
        {
            Row(new DateTime(2026, 4, 30, 5, 45, 0, DateTimeKind.Utc),  24, v.AddHours(5),  0.10),
            Row(new DateTime(2026, 4, 30, 10, 7, 0, DateTimeKind.Utc),  24, v.AddHours(10), 0.20),
        };
        var incoming = new[]
        {
            Row(new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc), 24, v.AddHours(10), 0.25),
        };

        var merged = PrecipPredictCommand.MergeRows(existing, incoming);

        merged.Should().HaveCount(3);
        merged.Select(r => r.PredictionMadeAtUtc.Hour).Should().Equal(5, 10, 10);
    }

    [Fact]
    public void MergeRows_dedupes_exact_repeats_within_one_run_to_a_single_row()
    {
        // A retry that re-emits the same (PMT, lead, ValidTime) row should
        // collapse to one. Choice of payload between identical-keyed rows is
        // unimportant; that they collapse is.
        var made = new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc);
        var valid = new DateTime(2026, 5, 1, 10, 0, 0, DateTimeKind.Utc);
        var rows = new[] { Row(made, 24, valid, 0.5), Row(made, 24, valid, 0.5) };

        PrecipPredictCommand.MergeRows(Array.Empty<PrecipPredictionRow>(), rows)
            .Should().ContainSingle();
    }

    // ---- BuildHourlyTargets ------------------------------------------------

    [Fact]
    public void BuildHourlyTargets_emits_24_hours_per_lead_at_correct_target_day()
    {
        // anchor 2026-04-30 10:00 UTC, leads {24, 48, 72} ->
        //   24h target day = 2026-05-01, hours 00..23
        //   48h target day = 2026-05-02, hours 00..23
        //   72h target day = 2026-05-03, hours 00..23
        var anchor = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var leads = new[] { 24, 48, 72 };

        var targets = PrecipPredictCommand.BuildHourlyTargets(anchor, leads);

        targets.Should().HaveCount(72);
        targets.Where(t => t.Lead == 24).Select(t => t.Valid)
            .Should().BeEquivalentTo(Enumerable.Range(0, 24)
                .Select(h => new DateTime(2026, 5, 1, h, 0, 0, DateTimeKind.Utc)));
        targets.Where(t => t.Lead == 48).Select(t => t.Valid).First()
            .Should().Be(new DateTime(2026, 5, 2, 0, 0, 0, DateTimeKind.Utc));
        targets.Where(t => t.Lead == 72).Select(t => t.Valid).Last()
            .Should().Be(new DateTime(2026, 5, 3, 23, 0, 0, DateTimeKind.Utc));
    }

    [Fact]
    public void BuildHourlyTargets_uses_anchor_date_at_UTC_midnight_not_anchor_hour()
    {
        // Two anchors on the same UTC day at different hours produce IDENTICAL
        // target days. The lead-bucket filter at SQL time picks the right freshest
        // cycle per row, so per-hour anchor drift doesn't shift the target day —
        // matches the dry-window blender's anchorDate convention.
        var morning = PrecipPredictCommand.BuildHourlyTargets(
            new DateTime(2026, 4, 30, 5, 45, 0, DateTimeKind.Utc), new[] { 24 });
        var evening = PrecipPredictCommand.BuildHourlyTargets(
            new DateTime(2026, 4, 30, 22, 30, 0, DateTimeKind.Utc), new[] { 24 });

        morning.Select(t => t.Valid).Should().Equal(evening.Select(t => t.Valid));
        morning.First().Valid.Should().Be(new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc));
    }

    [Fact]
    public void MergeRows_orders_by_ValidTime_then_LeadHours()
    {
        // Downstream consumers iterate in chronological order; baking the sort
        // into the merge saves a re-sort at every read.
        var made = new DateTime(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, 48, new DateTime(2026, 5, 2, 10, 0, 0, DateTimeKind.Utc), 0.4),
            Row(made, 24, new DateTime(2026, 5, 1, 5, 0, 0, DateTimeKind.Utc),  0.1),
            Row(made, 24, new DateTime(2026, 5, 1, 10, 0, 0, DateTimeKind.Utc), 0.2),
            Row(made, 72, new DateTime(2026, 5, 3, 10, 0, 0, DateTimeKind.Utc), 0.3),
        };

        var merged = PrecipPredictCommand.MergeRows(Array.Empty<PrecipPredictionRow>(), rows);

        merged.Select(r => (r.ValidTimeUtc, r.LeadHours)).Should().Equal(
            (new DateTime(2026, 5, 1,  5, 0, 0, DateTimeKind.Utc), 24),
            (new DateTime(2026, 5, 1, 10, 0, 0, DateTimeKind.Utc), 24),
            (new DateTime(2026, 5, 2, 10, 0, 0, DateTimeKind.Utc), 48),
            (new DateTime(2026, 5, 3, 10, 0, 0, DateTimeKind.Utc), 72));
    }
}
