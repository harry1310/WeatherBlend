using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Unit tests for <see cref="SeriesDedup"/> — the shared per-valid-time
/// collapse behind every site chart/table. The tie-break rules are the
/// whole point of the helper (the hand-rolled copies it replaced diverged
/// on exactly these), so each rule gets an explicit case:
///   * LatestPerValid: smallest lead wins; freshest made-at on lead ties.
///   * LatestPerValid (no made-at overload): smallest lead, source order
///     on ties — for sources without a made-at column.
///   * FreshestPerValid: freshest made-at wins; source order on ties
///     (LINQ OrderBy is stable).
/// </summary>
public class SeriesDedupTests
{
    private sealed record Row(DateTime Valid, int Lead, DateTime MadeAt, string Tag);

    private static readonly DateTime V1 = new(2026, 6, 10, 12, 0, 0, DateTimeKind.Utc);
    private static readonly DateTime V2 = new(2026, 6, 10, 13, 0, 0, DateTimeKind.Utc);
    private static readonly DateTime T0 = new(2026, 6, 9, 6, 0, 0, DateTimeKind.Utc);

    [Fact]
    public void LatestPerValid_picks_smallest_lead_per_valid_time()
    {
        var rows = new[]
        {
            new Row(V1, 48, T0, "v1-48"),
            new Row(V1, 24, T0, "v1-24"),
            new Row(V2, 24, T0, "v2-24"),
            new Row(V2, 72, T0, "v2-72"),
        };

        var result = rows.LatestPerValid(r => r.Valid, r => r.Lead, r => r.MadeAt).ToList();

        result.Should().HaveCount(2);
        result.Select(r => r.Tag).Should().Equal("v1-24", "v2-24");
    }

    [Fact]
    public void LatestPerValid_breaks_lead_ties_by_freshest_made_at()
    {
        var rows = new[]
        {
            new Row(V1, 24, T0,            "stale"),
            new Row(V1, 24, T0.AddHours(6), "fresh"),
            new Row(V1, 24, T0.AddHours(3), "middle"),
        };

        var result = rows.LatestPerValid(r => r.Valid, r => r.Lead, r => r.MadeAt).Single();

        result.Tag.Should().Be("fresh");
    }

    [Fact]
    public void LatestPerValid_prefers_smaller_lead_even_when_larger_lead_is_fresher()
    {
        // The lead ordering is primary — a stale shorter-lead row still
        // beats a fresher longer-lead one (shortest lead = most recent
        // NWP cycle covering the hour, which is the product decision).
        var rows = new[]
        {
            new Row(V1, 48, T0.AddHours(12), "fresher-but-far"),
            new Row(V1, 24, T0,              "staler-but-near"),
        };

        var result = rows.LatestPerValid(r => r.Valid, r => r.Lead, r => r.MadeAt).Single();

        result.Tag.Should().Be("staler-but-near");
    }

    [Fact]
    public void LatestPerValid_without_made_at_keeps_source_order_on_lead_ties()
    {
        var rows = new[]
        {
            new Row(V1, 24, T0.AddHours(1), "first-in-source"),
            new Row(V1, 24, T0.AddHours(9), "second-in-source"),
        };

        // No made-at selector → stable sort → first source row wins,
        // mirroring the original Met-Office-Spot call sites exactly.
        var result = rows.LatestPerValid(r => r.Valid, r => r.Lead).Single();

        result.Tag.Should().Be("first-in-source");
    }

    [Fact]
    public void FreshestPerValid_picks_freshest_made_at_per_valid_time()
    {
        var rows = new[]
        {
            new Row(V1, 24, T0,             "v1-old"),
            new Row(V1, 24, T0.AddHours(6), "v1-new"),
            new Row(V2, 24, T0.AddHours(2), "v2-only"),
        };

        var result = rows.FreshestPerValid(r => r.Valid, r => r.MadeAt).ToList();

        result.Select(r => r.Tag).Should().Equal("v1-new", "v2-only");
    }

    [Fact]
    public void FreshestPerValid_ignores_lead_entirely()
    {
        // A fresher long-lead row beats a staler short-lead row — that is
        // the deliberate contrast with LatestPerValid for the per-lead
        // forecast pages (already filtered to one lead) vs cross-lead picks.
        var rows = new[]
        {
            new Row(V1, 12, T0,             "short-lead-stale"),
            new Row(V1, 96, T0.AddHours(1), "long-lead-fresh"),
        };

        var result = rows.FreshestPerValid(r => r.Valid, r => r.MadeAt).Single();

        result.Tag.Should().Be("long-lead-fresh");
    }

    [Fact]
    public void FreshestPerValid_keeps_source_order_on_made_at_ties()
    {
        var rows = new[]
        {
            new Row(V1, 24, T0, "first-in-source"),
            new Row(V1, 48, T0, "second-in-source"),
        };

        var result = rows.FreshestPerValid(r => r.Valid, r => r.MadeAt).Single();

        result.Tag.Should().Be("first-in-source");
    }

    [Fact]
    public void Composite_keys_group_independently()
    {
        // Several call sites collapse on composite keys — e.g. the dry-window
        // tables on (TargetDate, Lead) and the start-hour grid on
        // (WindowHours, StartHour). The generic key must keep those cells
        // separate rather than folding across the second component.
        var rows = new[]
        {
            new Row(V1, 24, T0,             "v1-l24-old"),
            new Row(V1, 24, T0.AddHours(1), "v1-l24-new"),
            new Row(V1, 48, T0.AddHours(9), "v1-l48"),
            new Row(V2, 24, T0,             "v2-l24"),
        };

        var result = rows
            .FreshestPerValid(r => (r.Valid, r.Lead), r => r.MadeAt)
            .Select(r => r.Tag)
            .ToList();

        result.Should().Equal("v1-l24-new", "v1-l48", "v2-l24");
    }

    [Fact]
    public void Groups_are_emitted_in_first_seen_order()
    {
        // Callers re-sort by valid time afterwards, but the helper itself
        // must follow the LINQ GroupBy contract (first-seen group order) so
        // converted call sites that did NOT re-sort behave identically.
        var rows = new[]
        {
            new Row(V2, 24, T0, "v2"),
            new Row(V1, 24, T0, "v1"),
        };

        rows.LatestPerValid(r => r.Valid, r => r.Lead, r => r.MadeAt)
            .Select(r => r.Tag)
            .Should().Equal("v2", "v1");
    }
}
