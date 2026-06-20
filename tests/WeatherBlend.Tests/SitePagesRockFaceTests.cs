using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Face-aware rock-surface rendering (SENNEN_ROCK_TEMP_PLAN.md S5): the
/// overview tile badge collapses multi-face hours to the BEST (driest,
/// furthest-from-dew) face after per-face lead dedup — the climbable aspect —
/// and the chart helpers give each face a stable label/colour distinct from
/// the shared dew/air lines.
/// </summary>
public class SitePagesRockFaceTests
{
    private static SitePages.RockSurfaceForecastPoint Row(
        string face, double marginC, int leadHours = 24, DateTime? valid = null) => new(
        Version: "v1",
        PredictedAtUtc: new DateTime(2026, 6, 13, 6, 0, 0, DateTimeKind.Utc),
        ValidTimeUtc: valid ?? new DateTime(2026, 6, 14, 12, 0, 0, DateTimeKind.Utc),
        LeadHours: leadHours,
        RockSurfaceTempC: 10 + marginC,
        AirTempC: 11,
        DewPointC: 10,
        CondensationMarginC: marginC,
        GreasinessStatus: marginC <= 0 ? "condensation" : marginC <= 3 ? "potentially_greasy" : "dry",
        LocationName: "sennen_cove",
        Face: face);

    [Fact]
    public void Best_face_wins_the_tile_badge()
    {
        var rows = new[] { Row("west", 4.0), Row("southwest", 2.5), Row("south", -0.5) };

        var byValid = SitePages.CollapseRockToBestFace(rows);

        byValid.Should().HaveCount(1);
        byValid.Values.Single().Face.Should().Be("west", "the driest, furthest-from-dew wall is the climbable aspect");
        byValid.Values.Single().GreasinessStatus.Should().Be("dry");
    }

    [Fact]
    public void Stale_lead_rows_are_deduped_per_face_before_the_best_face_pick()
    {
        // West's lead-48 row (stale cycle) shows margin −1, but its fresh
        // lead-24 row is dry at +5 — the badge must reflect fresh data, so the
        // best face is west at +5, not the stale −1.
        var rows = new[]
        {
            Row("west", -1.0, leadHours: 48),
            Row("west", 5.0, leadHours: 24),
            Row("southwest", 2.0, leadHours: 24),
        };

        var best = SitePages.CollapseRockToBestFace(rows).Values.Single();

        best.Face.Should().Be("west");
        best.CondensationMarginC.Should().Be(5.0);
    }

    [Fact]
    public void Single_empty_face_reduces_to_one_row_per_valid()
    {
        var rows = new[] { Row("", 1.5, leadHours: 48), Row("", 2.5, leadHours: 24) };

        var byValid = SitePages.CollapseRockToBestFace(rows);

        byValid.Should().HaveCount(1);
        byValid.Values.Single().LeadHours.Should().Be(24, "whole-crag mode keeps the smallest-lead-per-valid rule");
    }

    [Fact]
    public void Face_labels_and_colours_are_stable_and_distinct()
    {
        SitePages.RockFaceLabel("").Should().Be("Rock surface");
        SitePages.RockFaceLabel("west").Should().Be("Rock — west face");

        var faceColours = new[] { "west", "southwest", "south" }.Select(SitePages.RockFaceColor).ToList();
        faceColours.Should().OnlyHaveUniqueItems();
        faceColours.Should().NotContain(new[] { "#2e7d32", "#1565c0" }, "dew point and air keep their colours");
    }
}
