using FluentAssertions;
using WeatherBlend.Predict;
using Xunit;

namespace WeatherBlend.Tests;

public class PredictForecastFiltersTests
{
    [Fact]
    public void LiveCycleAsOf_includes_all_required_predicates()
    {
        var frag = PredictForecastFilters.LiveCycleAsOf(
            locationName: "Bonehill Rocks",
            asOfRunTime: new DateTime(2026, 4, 23, 14, 0, 0, DateTimeKind.Utc),
            earliestValid: new DateTime(2026, 4, 24, 14, 0, 0, DateTimeKind.Utc),
            latestValid:   new DateTime(2026, 4, 26, 14, 0, 0, DateTimeKind.Utc));

        frag.Should().Contain("LocationName = 'Bonehill Rocks'");
        frag.Should().Contain("RunTimeSource IS NULL OR RunTimeSource <> 'offset_day'");
        frag.Should().Contain("RunTimeUtc <= TIMESTAMP '2026-04-23 14:00:00'");
        frag.Should().Contain("ValidTimeUtc BETWEEN TIMESTAMP '2026-04-24 14:00:00'");
        frag.Should().Contain("AND TIMESTAMP '2026-04-26 14:00:00'");
    }

    [Fact]
    public void LiveCycleAsOf_escapes_single_quotes_in_location_name()
    {
        // SQL-injection guard for the only interpolated string literal.
        var frag = PredictForecastFilters.LiveCycleAsOf(
            locationName: "O'Malley's Point",
            asOfRunTime: new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc),
            earliestValid: new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc),
            latestValid:   new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc));

        frag.Should().Contain("'O''Malley''s Point'");
        frag.Should().NotContain("'O'Malley's Point'");
    }
}
