using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

public class MetOfficeObservationsClientTests
{
    private const string Loc = "bonehill_rocks";
    private const string Geohash = "gcj8ds";
    private const string Area = "Devon";

    private static JsonElement Payload(string json) => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void ParseNearest_picks_first_entry()
    {
        // Shape taken from the Land Observations OpenAPI spec v1.0.
        var json = """
        [
          {"geohash":"gcj8ds","area":"Devon","region":"sw","country":"England","olson_time_zone":"Europe/London"},
          {"geohash":"gcj8dt","area":"Devon","region":"sw","country":"England","olson_time_zone":"Europe/London"}
        ]
        """;

        var near = MetOfficeObservationsClient.ParseNearest(Payload(json));

        near.Should().NotBeNull();
        near!.Geohash.Should().Be("gcj8ds");
        near.Area.Should().Be("Devon");
        near.Region.Should().Be("sw");
        near.Country.Should().Be("England");
    }

    [Fact]
    public void ParseNearest_returns_null_for_empty_array()
    {
        MetOfficeObservationsClient.ParseNearest(Payload("[]")).Should().BeNull();
    }

    [Fact]
    public void Parse_maps_all_documented_fields()
    {
        // Exact shape from the API spec sample (one observation per array element).
        var json = """
        [{
          "datetime":"2022-12-14T11:00:00Z",
          "humidity":70,
          "mslp":991,
          "pressure_tendency":"R",
          "temperature":10,
          "visibility":2000,
          "weather_code":12,
          "wind_direction":"SWW",
          "wind_gust":2,
          "wind_speed":13.05
        }]
        """;

        var rows = MetOfficeObservationsClient.Parse(Payload(json), Loc, Geohash, Area);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.LocationName.Should().Be(Loc);
        r.Geohash.Should().Be(Geohash);
        r.Area.Should().Be(Area);
        r.ObservedTimeUtc.Should().Be(new DateTime(2022, 12, 14, 11, 0, 0, DateTimeKind.Utc));
        r.AirTemperature.Should().Be(10);
        r.Humidity.Should().Be(70);
        r.MeanSeaLevelPressure.Should().Be(991);       // hPa, per spec (contrast: Spot mslp is Pa)
        r.PressureTendency.Should().Be("R");
        r.Visibility.Should().Be(2000);
        r.WeatherCode.Should().Be(12);
        r.WindSpeed10m.Should().Be(13.05);
        r.WindGusts10m.Should().Be(2);
        r.WindDirection10m.Should().Be(247.5);         // "SWW" == WSW == 247.5°
    }

    [Fact]
    public void Parse_decodes_all_16_compass_points()
    {
        var mapping = new Dictionary<string, double>
        {
            ["N"] = 0, ["NNE"] = 22.5, ["NE"] = 45, ["ENE"] = 67.5,
            ["E"] = 90, ["ESE"] = 112.5, ["SE"] = 135, ["SSE"] = 157.5,
            ["S"] = 180, ["SSW"] = 202.5, ["SW"] = 225, ["WSW"] = 247.5,
            ["W"] = 270, ["WNW"] = 292.5, ["NW"] = 315, ["NNW"] = 337.5,
        };

        foreach (var (compass, expected) in mapping)
        {
            MetOfficeObservationsClient.CompassToDegrees(compass)
                .Should().Be(expected, $"compass point {compass} should map to {expected}°");
        }
    }

    [Fact]
    public void Parse_accepts_inverted_compass_variants_used_by_the_api()
    {
        // Spec example uses "SWW" for WSW; the letter-order is occasionally
        // inverted in responses, so accept both forms.
        MetOfficeObservationsClient.CompassToDegrees("SWW").Should().Be(247.5);
        MetOfficeObservationsClient.CompassToDegrees("NWW").Should().Be(292.5);
        MetOfficeObservationsClient.CompassToDegrees("NEE").Should().Be(67.5);
        MetOfficeObservationsClient.CompassToDegrees("SEE").Should().Be(112.5);
    }

    [Fact]
    public void Parse_returns_null_wind_direction_for_unrecognised_strings()
    {
        MetOfficeObservationsClient.CompassToDegrees(null).Should().BeNull();
        MetOfficeObservationsClient.CompassToDegrees("").Should().BeNull();
        MetOfficeObservationsClient.CompassToDegrees("XYZ").Should().BeNull();
    }

    [Fact]
    public void Parse_missing_optional_fields_become_null()
    {
        // Every metric field is nullable in the spec — only datetime is required.
        var json = """[{"datetime":"2026-04-24T11:00:00Z","temperature":8.3}]""";

        var rows = MetOfficeObservationsClient.Parse(Payload(json), Loc, Geohash, Area);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.AirTemperature.Should().Be(8.3);
        r.Humidity.Should().BeNull();
        r.MeanSeaLevelPressure.Should().BeNull();
        r.PressureTendency.Should().BeNull();
        r.Visibility.Should().BeNull();
        r.WeatherCode.Should().BeNull();
        r.WindSpeed10m.Should().BeNull();
        r.WindDirection10m.Should().BeNull();
        r.WindGusts10m.Should().BeNull();
    }

    [Fact]
    public void Parse_skips_entries_without_datetime()
    {
        var json = """
        [
          {"temperature":10},
          {"datetime":"2026-04-24T11:00:00Z","temperature":11}
        ]
        """;

        var rows = MetOfficeObservationsClient.Parse(Payload(json), Loc, Geohash, Area);

        rows.Should().HaveCount(1);
        rows[0].AirTemperature.Should().Be(11);
    }

    [Fact]
    public void Parse_sorts_rows_by_observation_time()
    {
        var json = """
        [
          {"datetime":"2026-04-24T13:00:00Z","temperature":13},
          {"datetime":"2026-04-24T11:00:00Z","temperature":11},
          {"datetime":"2026-04-24T12:00:00Z","temperature":12}
        ]
        """;

        var rows = MetOfficeObservationsClient.Parse(Payload(json), Loc, Geohash, Area);

        rows.Select(r => r.ObservedTimeUtc.Hour).Should().Equal(11, 12, 13);
    }

    [Fact]
    public void Parse_empty_array_returns_empty_list()
    {
        MetOfficeObservationsClient.Parse(Payload("[]"), Loc, Geohash, Area).Should().BeEmpty();
    }
}
