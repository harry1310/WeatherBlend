using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

public class OpenMeteoClientTests
{
    private const string Loc = "bonehill";
    private const string Model = "icon_seamless";

    private static JsonElement Payload(string json)
        => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void Parse_historical_sets_run_time_to_midnight_of_valid_day_and_lead_to_hour()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00","2025-06-10T06:00","2025-06-10T23:00"],
            "temperature_2m": [10.0, 12.5, 11.0]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, isHistorical: true);

        rows.Should().HaveCount(3);
        foreach (var r in rows)
        {
            r.RunTimeUtc.Should().Be(new DateTime(2025, 6, 10, 0, 0, 0, DateTimeKind.Utc));
            r.RunTimeUtc.Kind.Should().Be(DateTimeKind.Utc);
            r.ValidTimeUtc.Kind.Should().Be(DateTimeKind.Utc);
        }
        rows[0].LeadHours.Should().Be(0);
        rows[1].LeadHours.Should().Be(6);
        rows[2].LeadHours.Should().Be(23);
    }

    [Fact]
    public void Parse_maps_temperature_and_other_columns()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m": [15.3],
            "dew_point_2m": [10.1],
            "relative_humidity_2m": [70],
            "precipitation": [0.2],
            "precipitation_probability": [40],
            "cloud_cover": [80],
            "wind_speed_10m": [5.5],
            "wind_direction_10m": [270],
            "wind_gusts_10m": [9.0],
            "surface_pressure": [1012.5]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, isHistorical: true);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.LocationName.Should().Be(Loc);
        r.Model.Should().Be(Model);
        r.Temperature2m.Should().Be(15.3);
        r.DewPoint2m.Should().Be(10.1);
        r.RelativeHumidity2m.Should().Be(70);
        r.Precipitation.Should().Be(0.2);
        r.PrecipitationProbability.Should().Be(40);
        r.CloudCover.Should().Be(80);
        r.WindSpeed10m.Should().Be(5.5);
        r.WindDirection10m.Should().Be(270);
        r.WindGusts10m.Should().Be(9.0);
        r.SurfacePressure.Should().Be(1012.5);
    }

    [Fact]
    public void Parse_null_entries_become_null_doubles()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00","2025-06-10T01:00"],
            "temperature_2m": [null, 11.0]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, isHistorical: true);

        rows.Should().HaveCount(2);
        rows[0].Temperature2m.Should().BeNull();
        rows[1].Temperature2m.Should().Be(11.0);
    }

    [Fact]
    public void Parse_missing_hourly_returns_empty()
    {
        var json = """{ "latitude": 50.5, "longitude": -3.7 }""";

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, isHistorical: true);

        rows.Should().BeEmpty();
    }

    [Fact]
    public void Parse_missing_variable_column_backfills_nulls_matching_time_length()
    {
        // temperature_2m requested but not returned → Parse fills Col(...) with nulls.
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00","2025-06-10T01:00"],
            "dew_point_2m": [9.0, 9.5]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, isHistorical: true);

        rows.Should().HaveCount(2);
        rows.All(r => r.Temperature2m is null).Should().BeTrue();
        rows[0].DewPoint2m.Should().Be(9.0);
        rows[1].DewPoint2m.Should().Be(9.5);
    }

    [Fact]
    public void Parse_live_uses_wall_clock_run_time_so_lead_varies()
    {
        // Live mode: RunTime is "now floored to hour" — different valid times should
        //   produce monotonically different LeadHours (spacing matches valid-time spacing).
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00","2025-06-10T03:00","2025-06-10T06:00"],
            "temperature_2m": [10.0, 11.0, 12.0]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, isHistorical: false);

        rows.Should().HaveCount(3);
        var r0 = rows[0].RunTimeUtc;
        rows.All(r => r.RunTimeUtc == r0).Should().BeTrue();
        (rows[1].LeadHours - rows[0].LeadHours).Should().Be(3);
        (rows[2].LeadHours - rows[1].LeadHours).Should().Be(3);
    }
}
