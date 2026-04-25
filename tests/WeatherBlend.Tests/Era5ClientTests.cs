using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

public class Era5ClientTests
{
    private const string Loc = "bonehill";

    private static JsonElement Payload(string json)
        => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void Parse_maps_all_columns_including_radiation()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m": [15.3],
            "dew_point_2m": [10.1],
            "relative_humidity_2m": [70],
            "precipitation": [0.0],
            "rain": [0.0],
            "snowfall": [0.0],
            "cloud_cover": [40],
            "cloud_cover_low": [20],
            "cloud_cover_mid": [10],
            "cloud_cover_high": [10],
            "wind_speed_10m": [4.5],
            "wind_direction_10m": [225],
            "wind_gusts_10m": [8.0],
            "surface_pressure": [1011.5],
            "visibility": [25000],
            "shortwave_radiation": [600.5],
            "direct_radiation": [450.0],
            "diffuse_radiation": [150.5]
          }
        }
        """;

        var rows = Era5Client.Parse(Payload(json), Loc);

        rows.Should().HaveCount(1);
        var r = rows.Single();
        r.LocationName.Should().Be(Loc);
        r.Temperature2m.Should().Be(15.3);
        r.DewPoint2m.Should().Be(10.1);
        r.RelativeHumidity2m.Should().Be(70);
        r.CloudCover.Should().Be(40);
        r.WindSpeed10m.Should().Be(4.5);
        r.WindDirection10m.Should().Be(225);
        r.WindGusts10m.Should().Be(8.0);
        r.SurfacePressure.Should().Be(1011.5);
        r.Visibility.Should().Be(25000);
        r.ShortwaveRadiation.Should().Be(600.5);
        r.DirectRadiation.Should().Be(450.0);
        r.DiffuseRadiation.Should().Be(150.5);
    }

    [Fact]
    public void Parse_null_entries_become_null_doubles()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00","2025-06-10T01:00"],
            "temperature_2m": [null, 11.0],
            "shortwave_radiation": [0.0, null]
          }
        }
        """;

        var rows = Era5Client.Parse(Payload(json), Loc);

        rows.Should().HaveCount(2);
        rows[0].Temperature2m.Should().BeNull();
        rows[1].Temperature2m.Should().Be(11.0);
        rows[0].ShortwaveRadiation.Should().Be(0.0);
        rows[1].ShortwaveRadiation.Should().BeNull();
    }

    [Fact]
    public void Parse_missing_hourly_returns_empty()
    {
        var rows = Era5Client.Parse(Payload("""{ "latitude": 50.5 }"""), Loc);
        rows.Should().BeEmpty();
    }

    [Fact]
    public void Parse_missing_radiation_columns_default_to_null()
    {
        // Older ERA5 files (or stripped-down test fixtures) shouldn't fail just
        // because the radiation fields are absent — they should land as null.
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00"],
            "temperature_2m": [10.0]
          }
        }
        """;

        var rows = Era5Client.Parse(Payload(json), Loc);

        rows.Single().ShortwaveRadiation.Should().BeNull();
        rows.Single().DirectRadiation.Should().BeNull();
        rows.Single().DiffuseRadiation.Should().BeNull();
    }
}
