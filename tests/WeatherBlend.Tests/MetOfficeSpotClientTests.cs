using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

public class MetOfficeSpotClientTests
{
    private const string Loc = "bonehill_rocks";
    private const string Tag = "met_office_spot";

    private static JsonElement Payload(string json)
        => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void Parse_extracts_run_time_and_maps_fields()
    {
        // Real Global Spot hourly response shape, trimmed.
        var json = """
        {
          "type": "FeatureCollection",
          "features": [{
            "type": "Feature",
            "geometry": {"type":"Point","coordinates":[-3.811,50.5745,239.0]},
            "properties": {
              "requestPointDistance": 1585.307,
              "modelRunDate": "2026-04-24T17:00Z",
              "timeSeries": [
                {
                  "time": "2026-04-24T17:00Z",
                  "screenTemperature": 14.5,
                  "screenDewPointTemperature": -0.56,
                  "screenRelativeHumidity": 36.05,
                  "precipitationRate": 0.2,
                  "probOfPrecipitation": 10,
                  "totalSnowAmount": 0,
                  "windSpeed10m": 4.73,
                  "windDirectionFrom10m": 109,
                  "windGustSpeed10m": 8.75,
                  "mslp": 102180,
                  "visibility": 54314
                }
              ]
            }
          }]
        }
        """;

        var rows = MetOfficeSpotClient.Parse(Payload(json), Loc, Tag);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.LocationName.Should().Be(Loc);
        r.Model.Should().Be(Tag);
        r.RunTimeUtc.Should().Be(new DateTime(2026, 4, 24, 17, 0, 0, DateTimeKind.Utc));
        r.ValidTimeUtc.Should().Be(new DateTime(2026, 4, 24, 17, 0, 0, DateTimeKind.Utc));
        r.LeadHours.Should().Be(0);
        r.RunTimeSource.Should().Be(RunTimeSources.Reported);
        r.Temperature2m.Should().Be(14.5);
        r.DewPoint2m.Should().Be(-0.56);
        r.RelativeHumidity2m.Should().Be(36.05);
        r.Precipitation.Should().Be(0.2);
        r.PrecipitationProbability.Should().Be(10);
        r.Snowfall.Should().Be(0);
        r.WindSpeed10m.Should().Be(4.73);
        r.WindDirection10m.Should().Be(109);
        r.WindGusts10m.Should().Be(8.75);
        r.Visibility.Should().Be(54314);
    }

    [Fact]
    public void Parse_leaves_surface_pressure_null_because_mslp_is_not_station_pressure()
    {
        // mslp is mean-sea-level pressure in Pa; for a 393m site that's ~50 hPa
        // higher than station pressure. Writing it into SurfacePressure would
        // introduce a systematic blender bias — see MetOfficeSpotClient docstring.
        var json = """
        {"features":[{"properties":{
          "modelRunDate":"2026-04-24T17:00Z",
          "timeSeries":[{"time":"2026-04-24T17:00Z","mslp":102180}]
        }}]}
        """;

        var rows = MetOfficeSpotClient.Parse(Payload(json), Loc, Tag);

        rows.Should().HaveCount(1);
        rows[0].SurfacePressure.Should().BeNull();
    }

    [Fact]
    public void Parse_fields_missing_on_step_become_null()
    {
        var json = """
        {"features":[{"properties":{
          "modelRunDate":"2026-04-24T17:00Z",
          "timeSeries":[{"time":"2026-04-24T17:00Z","screenTemperature":15.0}]
        }}]}
        """;

        var rows = MetOfficeSpotClient.Parse(Payload(json), Loc, Tag);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.Temperature2m.Should().Be(15.0);
        r.DewPoint2m.Should().BeNull();
        r.Precipitation.Should().BeNull();
        r.WindSpeed10m.Should().BeNull();
        r.Visibility.Should().BeNull();
    }

    [Fact]
    public void Parse_computes_lead_hours_from_modelRunDate()
    {
        var json = """
        {"features":[{"properties":{
          "modelRunDate":"2026-04-24T17:00Z",
          "timeSeries":[
            {"time":"2026-04-24T15:00Z","screenTemperature":12.0},
            {"time":"2026-04-24T17:00Z","screenTemperature":14.5},
            {"time":"2026-04-24T20:00Z","screenTemperature":10.0},
            {"time":"2026-04-25T17:00Z","screenTemperature":12.0}
          ]
        }}]}
        """;

        var rows = MetOfficeSpotClient.Parse(Payload(json), Loc, Tag);

        rows.Select(r => r.LeadHours).Should().Equal(-2, 0, 3, 24);
    }

    [Fact]
    public void Parse_empty_features_returns_empty_list()
    {
        MetOfficeSpotClient.Parse(Payload("""{"features":[]}"""), Loc, Tag)
            .Should().BeEmpty();
        MetOfficeSpotClient.Parse(Payload("""{}"""), Loc, Tag)
            .Should().BeEmpty();
    }

    [Fact]
    public void Parse_missing_modelRunDate_returns_empty()
    {
        // Without a run time we can't compute LeadHours honestly, so we bail
        // rather than fabricate a value.
        var json = """
        {"features":[{"properties":{
          "timeSeries":[{"time":"2026-04-24T17:00Z","screenTemperature":14.5}]
        }}]}
        """;

        MetOfficeSpotClient.Parse(Payload(json), Loc, Tag).Should().BeEmpty();
    }
}
