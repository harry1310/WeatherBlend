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
            "surface_pressure": [1012.5],
            "shortwave_radiation": [420.1],
            "direct_radiation": [310.5],
            "diffuse_radiation": [109.6]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);

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
        r.ShortwaveRadiation.Should().Be(420.1);
        r.DirectRadiation.Should().Be(310.5);
        r.DiffuseRadiation.Should().Be(109.6);
    }

    [Fact]
    public void Parse_radiation_columns_default_to_null_when_absent()
    {
        // Older payloads (or model+date combinations that don't carry radiation)
        // should leave the new fields null rather than fail to deserialise.
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00"],
            "temperature_2m": [10.0]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);

        rows.Single().ShortwaveRadiation.Should().BeNull();
        rows.Single().DirectRadiation.Should().BeNull();
        rows.Single().DiffuseRadiation.Should().BeNull();
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

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);

        rows.Should().HaveCount(2);
        rows[0].Temperature2m.Should().BeNull();
        rows[1].Temperature2m.Should().Be(11.0);
    }

    [Fact]
    public void Parse_missing_hourly_returns_empty()
    {
        var json = """{ "latitude": 50.5, "longitude": -3.7 }""";

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);

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

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);

        rows.Should().HaveCount(2);
        rows.All(r => r.Temperature2m is null).Should().BeTrue();
        rows[0].DewPoint2m.Should().Be(9.0);
        rows[1].DewPoint2m.Should().Be(9.5);
    }

    [Fact]
    public void Parse_live_without_reported_time_tags_synthesised()
    {
        var json = """{ "hourly": { "time": ["2025-06-10T06:00"], "temperature_2m": [12.0] } }""";
        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);
        rows.Single().RunTimeSource.Should().Be(WeatherBlend.Models.RunTimeSources.Synthesised);
    }

    [Fact]
    public void Parse_live_with_reported_time_sets_run_time_and_tags_reported()
    {
        var reported = new DateTime(2025, 6, 10, 0, 0, 0, DateTimeKind.Utc);
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T01:00","2025-06-10T06:00","2025-06-11T00:00"],
            "temperature_2m": [10.0, 11.0, 12.0]
          }
        }
        """;

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model, reportedRunTime: reported);

        rows.Should().HaveCount(3);
        foreach (var r in rows)
        {
            r.RunTimeUtc.Should().Be(reported);
            r.RunTimeSource.Should().Be(WeatherBlend.Models.RunTimeSources.Reported);
        }
        rows[0].LeadHours.Should().Be(1);
        rows[1].LeadHours.Should().Be(6);
        rows[2].LeadHours.Should().Be(24);
    }

    [Fact]
    public void ParsePreviousRuns_emits_one_row_per_offset_with_lead_24N_and_runtime_back_shift()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m_previous_day1": [15.0],
            "temperature_2m_previous_day2": [14.5],
            "temperature_2m_previous_day3": [14.0],
            "temperature_2m_previous_day4": [13.5],
            "temperature_2m_previous_day5": [13.0],
            "temperature_2m_previous_day6": [12.5],
            "temperature_2m_previous_day7": [12.0]
          }
        }
        """;

        var rows = OpenMeteoClient.ParsePreviousRuns(
            Payload(json), Loc, Model, new[] { "temperature_2m" });

        rows.Should().HaveCount(7);
        var valid = new DateTime(2025, 6, 10, 12, 0, 0, DateTimeKind.Utc);

        for (int n = 1; n <= 7; n++)
        {
            var r = rows[n - 1];
            r.ValidTimeUtc.Should().Be(valid);
            r.LeadHours.Should().Be(24 * n);
            r.RunTimeUtc.Should().Be(valid.AddHours(-24 * n));
            r.RunTimeSource.Should().Be(WeatherBlend.Models.RunTimeSources.OffsetDay);
            r.Model.Should().Be(Model);
            r.LocationName.Should().Be(Loc);
            r.Temperature2m.Should().Be(15.5 - 0.5 * n);
        }
    }

    [Fact]
    public void ParsePreviousRuns_drops_rows_where_all_variables_are_null_for_that_offset()
    {
        // previous_day3 is all-null for this hour → the offset=3 row for that hour is dropped.
        // Other offsets still emit.
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m_previous_day1": [15.0],
            "temperature_2m_previous_day2": [14.5],
            "temperature_2m_previous_day3": [null],
            "temperature_2m_previous_day4": [13.5],
            "temperature_2m_previous_day5": [13.0],
            "temperature_2m_previous_day6": [12.5],
            "temperature_2m_previous_day7": [12.0]
          }
        }
        """;

        var rows = OpenMeteoClient.ParsePreviousRuns(
            Payload(json), Loc, Model, new[] { "temperature_2m" });

        rows.Should().HaveCount(6);
        rows.Select(r => r.LeadHours).Should().NotContain(72);
    }

    [Fact]
    public void ParsePreviousRuns_maps_multiple_variables_per_offset()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T00:00"],
            "temperature_2m_previous_day1": [10.0],
            "temperature_2m_previous_day2": [null],
            "temperature_2m_previous_day3": [null],
            "temperature_2m_previous_day4": [null],
            "temperature_2m_previous_day5": [null],
            "temperature_2m_previous_day6": [null],
            "temperature_2m_previous_day7": [null],
            "precipitation_previous_day1": [0.5],
            "precipitation_previous_day2": [null],
            "precipitation_previous_day3": [null],
            "precipitation_previous_day4": [null],
            "precipitation_previous_day5": [null],
            "precipitation_previous_day6": [null],
            "precipitation_previous_day7": [null],
            "wind_speed_10m_previous_day1": [3.2],
            "wind_speed_10m_previous_day2": [null],
            "wind_speed_10m_previous_day3": [null],
            "wind_speed_10m_previous_day4": [null],
            "wind_speed_10m_previous_day5": [null],
            "wind_speed_10m_previous_day6": [null],
            "wind_speed_10m_previous_day7": [null]
          }
        }
        """;

        var rows = OpenMeteoClient.ParsePreviousRuns(
            Payload(json), Loc, Model, new[] { "temperature_2m", "precipitation", "wind_speed_10m" });

        // Only offset=1 survives the all-null filter; six other offsets drop.
        rows.Should().HaveCount(1);
        var r = rows.Single();
        r.LeadHours.Should().Be(24);
        r.Temperature2m.Should().Be(10.0);
        r.Precipitation.Should().Be(0.5);
        r.WindSpeed10m.Should().Be(3.2);
    }

    [Fact]
    public void ParsePreviousRuns_maps_radiation_columns_per_offset()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "shortwave_radiation_previous_day1": [501.0],
            "shortwave_radiation_previous_day2": [null],
            "shortwave_radiation_previous_day3": [null],
            "shortwave_radiation_previous_day4": [null],
            "shortwave_radiation_previous_day5": [null],
            "shortwave_radiation_previous_day6": [null],
            "shortwave_radiation_previous_day7": [null],
            "direct_radiation_previous_day1": [380.0],
            "direct_radiation_previous_day2": [null],
            "direct_radiation_previous_day3": [null],
            "direct_radiation_previous_day4": [null],
            "direct_radiation_previous_day5": [null],
            "direct_radiation_previous_day6": [null],
            "direct_radiation_previous_day7": [null],
            "diffuse_radiation_previous_day1": [121.0],
            "diffuse_radiation_previous_day2": [null],
            "diffuse_radiation_previous_day3": [null],
            "diffuse_radiation_previous_day4": [null],
            "diffuse_radiation_previous_day5": [null],
            "diffuse_radiation_previous_day6": [null],
            "diffuse_radiation_previous_day7": [null]
          }
        }
        """;

        var rows = OpenMeteoClient.ParsePreviousRuns(
            Payload(json), Loc, Model, new[] { "shortwave_radiation", "direct_radiation", "diffuse_radiation" });

        rows.Should().HaveCount(1);
        var r = rows.Single();
        r.LeadHours.Should().Be(24);
        r.ShortwaveRadiation.Should().Be(501.0);
        r.DirectRadiation.Should().Be(380.0);
        r.DiffuseRadiation.Should().Be(121.0);
    }

    [Fact]
    public void ParsePreviousRuns_missing_hourly_returns_empty()
    {
        var rows = OpenMeteoClient.ParsePreviousRuns(
            Payload("""{ "latitude": 50.5 }"""), Loc, Model, new[] { "temperature_2m" });
        rows.Should().BeEmpty();
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

        var rows = OpenMeteoClient.Parse(Payload(json), Loc, Model);

        rows.Should().HaveCount(3);
        var r0 = rows[0].RunTimeUtc;
        rows.All(r => r.RunTimeUtc == r0).Should().BeTrue();
        (rows[1].LeadHours - rows[0].LeadHours).Should().Be(3);
        (rows[2].LeadHours - rows[1].LeadHours).Should().Be(3);
    }

    // ---- Historical-forecast (0h-lead "nowcast" source) ---------------------

    [Fact]
    public void ParseHistoricalForecast_maps_surface_and_pressure_columns_lead0_histforecast()
    {
        // The widened hist-forecast pull carries surface AND pressure fields;
        // every row is lead-0, RunTime == ValidTime, tagged hist_forecast.
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m": [15.3],
            "precipitation": [0.4],
            "relative_humidity_2m": [88],
            "cloud_cover": [75],
            "wind_speed_10m": [6.0],
            "surface_pressure": [1009.0],
            "temperature_850hPa": [8.1],
            "relative_humidity_850hPa": [92]
          }
        }
        """;

        var rows = OpenMeteoClient.ParseHistoricalForecast(Payload(json), Loc, Model);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.RunTimeSource.Should().Be(WeatherBlend.Models.RunTimeSources.HistForecast);
        r.LeadHours.Should().Be(0);
        r.RunTimeUtc.Should().Be(r.ValidTimeUtc);
        r.Temperature2m.Should().Be(15.3);
        r.Precipitation.Should().Be(0.4);
        r.RelativeHumidity2m.Should().Be(88);
        r.CloudCover.Should().Be(75);
        r.WindSpeed10m.Should().Be(6.0);
        r.SurfacePressure.Should().Be(1009.0);
        r.Temperature850hPa.Should().Be(8.1);
        r.RelativeHumidity850hPa.Should().Be(92);
    }

    [Fact]
    public void ParseHistoricalForecast_keeps_surface_only_hour_when_pressure_absent()
    {
        // UKMO/AIFS don't archive upper-air — a surface-only hour must survive
        // on its surface fields alone (the old pressure-only parser dropped it).
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m": [14.0],
            "precipitation": [1.2]
          }
        }
        """;

        var rows = OpenMeteoClient.ParseHistoricalForecast(Payload(json), Loc, Model);

        rows.Should().HaveCount(1);
        rows[0].Precipitation.Should().Be(1.2);
        rows[0].Temperature850hPa.Should().BeNull();
    }

    [Fact]
    public void ParseHistoricalForecast_drops_fully_empty_hour()
    {
        var json = """
        {
          "hourly": {
            "time": ["2025-06-10T12:00"],
            "temperature_2m": [null],
            "precipitation": [null],
            "temperature_850hPa": [null]
          }
        }
        """;

        OpenMeteoClient.ParseHistoricalForecast(Payload(json), Loc, Model)
            .Should().BeEmpty();
    }
}
