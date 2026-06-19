using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

public class WeatherLinkClientTests
{
    private const string Loc = "sennen_cove";
    private const string Station = "115072";

    private static JsonElement Payload(string json) => JsonDocument.Parse(json).RootElement;

    // Epoch seconds (UTC). 2026-06-09:
    //   11:00 = 1781002800, 11:15 = 1781003700, 12:00 = 1781006400
    private const long Ts1100 = 1781002800;
    private const long Ts1115 = 1781003700;
    private const long Ts1200 = 1781006400;

    /// <summary>
    /// Two ISS (sensor_type 43) records in UTC hour 11 plus one in hour 12.
    /// Temps in °F, winds in mph, rainfall in mm — exactly as WeatherLink serves.
    /// A second sensor (a non-ISS type) is present to prove it's ignored.
    /// </summary>
    private const string SampleJson = """
    {
      "station_id": 115072,
      "sensors": [
        {
          "sensor_type": 37,
          "data": [ { "ts": 1781002800, "bar_sea_level": 1012.3 } ]
        },
        {
          "sensor_type": 43,
          "data": [
            {
              "ts": 1781002800,
              "temp_avg": 50.0, "temp_hi": 53.6, "temp_lo": 48.2,
              "hum_last": 80, "dew_point_last": 41.0,
              "rainfall_mm": 0.2, "rain_rate_hi_mm": 1.0,
              "wind_speed_avg": 10.0, "wind_speed_hi": 20.0,
              "wind_dir_of_prevail": 90
            },
            {
              "ts": 1781003700,
              "temp_avg": 59.0, "temp_hi": 68.0, "temp_lo": 50.0,
              "hum_last": 60, "dew_point_last": 50.0,
              "rainfall_mm": 0.3, "rain_rate_hi_mm": 2.0,
              "wind_speed_avg": 20.0, "wind_speed_hi": 30.0,
              "wind_dir_of_prevail": 180
            },
            {
              "ts": 1781006400,
              "temp_avg": 68.0, "temp_hi": 68.0, "temp_lo": 68.0,
              "hum_last": 55, "dew_point_last": 53.6,
              "rainfall_mm": 0.0, "rain_rate_hi_mm": 0.0,
              "wind_speed_avg": 0.0, "wind_speed_hi": 0.0,
              "wind_dir_of_prevail": null
            }
          ]
        }
      ]
    }
    """;

    [Fact]
    public void Parse_aggregates_two_utc_hours_with_correct_unit_conversions()
    {
        var rows = WeatherLinkClient.Parse(Payload(SampleJson), Loc, Station);

        rows.Should().HaveCount(2);
        rows.Select(r => r.ObservedTimeUtc).Should().BeInAscendingOrder();

        var h11 = rows[0];
        h11.LocationName.Should().Be(Loc);
        h11.StationId.Should().Be(Station);
        h11.ObservedTimeUtc.Should().Be(DateTimeOffset.FromUnixTimeSeconds(Ts1100).UtcDateTime);
        h11.ObservedTimeUtc.Kind.Should().Be(DateTimeKind.Utc);

        // °F → °C, mean of temp_avg {50,59}°F = {10,15}°C → 12.5
        h11.Temperature2m.Should().BeApproximately(12.5, 1e-9);
        // max temp_hi {53.6,68}°F = {12,20}°C → 20
        h11.TemperatureHigh.Should().BeApproximately(20.0, 1e-9);
        // min temp_lo {48.2,50}°F = {9,10}°C → 9
        h11.TemperatureLow.Should().BeApproximately(9.0, 1e-9);
        // mean hum {80,60} → 70
        h11.Humidity.Should().BeApproximately(70.0, 1e-9);
        // mean dew {41,50}°F = {5,10}°C → 7.5
        h11.DewPoint.Should().BeApproximately(7.5, 1e-9);
        // SUM rainfall {0.2,0.3} → 0.5
        h11.RainfallMm.Should().BeApproximately(0.5, 1e-9);
        // max rain rate {1,2} → 2
        h11.RainRateMmHr.Should().BeApproximately(2.0, 1e-9);
        // mean wind mph {10,20} → m/s mean of {4.4704, 8.9408} = 6.7056
        h11.WindSpeed10m.Should().BeApproximately(6.7056, 1e-9);
        // max gust {20,30} mph → 30 * 0.44704 = 13.4112 m/s
        h11.WindGust10m.Should().BeApproximately(13.4112, 1e-9);
        // circular mean of {90,180} → 135
        h11.WindDirection10m.Should().BeApproximately(135.0, 1e-6);

        var h12 = rows[1];
        h12.ObservedTimeUtc.Should().Be(DateTimeOffset.FromUnixTimeSeconds(Ts1200).UtcDateTime);
        h12.Temperature2m.Should().BeApproximately(20.0, 1e-9);   // 68°F
        h12.RainfallMm.Should().BeApproximately(0.0, 1e-9);
        h12.WindSpeed10m.Should().BeApproximately(0.0, 1e-9);
        h12.WindDirection10m.Should().BeNull();                   // only record's dir was null
    }

    [Fact]
    public void Parse_ignores_non_iss_sensors()
    {
        // The sensor_type 37 record (bar_sea_level only, no temp) must not
        // produce a row — only sensor_type 43 carries the weather archive.
        var rows = WeatherLinkClient.Parse(Payload(SampleJson), Loc, Station);
        rows.Should().OnlyContain(r => r.Temperature2m != null);
    }

    [Fact]
    public void AggregateHourly_skips_hours_with_no_valid_temperature()
    {
        // An hour whose only record has a null temp_avg is dropped.
        var records = new[]
        {
            new WeatherLinkClient.IssRecord(
                Ts: DateTimeOffset.FromUnixTimeSeconds(Ts1200).UtcDateTime,
                TempAvgC: null, TempHiC: null, TempLoC: null,
                HumidityPct: 80, DewPointC: null,
                RainfallMm: 0.5, RainRateMmHr: 1.0,
                WindSpeedMs: 3.0, WindGustMs: 5.0, WindDirDeg: 90),
        };

        WeatherLinkClient.AggregateHourly(records, Loc, Station).Should().BeEmpty();
    }

    [Fact]
    public void CircularMean_wraps_across_north()
    {
        // 350° and 10° straddle North — the circular mean is 0°, NOT 180°.
        WeatherLinkClient.CircularMeanDegrees(new double?[] { 350, 10 })
            .Should().BeApproximately(0.0, 1e-6);
    }

    [Fact]
    public void CircularMean_skips_nulls_and_returns_null_when_all_null()
    {
        WeatherLinkClient.CircularMeanDegrees(new double?[] { null, 90, null })
            .Should().BeApproximately(90.0, 1e-6);
        WeatherLinkClient.CircularMeanDegrees(new double?[] { null, null }).Should().BeNull();
    }

    [Fact]
    public void Parse_missing_sensors_returns_empty()
    {
        WeatherLinkClient.Parse(Payload("""{"station_id":1}"""), Loc, Station).Should().BeEmpty();
        WeatherLinkClient.Parse(Payload("[]"), Loc, Station).Should().BeEmpty();
    }

    [Fact]
    public void Parse_null_optional_fields_become_null_but_row_still_emitted_when_temp_present()
    {
        var json = """
        {
          "sensors": [
            { "sensor_type": 43, "data": [
              { "ts": 1781002800, "temp_avg": 50.0 }
            ] }
          ]
        }
        """;

        var rows = WeatherLinkClient.Parse(Payload(json), Loc, Station);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.Temperature2m.Should().BeApproximately(10.0, 1e-9);
        r.TemperatureHigh.Should().BeNull();
        r.Humidity.Should().BeNull();
        r.DewPoint.Should().BeNull();
        r.RainfallMm.Should().BeNull();
        r.RainRateMmHr.Should().BeNull();
        r.WindSpeed10m.Should().BeNull();
        r.WindGust10m.Should().BeNull();
        r.WindDirection10m.Should().BeNull();
    }
}
