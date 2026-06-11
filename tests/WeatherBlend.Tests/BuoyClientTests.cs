using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

public class BuoyClientTests
{
    private const string Loc = "sennen_cove";
    private const string Slug = "sevenstones_62107";

    [Fact]
    public void ParseRealtime_maps_identifiers_and_dedups_timestamps()
    {
        var json = """
        [
          {"telemetryId":2,"timestamp":"2026-06-11T20:00:00","isForecast":false,
           "results":[{"identifier":"Hm0","value":"1.90"},{"identifier":"TEMP","value":"13.60"},{"identifier":"Tz","value":"7.0"}]},
          {"telemetryId":1,"timestamp":"2026-06-11T20:00:00","isForecast":false,
           "results":[{"identifier":"Hm0","value":"1.85"}]},
          {"telemetryId":3,"timestamp":"2026-06-11T21:00:00","isForecast":true,
           "results":[{"identifier":"Hm0","value":"2.10"}]},
          {"telemetryId":4,"timestamp":"2026-06-11T19:30:00","isForecast":false,
           "results":[{"identifier":"Hm0","value":"1.80"},{"identifier":"Tpeak","value":"12.5"},
                      {"identifier":"W_PDIR","value":"281"},{"identifier":"W_SPR","value":"24"}]}
        ]
        """;

        var rows = BuoyClient.ParseRealtime(JsonDocument.Parse(json).RootElement, Loc, Slug);

        // Forecast record dropped; duplicate 20:00 collapses (last record wins).
        rows.Should().HaveCount(2);

        var r1930 = rows[0];
        r1930.ValidTimeUtc.Should().Be(new DateTime(2026, 6, 11, 19, 30, 0, DateTimeKind.Utc));
        r1930.WaveHeight.Should().Be(1.80);
        r1930.PeakPeriod.Should().Be(12.5);
        r1930.WaveDirection.Should().Be(281);
        r1930.DirectionalSpread.Should().Be(24);

        var r2000 = rows[1];
        r2000.Source.Should().Be(Slug);
        r2000.WaveHeight.Should().Be(1.85);   // later array entry wins
        r2000.SeaSurfaceTemperature.Should().BeNull();
    }

    [Fact]
    public void ParseErddapCsv_skips_units_row_non_qc_and_nan_rows()
    {
        var csv = "PLATFORMCODE,time,VHM0,VHM0_QC\r\n" +
                  ",UTC,m,1\r\n" +
                  "6200107,2024-01-02T12:00:00Z,4.5,1\r\n" +
                  "6200107,2024-01-02T12:00:00Z,NaN,-127\r\n" +
                  "6200107,2024-01-02T13:00:00Z,4.6,0\r\n" +
                  "6200107,2024-01-02T14:00:00Z,4.7,1\r\n";

        var values = BuoyClient.ParseErddapCsv(csv, "VHM0");

        values.Should().HaveCount(2);
        values[new DateTime(2024, 1, 2, 12, 0, 0, DateTimeKind.Utc)].Should().Be(4.5);
        values[new DateTime(2024, 1, 2, 14, 0, 0, DateTimeKind.Utc)].Should().Be(4.7);
    }

    [Fact]
    public void MergeArchive_unions_variables_on_timestamp()
    {
        var t1 = new DateTime(2024, 1, 2, 12, 0, 0, DateTimeKind.Utc);
        var t2 = new DateTime(2024, 1, 2, 13, 0, 0, DateTimeKind.Utc);
        var byVar = new Dictionary<string, Dictionary<DateTime, double>>
        {
            ["VHM0"] = new() { [t1] = 4.5, [t2] = 4.6 },
            ["VTZA"] = new() { [t1] = 8.0 },
            ["VTPK"] = new() { [t2] = 14.3 },
        };

        var rows = BuoyClient.MergeArchive(byVar, Loc, Slug);

        rows.Should().HaveCount(2);
        rows[0].WaveHeight.Should().Be(4.5);
        rows[0].WavePeriod.Should().Be(8.0);
        rows[0].PeakPeriod.Should().BeNull();
        rows[1].PeakPeriod.Should().Be(14.3);
        rows[1].WaveDirection.Should().BeNull();
    }
}
