using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Collect;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

public class MarineClientTests
{
    private const string Loc = "sennen_cove";
    private const string Model = "meteofrance_wave";

    private static JsonElement Payload(string json)
        => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void Parse_maps_wave_components_and_site_extras()
    {
        var json = """
        {
          "hourly": {
            "time": ["2026-06-11T12:00"],
            "wave_height": [1.52],
            "wave_period": [6.35],
            "wave_direction": [276],
            "wind_wave_height": [0.40],
            "wind_wave_period": [3.2],
            "wind_wave_direction": [250],
            "swell_wave_height": [1.40],
            "swell_wave_period": [9.5],
            "swell_wave_direction": [281],
            "secondary_swell_wave_height": [0.30],
            "secondary_swell_wave_period": [12.1],
            "secondary_swell_wave_direction": [200],
            "sea_level_height_msl": [-1.38],
            "sea_surface_temperature": [13.4]
          }
        }
        """;

        var run = new DateTime(2026, 6, 11, 12, 0, 0, DateTimeKind.Utc);
        var rows = MarineClient.Parse(Payload(json), Loc, Model, run, RunTimeSources.Synthesised);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.LocationName.Should().Be(Loc);
        r.Model.Should().Be(Model);
        r.RunTimeUtc.Should().Be(run);
        r.LeadHours.Should().Be(0);
        r.RunTimeSource.Should().Be(RunTimeSources.Synthesised);
        r.WaveHeight.Should().Be(1.52);
        r.WavePeriod.Should().Be(6.35);
        r.WaveDirection.Should().Be(276);
        r.WindWaveHeight.Should().Be(0.40);
        r.SwellWaveHeight.Should().Be(1.40);
        r.SwellWavePeriod.Should().Be(9.5);
        r.SwellWaveDirection.Should().Be(281);
        r.SecondarySwellWaveHeight.Should().Be(0.30);
        r.SeaLevelHeightMsl.Should().Be(-1.38);
        r.SeaSurfaceTemperature.Should().Be(13.4);
    }

    [Fact]
    public void Parse_hindcast_drops_all_null_hours_and_is_lead_unlabelled()
    {
        // Pre-archive hindcast chunks return all-null columns — those hours
        // must vanish (0-row chunk, not scaffold), and surviving rows are
        // lead-unlabelled placeholders (RunTime = ValidTime, LeadHours = 0).
        var json = """
        {
          "hourly": {
            "time": ["2023-01-15T00:00", "2023-01-15T01:00"],
            "wave_height": [null, 4.02],
            "wave_period": [null, 8.1]
          }
        }
        """;

        var rows = MarineClient.Parse(Payload(json), Loc, Model, runTime: null, RunTimeSources.HistForecast);

        rows.Should().HaveCount(1);
        rows[0].ValidTimeUtc.Should().Be(new DateTime(2023, 1, 15, 1, 0, 0, DateTimeKind.Utc));
        rows[0].RunTimeUtc.Should().Be(rows[0].ValidTimeUtc);
        rows[0].LeadHours.Should().Be(0);
        rows[0].RunTimeSource.Should().Be(RunTimeSources.HistForecast);
        rows[0].WaveHeight.Should().Be(4.02);
    }

    [Fact]
    public void ParseOffsetDays_emits_one_row_per_offset_with_offset_day_convention()
    {
        var json = """
        {
          "hourly": {
            "time": ["2026-06-11T12:00"],
            "wave_height_previous_day1": [1.58],
            "wave_height_previous_day2": [1.61],
            "swell_wave_height_previous_day1": [1.30]
          }
        }
        """;

        var rows = MarineClient.ParseOffsetDays(
            Payload(json), Loc, Model, new[] { "wave_height", "swell_wave_height" });

        // Offsets 3..7 have no columns at all → all-null → dropped.
        rows.Should().HaveCount(2);

        var d1 = rows.Single(r => r.LeadHours == 24);
        d1.RunTimeSource.Should().Be(RunTimeSources.OffsetDay);
        d1.RunTimeUtc.Should().Be(new DateTime(2026, 6, 10, 12, 0, 0, DateTimeKind.Utc));
        d1.WaveHeight.Should().Be(1.58);
        d1.SwellWaveHeight.Should().Be(1.30);

        var d2 = rows.Single(r => r.LeadHours == 48);
        d2.WaveHeight.Should().Be(1.61);
        d2.SwellWaveHeight.Should().BeNull();
    }

    [Fact]
    public void ParseWaveTruth_maps_triple_and_drops_unpublished_hours()
    {
        var json = """
        {
          "hourly": {
            "time": ["2020-06-01T00:00", "2020-06-01T01:00"],
            "wave_height": [1.34, null],
            "wave_period": [5.95, null],
            "wave_direction": [112, null]
          }
        }
        """;

        var rows = MarineClient.ParseWaveTruth(Payload(json), Loc);

        rows.Should().HaveCount(1);
        var r = rows[0];
        r.Source.Should().Be(MarineClient.Era5OceanModel);
        r.WaveHeight.Should().Be(1.34);
        r.WavePeriod.Should().Be(5.95);
        r.WaveDirection.Should().Be(112);
        r.PeakPeriod.Should().BeNull();
    }

    [Fact]
    public void Parse_returns_empty_when_hourly_block_missing()
    {
        MarineClient.Parse(Payload("""{"error":true}"""), Loc, Model, null, RunTimeSources.HistForecast)
            .Should().BeEmpty();
        MarineClient.ParseWaveTruth(Payload("""{"error":true}"""), Loc)
            .Should().BeEmpty();
    }
}
