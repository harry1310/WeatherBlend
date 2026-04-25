using FluentAssertions;
using WeatherBlend.Config;
using Xunit;

namespace WeatherBlend.Tests;

public class ConfigTests
{
    [Fact]
    public void AppConfig_default_has_empty_collections()
    {
        var c = new AppConfig();
        c.Models.Should().BeEmpty();
        c.Variables.Forecast.Should().BeEmpty();
        c.Variables.Era5.Should().BeEmpty();
        c.LeadHours.Should().BeEmpty();
    }

    [Fact]
    public void VariablesConfig_holds_forecast_and_era5_lists_separately()
    {
        var v = new VariablesConfig
        {
            Forecast = new() { "temperature_2m", "precipitation" },
            Era5 = new() { "temperature_2m", "shortwave_radiation" },
        };
        v.Forecast.Should().Contain("precipitation");
        v.Era5.Should().Contain("shortwave_radiation");
        v.Forecast.Should().NotContain("shortwave_radiation");
        v.Era5.Should().NotContain("precipitation");
    }

    [Fact]
    public void LocationConfig_populates()
    {
        var loc = new LocationConfig
        {
            Name = "bonehill_rocks",
            Latitude = 50.5831,
            Longitude = -3.7931,
            ElevationMeters = 393
        };
        loc.Latitude.Should().BeApproximately(50.5831, 0.0001);
        loc.ElevationMeters.Should().Be(393);
    }

    [Fact]
    public void HttpConfig_defaults_open_meteo_backfill_delay_to_15s()
    {
        // Picked to keep us at ~4 calls/min — under the per-hour token bucket
        // that bit the 2026-04-25 backfill. Lowering this is the easiest way
        // to reintroduce 429s, so locking the default down with a test.
        var c = new HttpConfig();
        c.OpenMeteoBackfillDelaySeconds.Should().Be(15);
    }

    [Fact]
    public void HttpConfig_open_meteo_backfill_delay_is_configurable()
    {
        var c = new HttpConfig { OpenMeteoBackfillDelaySeconds = 30 };
        c.OpenMeteoBackfillDelaySeconds.Should().Be(30);
    }
}
