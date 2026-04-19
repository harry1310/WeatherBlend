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
        c.Variables.Should().BeEmpty();
        c.LeadHours.Should().BeEmpty();
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
}
