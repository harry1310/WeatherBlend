using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using WeatherBlend.Config;
using WeatherBlend.Predict;
using Xunit;

namespace WeatherBlend.Tests;

public class PredictLocationResolverTests
{
    private static AppConfig TwoLocationConfig() => new()
    {
        Locations =
        {
            new LocationConfig { Name = "bonehill_rocks" },
            new LocationConfig { Name = "membury_devon" },
        },
    };

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Null_or_blank_override_resolves_to_primary_location(string? overrideValue)
    {
        var cfg = TwoLocationConfig();

        var (location, exitCode) = PredictLocationResolver.Resolve(cfg, overrideValue, NullLogger.Instance);

        exitCode.Should().Be(0);
        location.Should().BeSameAs(cfg.Location);
        location!.Name.Should().Be("bonehill_rocks");
    }

    [Theory]
    [InlineData("membury_devon")]
    [InlineData("MEMBURY_DEVON")]
    [InlineData("Membury_Devon")]
    public void Matching_override_resolves_case_insensitively(string overrideValue)
    {
        var cfg = TwoLocationConfig();

        var (location, exitCode) = PredictLocationResolver.Resolve(cfg, overrideValue, NullLogger.Instance);

        exitCode.Should().Be(0);
        location.Should().NotBeNull();
        location!.Name.Should().Be("membury_devon");
    }

    [Fact]
    public void Unknown_override_returns_null_and_exit_code_2()
    {
        var cfg = TwoLocationConfig();

        var (location, exitCode) = PredictLocationResolver.Resolve(cfg, "atlantis", NullLogger.Instance);

        location.Should().BeNull();
        exitCode.Should().Be(2);
    }
}
