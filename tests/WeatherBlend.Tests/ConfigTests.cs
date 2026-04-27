using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
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
    public void HttpConfig_defaults_previous_runs_backfill_delay_to_15s()
    {
        // Picked to keep us at ~4 calls/min — under the per-hour token bucket
        // that bit the 2026-04-25 previous-runs backfill. Lowering this is the
        // easiest way to reintroduce 429s, so locking the default down with a test.
        var c = new HttpConfig();
        c.PreviousRunsBackfillDelaySeconds.Should().Be(15);
    }

    [Fact]
    public void HttpConfig_previous_runs_backfill_delay_is_configurable()
    {
        var c = new HttpConfig { PreviousRunsBackfillDelaySeconds = 30 };
        c.PreviousRunsBackfillDelaySeconds.Should().Be(30);
    }

    [Fact]
    public void BlendersConfig_get_throws_when_target_or_featureSet_missing()
    {
        var b = new BlendersConfig
        {
            Items = { new BlenderConfig { Target = "temperature", FeatureSet = "lean" } }
        };
        var act = () => b.Get("temperature", "rich");
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*temperature*rich*");
    }

    [Fact]
    public void BlenderConfig_required_and_optional_fall_back_to_default_when_no_override()
    {
        var c = new BlenderConfig
        {
            RequiredModels = new() { "gfs_seamless", "ecmwf_ifs025" },
            OptionalModels = new() { "ukmo_seamless" },
            PerLeadOverrides = new()
            {
                new() { Lead = 120, RequiredModels = new() { "gfs_seamless" }, OptionalModels = new() }
            },
        };
        c.RequiredForLead(24).Should().Equal("gfs_seamless", "ecmwf_ifs025");
        c.OptionalForLead(24).Should().Equal("ukmo_seamless");
        c.RequiredForLead(120).Should().Equal("gfs_seamless");
        c.OptionalForLead(120).Should().BeEmpty();
    }

    [Fact]
    public void BlendersConfig_binds_from_yaml()
    {
        // The actual config.yaml shipped with the project should bind cleanly to
        // BlendersConfig with all 10 expected (target, featureSet) entries. This
        // pins the binding contract so adding a new blender requires updating
        // the test alongside the config — and so a regression in the YAML schema
        // (e.g. a typo in a property name) is caught before training does it.
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new Microsoft.Extensions.Configuration.ConfigurationBuilder()
            .AddYamlFile(configPath, optional: false)
            .Build();
        var bound = new AppConfig();
        cfg.Bind(bound);

        bound.Blenders.Items.Should().HaveCount(10);
        var keys = bound.Blenders.Items.Select(b => $"{b.Target}/{b.FeatureSet}").ToArray();
        keys.Should().Contain(new[]
        {
            "temperature/lean", "temperature/rich",
            "precipitation/lean", "precipitation/rich",
            "dry_window/base", "dry_window/shape",
            "wind/default", "humidity/default", "cloud/default", "radiation/default",
        });

        // Lean temp: 5 strict + AIFS optional. MF dropped at 120h, AIFS still optional.
        var leanTemp = bound.Blenders.Get("temperature", "lean");
        leanTemp.RequiredModels.Should().HaveCount(5);
        leanTemp.OptionalModels.Should().Equal("ecmwf_aifs025_single");
        leanTemp.RequiredForLead(120).Should().HaveCount(4);
        leanTemp.RequiredForLead(120).Should().NotContain("meteofrance_seamless");
        leanTemp.OptionalForLead(120).Should().Equal("ecmwf_aifs025_single");

        // Lean precip: nothing required, all 6 optional (5 NWPs + AIFS, COALESCE-any).
        var leanPrecip = bound.Blenders.Get("precipitation", "lean");
        leanPrecip.RequiredModels.Should().BeEmpty();
        leanPrecip.OptionalModels.Should().HaveCount(6);
        leanPrecip.OptionalForLead(120).Should().HaveCount(5);

        // Wind: 4 strict + UKMO optional + AIFS optional, MF excluded entirely.
        var wind = bound.Blenders.Get("wind", "default");
        wind.RequiredModels.Should().HaveCount(4);
        wind.OptionalModels.Should().Equal("ukmo_seamless", "ecmwf_aifs025_single");
        wind.RequiredModels.Should().NotContain("meteofrance_seamless");
        wind.OptionalModels.Should().NotContain("meteofrance_seamless");
    }
}
