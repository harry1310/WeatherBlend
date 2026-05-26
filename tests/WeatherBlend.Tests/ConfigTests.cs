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
    public void AppConfig_Location_returns_first_of_Locations_for_back_compat()
    {
        // The 29+ existing call sites use `_cfg.Location.X` — keeping
        // Location as a back-compat accessor over Locations[0] keeps
        // them all working after the 2026-05-11 multi-location refactor.
        var cfg = new AppConfig();
        cfg.Locations.Add(new LocationConfig { Name = "first",  Latitude = 1.0 });
        cfg.Locations.Add(new LocationConfig { Name = "second", Latitude = 2.0 });

        cfg.Location.Name.Should().Be("first");
        cfg.Location.Latitude.Should().Be(1.0);
        cfg.Locations.Should().HaveCount(2);
    }

    [Fact]
    public void AppConfig_Location_returns_empty_when_no_Locations_configured()
    {
        // Defensive: code paths that read Location on a fresh AppConfig
        // shouldn't NRE. Returns an empty LocationConfig instead.
        var cfg = new AppConfig();
        cfg.Locations.Should().BeEmpty();
        cfg.Location.Should().NotBeNull();
        cfg.Location.Name.Should().BeEmpty();
    }

    [Fact]
    public void AppConfig_binds_multiple_Locations_from_yaml()
    {
        // The shipped config.yaml has both Bonehill (primary, rainfall + METAR)
        // and Membury (rainfall + METAR — added 2026-05-14 alongside the
        // temperature rollout). Binding contract: Locations[0] = Bonehill,
        // Locations[1] = Membury. Both METAR blocks point at the same ICAOs
        // (EGTE primary, EGDY fallback) because they're the closest reliable
        // airports to both Dartmoor and East Devon — the read-side filter is
        // per-ICAO so the duplicate mapping isn't a storage concern.
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new Microsoft.Extensions.Configuration.ConfigurationBuilder()
            .AddYamlFile(configPath, optional: false)
            .Build();
        var bound = new AppConfig();
        cfg.Bind(bound);

        bound.Locations.Should().HaveCount(2);
        bound.Locations[0].Name.Should().Be("bonehill_rocks");
        // Bonehill rainfall: Bellever + Bovey + Hexworthy + Princetown
        // (Princetown re-added 2026-05-25 in anticipation of Phase 3o's
        // 4-station pool — per the cleanup plan its history is back-
        // filled to 2022-01-01, matching the other three. Princetown is
        // 3o-only — the retrain-blenders.yml 3a step explicitly filters
        // it out via `grep -v "^Princetown$"`).
        bound.Locations[0].Rainfall.Stations.Should().HaveCount(4);
        bound.Locations[0].Rainfall.Stations.Select(s => s.Name).Should()
            .Contain("Bellever Dartmoor")
            .And.Contain("Bovey Tracey")
            .And.Contain("Dartmoor nr Hexworthy")
            .And.Contain("Princetown");
        bound.Locations[0].Metar.Primary.Should().Be("EGTE");
        bound.Locations[0].Metar.Fallback.Should().Be("EGDY");
        bound.Locations[1].Name.Should().Be("membury_devon");
        bound.Locations[1].Rainfall.Stations.Should().HaveCount(3);
        bound.Locations[1].Rainfall.Stations.Select(s => s.Name).Should()
            .Contain("Chards Snowdon Hill")
            .And.Contain("Goren")
            .And.Contain("Raymonds Hill");
        bound.Locations[1].Metar.Primary.Should().Be("EGTE");
        bound.Locations[1].Metar.Fallback.Should().Be("EGDY");
        // Back-compat accessor still points to the primary.
        bound.Location.Name.Should().Be("bonehill_rocks");
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

        // Lean precip: nothing required, 7 optional (5 NWPs + AIFS + JMA, COALESCE-any).
        var leanPrecip = bound.Blenders.Get("precipitation", "lean");
        leanPrecip.RequiredModels.Should().BeEmpty();
        leanPrecip.OptionalModels.Should().HaveCount(7);
        leanPrecip.OptionalForLead(120).Should().HaveCount(6);

        // Wind: 4 strict + UKMO/AIFS optional, MF excluded entirely.
        var wind = bound.Blenders.Get("wind", "default");
        wind.RequiredModels.Should().HaveCount(4);
        wind.OptionalModels.Should().Equal("ukmo_seamless", "ecmwf_aifs025_single");
        wind.RequiredModels.Should().NotContain("meteofrance_seamless");
        wind.OptionalModels.Should().NotContain("meteofrance_seamless");
    }
}
