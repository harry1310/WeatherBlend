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

        bound.Locations.Should().HaveCount(3);
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
        // Sennen, Cornwall (added 2026-06-05, sea cliff). Data-only for now
        // (tabs: [] keeps it off the site picker); 3 nearest EA gauges; METAR
        // = Culdrose (EGDR) / Newquay (EGHQ).
        bound.Locations[2].Name.Should().Be("sennen_cove");
        bound.Locations[2].Tabs.Should().BeEmpty();
        bound.Locations[2].Metar.Primary.Should().Be("EGDR");
        bound.Locations[2].Rainfall.Stations.Select(s => s.Name).Should()
            .Contain("Trengwainton")
            .And.Contain("St Ives Towednack")
            .And.Contain("St Erth");
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

        // wind_speed_lgb dropped 2026-06-10 — the model moved to Python
        // (Option B CQR cutover); its NWP scope is pinned in
        // WeatherProbabilistic's train_wind_speed_pi.py now.
        bound.Blenders.Items.Should().HaveCount(11);
        var keys = bound.Blenders.Items.Select(b => $"{b.Target}/{b.FeatureSet}").ToArray();
        keys.Should().Contain(new[]
        {
            "temperature/lean", "temperature/rich",
            "precipitation/lean", "precipitation/rich",
            "dry_window/base", "dry_window/shape",
            "wind/default", "humidity/default", "cloud/default", "radiation/default",
            "wind_gust/default",
        });

        // Lean temp (2026-06-01 minimal-required policy): only gfs+ecmwf
        // required; icon/mf/gem/aifs optional; no perLeadOverrides (both
        // required models carry data at every lead incl. 120h). See the
        // TemperatureRequiredModels_stay_minimal guard below for why.
        var leanTemp = bound.Blenders.Get("temperature", "lean");
        leanTemp.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025");
        leanTemp.OptionalModels.Should().Equal(
            "icon_seamless", "meteofrance_seamless", "gem_seamless", "ecmwf_aifs025_single");
        leanTemp.PerLeadOverrides.Should().BeEmpty();
        leanTemp.RequiredForLead(120).Should().Equal("gfs_seamless", "ecmwf_ifs025");

        // Lean precip: nothing required, 7 optional (5 NWPs + AIFS + JMA, COALESCE-any).
        var leanPrecip = bound.Blenders.Get("precipitation", "lean");
        leanPrecip.RequiredModels.Should().BeEmpty();
        leanPrecip.OptionalModels.Should().HaveCount(7);
        leanPrecip.OptionalForLead(120).Should().HaveCount(6);

        // Wind: 3 strict (gfs/ecmwf/icon) + GEM/UKMO/AIFS optional, MF excluded
        // entirely. gem_seamless demoted required→optional 2026-06-05 (GEM-outage
        // robustness — see the ElementBlenders_never_require_gem guard below).
        var wind = bound.Blenders.Get("wind", "default");
        wind.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025", "icon_seamless");
        wind.OptionalModels.Should().Equal("gem_seamless", "ukmo_seamless", "ecmwf_aifs025_single");
        wind.RequiredModels.Should().NotContain("meteofrance_seamless");
        wind.OptionalModels.Should().NotContain("meteofrance_seamless");
    }

    // The two NWPs we trust to publish to 120h with reliable Open-Meteo
    // ingestion. Any model OUTSIDE this set, if made REQUIRED for temperature,
    // can silently truncate long-lead predictions the moment its feed lapses
    // or its horizon falls short — the row-drop gate kills every row missing a
    // required model.
    private static readonly string[] ReliableTo120h = { "gfs_seamless", "ecmwf_ifs025" };

    [Fact]
    public void TemperatureRequiredModels_stay_minimal_so_one_feed_outage_cannot_truncate_leads()
    {
        // Regression guard for the 2026-06-01 incident: Open-Meteo's GEM
        // (cmc_gem_gdps) ingestion stalled (frozen at 2026-05-26). Because
        // gem_seamless was REQUIRED for temperature 2b/2c, the post-pivot
        // "every required model NOT NULL" gate silently dropped EVERY
        // 72/96/120h temperature row site-wide — nothing on the overview or
        // temperature charts beyond +2 days, worsening by a day each day.
        //
        // The fix made the temperature required set minimal (gfs+ecmwf only,
        // both publish to 120h with reliable ingestion); every other model is
        // optional, so a single lapsed/short-horizon feed leaves a NaN slot
        // LightGBM handles natively instead of nuking the row. This test pins
        // that invariant at EVERY lead (base list + per-lead overrides) so a
        // future edit can't quietly reintroduce a fragile required model.
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new Microsoft.Extensions.Configuration.ConfigurationBuilder()
            .AddYamlFile(configPath, optional: false)
            .Build();
        var bound = new AppConfig();
        cfg.Bind(bound);

        int[] leads = { 24, 48, 72, 96, 120 };
        foreach (var featureSet in new[] { "lean", "rich" })
        {
            var blender = bound.Blenders.Get("temperature", featureSet);
            blender.RequiredModels.Should().OnlyContain(
                m => ReliableTo120h.Contains(m),
                $"temperature/{featureSet} base required set must stay within {{gfs,ecmwf}} " +
                "(see the 2026-06-01 GEM-outage incident)");
            foreach (var lead in leads)
            {
                blender.RequiredForLead(lead).Should().OnlyContain(
                    m => ReliableTo120h.Contains(m),
                    $"temperature/{featureSet} required set at lead {lead}h must stay within {{gfs,ecmwf}}");
            }
        }
    }

    [Fact]
    public void ElementBlenders_never_require_gem_so_a_gem_outage_cannot_zero_them()
    {
        // Regression guard for the 2026-06-05 incident (sibling to the
        // temperature guard above): Open-Meteo's GEM (cmc_gem_gdps) ingestion
        // froze at 2026-05-26. gem_seamless was REQUIRED across every element /
        // wind blender, so the post-pivot "every required model NOT NULL" gate
        // zeroed out wind / humidity / radiation / cloud / gust / wind_speed_lgb
        // from 06-02 — which starved the feels-like / UTCI derivation (no element
        // inputs → no feels-like, the original bug report). Demoting gem to
        // OPTIONAL everywhere keeps it as a feature (LightGBM handles the NaN
        // slot natively, preserving cloud's bracketing-corrector use) while
        // making a gem outage a graceful degradation. This pins that gem is
        // never required at any lead so a future edit can't reintroduce the
        // fragility. (wind_mvn direction is Python + median-imputes missing
        // NWPs, so it has no required gate to guard here.)
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new Microsoft.Extensions.Configuration.ConfigurationBuilder()
            .AddYamlFile(configPath, optional: false)
            .Build();
        var bound = new AppConfig();
        cfg.Bind(bound);

        // (wind_speed_lgb left this list 2026-06-10 — Python-trained now; its
        // required set never included gem anyway.)
        var elementBlenders = new[]
        {
            ("wind", "default"), ("humidity", "default"), ("cloud", "default"),
            ("radiation", "default"), ("wind_gust", "default"),
        };
        int[] leads = { 24, 48, 72, 96, 120 };
        foreach (var (target, featureSet) in elementBlenders)
        {
            var blender = bound.Blenders.Get(target, featureSet);
            blender.RequiredModels.Should().NotContain("gem_seamless",
                $"{target}/{featureSet} must not require gem_seamless (2026-06-05 GEM-outage robustness)");
            foreach (var lead in leads)
            {
                blender.RequiredForLead(lead).Should().NotContain("gem_seamless",
                    $"{target}/{featureSet} must not require gem_seamless at lead {lead}h");
            }
        }
    }

    [Fact]
    public void RockSurface_block_binds_with_expected_defaults()
    {
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new Microsoft.Extensions.Configuration.ConfigurationBuilder()
            .AddYamlFile(configPath, optional: false)
            .Build();
        var bound = new AppConfig();
        cfg.Bind(bound);

        var rs = bound.RockSurface;
        rs.GreasyMarginC.Should().Be(3.0, "Harry's within-3°C greasy threshold");
        rs.MuScale.Should().Be(0.3);
        rs.LwCloudK.Should().Be(0.54, "GFS-DLWRF-calibrated cloud-enhancement scale");
        rs.LwClearK.Should().Be(1.0);
        rs.Substeps.Should().Be(6);
        rs.SpinupHours.Should().Be(48);
        rs.Albedo.Should().Be(0.30);
        rs.EpsRock.Should().Be(0.95);
    }
}
