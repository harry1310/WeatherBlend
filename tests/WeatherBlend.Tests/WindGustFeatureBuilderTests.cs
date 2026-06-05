using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Gust;
using Xunit;

namespace WeatherBlend.Tests;

public class WindGustFeatureBuilderTests
{
    private static BlendersConfig LoadShippedConfig()
    {
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new ConfigurationBuilder().AddYamlFile(configPath, optional: false).Build();
        var bound = new AppConfig();
        cfg.Bind(bound);
        return bound.Blenders;
    }

    [Fact]
    public void BuildSpec_lead_24_matches_4_NWP_production_scope()
    {
        var spec = WindGustFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);

        spec.Target.Should().Be("wind_gust");
        spec.FeatureSet.Should().Be("default");
        // gem_seamless demoted required→optional 2026-06-05 (GEM-outage
        // robustness — see ConfigTests.ElementBlenders_never_require_gem). gem
        // keeps its feature slot (still in optional + Models), so a gem outage
        // degrades the gust blend gracefully instead of zeroing it.
        spec.RequiredModels.Should().Equal("gfs_seamless", "icon_seamless");
        spec.OptionalModels.Should().Equal("ukmo_seamless", "gem_seamless");
        // spec.Models is required ∪ optional reordered by CanonicalModelOrder,
        // so UKMO sits before GEM. Unchanged by the gem required→optional move.
        spec.Models.Should().Equal(
            "gfs_seamless", "icon_seamless", "ukmo_seamless", "gem_seamless");
        // 4 gust + 4 ratio + 2 spread = 10 features (production-scope shape,
        // matches the 2026-05-27 bake-off's minimal variant at MAE 1.04).
        spec.FeatureNames.Should().HaveCount(10);
        spec.FeatureNames.Should().ContainInOrder(
            "gust_gfs", "gust_icon", "gust_ukmo", "gust_gem",
            "gust_ratio_gfs", "gust_ratio_icon", "gust_ratio_ukmo", "gust_ratio_gem",
            "gust_ratio_mean", "gust_ratio_std");
    }

    [Fact]
    public void ComposeRow_packs_gust_then_ratio_then_spread_with_correct_clip_bounds()
    {
        var spec = WindGustFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        // 4 NWPs in spec.Models order: GFS / ICON / UKMO / GEM (canonical-order
        // intersection). Synthetic row exercises the ratio clip on both ends.
        //
        // GFS:  12 / 8   -> ratio 1.50 (in-range)
        // ICON: 20 / 4   -> ratio 5.00 -> CLIPPED to 4.0 (upper)
        // UKMO: 2.5 / 0.2 -> denom max(0.2, 0.5)=0.5 -> ratio 5.00 -> clipped 4.0
        // GEM:  0.3 / 5  -> ratio 0.06 -> CLIPPED to 0.5 (lower)
        var gusts  = new double[] { 12.0, 20.0, 2.5, 0.3 };
        var speeds = new double[] {  8.0,  4.0, 0.2, 5.0 };
        var validTime = new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var row = WindGustFeatureBuilder.ComposeRow(spec, validTime, gusts, speeds, era5GustMs: 15.5);

        row.ValidTimeUtc.Should().Be(validTime);
        row.Label.Should().Be(15.5f);

        // [0..3] = per-NWP gust passed through unmodified
        row.Features[0].Should().BeApproximately(12.0f, 1e-5f);
        row.Features[1].Should().BeApproximately(20.0f, 1e-5f);
        row.Features[2].Should().BeApproximately(2.5f, 1e-5f);
        row.Features[3].Should().BeApproximately(0.3f, 1e-5f);

        // [4..7] = per-NWP ratios with clip applied
        row.Features[4].Should().BeApproximately(1.50f, 1e-5f);  // 12/8 in-range
        row.Features[5].Should().Be(WindGustFeatureBuilder.RatioMax);  // 20/4 = 5 -> clipped to 4.0
        row.Features[6].Should().Be(WindGustFeatureBuilder.RatioMax);  // 2.5 / max(0.2,0.5)=0.5 = 5 -> 4.0
        row.Features[7].Should().Be(WindGustFeatureBuilder.RatioMin);  // 0.3/5 = 0.06 -> 0.5

        // [8] = ratio mean = (1.5 + 4.0 + 4.0 + 0.5) / 4 = 2.5
        row.Features[8].Should().BeApproximately(2.5f, 1e-5f);
        // [9] = ratio std (population) — variance = mean((x-2.5)^2)
        // residuals: -1.0, 1.5, 1.5, -2.0 -> squared 1, 2.25, 2.25, 4
        // -> variance = 9.5/4 = 2.375 -> std ≈ 1.5411
        row.Features[9].Should().BeApproximately(1.5411f, 1e-3f);
    }

    [Fact]
    public void ComposeRow_handles_NaN_inputs_via_skip_semantics()
    {
        // UKMO silent (NaN gust + NaN wsp) — its ratio should be NaN and skipped
        // from spread; spread is computed over the 3 present NWPs only.
        var spec = WindGustFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var gusts  = new double[] { 10.0, 12.0, 8.0, double.NaN };
        var speeds = new double[] {  6.0,  7.0, 5.0, double.NaN };
        var validTime = new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var row = WindGustFeatureBuilder.ComposeRow(spec, validTime, gusts, speeds, era5GustMs: 11.0);

        // UKMO gust + ratio both NaN
        float.IsNaN(row.Features[3]).Should().BeTrue("UKMO gust input was NaN");
        float.IsNaN(row.Features[7]).Should().BeTrue("UKMO ratio derived from NaN inputs");

        // Ratios: 10/6=1.667, 12/7=1.714, 8/5=1.600 (all in-range)
        row.Features[4].Should().BeApproximately(10.0f / 6.0f, 1e-4f);
        row.Features[5].Should().BeApproximately(12.0f / 7.0f, 1e-4f);
        row.Features[6].Should().BeApproximately(8.0f / 5.0f, 1e-4f);

        // Spread over 3 present ratios — UKMO NaN skipped.
        var expectedMean = (10.0f / 6.0f + 12.0f / 7.0f + 8.0f / 5.0f) / 3.0f;
        row.Features[8].Should().BeApproximately(expectedMean, 1e-4f);
        float.IsNaN(row.Features[9]).Should().BeFalse("std should be finite when ≥1 ratio is present");
    }

    [Fact]
    public void ComposeRow_low_wind_uses_floor_in_ratio_denominator()
    {
        // gust=4, wsp=0.1 -> raw ratio would be 40 (degenerate); floor of 0.5
        // means denom is 0.5, so ratio = 8.0 -> clipped to 4.0.
        var spec = WindGustFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var gusts  = new double[] { 4.0, 4.0, 4.0, 4.0 };
        var speeds = new double[] { 0.1, 0.05, 0.0, -1.0 };  // last 2 nonsensical but should still floor
        var validTime = new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var row = WindGustFeatureBuilder.ComposeRow(spec, validTime, gusts, speeds, era5GustMs: 5.0);

        // All four ratios should be at the upper clip (4.0) because the floor
        // makes denom = 0.5 → raw = 8.0 → clipped to 4.0.
        row.Features[4].Should().Be(WindGustFeatureBuilder.RatioMax);
        row.Features[5].Should().Be(WindGustFeatureBuilder.RatioMax);
        row.Features[6].Should().Be(WindGustFeatureBuilder.RatioMax);
        row.Features[7].Should().Be(WindGustFeatureBuilder.RatioMax);
    }

    [Fact]
    public void ComposeRow_throws_when_input_arrays_size_mismatch_spec_models()
    {
        var spec = WindGustFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var threeGusts = new double[] { 10.0, 11.0, 12.0 };  // missing UKMO slot
        var fourSpeeds = new double[] { 6.0, 7.0, 8.0, 5.0 };
        var validTime = new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var act = () => WindGustFeatureBuilder.ComposeRow(spec, validTime, threeGusts, fourSpeeds, 10.0);
        act.Should().Throw<ArgumentException>().WithMessage("Expected 4 gusts*");
    }
}
