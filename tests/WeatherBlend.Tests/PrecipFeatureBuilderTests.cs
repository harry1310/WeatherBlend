using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipFeatureBuilderTests
{
    // -----------------------------------------------------------------------
    // New canonical path (BlenderSpec / BinaryTrainingRow). Phase 2 refactor.
    // -----------------------------------------------------------------------

    private static BlendersConfig LoadShippedConfig()
    {
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.yaml");
        var cfg = new ConfigurationBuilder().AddYamlFile(configPath, optional: false).Build();
        var bound = new AppConfig();
        cfg.Bind(bound);
        return bound.Blenders;
    }

    [Fact]
    public void BuildSpec_lean_precip_lead_24_has_6_optional_models_and_27_features()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        spec.Target.Should().Be("precipitation");
        spec.FeatureSet.Should().Be("lean");
        spec.RequiredModels.Should().BeEmpty();
        spec.OptionalModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless",
            "ecmwf_aifs025_single", "jma_seamless");
        // Layout: 7 precip + 4 spread + 7 covariates + 4 calendar = 22 (prob_* dropped).
        spec.FeatureNames.Should().HaveCount(22);
        spec.FeatureNames.Should().StartWith(new[]
        {
            "precip_gfs", "precip_ecmwf", "precip_icon", "precip_mf", "precip_gem", "precip_aifs", "precip_jma",
        });
        spec.FeatureNames.Should().NotContain("prob_gfs");
        spec.FeatureNames.Should().Contain("precip_agreement_wet_01");
    }

    [Fact]
    public void BuildSpec_lean_precip_lead_120_drops_MF_to_5_models_and_25_features()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 120);
        spec.OptionalModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ecmwf_aifs025_single", "jma_seamless");
        // 6 precip + 4 spread + 7 covariates + 4 calendar = 21 (prob_* dropped).
        spec.FeatureNames.Should().HaveCount(21);
        spec.FeatureNames.Should().NotContain("precip_mf");
        spec.FeatureNames.Should().NotContain("prob_mf");
        spec.FeatureNames.Should().NotContain("precip_ukmo");
        spec.FeatureNames.Should().Contain("precip_aifs");
        spec.FeatureNames.Should().Contain("precip_jma");
    }

    [Fact]
    public void ComposeRow_with_spec_packs_features_and_label()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        // 7 models in spec at lead 24h: gfs/ecmwf/icon/mf/gem/aifs/jma.
        var precip = new[] { 0.0, 0.05, 0.2, 0.5, 1.0, 0.3, 0.4 };
        var row = PrecipFeatureBuilder.ComposeRow(
            spec, new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            precip,
            rhMean: 80, dewDepressionMean: 2,
            cloudLowMean: 50, cloudMidMean: 30, cloudHighMean: 20,
            capeMean: 100, windSpeedMean: 5,
            truthMmHour: 0.3);

        row.Features.Length.Should().Be(spec.FeatureCount);
        for (int i = 0; i < precip.Length; i++)
            row.Features[i].Should().BeApproximately((float)precip[i], 1e-5f);
        row.Label.Should().BeTrue();          // 0.3 mm/h ≥ 0.1 mm threshold
        row.TruthMmHour.Should().BeApproximately(0.3f, 1e-5f);
        // Mean of (0.0, 0.05, 0.2, 0.5, 1.0, 0.3, 0.4) = 2.45/7 ≈ 0.35; max = 1.0;
        // agreement = 5/7 (precip≥0.1mm: 0.2, 0.5, 1.0, 0.3, 0.4).
        row.Features[spec.IndexOf("precip_mean")].Should().BeApproximately(2.45f / 7f, 1e-4f);
        row.Features[spec.IndexOf("precip_max")].Should().BeApproximately(1.0f, 1e-4f);
        row.Features[spec.IndexOf("precip_agreement_wet_01")].Should().BeApproximately(5f / 7f, 1e-4f);
    }

    [Fact]
    public void ComposeRow_throws_when_perModelPrecip_count_does_not_match_spec()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);  // 7 models with JMA
        var act = () => PrecipFeatureBuilder.ComposeRow(
            spec, DateTime.UtcNow, new[] { 0.0, 0.0 },
            0, 0, 0, 0, 0, 0, 0, 0);
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData(0.0,  false)]
    [InlineData(0.05, false)]
    [InlineData(0.1,  true)]
    [InlineData(1.5,  true)]
    public void Label_respects_0_1mm_wet_threshold(double truthMm, bool expected)
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var precip = new double[spec.Models.Count];
        var row = PrecipFeatureBuilder.ComposeRow(
            spec, new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            precip,
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0, cloudHighMean: 0,
            capeMean: 0, windSpeedMean: 0,
            truthMmHour: truthMm);

        row.Label.Should().Be(expected);
        row.TruthMmHour.Should().BeApproximately((float)truthMm, 1e-6f);
    }

    [Fact]
    public void ComposeRow_aggregates_skip_nan_per_model_slots()
    {
        // Three non-null precip values: 0.2, 0.5, 1.0 — mean 1.7/3, max 1.0, agreement 1.0.
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var precip = new double[spec.Models.Count];
        for (int i = 0; i < precip.Length; i++) precip[i] = double.NaN;
        precip[0] = 0.2; precip[2] = 0.5; precip[4] = 1.0;

        var row = PrecipFeatureBuilder.ComposeRow(
            spec, new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            precip,
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0, cloudHighMean: 0,
            capeMean: 0, windSpeedMean: 0,
            truthMmHour: 0.0);

        row.Features[spec.IndexOf("precip_mean")].Should().BeApproximately(1.7f / 3f, 1e-4f);
        row.Features[spec.IndexOf("precip_max")].Should().BeApproximately(1.0f, 1e-4f);
        row.Features[spec.IndexOf("precip_agreement_wet_01")].Should().BeApproximately(1.0f, 1e-4f);
    }
}
