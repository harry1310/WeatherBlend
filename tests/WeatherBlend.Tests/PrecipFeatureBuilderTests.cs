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
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless", "ecmwf_aifs025_single");
        // Layout: 6 precip + 6 prob + 4 spread + 7 covariates + 4 calendar = 27.
        spec.FeatureNames.Should().HaveCount(27);
        spec.FeatureNames.Should().StartWith(new[]
        {
            "precip_gfs", "precip_ecmwf", "precip_icon", "precip_mf", "precip_gem", "precip_aifs",
            "prob_gfs", "prob_ecmwf", "prob_icon", "prob_mf", "prob_gem", "prob_aifs",
        });
        spec.FeatureNames.Should().Contain("precip_agreement_wet_01");
    }

    [Fact]
    public void BuildSpec_lean_precip_lead_120_drops_MF_to_5_models_and_25_features()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 120);
        spec.OptionalModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ecmwf_aifs025_single");
        // 5 precip + 5 prob + 4 spread + 7 covariates + 4 calendar = 25.
        spec.FeatureNames.Should().HaveCount(25);
        spec.FeatureNames.Should().NotContain("precip_mf");
        spec.FeatureNames.Should().NotContain("prob_mf");
        spec.FeatureNames.Should().NotContain("precip_ukmo");
        spec.FeatureNames.Should().Contain("precip_aifs");
    }

    [Fact]
    public void ComposeRow_with_spec_packs_features_and_label()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        // 6 models in spec at lead 24h: gfs/ecmwf/icon/mf/gem/aifs.
        var precip = new[] { 0.0, 0.05, 0.2, 0.5, 1.0, 0.3 };
        var probs  = new[] { 0.1, 0.2,  0.3, 0.4, 0.5, 0.6 };
        var row = PrecipFeatureBuilder.ComposeRow(
            spec, new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            precip, probs,
            rhMean: 80, dewDepressionMean: 2,
            cloudLowMean: 50, cloudMidMean: 30, cloudHighMean: 20,
            capeMean: 100, windSpeedMean: 5,
            truthMmHour: 0.3);

        row.Features.Length.Should().Be(spec.FeatureCount);
        for (int i = 0; i < precip.Length; i++)
            row.Features[i].Should().BeApproximately((float)precip[i], 1e-5f);
        row.Label.Should().BeTrue();          // 0.3 mm/h ≥ 0.1 mm threshold
        row.TruthMmHour.Should().BeApproximately(0.3f, 1e-5f);
        // Mean of (0.0, 0.05, 0.2, 0.5, 1.0, 0.3) = 0.3417; max = 1.0;
        // agreement = 4/6 (precip≥0.1mm: 0.2, 0.5, 1.0, 0.3).
        row.Features[spec.IndexOf("precip_mean")].Should().BeApproximately(2.05f / 6f, 1e-4f);
        row.Features[spec.IndexOf("precip_max")].Should().BeApproximately(1.0f, 1e-4f);
        row.Features[spec.IndexOf("precip_agreement_wet_01")].Should().BeApproximately(4f / 6f, 1e-4f);
    }

    [Fact]
    public void ComposeRow_throws_when_perModelPrecip_count_does_not_match_spec()
    {
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);  // 6 models incl AIFS
        var act = () => PrecipFeatureBuilder.ComposeRow(
            spec, DateTime.UtcNow, new[] { 0.0, 0.0 }, new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            0, 0, 0, 0, 0, 0, 0, 0);
        act.Should().Throw<ArgumentException>();
    }

    // -----------------------------------------------------------------------
    // Legacy fixed-shape path (kept until rich precip migrates in Phase 3)
    // -----------------------------------------------------------------------

    [Fact]
    public void OccurrenceFeatureNames_has_exactly_27_columns_in_expected_order()
    {
        PrecipFeatureBuilder.OccurrenceFeatureNames.Should().HaveCount(27);
        PrecipFeatureBuilder.OccurrenceFeatureNames.Should().ContainInOrder(
            "precip_gfs", "precip_ecmwf", "precip_icon", "precip_mf", "precip_ukmo", "precip_gem",
            "prob_gfs",   "prob_ecmwf",   "prob_icon",   "prob_mf",   "prob_ukmo",   "prob_gem",
            "precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01",
            "rh_mean", "dew_depression_mean",
            "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
            "cape_mean", "wind_speed_mean",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos");
    }

    [Fact]
    public void ModelColumns_tuple_order_aligns_with_feature_name_order()
    {
        // The first 6 feature slots are the per-model precip cols, in ModelColumns order.
        var featurePrecipCols = PrecipFeatureBuilder.OccurrenceFeatureNames.Take(6).ToArray();
        var modelPrecipCols = PrecipFeatureBuilder.ModelColumns.Select(m => m.PrecipCol).ToArray();
        featurePrecipCols.Should().Equal(modelPrecipCols);

        // Same invariant for the 6 prob cols in slots 6..11.
        var featureProbCols = PrecipFeatureBuilder.OccurrenceFeatureNames.Skip(6).Take(6).ToArray();
        var modelProbCols = PrecipFeatureBuilder.ModelColumns.Select(m => m.ProbCol).ToArray();
        featureProbCols.Should().Equal(modelProbCols);
    }

    [Fact]
    public void ComposeRow_computes_ensemble_mean_std_max_and_agreement_from_six_models()
    {
        // Six model precip values: 0.0, 0.05, 0.2, 0.5, 1.0, 2.0
        // Mean = 3.75 / 6 = 0.625. Max = 2.0.
        // WetCount (>=0.1mm): 0.2, 0.5, 1.0, 2.0 → 4 → agreement = 4/6 ≈ 0.6667.
        var row = PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new[] { 0.0, 0.05, 0.2, 0.5, 1.0, 2.0 },
            perModelProb:   new[] { 0.0, 0.0,  0.0, 0.0, 0.0, 0.0 },
            rhMean: 80, dewDepressionMean: 2, cloudLowMean: 50, cloudMidMean: 30,
            cloudHighMean: 20, capeMean: 100, windSpeedMean: 5,
            truthMmHour: 0.0);

        row.PrecipMean.Should().BeApproximately(0.625f, 1e-4f);
        row.PrecipMax.Should().BeApproximately(2.0f, 1e-4f);
        row.PrecipAgreementWet01.Should().BeApproximately(4f / 6f, 1e-4f);
    }

    [Fact]
    public void ComposeRow_ensemble_aggregates_skip_nan_models()
    {
        // Three non-null precip values: 0.2, 0.5, 1.0. Mean = 1.7/3 ≈ 0.5667, max = 1.0.
        // WetCount (>=0.1mm) = 3 of 3 non-null → agreement = 1.0.
        var row = PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new[] { 0.2, double.NaN, 0.5, double.NaN, 1.0, double.NaN },
            perModelProb:   new[] { 0.0, 0.0,       0.0, 0.0,       0.0, 0.0       },
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0,
            cloudHighMean: 0, capeMean: 0, windSpeedMean: 0,
            truthMmHour: 0.0);

        row.PrecipMean.Should().BeApproximately(1.7f / 3f, 1e-4f);
        row.PrecipMax.Should().BeApproximately(1.0f, 1e-4f);
        row.PrecipAgreementWet01.Should().BeApproximately(1.0f, 1e-4f);
    }

    [Fact]
    public void ComposeRow_aggregates_become_nan_when_all_models_missing()
    {
        var row = PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: Enumerable.Repeat(double.NaN, 6).ToArray(),
            perModelProb:   Enumerable.Repeat(double.NaN, 6).ToArray(),
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0,
            cloudHighMean: 0, capeMean: 0, windSpeedMean: 0,
            truthMmHour: 0.0);

        float.IsNaN(row.PrecipMean).Should().BeTrue();
        float.IsNaN(row.PrecipMax).Should().BeTrue();
        float.IsNaN(row.PrecipAgreementWet01).Should().BeTrue();
        // std collapses to 0 when presentCount<=1 by construction.
        row.PrecipStd.Should().Be(0f);
    }

    [Fact]
    public void ComposeRow_std_is_zero_when_only_one_model_present()
    {
        var row = PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new[] { 0.5, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN },
            perModelProb:   new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0,
            cloudHighMean: 0, capeMean: 0, windSpeedMean: 0,
            truthMmHour: 0.0);

        row.PrecipStd.Should().Be(0f);  // variance of a single point is 0, not NaN
        row.PrecipMean.Should().BeApproximately(0.5f, 1e-6f);
    }

    [Theory]
    [InlineData(0.0,  false)]   // dry
    [InlineData(0.05, false)]   // just below threshold
    [InlineData(0.1,  true)]    // exactly at threshold → wet (>=)
    [InlineData(1.5,  true)]    // clearly wet
    public void ComposeRow_WetBinary_label_respects_0_1mm_threshold(double truthMm, bool expected)
    {
        var row = PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            perModelProb:   new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0,
            cloudHighMean: 0, capeMean: 0, windSpeedMean: 0,
            truthMmHour: truthMm);

        row.WetBinary.Should().Be(expected);
        row.PrecipMmHour.Should().BeApproximately((float)truthMm, 1e-6f);
    }

    [Theory]
    [InlineData(0,   0.0,   1.0)]   // sin(0)=0, cos(0)=1
    [InlineData(6,   1.0,   0.0)]   // π/2
    [InlineData(12,  0.0,  -1.0)]   // π
    [InlineData(18, -1.0,   0.0)]   // 3π/2
    public void ComposeRow_hour_encoding_matches_unit_circle(int hour, double expectedSin, double expectedCos)
    {
        var row = PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, hour, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            perModelProb:   new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0,
            cloudHighMean: 0, capeMean: 0, windSpeedMean: 0,
            truthMmHour: 0.0);

        row.HourSin.Should().BeApproximately((float)expectedSin, 1e-5f);
        row.HourCos.Should().BeApproximately((float)expectedCos, 1e-5f);
    }

    [Fact]
    public void ComposeRow_throws_on_wrong_precip_array_length()
    {
        var act = () => PrecipFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 0, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new[] { 0.0, 0.0 },    // only 2, not 6
            perModelProb:   new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            rhMean: 0, dewDepressionMean: 0, cloudLowMean: 0, cloudMidMean: 0,
            cloudHighMean: 0, capeMean: 0, windSpeedMean: 0,
            truthMmHour: 0.0);

        act.Should().Throw<ArgumentException>();
    }
}
