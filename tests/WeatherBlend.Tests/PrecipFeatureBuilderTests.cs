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
    public void BuildSpec_lean_precip_lead_96_mirrors_lead_120_dropping_MF()
    {
        // Lead 96h was added 2026-04-29. Same MF-cap reasoning as temperature.
        var spec = PrecipFeatureBuilder.BuildSpec(LoadShippedConfig(), 96);
        spec.OptionalModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ecmwf_aifs025_single", "jma_seamless");
        spec.FeatureNames.Should().HaveCount(21);
        spec.FeatureNames.Should().NotContain("precip_mf");
    }

    [Fact]
    public void BuildSpec_rich_precip_lead_96_drops_MF_keeps_UKMO_AIFS_JMA()
    {
        var spec = WeatherBlend.Train.PrecipRichFeatureBuilder.BuildSpec(LoadShippedConfig(), 96);
        spec.OptionalModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ukmo_seamless", "gem_seamless",
            "ecmwf_aifs025_single", "jma_seamless");
        spec.FeatureNames.Should().NotContain("precip_mf");
        spec.FeatureNames.Should().Contain("precip_ukmo");
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

    // --- Predict-time upper-air (UpperAirValuesFor) — must reproduce the
    //     trainer's uaValues construction so 3c/3o predict feeds the model the
    //     same column layout it was trained on. (2026-06-02 UA productionisation.)

    [Fact]
    public void UpperAirValuesFor_picks_freshest_entry_at_or_before_valid_with_means()
    {
        // 4 models × 10 cols = 40, model-major. t850 is col 0 of each model's
        // block (idx 0,10,20,30); rh850 is col 9 (idx 9,19,29,39). Build two
        // exact-grid entries; query between them → expect the EARLIER one.
        const int width = 40;
        double[] Block(double t850, double rh850)
        {
            var a = new double[width];
            Array.Fill(a, double.NaN);
            for (int k = 0; k < 4; k++) { a[10 * k + 0] = t850 + k; a[10 * k + 9] = rh850 + k; }
            return a;
        }
        var early = new DateTime(2026, 6, 1, 0, 0, 0, DateTimeKind.Utc);
        var late  = new DateTime(2026, 6, 1, 6, 0, 0, DateTimeKind.Utc);
        var asof = new List<(DateTime, double[])> { (early, Block(10, 80)), (late, Block(20, 90)) };

        // Query at 03:00Z — freshest ≤ V is the 00:00Z (early) entry.
        var v = PrecipFeatureBuilder.UpperAirValuesFor(asof, early.AddHours(3));
        v.Length.Should().Be(42, "40 model×col + t850_mean + rh850_mean");
        v[0].Should().Be(10.0);                       // gfs t850 from the early block
        // t850_mean = mean over models of (10,11,12,13) = 11.5; rh850_mean of (80..83)=81.5
        v[40].Should().BeApproximately(11.5, 1e-9);
        v[41].Should().BeApproximately(81.5, 1e-9);

        // Query before the first entry → all-NaN (missing; LightGBM tolerant).
        var none = PrecipFeatureBuilder.UpperAirValuesFor(asof, early.AddHours(-1));
        none.Length.Should().Be(42);
        none.Should().OnlyContain(x => double.IsNaN(x));
    }

    [Fact]
    public void UpperAirAsofTime_returns_the_valid_time_of_the_selected_entry()
    {
        var early = new DateTime(2026, 6, 1, 0, 0, 0, DateTimeKind.Utc);
        var late  = new DateTime(2026, 6, 1, 6, 0, 0, DateTimeKind.Utc);
        var asof = new List<(DateTime, double[])>
        {
            (early, new double[1]),
            (late,  new double[1]),
        };

        // Freshest ≤ 03:00Z is the 00:00Z entry — 3h reach-back.
        PrecipFeatureBuilder.UpperAirAsofTime(asof, early.AddHours(3)).Should().Be(early);
        // Exactly on the late entry → 0h reach-back (freshest).
        PrecipFeatureBuilder.UpperAirAsofTime(asof, late).Should().Be(late);
        // Before the first entry → null (no UA in hand).
        PrecipFeatureBuilder.UpperAirAsofTime(asof, early.AddHours(-1)).Should().BeNull();
    }
}
