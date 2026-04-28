using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

public class FeatureBuilderTests
{
    // -----------------------------------------------------------------------
    // New canonical path (BlenderSpec / RegressionTrainingRow)
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
    public void BuildSpec_lean_temperature_lead_24_uses_5_required_plus_aifs_optional()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        spec.Target.Should().Be("temperature");
        spec.FeatureSet.Should().Be("lean");
        spec.RequiredModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless");
        spec.OptionalModels.Should().Equal("ecmwf_aifs025_single");
        spec.Models.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless", "ecmwf_aifs025_single");
        // 6 per-model temps + 3 spread + 4 calendar = 13 features.
        spec.FeatureNames.Should().HaveCount(13);
        spec.FeatureNames.Should().ContainInOrder(
            "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_gem", "temp_aifs",
            "temp_mean", "temp_std", "temp_range",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos");
    }

    [Fact]
    public void BuildSpec_lean_temperature_lead_120_drops_MF_keeps_aifs_optional()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 120);
        spec.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless");
        spec.OptionalModels.Should().Equal("ecmwf_aifs025_single");
        spec.Models.Should().HaveCount(5);
        // 5 per-model temps + 3 spread + 4 calendar = 12 features.
        spec.FeatureNames.Should().HaveCount(12);
        spec.FeatureNames.Should().NotContain("temp_mf");
        spec.FeatureNames.Should().NotContain("temp_ukmo");
        spec.FeatureNames.Should().Contain("temp_aifs");
    }

    [Fact]
    public void BuildSpec_lean_temperature_lead_96_mirrors_lead_120_dropping_MF()
    {
        // Lead 96h was added 2026-04-29. Open-Meteo Previous Runs caps
        // meteofrance_seamless at ~72h, so MF must be excluded from required
        // at any lead ≥96h or training would yield zero rows.
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 96);
        spec.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless");
        spec.OptionalModels.Should().Equal("ecmwf_aifs025_single");
        spec.FeatureNames.Should().HaveCount(12);
        spec.FeatureNames.Should().NotContain("temp_mf");
    }

    [Fact]
    public void BuildSpec_rich_temperature_lead_96_drops_MF_keeps_UKMO_and_aifs()
    {
        var spec = WeatherBlend.Train.RichFeatureBuilder.BuildSpec(LoadShippedConfig(), 96);
        spec.RequiredModels.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ukmo_seamless", "gem_seamless");
        spec.OptionalModels.Should().Equal("ecmwf_aifs025_single");
        spec.Models.Should().HaveCount(6);
        spec.FeatureNames.Should().NotContain("temp_mf");
        spec.FeatureNames.Should().Contain("temp_ukmo");
        spec.FeatureNames.Should().Contain("temp_aifs");
    }

    [Fact]
    public void ComposeRow_with_spec_packs_features_in_declared_order()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        // Per-model temps in spec order: gfs/ecmwf/icon/mf/gem/aifs.
        var temps = new[] { 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };
        var row = FeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            temps,
            windDirMeanDeg: 180.0,
            era5Temp: 15.0);

        row.Features.Length.Should().Be(spec.FeatureCount);
        // First 6 entries are the per-model temps.
        for (int i = 0; i < temps.Length; i++) row.Features[i].Should().BeApproximately((float)temps[i], 1e-5f);
        // Then mean/std/range over [10..20] (6 values, mean=15, range=10).
        row.Features[spec.IndexOf("temp_mean")].Should().BeApproximately(15f, 1e-4f);
        row.Features[spec.IndexOf("temp_range")].Should().BeApproximately(10f, 1e-4f);
        row.Label.Should().BeApproximately(15f, 1e-5f);
        row.WindDirMean.Should().BeApproximately(180f, 1e-5f);
    }

    [Fact]
    public void ComposeRow_throws_when_temps_count_does_not_match_spec_models()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);  // 6 models incl. AIFS
        var act = () => FeatureBuilder.ComposeRow(
            spec, DateTime.UtcNow, new[] { 1.0, 2.0 }, double.NaN, 0.0);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ComposeRow_lead_120_packs_5_per_model_values_no_mf_no_ukmo_with_aifs()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 120);
        // Spec order at 120h: gfs, ecmwf, icon, gem, aifs.
        var temps = new[] { 5.0, 6.0, 7.0, 8.0, 9.0 };
        var row = FeatureBuilder.ComposeRow(
            spec, new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            temps, windDirMeanDeg: 90, era5Temp: 7.0);
        row.Features.Length.Should().Be(12);
        spec.FeatureNames.Should().NotContain("temp_mf");
        spec.FeatureNames.Should().NotContain("temp_ukmo");
        spec.FeatureNames.Should().Contain("temp_aifs");
    }

    [Theory]
    [InlineData(0,   0.0,   1.0)]
    [InlineData(6,   1.0,   0.0)]
    [InlineData(12,  0.0,  -1.0)]
    [InlineData(18, -1.0,   0.0)]
    public void Cyclical_hour_encoding_matches_unit_circle(int hour, double expectedSin, double expectedCos)
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var temps = new double[spec.Models.Count];
        for (int i = 0; i < temps.Length; i++) temps[i] = 1.0 + i;
        var row = FeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 1, 1, hour, 0, 0, DateTimeKind.Utc),
            temps,
            windDirMeanDeg: double.NaN,
            era5Temp: 0.0);

        row.Features[spec.IndexOf("hour_sin")].Should().BeApproximately((float)expectedSin, 1e-5f);
        row.Features[spec.IndexOf("hour_cos")].Should().BeApproximately((float)expectedCos, 1e-5f);
    }

    [Fact]
    public void Cyclical_doy_encoding_wraps_to_near_zero_at_year_boundary()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var temps = Enumerable.Repeat(1.0, spec.Models.Count).ToArray();
        var row = FeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            temps,
            windDirMeanDeg: double.NaN,
            era5Temp: 0.0);

        row.Features[spec.IndexOf("doy_sin")].Should().BeApproximately(0f, 1e-5f);
        row.Features[spec.IndexOf("doy_cos")].Should().BeApproximately(1f, 1e-5f);
    }

    [Fact]
    public void ComposeRow_spread_is_NaN_safe_when_one_model_is_missing()
    {
        // Population std of present values must stay finite when one slot is NaN.
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);  // 6 models
        // Five present values {10,12,14,16,18} → mean 14, range 8, std sqrt(8)≈2.8284.
        var temps = new[] { 10.0, 12.0, 14.0, 16.0, 18.0, double.NaN };
        var row = FeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            temps,
            windDirMeanDeg: double.NaN,
            era5Temp: 14.0);

        row.Features[spec.IndexOf("temp_mean")].Should().BeApproximately(14f, 1e-4f);
        row.Features[spec.IndexOf("temp_range")].Should().BeApproximately(8f, 1e-4f);
        row.Features[spec.IndexOf("temp_std")].Should().BeApproximately(2.8284f, 1e-3f);
        // The missing slot stays NaN — LightGBM treats as missing.
        float.IsNaN(row.Features[5]).Should().BeTrue();
    }

    [Fact]
    public void RegressionDataset_split_is_chronological_and_non_overlapping()
    {
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var temps = Enumerable.Repeat(1.0, spec.Models.Count).ToArray();
        var rows = Enumerable.Range(0, 20).Select(i => FeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddHours(i),
            temps,
            windDirMeanDeg: 0.0,
            era5Temp: 1.0)).ToList();

        var ds = RegressionDataset.Split(rows);

        ds.Train.Count.Should().Be(14);
        ds.Val.Count.Should().Be(3);
        ds.Test.Count.Should().Be(3);
        ds.TrainEnd.Should().BeBefore(ds.ValStart);
        ds.ValEnd.Should().BeBefore(ds.TestStart);
    }

    [Fact]
    public void LeadModelFileName_formats_as_leadNh_zip()
    {
        WeatherBlend.Train.ModelArtifact.LeadModelFileName(24).Should().Be("lead_24h.zip");
        WeatherBlend.Train.ModelArtifact.LeadModelFileName(72).Should().Be("lead_72h.zip");
    }

    [Fact]
    public void TemperatureTrainer_end_to_end_on_synthetic_data_produces_non_nan_metrics()
    {
        // Tiny synthetic set: ERA5 truth = mean of per-model temps + small noise so the
        // trainer has something to fit. Enough rows to survive 70/15/15.
        var spec = FeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var rng = new Random(42);
        var rows = new List<RegressionTrainingRow>();
        var start = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        for (int i = 0; i < 400; i++)
        {
            var temps = new double[spec.Models.Count];
            for (int m = 0; m < temps.Length; m++) temps[m] = 10.0 + rng.NextDouble() * 5.0;
            var mean = temps.Average();
            var era5 = mean + (rng.NextDouble() - 0.5) * 0.2;
            rows.Add(FeatureBuilder.ComposeRow(spec, start.AddHours(i), temps, 180.0, era5));
        }

        var ds = RegressionDataset.Split(rows);
        var hp = new TemperatureTrainer.Hyperparameters(NumberOfIterations: 50, EarlyStoppingRound: 10);
        var trained = TemperatureTrainer.TrainVector(ds.Train, ds.Val, spec, hp);
        var predicted = TemperatureTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);

        predicted.Should().HaveCount(ds.Test.Count);
        predicted.All(x => !double.IsNaN(x) && !double.IsInfinity(x)).Should().BeTrue();

        var actual = ds.Test.Select(x => (double)x.Label).ToArray();
        var stats = WeatherBlend.Evaluate.Metrics.Compute(predicted, actual);
        double.IsNaN(stats.Mae).Should().BeFalse();
        stats.Mae.Should().BeLessThan(5.0);
    }
}
