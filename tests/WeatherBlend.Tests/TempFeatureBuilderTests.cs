using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Temp;
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
    public void BuildSpec_lean_temperature_lead_24_requires_only_gfs_ecmwf_rest_optional()
    {
        // 2026-06-01 (commit 42f591d): temp lean requires only gfs+ecmwf; icon/
        // mf/gem/aifs are optional so a single lapsed feed can't truncate rows.
        // The model UNION (and feature schema) is unchanged — only the
        // required/optional split moved.
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        spec.Target.Should().Be("temperature");
        spec.FeatureSet.Should().Be("lean");
        spec.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025");
        spec.OptionalModels.Should().Equal(
            "icon_seamless", "meteofrance_seamless", "gem_seamless", "ecmwf_aifs025_single");
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
    public void BuildSpec_lean_temperature_lead_120_keeps_full_union_no_per_lead_drop()
    {
        // Post-42f591d there are NO perLeadOverrides: the required set is a flat
        // {gfs, ecmwf} at every lead, and MF/gem/aifs stay in the union as
        // optional (a short-horizon feed just leaves a NaN slot rather than
        // truncating the row). So lead 120 carries the SAME 6-model union as 24.
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 120);
        spec.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025");
        spec.OptionalModels.Should().Equal(
            "icon_seamless", "meteofrance_seamless", "gem_seamless", "ecmwf_aifs025_single");
        spec.Models.Should().HaveCount(6);
        spec.FeatureNames.Should().HaveCount(13);
        spec.FeatureNames.Should().Contain("temp_mf");      // no longer dropped at long lead
        spec.FeatureNames.Should().NotContain("temp_ukmo"); // lean never carries ukmo
        spec.FeatureNames.Should().Contain("temp_aifs");
    }

    [Fact]
    public void BuildSpec_lean_temperature_lead_96_matches_lead_24_flat_required()
    {
        // Flat required set ⇒ model membership no longer varies by lead (42f591d
        // removed the MF/gem-dropping perLeadOverrides). The lead-96 spec is
        // identical to lead-24's.
        var spec96 = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 96);
        var spec24 = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        spec96.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025");
        spec96.Models.Should().Equal(spec24.Models);
        spec96.FeatureNames.Should().Equal(spec24.FeatureNames);
        spec96.FeatureNames.Should().Contain("temp_mf");
    }

    [Fact]
    public void BuildSpec_rich_temperature_lead_96_requires_only_gfs_ecmwf_union_7()
    {
        // Rich mirrors lean's minimal-required policy (42f591d) but keeps ukmo
        // in the union → 7 models. No per-lead dropping, so MF + ukmo are both
        // present at lead 96.
        var spec = WeatherBlend.Train.TempRichFeatureBuilder.BuildSpec(LoadShippedConfig(), 96);
        spec.RequiredModels.Should().Equal("gfs_seamless", "ecmwf_ifs025");
        spec.OptionalModels.Should().Equal(
            "icon_seamless", "meteofrance_seamless", "ukmo_seamless", "gem_seamless", "ecmwf_aifs025_single");
        spec.Models.Should().HaveCount(7);
        spec.FeatureNames.Should().Contain("temp_mf");
        spec.FeatureNames.Should().Contain("temp_ukmo");
        spec.FeatureNames.Should().Contain("temp_aifs");
    }

    [Fact]
    public void ComposeRow_with_spec_packs_features_in_declared_order()
    {
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        // Per-model temps in spec order: gfs/ecmwf/icon/mf/gem/aifs.
        var temps = new[] { 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };
        var row = TempFeatureBuilder.ComposeRow(
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
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);  // 6 models incl. AIFS
        var act = () => TempFeatureBuilder.ComposeRow(
            spec, DateTime.UtcNow, new[] { 1.0, 2.0 }, double.NaN, 0.0);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ComposeRow_lead_120_packs_6_per_model_values_full_lean_union()
    {
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 120);
        // Flat 6-model lean union at 120h: gfs, ecmwf, icon, mf, gem, aifs
        // (no per-lead dropping post-42f591d).
        var temps = new[] { 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var row = TempFeatureBuilder.ComposeRow(
            spec, new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            temps, windDirMeanDeg: 90, era5Temp: 7.0);
        row.Features.Length.Should().Be(13);
        spec.FeatureNames.Should().Contain("temp_mf");      // present post-42f591d
        spec.FeatureNames.Should().NotContain("temp_ukmo"); // lean
        spec.FeatureNames.Should().Contain("temp_aifs");
    }

    [Theory]
    [InlineData(0,   0.0,   1.0)]
    [InlineData(6,   1.0,   0.0)]
    [InlineData(12,  0.0,  -1.0)]
    [InlineData(18, -1.0,   0.0)]
    public void Cyclical_hour_encoding_matches_unit_circle(int hour, double expectedSin, double expectedCos)
    {
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var temps = new double[spec.Models.Count];
        for (int i = 0; i < temps.Length; i++) temps[i] = 1.0 + i;
        var row = TempFeatureBuilder.ComposeRow(
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
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var temps = Enumerable.Repeat(1.0, spec.Models.Count).ToArray();
        var row = TempFeatureBuilder.ComposeRow(
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
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);  // 6 models
        // Five present values {10,12,14,16,18} → mean 14, range 8, std sqrt(8)≈2.8284.
        var temps = new[] { 10.0, 12.0, 14.0, 16.0, 18.0, double.NaN };
        var row = TempFeatureBuilder.ComposeRow(
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
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var temps = Enumerable.Repeat(1.0, spec.Models.Count).ToArray();
        var rows = Enumerable.Range(0, 20).Select(i => TempFeatureBuilder.ComposeRow(
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
        var spec = TempFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var rng = new Random(42);
        var rows = new List<RegressionTrainingRow>();
        var start = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        for (int i = 0; i < 400; i++)
        {
            var temps = new double[spec.Models.Count];
            for (int m = 0; m < temps.Length; m++) temps[m] = 10.0 + rng.NextDouble() * 5.0;
            var mean = temps.Average();
            var era5 = mean + (rng.NextDouble() - 0.5) * 0.2;
            rows.Add(TempFeatureBuilder.ComposeRow(spec, start.AddHours(i), temps, 180.0, era5));
        }

        var ds = RegressionDataset.Split(rows);
        var hp = new TempTrainer.Hyperparameters(NumberOfIterations: 50, EarlyStoppingRound: 10);
        var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);
        var predicted = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);

        predicted.Should().HaveCount(ds.Test.Count);
        predicted.All(x => !double.IsNaN(x) && !double.IsInfinity(x)).Should().BeTrue();

        var actual = ds.Test.Select(x => (double)x.Label).ToArray();
        var stats = WeatherBlend.Evaluate.Temp.TempMetrics.Compute(predicted, actual);
        double.IsNaN(stats.Mae).Should().BeFalse();
        stats.Mae.Should().BeLessThan(5.0);
    }
}
