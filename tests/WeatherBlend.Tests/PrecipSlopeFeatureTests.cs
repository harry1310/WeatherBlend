using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Oro;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// The dProg/dt forecast-trend slope feature: per-phase arm wiring (PROV → 3c, AGG → 3o)
/// and the leak-safe least-squares slope. Locks the contract the production retrain relies on.
/// </summary>
public class PrecipSlopeFeatureTests
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
    public void Slope_per_provider_arm_appends_one_column_per_model_after_the_baseline()
    {
        var cfg = LoadShippedConfig();
        var baseline = PrecipRichFeatureBuilder.BuildSpec(cfg, 24);
        var prov = PrecipRichFeatureBuilder.BuildSpec(cfg, 24,
            slopeArm: PrecipRichFeatureBuilder.SlopeArm.PerProvider);

        // One pslope_<model> per model, and NOTHING else changes.
        prov.FeatureNames.Count.Should().Be(baseline.FeatureNames.Count + baseline.Models.Count);
        prov.FeatureNames.Count(f => f.StartsWith("pslope_")).Should().Be(prov.Models.Count);
        prov.FeatureNames.Should().NotContain("pslope_mean");
        prov.FeatureSet.Should().Be("rich-pslope");
        // Slope is APPENDED last — every baseline index is untouched.
        prov.FeatureNames.Take(baseline.FeatureNames.Count).Should().Equal(baseline.FeatureNames);
    }

    [Fact]
    public void Slope_aggregate_arm_appends_only_mean_and_std()
    {
        var cfg = LoadShippedConfig();
        var baseline = PrecipRichFeatureBuilder.BuildSpec(cfg, 24);
        var agg = PrecipRichFeatureBuilder.BuildSpec(cfg, 24,
            slopeArm: PrecipRichFeatureBuilder.SlopeArm.Aggregate);

        agg.FeatureNames.Count.Should().Be(baseline.FeatureNames.Count + 2);
        agg.FeatureNames.Should().Contain("pslope_mean").And.Contain("pslope_std");
        agg.FeatureNames.Count(f => f.StartsWith("pslope_")).Should().Be(2);
        agg.FeatureSet.Should().Be("rich-aslope");
        agg.FeatureNames.Take(baseline.FeatureNames.Count).Should().Equal(baseline.FeatureNames);
    }

    [Fact]
    public void Oro_3o_inherits_the_aggregate_slope_in_the_rich_prefix_before_terrain()
    {
        var cfg = LoadShippedConfig();
        var oro = PrecipRichOroFeatureBuilder.BuildSpec(cfg, 24, withUpperAir: false, includeStationId: true,
            slopeArm: PrecipRichFeatureBuilder.SlopeArm.Aggregate);

        oro.FeatureSet.Should().Be("rich-oro-aslope");
        oro.FeatureNames.Should().Contain("pslope_mean").And.Contain("pslope_std");
        oro.FeatureNames.Should().NotContain("pslope_gfs");  // AGG arm, not PROV
        // Slope lives in the rich prefix, BEFORE the terrain block — so the oro builder's
        // "strip the last N terrain columns" reconstruction still recovers the slope.
        var names = oro.FeatureNames.ToList();
        var terrainStart = names.IndexOf("oro_elevation_vs_cell_m");
        terrainStart.Should().BeGreaterThan(0);
        names.IndexOf("pslope_mean").Should().BeLessThan(terrainStart);
        names.IndexOf("pslope_std").Should().BeLessThan(terrainStart);
    }

    [Fact]
    public void LeadSafeSlope_recovers_a_clean_linear_trend_and_excludes_leads_at_or_below_L()
    {
        // [p@48, p@72, p@96, p@120] rising 1mm per 24h of lead → slope = 1/24 per hour.
        var clean = new[] { 2.0, 3.0, 4.0, 5.0 };
        PrecipRichFeatureBuilder.LeadSafeSlope(clean, 24).Should().BeApproximately(1.0 / 24, 1e-9); // {48,72,96,120}
        PrecipRichFeatureBuilder.LeadSafeSlope(clean, 48).Should().BeApproximately(1.0 / 24, 1e-9); // {72,96,120}
        PrecipRichFeatureBuilder.LeadSafeSlope(clean, 72).Should().BeApproximately(1.0 / 24, 1e-9); // {96,120}
        double.IsNaN(PrecipRichFeatureBuilder.LeadSafeSlope(clean, 96)).Should().BeTrue();   // only {120} → n<2
        double.IsNaN(PrecipRichFeatureBuilder.LeadSafeSlope(clean, 120)).Should().BeTrue();  // none > 120
    }

    [Fact]
    public void LeadSafeSlope_leak_guard_drops_the_lead_at_or_below_L()
    {
        // A spike at lead 48. Predicting at L=48 MUST exclude it (leak guard) and recover the
        // clean +1/24 trend from {72,96,120}; predicting at L=24 includes it and is pulled negative.
        var spikeAt48 = new[] { 100.0, 3.0, 4.0, 5.0 };
        PrecipRichFeatureBuilder.LeadSafeSlope(spikeAt48, 48).Should().BeApproximately(1.0 / 24, 1e-9);
        PrecipRichFeatureBuilder.LeadSafeSlope(spikeAt48, 24).Should().BeLessThan(0); // lead-48 spike included
    }

    [Fact]
    public void ComputeSlopeValues_emits_the_arm_shape_used_by_both_train_and_predict()
    {
        // This is the shared composer both BuildForLead (train) and PrecipPredictCommand
        // (predict) call — so the vector can't diverge between them (the bug the smoke caught).
        var cfg = LoadShippedConfig();
        var prov = PrecipRichFeatureBuilder.BuildSpec(cfg, 24, slopeArm: PrecipRichFeatureBuilder.SlopeArm.PerProvider);
        var agg = PrecipRichFeatureBuilder.BuildSpec(cfg, 24, slopeArm: PrecipRichFeatureBuilder.SlopeArm.Aggregate);
        var none = PrecipRichFeatureBuilder.BuildSpec(cfg, 24);
        var valid = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        // Every model: a clean +1/24 trend [2,3,4,5] at leads {48,72,96,120}.
        var trend = prov.Models.ToDictionary(m => m,
            m => new Dictionary<DateTime, double[]> { [valid] = new[] { 2.0, 3.0, 4.0, 5.0 } });

        var provVals = PrecipRichFeatureBuilder.ComputeSlopeValues(prov, trend, valid);
        provVals!.Length.Should().Be(prov.Models.Count);
        provVals.Should().OnlyContain(v => Math.Abs(v - 1.0 / 24) < 1e-9);

        var aggVals = PrecipRichFeatureBuilder.ComputeSlopeValues(agg, trend, valid);
        aggVals!.Length.Should().Be(2);
        aggVals[0].Should().BeApproximately(1.0 / 24, 1e-9); // mean
        aggVals[1].Should().BeApproximately(0.0, 1e-9);      // std (models identical)

        // No arm, or no trend data → null (ComposeRow then emits the no-slope vector).
        PrecipRichFeatureBuilder.ComputeSlopeValues(none, trend, valid).Should().BeNull();
        PrecipRichFeatureBuilder.ComputeSlopeValues(prov, null, valid).Should().BeNull();
    }

    [Fact]
    public void AggregateSlopes_is_mean_and_population_std_NaN_skipping()
    {
        var (mean, std) = PrecipRichFeatureBuilder.AggregateSlopes(new[] { 1.0, 2.0, 3.0 });
        mean.Should().BeApproximately(2.0, 1e-9);
        std.Should().BeApproximately(Math.Sqrt(2.0 / 3), 1e-9);

        PrecipRichFeatureBuilder.AggregateSlopes(new[] { double.NaN, 2.0, double.NaN }).Mean.Should().Be(2.0);
        PrecipRichFeatureBuilder.AggregateSlopes(new[] { double.NaN, 2.0 }).Std.Should().Be(0.0); // n == 1
        double.IsNaN(PrecipRichFeatureBuilder.AggregateSlopes(new[] { double.NaN, double.NaN }).Mean).Should().BeTrue();
    }
}
