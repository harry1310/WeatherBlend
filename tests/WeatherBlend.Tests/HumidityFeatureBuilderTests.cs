using FluentAssertions;
using Microsoft.Extensions.Configuration;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Humidity;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Layout pins for the humidity feature vector — added with the 2026-06-10
/// rich-feature change (+6 cross-variable ensemble means, 17→23 at N=5).
/// ComposeRow is schema-gated on "temp_mean" so the same code path must keep
/// serving pre-change 17-feature bundles until the Sunday retrain; these
/// tests pin both layouts.
/// </summary>
public class HumidityFeatureBuilderTests
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
    public void BuildSpec_appends_aux_means_at_END_so_legacy_offsets_are_preserved()
    {
        var spec = HumidityFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var n = spec.Models.Count;

        // N rh + N dp + 3 spread + 4 calendar + 6 aux = 2N + 13.
        spec.FeatureNames.Should().HaveCount(2 * n + 13);
        // Legacy block offsets unchanged (spread at 2N, calendar at 2N+3) —
        // pre-change bundles keep deserialising against the same indices.
        spec.FeatureNames[2 * n].Should().Be("rh_mean");
        spec.FeatureNames[2 * n + 3].Should().Be("hour_sin");
        // Aux block strictly at the tail, in RichAuxNames order (ComposeRow
        // fills the tail slots positionally).
        spec.FeatureNames.Skip(2 * n + 7).Should().Equal(HumidityFeatureBuilder.RichAuxNames);
    }

    [Fact]
    public void ComposeRow_packs_aux_tail_and_leaves_legacy_blocks_untouched()
    {
        var spec = HumidityFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var n = spec.Models.Count;
        var rh = Enumerable.Range(0, n).Select(i => 70.0 + i).ToArray();
        var dp = Enumerable.Range(0, n).Select(i => 8.0 + i).ToArray();
        var aux = new[] { 12.5, 3.25, 40.0, 25.0, 10.0, 6.5 };
        var valid = new DateTime(2026, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var row = HumidityFeatureBuilder.ComposeRow(spec, valid, rh, dp, era5Rh: 75.0, aux);

        row.Features.Should().HaveCount(2 * n + 13);
        // rh / dp blocks first.
        row.Features[0].Should().Be(70.0f);
        row.Features[n].Should().Be(8.0f);
        // rh spread over 70..70+(n-1): mean unchanged by the aux addition.
        row.Features[2 * n].Should().BeApproximately((float)rh.Average(), 1e-4f);
        // Aux tail at 2N+7.. in RichAuxNames order.
        for (int i = 0; i < aux.Length; i++)
            row.Features[2 * n + 7 + i].Should().Be((float)aux[i]);
        row.Label.Should().Be(75.0f);
    }

    [Fact]
    public void ComposeRow_throws_when_spec_has_aux_block_but_no_aux_supplied()
    {
        var spec = HumidityFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var n = spec.Models.Count;
        var rh = Enumerable.Repeat(70.0, n).ToArray();
        var dp = Enumerable.Repeat(8.0, n).ToArray();
        var valid = new DateTime(2026, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var act = () => HumidityFeatureBuilder.ComposeRow(spec, valid, rh, dp, era5Rh: 75.0);

        act.Should().Throw<ArgumentException>().WithParameterName("aux");
    }

    [Fact]
    public void ComposeRow_serves_legacy_17_feature_spec_without_aux()
    {
        // A pre-2026-06-10 bundle's saved feature_schema.json has no
        // temp_mean — replicate that spec shape and check the legacy path.
        var built = HumidityFeatureBuilder.BuildSpec(LoadShippedConfig(), 24);
        var legacy = new BlenderSpec
        {
            Target = built.Target,
            FeatureSet = built.FeatureSet,
            LeadHours = built.LeadHours,
            RequiredModels = built.RequiredModels,
            OptionalModels = built.OptionalModels,
            Models = built.Models,
            FeatureNames = built.FeatureNames
                .Where(f => !HumidityFeatureBuilder.RichAuxNames.Contains(f)).ToList(),
            DataSource = built.DataSource,
            Tier = built.Tier,
        };
        var n = legacy.Models.Count;
        var rh = Enumerable.Repeat(70.0, n).ToArray();
        var dp = Enumerable.Repeat(8.0, n).ToArray();
        var valid = new DateTime(2026, 6, 15, 12, 0, 0, DateTimeKind.Utc);

        var row = HumidityFeatureBuilder.ComposeRow(legacy, valid, rh, dp, era5Rh: 75.0);

        row.Features.Should().HaveCount(2 * n + 7);
        // Calendar block still terminates the vector for legacy bundles.
        row.Features[2 * n + 6].Should().BeApproximately(
            (float)Math.Cos(2.0 * Math.PI * (valid.DayOfYear - 1) / 365.0), 1e-4f);
    }
}
