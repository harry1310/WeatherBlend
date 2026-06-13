using FluentAssertions;
using WeatherBlend.Train.Oro;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// 3oni no-station-id terrain spec (2026-06-13). 3oni is 3o minus the trailing
/// oro_station_id, so the pooled model can extrapolate to the ungauged Bonehill
/// tor (an unseen station id would otherwise route down a gauge's default
/// split). These pin the feature-builder contract: the no-id variant drops
/// exactly one terrain feature (the station id), same order, and the default
/// (id-on) path is unchanged at 9.
/// </summary>
public class PrecipOroNoIdSpecTests
{
    [Fact]
    public void TerrainNames_drops_only_the_station_id_when_excluded()
    {
        var withId = PrecipRichOroFeatureBuilder.TerrainNamesFor(includeStationId: true);
        var noId = PrecipRichOroFeatureBuilder.TerrainNamesFor(includeStationId: false);

        withId.Should().HaveCount(9);
        withId.Should().Contain("oro_station_id");
        withId[^1].Should().Be("oro_station_id", "station id is the trailing terrain feature");

        noId.Should().HaveCount(8);
        noId.Should().NotContain("oro_station_id");
        noId.Should().Equal(withId.Take(8), "the 8 retained names are the id-on list minus the trailing id, same order");
    }

    [Fact]
    public void TerrainCount_is_9_with_id_and_8_without()
    {
        PrecipRichOroFeatureBuilder.TerrainCountFor(includeStationId: true).Should().Be(9);
        PrecipRichOroFeatureBuilder.TerrainCountFor(includeStationId: false).Should().Be(8);
    }
}
