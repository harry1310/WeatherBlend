using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PredictCommandTests
{
    [Fact]
    public void HashFeatures_is_deterministic_for_identical_inputs()
    {
        var row = FeatureBuilder.ComposeRow(
            new DateTime(2026, 4, 22, 20, 0, 0, DateTimeKind.Utc),
            new[] { 9.1, 9.3, 9.5, 9.4, 9.2, 9.6 },
            windDirMeanDeg: 225.0,
            era5Temp: double.NaN);

        var h1 = PredictCommand.HashFeatures(row);
        var h2 = PredictCommand.HashFeatures(row);

        h1.Should().Be(h2);
        h1.Should().HaveLength(64);        // SHA-256 hex
        h1.Should().MatchRegex("^[0-9a-f]+$");
    }

    [Fact]
    public void HashFeatures_differs_when_any_feature_changes()
    {
        var baseRow = FeatureBuilder.ComposeRow(
            new DateTime(2026, 4, 22, 20, 0, 0, DateTimeKind.Utc),
            new[] { 9.1, 9.3, 9.5, 9.4, 9.2, 9.6 },
            windDirMeanDeg: 225.0,
            era5Temp: double.NaN);

        // Bump one model temp by 0.01 — everything downstream (mean/std/range)
        // and therefore the feature vector changes, so the hash must too.
        var bumpedRow = FeatureBuilder.ComposeRow(
            new DateTime(2026, 4, 22, 20, 0, 0, DateTimeKind.Utc),
            new[] { 9.11, 9.3, 9.5, 9.4, 9.2, 9.6 },
            windDirMeanDeg: 225.0,
            era5Temp: double.NaN);

        PredictCommand.HashFeatures(baseRow)
            .Should().NotBe(PredictCommand.HashFeatures(bumpedRow));
    }

    [Fact]
    public void HashFeatures_does_not_depend_on_era5_or_wind_dir()
    {
        // Era5Temp and WindDirMean are NOT in the 13-feature set — they're stored
        // on TrainingRow for downstream use but should not affect the hash.
        var rowA = FeatureBuilder.ComposeRow(
            new DateTime(2026, 4, 22, 20, 0, 0, DateTimeKind.Utc),
            new[] { 9.1, 9.3, 9.5, 9.4, 9.2, 9.6 },
            windDirMeanDeg: 10.0,
            era5Temp: 5.0);

        var rowB = FeatureBuilder.ComposeRow(
            new DateTime(2026, 4, 22, 20, 0, 0, DateTimeKind.Utc),
            new[] { 9.1, 9.3, 9.5, 9.4, 9.2, 9.6 },
            windDirMeanDeg: 270.0,
            era5Temp: double.NaN);

        PredictCommand.HashFeatures(rowA).Should().Be(PredictCommand.HashFeatures(rowB));
    }
}
