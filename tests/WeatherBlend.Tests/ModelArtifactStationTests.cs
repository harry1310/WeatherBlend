using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class ModelArtifactStationTests
{
    private static string FreshRoot() =>
        Path.Combine(Path.GetTempPath(), $"wb_models_{Guid.NewGuid():N}");

    private static string Norm(string p) => p.Replace('\\', '/');

    [Fact]
    public void BuildStationVersionDir_nests_station_under_target()
    {
        var root = FreshRoot();
        var now = new DateTime(2026, 4, 23, 12, 34, 56, DateTimeKind.Utc);

        var dir = ModelArtifact.BuildStationVersionDir(root, "precipitation", "ea_bellever_dartmoor", now);

        Norm(dir).Should().EndWith("precipitation/ea_bellever_dartmoor/v2026-04-23_123456");
    }

    [Fact]
    public void UpdateStationManifest_creates_per_station_entry_and_preserves_others()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_120000");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_princetown",        "v2026-04-23_120500");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_121000");

            ModelArtifact.ListStations(root, "precipitation")
                .Should().BeEquivalentTo(new[] { "ea_bellever_dartmoor", "ea_princetown" });

            // Bellever now points to the second train, with both versions recorded.
            var bellDir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bellever_dartmoor", "current");
            Norm(bellDir).Should().EndWith("ea_bellever_dartmoor/v2026-04-23_121000");

            // Princetown untouched by the second Bellever update.
            var princetownDir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_princetown", "current");
            Norm(princetownDir).Should().EndWith("ea_princetown/v2026-04-23_120500");
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ResolveStationVersionDir_accepts_explicit_version()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_princetown", "v2026-04-23_120000");

            var dir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_princetown", "v2026-04-22_090000");

            Norm(dir).Should().EndWith("ea_princetown/v2026-04-22_090000");
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ResolveStationVersionDir_throws_when_manifest_missing()
    {
        var root = FreshRoot();

        var act = () => ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_princetown", "current");

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void ResolveStationVersionDir_throws_when_station_absent_from_manifest()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v1");

            var act = () => ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_never_trained", "current");

            act.Should().Throw<InvalidOperationException>();
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ListStations_returns_empty_when_manifest_missing()
    {
        var root = FreshRoot();

        ModelArtifact.ListStations(root, "precipitation").Should().BeEmpty();
    }
}
