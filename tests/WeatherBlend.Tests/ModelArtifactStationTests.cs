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
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bovey_tracey",        "v2026-04-23_120500");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_121000");

            ModelArtifact.ListStations(root, "precipitation")
                .Should().BeEquivalentTo(new[] { "ea_bellever_dartmoor", "ea_bovey_tracey" });

            // Bellever now points to the second train, with both versions recorded.
            var bellDir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bellever_dartmoor", "current");
            Norm(bellDir).Should().EndWith("ea_bellever_dartmoor/v2026-04-23_121000");

            // Bovey Tracey untouched by the second Bellever update.
            var boveyDir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bovey_tracey", "current");
            Norm(boveyDir).Should().EndWith("ea_bovey_tracey/v2026-04-23_120500");
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
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bovey_tracey", "v2026-04-23_120000");

            var dir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bovey_tracey", "v2026-04-22_090000");

            Norm(dir).Should().EndWith("ea_bovey_tracey/v2026-04-22_090000");
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

        var act = () => ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bovey_tracey", "current");

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

    [Fact]
    public void UpdateStationManifest_seeds_Active_as_single_current_for_backcompat()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-22_071842");

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Equal("v2026-04-22_071842");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void SetStationActive_replaces_list_independently_of_Current()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v3a_lean");
            ModelArtifact.AppendStationVersion(root, "precipitation", "ea_bellever_dartmoor", "v3c_rich");
            ModelArtifact.SetStationActive(root, "precipitation", "ea_bellever_dartmoor",
                new[] { "v3a_lean", "v3c_rich" });

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Equal("v3a_lean", "v3c_rich");

            // Current is unchanged by SetStationActive — the lean 3a remains the champion.
            var dir = ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bellever_dartmoor", "current");
            Norm(dir).Should().EndWith("ea_bellever_dartmoor/v3a_lean");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationActive_falls_back_to_Current_when_Active_empty()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bovey_tracey", "v_legacy");
            // Simulate a legacy manifest: clear Active so we fall back to [Current].
            ModelArtifact.SetStationActive(root, "precipitation", "ea_bovey_tracey", Array.Empty<string>());

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bovey_tracey")
                .Should().Equal("v_legacy");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationActive_returns_empty_when_station_absent()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v1");

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_never_trained")
                .Should().BeEmpty();
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void AppendStationVersion_creates_entry_with_empty_Current_and_Active()
    {
        // Mirrors the temperature AppendVersion semantics: registers a version in history
        // without making it the champion or adding it to Active.
        var root = FreshRoot();
        try
        {
            ModelArtifact.AppendStationVersion(root, "precipitation", "ea_bovey_tracey", "v_future_challenger");

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bovey_tracey")
                .Should().BeEmpty();
            var act = () => ModelArtifact.ResolveStationVersionDir(root, "precipitation", "ea_bovey_tracey", "current");
            act.Should().Throw<InvalidOperationException>();
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }
}
