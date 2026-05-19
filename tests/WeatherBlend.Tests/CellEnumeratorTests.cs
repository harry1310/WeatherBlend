using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins <see cref="CellEnumerator"/> — the single manifest walk that predict /
/// verify / render will share in Phase C. The enumerator must reproduce the
/// existing station × Active-version × lead universe, attribute each cell to
/// its pinned location, and tolerate a stale orphan Active version.
/// </summary>
public class CellEnumeratorTests
{
    private static string FreshRoot() =>
        Path.Combine(Path.GetTempPath(), $"wb_cells_{Guid.NewGuid():N}");

    /// <summary>Seed a bundle (training_metadata.json + feature_schema.json)
    /// and promote it into the manifest under <paramref name="phase"/>.</summary>
    private static void Seed(
        string root, string target, string station, string version,
        string phase, string location, params int[] leads)
    {
        var dir = Path.Combine(root, target, station, version);
        Directory.CreateDirectory(dir);

        ModelArtifact.SaveTrainingMetadata(dir, new ModelArtifact.TrainingMetadata
        {
            Version = version,
            Target = target,
            Phase = phase,
            LocationName = location,
            TrainedAtUtc = DateTime.UtcNow,
        });

        var schema = new ModelArtifact.FeatureSchema();
        foreach (var lead in leads)
            schema.Leads[lead.ToString()] = new ModelArtifact.LeadSchema
            {
                Target = target,
                LeadHours = lead,
            };
        File.WriteAllText(
            Path.Combine(dir, ModelArtifact.FeatureSchemaFileName),
            JsonSerializer.Serialize(schema));

        ModelArtifact.PromoteStationVersion(root, target, station, version, newPhase: phase);
    }

    [Fact]
    public void Enumerate_reproduces_the_station_version_lead_universe()
    {
        var root = FreshRoot();
        try
        {
            // precipitation: Bellever (Bonehill) has a 3a champion + 3c
            // challenger; Chards (Membury) has a single 3a.
            Seed(root, "precipitation", "ea_bellever_dartmoor", "v2026-05-01_100000", "3a", "bonehill_rocks", 24, 48);
            Seed(root, "precipitation", "ea_bellever_dartmoor", "v2026-05-02_100000", "3c", "bonehill_rocks", 24, 48);
            Seed(root, "precipitation", "ea_chards_snowdon_hill", "v2026-05-01_110000", "3a", "membury_devon", 24);
            // temperature: location-keyed station entry.
            Seed(root, "temperature", "bonehill_rocks", "v2026-05-01_120000", "2b", "bonehill_rocks", 24, 48, 72);

            var cells = CellEnumerator.Enumerate(root, new[] { "precipitation", "temperature" });

            // 2×2 (Bellever) + 1×1 (Chards) + 1×3 (temperature) = 8 cells.
            cells.Should().HaveCount(8);

            cells.Should().ContainEquivalentOf(new CellVersion(
                new Cell("bonehill_rocks", "ea_bellever_dartmoor", "precipitation", "3a", 24),
                "v2026-05-01_100000"));
            cells.Should().ContainEquivalentOf(new CellVersion(
                new Cell("bonehill_rocks", "ea_bellever_dartmoor", "precipitation", "3c", 48),
                "v2026-05-02_100000"));
            // Temperature is location-keyed: Station == Location.
            cells.Should().ContainEquivalentOf(new CellVersion(
                new Cell("bonehill_rocks", "bonehill_rocks", "temperature", "2b", 72),
                "v2026-05-01_120000"));

            // Chards is attributed to Membury, not the primary location.
            cells.Where(c => c.Cell.Station == "ea_chards_snowdon_hill")
                .Should().OnlyContain(c => c.Cell.Location == "membury_devon");
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Enumerate_skips_an_Active_version_with_no_directory_on_disk()
    {
        var root = FreshRoot();
        try
        {
            Seed(root, "precipitation", "ea_bellever_dartmoor", "v2026-05-01_100000", "3a", "bonehill_rocks", 24);
            // Add a phantom version to Active that was never written to disk.
            var real = ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bellever_dartmoor").ToList();
            ModelArtifact.SetStationActive(root, "precipitation", "ea_bellever_dartmoor",
                real.Append("v2026-05-09_999999").ToList());

            var cells = CellEnumerator.Enumerate(root, new[] { "precipitation" });

            // Only the real version's single lead survives.
            cells.Should().ContainSingle()
                .Which.Version.Should().Be("v2026-05-01_100000");
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
