using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class ModelArtifactStationTests
{
    private static string FreshRoot() =>
        Path.Combine(Path.GetTempPath(), $"wb_models_{Guid.NewGuid():N}");

    private static string Norm(string p) => p.Replace('\\', '/');

    /// <summary>Create a station version dir with a training_metadata.json
    /// carrying the given phase + location. The promote helpers read the
    /// pinned location back off disk, and the derived-champion resolver
    /// (<see cref="ModelArtifact.ResolveStationChampionVersion"/>) reads the
    /// phase — so a version that should resolve as champion must be seeded
    /// with a phase in the target's lineup. Precipitation is a fallback chain
    /// (3o → 3c → 3a, champion-first), so the highest-priority phase a station
    /// has a bundle of wins.</summary>
    private static void SeedBundle(
        string root, string target, string station, string version,
        string phase, string location = "bonehill_rocks")
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
    }

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
            // Seed bundle metadata (phase 3a = precipitation champion) so the
            // derived-champion "current" resolver can identify each version.
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_120000", "3a");
            SeedBundle(root, "precipitation", "ea_bovey_tracey",        "v2026-04-23_120500", "3a");
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_121000", "3a");

            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_120000");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bovey_tracey",        "v2026-04-23_120500");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v2026-04-23_121000");

            ModelArtifact.ListStations(root, "precipitation")
                .Should().BeEquivalentTo(new[] { "ea_bellever_dartmoor", "ea_bovey_tracey" });

            // Bellever's champion is now the second train (UpdateStationManifest
            // resets Active to the single new version).
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
    public void UpdateStationManifest_seeds_Active_as_single_entry()
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
    public void SetStationActive_replaces_list_and_champion_follows_fallback_chain()
    {
        var root = FreshRoot();
        try
        {
            // 3a-lean + 3c-rich both Active; no 3o bundle for this station.
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v3a_lean", "3a");
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v3c_rich", "3c");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v3a_lean");
            ModelArtifact.AppendStationVersion(root, "precipitation", "ea_bellever_dartmoor", "v3c_rich");
            ModelArtifact.SetStationActive(root, "precipitation", "ea_bellever_dartmoor",
                new[] { "v3a_lean", "v3c_rich" });

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Equal("v3a_lean", "v3c_rich");

            // The champion is derived by walking the precipitation lineup
            // champion-first (3o → 3c → 3a, 2026-06-06) and returning the
            // first phase this station has an Active bundle of. With no 3o
            // bundle here, 3c wins over 3a — exactly the per-station fallback
            // that makes Bonehill 3o, Membury 3c, and Sennen 3a.
            ModelArtifact.ResolveStationChampionVersion(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Be("v3c_rich");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationChampionVersion_prefers_3o_over_3c_and_3a()
    {
        var root = FreshRoot();
        try
        {
            // Full Bonehill stack: 3o (champion phase) + 3c + 3a all Active.
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v3a_lean", "3a");
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v3c_rich", "3c");
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v3o_oro", "3o");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bellever_dartmoor", "v3a_lean");
            ModelArtifact.AppendStationVersion(root, "precipitation", "ea_bellever_dartmoor", "v3c_rich");
            ModelArtifact.AppendStationVersion(root, "precipitation", "ea_bellever_dartmoor", "v3o_oro");
            ModelArtifact.SetStationActive(root, "precipitation", "ea_bellever_dartmoor",
                new[] { "v3a_lean", "v3c_rich", "v3o_oro" });

            // 3o is first in the lineup, so it wins outright.
            ModelArtifact.ResolveStationChampionVersion(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Be("v3o_oro");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationChampionVersion_falls_back_to_3a_when_only_lean_trained()
    {
        var root = FreshRoot();
        try
        {
            // A station that only has the universal 3a fallback bundle (no
            // richer 3o/3c trained for it yet) resolves its champion to 3a.
            SeedBundle(root, "precipitation", "ea_trengwainton", "v3a_lean", "3a", "sennen_cove");
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_trengwainton", "v3a_lean");

            ModelArtifact.ResolveStationChampionVersion(root, "precipitation", "ea_trengwainton")
                .Should().Be("v3a_lean");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationActive_returns_empty_when_Active_cleared()
    {
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bovey_tracey", "v_legacy");
            // Clearing Active demotes the station — there is no Current
            // pointer to fall back to any more.
            ModelArtifact.SetStationActive(root, "precipitation", "ea_bovey_tracey", Array.Empty<string>());

            ModelArtifact.ResolveStationActive(root, "precipitation", "ea_bovey_tracey")
                .Should().BeEmpty();
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
    public void AppendStationVersion_creates_entry_with_empty_Active()
    {
        // Mirrors the temperature AppendVersion semantics: registers a version
        // in history without adding it to Active. A station with no Active
        // version has no champion.
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

    // ----- Phase B: StationEntry.Location -------------------------------

    [Fact]
    public void PromoteStationVersion_pins_Location_from_bundle_metadata()
    {
        var root = FreshRoot();
        try
        {
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a", "bonehill_rocks");
            ModelArtifact.PromoteStationVersion(
                root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a");

            ModelArtifact.ResolveStationLocation(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Be("bonehill_rocks");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void PromoteStationVersion_throws_when_a_later_bundle_swaps_location()
    {
        var root = FreshRoot();
        try
        {
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a", "bonehill_rocks");
            ModelArtifact.PromoteStationVersion(
                root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a");

            // A bundle whose metadata claims a different location for the
            // same station slot — a misrouted trainer; promote must refuse.
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v2_phase3a", "3a", "membury_devon");
            var act = () => ModelArtifact.PromoteStationVersion(
                root, "precipitation", "ea_bellever_dartmoor", "v2_phase3a", "3a");

            act.Should().Throw<InvalidOperationException>()
               .WithMessage("*pinned to*bonehill_rocks*");

            // The refused promote left the manifest intact.
            ModelArtifact.ResolveStationLocation(root, "precipitation", "ea_bellever_dartmoor")
                .Should().Be("bonehill_rocks");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationActive_with_location_returns_active_when_location_matches()
    {
        var root = FreshRoot();
        try
        {
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a", "bonehill_rocks");
            ModelArtifact.PromoteStationVersion(
                root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a");

            ModelArtifact.ResolveStationActive(
                    root, "precipitation", "bonehill_rocks", "ea_bellever_dartmoor")
                .Should().Equal("v1_phase3a");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationActive_with_location_throws_on_location_mismatch()
    {
        var root = FreshRoot();
        try
        {
            SeedBundle(root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a", "bonehill_rocks");
            ModelArtifact.PromoteStationVersion(
                root, "precipitation", "ea_bellever_dartmoor", "v1_phase3a", "3a");

            var act = () => ModelArtifact.ResolveStationActive(
                root, "precipitation", "membury_devon", "ea_bellever_dartmoor");

            act.Should().Throw<InvalidOperationException>();
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void ResolveStationActive_with_location_tolerates_legacy_unpinned_entry()
    {
        // UpdateStationManifest doesn't pin a location — a pre-Phase-B
        // entry has Location="". The location-checked resolve must not
        // throw against such an entry (nothing to verify).
        var root = FreshRoot();
        try
        {
            ModelArtifact.UpdateStationManifest(root, "precipitation", "ea_bovey_tracey", "v_legacy");

            ModelArtifact.ResolveStationActive(
                    root, "precipitation", "bonehill_rocks", "ea_bovey_tracey")
                .Should().Equal("v_legacy");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void PromoteStationVersion_evicts_retired_phase_but_keeps_current_sibling()
    {
        // Reproduces the 2026-06-09 stale-element bug: a pre-rename `lean-wind`
        // version stranded in Active because ComposeActive's exact phase-string
        // match treated it as a different phase and preserved it forever.
        // Passing the target's current phase lineup (knownPhases) must evict the
        // retired `lean-wind` while preserving the legitimate `wind_speed_lgb`
        // sibling (a different but CURRENT phase).
        var root = FreshRoot();
        try
        {
            var lineup = new HashSet<string>(StringComparer.Ordinal)
                { "wind", "wind_speed_lgb", "wind_blend" };

            // Old stranded state: lean-wind promoted with no lineup (legacy).
            SeedBundle(root, "wind", "bonehill_rocks", "v1_lean-wind", "lean-wind");
            ModelArtifact.PromoteStationVersion(
                root, "wind", "bonehill_rocks", "v1_lean-wind", "lean-wind");

            // Current sibling phase lands — lean-wind is retired (not in lineup) → evicted.
            SeedBundle(root, "wind", "bonehill_rocks", "v2_wind_speed_lgb", "wind_speed_lgb");
            ModelArtifact.PromoteStationVersion(
                root, "wind", "bonehill_rocks", "v2_wind_speed_lgb", "wind_speed_lgb",
                knownPhases: lineup);

            // Champion `wind` retrains — wind_speed_lgb is a CURRENT sibling, preserved.
            SeedBundle(root, "wind", "bonehill_rocks", "v3_wind", "wind");
            ModelArtifact.PromoteStationVersion(
                root, "wind", "bonehill_rocks", "v3_wind", "wind",
                knownPhases: lineup);

            ModelArtifact.ResolveStationActive(root, "wind", "bonehill_rocks")
                .Should().BeEquivalentTo("v2_wind_speed_lgb", "v3_wind");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void PromoteStationVersion_without_knownPhases_preserves_retired_phase()
    {
        // Backward-compat guard: every non-element caller passes no lineup, so the
        // retired-phase eviction must stay OFF for them — a different-phase entry
        // is preserved exactly as before (this is the pre-fix behaviour, retained
        // for precipitation / temperature / dry-window).
        var root = FreshRoot();
        try
        {
            SeedBundle(root, "wind", "bonehill_rocks", "v1_lean-wind", "lean-wind");
            ModelArtifact.PromoteStationVersion(
                root, "wind", "bonehill_rocks", "v1_lean-wind", "lean-wind");

            SeedBundle(root, "wind", "bonehill_rocks", "v2_wind", "wind");
            ModelArtifact.PromoteStationVersion(
                root, "wind", "bonehill_rocks", "v2_wind", "wind");  // no knownPhases

            ModelArtifact.ResolveStationActive(root, "wind", "bonehill_rocks")
                .Should().BeEquivalentTo("v1_lean-wind", "v2_wind");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
    }
}
