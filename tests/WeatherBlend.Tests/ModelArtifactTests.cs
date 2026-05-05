using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class ModelArtifactTests : IDisposable
{
    private readonly string _root;

    public ModelArtifactTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-artifact-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    // ---- UpdateManifest --------------------------------------------------------

    [Fact]
    public void UpdateManifest_creates_manifest_on_first_call()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v2026-04-21_201231");

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        File.Exists(path).Should().BeTrue();

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;
        manifest.Target.Should().Be("temperature");
        manifest.Current.Should().Be("v2026-04-21_201231");
        manifest.Versions.Should().ContainSingle().Which.Should().Be("v2026-04-21_201231");
    }

    [Fact]
    public void UpdateManifest_appends_new_version_and_advances_Current()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.UpdateManifest(_root, "temperature", "v2");
        ModelArtifact.UpdateManifest(_root, "temperature", "v3");

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;

        manifest.Current.Should().Be("v3");
        manifest.Versions.Should().ContainInOrder("v1", "v2", "v3");
    }

    [Fact]
    public void UpdateManifest_does_not_duplicate_an_already_listed_version()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.UpdateManifest(_root, "temperature", "v2");
        ModelArtifact.UpdateManifest(_root, "temperature", "v1"); // re-pointing Current back to v1

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;

        manifest.Current.Should().Be("v1");
        manifest.Versions.Should().Equal("v1", "v2");
    }

    [Fact]
    public void UpdateManifest_leaves_no_tmp_file_behind()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");

        var dir = Path.Combine(_root, "temperature");
        Directory.EnumerateFiles(dir, "*.tmp").Should().BeEmpty(
            "temp file should be moved over the manifest, not left behind");
    }

    // ---- Active-list semantics (champion/challenger) ---------------------------

    [Fact]
    public void UpdateManifest_resets_Active_to_single_entry_for_legacy_single_version_flow()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().ContainSingle().Which.Should().Be("v1");
    }

    [Fact]
    public void AppendVersion_adds_to_Versions_without_changing_Current_or_Active()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.AppendVersion(_root, "temperature", "v2-challenger");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Current.Should().Be("v1");
        manifest.Versions.Should().Equal("v1", "v2-challenger");
        manifest.Active.Should().ContainSingle().Which.Should().Be("v1",
            "AppendVersion must not touch Active — training a challenger doesn't promote it");
    }

    [Fact]
    public void SetActive_overrides_Active_list_without_touching_Current()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.AppendVersion(_root, "temperature", "v2");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v1", "v2" });

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Current.Should().Be("v1");
        manifest.Active.Should().Equal("v1", "v2");
    }

    [Fact]
    public void SetActive_deduplicates_repeated_entries()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v1", "v1", "v2", "v2" });

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().Equal("v1", "v2");
    }

    [Fact]
    public void ResolveActive_returns_the_explicit_list_when_set()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.AppendVersion(_root, "temperature", "v2");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v1", "v2" });

        ModelArtifact.ResolveActive(_root, "temperature").Should().Equal("v1", "v2");
    }

    [Fact]
    public void ResolveActive_falls_back_to_Current_when_Active_empty_legacy_manifest()
    {
        // Simulate a manifest written by an older build that predates the Active field.
        var dir = Path.Combine(_root, "temperature");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, ModelArtifact.ManifestFileName),
            """{"Target":"temperature","Current":"v-legacy","Versions":["v-legacy"],"Active":[],"Stations":{}}""");

        ModelArtifact.ResolveActive(_root, "temperature").Should()
            .ContainSingle().Which.Should().Be("v-legacy");
    }

    [Fact]
    public void ResolveActive_returns_empty_when_manifest_absent()
    {
        ModelArtifact.ResolveActive(_root, "temperature").Should().BeEmpty();
    }

    // ---- ResolveVersionDir ------------------------------------------------------

    [Fact]
    public void ResolveVersionDir_returns_explicit_version_path_without_consulting_manifest()
    {
        // No manifest at all — explicit lookup should still work.
        var dir = ModelArtifact.ResolveVersionDir(_root, "temperature", "v2026-04-21_201231");
        dir.Should().Be(Path.Combine(_root, "temperature", "v2026-04-21_201231"));
    }

    [Fact]
    public void ResolveVersionDir_resolves_current_via_manifest()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.UpdateManifest(_root, "temperature", "v2");

        var dir = ModelArtifact.ResolveVersionDir(_root, "temperature", "current");
        dir.Should().Be(Path.Combine(_root, "temperature", "v2"));
    }

    [Fact]
    public void ResolveVersionDir_throws_when_current_requested_without_manifest()
    {
        var act = () => ModelArtifact.ResolveVersionDir(_root, "temperature", "current");
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*train a model first*");
    }

    // ---- TrainingMetadata round-trip -------------------------------------------

    [Fact]
    public void TrainingMetadata_round_trips_through_disk()
    {
        var versionDir = Path.Combine(_root, "temperature", "v-rt");
        Directory.CreateDirectory(versionDir);

        var original = new ModelArtifact.TrainingMetadata
        {
            Version = "v-rt",
            Target = "temperature",
            Phase = "2b",
            DataSource = "previous_runs_api",
            TrainedAtUtc = new DateTime(2026, 4, 21, 20, 12, 31, DateTimeKind.Utc),
            Hyperparameters = new Dictionary<string, object>
            {
                ["learningRate"] = 0.05,
                ["numberOfLeaves"] = 31,
            },
            TestMae = new Dictionary<string, double>
            {
                ["lead_24h_blend"] = 1.234,
                ["lead_48h_blend"] = 1.511,
                ["lead_72h_blend"] = 1.802,
            },
            DeviationsFromBrief = new List<string> { "L2 objective", "no monotone constraints" },
            PerLead = new Dictionary<string, ModelArtifact.PerLeadStats>
            {
                ["24"] = new ModelArtifact.PerLeadStats
                {
                    LeadHours = 24,
                    DataRangeTrain = "2024-06-01 → 2025-10-15",
                    TrainRows = 6000, ValRows = 1000, TestRows = 1000,
                    TestCalendarMonths = 6,
                    BestSingle = "temp_ecmwf",
                    BestSingleValMae = 1.60,
                    BlendTestMae = 1.23, BlendTestRmse = 1.70, BlendTestBias = -0.05,
                },
            },
        };

        ModelArtifact.SaveTrainingMetadata(versionDir, original);
        var reloaded = ModelArtifact.LoadTrainingMetadata(versionDir);

        reloaded.Version.Should().Be(original.Version);
        reloaded.Phase.Should().Be(original.Phase);
        reloaded.TrainedAtUtc.Should().Be(original.TrainedAtUtc);
        reloaded.TestMae.Should().BeEquivalentTo(original.TestMae);
        reloaded.DeviationsFromBrief.Should().Equal(original.DeviationsFromBrief);
        reloaded.PerLead.Should().ContainKey("24");
        reloaded.PerLead["24"].BestSingle.Should().Be("temp_ecmwf");
        reloaded.PerLead["24"].BlendTestMae.Should().BeApproximately(1.23, 1e-9);
    }

    [Fact]
    public void LoadTrainingMetadata_throws_for_missing_file()
    {
        var versionDir = Path.Combine(_root, "temperature", "v-missing");
        Directory.CreateDirectory(versionDir);

        var act = () => ModelArtifact.LoadTrainingMetadata(versionDir);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Missing training metadata*");
    }

    // ---- SavePerLeadFeatureImportance / LoadPerLeadFeatureImportance ----------

    [Fact]
    public void PerLeadFeatureImportance_round_trips_with_sort_order_preserved()
    {
        var versionDir = Path.Combine(_root, "temperature", "v-fi");
        Directory.CreateDirectory(versionDir);

        var byLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>
        {
            [24] = new[] { ("temp_ecmwf", 0.42), ("temp_gfs", 0.31), ("hour_sin", 0.04) },
            [48] = new[] { ("temp_ecmwf", 0.50), ("temp_mean", 0.20) },
            [72] = Array.Empty<(string, double)>(),
        };

        ModelArtifact.SavePerLeadFeatureImportance(versionDir, byLead);
        var loaded = ModelArtifact.LoadPerLeadFeatureImportance(versionDir);

        loaded.Keys.Should().BeEquivalentTo(new[] { 24, 48, 72 });
        loaded[24].Select(t => t.Name).Should().Equal("temp_ecmwf", "temp_gfs", "hour_sin");
        loaded[24][0].Gain.Should().BeApproximately(0.42, 1e-9);
        loaded[48].Should().HaveCount(2);
        loaded[72].Should().BeEmpty();
    }

    [Fact]
    public void LoadPerLeadFeatureImportance_returns_empty_for_missing_file()
    {
        var versionDir = Path.Combine(_root, "temperature", "v-none");
        Directory.CreateDirectory(versionDir);

        var loaded = ModelArtifact.LoadPerLeadFeatureImportance(versionDir);
        loaded.Should().BeEmpty();
    }

    // ---- BuildVersionDir --------------------------------------------------------

    [Fact]
    public void BuildVersionDir_encodes_utc_timestamp_and_uses_forward_slashes()
    {
        var ts = new DateTime(2026, 4, 21, 20, 12, 31, DateTimeKind.Utc);
        var dir = ModelArtifact.BuildVersionDir("data/models", "temperature", ts);

        dir.Should().Be("data/models/temperature/v2026-04-21_201231");
        dir.Should().NotContain("\\", "paths are always normalised to forward slashes");
    }

    [Fact]
    public void BuildVersionDir_appends_suffix_when_provided()
    {
        var ts = new DateTime(2026, 4, 21, 20, 12, 31, DateTimeKind.Utc);
        var dir = ModelArtifact.BuildVersionDir("data/models", "temperature", ts, suffix: "retrain");

        dir.Should().Be("data/models/temperature/v2026-04-21_201231_retrain");
    }

    // ---- Concurrency: atomic rename + file lock --------------------------------

    /// <summary>
    /// Reader running concurrently with a writer should never observe a missing
    /// or half-written manifest. Regression for the prior Delete-then-Move
    /// pattern that left a microsecond window where the file didn't exist.
    /// </summary>
    [Fact]
    public void Manifest_is_always_present_during_concurrent_writes()
    {
        // Seed so the reader has something to read before the writer starts.
        ModelArtifact.UpdateManifest(_root, "temperature", "v0");

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        var observedMissing = 0;
        var observedReads = 0;

        // Reader thread — hammers ResolveActive while the writer churns.
        var reader = Task.Run(() =>
        {
            while (!cts.IsCancellationRequested)
            {
                try
                {
                    var active = ModelArtifact.ResolveActive(_root, "temperature");
                    if (active.Count == 0) Interlocked.Increment(ref observedMissing);
                    Interlocked.Increment(ref observedReads);
                }
                catch (IOException) { Interlocked.Increment(ref observedMissing); }
            }
        });

        // Writer thread — bumps Current 200× in tight succession.
        var writer = Task.Run(() =>
        {
            for (int i = 1; i <= 200 && !cts.IsCancellationRequested; i++)
            {
                ModelArtifact.UpdateManifest(_root, "temperature", $"v{i}");
            }
        });

        writer.Wait();
        cts.Cancel();
        reader.Wait();

        observedReads.Should().BeGreaterThan(50,
            "reader should have completed many reads while the writer was active");
        observedMissing.Should().Be(0,
            "manifest should never appear missing or half-written under concurrent updates");
    }

    /// <summary>
    /// N concurrent threads each appending a unique version should produce a
    /// final manifest that contains all N versions — no lost updates. Regression
    /// for the read-mutate-write trample race fixed by the file lock.
    /// </summary>
    [Fact]
    public void Concurrent_AppendVersion_does_not_lose_updates()
    {
        const int N = 30;

        var threads = Enumerable.Range(0, N).Select(i => new Thread(() =>
        {
            ModelArtifact.AppendVersion(_root, "temperature", $"v-thread-{i:D2}");
        })).ToList();

        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join();

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;

        manifest.Versions.Should().HaveCount(N);
        for (int i = 0; i < N; i++)
            manifest.Versions.Should().Contain($"v-thread-{i:D2}");
    }

    // ---- PromoteVersion (idempotent same-phase replacement) ------------------
    //
    // The promote helpers fix the load-bearing footgun in
    // UpdateManifest / UpdateStationManifest: the older helpers reset Active to
    // [new], which silently kicked any active challenger phase out of the rotation
    // on every champion retrain. The promote variants read each existing Active
    // entry's training_metadata.Phase and only replace entries with the same
    // phase as the new version, leaving all other phases (champions or
    // challengers) untouched. Tests below pin both the same-phase replacement
    // rule and the "preserve other phases" invariant.

    private static void WritePhaseMetadata(string dir, string version, string phase)
    {
        Directory.CreateDirectory(dir);
        var meta = new ModelArtifact.TrainingMetadata
        {
            Version = version,
            Target = "temperature",
            Phase = phase,
            DataSource = "test",
            TrainedAtUtc = DateTime.UtcNow,
        };
        ModelArtifact.SaveTrainingMetadata(dir, meta);
    }

    [Fact]
    public void PromoteVersionAsChampion_replaces_same_phase_and_preserves_others()
    {
        // Setup: 2b champion (v_base_old) + 2c challenger (v_2c).
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base_old"), "v_base_old", "2b");
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c"),       "v_2c",       "2c");
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base_old");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base_old", "v_2c" });

        // Retrain 2b → v_base_new. Expected: v_base_old is dropped from Active
        // (replaced by v_base_new), v_2c stays.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base_new"), "v_base_new", "2b");
        ModelArtifact.PromoteVersionAsChampion(_root, "temperature", "v_base_new", newPhase: "2b");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Current.Should().Be("v_base_new");
        manifest.Active.Should().BeEquivalentTo(new[] { "v_2c", "v_base_new" });   // 2c preserved
        manifest.Versions.Should().Contain(new[] { "v_base_old", "v_base_new" });
    }

    [Fact]
    public void PromoteVersionAsChallenger_replaces_same_phase_and_does_not_change_Current()
    {
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"),    "v_base",    "2b");
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_old"),  "v_2c_old",  "2c");
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2c_old" });

        // Retrain 2c → v_2c_new. Expected: v_2c_old replaced by v_2c_new in
        // Active, v_base preserved, Current still v_base.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_new"), "v_2c_new", "2c");
        ModelArtifact.PromoteVersionAsChallenger(_root, "temperature", "v_2c_new", newPhase: "2c");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Current.Should().Be("v_base");                                  // unchanged
        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2c_new" }); // 2c rotated
    }

    [Fact]
    public void PromoteVersion_is_idempotent_across_repeated_same_phase_retrains()
    {
        // Run two 2c retrains back-to-back. Active should converge to one 2c
        // entry per run, not accumulate stale 2c versions across runs.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"),  "v_base",  "2b");
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");

        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_a"), "v_2c_a", "2c");
        ModelArtifact.PromoteVersionAsChallenger(_root, "temperature", "v_2c_a", newPhase: "2c");

        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_b"), "v_2c_b", "2c");
        ModelArtifact.PromoteVersionAsChallenger(_root, "temperature", "v_2c_b", newPhase: "2c");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        // Active has exactly one 2c (the latest), not both.
        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2c_b" });
        // Versions accumulates both 2c entries (history is preserved).
        manifest.Versions.Should().Contain(new[] { "v_base", "v_2c_a", "v_2c_b" });
    }

    [Fact]
    public void PromoteVersion_preserves_entry_with_unreadable_metadata()
    {
        // An Active entry whose training_metadata is missing or malformed has
        // unknown phase. The promote helper PRESERVES it rather than silently
        // dropping — caller can clean up explicitly via SetActive if intended.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"), "v_base", "2b");
        // v_orphan has no metadata file.
        Directory.CreateDirectory(Path.Combine(_root, "temperature", "v_orphan"));
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_orphan" });

        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base_new"), "v_base_new", "2b");
        ModelArtifact.PromoteVersionAsChampion(_root, "temperature", "v_base_new", newPhase: "2b");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        // v_orphan stays. Only v_base (matching phase 2b) is replaced.
        manifest.Active.Should().BeEquivalentTo(new[] { "v_orphan", "v_base_new" });
    }

    [Fact]
    public void PromoteStationVersionAsChampion_replaces_same_phase_per_station_only()
    {
        // Phase 3a champion + 3c challenger at one station. Retraining 3a
        // replaces only the 3a entry, leaves 3c — and doesn't touch the
        // OTHER station's manifest entry at all.
        var stationA = "ea_bellever_dartmoor";
        var stationB = "ea_princetown";

        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationA, "v_3a_old"), "v_3a_old", "3a");
        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationA, "v_3c"),     "v_3c",     "3c");
        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationB, "v_3a_b"),   "v_3a_b",   "3a");

        ModelArtifact.UpdateStationManifest(_root, "precipitation", stationA, "v_3a_old");
        ModelArtifact.SetStationActive(_root, "precipitation", stationA, new[] { "v_3a_old", "v_3c" });
        ModelArtifact.UpdateStationManifest(_root, "precipitation", stationB, "v_3a_b");

        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationA, "v_3a_new"), "v_3a_new", "3a");
        ModelArtifact.PromoteStationVersionAsChampion(
            _root, "precipitation", stationA, "v_3a_new", newPhase: "3a");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "precipitation", ModelArtifact.ManifestFileName)))!;

        manifest.Stations[stationA].Current.Should().Be("v_3a_new");
        manifest.Stations[stationA].Active.Should().BeEquivalentTo(new[] { "v_3c", "v_3a_new" });
        // Other station untouched.
        manifest.Stations[stationB].Current.Should().Be("v_3a_b");
        manifest.Stations[stationB].Active.Should().BeEquivalentTo(new[] { "v_3a_b" });
    }

    [Fact]
    public void PromoteStationVersionAsChallenger_replaces_same_phase_3e_idempotently()
    {
        // Direct regression for the dry-window 3e use case the user shipped:
        // re-running 3e training should replace the previous 3e in Active
        // without disturbing the 3b champion entry.
        var station = "ea_bellever_dartmoor/window_4h";

        WritePhaseMetadata(Path.Combine(_root, "dry_window", station, "v_3b"),    "v_3b",    "3b");
        WritePhaseMetadata(Path.Combine(_root, "dry_window", station, "v_3e_a"),  "v_3e_a",  "3e");
        ModelArtifact.UpdateStationManifest(_root, "dry_window", station, "v_3b");
        ModelArtifact.SetStationActive(_root, "dry_window", station, new[] { "v_3b", "v_3e_a" });

        WritePhaseMetadata(Path.Combine(_root, "dry_window", station, "v_3e_b"), "v_3e_b", "3e");
        ModelArtifact.PromoteStationVersionAsChallenger(
            _root, "dry_window", station, "v_3e_b", newPhase: "3e");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "dry_window", ModelArtifact.ManifestFileName)))!;

        manifest.Stations[station].Current.Should().Be("v_3b");                          // unchanged
        manifest.Stations[station].Active.Should().BeEquivalentTo(new[] { "v_3b", "v_3e_b" });
    }

    /// <summary>
    /// Concurrent SetActive + AppendVersion must compose: every appended version
    /// survives, and the final SetActive winner is one of the values written.
    /// Mirrors the chained train-then-calibrate flow that previously trampled.
    /// </summary>
    [Fact]
    public void Concurrent_SetActive_and_AppendVersion_compose_without_loss()
    {
        const int N = 20;
        ModelArtifact.UpdateManifest(_root, "temperature", "v-base");

        var threads = new List<Thread>();
        for (int i = 0; i < N; i++)
        {
            var idx = i;
            threads.Add(new Thread(() =>
            {
                ModelArtifact.AppendVersion(_root, "temperature", $"v-{idx:D2}");
                ModelArtifact.SetActive(_root, "temperature", new[] { "v-base", $"v-{idx:D2}" });
            }));
        }

        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join();

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;

        // All N appended versions plus v-base must be present.
        manifest.Versions.Should().Contain("v-base");
        for (int i = 0; i < N; i++)
            manifest.Versions.Should().Contain($"v-{i:D2}");

        // Active is whichever thread wrote last; must contain v-base + exactly one v-{i}.
        manifest.Active.Should().HaveCount(2);
        manifest.Active.Should().Contain("v-base");
        manifest.Active.Should().ContainSingle(v => v.StartsWith("v-", StringComparison.Ordinal) && v != "v-base");
    }

    // ---- ChampionByLead -------------------------------------------------------

    [Fact]
    public void ResolveChampionForLead_falls_back_to_Current_when_no_per_lead_override()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v-2b");
        ModelArtifact.ResolveChampionForLead(_root, "temperature", 24).Should().Be("v-2b");
        ModelArtifact.ResolveChampionForLead(_root, "temperature", 12).Should().Be("v-2b");
    }

    [Fact]
    public void SetChampionForLead_pins_per_lead_override_without_touching_Current()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v-2b");
        ModelArtifact.SetChampionForLead(_root, "temperature", 12, "v-2d");

        ModelArtifact.ResolveChampionForLead(_root, "temperature", 12).Should().Be("v-2d");
        ModelArtifact.ResolveChampionForLead(_root, "temperature", 24).Should().Be("v-2b");

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;
        manifest.Current.Should().Be("v-2b"); // Current untouched
        manifest.ChampionByLead.Should().ContainKey(12).WhoseValue.Should().Be("v-2d");
    }

    [Fact]
    public void SetChampionForLead_with_empty_string_clears_the_pin()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v-2b");
        ModelArtifact.SetChampionForLead(_root, "temperature", 12, "v-2d");
        ModelArtifact.SetChampionForLead(_root, "temperature", 12, "");

        ModelArtifact.ResolveChampionForLead(_root, "temperature", 12).Should().Be("v-2b");

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;
        manifest.ChampionByLead.Should().NotContainKey(12);
    }

    [Fact]
    public void ResolveChampionForLead_pure_overload_works_against_in_memory_manifest()
    {
        var manifest = new ModelArtifact.Manifest
        {
            Current = "v-2b",
            ChampionByLead = new Dictionary<int, string> { [12] = "v-2d" },
        };
        ModelArtifact.ResolveChampionForLead(manifest, 12).Should().Be("v-2d");
        ModelArtifact.ResolveChampionForLead(manifest, 24).Should().Be("v-2b");
    }
}
