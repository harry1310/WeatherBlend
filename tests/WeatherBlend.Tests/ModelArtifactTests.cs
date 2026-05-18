using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Exact12h;
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
        manifest.Versions.Should().ContainSingle().Which.Should().Be("v2026-04-21_201231");
    }

    [Fact]
    public void UpdateManifest_appends_new_version_and_advances_Active()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.UpdateManifest(_root, "temperature", "v2");
        ModelArtifact.UpdateManifest(_root, "temperature", "v3");

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;

        manifest.Active.Should().Equal("v3");
        manifest.Versions.Should().ContainInOrder("v1", "v2", "v3");
    }

    [Fact]
    public void UpdateManifest_does_not_duplicate_an_already_listed_version()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.UpdateManifest(_root, "temperature", "v2");
        ModelArtifact.UpdateManifest(_root, "temperature", "v1"); // re-pointing Active back to v1

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;

        manifest.Active.Should().Equal("v1");
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
    public void AppendVersion_adds_to_Versions_without_changing_Active()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.AppendVersion(_root, "temperature", "v2-challenger");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Versions.Should().Equal("v1", "v2-challenger");
        manifest.Active.Should().ContainSingle().Which.Should().Be("v1",
            "AppendVersion must not touch Active — training a challenger doesn't promote it");
    }

    [Fact]
    public void SetActive_overrides_Active_list()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v1");
        ModelArtifact.AppendVersion(_root, "temperature", "v2");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v1", "v2" });

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

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
    public void ResolveActive_returns_empty_when_Active_empty()
    {
        // A manifest with an empty Active list has nothing live — there is
        // no Current pointer to fall back to any more.
        var dir = Path.Combine(_root, "temperature");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, ModelArtifact.ManifestFileName),
            """{"Target":"temperature","Versions":["v-legacy"],"Active":[],"Stations":{}}""");

        ModelArtifact.ResolveActive(_root, "temperature").Should().BeEmpty();
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
            .WithMessage("*champion version*");
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
            LocationName = "bonehill_rocks",
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
        reloaded.LocationName.Should().Be("bonehill_rocks");
        reloaded.TrainedAtUtc.Should().Be(original.TrainedAtUtc);
        reloaded.TestMae.Should().BeEquivalentTo(original.TestMae);
        reloaded.DeviationsFromBrief.Should().Equal(original.DeviationsFromBrief);
        reloaded.PerLead.Should().ContainKey("24");
        reloaded.PerLead["24"].BestSingle.Should().Be("temp_ecmwf");
        reloaded.PerLead["24"].BlendTestMae.Should().BeApproximately(1.23, 1e-9);
    }

    [Fact]
    public void TrainingMetadata_throws_when_LocationName_field_is_missing()
    {
        // Phase A tightening (Task #21): LocationName is [JsonRequired].
        // A bundle without it is a corrupt/incomplete write and must fail
        // at deserialise — silently treating it as null lets the predict
        // path either pick the wrong NWP source or fall back to legacy
        // behaviour, both of which are the bug we shipped Phase A to kill.
        // Pre-tightening (Tasks 15-20) the loader returned null and the
        // predict commands warn-then-fallback. After tightening, only
        // bundles written by trainers that thread LocationName through
        // RetrainGuard.BuildCheckAndSave will load — every other path is
        // a bundle we should not score against.
        var versionDir = Path.Combine(_root, "temperature", "v-no-location");
        Directory.CreateDirectory(versionDir);
        var jsonWithoutLocation = """
            {
              "Version": "v-no-location",
              "Target": "temperature",
              "Phase": "2b",
              "DataSource": "previous_runs_api",
              "TrainedAtUtc": "2026-04-21T20:12:31Z",
              "Hyperparameters": {},
              "TestMae": {},
              "DeviationsFromBrief": [],
              "PerLead": {}
            }
            """;
        File.WriteAllText(
            Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName),
            jsonWithoutLocation);

        var act = () => ModelArtifact.LoadTrainingMetadata(versionDir);

        act.Should().Throw<JsonException>()
            .WithMessage("*LocationName*");
    }

    [Fact]
    public void BlenderSpecs_round_trip_through_disk_preserving_structured_fields()
    {
        // Lock the on-disk contract for the structured-spec fields added
        // 2026-05-07 (DataSource / Tier / UkvStrategy). A round-trip via
        // SaveBlenderSpecs → LoadBlenderSpecs must preserve every field
        // — predict relies on FeatureNames + Models, and the Spec page now
        // relies on the structured fields. Anything dropped silently here
        // would re-introduce the dash-everywhere class of bug.
        var versionDir = Path.Combine(_root, "temperature", "v-spec-rt");
        Directory.CreateDirectory(versionDir);

        var original = new Dictionary<int, BlenderSpec>
        {
            [12] = new BlenderSpec
            {
                Target = "temperature",
                FeatureSet = "exact-l12-T2",
                LeadHours = 12,
                RequiredModels = new[] { "gfs_ncep", "ecmwf_aifs_oper" },
                OptionalModels = new[] { "ecmwf_ifs_oper", "met_office_global" },
                Models = new[] { "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global" },
                FeatureNames = new[] { "temp_gfs", "temp_ifs", "temp_aifs", "temp_moglobal", "temp_ukv", "temp_mean" },
                DataSource = BlenderDataSource.ExactRuntimeS3,
                Tier = "T2",
                UkvStrategy = Exact12hFeatureBuilder.UkvPickStrategy.Strict,
            },
            [48] = new BlenderSpec
            {
                Target = "temperature",
                FeatureSet = "lean",
                LeadHours = 48,
                RequiredModels = new[] { "gfs_seamless", "ecmwf_ifs025" },
                OptionalModels = new[] { "ecmwf_aifs025_single" },
                Models = new[] { "gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025_single" },
                FeatureNames = new[] { "temp_gfs", "temp_ecmwf", "temp_aifs", "temp_mean" },
                DataSource = BlenderDataSource.OpenMeteoPreviousRuns,
                Tier = "lean",
                UkvStrategy = null,
            },
        };

        ModelArtifact.SaveBlenderSpecs(versionDir, original);
        var reloaded = ModelArtifact.LoadBlenderSpecs(versionDir);

        reloaded.Should().HaveCount(2);
        reloaded[12].DataSource.Should().Be(BlenderDataSource.ExactRuntimeS3);
        reloaded[12].Tier.Should().Be("T2");
        reloaded[12].UkvStrategy.Should().Be(Exact12hFeatureBuilder.UkvPickStrategy.Strict);
        reloaded[12].RequiredModels.Should().Equal("gfs_ncep", "ecmwf_aifs_oper");
        reloaded[12].FeatureNames.Should().Contain("temp_ukv");

        reloaded[48].DataSource.Should().Be(BlenderDataSource.OpenMeteoPreviousRuns);
        reloaded[48].Tier.Should().Be("lean");
        reloaded[48].UkvStrategy.Should().BeNull();
    }

    [Fact]
    public void BlenderSpecs_load_handles_legacy_schemas_without_structured_fields()
    {
        // Legacy feature_schema.json files trained before the 2026-05-07
        // structured-spec migration don't have DataSource / Tier /
        // UkvStrategy keys. JSON deserialisation should default these to
        // empty strings / null without throwing — load happens on every
        // render-site run, so a deserialisation failure here would take
        // down the whole site.
        var versionDir = Path.Combine(_root, "temperature", "v-legacy");
        Directory.CreateDirectory(versionDir);
        var legacyJson = """
            {
              "Leads": {
                "24": {
                  "Target": "temperature",
                  "FeatureSet": "lean",
                  "LeadHours": 24,
                  "RequiredModels": ["gfs_seamless"],
                  "OptionalModels": [],
                  "Models": ["gfs_seamless"],
                  "FeatureNames": ["temp_gfs", "temp_mean"]
                }
              }
            }
            """;
        File.WriteAllText(Path.Combine(versionDir, "feature_schema.json"), legacyJson);

        var reloaded = ModelArtifact.LoadBlenderSpecs(versionDir);

        reloaded.Should().HaveCount(1);
        reloaded[24].DataSource.Should().BeEmpty();
        reloaded[24].Tier.Should().BeEmpty();
        reloaded[24].UkvStrategy.Should().BeNull();
        reloaded[24].FeatureNames.Should().Equal("temp_gfs", "temp_mean");
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

    /// <summary>
    /// Test helper. Writes both training_metadata.json (for phase) and
    /// feature_schema.json (for the lead set ComposeActive reads).
    /// When no leads are passed, defaults to {24, 48, 72} — a sensible
    /// "covers the common forecast leads" set so legacy promote tests
    /// (which assert "same phase replaces") see fully overlapping
    /// lead-sets between old and new entries and trigger the
    /// replacement rule. Tests that need disjoint or partial overlap
    /// pass explicit leads.
    /// </summary>
    private static void WritePhaseMetadata(string dir, string version, string phase, params int[] leads)
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
        var effectiveLeads = leads.Length > 0 ? leads : new[] { 24, 48, 72 };
        var specs = effectiveLeads.ToDictionary(
            l => l,
            l => new BlenderSpec
            {
                Target = "temperature",
                FeatureSet = "test",
                LeadHours = l,
                FeatureNames = new[] { "test_feature" },
            });
        ModelArtifact.SaveBlenderSpecs(dir, specs);
    }

    [Fact]
    public void PromoteVersion_replaces_same_phase_and_preserves_others()
    {
        // Setup: 2b champion (v_base_old) + 2c challenger (v_2c).
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base_old"), "v_base_old", "2b");
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c"),       "v_2c",       "2c");
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base_old");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base_old", "v_2c" });

        // Retrain 2b → v_base_new. Expected: v_base_old is dropped from Active
        // (replaced by v_base_new), v_2c stays.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base_new"), "v_base_new", "2b");
        ModelArtifact.PromoteVersion(_root, "temperature", "v_base_new", newPhase: "2b");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().BeEquivalentTo(new[] { "v_2c", "v_base_new" });   // 2c preserved
        manifest.Versions.Should().Contain(new[] { "v_base_old", "v_base_new" });
    }

    [Fact]
    public void PromoteVersion_replaces_same_phase_challenger_and_preserves_champion()
    {
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"),    "v_base",    "2b");
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_old"),  "v_2c_old",  "2c");
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2c_old" });

        // Retrain 2c → v_2c_new. Expected: v_2c_old replaced by v_2c_new in
        // Active, v_base preserved.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_new"), "v_2c_new", "2c");
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2c_new", newPhase: "2c");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

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
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2c_a", newPhase: "2c");

        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2c_b"), "v_2c_b", "2c");
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2c_b", newPhase: "2c");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        // Active has exactly one 2c (the latest), not both.
        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2c_b" });
        // Versions accumulates both 2c entries (history is preserved).
        manifest.Versions.Should().Contain(new[] { "v_base", "v_2c_a", "v_2c_b" });
    }

    [Fact]
    public void PromoteVersion_keeps_same_phase_versions_with_disjoint_lead_sets()
    {
        // Same-phase entries with FULLY DISJOINT lead sets coexist —
        // neither emits predictions at leads the other covers, so no
        // duplicate (composite, lead, valid) parquet rows on disk.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"), "v_base", "2b", 24, 48, 72);
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_short"), "v_2d_short", "2d", 12, 24, 48);
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2d_short" });

        // Add a 2d at the OTHER lead bucket — disjoint from v_2d_short's
        // {12,24,48}. Both 2d versions should survive Active.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_long"), "v_2d_long", "2d", 72, 96, 120);
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2d_long", newPhase: "2d");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2d_short", "v_2d_long" });
    }

    [Fact]
    public void PromoteVersion_drops_same_phase_version_with_any_overlap_on_leads()
    {
        // Operator runs a wider 2d retrain that supersedes a narrower
        // existing 2d. Because the new lead-set {12,24,48,72} OVERLAPS
        // with the existing {48,72} on at least one lead, V_old must be
        // dropped — keeping it would produce duplicate predictions at
        // (composite=temperature, lead=48 / 72) tuples on disk.
        //
        // Regression test for the 2026-05-08 second-iteration bug where
        // an equality-only check kept both versions Active when one's
        // lead-set strictly contained the other's.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"), "v_base", "2b", 24, 48, 72);
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_old"), "v_2d_old", "2d", 48, 72);
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2d_old" });

        // Retrain 2d covering MORE leads. {12,24,48,72} ∩ {48,72} = {48,72}
        // ≠ ∅ → V_old superseded.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_wide"), "v_2d_wide", "2d", 12, 24, 48, 72);
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2d_wide", newPhase: "2d");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2d_wide" });
        manifest.Active.Should().NotContain("v_2d_old");
    }

    [Fact]
    public void PromoteVersion_partial_overlap_drops_existing_even_when_existing_has_unique_leads()
    {
        // The footgun documented on ComposeActive: a partial-coverage
        // retrain that overlaps an existing wider version supersedes it,
        // potentially losing the unique-to-old leads' coverage. Tested
        // here so the behaviour is locked, not a surprise.
        //
        // V_old {12,24} + V_new {24} → overlap on {24} → drop V_old.
        // Lead 12 loses its 2d source (would need a separate retrain to
        // restore). Operationally: don't run partial retrains unless you
        // mean to abandon the leads you're not touching.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"), "v_base", "2b", 24, 48);
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_wide"), "v_2d_wide", "2d", 12, 24);
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2d_wide" });

        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_narrow"), "v_2d_narrow", "2d", 24);
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2d_narrow", newPhase: "2d");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2d_narrow" });
        manifest.Active.Should().NotContain("v_2d_wide");
    }

    [Fact]
    public void PromoteVersion_replaces_same_phase_same_lead_set()
    {
        // Re-train idempotency: training the SAME (phase, lead-set) again
        // replaces the prior entry — no duplicate Active rows for the same
        // configuration. This is the only case ComposeActive treats as
        // a same-config replacement.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"), "v_base", "2b", 24, 48, 72);
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_a"), "v_2d_a", "2d", 12, 24, 48);
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2d_a" });

        // Re-train 2d on the SAME leads.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_b"), "v_2d_b", "2d", 12, 24, 48);
        ModelArtifact.PromoteVersion(_root, "temperature", "v_2d_b", newPhase: "2d");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        // v_2d_a replaced by v_2d_b — same phase, same lead set.
        manifest.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2d_b" });
    }

    [Fact]
    public void PromoteVersion_preserves_existing_when_new_version_lacks_schema()
    {
        // If we can't read the new version's lead set (no
        // feature_schema.json on disk), TryReadVersionLeads returns null.
        // ComposeActive then falls back to the conservative "preserve
        // everything we can't compare" rule — a missing schema can't be
        // shown to overlap with anything, so the existing version stays.
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_base"), "v_base", "2b", 24, 48);
        WritePhaseMetadata(Path.Combine(_root, "temperature", "v_2d_old"), "v_2d_old", "2d", 12, 24);
        ModelArtifact.UpdateManifest(_root, "temperature", "v_base");
        ModelArtifact.SetActive(_root, "temperature", new[] { "v_base", "v_2d_old" });

        // New 2d version: write the metadata + schema, then DELETE the
        // schema to simulate an unreadable artefact (e.g. a half-written
        // train run, or a pre-Phase-1 schema). Promote should preserve
        // v_2d_old conservatively.
        var newDir = Path.Combine(_root, "temperature", "v_2d_new");
        WritePhaseMetadata(newDir, "v_2d_new", "2d", 12, 24);
        File.Delete(Path.Combine(newDir, ModelArtifact.FeatureSchemaFileName));

        ModelArtifact.PromoteVersion(_root, "temperature", "v_2d_new", newPhase: "2d");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        manifest.Active.Should().Contain("v_2d_old");  // preserved despite shared phase + leads
        manifest.Active.Should().Contain("v_2d_new");  // appended
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
        ModelArtifact.PromoteVersion(_root, "temperature", "v_base_new", newPhase: "2b");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName)))!;

        // v_orphan stays. Only v_base (matching phase 2b) is replaced.
        manifest.Active.Should().BeEquivalentTo(new[] { "v_orphan", "v_base_new" });
    }

    [Fact]
    public void PromoteStationVersion_replaces_same_phase_per_station_only()
    {
        // Phase 3a champion + 3c challenger at one station. Retraining 3a
        // replaces only the 3a entry, leaves 3c — and doesn't touch the
        // OTHER station's manifest entry at all.
        var stationA = "ea_bellever_dartmoor";
        var stationB = "ea_bovey_tracey";

        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationA, "v_3a_old"), "v_3a_old", "3a");
        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationA, "v_3c"),     "v_3c",     "3c");
        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationB, "v_3a_b"),   "v_3a_b",   "3a");

        ModelArtifact.UpdateStationManifest(_root, "precipitation", stationA, "v_3a_old");
        ModelArtifact.SetStationActive(_root, "precipitation", stationA, new[] { "v_3a_old", "v_3c" });
        ModelArtifact.UpdateStationManifest(_root, "precipitation", stationB, "v_3a_b");

        WritePhaseMetadata(Path.Combine(_root, "precipitation", stationA, "v_3a_new"), "v_3a_new", "3a");
        ModelArtifact.PromoteStationVersion(
            _root, "precipitation", stationA, "v_3a_new", newPhase: "3a");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "precipitation", ModelArtifact.ManifestFileName)))!;

        manifest.Stations[stationA].Active.Should().BeEquivalentTo(new[] { "v_3c", "v_3a_new" });
        // Other station untouched.
        manifest.Stations[stationB].Active.Should().BeEquivalentTo(new[] { "v_3a_b" });
    }

    [Fact]
    public void PromoteStationVersion_replaces_same_phase_3e_idempotently()
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
        ModelArtifact.PromoteStationVersion(
            _root, "dry_window", station, "v_3e_b", newPhase: "3e");

        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(Path.Combine(_root, "dry_window", ModelArtifact.ManifestFileName)))!;

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
    public void ResolveChampionForLead_falls_back_to_champion_version_when_no_per_lead_override()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v-2b");
        ModelArtifact.ResolveChampionForLead(_root, "temperature", 24).Should().Be("v-2b");
        ModelArtifact.ResolveChampionForLead(_root, "temperature", 12).Should().Be("v-2b");
    }

    [Fact]
    public void SetChampionForLead_pins_per_lead_override()
    {
        ModelArtifact.UpdateManifest(_root, "temperature", "v-2b");
        ModelArtifact.SetChampionForLead(_root, "temperature", 12, "v-2d");

        ModelArtifact.ResolveChampionForLead(_root, "temperature", 12).Should().Be("v-2d");
        ModelArtifact.ResolveChampionForLead(_root, "temperature", 24).Should().Be("v-2b");

        var path = Path.Combine(_root, "temperature", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;
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

    // ---- Per-station ChampionByLead (Phase 3d) -------------------------------

    [Fact]
    public void SetStationChampionForLead_pins_per_lead_override_per_station()
    {
        ModelArtifact.UpdateStationManifest(_root, "precipitation", "ea_bellever_dartmoor", "v-3a");
        ModelArtifact.UpdateStationManifest(_root, "precipitation", "ea_dartmoor_nr_hexworthy", "v-3a-h");
        ModelArtifact.SetStationChampionForLead(_root, "precipitation", "ea_bellever_dartmoor", 12, "v-3d");

        var path = Path.Combine(_root, "precipitation", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;
        manifest.Stations["ea_bellever_dartmoor"].ChampionByLead.Should().ContainKey(12)
            .WhoseValue.Should().Be("v-3d");
        // The other station's per-lead pins are independent.
        manifest.Stations["ea_dartmoor_nr_hexworthy"].ChampionByLead.Should().BeEmpty();
    }

    [Fact]
    public void SetStationChampionForLead_with_empty_string_clears_the_pin()
    {
        ModelArtifact.UpdateStationManifest(_root, "precipitation", "ea_bellever_dartmoor", "v-3a");
        ModelArtifact.SetStationChampionForLead(_root, "precipitation", "ea_bellever_dartmoor", 12, "v-3d");
        ModelArtifact.SetStationChampionForLead(_root, "precipitation", "ea_bellever_dartmoor", 12, "");

        var path = Path.Combine(_root, "precipitation", ModelArtifact.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path))!;
        manifest.Stations["ea_bellever_dartmoor"].ChampionByLead.Should().NotContainKey(12);
    }
}
