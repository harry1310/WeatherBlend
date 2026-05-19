using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Exact12h;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// ModelArtifact coverage for the artefact round-trips (training_metadata,
/// blender specs, feature importance), the manifest-write concurrency
/// guarantees, and the lead-overlap-aware PromoteStationVersion / ComposeActive
/// rules. The basic per-station manifest CRUD (create / append / resolve /
/// version-dir) is exercised in <see cref="ModelArtifactStationTests"/>.
///
/// Every target manifest is Stations-keyed — there is no flat layout.
/// Temperature / element_* targets use a single entry keyed by location;
/// these tests use that location slug as the station key.
/// </summary>
public class ModelArtifactTests : IDisposable
{
    private readonly string _root;

    private const string Station = "bonehill_rocks";

    public ModelArtifactTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-artifact-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private ModelArtifact.Manifest ReadManifest(string target)
        => JsonSerializer.Deserialize<ModelArtifact.Manifest>(
               File.ReadAllText(Path.Combine(_root, target, ModelArtifact.ManifestFileName)))!;

    private string VersionDir(string target, string station, string version)
        => Path.Combine(_root, target, station, version);

    // ---- TrainingMetadata round-trip -------------------------------------------

    [Fact]
    public void TrainingMetadata_round_trips_through_disk()
    {
        var versionDir = VersionDir("temperature", Station, "v-rt");
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
        var versionDir = VersionDir("temperature", Station, "v-no-location");
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
    public void LoadTrainingMetadata_throws_for_missing_file()
    {
        var versionDir = VersionDir("temperature", Station, "v-missing");
        Directory.CreateDirectory(versionDir);

        var act = () => ModelArtifact.LoadTrainingMetadata(versionDir);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Missing training metadata*");
    }

    // ---- BlenderSpecs round-trip -----------------------------------------------

    [Fact]
    public void BlenderSpecs_round_trip_through_disk_preserving_structured_fields()
    {
        // Lock the on-disk contract for the structured-spec fields added
        // 2026-05-07 (DataSource / Tier / UkvStrategy). A round-trip via
        // SaveBlenderSpecs → LoadBlenderSpecs must preserve every field
        // — predict relies on FeatureNames + Models, and the Spec page now
        // relies on the structured fields. Anything dropped silently here
        // would re-introduce the dash-everywhere class of bug.
        var versionDir = VersionDir("temperature", Station, "v-spec-rt");
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
        var versionDir = VersionDir("temperature", Station, "v-legacy");
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

    // ---- SavePerLeadFeatureImportance / LoadPerLeadFeatureImportance ----------

    [Fact]
    public void PerLeadFeatureImportance_round_trips_with_sort_order_preserved()
    {
        var versionDir = VersionDir("temperature", Station, "v-fi");
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
        var versionDir = VersionDir("temperature", Station, "v-none");
        Directory.CreateDirectory(versionDir);

        var loaded = ModelArtifact.LoadPerLeadFeatureImportance(versionDir);
        loaded.Should().BeEmpty();
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
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v0");

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        var observedMissing = 0;
        var observedReads = 0;

        // Reader thread — hammers ResolveStationActive while the writer churns.
        var reader = Task.Run(() =>
        {
            while (!cts.IsCancellationRequested)
            {
                try
                {
                    var active = ModelArtifact.ResolveStationActive(_root, "temperature", Station);
                    if (active.Count == 0) Interlocked.Increment(ref observedMissing);
                    Interlocked.Increment(ref observedReads);
                }
                catch (IOException) { Interlocked.Increment(ref observedMissing); }
            }
        });

        // Writer thread — bumps Active 200× in tight succession.
        var writer = Task.Run(() =>
        {
            for (int i = 1; i <= 200 && !cts.IsCancellationRequested; i++)
            {
                ModelArtifact.UpdateStationManifest(_root, "temperature", Station, $"v{i}");
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
    public void Concurrent_AppendStationVersion_does_not_lose_updates()
    {
        const int N = 30;

        var threads = Enumerable.Range(0, N).Select(i => new Thread(() =>
        {
            ModelArtifact.AppendStationVersion(_root, "temperature", Station, $"v-thread-{i:D2}");
        })).ToList();

        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join();

        var versions = ReadManifest("temperature").Stations[Station].Versions;
        versions.Should().HaveCount(N);
        for (int i = 0; i < N; i++)
            versions.Should().Contain($"v-thread-{i:D2}");
    }

    /// <summary>
    /// Concurrent SetStationActive + AppendStationVersion must compose: every
    /// appended version survives, and the final SetStationActive winner is one
    /// of the values written. Mirrors the chained train-then-calibrate flow
    /// that previously trampled.
    /// </summary>
    [Fact]
    public void Concurrent_SetStationActive_and_AppendStationVersion_compose_without_loss()
    {
        const int N = 20;
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v-base");

        var threads = new List<Thread>();
        for (int i = 0; i < N; i++)
        {
            var idx = i;
            threads.Add(new Thread(() =>
            {
                ModelArtifact.AppendStationVersion(_root, "temperature", Station, $"v-{idx:D2}");
                ModelArtifact.SetStationActive(_root, "temperature", Station,
                    new[] { "v-base", $"v-{idx:D2}" });
            }));
        }

        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join();

        var entry = ReadManifest("temperature").Stations[Station];

        // All N appended versions plus v-base must be present.
        entry.Versions.Should().Contain("v-base");
        for (int i = 0; i < N; i++)
            entry.Versions.Should().Contain($"v-{i:D2}");

        // Active is whichever thread wrote last; must contain v-base + exactly one v-{i}.
        entry.Active.Should().HaveCount(2);
        entry.Active.Should().Contain("v-base");
        entry.Active.Should().ContainSingle(v => v.StartsWith("v-", StringComparison.Ordinal) && v != "v-base");
    }

    // ---- PromoteStationVersion (idempotent same-phase replacement) -----------
    //
    // PromoteStationVersion fixes the load-bearing footgun in
    // UpdateStationManifest / SetStationActive: those reset Active to [new],
    // which silently kicked any active challenger phase out of the rotation on
    // every champion retrain. Promote reads each existing Active entry's
    // training_metadata.Phase and only replaces entries with the same phase as
    // the new version (lead-overlap-aware via ComposeActive), leaving all other
    // phases (champions or challengers) untouched. Tests below pin both the
    // same-phase replacement rule and the "preserve other phases" invariant.

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
            LocationName = "bonehill_rocks",
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
    public void PromoteStationVersion_replaces_same_phase_and_preserves_others()
    {
        // Setup: 2b champion (v_base_old) + 2c challenger (v_2c).
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base_old"), "v_base_old", "2b");
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2c"),       "v_2c",       "2c");
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base_old");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base_old", "v_2c" });

        // Retrain 2b → v_base_new. Expected: v_base_old is dropped from Active
        // (replaced by v_base_new), v_2c stays.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base_new"), "v_base_new", "2b");
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_base_new", newPhase: "2b");

        var entry = ReadManifest("temperature").Stations[Station];
        entry.Active.Should().BeEquivalentTo(new[] { "v_2c", "v_base_new" });   // 2c preserved
        entry.Versions.Should().Contain(new[] { "v_base_old", "v_base_new" });
    }

    [Fact]
    public void PromoteStationVersion_replaces_same_phase_challenger_and_preserves_champion()
    {
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"),   "v_base",   "2b");
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2c_old"), "v_2c_old", "2c");
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_2c_old" });

        // Retrain 2c → v_2c_new. Expected: v_2c_old replaced by v_2c_new in
        // Active, v_base preserved.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2c_new"), "v_2c_new", "2c");
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2c_new", newPhase: "2c");

        ReadManifest("temperature").Stations[Station].Active
            .Should().BeEquivalentTo(new[] { "v_base", "v_2c_new" }); // 2c rotated
    }

    [Fact]
    public void PromoteStationVersion_is_idempotent_across_repeated_same_phase_retrains()
    {
        // Run two 2c retrains back-to-back. Active should converge to one 2c
        // entry per run, not accumulate stale 2c versions across runs.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b");
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");

        WritePhaseMetadata(VersionDir("temperature", Station, "v_2c_a"), "v_2c_a", "2c");
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2c_a", newPhase: "2c");

        WritePhaseMetadata(VersionDir("temperature", Station, "v_2c_b"), "v_2c_b", "2c");
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2c_b", newPhase: "2c");

        var entry = ReadManifest("temperature").Stations[Station];
        // Active has exactly one 2c (the latest), not both.
        entry.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2c_b" });
        // Versions accumulates both 2c entries (history is preserved).
        entry.Versions.Should().Contain(new[] { "v_base", "v_2c_a", "v_2c_b" });
    }

    [Fact]
    public void PromoteStationVersion_keeps_same_phase_versions_with_disjoint_lead_sets()
    {
        // Same-phase entries with FULLY DISJOINT lead sets coexist —
        // neither emits predictions at leads the other covers, so no
        // duplicate (composite, lead, valid) parquet rows on disk.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b", 24, 48, 72);
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_short"), "v_2d_short", "2d", 12, 24, 48);
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_2d_short" });

        // Add a 2d at the OTHER lead bucket — disjoint from v_2d_short's
        // {12,24,48}. Both 2d versions should survive Active.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_long"), "v_2d_long", "2d", 72, 96, 120);
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2d_long", newPhase: "2d");

        ReadManifest("temperature").Stations[Station].Active
            .Should().BeEquivalentTo(new[] { "v_base", "v_2d_short", "v_2d_long" });
    }

    [Fact]
    public void PromoteStationVersion_drops_same_phase_version_with_any_overlap_on_leads()
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
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b", 24, 48, 72);
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_old"), "v_2d_old", "2d", 48, 72);
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_2d_old" });

        // Retrain 2d covering MORE leads. {12,24,48,72} ∩ {48,72} = {48,72}
        // ≠ ∅ → V_old superseded.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_wide"), "v_2d_wide", "2d", 12, 24, 48, 72);
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2d_wide", newPhase: "2d");

        var entry = ReadManifest("temperature").Stations[Station];
        entry.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2d_wide" });
        entry.Active.Should().NotContain("v_2d_old");
    }

    [Fact]
    public void PromoteStationVersion_partial_overlap_drops_existing_even_when_existing_has_unique_leads()
    {
        // The footgun documented on ComposeActive: a partial-coverage
        // retrain that overlaps an existing wider version supersedes it,
        // potentially losing the unique-to-old leads' coverage.
        //
        // V_old {12,24} + V_new {24} → overlap on {24} → drop V_old.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b", 24, 48);
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_wide"), "v_2d_wide", "2d", 12, 24);
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_2d_wide" });

        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_narrow"), "v_2d_narrow", "2d", 24);
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2d_narrow", newPhase: "2d");

        var entry = ReadManifest("temperature").Stations[Station];
        entry.Active.Should().BeEquivalentTo(new[] { "v_base", "v_2d_narrow" });
        entry.Active.Should().NotContain("v_2d_wide");
    }

    [Fact]
    public void PromoteStationVersion_replaces_same_phase_same_lead_set()
    {
        // Re-train idempotency: training the SAME (phase, lead-set) again
        // replaces the prior entry — no duplicate Active rows for the same
        // configuration.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b", 24, 48, 72);
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_a"), "v_2d_a", "2d", 12, 24, 48);
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_2d_a" });

        // Re-train 2d on the SAME leads.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_b"), "v_2d_b", "2d", 12, 24, 48);
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2d_b", newPhase: "2d");

        // v_2d_a replaced by v_2d_b — same phase, same lead set.
        ReadManifest("temperature").Stations[Station].Active
            .Should().BeEquivalentTo(new[] { "v_base", "v_2d_b" });
    }

    [Fact]
    public void PromoteStationVersion_preserves_existing_when_new_version_lacks_schema()
    {
        // If we can't read the new version's lead set (no
        // feature_schema.json on disk), TryReadVersionLeads returns null.
        // ComposeActive then falls back to the conservative "preserve
        // everything we can't compare" rule — a missing schema can't be
        // shown to overlap with anything, so the existing version stays.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b", 24, 48);
        WritePhaseMetadata(VersionDir("temperature", Station, "v_2d_old"), "v_2d_old", "2d", 12, 24);
        ModelArtifact.UpdateStationManifest(_root, "temperature", Station, "v_base");
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_2d_old" });

        // New 2d version: write the metadata + schema, then DELETE the
        // schema to simulate an unreadable artefact. Promote should
        // preserve v_2d_old conservatively.
        var newDir = VersionDir("temperature", Station, "v_2d_new");
        WritePhaseMetadata(newDir, "v_2d_new", "2d", 12, 24);
        File.Delete(Path.Combine(newDir, ModelArtifact.FeatureSchemaFileName));

        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_2d_new", newPhase: "2d");

        var entry = ReadManifest("temperature").Stations[Station];
        entry.Active.Should().Contain("v_2d_old");  // preserved despite shared phase + leads
        entry.Active.Should().Contain("v_2d_new");  // appended
    }

    [Fact]
    public void PromoteStationVersion_preserves_entry_with_unreadable_metadata()
    {
        // An Active entry whose training_metadata is missing or malformed has
        // unknown phase. The promote helper PRESERVES it rather than silently
        // dropping — caller can clean up explicitly via SetStationActive.
        WritePhaseMetadata(VersionDir("temperature", Station, "v_base"), "v_base", "2b");
        // v_orphan has no metadata file.
        Directory.CreateDirectory(VersionDir("temperature", Station, "v_orphan"));
        ModelArtifact.SetStationActive(_root, "temperature", Station, new[] { "v_base", "v_orphan" });

        WritePhaseMetadata(VersionDir("temperature", Station, "v_base_new"), "v_base_new", "2b");
        ModelArtifact.PromoteStationVersion(_root, "temperature", Station, "v_base_new", newPhase: "2b");

        // v_orphan stays. Only v_base (matching phase 2b) is replaced.
        ReadManifest("temperature").Stations[Station].Active
            .Should().BeEquivalentTo(new[] { "v_orphan", "v_base_new" });
    }

    [Fact]
    public void PromoteStationVersion_replaces_same_phase_per_station_only()
    {
        // Phase 3a champion + 3c challenger at one station. Retraining 3a
        // replaces only the 3a entry, leaves 3c — and doesn't touch the
        // OTHER station's manifest entry at all.
        var stationA = "ea_bellever_dartmoor";
        var stationB = "ea_bovey_tracey";

        WritePhaseMetadata(VersionDir("precipitation", stationA, "v_3a_old"), "v_3a_old", "3a");
        WritePhaseMetadata(VersionDir("precipitation", stationA, "v_3c"),     "v_3c",     "3c");
        WritePhaseMetadata(VersionDir("precipitation", stationB, "v_3a_b"),   "v_3a_b",   "3a");

        ModelArtifact.UpdateStationManifest(_root, "precipitation", stationA, "v_3a_old");
        ModelArtifact.SetStationActive(_root, "precipitation", stationA, new[] { "v_3a_old", "v_3c" });
        ModelArtifact.UpdateStationManifest(_root, "precipitation", stationB, "v_3a_b");

        WritePhaseMetadata(VersionDir("precipitation", stationA, "v_3a_new"), "v_3a_new", "3a");
        ModelArtifact.PromoteStationVersion(
            _root, "precipitation", stationA, "v_3a_new", newPhase: "3a");

        var manifest = ReadManifest("precipitation");
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

        WritePhaseMetadata(VersionDir("dry_window", station, "v_3b"),   "v_3b",   "3b");
        WritePhaseMetadata(VersionDir("dry_window", station, "v_3e_a"), "v_3e_a", "3e");
        ModelArtifact.UpdateStationManifest(_root, "dry_window", station, "v_3b");
        ModelArtifact.SetStationActive(_root, "dry_window", station, new[] { "v_3b", "v_3e_a" });

        WritePhaseMetadata(VersionDir("dry_window", station, "v_3e_b"), "v_3e_b", "3e");
        ModelArtifact.PromoteStationVersion(
            _root, "dry_window", station, "v_3e_b", newPhase: "3e");

        ReadManifest("dry_window").Stations[station].Active
            .Should().BeEquivalentTo(new[] { "v_3b", "v_3e_b" });
    }
}
