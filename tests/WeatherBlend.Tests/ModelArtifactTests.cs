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
}
