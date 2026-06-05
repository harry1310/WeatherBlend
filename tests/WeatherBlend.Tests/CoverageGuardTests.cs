using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using WeatherBlend.Commands;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict.Coverage;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Coverage-guard tests. The guard's job is to turn the 2026-06-05 GEM-outage
/// silent-degradation into a loud failure: an active, bundled phase that
/// produced no predictions this cycle must breach. The registry's locations:
/// filter is what keeps "expected here" declarative (no champion/challenger
/// distinction).
/// </summary>
public class CoverageGuardTests : IDisposable
{
    private readonly string _models = Path.Combine(
        Path.GetTempPath(), "wb_cov_" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_models)) Directory.Delete(_models, recursive: true); } catch { }
    }

    // Two-location, temperature + wind registry. temperature: 2b champion, 2c
    // challenger (both everywhere), 2d challenger (bonehill only). wind: wind
    // champion (bonehill only) + wind_blend challenger (bonehill only, retrain:
    // none = composed live).
    private const string TwoLocYaml = """
        targets:
          temperature:
            phases:
              - id: "2b"
                role: champion
                impl: dotnet
              - id: "2c"
                role: challenger
                impl: dotnet
              - id: "2d"
                role: challenger
                impl: dotnet
                locations: ["bonehill_rocks"]
          wind:
            phases:
              - id: "wind"
                role: champion
                impl: dotnet
                locations: ["bonehill_rocks"]
              - id: "wind_blend"
                role: challenger
                impl: dotnet
                locations: ["bonehill_rocks"]
                retrain: none
        """;

    private static readonly CoverageGuard.LocationSpec Bonehill = new("bonehill_rocks", Array.Empty<string>());
    private static readonly CoverageGuard.LocationSpec Membury = new("membury_devon", Array.Empty<string>());

    private void AddBundle(string target, string station, string version, string phase, string location)
    {
        var dir = Path.Combine(_models, target, station, version);
        Directory.CreateDirectory(dir);
        ModelArtifact.SaveTrainingMetadata(dir, new ModelArtifact.TrainingMetadata
        {
            Version = version,
            Target = target,
            Phase = phase,
            LocationName = location,
        });
    }

    private void SetActive(string target, string station, params string[] versions)
        => ModelArtifact.SetStationActive(_models, target, station, versions);

    /// <summary>Bonehill: 2b/2c/2d + wind all bundled & active. Membury: 2b/2c.</summary>
    private void SeedHealthyFixture()
    {
        AddBundle("temperature", "bonehill_rocks", "v_2b", "2b", "bonehill_rocks");
        AddBundle("temperature", "bonehill_rocks", "v_2c", "2c", "bonehill_rocks");
        AddBundle("temperature", "bonehill_rocks", "v_2d", "2d", "bonehill_rocks");
        SetActive("temperature", "bonehill_rocks", "v_2b", "v_2c", "v_2d");

        AddBundle("temperature", "membury_devon", "v_2b_m", "2b", "membury_devon");
        AddBundle("temperature", "membury_devon", "v_2c_m", "2c", "membury_devon");
        SetActive("temperature", "membury_devon", "v_2b_m", "v_2c_m");

        AddBundle("wind", "bonehill_rocks", "v_wind", "wind", "bonehill_rocks");
        SetActive("wind", "bonehill_rocks", "v_wind");
    }

    [Fact]
    public void All_active_bundled_cells_producing_passes()
    {
        SeedHealthyFixture();
        var produced = new HashSet<string> { "v_2b", "v_2c", "v_2d", "v_2b_m", "v_2c_m", "v_wind" };

        var result = CoverageGuard.Run(_models, PhaseRegistry.LoadFromYaml(TwoLocYaml),
            new[] { Bonehill, Membury }, c => produced.Contains(c.Version));

        result.Passed.Should().BeTrue();
        result.Breaches.Should().BeEmpty();
        // Bonehill 2b/2c/2d (3) + Membury 2b/2c (2) + wind (1). wind_blend is
        // retrain:none → skipped. 2d excluded for Membury by locations: filter.
        result.CellsChecked.Should().Be(6);
    }

    [Fact]
    public void Active_bundled_cell_producing_nothing_breaches_naming_the_cell()
    {
        // The GEM regression: Membury's 2b champion bundle is Active but produced
        // no rows this cycle (everything else fine).
        SeedHealthyFixture();
        var produced = new HashSet<string> { "v_2b", "v_2c", "v_2d", "v_2c_m", "v_wind" }; // v_2b_m missing

        var result = CoverageGuard.Run(_models, PhaseRegistry.LoadFromYaml(TwoLocYaml),
            new[] { Bonehill, Membury }, c => produced.Contains(c.Version));

        result.Passed.Should().BeFalse();
        result.Breaches.Should().ContainSingle();
        var b = result.Breaches[0];
        b.Target.Should().Be("temperature");
        b.StationKey.Should().Be("membury_devon");
        b.Phase.Should().Be("2b");
        b.Versions.Should().Contain("v_2b_m");
    }

    [Fact]
    public void Composed_phase_without_bundle_is_skipped_not_warned_or_breached()
    {
        // wind_blend is registry-active for bonehill, retrain:none, no bundle.
        // It must NOT breach (its inputs are covered) and must NOT warn.
        SeedHealthyFixture();
        var produced = new HashSet<string> { "v_2b", "v_2c", "v_2d", "v_2b_m", "v_2c_m", "v_wind" };

        var result = CoverageGuard.Run(_models, PhaseRegistry.LoadFromYaml(TwoLocYaml),
            new[] { Bonehill, Membury }, c => produced.Contains(c.Version));

        result.Passed.Should().BeTrue();
        result.Warnings.Should().NotContain(w => w.Phase == "wind_blend");
        result.Breaches.Should().NotContain(b => b.Phase == "wind_blend");
    }

    [Fact]
    public void Registry_active_phase_with_no_bundle_warns_but_does_not_breach()
    {
        // A retrained phase the registry ships but no Active bundle exists for —
        // a training/rollout gap, RetrainGuard's job, not a predict regression.
        const string yaml = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: dotnet
                  - id: "2e"
                    role: challenger
                    impl: dotnet
            """;
        AddBundle("temperature", "bonehill_rocks", "v_2b", "2b", "bonehill_rocks");
        SetActive("temperature", "bonehill_rocks", "v_2b"); // no 2e bundle at all

        var result = CoverageGuard.Run(_models, PhaseRegistry.LoadFromYaml(yaml),
            new[] { Bonehill }, _ => true);

        result.Passed.Should().BeTrue("an unbundled phase is a warning, not a breach");
        result.Warnings.Should().ContainSingle(w => w.Phase == "2e" && w.Target == "temperature");
        result.CellsChecked.Should().Be(1); // only 2b had a bundle to check
    }

    [Fact]
    public void Phase_excluded_by_locations_filter_is_not_expected()
    {
        // Membury must not be checked for wind / 2d (both bonehill-only).
        SeedHealthyFixture();
        var produced = new HashSet<string> { "v_2b", "v_2c", "v_2d", "v_2b_m", "v_2c_m", "v_wind" };

        var result = CoverageGuard.Run(_models, PhaseRegistry.LoadFromYaml(TwoLocYaml),
            new[] { Membury }, c => produced.Contains(c.Version));

        result.Passed.Should().BeTrue();
        result.Breaches.Should().BeEmpty();
        result.Warnings.Should().BeEmpty();
        result.CellsChecked.Should().Be(2); // Membury: 2b + 2c only
    }

    [Fact]
    public void Retired_active_version_whose_phase_is_not_in_registry_is_ignored()
    {
        // A demoted/retired bundle left in Active whose phase isn't shipping must
        // not breach or warn — it simply isn't matched to any expected phase.
        AddBundle("temperature", "bonehill_rocks", "v_2b", "2b", "bonehill_rocks");
        AddBundle("temperature", "bonehill_rocks", "v_iso", "3a_isotonic", "bonehill_rocks");
        SetActive("temperature", "bonehill_rocks", "v_2b", "v_iso");

        const string yaml = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: dotnet
            """;
        var result = CoverageGuard.Run(_models, PhaseRegistry.LoadFromYaml(yaml),
            new[] { Bonehill }, c => c.Version == "v_2b");

        result.Passed.Should().BeTrue();
        result.Breaches.Should().BeEmpty();
        result.Warnings.Should().BeEmpty();
        result.CellsChecked.Should().Be(1);
    }

    [Fact]
    public void Python_phases_are_skipped_when_the_impl_filter_excludes_them()
    {
        // 4a (python) is produced by a SEPARATE workflow on its own cadence, so
        // the predict/predict-and-render guard scopes to dotnet phases. With the
        // filter, an unproduced 4a must NOT breach; without it, it does — proving
        // the filter is what spares the cross-repo python phases.
        const string yaml = """
            targets:
              precipitation:
                phases:
                  - id: "3a"
                    role: champion
                    impl: dotnet
                  - id: "4a"
                    role: challenger
                    impl: python
            """;
        AddBundle("precipitation", "ea_bellever_dartmoor", "v_3a", "3a", "bonehill_rocks");
        AddBundle("precipitation", "ea_bellever_dartmoor", "v_4a", "4a", "bonehill_rocks");
        SetActive("precipitation", "ea_bellever_dartmoor", "v_3a", "v_4a");

        var loc = new CoverageGuard.LocationSpec("bonehill_rocks", new[] { "ea_bellever_dartmoor" });
        var produced = new HashSet<string> { "v_3a" }; // 3a produced, 4a did NOT

        var registry = PhaseRegistry.LoadFromYaml(yaml);

        var dotnetOnly = CoverageGuard.Run(_models, registry, new[] { loc },
            c => produced.Contains(c.Version), includePhase: p => p.Impl == PhaseImpl.Dotnet);
        dotnetOnly.Passed.Should().BeTrue("4a is python → excluded");
        dotnetOnly.CellsChecked.Should().Be(1);

        var unfiltered = CoverageGuard.Run(_models, registry, new[] { loc },
            c => produced.Contains(c.Version));
        unfiltered.Breaches.Should().ContainSingle(b => b.Phase == "4a",
            "without the impl filter the unproduced 4a breaches");
    }

    [Fact]
    public async Task Command_exit_code_is_0_when_covered_and_5_on_breach()
    {
        // End-to-end through PredictCoverageCommand: build a per-key-dir
        // (temperature) fixture + a real anchor partition; assert exit 0, then
        // delete the partition and assert exit 5. Uses the registry-injectable
        // overload so it doesn't depend on the live phases.yaml.
        AddBundle("temperature", "bonehill_rocks", "v_2b", "2b", "bonehill_rocks");
        SetActive("temperature", "bonehill_rocks", "v_2b");

        const string yaml = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: dotnet
            """;
        var registry = PhaseRegistry.LoadFromYaml(yaml);

        var predictionsRoot = Path.Combine(_models, "_preds");
        var anchor = WeatherBlend.Predict.PredictAnchor.Compute(DateTime.UtcNow, forDate: null);
        var partitionDir = Path.Combine(predictionsRoot, "temperature", "bonehill_rocks",
            "model_version=v_2b", $"date={anchor:yyyy-MM-dd}");
        Directory.CreateDirectory(partitionDir);
        var partition = Path.Combine(partitionDir, "predictions.parquet");
        await File.WriteAllTextAsync(partition, "x"); // existence is the signal for per-key-dir trees

        var cfg = new AppConfig
        {
            Storage = new StorageConfig { ModelsPath = _models, PredictionsPath = predictionsRoot },
            Locations = { new LocationConfig { Name = "bonehill_rocks" } },
        };
        var cmd = new PredictCoverageCommand(NullLogger<PredictCoverageCommand>.Instance, cfg);

        var ok = await cmd.RunAsync(forDate: null, locationOverride: null, registry, CancellationToken.None);
        ok.Should().Be(0);

        File.Delete(partition);
        var breach = await cmd.RunAsync(forDate: null, locationOverride: null, registry, CancellationToken.None);
        breach.Should().Be(5);
    }
}
