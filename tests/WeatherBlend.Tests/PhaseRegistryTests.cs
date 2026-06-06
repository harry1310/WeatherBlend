using FluentAssertions;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

public class PhaseRegistryTests
{
    /// <summary>
    /// The "main" YAML — three targets, mix of impls. Mirrors the
    /// production phases.yaml so tests catch a registry-shape regression
    /// too. Kept inline so a phase added to production doesn't silently
    /// break tests that don't actually exercise the change.
    /// </summary>
    private const string FullYaml = """
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
          precipitation:
            phases:
              - id: "3a"
                role: champion
                impl: dotnet
              - id: "3c"
                role: challenger
                impl: dotnet
              - id: "3d"
                role: challenger
                impl: dotnet
              - id: "4a"
                role: challenger
                impl: python
          dry_window:
            phases:
              - id: "3b"
                role: champion
                impl: dotnet
              - id: "3p"
                role: challenger
                impl: dotnet
        """;

    [Fact]
    public void ByTarget_returns_all_phases_in_yaml_order()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);

        reg.ByTarget["precipitation"].Should().BeEquivalentTo(
            new[] { "3a", "3c", "3d", "4a" },
            opts => opts.WithStrictOrdering());
    }

    [Fact]
    public void IsActive_returns_true_for_challenger_role_phase()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.IsActive("precipitation", "4a").Should().BeTrue();
    }

    [Fact]
    public void IsActive_returns_false_for_unknown_phase()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.IsActive("precipitation", "9z").Should().BeFalse();
    }

    [Fact]
    public void IsActive_returns_false_for_unknown_target()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.IsActive("wind", "anything").Should().BeFalse();
    }

    [Fact]
    public void Priority_is_zero_for_champion()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.Priority("precipitation", "3a").Should().Be(0);
    }

    [Fact]
    public void Priority_increases_along_challenger_chain()
    {
        // 3a=0, 3c=1, 3d=2, 4a=3 in YAML order.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.Priority("precipitation", "3a").Should().Be(0);
        reg.Priority("precipitation", "3c").Should().Be(1);
        reg.Priority("precipitation", "3d").Should().Be(2);
        reg.Priority("precipitation", "4a").Should().Be(3);
    }

    [Fact]
    public void AllPhases_returns_full_list_in_yaml_order()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.AllPhases("precipitation").Select(p => p.Id).Should().BeEquivalentTo(
            new[] { "3a", "3c", "3d", "4a" },
            opts => opts.WithStrictOrdering());
    }

    [Fact]
    public void PhasesForImpl_python_returns_only_python_phases()
    {
        // retrain-python.yml uses this query (transitively, via the YAML
        // it fetches). 4a is the only python phase.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        var pythonIds = reg.PhasesForImpl(PhaseImpl.Python)
            .Select(t => t.Phase.Id)
            .ToArray();
        pythonIds.Should().BeEquivalentTo(new[] { "4a" });
    }

    [Fact]
    public void PhasesForImpl_dotnet_returns_only_dotnet_phases()
    {
        // The mirror query — retrain-blenders.yml (slice 4) will use it.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        var dotnetIds = reg.PhasesForImpl(PhaseImpl.Dotnet)
            .Select(t => t.Phase.Id)
            .ToArray();
        dotnetIds.Should().BeEquivalentTo(new[] { "2b", "2c", "2d", "3a", "3c", "3d", "3b", "3p" });
    }

    [Fact]
    public void ByTargetAndLocation_excludes_a_phase_scoped_to_another_location()
    {
        // 2d carries `locations: [bonehill_rocks]`; 2b has no filter.
        // For membury_devon only 2b survives; for bonehill_rocks both do.
        const string yaml = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: dotnet
                  - id: "2d"
                    role: challenger
                    impl: dotnet
                    locations: ["bonehill_rocks"]
            """;
        var reg = PhaseRegistry.LoadFromYaml(yaml);

        reg.ByTargetAndLocation("temperature", "bonehill_rocks")
            .Should().BeEquivalentTo(new[] { "2b", "2d" }, o => o.WithStrictOrdering());
        reg.ByTargetAndLocation("temperature", "membury_devon")
            .Should().BeEquivalentTo(new[] { "2b" },
                "2d is scoped to bonehill_rocks; 2b has no filter so it applies everywhere");
        reg.ByTargetAndLocation("wind", "bonehill_rocks")
            .Should().BeEmpty("unknown target");
    }

    [Fact]
    public void AppliesToLocation_is_case_insensitive_and_true_when_filter_absent()
    {
        const string yaml = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: dotnet
                  - id: "2d"
                    role: challenger
                    impl: dotnet
                    locations: ["Bonehill_Rocks"]
            """;
        var reg = PhaseRegistry.LoadFromYaml(yaml);
        var phases = reg.AllPhases("temperature");

        phases.Single(p => p.Id == "2b").AppliesToLocation("anywhere").Should().BeTrue(
            "a phase with no locations: filter applies to every location");
        phases.Single(p => p.Id == "2d").AppliesToLocation("bonehill_rocks").Should().BeTrue(
            "location match is case-insensitive");
        phases.Single(p => p.Id == "2d").AppliesToLocation("membury_devon").Should().BeFalse();
    }

    [Fact]
    public void Default_phases_yaml_scopes_2d_and_3d_to_bonehill_only()
    {
        // The shipped phases.yaml pins the exact-runtime phases to
        // bonehill_rocks — they have no Membury S3 archive yet.
        var reg = PhaseRegistry.Default;
        reg.ByTargetAndLocation("temperature", "membury_devon").Should().NotContain("2d");
        reg.ByTargetAndLocation("temperature", "bonehill_rocks").Should().Contain("2d");
        reg.ByTargetAndLocation("precipitation", "membury_devon").Should().NotContain("3d");
        reg.ByTargetAndLocation("precipitation", "bonehill_rocks").Should().Contain("3d");
    }

    [Fact]
    public void LoadFromYaml_throws_when_targets_block_missing()
    {
        // Catch a copy-paste accident that nukes the targets block. We
        // want a loud failure at startup rather than a silent empty
        // registry that makes the Models page render nothing.
        Action act = () => PhaseRegistry.LoadFromYaml("foo: bar\n");
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*targets*");
    }

    [Fact]
    public void LoadFromYaml_throws_when_role_unknown()
    {
        const string bad = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: rumchamp
                    impl: dotnet
            """;
        Action act = () => PhaseRegistry.LoadFromYaml(bad);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*role*rumchamp*");
    }

    [Fact]
    public void LoadFromYaml_throws_when_impl_unknown()
    {
        const string bad = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: rust
            """;
        Action act = () => PhaseRegistry.LoadFromYaml(bad);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*impl*rust*");
    }

    [Fact]
    public void LoadFromYaml_throws_when_target_has_two_champions()
    {
        // Priority() would be ambiguous; better to fail at config-load
        // than silently let the second one shadow the first.
        const string bad = """
            targets:
              temperature:
                phases:
                  - id: "2b"
                    role: champion
                    impl: dotnet
                  - id: "2c"
                    role: champion
                    impl: dotnet
            """;
        Action act = () => PhaseRegistry.LoadFromYaml(bad);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*champion*");
    }

    [Fact]
    public void Default_loads_the_real_phases_yaml_from_disk()
    {
        // Smoke test on the real file shipped with the binary. If the
        // copy-to-output path or YAML schema breaks, the rest of the
        // suite stays green but production ActivePhasePolicy explodes
        // at startup. This test catches that.
        var reg = PhaseRegistry.Default;
        reg.ByTarget.Should().ContainKeys("temperature", "precipitation", "dry_window");
        reg.ByTarget["precipitation"].Should().Contain("4a");
    }

    [Fact]
    public void ActivePhasePolicy_facade_matches_registry_default()
    {
        // Existing call sites (SitePages.Models, SitePages.Skill, etc.)
        // hit the static façade. Make sure it still returns the same
        // shape the pre-YAML code did.
        ActivePhasePolicy.ByTarget.Should().BeEquivalentTo(PhaseRegistry.Default.ByTarget);
        ActivePhasePolicy.IsActive("precipitation", "3a").Should().BeTrue();
        // 3o is the precipitation champion since 2026-06-06 (3a demoted to
        // the universal fallback challenger). Champion → priority 0; 3a now
        // sits behind 3o (0) and 3c (1).
        ActivePhasePolicy.Priority("precipitation", "3o").Should().Be(0);
        ActivePhasePolicy.Priority("precipitation", "3a").Should().Be(2);
    }

    // ---- min_valid_time parsing (2026-05-26) ------------------------------

    [Fact]
    public void MinValidTime_absent_resolves_to_null()
    {
        // Existing phases without a min_valid_time entry must continue
        // to expose null — i.e. "no cutoff, train on the full history".
        // Regression: a parser that defaulted to 'now' or '1970-01-01'
        // would silently shift every existing phase's training window.
        var yaml = """
            targets:
              precipitation:
                phases:
                  - id: "3a"
                    role: champion
                    impl: dotnet
            """;
        var reg = PhaseRegistry.LoadFromYaml(yaml);
        var entry = reg.AllPhases("precipitation").Single(p => p.Id == "3a");
        entry.MinValidTime.Should().BeNull();
    }

    [Fact]
    public void MinValidTime_iso_date_parses_to_UTC_midnight()
    {
        // The shipping convention — a bare date string. Must parse to
        // midnight UTC; downstream feature-builder SQL embeds the value
        // as a DuckDB TIMESTAMP literal.
        var yaml = """
            targets:
              precipitation:
                phases:
                  - id: "3a"
                    role: champion
                    impl: dotnet
                    minValidTime: "2024-01-01"
            """;
        var reg = PhaseRegistry.LoadFromYaml(yaml);
        var entry = reg.AllPhases("precipitation").Single(p => p.Id == "3a");
        entry.MinValidTime.Should().Be(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc));
        entry.MinValidTime!.Value.Kind.Should().Be(DateTimeKind.Utc);
    }

    [Fact]
    public void MinValidTime_invalid_string_throws_with_useful_message()
    {
        // Defensive: a typo in the YAML must NOT silently become "no
        // cutoff". The user has to see the breakage. The thrown message
        // must include the target + phase id so debugging from a CI log
        // is one grep away.
        var bad = """
            targets:
              precipitation:
                phases:
                  - id: "3a"
                    role: champion
                    impl: dotnet
                    minValidTime: "not a date"
            """;
        Action act = () => PhaseRegistry.LoadFromYaml(bad);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*precipitation*3a*not a date*");
    }

    [Fact]
    public void MinValidTime_production_yaml_pins_2024_cutoff_on_affected_phases()
    {
        // Locks the 2026-05-26 cutoff decision in the shipped YAML.
        // Any future edit that removes the field from these phases will
        // trip this test, which is the intended forcing function — the
        // cutoff is load-bearing for 3c / 3b post-JMA-extension and
        // shouldn't be silently dropped.
        var reg = PhaseRegistry.Default;
        var expectedCutoff = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        foreach (var (target, phase) in new[]
        {
            ("temperature",    "2b"),
            ("temperature",    "2c"),
            ("precipitation",  "3a"),
            ("precipitation",  "3c"),
            ("precipitation",  "4a"),
            ("dry_window",     "3b"),
        })
        {
            var entry = reg.AllPhases(target).SingleOrDefault(p => p.Id == phase);
            entry.Should().NotBeNull($"phases.yaml must still declare {target}/{phase}");
            entry!.MinValidTime.Should().Be(expectedCutoff,
                $"{target}/{phase} must carry the 2024 cutoff " +
                "(see project_3a_3b_data_drift_2026-05-26 memory).");
        }

        // 3o is the explicit non-exception — its terrain + non-precip aux
        // features stay informative pre-2024, so it keeps the full range.
        var threeOh = reg.AllPhases("precipitation").Single(p => p.Id == "3o");
        threeOh.MinValidTime.Should().BeNull(
            "3o is the sole forecast-tree phase exempt from the 2024 cutoff.");
    }
}
