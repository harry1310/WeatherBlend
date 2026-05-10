using FluentAssertions;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

public class PhaseRegistryTests
{
    /// <summary>
    /// The "main" YAML — three targets, mix of impls, 5a as confidence.
    /// Mirrors the production phases.yaml so tests catch a registry-shape
    /// regression too. Kept inline so a phase added to production doesn't
    /// silently break tests that don't actually exercise the change.
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
              - id: "5a"
                role: confidence
                impl: python
          dry_window:
            phases:
              - id: "3b"
                role: champion
                impl: dotnet
              - id: "3g"
                role: challenger
                impl: dotnet
        """;

    [Fact]
    public void ByTarget_excludes_confidence_phases()
    {
        // 5a is role=confidence, and the Models page must not see it as a
        // prediction line. Same for any future confidence-role phases.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);

        reg.ByTarget["precipitation"].Should().BeEquivalentTo(
            new[] { "3a", "3c", "3d", "4a" },
            opts => opts.WithStrictOrdering());
    }

    [Fact]
    public void IsActive_returns_false_for_confidence_role_phase()
    {
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.IsActive("precipitation", "5a").Should().BeFalse(
            "5a is a CI overlay, not a prediction line");
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
        // 3a=0, 3c=1, 3d=2, 4a=3 — confidence-role 5a is skipped (counted
        // as MaxValue) so Priority indices stay dense across the
        // active-line list.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.Priority("precipitation", "3a").Should().Be(0);
        reg.Priority("precipitation", "3c").Should().Be(1);
        reg.Priority("precipitation", "3d").Should().Be(2);
        reg.Priority("precipitation", "4a").Should().Be(3);
        reg.Priority("precipitation", "5a").Should().Be(int.MaxValue,
            "confidence-role phases are excluded from priority sort");
    }

    [Fact]
    public void AllPhases_includes_confidence_in_yaml_order()
    {
        // Train workflows enumerate AllPhases — they MUST see 5a even
        // though it doesn't get a Models card. Otherwise the Sunday
        // sweep silently skips retraining the CI band.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        reg.AllPhases("precipitation").Select(p => p.Id).Should().BeEquivalentTo(
            new[] { "3a", "3c", "3d", "4a", "5a" },
            opts => opts.WithStrictOrdering());
    }

    [Fact]
    public void PhasesForImpl_python_returns_only_python_phases()
    {
        // retrain-python.yml uses this query (transitively, via the YAML
        // it fetches). 4a + 5a are python; everything else is dotnet.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        var pythonIds = reg.PhasesForImpl(PhaseImpl.Python)
            .Select(t => t.Phase.Id)
            .ToArray();
        pythonIds.Should().BeEquivalentTo(new[] { "4a", "5a" });
    }

    [Fact]
    public void PhasesForImpl_dotnet_returns_only_dotnet_phases()
    {
        // The mirror query — retrain-blenders.yml (slice 4) will use it.
        var reg = PhaseRegistry.LoadFromYaml(FullYaml);
        var dotnetIds = reg.PhasesForImpl(PhaseImpl.Dotnet)
            .Select(t => t.Phase.Id)
            .ToArray();
        dotnetIds.Should().BeEquivalentTo(new[] { "2b", "2c", "2d", "3a", "3c", "3d", "3b", "3g" });
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
        reg.AllPhases("precipitation").Should().Contain(p => p.Id == "5a" && p.Role == PhaseRole.Confidence);
    }

    [Fact]
    public void ActivePhasePolicy_facade_matches_registry_default()
    {
        // Existing call sites (SitePages.Models, SitePages.Skill, etc.)
        // hit the static façade. Make sure it still returns the same
        // shape the pre-YAML code did.
        ActivePhasePolicy.ByTarget.Should().BeEquivalentTo(PhaseRegistry.Default.ByTarget);
        ActivePhasePolicy.IsActive("precipitation", "3a").Should().BeTrue();
        ActivePhasePolicy.IsActive("precipitation", "5a").Should().BeFalse();
        ActivePhasePolicy.Priority("precipitation", "3a").Should().Be(0);
    }
}
