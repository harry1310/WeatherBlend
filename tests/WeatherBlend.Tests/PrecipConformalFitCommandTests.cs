using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Regression test for the 2026-05-26 retrain failure where adding Phase 3o
/// to phases.yaml without wiring a matching arm in PrecipConformalFitCommand
/// caused the post-train auto-conformal-fit hook to fall through to the
/// lean PrecipFeatureBuilder and crash with
///
///   System.InvalidOperationException: Feature pack mismatch: wrote 23, expected 68
///
/// Coverage rule: every active impl=dotnet precipitation phase in
/// phases.yaml MUST either be in <see cref="PrecipConformalFitCommand.HandledPhases"/>
/// (meaning the per-lead feature-builder dispatch has a matching arm) OR in
/// <see cref="PrecipConformalFitCommand.DocumentedSkipPhases"/> (meaning the
/// command knowingly skips it — currently 3d only, pending exact-runtime
/// feature-builder plumbing).
///
/// Adding a new precipitation phase that triggers conformal fit requires
/// editing TWO places: phases.yaml + PrecipConformalFitCommand.HandledPhases
/// (and the dispatch cascade inside FitOneAsync). This test surfaces step 2
/// when step 1 lands without it.
/// </summary>
public class PrecipConformalFitCommandTests
{
    [Fact]
    public void HandledPhases_or_DocumentedSkip_covers_every_active_dotnet_precipitation_phase()
    {
        // PhaseRegistry.Default loads the real phases.yaml at startup — this
        // test exercises the actual production config, not a fixture.
        var allPrecipPhases = PhaseRegistry.Default.AllPhases("precipitation");

        // The conformal fit is invoked from PrecipTrainCommand's train paths
        // after a successful promote. Only impl=dotnet phases reach the
        // .NET trainer; impl=python phases (4a) are served by their own
        // WP-side workflows that handle calibration independently.
        var dotnetPhases = allPrecipPhases
            .Where(p => p.Impl == PhaseImpl.Dotnet)
            .Select(p => p.Id)
            .ToHashSet(StringComparer.Ordinal);

        var coverage = PrecipConformalFitCommand.HandledPhases
            .Concat(PrecipConformalFitCommand.DocumentedSkipPhases)
            .ToHashSet(StringComparer.Ordinal);

        // Every impl=dotnet precip phase must be covered. The error
        // message tells the dev exactly what to do.
        var uncovered = dotnetPhases.Except(coverage).ToList();
        uncovered.Should().BeEmpty(
            "every active impl=dotnet precipitation phase in phases.yaml MUST be in either " +
            "PrecipConformalFitCommand.HandledPhases (with a matching feature-builder arm in " +
            "FitOneAsync's per-lead loop) OR PrecipConformalFitCommand.DocumentedSkipPhases " +
            "(documented as not-yet-supported, like 3d's exact-runtime feature builder). " +
            "Uncovered: [{0}]. Adding a phase to phases.yaml without doing this leaves the " +
            "per-lead dispatch falling through to the wrong feature builder — see the " +
            "2026-05-26 retrain failure that landed 3o without a HandledPhases entry and " +
            "crashed with 'Feature pack mismatch: wrote 23, expected 68'.",
            string.Join(", ", uncovered));
    }

    [Fact]
    public void HandledPhases_and_DocumentedSkip_are_disjoint()
    {
        // Sanity: a phase that's "handled" is by definition not "documented
        // skip", and vice versa. Both lists should be deliberate.
        PrecipConformalFitCommand.HandledPhases
            .Intersect(PrecipConformalFitCommand.DocumentedSkipPhases)
            .Should().BeEmpty(
                "a phase must be exactly one of: HandledPhases (has dispatch arm) " +
                "or DocumentedSkipPhases (deliberately skipped). Both means the " +
                "intent is unclear.");
    }
}
