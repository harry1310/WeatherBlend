using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins <see cref="DryWindowPredictCommand.PhaseRequiresClimatology"/>,
/// the helper that defends the 2026-05-26 3p-rejection regression.
///
/// The bug: a <c>File.Exists("dry_window_climatology.json")</c> check
/// sat above the 3p phase dispatch in <c>RunAsync</c>. Every 3p bundle
/// (which ships only <c>correlation.json</c> + <c>training_metadata.json</c>
/// by design — see <see cref="DryWindow3pPredictor"/>'s docstring)
/// was rejected with "missing climatology … skipping" before predict
/// even noticed the phase tag. Two retrain sweeps shipped 0 3p
/// predictions on R2.
///
/// The fix hoisted the 3p dispatch above the check (commit 4f3b445)
/// AND tightened the <c>File.Exists</c> guard to consult this helper
/// so a future re-order can't silently re-introduce the same bug —
/// see the in-source comment block.
/// </summary>
public class DryWindowPredictCommandTests
{
    [Fact]
    public void PhaseRequiresClimatology_3p_returns_false()
    {
        // Regression: 3p ships no climatology, must NOT trip the
        // File.Exists guard. The constant comes from the predictor type
        // to defend against a future rename of "3p".
        DryWindowPredictCommand
            .PhaseRequiresClimatology(DryWindow3pPredictor.Phase3p)
            .Should().BeFalse();
    }

    [Theory]
    [InlineData("3b")]    // legacy LightGBM dry-window — climatology mandatory
    [InlineData("3d")]    // exact-runtime mirror of 3b — same artefact set
    [InlineData("3z")]    // hypothetical future phase — defaults to "requires"
    [InlineData("")]      // empty/unknown phase — safer default is "requires"
    public void PhaseRequiresClimatology_returns_true_for_every_other_phase(string phase)
    {
        // Default posture is "requires climatology" — only the explicitly-
        // exempt phases get a pass. If a new climatology-free phase ships
        // it must be added here AND given a dispatch branch above the
        // File.Exists check in RunAsync. Failing this test is the
        // intended signal that both edits are needed.
        DryWindowPredictCommand
            .PhaseRequiresClimatology(phase)
            .Should().BeTrue();
    }
}
