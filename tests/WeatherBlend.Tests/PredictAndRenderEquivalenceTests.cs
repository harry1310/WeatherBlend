using System.Text.RegularExpressions;
using FluentAssertions;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Contract test: the legacy fused workflow predict-and-render.yml MUST
/// stay behaviourally identical to predict.yml + render-site.yml. It is
/// kept in the repo as the revert path for the predict/predict-3f/render
/// split (see cloudflare/scheduler-worker/src/index.ts hops D / F.1 / F);
/// if those hops break, production reverts by flipping Hop D back at
/// predict-and-render.yml and disabling F.1+F. That "revert" is only
/// safe if the fused workflow does EXACTLY what the split chain does.
///
/// Divergence surface this test pins:
///   1. predict.yml uses predict-all + predict-tail composites.
///   2. render-site.yml uses render-and-deploy composite.
///   3. predict-and-render.yml MUST reference every composite from (1) + (2).
///   4. predict-and-render.yml MUST NOT contain inline shell that
///      replicates a step that lives in a composite — that's how drift
///      historically crept in (a step gets edited in one workflow and
///      the inline duplicate in the other gets missed).
///
/// Canonical regressions this defends:
///   * 2026-05-26 cleanup: extracted predict-tail + render-and-deploy
///     composites specifically because the inline 4b synthesis step
///     was hand-duplicated in both predict.yml and predict-and-render.yml
///     and trivially divergeable. This test was added the same day to
///     stop a future edit from undoing the extraction.
/// </summary>
public class PredictAndRenderEquivalenceTests
{
    private static readonly string WorkflowsDir = FindRepoSubdir(Path.Combine(".github", "workflows"));

    [Fact]
    public void PredictAndRender_uses_every_composite_predict_uses()
    {
        var predictComposites = CompositeActionsUsedBy("predict.yml");
        var fusedComposites = CompositeActionsUsedBy("predict-and-render.yml");

        // Every composite the split predict.yml depends on must be present
        // in the fused workflow. Anything missing is divergence — the
        // fused workflow would be doing strictly less than predict.yml
        // and the revert wouldn't be a true revert.
        predictComposites.Should().BeSubsetOf(fusedComposites,
            because: "predict-and-render.yml is the revert target for predict.yml; " +
                     "every composite predict.yml uses must also be used here so the fused " +
                     "workflow does at least everything the split predict step does.");
    }

    [Fact]
    public void PredictAndRender_uses_every_composite_render_site_uses()
    {
        var renderComposites = CompositeActionsUsedBy("render-site.yml");
        var fusedComposites = CompositeActionsUsedBy("predict-and-render.yml");

        renderComposites.Should().BeSubsetOf(fusedComposites,
            because: "predict-and-render.yml is the revert target for render-site.yml; " +
                     "every composite render-site.yml uses must also be used here.");
    }

    [Theory]
    // Steps that USED to live as inline bash in predict-and-render.yml +
    // its split equivalents. After the 2026-05-26 extraction they live
    // ONLY in composites. If any of these strings reappears inline in
    // the workflow body, someone has unwound the extraction.
    [InlineData("phase4b-predict",       "predict-tail")]
    [InlineData("render-site $ARGS",     "render-and-deploy")]
    [InlineData("pages deploy data/site","render-and-deploy")]
    public void PredictAndRender_has_no_inline_step_duplicating_a_composite(string forbiddenInline, string canonicalComposite)
    {
        var workflow = File.ReadAllText(Path.Combine(WorkflowsDir, "predict-and-render.yml"));

        // Strip out the `uses: ./.github/actions/<name>` lines so we don't
        // false-positive on the composite reference itself (its name often
        // mentions the same keywords).
        var withoutComposites = Regex.Replace(workflow, @"uses:\s*\./\.github/actions/[A-Za-z0-9_\-]+", "");

        withoutComposites.Should().NotContain(forbiddenInline,
            because: $"'{forbiddenInline}' lives in the {canonicalComposite} composite (.github/actions/{canonicalComposite}/action.yml). " +
                     $"Inlining it in predict-and-render.yml is the exact divergence pattern this test exists to block — " +
                     $"the next edit to {canonicalComposite}'s logic would silently miss this inline copy. " +
                     $"Use `uses: ./.github/actions/{canonicalComposite}` instead.");
    }

    [Fact]
    public void All_three_workflows_use_the_same_sync_render_inputs_composite()
    {
        // The R2-sync composite was the FIRST extraction (2026-05-01); make
        // sure all three workflows still consume it. If render-site.yml ever
        // started inlining its own rclone steps, the drift would be silent
        // because no command would crash — it'd just produce stale output.
        foreach (var workflow in new[] { "predict.yml", "render-site.yml", "predict-and-render.yml" })
        {
            var composites = CompositeActionsUsedBy(workflow);
            composites.Should().Contain("sync-render-inputs",
                because: $"{workflow} must consume the shared R2-sync composite, not inline its own rclone steps.");
        }
    }

    private static HashSet<string> CompositeActionsUsedBy(string workflowFile)
    {
        var path = Path.Combine(WorkflowsDir, workflowFile);
        File.Exists(path).Should().BeTrue($"workflow {workflowFile} should exist");
        var yaml = File.ReadAllText(path);
        return Regex.Matches(yaml, @"uses:\s*\./\.github/actions/([A-Za-z0-9_\-]+)")
            .Select(m => m.Groups[1].Value)
            .ToHashSet(StringComparer.Ordinal);
    }

    private static string FindRepoSubdir(string relativePath)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, relativePath);
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException(
            $"Could not locate {relativePath} from test runner directory; " +
            $"expected to find it by walking parents of {AppContext.BaseDirectory}");
    }
}
