using System.Text.RegularExpressions;
using FluentAssertions;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Contract test for the .github/workflows/*.yml rclone-sync steps. Each workflow
/// invokes a set of commands; those commands read from a set of on-disk trees
/// (forecasts, predictions, models, truth/era5, truth/metar, truth/rainfall,
/// reports). If a workflow runs a command that reads a tree, the workflow must
/// pull that tree from R2 before the build step, otherwise the command crashes
/// in CI with an opaque DuckDB "No files found" error.
///
/// Composite-action awareness: workflows can delegate the actual rclone copy
/// to a composite action under <c>.github/actions/&lt;name&gt;</c>. The test
/// flattens the workflow file with the bodies of every <c>uses: ./.github/...</c>
/// it references, so the assertion holds whether the rclone step is inline or
/// inside the composite. Without this, the 2026-05-01 refactor that moved the
/// shared sync into <c>sync-render-inputs</c> would have broken the test.
///
/// Canonical regressions this test catches:
///   * 5627bcd — predict workflow lacked <c>data/truth/rainfall</c>
///   * 2026-05-01 — render-site workflow lacked <c>data/reports</c>
///     (the symptom: Models-page "Verify history" sections silently absent
///     because <c>VerifyHistoryFile</c> sidecars never landed on the runner).
/// </summary>
public class WorkflowDataSyncContractTests
{
    private static readonly string WorkflowsDir = FindRepoSubdir(Path.Combine(".github", "workflows"));
    private static readonly string ActionsDir   = FindRepoSubdir(Path.Combine(".github", "actions"));

    [Theory]
    [InlineData("predict.yml",             "data/forecasts")]
    [InlineData("predict.yml",             "data/models")]
    [InlineData("predict.yml",             "data/predictions")]
    [InlineData("predict.yml",             "data/truth/rainfall")]   // regression: 5627bcd
    [InlineData("verify.yml",              "data/predictions")]
    [InlineData("verify.yml",              "data/truth/era5")]
    [InlineData("verify.yml",              "data/truth/rainfall")]
    [InlineData("verify.yml",              "data/models")]
    [InlineData("render-site.yml",         "data/predictions")]
    [InlineData("render-site.yml",         "data/truth/era5")]
    [InlineData("render-site.yml",         "data/truth/metar")]
    [InlineData("render-site.yml",         "data/truth/rainfall")]
    [InlineData("render-site.yml",         "data/models")]
    [InlineData("render-site.yml",         "data/reports")]          // regression: 2026-05-01
    [InlineData("predict-and-render.yml",  "data/forecasts")]
    [InlineData("predict-and-render.yml",  "data/predictions")]
    [InlineData("predict-and-render.yml",  "data/truth/era5")]
    [InlineData("predict-and-render.yml",  "data/truth/metar")]
    [InlineData("predict-and-render.yml",  "data/truth/rainfall")]
    [InlineData("predict-and-render.yml",  "data/models")]
    [InlineData("predict-and-render.yml",  "data/reports")]
    public void Workflow_syncs_expected_tree_from_R2(string workflowFile, string expectedTree)
    {
        var path = Path.Combine(WorkflowsDir, workflowFile);
        File.Exists(path).Should().BeTrue($"workflow {workflowFile} should exist");

        var flattened = FlattenWithUsedActions(path);

        // Accept any rclone copy that lists the expected tree as its source
        // OR its destination. Real steps look like:
        //   rclone copy "r2:${R2_BUCKET}/data/truth/rainfall" ./data/truth/rainfall ...
        // Match on the tree path fragment — tolerant of quoting, interpolation,
        // and shell continuations but strict enough to catch a missing step.
        flattened.Should().Contain(expectedTree,
            because: $"{workflowFile} (with composite actions inlined) must rclone-copy {expectedTree}; " +
                     $"without it the CI run hits a DuckDB 'No files found' crash.");

        flattened.Should().Contain("rclone copy",
            because: $"{workflowFile} must rclone-copy its read dependencies from R2 " +
                     $"before the build/run step (inline or via a composite action).");
    }

    /// <summary>
    /// Returns the workflow YAML concatenated with the action.yml of every
    /// composite action it references via <c>uses: ./.github/actions/&lt;name&gt;</c>.
    /// One pass — composite actions don't currently use other composite actions in
    /// a way that affects the rclone surface, so a single level is enough. If that
    /// changes, this should grow into a recursive walk.
    /// </summary>
    private static string FlattenWithUsedActions(string workflowPath)
    {
        var yaml = File.ReadAllText(workflowPath);
        var actionNames = Regex.Matches(yaml, @"uses:\s*\./\.github/actions/([A-Za-z0-9_\-]+)")
            .Select(m => m.Groups[1].Value)
            .Distinct(StringComparer.Ordinal)
            .ToList();

        var combined = new System.Text.StringBuilder(yaml);
        foreach (var name in actionNames)
        {
            var actionPath = Path.Combine(ActionsDir, name, "action.yml");
            if (File.Exists(actionPath))
            {
                combined.AppendLine();
                combined.AppendLine($"# --- inlined from .github/actions/{name}/action.yml ---");
                combined.AppendLine(File.ReadAllText(actionPath));
            }
        }
        return combined.ToString();
    }

    private static string FindRepoSubdir(string relativePath)
    {
        // Tests run from bin/Debug/net10.0/, so walk up until we find the dir.
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
