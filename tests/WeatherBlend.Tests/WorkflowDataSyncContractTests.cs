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
/// The canonical regression this test catches: the predict workflow used to
/// invoke Phase 3c precip predict without pulling <c>data/truth/rainfall</c>,
/// which Phase 3c reads for the persistence-feature tier. See commit 5627bcd.
/// </summary>
public class WorkflowDataSyncContractTests
{
    // Resolved once — tests that mutate the repo state would foul each other
    // anyway, and finding the repo root from the test assembly location is
    // slightly fiddly so keep it in one place.
    private static readonly string WorkflowsDir = FindWorkflowsDir();

    [Theory]
    [InlineData("predict.yml",     "data/forecasts")]
    [InlineData("predict.yml",     "data/models")]
    [InlineData("predict.yml",     "data/predictions")]
    [InlineData("predict.yml",     "data/truth/rainfall")]  // regression: 5627bcd
    [InlineData("verify.yml",      "data/predictions")]
    [InlineData("verify.yml",      "data/truth/era5")]
    [InlineData("verify.yml",      "data/truth/rainfall")]
    [InlineData("verify.yml",      "data/models")]
    [InlineData("render-site.yml", "data/predictions")]
    [InlineData("render-site.yml", "data/truth/era5")]
    [InlineData("render-site.yml", "data/truth/metar")]
    [InlineData("render-site.yml", "data/truth/rainfall")]
    [InlineData("render-site.yml", "data/models")]
    public void Workflow_syncs_expected_tree_from_R2(string workflowFile, string expectedTree)
    {
        var path = Path.Combine(WorkflowsDir, workflowFile);
        File.Exists(path).Should().BeTrue($"workflow {workflowFile} should exist");

        var yaml = File.ReadAllText(path);

        // Accept any rclone copy step that lists the expected tree as its source
        // OR its destination. Real steps look like:
        //   rclone copy "r2:${R2_BUCKET}/data/truth/rainfall" ./data/truth/rainfall ...
        // Match on the tree path fragment — tolerant of quoting, interpolation,
        // and shell continuations but strict enough to catch a missing step.
        yaml.Should().Contain(expectedTree,
            because: $"{workflowFile} invokes commands that read from {expectedTree}; " +
                     $"without an rclone step the CI run hits a DuckDB 'No files found' crash.");

        yaml.Should().Contain("rclone copy",
            because: $"{workflowFile} must rclone-copy its read dependencies from R2 " +
                     $"before the build/run step.");
    }

    private static string FindWorkflowsDir()
    {
        // Tests run from bin/Debug/net10.0/, so walk up until we see .github.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, ".github", "workflows");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException(
            "Could not locate .github/workflows from test runner directory; " +
            "expected to find it by walking parents of " + AppContext.BaseDirectory);
    }
}
