using System.Text.RegularExpressions;
using FluentAssertions;
using Xunit;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// Guards verify.yml's R2 pull step. Two failure modes this locks down:
///
///  1. COMPLETENESS — verify must pull every tree it reads (predictions, era5,
///     rainfall, model metadata, reports). Dropping one makes verify silently
///     score nothing for that target (the same class as the sync-render-inputs
///     regressions).
///
///  2. WEIGHT — the data/models pull MUST exclude the per-cell 4a BART artefacts
///     (state_lead_*.rds / arrays_lead_*.npz, ~80% of the tree's bytes). verify
///     reads only the json metadata + reports, never the .rds/.npz. The
///     whole-tree copy crept past the 15-min job cap and the run was cancelled
///     mid-pull on 2026-06-01; this assertion stops anyone re-introducing it.
///
/// A static guard can't catch a purely data-growth-driven timeout, but it pins
/// the fix (the exclusion) + the dependency set so neither silently regresses.
/// </summary>
public class VerifyPullSmokeTests
{
    private static string LocateVerifyWorkflow()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, ".github", "workflows", "verify.yml");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new FileNotFoundException(
            ".github/workflows/verify.yml not found by ascent from " + AppContext.BaseDirectory);
    }

    private static readonly (string Tree, string Reader)[] RequiredTrees =
    {
        ("data/predictions",   "the predictions being scored"),
        ("data/truth/era5",    "temperature verify join key"),
        ("data/truth/rainfall","precipitation / dry-window verify join key"),
        ("data/truth/waves",   "wave_height verify join key (buoy Hs)"),
        ("data/models",        "training_metadata.json + MANIFEST.json drift thresholds"),
        ("data/reports",       "existing verify_*.json sidecars (additive writes)"),
    };

    [Fact]
    public void Verify_pulls_every_tree_it_reads()
    {
        var yaml = File.ReadAllText(LocateVerifyWorkflow());
        var missing = RequiredTrees
            .Where(t => !yaml.Contains($"r2:${{R2_BUCKET}}/{t.Tree}", StringComparison.Ordinal))
            .Select(t => $"'{t.Tree}' ({t.Reader})")
            .ToList();
        missing.Should().BeEmpty(
            "verify.yml must rclone every tree the verify command reads — a dropped tree silently "
            + "scores nothing for that target. Missing:\n  " + string.Join("\n  ", missing));
    }

    [Fact]
    public void Verify_models_pull_excludes_heavy_bart_artefacts()
    {
        var yaml = File.ReadAllText(LocateVerifyWorkflow());

        // Find the data/models rclone copy and the flags up to the next blank
        // line (the multi-line `\`-continued command).
        var m = Regex.Match(
            yaml,
            @"rclone copy ""r2:\$\{R2_BUCKET\}/data/models""(?<flags>(?:.|\n)*?)\n\s*\n",
            RegexOptions.None);
        m.Success.Should().BeTrue("verify.yml should contain a data/models rclone copy");
        var flags = m.Groups["flags"].Value;

        flags.Should().Contain("*.rds",
            "the data/models pull must --exclude '**/*.rds' (4a BART states verify never reads) — "
            + "without it the pull exceeds the job timeout, as on 2026-06-01.");
        flags.Should().Contain("*.npz",
            "the data/models pull must --exclude '**/*.npz' (4a BART arrays verify never reads).");
    }
}
