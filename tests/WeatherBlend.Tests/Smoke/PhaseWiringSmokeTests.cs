using System.Text.RegularExpressions;
using FluentAssertions;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// Keeps the active-phase registry (phases.yaml) in lockstep with the satellite
/// lists that MUST track it but live in separate files. A phase trained by the
/// .NET retrain (impl=dotnet, retrain != none) has to have:
///   * a case entry in scripts/sync_train_data.sh  (so its data trees are pulled), and
///   * a steps.phases.outputs.run_&lt;id&gt; gate in retrain-blenders.yml (so a
///     train/mint step actually runs for it).
///
/// Drift between these is invisible to the rest of the suite and only detonates
/// in a live Sunday retrain — which is exactly what happened 2026-05-31 when
/// `wind_blend` was added to phases.yaml as impl=dotnet without either wiring
/// (sync_train_data.sh exit 3). `wind_blend` is now `retrain: none` (it's
/// composed at predict time), so it's correctly excluded. This test turns that
/// class of silent drift into a PR-time / smoke-time failure.
///
/// Pure file inspection — no fixture, no R2, no build artefacts. Reads the
/// production phases.yaml + the two satellite files straight from the repo.
/// </summary>
[Trait("Category", "Smoke")]
public class PhaseWiringSmokeTests
{
    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "src", "WeatherBlend", "Config", "phases.yaml")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new FileNotFoundException(
            "repo root (src/WeatherBlend/Config/phases.yaml) not found by ascent from "
            + AppContext.BaseDirectory);
    }

    private static PhaseRegistry LoadRegistry(string repo)
        => PhaseRegistry.LoadFromYaml(File.ReadAllText(
            Path.Combine(repo, "src", "WeatherBlend", "Config", "phases.yaml")));

    /// <summary>Phase ids handled by sync_train_data.sh's case table. Case arms
    /// look like <c>2b|2c|wind|wind_gust_lgb)</c> or <c>3o)</c>; <c>all)</c> is
    /// the union convenience case, not a phase.</summary>
    private static HashSet<string> SyncTrainDataLabels(string repo)
    {
        var text = File.ReadAllText(Path.Combine(repo, "scripts", "sync_train_data.sh"));
        var labels = new HashSet<string>(StringComparer.Ordinal);
        foreach (Match m in Regex.Matches(text, @"^\s*([a-z0-9_]+(?:\|[a-z0-9_]+)*)\)", RegexOptions.Multiline))
            foreach (var tok in m.Groups[1].Value.Split('|'))
                labels.Add(tok);
        labels.Remove("all");
        return labels;
    }

    /// <summary>Phase ids that have a static train/mint step in retrain-blenders.yml,
    /// read from the <c>steps.phases.outputs.run_&lt;id&gt;</c> gate references (the
    /// dynamic <c>echo "run_${id}="</c> emit and the <c>run_id</c> shell var don't
    /// match this prefixed pattern).</summary>
    private static HashSet<string> RetrainBlendersRunGates(string repo)
    {
        var text = File.ReadAllText(Path.Combine(repo, ".github", "workflows", "retrain-blenders.yml"));
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (Match m in Regex.Matches(text, @"steps\.phases\.outputs\.run_([a-z0-9_]+)"))
            ids.Add(m.Groups[1].Value);
        return ids;
    }

    [Fact]
    public void Every_trained_dotnet_phase_is_wired_in_sync_and_retrain_blenders()
    {
        var repo = RepoRoot();
        var reg = LoadRegistry(repo);
        var sync = SyncTrainDataLabels(repo);
        var runGates = RetrainBlendersRunGates(repo);

        var trainedDotnet = reg.EnumerateAll()
            .Where(t => t.Phase.Impl == PhaseImpl.Dotnet && t.Phase.IsRetrained)
            .Select(t => t.Phase.Id)
            .Distinct()
            .ToList();

        trainedDotnet.Should().NotBeEmpty("phases.yaml must define at least one trained dotnet phase");

        var violations = new List<string>();
        foreach (var id in trainedDotnet)
        {
            if (!sync.Contains(id))
                violations.Add(
                    $"phase '{id}' is a trained dotnet phase but has NO case entry in " +
                    "scripts/sync_train_data.sh — its data trees won't be pulled and the retrain exits 3.");
            if (!runGates.Contains(id))
                violations.Add(
                    $"phase '{id}' is a trained dotnet phase but has NO steps.phases.outputs.run_{id} " +
                    "gate in retrain-blenders.yml — no train/mint step will run for it.");
        }

        violations.Should().BeEmpty(
            "phases.yaml must stay in lockstep with sync_train_data.sh + retrain-blenders.yml. " +
            "Either wire the phase in both, or set `retrain: none` if it's composed at predict time.\n"
            + string.Join("\n", violations));
    }

    [Fact]
    public void Every_retrain_blenders_run_gate_maps_to_a_dotnet_phase()
    {
        var repo = RepoRoot();
        var reg = LoadRegistry(repo);
        var runGates = RetrainBlendersRunGates(repo);
        var dotnetPhaseIds = reg.EnumerateAll()
            .Where(t => t.Phase.Impl == PhaseImpl.Dotnet)
            .Select(t => t.Phase.Id)
            .ToHashSet(StringComparer.Ordinal);

        var orphans = runGates.Where(id => !dotnetPhaseIds.Contains(id)).ToList();

        orphans.Should().BeEmpty(
            "every run_<id> train gate in retrain-blenders.yml should map to an impl=dotnet phase in "
            + "phases.yaml — a leftover gate for a removed phase is dead config. Orphans: "
            + string.Join(", ", orphans));
    }
}
