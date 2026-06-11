using System.Text.RegularExpressions;
using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Wiring-consistency contract for <c>Config/phases.yaml</c> — the test the
/// registry's header comment points at. phases.yaml is the REGISTRY, not
/// the whole wiring: a new phase also needs a retrain workflow step, a
/// <c>sync_train_data.sh</c> case entry, a predict handled-set entry and a
/// presentation-table row. Each of those hooks fails in its own delayed,
/// quiet way when missed (the 2026-05-31 retrain broke on a phase added
/// without <c>retrain: none</c>; 2026-05-28 wind_speed_lgb trained on
/// "Loaded 0 rows" when its data pull was missing), so this test pins all
/// of them to the registry at PR time.
///
/// The registry is loaded via <see cref="PhaseRegistry.Default"/> — the
/// same loader production uses (csproj copies phases.yaml to the test
/// output) — NOT a parallel YAML parse that could drift. The workflow and
/// shell script are read from the repo tree, mirroring
/// <see cref="WorkflowDataSyncContractTests"/>'s walk-up pattern.
/// </summary>
public class PhaseWiringConsistencyTests
{
    private static readonly PhaseRegistry Registry = PhaseRegistry.Default;

    /// <summary>
    /// impl=dotnet phases that a retrain actually produces (train or mint).
    /// <c>retrain: none</c> phases (wind_blend — composed live at predict
    /// time, no bundle) are deliberately excluded: the workflow's yq fanout
    /// filters them out before any train step or data pull is considered.
    /// </summary>
    private static IReadOnlyList<(string Target, PhaseEntry Phase)> RetrainedDotnetPhases =>
        Registry.PhasesForImpl(PhaseImpl.Dotnet)
            .Where(t => t.Phase.IsRetrained)
            .ToList();

    // ------------------------------------------------------------------
    // retrain-blenders.yml — every trained dotnet phase has a gated step
    // ------------------------------------------------------------------

    [Fact]
    public void Every_retrained_dotnet_phase_has_a_run_gate_in_retrain_blenders()
    {
        var workflow = File.ReadAllText(RepoFile(".github", "workflows", "retrain-blenders.yml"));

        foreach (var (target, phase) in RetrainedDotnetPhases)
        {
            // Marker = the per-step gate condition. The "Decide which phases
            // to run" step emits run_<id> dynamically for every yaml phase,
            // so the only literal `steps.phases.outputs.run_<id> ==` in the
            // file is each train/mint step's `if:` — exactly the hook a new
            // phase must add. The trailing " ==" stops run_wind matching
            // run_wind_gust_lgb's gate.
            var marker = $"steps.phases.outputs.run_{phase.Id} ==";
            workflow.Should().Contain(marker,
                because: $"phases.yaml registers {target}/{phase.Id} as impl=dotnet with a retrain " +
                         $"step, so retrain-blenders.yml needs a train step gated on run_{phase.Id} " +
                         "(otherwise the Sunday sweep silently never retrains it)");
        }
    }

    // ------------------------------------------------------------------
    // scripts/sync_train_data.sh — every trained dotnet phase declares
    // its data pulls (the script exits 3 on unknown ids)
    // ------------------------------------------------------------------

    [Fact]
    public void Every_retrained_dotnet_phase_is_in_sync_train_data_case_table()
    {
        var caseLabels = SyncTrainDataCaseLabels();

        foreach (var (target, phase) in RetrainedDotnetPhases)
        {
            caseLabels.Should().Contain(phase.Id,
                because: $"phases.yaml registers {target}/{phase.Id} as a trained dotnet phase; " +
                         "scripts/sync_train_data.sh exits 3 on ids missing from its case table, " +
                         "which would fail the retrain job's Pull step at run time");
        }
        // The inverse rule for composed phases (wind_blend, retrain: none)
        // is exclusion-from-fanout, not a case entry: the workflow's yq
        // filter drops them before the script is ever invoked, so they
        // intentionally have no entry here. impl=python phases likewise
        // declare their pulls in WeatherProbabilistic's own sync script.
    }

    /// <summary>
    /// Parses the phase ids out of the script's case table. Pattern lines
    /// look like <c>2b|2c|wind)</c> or <c>3o)</c> on their own line; the
    /// <c>all)</c> alias and the <c>*)</c> unknown-id catch-all are not
    /// phase ids and are excluded.
    /// </summary>
    private static HashSet<string> SyncTrainDataCaseLabels()
    {
        var script = File.ReadAllText(RepoFile("scripts", "sync_train_data.sh"));
        var labels = new HashSet<string>(StringComparer.Ordinal);
        foreach (Match m in Regex.Matches(script, @"^\s*([A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)\)\s*$", RegexOptions.Multiline))
        {
            foreach (var id in m.Groups[1].Value.Split('|'))
            {
                if (id != "all") labels.Add(id);
            }
        }
        labels.Should().NotBeEmpty("the case-table parse should find the PER-PHASE NEEDS TABLE; " +
                                   "if sync_train_data.sh's layout changed, update this regex");
        return labels;
    }

    // ------------------------------------------------------------------
    // Predict handled-sets (the L2 gate) stay subsets of the registry
    // ------------------------------------------------------------------

    [Fact]
    public void HandledPrecipPhases_is_a_subset_of_the_precipitation_registry()
    {
        var registered = Registry.ByTarget["precipitation"];
        PrecipPredictCommand.HandledPrecipPhases.Should().BeSubsetOf(registered,
            because: "PrecipPredictCommand only predicts phases that are registered in phases.yaml " +
                     "(the L1 gate) — a handled-set entry without a registry entry is a retired or " +
                     "mistyped phase that would never pass the registry check anyway");
    }

    [Fact]
    public void HandledElementPhases_is_a_subset_of_the_element_target_registries()
    {
        // Element phases span several single-phase targets (wind, humidity,
        // shortwave_radiation, cloud_cover, wind_gust) — union them. Phases
        // owned by other runtimes (wind_speed_lgb → Python predict,
        // wind_blend → WindBlendPredictCommand) are registered but
        // intentionally NOT handled, so subset (not equality) is the rule.
        var registered = Registry.ByTarget
            .Where(kv => kv.Key is "wind" or "humidity" or "shortwave_radiation" or "cloud_cover" or "wind_gust")
            .SelectMany(kv => kv.Value)
            .ToList();
        ElementPredictCommand.HandledElementPhases.Should().BeSubsetOf(registered);
    }

    // ------------------------------------------------------------------
    // Presentation — every registry phase carries a display block (the
    // old separate Site presentation tables moved INTO phases.yaml on
    // 2026-06-10, so this is now enforceable in one file)
    // ------------------------------------------------------------------

    [Fact]
    public void Every_registry_phase_has_display_fields()
    {
        foreach (var (target, phase) in Registry.EnumerateAll())
        {
            phase.Display.Should().NotBeNull(
                because: $"{target}/{phase.Id} is registered in phases.yaml; without a display " +
                         "block its lines/cards render fallback grey or untitled (the loader " +
                         "already rejects a display block without a color, so non-null is the " +
                         "whole invariant)");
        }
    }

    [Fact]
    public void Card_rendering_targets_carry_full_display_metadata()
    {
        // Precipitation + dry-window render per-phase SECTIONS (heading +
        // skill line + overlay label), not just coloured lines — a
        // colour-only display block would render an empty-description card.
        foreach (var target in new[] { "precipitation", "dry_window" })
        foreach (var phase in Registry.AllPhases(target))
        {
            phase.Display.Should().NotBeNull();
            phase.Display!.Title.Should().NotBeNullOrWhiteSpace(
                $"{target}/{phase.Id} needs display.title for its page heading");
            phase.Display.ShortTitle.Should().NotBeNullOrWhiteSpace(
                $"{target}/{phase.Id} needs display.shortTitle for chart titles");
            phase.Display.Description.Should().NotBeNullOrWhiteSpace(
                $"{target}/{phase.Id} needs display.description for the skill line");
            phase.Display.Label.Should().NotBeNullOrWhiteSpace(
                $"{target}/{phase.Id} needs display.label for overlay series");
        }
    }

    [Fact]
    public void Site_lookups_reflect_the_registry()
    {
        // The Site classes are thin lookups over the registry — assert they
        // expose every phase in registry order so render iteration can't
        // silently drop one (the failure mode the old hardcoded tables had).
        PrecipPhases.All.Select(p => p.Key).Should().Equal(Registry.ByTarget["precipitation"]);
        DryWindowPhases.All.Select(p => p.Key).Should().Equal(Registry.ByTarget["dry_window"]);
        foreach (var id in Registry.ByTarget["temperature"])
        {
            TempPhases.ColorFor(id).Should().NotBe("#9e9e9e",
                $"temperature phase {id} should resolve its registry colour, not the unknown-phase grey");
        }
    }

    [Fact]
    public void Phase3p_start_hour_curve_version_stays_in_sync_with_predict()
    {
        // The yaml display block carries the curve's on-disk model_version;
        // DryWindowPredictCommand writes the curve under its own constant.
        // They must agree or the renderer reads a curve that predict no
        // longer writes (or vice versa).
        DryWindowPhases.Phase3p.StartHourCurveVersion.Should()
            .Be(DryWindowPredictCommand.Phase3pStartHourVersion);
    }

    // ------------------------------------------------------------------
    // Repo-file location (same walk-up as WorkflowDataSyncContractTests)
    // ------------------------------------------------------------------

    private static string RepoFile(params string[] relativeParts)
    {
        var relative = Path.Combine(relativeParts);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new FileNotFoundException(
            $"Could not locate {relative} by walking parents of {AppContext.BaseDirectory}");
    }
}
