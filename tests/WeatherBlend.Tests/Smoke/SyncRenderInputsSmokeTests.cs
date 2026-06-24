using FluentAssertions;
using Xunit;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// Guards the completeness of <c>.github/actions/sync-render-inputs/action.yml</c>
/// — the single composite action that pulls every R2 tree the predict +
/// render path reads (shared by predict-and-render.yml and render-site.yml).
///
/// WeatherBlend's predict/render data provisioning is ALREADY centralised in
/// that action (unlike WP's old per-workflow hand-rolled rclone), but its
/// completeness was never smoke-tested — so a tree silently dropped from the
/// list only surfaced in production: the 2026-05-01 missing-<c>data/reports</c>
/// regression (Models page "Verify history" went blank) and the 2026-05-04
/// missing-<c>data/truth/met_office_obs</c> + <c>data/forecasts</c> regression
/// (temp-skill cross-check lines + per-NWP PoP overlay went silently empty).
///
/// This is the WB analogue of WeatherProbabilistic's predict-pull guards: it
/// turns "a render/predict input tree was removed from the canonical pull" into
/// a PR-time failure. Each required tree is annotated with the consumer that
/// reads it — adding a tree here is the same one-place edit as adding it to the
/// action. (Orographic JSON is intentionally NOT here: it's committed to git
/// under data/static/orographic, so the checkout provides it — no R2 pull.)
///
/// NOTE: this is a static completeness guard (catches a dropped tree). The
/// stronger run-it-and-fail form — a render smoke that provisions ONLY the
/// trees this action declares, then asserts RenderSiteCommand produces
/// non-empty pages — is the follow-up.
/// </summary>
[Trait("Category", "Smoke")]
public class SyncRenderInputsSmokeTests
{
    /// <summary>R2 tree → the predict/render consumer that reads it. If any of
    /// these is missing from the action's pull list, the smoke fails.</summary>
    private static readonly (string Tree, string Consumer)[] RequiredTrees =
    {
        ("data/forecasts",            "predict-all (every target's live feature vector) + RenderSiteCommand per-NWP PoP / Met Office Spot overlays"),
        ("data/models",               "predict-all (bundles + MANIFEST champion filter) + Models page metrics"),
        ("data/predictions",          "RenderSiteCommand home page / per-lead forecasts / skill charts + 4b mint inputs"),
        ("data/reports",              "Models page 'Verify history' table (verify_*.json sidecars) — the 2026-05-01 regression"),
        ("data/truth/era5",           "temperature skill chart truth"),
        ("data/truth/metar",          "temp-skill METAR comparison line"),
        ("data/truth/rainfall",       "precip/rain-skill truth + PrecipPredict rich/3d antecedent-rain features (LoadHourlyRain)"),
        ("data/truth/weatherlink",    "3c predict persistence for WeatherLink-sourced rainfall stations (Sennen → Lands End cove gauge)"),
        ("data/truth/met_office_obs", "temp-skill Met Office Land Obs cross-check line — the 2026-05-04 regression"),
    };

    private static string LocateRenderInputsAction()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(
                dir.FullName, ".github", "actions", "sync-render-inputs", "action.yml");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new FileNotFoundException(
            ".github/actions/sync-render-inputs/action.yml not found by ascent from "
            + AppContext.BaseDirectory);
    }

    [Fact]
    public void Sync_render_inputs_pulls_every_predict_and_render_tree()
    {
        var actionYaml = File.ReadAllText(LocateRenderInputsAction());

        var missing = RequiredTrees
            .Where(t => !actionYaml.Contains(t.Tree, StringComparison.Ordinal))
            .Select(t => $"'{t.Tree}' (needed by: {t.Consumer})")
            .ToList();

        missing.Should().BeEmpty(
            "sync-render-inputs/action.yml must pull every tree the predict + render path reads. "
            + "A dropped tree silently empties part of the site in production (2026-05-01 reports, "
            + "2026-05-04 met_office_obs/forecasts). Missing:\n  " + string.Join("\n  ", missing));
    }
}
