using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// "What NWPs feed each model at each lead" — single source of truth for
    /// the deployed feature spec, auto-generated from each Active version's
    /// <c>feature_schema.json</c>. Replaces the hand-edited
    /// <c>docs/MODEL_SPEC.md</c> reference with deployed truth: when a tier
    /// or picker changes, the table updates on next render with no extra
    /// curation. Sits as the 4th tab on the Models sub-nav so the per-target
    /// MAE / Brier pages stay metric-focused.
    ///
    /// Dedupe rule: per (composite, phase, lead), keep the freshest
    /// <see cref="FeatureSpecRow.TrainedAtUtc"/>. Older Active versions still
    /// emitting predictions (manifest hasn't been pruned) get filtered out
    /// so the table reads cleanly.
    /// </summary>
    public static string RenderModelsSpec(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("<section>");
        content.Append(RenderModelsSubNav("spec"));
        content.Append("""
              <hgroup>
                <h2>Feature spec</h2>
                <p>What NWPs feed each blender at each lead, derived from each Active
                version's <code>feature_schema.json</code>. Required = must be present;
                Optional = NaN allowed. UKV mode parsed from the feature-set tag.
                Auto-generated — when this disagrees with a hand-written doc, the
                table is right by construction.</p>
              </hgroup>
            """);

        if (input.FeatureSpecRows.Count == 0)
        {
            content.Append("<p><em>No feature schemas on disk. Train at least one model per target, then rclone <code>data/models</code> from R2.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Feature spec", "models", content.ToString());
        }

        // Filter to phases currently shipping. A version's predictions can
        // outlive its phase being deactivated (e.g. 3a_isotonic) — those rows
        // would clutter the spec without being load-bearing.
        var live = input.FeatureSpecRows
            .Where(r => IsActivePhase(r.Composite, r.Phase))
            // Per (composite, phase, lead), keep the freshest TrainedAt. New
            // training mints a new version; older ones may still be in the
            // window but the spec we want is the latest.
            .GroupBy(r => (r.Composite, r.Phase, r.LeadHours))
            .Select(g => g.OrderByDescending(r => r.TrainedAtUtc).First())
            .ToList();

        if (live.Count == 0)
        {
            content.Append("<p><em>No active phase schemas to render.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Feature spec", "models", content.ToString());
        }

        // Group by target prefix so each target gets its own table — three
        // tables on one page reads better than one wide table mixing
        // temperature MAE-blenders with per-station precip Brier-blenders.
        foreach (var (targetKey, targetLabel) in new[]
        {
            ("temperature",   "Temperature"),
            ("precipitation", "Precipitation"),
            ("dry_window",    "Dry-window"),
        })
        {
            var targetRows = live
                .Where(r => r.Composite == targetKey || r.Composite.StartsWith(targetKey + "/", StringComparison.Ordinal))
                .OrderBy(r => r.Composite, StringComparer.Ordinal)
                .ThenBy(r => ActivePhasePolicy.Priority(targetKey, r.Phase))
                .ThenBy(r => r.LeadHours)
                .ToList();
            if (targetRows.Count == 0) continue;

            content.Append(Ci, $"""<h3>{Escape(targetLabel)}</h3>""");
            content.Append(RenderSpecTable(targetRows));
        }

        content.Append("</section>");
        return WrapPage(input, "Feature spec", "models", content.ToString());
    }

    private static string RenderSpecTable(IReadOnlyList<FeatureSpecRow> rows)
    {
        var t = new StringBuilder();
        t.Append("""
            <table>
              <thead>
                <tr>
                  <th>Composite</th>
                  <th>Phase</th>
                  <th>Lead</th>
                  <th>Required</th>
                  <th>Optional</th>
                  <th>UKV</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
            """);
        foreach (var r in rows)
        {
            var (source, ukvMode) = InterpretFeatureSet(r.FeatureSet, r.OptionalModels);
            t.Append(Ci, $"""
                <tr>
                  <td>{Escape(PrettyComposite(r.Composite))}</td>
                  <td>{Escape(r.Phase)}</td>
                  <td>+{r.LeadHours}h</td>
                  <td>{Escape(string.Join(", ", r.RequiredModels.Select(NwpShort)))}</td>
                  <td>{Escape(string.Join(", ", r.OptionalModels.Select(NwpShort)))}</td>
                  <td>{Escape(ukvMode)}</td>
                  <td>{Escape(source)}</td>
                </tr>
                """);
        }
        t.Append("</tbody></table>");
        return t.ToString();
    }

    /// <summary>
    /// Decode the <c>FeatureSet</c> tag stored on disk into two human columns:
    ///   * data source — "Open-Meteo previous_runs" (lean / rich) vs "Exact-runtime S3" (exact-l*).
    ///   * UKV mode — "Strict" (cycles {0,6,12,18}Z, lead = target), "Averaging"
    ///     (cycles {3,15}Z averaged across two leads), or "—" (UKV not in the
    ///     spec). The Strict-vs-Averaging distinction can't be read from the
    ///     tag alone, so we infer per-target: temp 2d uses Strict, precip 3d
    ///     uses Averaging (decision baked in 2026-05-06 bake-off).
    /// </summary>
    private static (string Source, string UkvMode) InterpretFeatureSet(
        string featureSet, IReadOnlyList<string> optionalModels)
    {
        var hasUkv = optionalModels.Any(m => m == "met_office_ukv");
        var source = featureSet.StartsWith("exact-", StringComparison.Ordinal)
            ? "Exact-runtime S3"
            : "Open-Meteo previous_runs";

        if (!hasUkv) return (source, "—");

        // Strict vs Averaging: the tag carries a tier suffix (T2 / P1 / P2)
        // that we use as proxy. Temp tiers are Strict; precip tiers are
        // Averaging (ukv_target_aware_picker_shipped 2026-05-06).
        var ukvMode = featureSet.Contains("-T", StringComparison.Ordinal)
            ? "Strict"
            : featureSet.Contains("-P", StringComparison.Ordinal)
                ? "Averaging"
                : "Optional";
        return (source, ukvMode);
    }

    /// <summary>
    /// Map raw NWP id (Open-Meteo or S3 form) to the short label used in the
    /// existing tables (GFS, ECMWF, AIFS, ICON, MF, UKMO Global, UKV, GEM, JMA).
    /// Unknown ids fall back to themselves so a newly-added model doesn't
    /// silently disappear.
    /// </summary>
    private static string NwpShort(string nwpId) => nwpId switch
    {
        "gfs_seamless"               => "GFS",
        "gfs_ncep"                   => "GFS",
        "ecmwf_ifs025"               => "ECMWF",
        "ecmwf_ifs_oper"             => "ECMWF",
        "ecmwf_aifs025_single"       => "AIFS",
        "ecmwf_aifs_oper"            => "AIFS",
        "icon_seamless"              => "ICON",
        "meteofrance_seamless"       => "MF",
        "ukmo_seamless"              => "UKMO",
        "met_office_global"          => "UKMO Global",
        "met_office_ukv"             => "UKV",
        "gem_seamless"               => "GEM",
        "jma_seamless"               => "JMA",
        _                            => nwpId,
    };
}
