using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

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

    /// <summary>
    /// True when the row's feature vector includes a UKV column. UKV is
    /// joined via the (cycle, lead) picker tables outside the normal NWP
    /// pivot, so it never appears in <see cref="FeatureSpecRow.RequiredModels"/>
    /// or <see cref="FeatureSpecRow.OptionalModels"/>. The reliable on-disk
    /// signal is `temp_ukv` (temperature) or `precip_ukv` (precipitation)
    /// in <see cref="FeatureSpecRow.FeatureNames"/>. Some FeatureSet tags
    /// carry a `-ukv` suffix (3d at 48/72) but others don't (2d at 12/24)
    /// — that suffix is unreliable; FeatureNames is canonical.
    /// </summary>
    private static bool UkvIsUsed(FeatureSpecRow row)
        => row.FeatureNames.Any(n =>
            string.Equals(n, "temp_ukv", StringComparison.Ordinal)
            || string.Equals(n, "precip_ukv", StringComparison.Ordinal));

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
            var (source, ukvMode) = InterpretFeatureSet(r);
            // Optional column doubles up to surface UKV when present, since
            // UKV isn't stored in OptionalModels (joined via picker tables
            // outside the normal Required/Optional buckets) but reads as an
            // optional input from the user's perspective. Append "UKV"
            // explicitly when a temp_ukv / precip_ukv column is in
            // FeatureNames so the column lists the actual on-disk inputs.
            var optionalDisplay = r.OptionalModels.Select(NwpShort).ToList();
            if (UkvIsUsed(r) && !optionalDisplay.Contains("UKV", StringComparer.Ordinal))
                optionalDisplay.Add("UKV");
            t.Append(Ci, $"""
                <tr>
                  <td>{Escape(PrettyComposite(r.Composite))}</td>
                  <td>{Escape(r.Phase)}</td>
                  <td>+{r.LeadHours}h</td>
                  <td>{Escape(string.Join(", ", r.RequiredModels.Select(NwpShort)))}</td>
                  <td>{Escape(string.Join(", ", optionalDisplay))}</td>
                  <td>{Escape(ukvMode)}</td>
                  <td>{Escape(source)}</td>
                </tr>
                """);
        }
        t.Append("</tbody></table>");
        return t.ToString();
    }

    /// <summary>
    /// Decode the row into two human columns: data source ("Open-Meteo
    /// previous_runs" / "Exact-runtime S3") and UKV mode ("Strict" /
    /// "Averaging" / "—").
    ///
    /// Two paths:
    /// 1. **Structured fields (Phase 2 — primary path).** When a row has
    ///    a non-empty <see cref="FeatureSpecRow.DataSource"/> and the
    ///    <see cref="FeatureSpecRow.UkvStrategy"/> is either set or null
    ///    explicitly, the renderer reads the answers directly from these
    ///    fields. No string parsing, no inference. This is the path every
    ///    new training run (post-2026-05-07) takes.
    /// 2. **Legacy inference (fallback).** When DataSource is empty
    ///    (schemas predating the structured-spec migration), fall back to
    ///    inferring from FeatureSet prefix + FeatureNames-contains-temp_ukv
    ///    /precip_ukv. Same logic as before Phase 2 — preserved so old
    ///    versions still in the predictions window render correctly.
    ///
    /// History of bugs in the legacy path that the structured fields
    /// retire (so future me doesn't reintroduce them on a refactor):
    ///   * commit dce1d56 — checked OptionalModels for "met_office_ukv".
    ///     Never present, so every cell rendered "—".
    ///   * commit 556e681 — checked FeatureSet for "-ukv" suffix. Hit 3d
    ///     correctly but missed 2d ("exact-l12-T2", no suffix, temp_ukv
    ///     IS in FeatureNames) so 2d still rendered "—".
    /// </summary>
    private static (string Source, string UkvMode) InterpretFeatureSet(FeatureSpecRow row)
    {
        // Path 1 — structured fields. Non-empty DataSource means the row
        // came from a post-Phase-1 schema. UkvStrategy is either set
        // (Strict / Averaging) or explicitly null (no UKV).
        if (!string.IsNullOrEmpty(row.DataSource))
        {
            var source = row.DataSource switch
            {
                BlenderDataSource.ExactRuntimeS3 => "Exact-runtime S3",
                BlenderDataSource.OpenMeteoPreviousRuns => "Open-Meteo previous_runs",
                _ => row.DataSource, // unknown source — surface as-is, don't hide
            };
            var ukvMode = row.UkvStrategy ?? "—";
            return (source, ukvMode);
        }

        // Path 2 — legacy inference. Source from FeatureSet prefix; UKV
        // presence from FeatureNames; mode from the FeatureSet tier letter.
        var legacySource = row.FeatureSet.StartsWith("exact-", StringComparison.Ordinal)
            ? "Exact-runtime S3"
            : "Open-Meteo previous_runs";

        if (!UkvIsUsed(row)) return (legacySource, "—");

        var legacyMode = row.FeatureSet.Contains("-T", StringComparison.Ordinal)
            ? "Strict"
            : row.FeatureSet.Contains("-P", StringComparison.Ordinal)
                ? "Averaging"
                : "Optional";
        return (legacySource, legacyMode);
    }

    /// <summary>
    /// Map raw NWP id (Open-Meteo or S3 form) to the short label used in the
    /// existing tables. The 8 blender-input ids delegate to
    /// <see cref="Nwp.DisplayLabel"/>; the raw-archive variants
    /// (<c>*_oper</c>, <c>gfs_ncep</c>, <c>met_office_global</c>,
    /// <c>met_office_ukv</c>) collapse to the same short label as their
    /// blender-input twin so a row sourced from raw S3 reads consistently
    /// with the Open-Meteo blender row.
    /// Unknown ids fall back to themselves so a newly-added model doesn't
    /// silently disappear.
    /// </summary>
    private static string NwpShort(string nwpId) => nwpId switch
    {
        "gfs_ncep"           => "GFS",
        "ecmwf_ifs_oper"     => "ECMWF",
        "ecmwf_aifs_oper"    => "AIFS",
        "met_office_global"  => "UKMO Global",
        "met_office_ukv"     => "UKV",
        _                    => Nwp.DisplayLabel(nwpId),
    };
}
