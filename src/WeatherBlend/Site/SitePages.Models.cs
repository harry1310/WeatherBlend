using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Held-out test performance for the latest active blender per (composite × phase).
    /// One small card per (target × phase) with a short prose description plus a
    /// tight per-lead table comparing the blend's test score to the best single NWP
    /// scored on the same test slice (apples-to-apples).
    ///
    /// Latest-only: the <see cref="SiteInputs.ModelSummaries"/> list grew unwieldy
    /// once each composite had multiple historical artefacts. This page now keeps
    /// only the most recently trained version per (composite, phase) so the eye
    /// can scan it. Older versions still live on disk; visit the artefact files
    /// directly if you need the history.
    /// </summary>
    public static string RenderModels(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Models</h2>
                <p>Held-out test performance for the latest active blender per phase, against ERA5 (temperature) or EA rainfall (precipitation, dry window).
                   Temperature scores are MAE in °C; precipitation and dry-window scores are Brier — lower is better in both cases.
                   The Δ column compares the blend to the best single NWP <em>on the same test slice</em>; negative means the blend wins.</p>
              </hgroup>
            """);

        if (input.ModelSummaries.Count == 0)
        {
            content.Append("<p><em>No training metadata on disk. Run <code>train</code> for each target, then rclone <code>data/models</code> from R2.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Models", "models", content.ToString());
        }

        // Order: temperature → precipitation → dry window, then alphabetical within each.
        // Matches the roadmap and the reader's mental model of "phase 2 → 3a → 3b".
        static int TargetOrder(string composite) => composite.Split('/')[0] switch
        {
            "temperature" => 0,
            "precipitation" => 1,
            "dry_window" => 2,
            _ => 3,
        };

        // Latest-only per (composite, phase), filtered to phases that are
        // currently shipping. Demoted phases (3a_isotonic + 3d-calibrated were
        // retired 2026-04-29 after PAV calibration didn't move test Brier; the
        // one-off temperature 2b_redo retrain is also reference-only) still
        // have prediction rows in the rolling window so they leak into
        // ModelSummaries — explicit allowlist keeps the Models page focused on
        // "what's actually being used right now".
        var latestPerFamily = input.ModelSummaries
            .Where(m => IsActivePhase(m.Composite, m.Phase))
            .GroupBy(m => (m.Composite, m.Phase), comparer: null!)
            .Select(g => g.OrderByDescending(m => m.TrainedAtUtc).First())
            .GroupBy(m => m.Composite, StringComparer.Ordinal)
            .OrderBy(g => TargetOrder(g.Key))
            .ThenBy(g => g.Key, StringComparer.Ordinal);

        foreach (var group in latestPerFamily)
        {
            content.Append(Ci, $"""<h3>{Escape(PrettyComposite(group.Key))}</h3>""");
            // Within a composite (e.g. one temperature block), order phase-cards
            // by champion → challenger so a glance at the page reads "lean first,
            // rich underneath". TrainedAt-descending was deterministic but ordered
            // by training-cycle accident; readers expect the lean champion on top.
            var ordered = group
                .OrderBy(m => PhasePriority(group.Key, m.Phase))
                .ThenByDescending(m => m.TrainedAtUtc)
                .ToList();
            foreach (var m in ordered)
                content.Append(RenderBlenderCard(group.Key, m, input.VerifyHistory));
        }

        content.Append("</section>");
        return WrapPage(input, "Models", "models", content.ToString());
    }

    /// <summary>
    /// Maps a Models-page composite key
    /// (<c>"temperature"</c> / <c>"precipitation/{station}"</c> /
    /// <c>"dry_window/{station}/{N}h"</c>) to the
    /// <see cref="WeatherBlend.Models.VerifyHistoryFile.Target"/> +
    /// <see cref="WeatherBlend.Models.VerifyHistoryRow.Station"/> +
    /// <see cref="WeatherBlend.Models.VerifyHistoryRow.WindowHours"/> tuple a
    /// verify-history file uses, so the per-card history filter can pick out
    /// rows matching this exact card.
    /// </summary>
    private static (string Target, string? Station, int? WindowHours) ParseCompositeForHistory(string composite)
    {
        var parts = composite.Split('/');
        if (parts.Length == 1) return (parts[0], null, null);
        if (parts.Length == 2) return (parts[0], parts[1], null);
        if (parts.Length == 3 && parts[2].EndsWith("h", StringComparison.Ordinal)
            && int.TryParse(parts[2][..^1], out var w))
            return (parts[0], parts[1], w);
        return (composite, null, null);
    }

    private static string RenderBlenderCard(string composite, ModelSummary m,
        IReadOnlyList<WeatherBlend.Models.VerifyHistoryFile> verifyHistory)
    {
        var description = PhaseDescription(composite, m.Phase);
        var phaseLabel = string.IsNullOrEmpty(m.Phase) ? "—" : m.Phase;

        var tbody = new StringBuilder();
        foreach (var lead in Leads.Full)
        {
            if (!m.PerLead.TryGetValue(lead, out var s)) continue;

            // Δ% vs best single, computed on the SAME test slice (apples-to-apples).
            // BestSingleTestMae populated from 2026-04-29+; older artefacts fall back
            // to the val number with a "(val)" tag so the mismatch is visible.
            var bestRef = s.BestSingleTestMae > 0 && !double.IsNaN(s.BestSingleTestMae)
                ? s.BestSingleTestMae : s.BestSingleValMae;
            var bestRefIsTest = s.BestSingleTestMae > 0 && !double.IsNaN(s.BestSingleTestMae);

            string deltaCell;
            if (bestRef > 0 && !double.IsNaN(bestRef) && !double.IsNaN(s.BlendTestScore))
            {
                var pct = (s.BlendTestScore - bestRef) / bestRef * 100.0;
                var cls = pct < 0 ? "delta-good" : "delta-bad";
                deltaCell = $"""<td class="num {cls}">{pct.ToString("+0.0;-0.0;0.0", Ci)}%</td>""";
            }
            else
            {
                deltaCell = """<td class="num">—</td>""";
            }

            var bestLabel = bestRefIsTest ? "" : " val";
            tbody.Append(Ci, $"""
                <tr>
                  <td>+{lead}h</td>
                  <td class="num"><strong>{s.BlendTestScore.ToString("0.000", Ci)}</strong></td>
                  <td class="num">{Escape(string.IsNullOrEmpty(s.BestSingle) ? "—" : s.BestSingle)} <small>({bestRef.ToString("0.000", Ci)}{bestLabel})</small></td>
                  {deltaCell}
                </tr>
                """);
        }

        var historyHtml = RenderVerifyHistorySection(composite, m.Phase, verifyHistory);

        return $"""
            <article class="blender-card">
              <header>
                <hgroup>
                  <h4>Phase {Escape(phaseLabel)} <small>· <code>{Escape(m.Version)}</code></small></h4>
                  <p class="skill-line">{Escape(description)} Trained {m.TrainedAtUtc:yyyy-MM-dd}. Metric: {Escape(m.MetricLabel)}.</p>
                </hgroup>
              </header>
              <table>
                <thead>
                  <tr>
                    <th>Lead</th>
                    <th class="num">Blend</th>
                    <th class="num">Best single</th>
                    <th class="num">Δ vs best</th>
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
              {historyHtml}
            </article>
            """;
    }

    /// <summary>
    /// Per-card "Verify history" sub-section: one row per past weekly verify
    /// run for this card's (target, station, windowHours, phase) lineage,
    /// with per-lead blend metric columns and a drift indicator. Filters
    /// <paramref name="verifyHistory"/> to runs whose <c>Target</c>,
    /// <c>Station</c>, <c>WindowHours</c> and <c>Phase</c> match the card,
    /// sorted newest-first.
    ///
    /// Phase-based (not version-based) match is deliberate: the card always
    /// represents the current Active version of a phase, but verify can only
    /// score versions whose paired truth window has fully landed (the
    /// 5-day ERA5 latency etc.). A freshly retrained champion therefore
    /// has no exact-version verify rows for ~5 days. Matching by phase
    /// keeps the same lineage's history visible across retrains.
    ///
    /// Renders nothing when no matching history is on disk — keeps the card
    /// clean for fresh-deploy / pre-first-verify states.
    /// </summary>
    private static string RenderVerifyHistorySection(string composite, string phase,
        IReadOnlyList<WeatherBlend.Models.VerifyHistoryFile> verifyHistory)
    {
        if (verifyHistory.Count == 0 || string.IsNullOrEmpty(phase)) return "";
        var (target, station, windowHours) = ParseCompositeForHistory(composite);

        // Find every (file, row) pair that matches this card. Each matching
        // file contributes one row to the history table; the per-lead values
        // come from rows in that file.
        var matchingFiles = verifyHistory
            .Where(f => f.Target == target)
            .Select(f =>
            {
                var rows = f.Rows.Where(r =>
                        string.Equals(r.Phase, phase, StringComparison.Ordinal)
                        && r.Station == station
                        && r.WindowHours == windowHours)
                    .ToList();
                return (File: f, Rows: rows);
            })
            .Where(x => x.Rows.Count > 0)
            .OrderByDescending(x => x.File.AsOfUtc)
            .ToList();

        if (matchingFiles.Count == 0)
        {
            // Empty-state explanation rather than silently omitting the section.
            // Two ways a card lands here:
            //   1. The phase is genuinely brand-new and verify hasn't run since
            //      training (will populate at the next weekly cycle + 5d ERA5 lag).
            //   2. The phase tag drifted between trainings (e.g. an old "2b_redo"
            //      version's verify rows can't match a current "2b" card). The
            //      message names the phase being filtered for so the cause is
            //      visible without diff'ing JSON files.
            return $"""
                <details open>
                  <summary><strong>Verify history</strong> <small>(no runs yet)</small></summary>
                  <p class="skill-line">No verify rows on disk match this card's phase ({Escape(phase)}). Either the next verify (twice-weekly Mon + Thu 09:30 UTC, then 5d ERA5 latency) hasn't yet scored predictions made by this version, or older verify files used a different phase tag for this lineage. Re-check after the next Mon/Thu cycle.</p>
                </details>
                """;
        }

        // Lead set for the columns: union across every matching run, sorted
        // ascending. Different runs may have different leads (a target's
        // training set changed mid-history), so a fixed Leads.Full would
        // either over- or under-count columns.
        var allLeads = matchingFiles
            .SelectMany(x => x.Rows.Select(r => r.LeadHours))
            .Distinct()
            .OrderBy(l => l)
            .ToList();
        var metricLabel = matchingFiles[0].File.MetricLabel;

        var tbody = new StringBuilder();
        foreach (var (file, rows) in matchingFiles)
        {
            // A single verify file can contain multiple versions of the same
            // phase (e.g. v..._phase3a from Apr 23 + v..._phase3a from Apr 26),
            // so per-lead values are picked from the latest version (lexicographic
            // sort works because every version starts with `vYYYY-MM-DD_HHmmss`).
            var byLead = rows
                .GroupBy(r => r.LeadHours)
                .ToDictionary(
                    g => g.Key,
                    g => g.OrderByDescending(r => r.ModelVersion, StringComparer.Ordinal).First());
            var driftAny = byLead.Values.Any(r => r.DriftFlag);

            // Distinct versions scored across all leads in this run. Usually
            // one (the active version of this phase at verify-time), but a
            // recent retrain leaves the predict tree carrying both old + new
            // for a few days so verify scores both. Showing them all keeps
            // the reader honest about which version the row's numbers came
            // from — matters when the row's drift is on the OLD version.
            var versionsScored = byLead.Values
                .Select(r => r.ModelVersion)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(v => v, StringComparer.Ordinal)
                .ToList();
            var versionCellHtml = string.Join(
                "<br>",
                versionsScored.Select(v => $"<code>{Escape(v)}</code>"));

            var leadCells = new StringBuilder();
            foreach (var lead in allLeads)
            {
                if (byLead.TryGetValue(lead, out var r))
                {
                    var driftCls = r.DriftFlag ? " delta-bad" : "";
                    leadCells.Append(Ci, $"""<td class="num{driftCls}">{r.BlendMetric.ToString("0.000", Ci)}</td>""");
                }
                else
                {
                    leadCells.Append("""<td class="num">—</td>""");
                }
            }

            var driftCell = driftAny
                ? "<td class=\"num delta-bad\">⚠</td>"
                : "<td class=\"num delta-good\">✓</td>";

            tbody.Append(Ci, $"""
                <tr>
                  <td><time datetime="{file.AsOfUtc:yyyy-MM-dd}">{file.AsOfUtc:ddd yyyy-MM-dd}</time></td>
                  <td><small>{versionCellHtml}</small></td>
                  {leadCells}
                  {driftCell}
                </tr>
                """);
        }

        var leadHeaders = new StringBuilder();
        foreach (var lead in allLeads)
            leadHeaders.Append(Ci, $"""<th class="num">+{lead}h</th>""");

        return $"""
            <details open>
              <summary><strong>Verify history</strong> <small>({matchingFiles.Count} run{(matchingFiles.Count == 1 ? "" : "s")})</small></summary>
              <p class="skill-line">Weekly Brier/MAE on the held-out rolling window — one row per verify run, drift flag in the last column. Metric: {Escape(metricLabel)}. Version column names which trained model the row's numbers came from — a freshly retrained champion shows zero rows here for ~14d (one verify cycle plus 5d ERA5 latency), so a row labelled with an older version is the previous lineage's history under the same phase.</p>
              <table>
                <thead>
                  <tr>
                    <th>Run (UTC)</th>
                    <th>Version</th>
                    {leadHeaders}
                    <th class="num">Drift</th>
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
            </details>
            """;
    }

    /// <summary>
    /// True iff the (target, phase) pair should render on the Models page.
    /// Thin adapter over <see cref="ActivePhasePolicy.IsActive"/> that splits
    /// the composite key into its target prefix — the page filters every
    /// version emitted into the rolling window through this so retired
    /// phases (still on disk) don't leak onto the cards.
    /// </summary>
    private static bool IsActivePhase(string composite, string phase)
        => ActivePhasePolicy.IsActive(composite.Split('/')[0], phase);

    /// <summary>
    /// Sort key for a phase within a composite — index in the
    /// champion-first <see cref="ActivePhasePolicy"/> list, so lean ranks
    /// above rich. Unknown phases sort last.
    /// </summary>
    private static int PhasePriority(string composite, string phase)
        => ActivePhasePolicy.Priority(composite.Split('/')[0], phase);

    /// <summary>
    /// One-sentence prose summary of what each phase actually does — composed for
    /// the Models page so the reader doesn't need to know the codebase to read the
    /// table. Keeps the per-blender card honest without trying to recompute feature
    /// counts at render time. Falls back to "Phase {x} blender." for unknown phases.
    /// </summary>
    private static string PhaseDescription(string composite, string phase)
    {
        var target = composite.Split('/')[0];
        return (target, phase) switch
        {
            ("temperature", "2b")
                => "Lean blender — six per-NWP temperatures, their mean/std/range, and cyclical hour/day-of-year encodings (~13 features).",
            ("temperature", "2c")
                => "Rich blender — adds per-NWP dew point, RH, cloud {total/low/mid/high}, wind speed/direction/gusts, surface pressure, plus cross-model aggregates (~88 features).",
            ("precipitation", "3a")
                => "Lean P(wet ≥ 0.1 mm/h) classifier — six per-NWP precipitation rates and ensemble agreement.",
            ("precipitation", "3c")
                => "Rich P(wet) classifier — adds per-NWP cloud, humidity, CAPE, dew-point depression with feature-importance pruning (~55 features).",
            ("dry_window", "3b")
                => "Per-(station, window) classifier for whether at least one N-hour dry block occurs in the target UTC day.",
            ("dry_window", "3g")
                => "Parameter-free MC — 10,000 Bernoulli draws over Phase 3a's hourly P(wet); the prediction is the fraction whose longest dry run reaches the window length. Cross-window monotonicity holds by construction.",
            _ => $"Phase {phase} blender.",
        };
    }

    /// <summary>
    /// Pretty-print a composite key. Examples:
    ///   "temperature"                              → "Temperature"
    ///   "precipitation/ea_bellever_dartmoor"       → "Precipitation — Bellever Dartmoor"
    ///   "dry_window/ea_bellever_dartmoor/6h"       → "Dry window — Bellever Dartmoor — 6-hour"
    /// </summary>
    private static string PrettyComposite(string key)
    {
        var parts = key.Split('/');
        if (parts.Length == 0) return key;

        var target = parts[0] switch
        {
            "temperature" => "Temperature",
            "precipitation" => "Precipitation",
            "dry_window" => "Dry window",
            _ => parts[0],
        };
        if (parts.Length == 1) return target;

        var station = PrettyStation(parts[1]);
        if (parts.Length == 2) return $"{target} — {station}";

        // dry-window: third part is "{N}h"
        var window = parts[2].TrimEnd('h');
        return $"{target} — {station} — {window}-hour";
    }
}
