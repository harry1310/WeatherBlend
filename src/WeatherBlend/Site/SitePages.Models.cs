using System.Text;
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
            // Within a composite (e.g. one temperature block), order phase-cards by
            // most recently trained first.
            var ordered = group.OrderByDescending(m => m.TrainedAtUtc).ToList();
            foreach (var m in ordered)
                content.Append(RenderBlenderCard(group.Key, m));
        }

        content.Append("</section>");
        return WrapPage(input, "Models", "models", content.ToString());
    }

    private static string RenderBlenderCard(string composite, ModelSummary m)
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
            </article>
            """;
    }

    /// <summary>
    /// Phase allowlist per target — what we actually want surfaced on the
    /// Models page. The full <c>training_metadata.Phase</c> universe includes
    /// retired bookkeeping artefacts (3a_isotonic + 3d-calibrated were dropped
    /// 2026-04-29; the one-off 2b_redo retrain stays on disk for verify history).
    /// Keeps the page focused on the live champion + its current challenger.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, IReadOnlySet<string>> ActivePhasesByTarget =
        new Dictionary<string, IReadOnlySet<string>>(StringComparer.Ordinal)
        {
            ["temperature"]   = new HashSet<string>(StringComparer.Ordinal) { "2b", "2c" },
            ["precipitation"] = new HashSet<string>(StringComparer.Ordinal) { "3a", "3c" },
            ["dry_window"]    = new HashSet<string>(StringComparer.Ordinal) { "3b", "3d-shape" },
        };

    /// <summary>
    /// True iff the (target, phase) pair should render on the Models page. The
    /// renderer reads every prediction in the rolling window via DuckDB; any
    /// version whose phase has been retired but still has window-resident
    /// prediction rows would otherwise leak through.
    /// </summary>
    private static bool IsActivePhase(string composite, string phase)
    {
        var target = composite.Split('/')[0];
        return ActivePhasesByTarget.TryGetValue(target, out var allowed)
            && allowed.Contains(phase);
    }

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
            ("dry_window", "3d-shape")
                => "Phase 3b features plus seven within-day shape descriptors (precip distribution moments).",
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
