using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Held-out test performance for every active blender. One small card per blender
    /// (target × version) with a short prose description of what the blender is, plus
    /// a tight per-lead table comparing its blind-test score against the best single
    /// NWP at the same lead. The numeric counterpart to the Skill page — Skill shows
    /// the eyeball trajectory; Models shows what training measured.
    ///
    /// Replaces the previous one-big-table-per-composite layout, which was hard to
    /// scan once we had three precipitation stations × three blender phases × four
    /// leads on screen. Each card now answers "what is this blender, and is it
    /// worth running?" in one glance.
    /// </summary>
    public static string RenderModels(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Models</h2>
                <p>Held-out test performance for each active blender, against ERA5 (temperature) or EA rainfall (precipitation, dry window).
                   Temperature scores are MAE in °C; precipitation and dry-window scores are Brier — lower is better in both cases.
                   The Δ column shows blend-vs-best-single as a percentage; negative means the blend wins.</p>
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
        var grouped = input.ModelSummaries
            .GroupBy(m => m.Composite, StringComparer.Ordinal)
            .OrderBy(g => TargetOrder(g.Key))
            .ThenBy(g => g.Key, StringComparer.Ordinal);

        foreach (var group in grouped)
        {
            content.Append(Ci, $"""<h3>{Escape(PrettyComposite(group.Key))}</h3>""");
            // Newest trained first within a composite group — readers most often want
            // to see the most recent challenger / champion at the top.
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
        foreach (var lead in PocLeads)
        {
            if (!m.PerLead.TryGetValue(lead, out var s)) continue;

            // Δ% vs best single. Both metrics are "lower is better" (MAE for temp,
            // Brier for precip / dry-window), so a negative Δ means the blend won.
            // Guard against best-single being 0 or NaN — emit "—" rather than ÷0.
            string deltaCell;
            if (s.BestSingleValMae > 0 && !double.IsNaN(s.BestSingleValMae) && !double.IsNaN(s.BlendTestScore))
            {
                var pct = (s.BlendTestScore - s.BestSingleValMae) / s.BestSingleValMae * 100.0;
                var cls = pct < 0 ? "delta-good" : "delta-bad";
                deltaCell = $"""<td class="num {cls}">{pct.ToString("+0.0;-0.0;0.0", Ci)}%</td>""";
            }
            else
            {
                deltaCell = """<td class="num">—</td>""";
            }

            tbody.Append(Ci, $"""
                <tr>
                  <td>+{lead}h</td>
                  <td class="num"><strong>{s.BlendTestScore.ToString("0.000", Ci)}</strong></td>
                  <td class="num">{Escape(string.IsNullOrEmpty(s.BestSingle) ? "—" : s.BestSingle)} <small>({s.BestSingleValMae.ToString("0.000", Ci)})</small></td>
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
            ("temperature", "2b") or ("temperature", "2b_redo")
                => "Lean blender — six per-NWP temperatures, their mean/std/range, and cyclical hour/day-of-year encodings (~13 features).",
            ("temperature", "2c")
                => "Rich blender — adds per-NWP dew point, RH, cloud {total/low/mid/high}, wind speed/direction/gusts, surface pressure, plus cross-model aggregates (~88 features).",
            ("precipitation", "3a")
                => "Lean P(wet ≥ 0.1 mm/h) classifier — six per-NWP precipitation rates and ensemble agreement.",
            ("precipitation", "3a_isotonic")
                => "Phase 3a output passed through PAV isotonic calibration — same features, post-hoc score-to-probability remap.",
            ("precipitation", "3c")
                => "Rich P(wet) classifier — adds per-NWP cloud, humidity, CAPE, dew-point depression with feature-importance pruning (~55 features).",
            ("dry_window", "3b")
                => "Per-(station, window) classifier for whether at least one N-hour dry block occurs in the target UTC day.",
            ("dry_window", "3d-shape")
                => "Phase 3b features plus seven within-day shape descriptors (precip distribution moments).",
            ("dry_window", "3d-calibrated")
                => "Phase 3b output wrapped with per-lead PAV isotonic calibration.",
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
