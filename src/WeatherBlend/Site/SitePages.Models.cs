using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Held-out test performance for every active blender. One table per composite
    /// (temperature, precipitation per station, dry-window per station × window) with
    /// one row per version × lead. This is the numeric counterpart to the Skill page —
    /// Skill shows the eyeball trajectory; Models shows the blind-test metric that
    /// was actually computed during training.
    ///
    /// Scope is deliberately narrow: no hyperparameter / importance / calibration
    /// tables. The raw <c>training_metadata.json</c> is on disk (and in R2) for the
    /// curious; this page exists so a reader can answer "which blender is best at
    /// +48h?" in one glance.
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
                   Test slices are time-split (walk-forward): train → validation → test, in that order, with no data leakage across the boundary.</p>
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
            var ordered = group.OrderByDescending(m => m.TrainedAtUtc).ToList();
            var first = ordered[0];
            content.Append(Ci, $"""
                <article>
                  <h3>{Escape(PrettyComposite(group.Key))}</h3>
                  <p class="skill-line">Metric: {Escape(first.MetricLabel)}.</p>
                """);

            content.Append(RenderComposite(ordered));
            content.Append("</article>");
        }

        content.Append("</section>");
        return WrapPage(input, "Models", "models", content.ToString());
    }

    private static string RenderComposite(IReadOnlyList<ModelSummary> ordered)
    {
        // One row per (version, lead). Sorting: newest trained first, then +24 → +48 → +72.
        // Each version contributes up to three rows; the first row in a version is the
        // only one that prints phase / trained-at so the table reads per-version.
        var tbody = new StringBuilder();
        foreach (var m in ordered)
        {
            bool firstLeadRow = true;
            foreach (var lead in new[] { 24, 48, 72 })
            {
                if (!m.PerLead.TryGetValue(lead, out var s)) continue;

                var versionCell = firstLeadRow
                    ? $"""<td rowspan="{CountLeads(m)}"><code>{Escape(m.Version)}</code></td>"""
                    : "";
                var phaseCell = firstLeadRow
                    ? $"""<td rowspan="{CountLeads(m)}">{Escape(string.IsNullOrEmpty(m.Phase) ? "—" : m.Phase)}</td>"""
                    : "";
                var trainedCell = firstLeadRow
                    ? $"""<td rowspan="{CountLeads(m)}"><time>{m.TrainedAtUtc:yyyy-MM-dd}</time></td>"""
                    : "";

                tbody.Append(Ci, $"""
                    <tr>
                      {versionCell}
                      {phaseCell}
                      {trainedCell}
                      <td>+{lead}h</td>
                      <td class="num"><strong>{s.BlendTestScore.ToString("0.000", Ci)}</strong></td>
                      <td class="num">{s.BlendTestRmse.ToString("0.000", Ci)}</td>
                      <td class="num">{s.BlendTestBias.ToString("+0.00;-0.00;0.00", Ci)}</td>
                      <td class="num">{Escape(string.IsNullOrEmpty(s.BestSingle) ? "—" : s.BestSingle)}</td>
                      <td class="num">{s.BestSingleValMae.ToString("0.000", Ci)}</td>
                      <td class="num">{s.TestRows}</td>
                    </tr>
                    """);
                firstLeadRow = false;
            }
        }

        return $"""
            <figure>
              <table>
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Phase</th>
                    <th>Trained</th>
                    <th>Lead</th>
                    <th class="num">Blend score</th>
                    <th class="num">RMSE</th>
                    <th class="num">Bias</th>
                    <th class="num">Best single</th>
                    <th class="num">Best-single val</th>
                    <th class="num">Test rows</th>
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
            </figure>
            """;
    }

    private static int CountLeads(ModelSummary m)
        => new[] { 24, 48, 72 }.Count(lead => m.PerLead.ContainsKey(lead));

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
