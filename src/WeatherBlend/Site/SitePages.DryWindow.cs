using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    public static string RenderDryWindow(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Dry window — P(∃ N-hour dry block in target UTC day)</h2>
                <p>Phase 3b per-station, per-window blender. 18 LightGBM classifiers (Bellever + Princetown × {3, 4, 6}h × leads 24/48/72h). Truth from EA Hydrology gauges with a 4-of-4 hourly gate.</p>
              </hgroup>
            """);

        if (input.DryWindowPredictions.Count == 0)
        {
            content.Append("<p><em>No dry-window predictions in window. Run <code>predict --target dry-window --truth-station all --window all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry window", "dry-window", content.ToString());
        }

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = new[] { 24, 48, 72 };

        foreach (var station in stations)
        {
            content.Append(Ci, $"<h3>{Escape(PrettyStation(station))}</h3>");

            foreach (var window in windows)
            {
                // Latest prediction per (target_date, lead). Keep target dates strictly
                // forward of "yesterday" to avoid stale rows dominating the view.
                var cutoff = input.GeneratedAtUtc.Date.AddDays(-1);
                var latest = input.DryWindowPredictions
                    .Where(d => d.Station == station && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

                if (latest.Count == 0)
                {
                    content.Append(Ci, $"<h4>{window}-hour dry window</h4><p><em>No predictions on or after {cutoff:yyyy-MM-dd}.</em></p>");
                    continue;
                }

                var dates = latest.Select(d => d.TargetDateUtc).Distinct().OrderBy(d => d).ToList();

                var tbody = new StringBuilder();
                foreach (var date in dates)
                {
                    var byLead = latest.Where(d => d.TargetDateUtc == date).ToDictionary(d => d.LeadHours);
                    var anyClim = byLead.Values.FirstOrDefault();
                    var clim = anyClim is null ? "—" : anyClim.ClimatologyProbHasDryWindow.ToString("0.00", Ci);

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                        {
                            // Bold cell when the blender meaningfully diverges from climatology.
                            var diff = d.ProbHasDryWindow - d.ClimatologyProbHasDryWindow;
                            var cls = Math.Abs(diff) >= 0.10 ? " class=\"num strong\"" : " class=\"num\"";
                            leadCells.Append(Ci, $"<td{cls}>{d.ProbHasDryWindow.ToString("0.00", Ci)}</td>");
                        }
                        else
                        {
                            leadCells.Append("<td class=\"num\">—</td>");
                        }
                    }

                    var agreementCell = byLead.Values
                        .Select(d => d.AgreementHasDryWindow)
                        .FirstOrDefault(a => a.HasValue);
                    var agreement = agreementCell.HasValue ? agreementCell.Value.ToString("0.00", Ci) : "—";

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time>{date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          <td class="num">{clim}</td>
                          <td class="num">{agreement}</td>
                        </tr>
                        """);
                }

                content.Append(Ci, $"""
                    <h4>{window}-hour dry window</h4>
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
                            <th class="num">Climatology</th>
                            <th class="num">Model agreement</th>
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);
            }
        }

        content.Append("""
            <p class="skill-line">A dry "hour" requires all four 15-min EA gauge readings to be ≤ 0.1 mm. Cross-midnight dry stretches are not credited (UTC-day boundary). Daylight filtering is deferred to the application layer.</p>
            </section>
            """);
        return WrapPage(input, "Dry window", "dry-window", content.ToString());
    }
}
