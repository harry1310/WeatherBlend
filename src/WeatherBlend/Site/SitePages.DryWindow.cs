using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Dry-window page. <paramref name="stationSlug"/> picks which station to render;
    /// <c>null</c> means the first station, which ships as <c>dry-window.html</c>. The
    /// other stations ship as <c>dry-window-{slug}.html</c>. Each variant shows all
    /// three phases (3b, 3d-shape, 3d-calibrated) × all windows for the one station.
    /// </summary>
    public static string RenderDryWindow(SiteInputs input, string? stationSlug = null)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Dry window — P(∃ N-hour dry block in target UTC day)</h2>
                <p>Per-station, per-window blender. Bellever + Princetown × {3, 4, 6}h × leads 24/48/72h. Truth from EA Hydrology gauges with a 4-of-4 hourly gate.
                   Three phases run side-by-side via the per-composite <code>Active</code> manifest list — <strong>Phase 3b (lean, 53 features)</strong> as the production champion,
                   <strong>Phase 3d-shape (60 features)</strong> adding 7 within-day shape features derived from the ensemble-mean hourly precip vector,
                   and <strong>Phase 3d-calibrated</strong> wrapping 3b's saved model in a per-lead pool-adjacent-violators isotonic remapping.
                   Tables below are grouped by phase so feature-richness vs post-hoc calibration deltas are visible at a glance.</p>
              </hgroup>
            """);

        if (input.DryWindowPredictions.Count == 0)
        {
            content.Append("<p><em>No dry-window predictions in window. Run <code>predict --target dry-window --truth-station all --window all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry window", "dry-window", content.ToString());
        }

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var currentStation = ResolveStationFromSlug(stations, stationSlug);

        if (currentStation is not null)
            content.Append(RenderStationSubNav("dry-window", stations, currentStation));

        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = new[] { 24, 48, 72 };
        // Forward of "yesterday" so stale rows can't dominate the view.
        var cutoff = input.GeneratedAtUtc.Date.AddDays(-1);

        if (currentStation is null)
        {
            content.Append("<p><em>No dry-window predictions for the selected station.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry window", "dry-window", content.ToString());
        }

        content.Append(Ci, $"<h3>{Escape(PrettyStation(currentStation))}</h3>");

        foreach (var window in windows)
        {
            var windowRows = input.DryWindowPredictions
                .Where(d => d.Station == currentStation && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                .ToList();

            if (windowRows.Count == 0)
            {
                content.Append(Ci, $"<h4>{window}-hour dry window</h4><p><em>No predictions on or after {cutoff:yyyy-MM-dd}.</em></p>");
                continue;
            }

            content.Append(Ci, $"<h4>{window}-hour dry window</h4>");

            bool anyRendered = false;
            foreach (var phase in DryWindowPhases.All)
            {
                var phaseRows = windowRows
                    .Where(d => DryWindowPhases.Bucket(input.PhaseByVersion, d.Version) == phase)
                    .ToList();
                if (phaseRows.Count == 0) continue;
                anyRendered = true;

                // Latest prediction per (target_date, lead) within this phase bucket.
                var latest = phaseRows
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

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
                    <h5>{Escape(phase.LongTitle)}</h5>
                    <p class="skill-line">{Escape(phase.Description)}</p>
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

            if (!anyRendered)
            {
                content.Append("<p><em>No predictions in known phase buckets for this window.</em></p>");
            }
        }

        content.Append("""
            <p class="skill-line">A dry "hour" requires all four 15-min EA gauge readings to be ≤ 0.1 mm. Cross-midnight dry stretches are not credited (UTC-day boundary). Daylight filtering is deferred to the application layer.</p>
            </section>
            """);
        return WrapPage(input, "Dry window", "dry-window", content.ToString());
    }
}
