using System.Text;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Dry daytime window page. <paramref name="stationSlug"/> picks which station
    /// to render; <c>null</c> means the first station, which ships as
    /// <c>dry-window.html</c> (filename preserved so existing links don't break).
    /// The other stations ship as <c>dry-window-{slug}.html</c>. Each variant
    /// shows the active phases × all windows for the one station.
    /// </summary>
    public static string RenderDryWindow(SiteInputs input, string? stationSlug = null)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Dry daytime window — P(∃ N-hour dry block in 09–18 local)</h2>
                <p>Per-station, per-window blender. Truth from EA Hydrology gauges with a 4-of-4 hourly gate.
                   The label asks "is there a contiguous N-hour dry block somewhere within 09:00–18:00 local time?" — the realistic
                   outdoor-walking window at Bonehill year-round, in BST or GMT (DST handled per target day).
                   Currently shipping <strong>Phase 3b (lean, 59 features)</strong>: per-NWP precip aggregates (sum, max-hour, wet-hour count, longest dry run, has-dry-window indicator, max prob),
                   ensemble cross-NWP summaries, RH/dew-depression/cloud/CAPE/wind covariates, and DOY calendar encodings. Bellever's blender ships PAV-calibrated;
                   Princetown and Hexworthy ship raw (their raw outputs were already well-calibrated, PAV overfit the small validation slice).
                   Phase 3d-shape (60-feature variant adding within-window rain-structure features) was tested 2026-04-29 and gave no consistent Brier improvement, so it doesn't render.</p>
              </hgroup>
            """);

        if (input.DryWindowPredictions.Count == 0)
        {
            content.Append("<p><em>No dry-window predictions in window. Run <code>predict --target dry-window --truth-station all --window all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry daytime window", "dry-window", content.ToString());
        }

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var currentStation = ResolveStationFromSlug(stations, stationSlug);

        if (currentStation is not null)
            content.Append(RenderStationSubNav("dry-window", stations, currentStation));

        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = Leads.Short;
        // Forward of "yesterday" so stale rows can't dominate the view.
        var cutoff = input.GeneratedAtUtc.Date.AddDays(-1);

        if (currentStation is null)
        {
            content.Append("<p><em>No dry-window predictions for the selected station.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry daytime window", "dry-window", content.ToString());
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

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                        {
                            // Cell text colour walks the green→red gradient so the eye
                            // can scan the column without reading every digit.
                            var color = ProbabilityColor(d.ProbHasDryWindow);
                            leadCells.Append(Ci, $"<td class=\"num\" style=\"color: {color}; font-weight: 600\">{d.ProbHasDryWindow.ToString("0.00", Ci)}</td>");
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
                          <td class="num">{agreement}</td>
                        </tr>
                        """);
                }

                // When only one phase is active, the per-phase header is just
                // noise — the section already names the window. Skip it and let
                // the table sit directly under the <h4>. The header returns
                // automatically the moment a second phase joins All.
                if (DryWindowPhases.All.Count > 1)
                {
                    content.Append(Ci, $"""
                        <h5>{Escape(phase.LongTitle)}</h5>
                        <p class="skill-line">{Escape(phase.Description)}</p>
                        """);
                }
                content.Append(Ci, $"""
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
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
            <p class="skill-line">A dry "hour" requires all four 15-min EA gauge readings to be ≤ 0.1 mm. Search is bounded to 09:00–18:00 local time (Europe/London, DST-aware) — overnight dry stretches don't count, and a dry block that bridges 18:00 into the evening isn't credited. Cross-midnight dry stretches are not credited (UTC-day boundary).</p>
            </section>
            """);
        return WrapPage(input, "Dry daytime window", "dry-window", content.ToString());
    }
}
