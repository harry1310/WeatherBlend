using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Renders <c>forecasts-{lead}h.html</c> — one page per POC lead (24/48/72) showing
    /// what every active blender is saying for that single horizon. A sub-nav at the top
    /// lets the reader switch between leads without hopping back to the main nav.
    /// Composition per page:
    ///   1. Temperature — per-NWP breakdown table. The champion blend value itself is
    ///      on the home page; this page exists to show which NWPs drove the blend up
    ///      or down at this horizon.
    ///   2. Precipitation — one P(wet) chart + hourly-detail table per EA station.
    /// Dry-window sits on its own tab because its display unit is the UTC day, not the
    /// hourly valid-time used by temperature and P(wet).
    /// </summary>
    public static string RenderForecasts(SiteInputs input, int lead)
    {
        var body = new StringBuilder();
        body.Append(RenderLeadSubNav(lead));

        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>+{lead}h forecast</h2>
                <p>Latest blender outputs valid around the {lead}-hour horizon from right now.
                   Temperature is the champion model; precipitation is P(wet ≥ 0.1 mm/h)
                   per EA Hydrology gauge.</p>
              </hgroup>
            """);

        body.Append(RenderTempSection(input, lead));
        body.Append("<hr/>");
        body.Append(RenderPrecipSection(input, lead));

        body.Append("</section>");
        return WrapPage(input, $"Forecasts +{lead}h", "forecasts", body.ToString());
    }

    private static string RenderLeadSubNav(int current)
    {
        var items = new StringBuilder();
        foreach (var lead in PocLeads)
        {
            var cls = lead == current ? " class=\"active\"" : "";
            items.Append(Ci, $"""<li><a href="forecasts-{lead}h.html"{cls}>+{lead}h</a></li>""");
        }
        return $"""<nav class="lead-nav"><ul>{items}</ul></nav>""";
    }

    private static string RenderTempSection(SiteInputs input, int lead)
    {
        // Per-NWP breakdown table only. The champion blend value itself is on the home
        // page; repeating it here was duplication. Challenger lines at this lead live on
        // the Skill page.
        var future = input.Predictions
            .Where(p => p.LeadHours == lead
                        && (string.IsNullOrEmpty(input.CurrentVersion) || p.ModelVersion == input.CurrentVersion)
                        && p.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
            .OrderBy(p => p.ValidTimeUtc)
            .Take(24)
            .ToList();

        var s = new StringBuilder();
        s.Append("<h3>Temperature — per-model inputs</h3>");
        s.Append("<p class=\"skill-line\">Each row is one valid time at this lead. The Blend column is what the home-page card shows; the per-NWP columns are the raw model values the blender saw.</p>");

        if (future.Count == 0)
        {
            s.Append("<p><em>No +").Append(lead).Append("h temperature forecast available.</em></p>");
            return s.ToString();
        }

        var tbody = new StringBuilder();
        foreach (var p in future)
        {
            tbody.Append(Ci, $"""
                <tr>
                  <td><time>{p.ValidTimeUtc:MM-dd HH:mm}Z</time></td>
                  <td class="num"><strong>{p.BlendTemperature.ToString("0.0", Ci)}</strong></td>
                  <td class="num">{FmtNullable(p.TempGfs, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempEcmwf, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempIcon, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempMf, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempUkmo, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempGem, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempMean, "0.0")}</td>
                  <td class="num">{FmtNullable(p.TempStd, "0.00")}</td>
                </tr>
                """);
        }

        s.Append(Ci, $"""
            <figure>
              <table>
                <thead>
                  <tr>
                    <th>Valid time (UTC)</th>
                    <th class="num">Blend</th>
                    <th class="num">GFS</th>
                    <th class="num">ECMWF</th>
                    <th class="num">ICON</th>
                    <th class="num">MF</th>
                    <th class="num">UKMO</th>
                    <th class="num">GEM</th>
                    <th class="num">Mean</th>
                    <th class="num">Std</th>
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
            </figure>
            """);
        return s.ToString();
    }

    private static string RenderPrecipSection(SiteInputs input, int lead)
    {
        var s = new StringBuilder();
        s.Append("<h3>Precipitation — P(wet ≥ 0.1 mm/h)</h3>");

        var stations = input.PrecipPredictions
            .Select(p => p.Station).Distinct()
            .OrderBy(st => st, StringComparer.Ordinal).ToList();

        if (stations.Count == 0)
        {
            s.Append("<p><em>No precipitation predictions in window.</em></p>");
            return s.ToString();
        }

        foreach (var station in stations)
        {
            // One chart per station at this lead, with the station's champion version as
            // the primary line and climatology as a reference. Challenger lines (3a/3c)
            // live on the Skill page.
            input.PrecipCurrentByStation.TryGetValue(station, out var champion);

            var latestPerValid = input.PrecipPredictions
                .Where(r => r.Station == station
                            && r.LeadHours == lead
                            && (string.IsNullOrEmpty(champion) || r.Version == champion)
                            && r.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
                .GroupBy(r => r.ValidTimeUtc)
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .ToList();

            s.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

            if (latestPerValid.Count == 0)
            {
                s.Append(RenderEmptyChart(
                    $"P(wet) — {PrettyStation(station)} — +{lead}h",
                    "No champion forecast at this lead in the forward window."));
                continue;
            }

            var series = new List<LineSeries>
            {
                new($"P(wet) +{lead}h", "#7c4dff",
                    latestPerValid.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet)).ToList()),
            };

            var climPts = latestPerValid
                .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ClimatologyPWet))
                .ToList();
            if (climPts.Count > 0)
                series.Add(new LineSeries("Climatology", "#9e9e9e", climPts));

            s.Append(LineChartRenderer.Render(new LineChartSpec
            {
                Title = $"P(wet) — {PrettyStation(station)} — +{lead}h",
                XLabel = "Valid time (UTC)",
                YLabel = "Probability",
                Series = series,
                Height = 260,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));

            // Compact per-NWP hourly breakdown for the next ~48h at this lead. Same
            // columns as the temperature table above, so the page reads consistently.
            var detail = latestPerValid.Take(24).ToList();
            if (detail.Count == 0) continue;

            var tbody = new StringBuilder();
            foreach (var r in detail)
            {
                tbody.Append(Ci, $"""
                    <tr>
                      <td><time>{r.ValidTimeUtc:MM-dd HH:mm}Z</time></td>
                      <td class="num"><strong>{r.ProbWet.ToString("0.00", Ci)}</strong></td>
                      <td class="num">{r.ClimatologyPWet.ToString("0.00", Ci)}</td>
                      <td class="num">{FmtNullable(r.PrecipGfs)}</td>
                      <td class="num">{FmtNullable(r.PrecipEcmwf)}</td>
                      <td class="num">{FmtNullable(r.PrecipIcon)}</td>
                      <td class="num">{FmtNullable(r.PrecipMf)}</td>
                      <td class="num">{FmtNullable(r.PrecipUkmo)}</td>
                      <td class="num">{FmtNullable(r.PrecipGem)}</td>
                    </tr>
                    """);
            }

            s.Append(Ci, $"""
                <figure>
                  <table>
                    <thead>
                      <tr>
                        <th>Valid time (UTC)</th>
                        <th class="num">P(wet)</th>
                        <th class="num">Clim.</th>
                        <th class="num">GFS mm</th>
                        <th class="num">ECMWF mm</th>
                        <th class="num">ICON mm</th>
                        <th class="num">MF mm</th>
                        <th class="num">UKMO mm</th>
                        <th class="num">GEM mm</th>
                      </tr>
                    </thead>
                    <tbody>
                {tbody}    </tbody>
                  </table>
                </figure>
                """);
        }

        return s.ToString();
    }
}
