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
        // Per-NWP overlay chart — Blend on top, the six NWP inputs underneath. The
        // champion blend value itself is on the home page; this chart exists so the
        // reader can see which NWPs the blender's leaning on at this horizon and
        // where they disagree. Challenger blender lines at this lead live on Skill.
        //
        // Pool across all blender versions before picking the freshest per
        // ValidTime: the champion only emits a few rows per anchor, so a strict
        // version filter collapses the chart to a single point per lead. Pooling
        // gives the X axis a real time spread without changing the per-NWP story
        // (the NWP columns are identical across blender versions for the same
        // valid hour).
        var future = input.Predictions
            .Where(p => p.LeadHours == lead
                        && p.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        var s = new StringBuilder();
        s.Append("<h3>Temperature — blend vs per-model inputs</h3>");
        s.Append("<p class=\"skill-line\">One series per NWP plus the champion blend, valid times at this lead. Hover for exact values.</p>");

        if (future.Count == 0)
        {
            s.Append("<p><em>No +").Append(lead).Append("h temperature forecast available.</em></p>");
            return s.ToString();
        }

        // NWPs first so the brand-purple Blend draws last and sits visually on top.
        // Colours chosen for hue separation; UKMO uses indigo rather than purple to
        // stay distinct from the brand colour reserved for the blend itself.
        var nwpSpecs = new (string Label, string Color, Func<PredictionRow, double?> Get)[]
        {
            ("GFS",   "#ef5350", p => p.TempGfs),
            ("ECMWF", "#42a5f5", p => p.TempEcmwf),
            ("ICON",  "#66bb6a", p => p.TempIcon),
            ("MF",    "#ffa726", p => p.TempMf),
            ("UKMO",  "#5c6bc0", p => p.TempUkmo),
            ("GEM",   "#26a69a", p => p.TempGem),
        };

        var series = new List<LineSeries>();
        foreach (var (label, color, get) in nwpSpecs)
        {
            var pts = future
                .Select(p => (Valid: p.ValidTimeUtc, Val: get(p)))
                .Where(t => t.Val.HasValue)
                .Select(t => (X: t.Valid.ToOADate(), Y: t.Val!.Value))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries(label, color, pts));
        }
        var blendPts = future
            .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.BlendTemperature))
            .ToList();
        series.Add(new LineSeries("Blend", "#7c4dff", blendPts));

        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Temperature — +{lead}h",
            XLabel = "Valid time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 360,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
        }));
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
