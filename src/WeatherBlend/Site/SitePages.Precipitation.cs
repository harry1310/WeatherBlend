using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    public static string RenderPrecipitation(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Precipitation — P(wet ≥ 0.1 mm/h)</h2>
                <p>Per-station hourly occurrence blender. EA Hydrology gauges as truth; LightGBM per lead-time bucket.
                   Three blenders are active per station — <strong>Phase 3a (lean, 27 features)</strong> as the original champion,
                   <strong>Phase 3a_isotonic</strong> (3a's output re-mapped by a per-lead pool-adjacent-violators isotonic regression
                   fit on the validation slice), and <strong>Phase 3c (rich, 55 features)</strong> as the feature-richness challenger
                   with per-model humidity, surface pressure, and EA trailing-rainfall persistence. All three produce predictions
                   every cycle; charts below are grouped by phase so differences between feature richness and post-hoc calibration
                   are visible at a glance.</p>
              </hgroup>
            """);

        if (input.PrecipPredictions.Count == 0)
        {
            content.Append("<p><em>No precipitation predictions in window. Run <code>predict --target precipitation --truth-station all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Precipitation", "precipitation", content.ToString());
        }

        var stations = input.PrecipPredictions.Select(p => p.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();

        // Lead-bucket palette matches the temperature overview chart.
        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),
            (48, "#7c4dff"),
            (72, "#4527a0"),
        };

        foreach (var station in stations)
        {
            var stationRows = input.PrecipPredictions.Where(p => p.Station == station).ToList();

            // Headline: peak P(wet) at lead 24h over the next 72h, drawn from 3c if
            // available (the challenger is better-calibrated at +24h), otherwise 3a.
            var phaseForHeadline = stationRows.Any(r => PrecipPhases.Bucket(input.PhaseByVersion, r.Version) == PrecipPhases.Phase3c)
                ? PrecipPhases.Phase3c : PrecipPhases.Phase3a;
            var headlineRows = stationRows
                .Where(r => PrecipPhases.Bucket(input.PhaseByVersion, r.Version) == phaseForHeadline
                         && r.LeadHours == 24
                         && r.ValidTimeUtc >= input.GeneratedAtUtc)
                .GroupBy(r => r.ValidTimeUtc)
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .ToList();
            string headline;
            if (headlineRows.Count > 0)
            {
                var peak = headlineRows.OrderByDescending(r => r.ProbWet).First();
                var meanP = headlineRows.Average(r => r.ProbWet);
                headline = $"Next {headlineRows.Count}h at +24h (Phase {phaseForHeadline.Key}): mean P(wet) = {meanP.ToString("0.00", Ci)}, peak {peak.ProbWet.ToString("0.00", Ci)} at {peak.ValidTimeUtc:yyyy-MM-dd HH:mm}Z.";
            }
            else
            {
                headline = "No +24h-lead forecast in the future window.";
            }

            content.Append(Ci, $"""
                <article>
                  <hgroup>
                    <h3>{Escape(PrettyStation(station))}</h3>
                    <p class="skill-line">{Escape(headline)}</p>
                  </hgroup>
                """);

            // One chart per phase present for this station.
            bool anyRendered = false;
            foreach (var spec in PrecipPhases.All)
            {
                var phaseRows = stationRows
                    .Where(r => PrecipPhases.Bucket(input.PhaseByVersion, r.Version) == spec)
                    .ToList();
                if (phaseRows.Count == 0) continue;
                anyRendered = true;

                var latestPerLead = phaseRows
                    .GroupBy(r => (r.LeadHours, r.ValidTimeUtc))
                    .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                    .ToList();

                var series = new List<LineSeries>();
                foreach (var (lead, color) in leadSpecs)
                {
                    var pts = latestPerLead
                        .Where(r => r.LeadHours == lead && r.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
                        .OrderBy(r => r.ValidTimeUtc)
                        .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                        .ToList();
                    if (pts.Count > 0)
                        series.Add(new LineSeries($"Blend +{lead}h", color, pts));
                }

                // Climatology baseline — identical per valid time, pull it from whichever
                // lead has data and overlay as a reference line.
                var climPts = latestPerLead
                    .Where(r => r.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
                    .GroupBy(r => r.ValidTimeUtc)
                    .Select(g => g.First())
                    .OrderBy(r => r.ValidTimeUtc)
                    .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ClimatologyPWet))
                    .ToList();
                if (climPts.Count > 0)
                    series.Add(new LineSeries("Climatology", "#9e9e9e", climPts));

                content.Append(Ci, $"""
                    <h4>{Escape(spec.LongTitle)}</h4>
                    <p class="skill-line">{Escape(spec.Description)}</p>
                    """);

                if (series.Count == 0)
                {
                    content.Append(RenderEmptyChart(
                        $"P(wet) — {PrettyStation(station)} — {spec.Key}",
                        "No forecast valid times in the future window."));
                }
                else
                {
                    content.Append(LineChartRenderer.Render(new LineChartSpec
                    {
                        Title = $"P(wet ≥ 0.1 mm/h) — {PrettyStation(station)} — Phase {spec.Key}",
                        XLabel = "Valid time (UTC)",
                        YLabel = "Probability",
                        Series = series,
                        Height = 300,
                        FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                        FormatY = v => v.ToString("0.00", Ci),
                    }));
                }
            }

            // Champion-vs-challenger overlay at +24h — the most skillful lead and
            // the one the dog-walk user actually cares about. Drops in if both
            // phases produced 24h rows for this station. "Now minus 1h" to anchor
            // at the forecast trajectory rather than full history.
            var cvc24 = BuildChampionVsChallengerSeries(stationRows, input.PhaseByVersion, input.GeneratedAtUtc.AddHours(-1), leadHours: 24);
            if (cvc24.Count >= 2)
            {
                content.Append(Ci, $"""
                    <h4>Three-way comparison — +24h lead</h4>
                    <p class="skill-line">3a (raw), 3a_isotonic (calibrated), and 3c (rich) on the same axis. If 3a_isotonic tracks 3c, calibration alone did the work the extra features were supposed to do.</p>
                    """);
                content.Append(LineChartRenderer.Render(new LineChartSpec
                {
                    Title = $"3a vs 3a_isotonic vs 3c — {PrettyStation(station)} — +24h",
                    XLabel = "Valid time (UTC)",
                    YLabel = "Probability",
                    Series = cvc24,
                    Height = 280,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                    FormatY = v => v.ToString("0.00", Ci),
                }));
            }

            if (!anyRendered)
            {
                content.Append(RenderEmptyChart($"P(wet) — {station}", "No forecast valid times in the future window."));
            }

            content.Append("</article>");
        }

        // Latest-per-(station, phase, lead, valid_time) detail table, capped to the
        // next 96h forward of generation to keep page weight reasonable.
        var future = input.PrecipPredictions
            .Where(p => p.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1) && p.ValidTimeUtc <= input.GeneratedAtUtc.AddHours(96))
            .GroupBy(p => (p.Station, p.Version, p.LeadHours, p.ValidTimeUtc))
            .Select(g => g.OrderByDescending(p => p.PredictedAtUtc).First())
            .OrderBy(p => p.Station, StringComparer.Ordinal)
            .ThenBy(p => p.ValidTimeUtc)
            .ThenBy(p => p.LeadHours)
            .ThenBy(p => PrecipPhases.Bucket(input.PhaseByVersion, p.Version).Key, StringComparer.Ordinal)
            .ToList();

        if (future.Count > 0)
        {
            var tbody = new StringBuilder();
            foreach (var p in future)
            {
                var phase = PrecipPhases.Bucket(input.PhaseByVersion, p.Version);
                tbody.Append(Ci, $"""
                    <tr>
                      <td>{Escape(PrettyStation(p.Station))}</td>
                      <td>{Escape(phase.Key)}</td>
                      <td><time>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</time></td>
                      <td>{p.LeadHours}h</td>
                      <td class="num"><strong>{p.ProbWet.ToString("0.00", Ci)}</strong></td>
                      <td class="num">{p.ClimatologyPWet.ToString("0.00", Ci)}</td>
                      <td><small><code>{Escape(p.Version)}</code></small></td>
                    </tr>
                    """);
            }

            content.Append(Ci, $"""
                <h3>Hourly detail (next 96h)</h3>
                <figure>
                  <table>
                    <thead>
                      <tr>
                        <th>Station</th>
                        <th>Phase</th>
                        <th>Valid time</th>
                        <th>Lead</th>
                        <th class="num">P(wet)</th>
                        <th class="num">Climatology</th>
                        <th>Version</th>
                      </tr>
                    </thead>
                    <tbody>
                {tbody}    </tbody>
                  </table>
                </figure>
                """);
        }

        content.Append("</section>");
        return WrapPage(input, "Precipitation", "precipitation", content.ToString());
    }

    /// <summary>
    /// Build a champion-vs-challenger line series at a single lead for one station,
    /// one entry per phase in <see cref="PrecipPhases.Comparable"/>. Caller sets the
    /// earliest valid time to include — the precipitation page uses "now minus 1h" to
    /// get the forecast trajectory, while forecast-vs-truth uses <c>WindowStartUtc</c>
    /// to see historical predictions alongside the observed rainfall line.
    /// </summary>
    private static List<LineSeries> BuildChampionVsChallengerSeries(
        IReadOnlyList<PrecipForecastPoint> stationRows,
        IReadOnlyDictionary<string, string> phaseByVersion,
        DateTime minValidTimeUtc,
        int leadHours)
    {
        var series = new List<LineSeries>();
        foreach (var phase in PrecipPhases.Comparable)
        {
            var pts = stationRows
                .Where(r => r.LeadHours == leadHours
                         && PrecipPhases.Bucket(phaseByVersion, r.Version) == phase
                         && r.ValidTimeUtc >= minValidTimeUtc)
                .GroupBy(r => r.ValidTimeUtc)
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries(phase.ChampionVsChallengerLabel!, phase.Color!, pts));
        }
        return series;
    }
}
