using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Merged skill view: "eyeball" side-by-side plots of every active version against
    /// truth, plus rolling quantitative MAE per lead for the temperature blender. The
    /// two used to live on separate pages (forecast-vs-truth + verify) but the reader
    /// ends up cross-checking them anyway, so a single page keeps the story — how good
    /// is each blender vs reality? — in one place.
    /// </summary>
    public static string RenderSkill(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Skill</h2>
                <p>Eyeball comparisons first — each active blender's trajectory plotted against
                   observed truth. Rolling quantitative MAE by lead follows, so you can see
                   whether the visual impression holds up under aggregation.</p>
              </hgroup>
            """);

        content.Append("<h3>Temperature — vs truth</h3>");
        content.Append(RenderTempVsTruthBlock(input));

        content.Append("<hr/><h3>Temperature — rolling MAE</h3>");
        content.Append(RenderRollingMaeBlock(input));

        content.Append("<hr/><h3>Precipitation — P(wet) vs observed rainfall</h3>");
        content.Append(RenderPrecipVsTruthBlock(input));

        content.Append("<hr/><h3>Dry window — predicted vs observed</h3>");
        content.Append(Ci, $"""
            <p class="skill-line">One row per target UTC day × window length. The "observed" column is blank for dates
               beyond the last full rainfall day.</p>
            """);
        content.Append(RenderDryWindowVsTruthTable(input));

        content.Append("</section>");
        return WrapPage(input, "Skill", "skill", content.ToString());
    }

    // -------------------------------------------------------------------------------
    // Eyeball: temperature vs truth, grouped by phase so champion/challenger lines at
    // different leads don't pile up in one unreadable chart.
    // -------------------------------------------------------------------------------
    private static string RenderTempVsTruthBlock(SiteInputs input)
    {
        var content = new StringBuilder();

        var versionsByPhase = input.PhaseByVersion
            .GroupBy(kv => BucketPhase(kv.Value))
            .ToDictionary(g => g.Key, g => g.Select(kv => kv.Key).ToHashSet(StringComparer.Ordinal));
        var untagged = input.Predictions
            .Select(p => p.ModelVersion).Distinct()
            .Where(v => !input.PhaseByVersion.ContainsKey(v))
            .ToHashSet(StringComparer.Ordinal);
        if (untagged.Count > 0)
        {
            if (!versionsByPhase.TryGetValue("other", out var others))
                versionsByPhase["other"] = new HashSet<string>(untagged, StringComparer.Ordinal);
            else
                foreach (var v in untagged) others.Add(v);
        }

        (string Key, string Title, string Description)[] phaseSpecs =
        {
            ("2b", "Phase 2b lean (13 features)",
                "Six per-model temperatures, their mean/std/range, and cyclical hour/day-of-year encodings. The original champion."),
            ("2c", "Phase 2c rich (88 features)",
                "Adds per-model dew point, RH, cloud {total/low/mid/high}, wind speed/dir/gusts, surface pressure, plus cross-model aggregates. Challenger."),
            ("other", "Other versions",
                "Versions with no training metadata on disk — typically pre-2b experiments left in the manifest."),
        };
        bool anyDrawn = false;
        foreach (var spec in phaseSpecs)
        {
            if (!versionsByPhase.TryGetValue(spec.Key, out var versions) || versions.Count == 0)
                continue;
            anyDrawn = true;
            var filtered = input.Predictions.Where(p => versions.Contains(p.ModelVersion)).ToList();
            content.Append(Ci, $"""
                <article>
                  <hgroup>
                    <h4>{Escape(spec.Title)}</h4>
                    <p class="skill-line">{Escape(spec.Description)}</p>
                  </hgroup>
                """);
            content.Append(RenderTempVsTruthChart(input, filtered, spec.Key));
            content.Append("</article>");
        }

        if (!anyDrawn)
            content.Append("<p><em>No temperature predictions in window — run <code>predict</code>.</em></p>");

        return content.ToString();
    }

    private static string RenderTempVsTruthChart(
        SiteInputs input,
        IReadOnlyList<PredictionRow> filtered,
        string phaseKey)
    {
        var series = new List<LineSeries>();

        var era5Pts = input.TruthByTime
            .Where(kv => kv.Key >= input.WindowStartUtc)
            .OrderBy(kv => kv.Key)
            .Select(kv => (X: kv.Key.ToOADate(), Y: kv.Value))
            .ToList();
        if (era5Pts.Count > 0)
            series.Add(new LineSeries("ERA5 (truth)", "#ef5350", era5Pts));

        var metarPts = input.MetarByTime
            .Where(m => m.ObservedTimeUtc >= input.WindowStartUtc)
            .OrderBy(m => m.ObservedTimeUtc)
            .Select(m => (X: m.ObservedTimeUtc.ToOADate(), Y: m.Temperature2m))
            .ToList();
        if (metarPts.Count > 0)
        {
            var label = string.IsNullOrWhiteSpace(input.MetarStation) ? "METAR" : $"METAR {input.MetarStation}";
            series.Add(new LineSeries(label, "#ffa726", metarPts));
        }

        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),
            (48, "#7c4dff"),
            (72, "#4527a0"),
        };
        foreach (var (lead, color) in leadSpecs)
        {
            var pts = filtered
                .Where(p => p.LeadHours == lead)
                .GroupBy(p => p.ValidTimeUtc)
                .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
                .OrderBy(p => p.ValidTimeUtc)
                .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.BlendTemperature))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries($"Blend +{lead}h", color, pts));
        }

        if (series.Count == 0)
            return RenderEmptyChart($"Temperature — phase {phaseKey}", "No overlap between predictions and truth in window.");

        return LineChartRenderer.Render(new LineChartSpec
        {
            Title = $"Temperature vs truth — phase {phaseKey}",
            XLabel = "Time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 360,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
        });
    }

    // -------------------------------------------------------------------------------
    // Rolling quantitative MAE per (version, lead). One chart per lead — one line per
    // version. This is the "does the eyeball match the numbers?" cross-check.
    // -------------------------------------------------------------------------------
    private static string RenderRollingMaeBlock(SiteInputs input)
    {
        if (input.RollingMae.Count == 0)
            return "<p><em>No rolling MAE points computed — the window is too short or there's no matching ERA5 truth yet.</em></p>";

        var content = new StringBuilder();
        foreach (var lead in new[] { 24, 48, 72 })
        {
            var versions = input.RollingMae.Where(r => r.LeadHours == lead)
                .Select(r => r.ModelVersion).Distinct().OrderBy(v => v, StringComparer.Ordinal).ToList();

            var series = new List<LineSeries>();
            var palette = new[] { "#7c4dff", "#26a69a", "#ef5350", "#ffa726", "#42a5f5" };
            for (int i = 0; i < versions.Count; i++)
            {
                var v = versions[i];
                var pts = input.RollingMae
                    .Where(r => r.LeadHours == lead && r.ModelVersion == v)
                    .OrderBy(r => r.WindowEndUtc)
                    .Select(r => (X: r.WindowEndUtc.ToOADate(), Y: r.BlendMae))
                    .ToList();
                if (pts.Count > 0)
                    series.Add(new LineSeries(v, palette[i % palette.Length], pts));
            }

            content.Append(Ci, $"<h4>Lead +{lead}h</h4>");
            if (series.Count == 0)
            {
                content.Append(RenderEmptyChart($"Rolling MAE — lead {lead}h", "No scored predictions at this lead."));
                continue;
            }

            content.Append(LineChartRenderer.Render(new LineChartSpec
            {
                Title = $"Rolling MAE — lead +{lead}h",
                XLabel = "Window end (UTC)",
                YLabel = "MAE (°C)",
                Series = series,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));
        }
        return content.ToString();
    }

    // -------------------------------------------------------------------------------
    // Eyeball: precipitation vs observed rainfall. Overlays observed hourly rainfall
    // (capped at 1 mm/h so it shares the 0-1 probability axis) on a per-phase P(wet)
    // line, then a three-way 24h comparison of every phase against truth.
    // -------------------------------------------------------------------------------
    private static string RenderPrecipVsTruthBlock(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <p class="skill-line">P(wet) is the blender's probability that the next hour sees ≥ 0.1 mm.
               The rainfall line is the same 4-of-4 hourly aggregation used for verification — partial
               hours are dropped to avoid flipping wet↔dry at the boundary.</p>
            """);

        var stations = input.PrecipPredictions.Select(p => p.Station).Distinct()
            .OrderBy(s => s, StringComparer.Ordinal).ToList();
        if (stations.Count == 0)
        {
            content.Append("<p><em>No precipitation predictions in window.</em></p>");
            return content.ToString();
        }

        foreach (var station in stations)
            content.Append(RenderPrecipVsTruthChart(input, station));

        return content.ToString();
    }

    private static string RenderPrecipVsTruthChart(SiteInputs input, string station)
    {
        var content = new StringBuilder();
        content.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

        var stationPredictions = input.PrecipPredictions.Where(p => p.Station == station).ToList();

        List<(double X, double Y)> truthPts = new();
        if (input.RainfallTruth.TryGetValue(station, out var truth) && truth.Count > 0)
        {
            truthPts = truth
                .Where(kv => kv.Key >= input.WindowStartUtc)
                .OrderBy(kv => kv.Key)
                .Select(kv => (X: kv.Key.ToOADate(), Y: Math.Min(1.0, kv.Value)))
                .ToList();
        }

        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),
            (48, "#7c4dff"),
            (72, "#4527a0"),
        };
        bool anyRendered = false;
        foreach (var spec in PrecipPhases.All)
        {
            var phaseRows = stationPredictions
                .Where(p => PrecipPhases.Bucket(input.PhaseByVersion, p.Version) == spec)
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
                    .Where(r => r.LeadHours == lead)
                    .OrderBy(r => r.ValidTimeUtc)
                    .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                    .ToList();
                if (pts.Count > 0)
                    series.Add(new LineSeries($"P(wet) +{lead}h", color, pts));
            }
            if (truthPts.Count > 0)
                series.Add(new LineSeries("Observed mm/h (capped at 1)", "#ef5350", truthPts));

            if (series.Count == 0)
            {
                content.Append(RenderEmptyChart(
                    $"P(wet) vs observed — {station} — {spec.Key}",
                    "No forecast or truth in window."));
                continue;
            }

            content.Append(Ci, $"<h5>{Escape(spec.ShortTitle)}</h5>");
            content.Append(LineChartRenderer.Render(new LineChartSpec
            {
                Title = $"P(wet) vs observed rainfall — {PrettyStation(station)} — Phase {spec.Key}",
                XLabel = "Time (UTC)",
                YLabel = "Probability / mm·h⁻¹ (capped)",
                Series = series,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));
        }

        var cvc = BuildChampionVsChallengerSeries(stationPredictions, input.PhaseByVersion, input.WindowStartUtc, leadHours: 24);
        if (cvc.Count >= 2)
        {
            if (truthPts.Count > 0)
                cvc.Add(new LineSeries("Observed mm/h (capped at 1)", "#ef5350", truthPts));
            content.Append("<h5>Three-way comparison — +24h lead</h5>");
            content.Append(LineChartRenderer.Render(new LineChartSpec
            {
                Title = $"3a vs 3a_isotonic vs 3c vs observed — {PrettyStation(station)} — +24h",
                XLabel = "Time (UTC)",
                YLabel = "Probability / mm·h⁻¹ (capped)",
                Series = cvc,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));
        }

        if (!anyRendered)
            content.Append(RenderEmptyChart($"P(wet) vs observed — {station}", "No forecast or truth in window."));

        return content.ToString();
    }

    // -------------------------------------------------------------------------------
    // Dry-window vs observed. One table per station × window length showing each
    // lead's predicted probability alongside the observed verdict (dry / wet / —).
    // -------------------------------------------------------------------------------
    private static string RenderDryWindowVsTruthTable(SiteInputs input)
    {
        if (input.DryWindowPredictions.Count == 0)
            return "<p><em>No dry-window predictions in window.</em></p>";

        var observed = ComputeObservedDryWindows(input);

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct()
            .OrderBy(s => s, StringComparer.Ordinal).ToList();
        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = new[] { 24, 48, 72 };

        var content = new StringBuilder();

        foreach (var station in stations)
        {
            content.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

            foreach (var window in windows)
            {
                var cutoff = input.GeneratedAtUtc.Date.AddDays(-7);
                var latest = input.DryWindowPredictions
                    .Where(d => d.Station == station && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

                if (latest.Count == 0) continue;

                var dates = latest.Select(d => d.TargetDateUtc).Distinct().OrderBy(d => d).ToList();

                var tbody = new StringBuilder();
                foreach (var date in dates)
                {
                    var byLead = latest.Where(d => d.TargetDateUtc == date).ToDictionary(d => d.LeadHours);

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                            leadCells.Append(Ci, $"<td class=\"num\">{d.ProbHasDryWindow.ToString("0.00", Ci)}</td>");
                        else
                            leadCells.Append("<td class=\"num\">—</td>");
                    }

                    var observedCell = observed.TryGetValue((station, window, date), out var obs)
                        ? (obs ? "<strong>dry</strong>" : "wet")
                        : "—";

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time>{date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          <td>{observedCell}</td>
                        </tr>
                        """);
                }

                content.Append(Ci, $"""
                    <h5>{window}-hour dry window</h5>
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
                            <th>Observed</th>
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);
            }
        }

        return content.ToString();
    }

    // -------------------------------------------------------------------------------
    // Shared helpers — used to live on the old forecast-vs-truth / precipitation
    // pages. Kept as internal statics because SitePagesTests reaches for them.
    // -------------------------------------------------------------------------------

    private static string BucketPhase(string? phase)
    {
        if (string.IsNullOrWhiteSpace(phase)) return "other";
        if (phase.StartsWith("2b", StringComparison.OrdinalIgnoreCase)) return "2b";
        if (phase.Equals("2c", StringComparison.OrdinalIgnoreCase)) return "2c";
        return "other";
    }

    /// <summary>
    /// Build a champion-vs-challenger line series at a single lead for one station,
    /// one entry per phase in <see cref="PrecipPhases.Comparable"/>. Caller sets the
    /// earliest valid time to include.
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

    /// <summary>
    /// For every (station, window, target-date) triple referenced by a prediction, walk
    /// that day's hourly rainfall and check whether at least <c>window</c> consecutive
    /// hours are ≤ 0.1 mm within the UTC day. Returns no entry for days with incomplete
    /// 24-hour coverage.
    /// </summary>
    internal static IReadOnlyDictionary<(string Station, int WindowHours, DateTime TargetDate), bool>
        ComputeObservedDryWindows(SiteInputs input)
    {
        var result = new Dictionary<(string, int, DateTime), bool>();
        var triples = input.DryWindowPredictions
            .Select(d => (d.Station, d.WindowHours, d.TargetDateUtc.Date))
            .Distinct();

        foreach (var (station, window, date) in triples)
        {
            if (!input.RainfallTruth.TryGetValue(station, out var hourly) || hourly.Count == 0)
                continue;

            var hours = new double?[24];
            bool complete = true;
            for (int h = 0; h < 24; h++)
            {
                var ts = date.AddHours(h);
                if (hourly.TryGetValue(ts, out var mm)) hours[h] = mm;
                else { complete = false; break; }
            }
            if (!complete) continue;

            int run = 0;
            bool found = false;
            for (int h = 0; h < 24; h++)
            {
                if ((hours[h] ?? double.NaN) <= 0.1) { run++; if (run >= window) { found = true; break; } }
                else run = 0;
            }
            result[(station, window, date)] = found;
        }
        return result;
    }
}
