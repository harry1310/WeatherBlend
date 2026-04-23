using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Dedicated page that plots blend forecasts against observed truth, so the user can
    /// eyeball whether a given model version is actually tracking reality. Temperature is
    /// split into two charts (one per phase: 2b lean vs 2c rich) so champion/challenger
    /// lines don't overlap and the user can compare at a glance. Precipitation gets one
    /// chart per station with observed hourly rainfall overlaid on the P(wet) forecasts.
    /// Dry-window is a grid of observed-vs-predicted for completed target days.
    /// </summary>
    public static string RenderForecastVsTruth(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Forecast vs truth</h2>
                <p>Side-by-side view of what each active blender predicted vs what actually happened.
                   Temperature lines run forward past "now" because the blend projects 24/48/72h ahead;
                   truth (ERA5, METAR, rainfall) stops at the last observation.</p>
              </hgroup>
            """);

        // --- Temperature by phase --------------------------------------------------
        var versionsByPhase = input.PhaseByVersion
            .GroupBy(kv => BucketPhase(kv.Value))
            .ToDictionary(g => g.Key, g => g.Select(kv => kv.Key).ToHashSet(StringComparer.Ordinal));
        // Versions in predictions but with no metadata phase go to "other" — at least
        // they still get rendered somewhere rather than being silently dropped.
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
            ("2b", "Temperature — Phase 2b lean (13 features)",
                "Six per-model temperatures, their mean/std/range, and cyclical hour/day-of-year encodings. The original champion."),
            ("2c", "Temperature — Phase 2c rich (88 features)",
                "Adds per-model dew point, RH, cloud {total/low/mid/high}, wind speed/dir/gusts, surface pressure, plus cross-model aggregates. Trained to challenge 2b."),
            ("other", "Temperature — other versions",
                "Versions with no training metadata on disk — typically pre-2b experiments left in the manifest."),
        };
        foreach (var spec in phaseSpecs)
        {
            if (!versionsByPhase.TryGetValue(spec.Key, out var versions) || versions.Count == 0)
                continue;

            var filtered = input.Predictions.Where(p => versions.Contains(p.ModelVersion)).ToList();
            content.Append(Ci, $"""
                <article>
                  <hgroup>
                    <h3>{Escape(spec.Title)}</h3>
                    <p class="skill-line">{Escape(spec.Description)}</p>
                  </hgroup>
                """);
            content.Append(RenderTempVsTruthChart(input, filtered, spec.Key));
            content.Append("</article>");
        }

        if (versionsByPhase.Count == 0)
        {
            content.Append("<p><em>No temperature predictions with metadata — run <code>train</code> and <code>predict</code>.</em></p>");
        }

        // --- Precipitation per station ---------------------------------------------
        content.Append("<hr/>");
        content.Append(Ci, $"""
            <h3>Precipitation — P(wet) vs observed rainfall</h3>
            <p class="skill-line">P(wet) is the blender's probability that the next hour sees ≥ 0.1 mm. The rainfall line is the same 4-of-4 hourly aggregation used for verification — partial hours are dropped to avoid flipping wet↔dry at the boundary.</p>
            """);
        var precipStations = input.PrecipPredictions.Select(p => p.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        if (precipStations.Count == 0)
        {
            content.Append("<p><em>No precipitation predictions in window.</em></p>");
        }
        else
        {
            foreach (var station in precipStations)
                content.Append(RenderPrecipVsTruthChart(input, station));
        }

        // --- Dry-window per station × window --------------------------------------
        content.Append("<hr/>");
        content.Append(Ci, $"""
            <h3>Dry window — predicted probability vs (eventual) observation</h3>
            <p class="skill-line">One row per target UTC day × window length. The "observed" column is blank for dates beyond the last full rainfall day. Rolling Brier skill scores are on the <a href="verify.html">verification page</a>.</p>
            """);
        content.Append(RenderDryWindowVsTruthTable(input));

        content.Append("</section>");
        return WrapPage(input, "Forecast vs truth", "forecast-vs-truth", content.ToString());
    }

    private static string BucketPhase(string? phase)
    {
        if (string.IsNullOrWhiteSpace(phase)) return "other";
        // "2b" and "2b_redo" both belong in the lean bucket. "2c" is rich. Anything else
        // that isn't one of those two is shown in the "other" bucket.
        if (phase.StartsWith("2b", StringComparison.OrdinalIgnoreCase)) return "2b";
        if (phase.Equals("2c", StringComparison.OrdinalIgnoreCase)) return "2c";
        return "other";
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
            // Latest prediction per valid-time — if the same (lead, valid_time) was
            // produced by two same-phase versions (rare, but possible across a retrain)
            // the newer PredictionMadeAtUtc wins.
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

    private static string RenderPrecipVsTruthChart(SiteInputs input, string station)
    {
        var content = new StringBuilder();
        content.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

        var stationPredictions = input.PrecipPredictions.Where(p => p.Station == station).ToList();

        // Precompute the observed-rainfall series once; it overlays on every phase chart.
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

            // Rainfall truth: same 0–1 axis by capping at 1. The capped line functions
            // as "was it meaningfully raining?" without a second Y-axis.
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

        // Champion-vs-challenger overlay at +24h with observed truth so the reader can
        // see which phase tracked reality better for the lead that matters most.
        var cvc = BuildChampionVsChallengerSeries(stationPredictions, input.PhaseByVersion, input.WindowStartUtc, leadHours: 24);
        // Drop the generatedAt clamp above — for forecast-vs-truth we want history too.
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

    private static string RenderDryWindowVsTruthTable(SiteInputs input)
    {
        if (input.DryWindowPredictions.Count == 0)
            return "<p><em>No dry-window predictions in window.</em></p>";

        // "Observed dry window existed?" per (station, date, window). Use the same
        // rainfall truth we already loaded for the precip chart — a dry hour is one
        // where the 4-of-4 hourly SUM is ≤ 0.1 mm, and a dry window is N consecutive
        // such hours all within the same UTC day.
        var observed = ComputeObservedDryWindows(input);

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
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

    /// <summary>
    /// For every (station, window, target-date) triple referenced by a prediction, walk
    /// that day's hourly rainfall and check whether at least <c>window</c> consecutive
    /// hours are ≤ 0.1 mm within the UTC day. Returns null for days with no complete
    /// 24-hour coverage.
    /// </summary>
    internal static IReadOnlyDictionary<(string Station, int WindowHours, DateTime TargetDate), bool>
        ComputeObservedDryWindows(SiteInputs input)
    {
        var result = new Dictionary<(string, int, DateTime), bool>();
        // Group by (station, date) so we only walk each day once.
        var triples = input.DryWindowPredictions
            .Select(d => (d.Station, d.WindowHours, d.TargetDateUtc.Date))
            .Distinct();

        foreach (var (station, window, date) in triples)
        {
            if (!input.RainfallTruth.TryGetValue(station, out var hourly) || hourly.Count == 0)
                continue;

            // Need all 24 hourly buckets for a fair verdict. If any are missing we
            // don't emit a row — this leaves the table cell as "—".
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
