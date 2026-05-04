using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Renders <c>forecasts-{lead}h.html</c> — one page per POC lead (24/48/72/96/120) showing
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
        foreach (var lead in Leads.Full)
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
        // The per-NWP label + colour table lives in SitePages.NwpsForTemperature();
        // dropping it here means a colour change or model addition is one edit, not
        // four-files-worth of grep-and-replace.
        var series = new List<LineSeries>();
        foreach (var nwp in NwpsForTemperature())
        {
            var pts = future
                .Select(p => (Valid: p.ValidTimeUtc, Val: nwp.Get(p)))
                .Where(t => t.Val.HasValue)
                .Select(t => (X: t.Valid.ToOADate(), Y: t.Val!.Value))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries(nwp.Label, nwp.Color, pts));
        }
        var blendPts = future
            .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.BlendTemperature))
            .ToList();
        series.Add(new LineSeries("Blend", NwpPalette.Blend, blendPts));

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
        s.Append("<h3>Precipitation — P(wet ≥ 0.1 mm/h) vs NWP inputs</h3>");
        s.Append("<p class=\"skill-line\">Top: blender's hourly P(wet) plus climatology. Bottom: each NWP's raw precip rate (mm/h) at the same valid times. Stacked rather than overlaid because the units don't share an axis.</p>");

        // Filter to active stations so a demoted-from-config station whose
        // historical predictions are still on disk doesn't get a panel.
        var stations = input.PrecipPredictions
            .Select(p => p.Station).Distinct()
            .Where(s => input.ActiveStationSlugs.Count == 0 || input.ActiveStationSlugs.Contains(s))
            .OrderBy(st => st, StringComparer.Ordinal).ToList();

        if (stations.Count == 0)
        {
            s.Append("<p><em>No precipitation predictions in window.</em></p>");
            return s.ToString();
        }

        // Same shared NWP table as the temperature chart, with JMA appended
        // (precip-only). Colours are matched so "ECMWF is blue" reads the same
        // up there and down here.
        var nwpSpecs = NwpsForPrecipitation();

        foreach (var station in stations)
        {
            // Pool across blender versions before picking freshest per ValidTime.
            // Same rationale as the temp chart: champion-only collapses to ~one
            // point per anchor; the per-NWP columns are identical regardless of
            // blender version anyway, so pooling doesn't change the input story.
            var latestPerValid = input.PrecipPredictions
                .Where(r => r.Station == station
                            && r.LeadHours == lead
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
                    "No forecast at this lead in the forward window."));
                continue;
            }

            // Top: P(wet) + climatology + per-NWP PoP, all on the [0, 1]
            // probability axis. Per-NWP PoP comes from Open-Meteo's
            // precipitation_probability (0..100 percent, divided by 100
            // here to share the axis). Only ~4 of 8 NWPs publish it
            // (GFS / ECMWF / ICON / GEM via Open-Meteo); the others have
            // no rows in NwpPrecipProbabilities and silently drop out of
            // the legend. Threshold each NWP uses for "any precip" varies
            // and isn't strictly our 0.1 mm/h training label, so the
            // overlay is direction-of-effect, not like-for-like — the
            // skill-line above the chart calls this out.
            var probSeries = new List<LineSeries>
            {
                new($"P(wet)", "#7c4dff",
                    latestPerValid.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet)).ToList()),
                new("Climatology", "#9e9e9e",
                    latestPerValid.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ClimatologyPWet)).ToList()),
            };

            // Filter per-NWP PoP to the same valid-time range the blend
            // covers at this lead, so the chart's X axis stays aligned and
            // we don't drag in points beyond the lead's forward window.
            var minValid = latestPerValid[0].ValidTimeUtc;
            var maxValid = latestPerValid[^1].ValidTimeUtc;
            var nwpPalette = NwpsForPrecipitation()
                .ToDictionary(np => np.Label, np => np.Color, StringComparer.Ordinal);
            foreach (var nwpGroup in input.NwpPrecipProbabilities
                .Where(p => p.ValidTimeUtc >= minValid && p.ValidTimeUtc <= maxValid)
                .GroupBy(p => p.Model)
                .OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                var label = LookupNwpLabel(nwpGroup.Key);
                if (!nwpPalette.TryGetValue(label, out var colour)) colour = "#999";
                var pts = nwpGroup
                    .OrderBy(p => p.ValidTimeUtc)
                    .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.ProbabilityPercent / 100.0))
                    .ToList();
                if (pts.Count > 0)
                    probSeries.Add(new LineSeries($"{label} PoP", colour, pts));
            }

            // Bottom: per-NWP precip rate, mm/h axis. Skip series where every
            // point is null so the legend stays clean.
            var nwpSeries = new List<LineSeries>();
            foreach (var nwp in nwpSpecs)
            {
                var pts = latestPerValid
                    .Select(r => (Valid: r.ValidTimeUtc, Val: nwp.Get(r)))
                    .Where(t => t.Val.HasValue)
                    .Select(t => (X: t.Valid.ToOADate(), Y: t.Val!.Value))
                    .ToList();
                if (pts.Count > 0)
                    nwpSeries.Add(new LineSeries(nwp.Label, nwp.Color, pts));
            }

            s.Append("<div class=\"chart-stack\">");
            s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"P(wet) — {PrettyStation(station)} — +{lead}h",
                XLabel = "",                               // X label only on the bottom chart
                YLabel = "Probability",
                Series = probSeries,
                Height = 220,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));
            s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = "",                                // title only on the top chart
                XLabel = "Valid time (UTC)",
                YLabel = "Precip rate (mm/h)",
                Series = nwpSeries,
                Height = 220,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.0", Ci),
            }));
            s.Append("</div>");

            s.Append(RenderPrecipDailySummaryTable(latestPerValid));
            s.Append(RenderPrecipHourlyConfidenceTable(latestPerValid));
        }

        return s.ToString();
    }

    /// <summary>
    /// Per-date summary table: for each UTC day in the forward window, show
    /// mean P(wet), the driest hour of the day (lowest P(wet) — the "go for
    /// a walk" hour), and counts of hours the conformal calibrator flagged
    /// as confident-wet / ambiguous / confident-dry. Sits between the chart
    /// and the collapsible hourly detail so a glance gives "is today /
    /// tomorrow / day-after wet?" without expanding rows.
    ///
    /// Rendered inline (not collapsible) — this IS the at-a-glance view.
    /// Skipped if no rows have agreement OR conformal data (legacy parquets).
    /// </summary>
    private static string RenderPrecipDailySummaryTable(
        IReadOnlyList<PrecipForecastPoint> latestPerValid)
    {
        if (latestPerValid.Count == 0) return "";

        var anyConformal = latestPerValid.Any(r => !string.IsNullOrEmpty(r.ConformalSetTag));
        var byDay = latestPerValid
            .GroupBy(r => r.ValidTimeUtc.Date)
            .OrderBy(g => g.Key)
            .ToList();

        var rows = new StringBuilder();
        foreach (var day in byDay)
        {
            var dayRows = day.OrderBy(r => r.ValidTimeUtc).ToList();
            var meanP = dayRows.Average(r => r.ProbWet);
            // Driest = minimum P(wet) — the hour the blender thinks is least
            // likely to be wet, i.e. the best "go now" hour for the day.
            var driestR = dayRows.MinBy(r => r.ProbWet)!;
            // Conformal vote tally — counts of hours in each set when the
            // calibrator's tagged this row. "Wet" = confident wet for precip
            // (positive class is wet here, no inversion vs dry-window page).
            int wet = 0, dry = 0, amb = 0, untagged = 0;
            foreach (var r in dayRows)
            {
                switch (r.ConformalSetTag)
                {
                    case "Wet":       wet++; break;
                    case "Dry":       dry++; break;
                    case "Ambiguous": amb++; break;
                    default:          untagged++; break;
                }
            }
            var conformalCells = anyConformal
                ? $"""
                    <td class="num"><span class="conf conf-high">{wet}</span></td>
                    <td class="num"><span class="conf conf-low">{amb}</span></td>
                    <td class="num"><span class="conf conf-high">{dry}</span></td>
                  """
                : "";
            // Tint mean-P(wet) cell so the "wet day vs dry day" reads at a glance.
            var meanColor = meanP >= 0.5 ? "#c62828"
                          : meanP >= 0.3 ? "#ef6c00"
                          : meanP >= 0.15 ? "#f9a825"
                          : "#2e7d32";
            rows.Append(Ci, $"""
                <tr>
                  <td><time datetime="{day.Key:yyyy-MM-dd}">{day.Key:ddd dd MMM}</time></td>
                  <td class="num" style="color: {meanColor}; font-weight: 600">{(meanP * 100).ToString("0", Ci)}%</td>
                  <td class="num">{(driestR.ProbWet * 100).ToString("0", Ci)}% <small>at {driestR.ValidTimeUtc:HH'Z'}</small></td>
                  <td class="num">{dayRows.Count}</td>
                  {conformalCells}
                </tr>
                """);
        }

        var conformalHeader = anyConformal
            ? """
              <th class="num" title="Hours the calibrator says are confidently wet at the 90% set">conf wet h</th>
              <th class="num" title="Hours the calibrator can't commit at the 90% set">amb h</th>
              <th class="num" title="Hours the calibrator says are confidently dry at the 90% set">conf dry h</th>
              """
            : "";
        return $"""
            <figure>
              <table>
                <thead>
                  <tr>
                    <th>Date (UTC)</th>
                    <th class="num">Mean P(wet)</th>
                    <th class="num" title="Lowest forecast P(wet) of the day — the best 'go now' hour">Driest hour</th>
                    <th class="num">n_h</th>
                    {conformalHeader}
                  </tr>
                </thead>
                <tbody>
            {rows}    </tbody>
              </table>
            </figure>
            """;
    }

    /// <summary>
    /// Compact hourly-detail table under each precip chart showing P(wet) +
    /// a confidence chip per upcoming hour. Confidence comes from the
    /// per-NWP wet-vote spread already persisted on
    /// <see cref="PrecipForecastPoint.AgreementWet01"/>: high = ensemble
    /// near-unanimous; low = NWPs split. Read this as "the headline P(wet)
    /// could be 30% but if the NWPs are 50/50 split, treat that as a
    /// genuinely uncertain forecast, not a confident '70% chance dry'."
    ///
    /// Skipped silently when the agreement column is missing on every row
    /// (older parquets pre-dating PrecipPredictionRow.PrecipAgreementWet01,
    /// or 3g-only outputs where the per-NWP features aren't computed).
    /// </summary>
    private static string RenderPrecipHourlyConfidenceTable(
        IReadOnlyList<PrecipForecastPoint> latestPerValid)
    {
        if (latestPerValid.Count == 0) return "";
        if (!latestPerValid.Any(r => r.AgreementWet01.HasValue)) return "";

        var rows = new StringBuilder();
        var anyConformal = latestPerValid.Any(r => !string.IsNullOrEmpty(r.ConformalSetTag));
        foreach (var r in latestPerValid)
        {
            // Agreement is in [0, 1] = fraction of NWPs voting wet that hour.
            // Confidence (= "unanimity") is high when agreement is near 0 or 1.
            // Map to {high, medium, low} so the chip reads at a glance.
            var (label, cls) = ConfidenceFromAgreement(r.AgreementWet01);
            var agreementCell = r.AgreementWet01.HasValue
                ? (r.AgreementWet01.Value * 100).ToString("0", Ci) + "%"
                : "—";
            // Tint the P(wet) cell by confidence: greyer when low-confidence,
            // bolder when high. Cheaper than chart annotations and reads at
            // table-scan speed.
            var pwetStyle = label switch
            {
                "high"   => "color: #4527a0; font-weight: 600",
                "medium" => "color: #7c4dff",
                "low"    => "color: #9e9e9e; font-style: italic",
                _        => "",
            };
            // Second confidence chip from the conformal calibrator (precip-
            // conformal-fit). Only rendered when at least one row in this
            // (station, lead) batch has a tag — keeps the column off legacy
            // forecasts that pre-date the calibrator.
            var conformalTd = anyConformal
                ? "<td>" + RenderConformalChip(r.ConformalSetTag) + "</td>"
                : "";
            rows.Append(Ci, $"""
                <tr>
                  <td><time datetime="{r.ValidTimeUtc:yyyy-MM-ddTHH:mm}Z">{r.ValidTimeUtc:MM-dd HH'Z'}</time></td>
                  <td class="num" style="{pwetStyle}">{(r.ProbWet * 100).ToString("0", Ci)}%</td>
                  <td class="num">{agreementCell}</td>
                  <td><span class="conf conf-{cls}">{label}</span></td>
                  {conformalTd}
                </tr>
                """);
        }

        var conformalTh = anyConformal
            ? "<th>Conformal <small>(90% set)</small></th>"
            : "";
        return $"""
            <details class="hourly-detail">
              <summary>Hourly P(wet) + NWP agreement</summary>
              <figure>
                <table>
                  <thead>
                    <tr>
                      <th>Valid time</th>
                      <th class="num">P(wet)</th>
                      <th class="num">NWPs wet</th>
                      <th>Confidence</th>
                      {conformalTh}
                    </tr>
                  </thead>
                  <tbody>
            {rows}      </tbody>
                </table>
              </figure>
            </details>
            """;
    }

    /// <summary>
    /// Conformal set chip — same {"Dry", "Wet", "Ambiguous"} tag the
    /// dry-window page uses. Confident dry/wet read as high-class colour;
    /// ambiguous reads low. "—" when no calibrator has been fit for this
    /// (version, lead).
    /// </summary>
    private static string RenderConformalChip(string? tag) => tag switch
    {
        "Wet"       => "<span class=\"conf conf-high\">confident wet</span>",
        "Dry"       => "<span class=\"conf conf-high\">confident dry</span>",
        "Ambiguous" => "<span class=\"conf conf-low\">ambiguous</span>",
        _           => "<span class=\"conf conf-unknown\">—</span>",
    };

    /// <summary>
    /// Bucket per-NWP wet-vote agreement into a 3-tier confidence label.
    /// Unanimity = 2 * |agreement - 0.5| — distance from the 0.5 split,
    /// so 0% wet (all dry) and 100% wet (all wet) both score 1.0
    /// (everyone unanimous), 50% wet scores 0 (worst split).
    /// ≥0.6 → high, ≥0.2 → medium, rest → low.
    /// Returns (label-text, css-class). Null agreement → "—" / "unknown"
    /// so the renderer degrades gracefully on legacy rows.
    /// </summary>
    private static (string Label, string Cls) ConfidenceFromAgreement(double? agreement)
    {
        if (!agreement.HasValue) return ("—", "unknown");
        var unanimity = 2.0 * Math.Abs(agreement.Value - 0.5);
        return unanimity switch
        {
            >= 0.6 => ("high", "high"),
            >= 0.2 => ("medium", "medium"),
            _      => ("low", "low"),
        };
    }
}
