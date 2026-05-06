using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Forecasts pages, split per variable per lead since the 2026-05-04 site
    /// rework. The reader picks one of three variables in the variable sub-nav
    /// (Temperature / Rain / Dry window), then for temp + rain a per-lead
    /// inner sub-nav (24 / 48 / 72 / 96 / 120 h). Dry-window is per-day so
    /// has no lead axis.
    /// </summary>
    public static string RenderForecastsTemp(SiteInputs input, int lead)
    {
        var body = new StringBuilder();
        body.Append("<section>");
        // Sub-navs first so their Y positions are fixed across all sub-tabs
        // and lead choices — flicking between them doesn't jolt the page as
        // chart heights vary.
        body.Append(RenderForecastsSubNav("temp"));
        body.Append(RenderLeadSubNav("forecasts-temp", lead));

        body.Append(Ci, $"""
              <hgroup>
                <h2>Temperature +{lead}h</h2>
                <p>Per-NWP lines plus blend (2b solid, 2c lighter).</p>
              </hgroup>
            """);

        body.Append(RenderTempSection(input, lead));

        body.Append("</section>");
        return WrapPage(input, $"Temperature forecast +{lead}h", "forecasts", body.ToString());
    }

    public static string RenderForecastsRain(SiteInputs input, int lead)
    {
        var body = new StringBuilder();
        body.Append("<section>");
        body.Append(RenderForecastsSubNav("rain"));
        body.Append(RenderLeadSubNav("forecasts-rain", lead));

        body.Append(Ci, $"""
              <hgroup>
                <h2>Rain +{lead}h</h2>
                <p>Per-station P(wet ≥ 0.1 mm/h) — 3a solid, 3c lighter — plus per-NWP precip rate (one chart, point forecast for Bonehill).</p>
              </hgroup>
            """);

        body.Append(RenderPrecipSection(input, lead));

        body.Append("</section>");
        return WrapPage(input, $"Rain forecast +{lead}h", "forecasts", body.ToString());
    }

    /// <summary>
    /// Variable sub-nav across the Forecasts pages — three pill links (Temperature
    /// / Rain / Dry window). Same shape as <see cref="RenderModelsSubNav"/> and
    /// <see cref="RenderSkillSubNav"/>. Temperature + Rain link to their +24h
    /// landing page; Dry window has no lead axis so links to the day-aggregate
    /// page directly.
    /// </summary>
    internal static string RenderForecastsSubNav(string activeSlug)
    {
        var entries = new (string Slug, string File, string Label)[]
        {
            ("temp",       "forecasts-temp-24h.html",  "Temperature"),
            ("rain",       "forecasts-rain-24h.html",  "Rain"),
            ("dry-window", "forecasts-dry-window.html", "Dry window"),
        };
        var s = new StringBuilder();
        s.Append("<nav class=\"lead-nav\"><ul>");
        foreach (var (slug, file, label) in entries)
        {
            var cls = slug == activeSlug ? " class=\"active\"" : "";
            s.Append(Ci, $"<li><a href=\"{file}\"{cls}>{Escape(label)}</a></li>");
        }
        s.Append("</ul></nav>");
        return s.ToString();
    }

    private static string RenderLeadSubNav(string pageBase, int current)
    {
        var items = new StringBuilder();
        // ForecastsTempRain prepends +12h for 2d/3d (exact-runtime) on top of
        // the standard {24, 48, 72, 96, 120} set used by 2b/2c/3a/3c.
        foreach (var lead in Leads.ForecastsTempRain)
        {
            var cls = lead == current ? " class=\"active\"" : "";
            items.Append(Ci, $"""<li><a href="{pageBase}-{lead}h.html"{cls}>+{lead}h</a></li>""");
        }
        return $"""<nav class="lead-nav"><ul>{items}</ul></nav>""";
    }

    private static string RenderTempSection(SiteInputs input, int lead)
    {
        // Per-NWP overlay chart with champion + challenger blend lines on top.
        // Champion (2b) draws solid in brand purple; challenger (2c), when
        // present at this lead, draws dashed in the same colour so the eye
        // reads them as paired ("same prediction, two methods").
        //
        // Two passes through input.Predictions:
        //   - The per-NWP columns (TempGfs / TempEcmwf / …) are identical
        //     across blender versions — pool by ValidTime regardless of
        //     ModelVersion and pick the freshest PMT.
        //   - For the blend line itself we partition by phase first so each
        //     phase's BlendTemperature time series is built from rows of
        //     the right version.
        var poolFuture = input.Predictions
            .Where(p => p.LeadHours == lead
                        && p.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        var s = new StringBuilder();
        s.Append("<h3>Temperature — blend vs per-model inputs</h3>");

        if (poolFuture.Count == 0)
        {
            s.Append("<p><em>No +").Append(lead).Append("h temperature forecast available.</em></p>");
            return s.ToString();
        }

        // NWPs first so the brand-purple Blend draws last and sits visually on top.
        var series = new List<LineSeries>();
        foreach (var nwp in NwpsForTemperature())
        {
            var pts = poolFuture
                .Select(p => (Valid: p.ValidTimeUtc, Val: nwp.Get(p)))
                .Where(t => t.Val.HasValue)
                .Select(t => (X: t.Valid.ToOADate(), Y: t.Val!.Value))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries(nwp.Label, nwp.Color, pts));
        }

        // Champion + challenger blend lines, ordered by ActivePhasePolicy
        // priority so the FIRST entry (champion) draws solid and any
        // subsequent entry (challenger) draws dashed in the same colour.
        // Skip any phase that has zero rows in the forward window — common
        // for fresh challengers whose predictions haven't reached this lead's
        // valid-time band yet.
        var orderedActivePhases = ActivePhasePolicy.ByTarget["temperature"];
        var blendByPhase = input.Predictions
            .Where(p => p.LeadHours == lead
                        && p.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
            .Where(p => input.PhaseByVersion.TryGetValue(p.ModelVersion, out var ph)
                        && orderedActivePhases.Contains(ph, StringComparer.Ordinal))
            .GroupBy(p => input.PhaseByVersion[p.ModelVersion])
            .ToDictionary(g => g.Key, g => g.GroupBy(r => r.ValidTimeUtc)
                                            .Select(gv => gv.OrderByDescending(r => r.PredictionMadeAtUtc).First())
                                            .OrderBy(r => r.ValidTimeUtc)
                                            .ToList());
        // Champion solid in deep purple, challenger solid in lighter purple —
        // same hue family so the eye reads them as paired ("same prediction,
        // two methods"), but distinguishable by saturation. Dashed-line
        // challenger was tried 2026-05-04 and reverted on user feedback.
        for (int i = 0; i < orderedActivePhases.Count; i++)
        {
            var phase = orderedActivePhases[i];
            if (!blendByPhase.TryGetValue(phase, out var phaseRows) || phaseRows.Count == 0) continue;
            // Exact-runtime phases (2d temp, 3d precip) get a distinct
            // magenta — different blender family from the offset_day-trained
            // 2c/3c challengers, so visually pop OUT of the purple family.
            var color = i == 0
                ? NwpPalette.Blend
                : (phase == "2d" || phase == "3d"
                    ? NwpPalette.BlendExactChallenger
                    : NwpPalette.BlendChallenger);
            var label = i == 0 ? $"Blend ({phase} champion)" : $"Blend ({phase} challenger)";
            var pts = phaseRows
                .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.BlendTemperature))
                .ToList();
            series.Add(new LineSeries(label, color, pts));
        }

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
        // Cache the latest per-(station, valid) rows for each station so we
        // build them once per render. The NWP precip rate chart hoisted below
        // the loop reads this cache — it's identical across stations so any
        // one entry will do.
        var latestPerValidByStation = new Dictionary<string, IReadOnlyList<PrecipForecastPoint>>(StringComparer.Ordinal);

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
            latestPerValidByStation[station] = latestPerValid;

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
            // overlay is direction-of-effect, not like-for-like.
            //
            // Champion + challenger P(wet) lines: P(wet) bucketed by
            // ActivePhasePolicy priority so champion (3a) draws solid in
            // brand purple, challenger (3c) draws dashed in the same colour.
            // Climatology + NWP PoP overlay sit on top in their own colours.
            var probSeries = new List<LineSeries>();
            var orderedPrecipPhases = ActivePhasePolicy.ByTarget["precipitation"];
            var precipByPhase = input.PrecipPredictions
                .Where(r => r.Station == station
                            && r.LeadHours == lead
                            && r.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
                .Where(r => input.PhaseByVersion.TryGetValue(r.Version, out var ph)
                            && orderedPrecipPhases.Contains(ph, StringComparer.Ordinal))
                .GroupBy(r => input.PhaseByVersion[r.Version])
                .ToDictionary(g => g.Key, g => g.GroupBy(r => r.ValidTimeUtc)
                                                 .Select(gv => gv.OrderByDescending(r => r.PredictedAtUtc).First())
                                                 .OrderBy(r => r.ValidTimeUtc)
                                                 .ToList());
            // Champion solid in deep purple, challenger solid in lighter
            // purple — same hue family, distinguishable by saturation.
            for (int i = 0; i < orderedPrecipPhases.Count; i++)
            {
                var phase = orderedPrecipPhases[i];
                if (!precipByPhase.TryGetValue(phase, out var phaseRows) || phaseRows.Count == 0) continue;
                // Exact-runtime phases (2d temp, 3d precip) get a distinct
            // magenta — different blender family from the offset_day-trained
            // 2c/3c challengers, so visually pop OUT of the purple family.
            var color = i == 0
                ? NwpPalette.Blend
                : (phase == "2d" || phase == "3d"
                    ? NwpPalette.BlendExactChallenger
                    : NwpPalette.BlendChallenger);
                var label = i == 0 ? $"P(wet) ({phase} champion)" : $"P(wet) ({phase} challenger)";
                var pts = phaseRows
                    .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                    .ToList();
                probSeries.Add(new LineSeries(label, color, pts));
            }
            // Climatology stays a single line — it's a station property, not
            // a blender output. Read off the champion-pooled rows above.
            probSeries.Add(new LineSeries("Climatology", "#9e9e9e",
                latestPerValid.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ClimatologyPWet)).ToList()));

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

            s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"P(wet) — {PrettyStation(station)} — +{lead}h",
                XLabel = "Valid time (UTC)",
                YLabel = "Probability",
                Series = probSeries,
                Height = 220,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));

            s.Append(RenderPrecipDailySummaryTable(latestPerValid));
            s.Append(RenderPrecipHourlyConfidenceTable(latestPerValid, station, input.PrecipConformalTau));
        }

        // Per-NWP precip rate (mm/h) — hoisted out of the per-station loop
        // because PrecipGfs / PrecipEcmwf / etc. are point forecasts for
        // Bonehill, not gauge-station-specific. Pick any non-empty station
        // to source the rows from; values for the NWP columns are identical.
        var sourceRows = latestPerValidByStation.Values.FirstOrDefault(r => r.Count > 0);
        if (sourceRows is { Count: > 0 })
        {
            var nwpSeries = new List<LineSeries>();
            foreach (var nwp in nwpSpecs)
            {
                var pts = sourceRows
                    .Select(r => (Valid: r.ValidTimeUtc, Val: nwp.Get(r)))
                    .Where(t => t.Val.HasValue)
                    .Select(t => (X: t.Valid.ToOADate(), Y: t.Val!.Value))
                    .ToList();
                if (pts.Count > 0)
                    nwpSeries.Add(new LineSeries(nwp.Label, nwp.Color, pts));
            }
            if (nwpSeries.Count > 0)
            {
                s.Append("<h4>NWP precip rate (mm/h) — point forecast at Bonehill</h4>");
                s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
                {
                    Title = $"NWP precip rate — Bonehill — +{lead}h",
                    XLabel = "Valid time (UTC)",
                    YLabel = "Precip rate (mm/h)",
                    Series = nwpSeries,
                    Height = 220,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                    FormatY = v => v.ToString("0.0", Ci),
                }));
            }
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
        IReadOnlyList<PrecipForecastPoint> latestPerValid,
        string station,
        IReadOnlyDictionary<(string Station, string Version, int LeadHours), double> precipConformalTau)
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
            // forecasts that pre-date the calibrator. τ shown alongside the
            // chip when the calibrator is available so the reader can see
            // the model's overall ambiguity zone (low τ = narrow ambiguity
            // band, model commits on more rows; high τ = wider band).
            string conformalTd = "";
            if (anyConformal)
            {
                var chip = RenderConformalChip(r.ConformalSetTag);
                var tauPart = precipConformalTau.TryGetValue((station, r.Version, r.LeadHours), out var tau)
                    ? string.Create(Ci, $" <small>τ={(tau * 100):0}%</small>")
                    : "";
                conformalTd = "<td>" + chip + tauPart + "</td>";
            }
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
            ? "<th>Conformal <small>(90% set · τ)</small></th>"
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
