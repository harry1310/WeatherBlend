using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// How many days of past valid-time history every forecast-page chart
    /// shows before today. The four per-lead chart specs in this file
    /// (temp + temp NWP overlay + rain + rain NWP overlay) all read the
    /// same axis lower bound from <see cref="ForecastChartXMin"/>, so this
    /// is the single knob to turn when tuning the historical context shown
    /// alongside the forward forecast.
    /// </summary>
    private const int ForecastChartHistoryDays = 3;

    /// <summary>
    /// Lower bound (OADate) of the X axis on every forecast-page chart.
    /// Computed once per render from <see cref="ForecastChartHistoryDays"/>
    /// so the four chart sites can't drift on the window length.
    /// </summary>
    private static double ForecastChartXMin(SiteInputs input)
        => input.GeneratedAtUtc.AddDays(-ForecastChartHistoryDays).ToOADate();

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
        // Phase D: each forecast variable (Temperature / Rain / Dry window) is
        // its own top-level tab in the per-loc site-nav, so the per-page
        // forecasts sub-nav that used to live here was redundant and got
        // removed. Lead-axis sub-nav remains — it's intra-page.
        body.Append(RenderLeadSubNav("forecasts-temp", lead));

        body.Append(Ci, $"""
              <hgroup>
                <h2>Temperature +{lead}h</h2>
                <p>Our blend forecast on top (champion solid, challengers lighter); the raw NWP inputs below.</p>
              </hgroup>
            """);

        body.Append(RenderTempSection(input, lead));

        body.Append("</section>");
        return WrapPage(input, $"Temperature forecast +{lead}h", "temperature", body.ToString());
    }

    /// <summary>
    /// Rain forecast page for the given lead. Phase D — the page lives at
    /// <c>/{slug}/forecasts-rain-{lead}h.html</c> per location, so no
    /// locationName parameter or per-page loc-switcher (the global chrome
    /// handles location switching). The active location is whichever loc
    /// the caller built <paramref name="input"/> for.
    /// </summary>
    public static string RenderForecastsRain(SiteInputs input, int lead)
    {
        var active = input.RenderingFor;
        var body = new StringBuilder();
        body.Append("<section>");
        body.Append(RenderLeadSubNav("forecasts-rain", lead));

        body.Append(Ci, $"""
              <hgroup>
                <h2>Rain +{lead}h</h2>
                <p>Per-station P(wet ≥ 0.1 mm/h) — 3a solid, 3c lighter — plus per-NWP precip rate (one chart, point forecast).</p>
              </hgroup>
            """);

        body.Append(RenderPrecipSection(input, lead, active));

        body.Append("</section>");
        return WrapPage(input, $"Rain forecast +{lead}h", "rain", body.ToString());
    }

    // RenderForecastsSubNav removed in Phase D — each forecast variable is
    // its own top-level tab in the per-loc site-nav now, so the per-page
    // pill row was redundant.

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
        // Two stacked charts, mirroring the rain tab: our blend forecast on
        // top, the raw NWP inputs below. They were one combined overlay until
        // 2026-05-22 — ~8 NWP lines plus champion + challenger blend on a
        // single axis read as noise. Splitting lets the eye take the blend
        // forecast first, then drop to the inputs for "why". Both charts
        // share one (xMin, xMax) so they line up vertically on a fixed grid.
        //
        // Sources, unchanged by the split:
        //   - Blend: input.Predictions, partitioned by phase so each phase's
        //     BlendTemperature series is built from its own version's rows.
        //   - NWP: input.NwpTemperatures (raw hourly forecast tree, deduped
        //     to freshest cycle per (Model, ValidTime), already scoped to
        //     this page's location).
        // No now-1h floor: phase 2d emits only at {0,6,12,18}Z valid hours so
        // the lead-12 chart often has 0-1 future-of-now rows per anchor —
        // show history for context; the outer windowStart bounds it.
        var poolFuture = input.Predictions
            .Where(p => p.LeadHours == lead)
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        var s = new StringBuilder();

        if (poolFuture.Count == 0)
        {
            s.Append("<h3>Temperature — our blend</h3>");
            s.Append("<p><em>No +").Append(lead).Append("h temperature forecast available.</em></p>");
            return s.ToString();
        }

        // Shared X window. XMax = furthest blend valid_time at this lead, so
        // each lead-tab shows a window matching what the +Xh blend covers —
        // the raw NWP tree has the same forward horizon on every tab, so
        // without this clip the tabs looked identical. Floored at
        // GeneratedAtUtc so a stale prediction tree can't invert the axis.
        var blendMaxValid = poolFuture[^1].ValidTimeUtc;
        if (blendMaxValid < input.GeneratedAtUtc) blendMaxValid = input.GeneratedAtUtc;
        var xMin = ForecastChartXMin(input);
        var xMax = blendMaxValid.ToOADate();

        // ---- Chart 1 (top): our blend — champion + challengers ----
        // Champion (priority 0) solid in brand purple; challenger in lighter
        // purple, or exact-runtime magenta for 2d so it pops out of the
        // purple hue family. Skip any phase with zero rows in the window.
        var orderedActivePhases = ActivePhasePolicy.ByTarget["temperature"];
        var blendByPhase = input.Predictions
            .Where(p => p.LeadHours == lead)
            .Where(p => input.PhaseByVersion.TryGetValue(p.ModelVersion, out var ph)
                        && orderedActivePhases.Contains(ph, StringComparer.Ordinal))
            .GroupBy(p => input.PhaseByVersion[p.ModelVersion])
            .ToDictionary(g => g.Key, g => g.GroupBy(r => r.ValidTimeUtc)
                                            .Select(gv => gv.OrderByDescending(r => r.PredictionMadeAtUtc).First())
                                            .OrderBy(r => r.ValidTimeUtc)
                                            .ToList());
        var blendSeries = new List<LineSeries>();
        for (int i = 0; i < orderedActivePhases.Count; i++)
        {
            var phase = orderedActivePhases[i];
            if (!blendByPhase.TryGetValue(phase, out var phaseRows) || phaseRows.Count == 0) continue;
            var color = i == 0
                ? NwpPalette.Blend
                : (phase == "2d" || phase == "3d"
                    ? NwpPalette.BlendExactChallenger
                    : NwpPalette.BlendChallenger);
            var label = i == 0 ? $"Blend ({phase} champion)" : $"Blend ({phase} challenger)";
            var pts = phaseRows
                .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.BlendTemperature))
                .ToList();
            blendSeries.Add(new LineSeries(label, color, pts));
        }

        s.Append("<h3>Temperature — our blend</h3>");
        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Blend temperature — +{lead}h",
            XLabel = "Valid time (UTC)",
            YLabel = "Temperature (°C)",
            Series = blendSeries,
            Height = 280,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        }));

        // ---- Chart 2 (below): raw NWP inputs ----
        var nwpSeries = new List<LineSeries>();
        var tempPalette = NwpsForTemperature()
            .ToDictionary(np => np.Label, np => np.Color, StringComparer.Ordinal);
        foreach (var grp in input.NwpTemperatures
            .GroupBy(t => t.Model)
            .OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var label = Nwp.DisplayLabel(grp.Key);
            if (!tempPalette.TryGetValue(label, out var colour)) colour = "#999";
            var pts = grp
                .OrderBy(t => t.ValidTimeUtc)
                .Select(t => (X: t.ValidTimeUtc.ToOADate(), Y: t.Temperature2m))
                .ToList();
            if (pts.Count > 0)
                nwpSeries.Add(new LineSeries(label, colour, pts));
        }

        s.Append("<h3>NWP temperature forecasts</h3>");
        if (nwpSeries.Count == 0)
        {
            s.Append("<p><em>No NWP temperature forecast in the window.</em></p>");
        }
        else
        {
            s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"NWP temperature — +{lead}h",
                XLabel = "Valid time (UTC)",
                YLabel = "Temperature (°C)",
                Series = nwpSeries,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.#", Ci) + "°",
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }

        return s.ToString();
    }

    /// <summary>
    /// History days for the rain forecast tab's shared X axis. Wider than
    /// <see cref="ForecastChartHistoryDays"/> (3) because the rain page has
    /// stacked panels (top P(wet) + 4a + 5a + NWP rate) and the eye benefits
    /// from a wider context window when comparing them top-to-bottom.
    /// User-set 2026-05-10 alongside the per-tab axis-unification work.
    /// </summary>
    private const int RainChartHistoryDays = 7;

    private static string RenderPrecipSection(SiteInputs input, int lead)
        => RenderPrecipSection(input, lead, activeLocation: null);

    /// <summary>
    /// Phase → colour for the main P(wet) chart. Champion (3a) and every
    /// challenger get their own distinct hue; all SOLID lines (no dashes)
    /// so the eye reads colour, not stroke pattern. Hand-picked to be
    /// visually distinct under web rendering (no two adjacent hues on
    /// the colour wheel; light/dark contrast varies).
    ///
    /// Pre-2026-05-12 every challenger shared NwpPalette.BlendChallenger
    /// (light purple) with the only distinction being 2d/3d (magenta) —
    /// adding 3e + 4b put 4 lines on identical colour, unreadable.
    /// </summary>
    private static string PrecipPhaseColor(string phase, bool isChampion)
    {
        if (isChampion) return NwpPalette.Blend;   // brand purple
        return phase switch
        {
            "3c" => "#1976D2",                       // blue — rich-feature LightGBM
            "3d" => NwpPalette.BlendExactChallenger, // magenta — exact-runtime (unchanged)
            "3e" => "#2E7D32",                       // green — TorchSharp MLP
            "4b" => "#F57C00",                       // orange — 2-way mean (headline stack)
            _    => NwpPalette.BlendChallenger,      // fallback for future challengers
        };
    }

    /// <summary>
    /// Each precip prediction row carries the LocationName whose NWP fed its
    /// featureset (PrecipPredictCommand's <c>_activeLocation.Name</c>). Multi-
    /// location renders MUST drop rows where a station's prediction came from
    /// a different location's NWP — Membury stations should only render
    /// membury_devon-sourced rows, Bonehill stations only bonehill_rocks-
    /// sourced rows. Anything else is a category error: the model was trained
    /// against one site's NWP and the prediction is being scored against
    /// another. Pre-2026-05-12 the SQL filter quietly enforced primary-only
    /// at the cost of dropping legitimate Membury rows entirely.
    ///
    /// The map is built from <see cref="SiteInputs.Locations"/> (each
    /// LocationDescriptor lists its RainStationSlugs). A station present in
    /// no location returns <c>null</c> — callers treat that as "allow any"
    /// so legacy single-location callers / tests that don't populate
    /// Locations keep working unchanged.
    /// </summary>
    private static IReadOnlyDictionary<string, string> BuildStationHomeLocation(SiteInputs input)
    {
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var loc in input.Locations)
        {
            foreach (var slug in loc.RainStationSlugs)
                map[slug] = loc.Name;
        }
        return map;
    }

    /// <summary>
    /// True iff <paramref name="row"/> was sourced from the right location's
    /// NWP for its <c>Station</c>. Returns true (allow) when the station has
    /// no home-location entry — legacy single-location callers / fixtures
    /// without a populated Locations list keep their old behaviour.
    /// </summary>
    private static bool RowMatchesStationHomeLocation(
        PrecipForecastPoint row,
        IReadOnlyDictionary<string, string> stationHomeLocation)
    {
        if (!stationHomeLocation.TryGetValue(row.Station, out var home)) return true;
        return string.Equals(row.LocationName, home, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Per-station rain panels for one lead. When <paramref name="activeLocation"/>
    /// is non-null, the station list is intersected with that location's
    /// rainfall slugs so the Bonehill tab only shows Bonehill stations and
    /// the Membury tab only shows Membury stations. Null = legacy
    /// "all active stations" behaviour for the single-location callers
    /// (tests + pre-2026-05-11 drives).
    /// </summary>
    private static string RenderPrecipSection(SiteInputs input, int lead, LocationDescriptor? activeLocation)
    {
        var s = new StringBuilder();
        s.Append("<h3>Precipitation — P(wet ≥ 0.1 mm/h)</h3>");

        // Drop rows where the prediction's NWP source location doesn't match
        // the station's home location. See BuildStationHomeLocation /
        // RowMatchesStationHomeLocation for the rationale (mismatched pairings
        // are a category error). Doing this once at the top keeps the four
        // downstream queries identical and removes the risk of forgetting one.
        var stationHomeLocation = BuildStationHomeLocation(input);
        var precipForLocation = input.PrecipPredictions
            .Where(r => RowMatchesStationHomeLocation(r, stationHomeLocation))
            .ToList();

        // Filter to active stations so a demoted-from-config station whose
        // historical predictions are still on disk doesn't get a panel.
        var stations = precipForLocation
            .Select(p => p.Station).Distinct()
            .Where(s => input.ActiveStationSlugs.Count == 0 || input.ActiveStationSlugs.Contains(s))
            .OrderBy(st => st, StringComparer.Ordinal).ToList();

        // When an active location is set, restrict to that location's
        // configured rainfall slugs. The Membury tab reads from
        // activeLocation.RainStationSlugs so a Bonehill station can never
        // bleed into the Membury panel and vice versa.
        if (activeLocation is not null)
        {
            var locationSlugs = activeLocation.RainStationSlugs.ToHashSet(StringComparer.OrdinalIgnoreCase);
            stations = stations.Where(locationSlugs.Contains).ToList();
        }

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

        // Pre-compute the latest-per-(station, valid) lists up front so we
        // can derive the page-wide X axis window before rendering any chart.
        // Every chart on this lead-tab (top P(wet), per-station 4a, per-station
        // 5a, page-level NWP precip rate) shares the same (xMin, xMax) so the
        // reader can scan top-to-bottom on a fixed time grid. Window is:
        //   xMin = now − RainChartHistoryDays (currently 7d)
        //   xMax = furthest valid_time present in OUR top P(wet) chart's data
        // Per the 2026-05-10 user direction: the NWP precip rate often
        // extends further forward (raw forecast tree, hourly to +168h) but
        // the page's anchor is OUR blend forecast — subordinate panels +
        // the NWP rate chart get clipped to that horizon so the eye reads
        // a single consistent forward edge across the tab.
        foreach (var station in stations)
        {
            var latestPerValid = precipForLocation
                .Where(r => r.Station == station && r.LeadHours == lead)
                .GroupBy(r => r.ValidTimeUtc)
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .ToList();
            latestPerValidByStation[station] = latestPerValid;
        }
        var pageXMin = input.GeneratedAtUtc.AddDays(-RainChartHistoryDays).ToOADate();
        var pageXMaxValid = latestPerValidByStation.Values
            .Where(rows => rows.Count > 0)
            .Select(rows => rows[^1].ValidTimeUtc)
            .DefaultIfEmpty(input.GeneratedAtUtc)
            .Max();
        // Floor at GeneratedAtUtc so a stale prediction tree (e.g. predict
        // workflow failed for several days) doesn't produce an inverted
        // axis (xMax < xMin); chart degrades to "history only" rather than
        // rendering empty.
        if (pageXMaxValid < input.GeneratedAtUtc) pageXMaxValid = input.GeneratedAtUtc;
        var pageXMax = pageXMaxValid.ToOADate();

        foreach (var station in stations)
        {
            // Pool across blender versions before picking freshest per ValidTime.
            // Same rationale as the temp chart: champion-only collapses to ~one
            // point per anchor; the per-NWP columns are identical regardless of
            // blender version anyway, so pooling doesn't change the input story.
            // No now-1h floor: phases like 3d emit only at {0,6,12,18}Z
            // valid hours, so the chart at lead 12 typically has 0-1
            // future-of-now rows per anchor. Filtering to future-only
            // emptied the chart for entire half-days. Show historical
            // points in context — the outer windowStart (~30d back) still
            // bounds the data, and xMin/xMax in the chart spec keep the
            // visible window at the page-wide rain window (see pageXMin/
            // pageXMax computed above).
            var latestPerValid = latestPerValidByStation[station];

            s.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

            if (latestPerValid.Count == 0)
            {
                s.Append(RenderEmptyChart(
                    $"P(wet) — {PrettyStation(station)} — +{lead}h",
                    "No forecast at this lead in the forward window."));
                continue;
            }

            // Top chart: our blended model P(wet) lines + climatology only.
            // Per-NWP PoP overlay moved to a page-level chart below the
            // station loop (2026-05-12 — was overcrowded with 5 blender
            // lines + 4 NWP PoP lines + climatology on one axis).
            //
            // Champion = 3a (solid brand purple). Each challenger has its
            // OWN distinct colour so 5+ phases stay readable. Dashed for
            // every challenger; solid only for the champion. Climatology
            // is grey.
            var probSeries = new List<LineSeries>();
            var orderedPrecipPhases = ActivePhasePolicy.ByTarget["precipitation"];
            var precipByPhase = precipForLocation
                .Where(r => r.Station == station
                            && r.LeadHours == lead)
                .Where(r => input.PhaseByVersion.TryGetValue(r.Version, out var ph)
                            && orderedPrecipPhases.Contains(ph, StringComparer.Ordinal))
                .GroupBy(r => input.PhaseByVersion[r.Version])
                .ToDictionary(g => g.Key, g => g.GroupBy(r => r.ValidTimeUtc)
                                                 .Select(gv => gv.OrderByDescending(r => r.PredictedAtUtc).First())
                                                 .OrderBy(r => r.ValidTimeUtc)
                                                 .ToList());
            for (int i = 0; i < orderedPrecipPhases.Count; i++)
            {
                var phase = orderedPrecipPhases[i];
                // 4a renders in its own standalone panel below
                // (RenderPhase4aPanel) — its dashed Q05/Q95 band would
                // crowd the main chart, and its own panel can carry
                // more posterior detail without competing for space.
                if (phase == "4a") continue;
                if (!precipByPhase.TryGetValue(phase, out var phaseRows) || phaseRows.Count == 0) continue;
                var color = PrecipPhaseColor(phase, isChampion: i == 0);
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

            s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"P(wet) — {PrettyStation(station)} — +{lead}h",
                XLabel = "Valid time (UTC)",
                YLabel = "Probability",
                Series = probSeries,
                Height = 220,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = pageXMin,
                XMax = pageXMax,
            }));

            // Clip the daily-summary + hourly-confidence tables to the
            // same time window as the chart on the tab. Without this they
            // listed every row in latestPerValid (which extends back to
            // windowStart ~30d for historical context) — flooding the
            // tables with weeks of past rows the chart doesn't show.
            // Convert page xMin/xMax (OADate) back to DateTime once, then
            // reuse for both tables.
            var visibleStart = DateTime.FromOADate(pageXMin);
            var visibleEnd = DateTime.FromOADate(pageXMax);
            var visiblePerValid = latestPerValid
                .Where(r => r.ValidTimeUtc >= visibleStart && r.ValidTimeUtc <= visibleEnd)
                .ToList();

            s.Append(RenderPrecipDailySummaryTable(visiblePerValid));
            s.Append(RenderPrecipHourlyConfidenceTable(visiblePerValid, station, input.PrecipConformalTau));
            s.Append(RenderPhase4aPanel(input, station, lead, pageXMin, pageXMax));
            s.Append(RenderBayesianCiPanel(input, station, lead, pageXMin, pageXMax));
        }

        // Page-level NWP precipitation_probability (PoP) chart — moved
        // out of the per-station main P(wet) chart on 2026-05-12. PoP
        // values are identical across rainfall stations (they're a point
        // forecast at the location's NWP grid cell) so it makes sense
        // to render once per page rather than per station. Sits above
        // the mm/h chart so the eye reads "is rain likely?" then "how
        // hard?" top-to-bottom. Only ~4 of 8 NWPs publish PoP — others
        // drop out of the legend silently.
        // input.NwpPrecipProbabilities / NwpPrecipRates are already scoped to
        // this page's location by RenderSiteCommand (single-location SiteInputs
        // invariant) — iterate directly.
        var activeLocDisplay = activeLocation?.DisplayName
            ?? PrimaryLocation(input.Locations)?.DisplayName
            ?? "primary";

        if (input.NwpPrecipProbabilities.Count > 0)
        {
            var popSeries = new List<LineSeries>();
            var popPalette = nwpSpecs
                .ToDictionary(np => np.Label, np => np.Color, StringComparer.Ordinal);
            foreach (var grp in input.NwpPrecipProbabilities
                .GroupBy(p => p.Model)
                .OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                var label = Nwp.DisplayLabel(grp.Key);
                if (!popPalette.TryGetValue(label, out var colour)) colour = "#999";
                var pts = grp
                    .OrderBy(p => p.ValidTimeUtc)
                    .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.ProbabilityPercent / 100.0))
                    .ToList();
                // Suffix " PoP" so the NWP lines on the rain chart don't
                // collide with the blender's "P(wet)" series label or the
                // (label-only) NWP precip-rate chart further down — eyeballing
                // the chart legend should immediately tell the reader this is
                // an NWP probability of precip, not the blender's output or
                // a mm/h rate. Lost in a 2026-05-12 refactor when the chart
                // moved out of the per-station loop; restored 2026-05-15.
                if (pts.Count > 0)
                    popSeries.Add(new LineSeries($"{label} PoP", colour, pts));
            }
            if (popSeries.Count > 0)
            {
                s.Append(Ci, $"<h4>NWP precipitation probability — point forecast at {Escape(activeLocDisplay)}</h4>");
                s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
                {
                    Title = $"NWP PoP — {activeLocDisplay} — +{lead}h",
                    XLabel = "Valid time (UTC)",
                    YLabel = "Probability",
                    Series = popSeries,
                    Height = 220,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                    FormatY = v => v.ToString("0.00", Ci),
                    TodayLineX = input.GeneratedAtUtc.ToOADate(),
                    XMin = pageXMin,
                    XMax = pageXMax,
                }));
            }
        }

        // Per-NWP precip rate (mm/h) — point forecast at Bonehill, hoisted
        // out of the per-station loop because PrecipGfs/PrecipEcmwf/etc are
        // identical across rainfall stations. Source: input.NwpPrecipRates
        // (raw forecast tree, hourly), switched 2026-05-07 from the
        // prediction-row source so the lead-12 page populates at hourly
        // density (3d only emits at {0,6,12,18}Z which made the prior
        // chart absent at lead 12 entirely).
        if (input.NwpPrecipRates.Count > 0)
        {
            var nwpSeries = new List<LineSeries>();
            var ratePalette = nwpSpecs
                .ToDictionary(np => np.Label, np => np.Color, StringComparer.Ordinal);
            foreach (var grp in input.NwpPrecipRates
                .GroupBy(t => t.Model)
                .OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                var label = Nwp.DisplayLabel(grp.Key);
                if (!ratePalette.TryGetValue(label, out var colour)) colour = "#999";
                var pts = grp
                    .OrderBy(t => t.ValidTimeUtc)
                    .Select(t => (X: t.ValidTimeUtc.ToOADate(), Y: t.PrecipMmPerHour))
                    .ToList();
                if (pts.Count > 0)
                    nwpSeries.Add(new LineSeries(label, colour, pts));
            }
            if (nwpSeries.Count > 0)
            {
                s.Append(Ci, $"<h4>NWP precip rate (mm/h) — point forecast at {Escape(activeLocDisplay)}</h4>");
                s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
                {
                    Title = $"NWP precip rate — {activeLocDisplay} — +{lead}h",
                    XLabel = "Valid time (UTC)",
                    YLabel = "Precip rate (mm/h)",
                    Series = nwpSeries,
                    Height = 220,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                    FormatY = v => v.ToString("0.0", Ci),
                    TodayLineX = input.GeneratedAtUtc.ToOADate(),
                    XMin = pageXMin,
                    XMax = pageXMax,
                }));
            }
        }

        return s.ToString();
    }

    /// <summary>
    /// Bayesian credible-interval panel — Phase 2e. Renders a small line
    /// chart with three series: posterior median P(wet) (solid, dark
    /// blue) plus q05 and q95 (dashed, light blue) bracketing the median
    /// (90% band, matching 4a's panel — was 80% / q10–q90 before 2026-05-10).
    /// Above the chart, a one-line summary giving median P(wet) average
    /// and CI80 width average across the rendered hours, plus a tier
    /// label (high / medium / low confidence) so the eye can read it
    /// without computing widths in their head.
    ///
    /// Sits below the existing per-station P(wet) chart on the precip
    /// forecast page. The Bayesian model is from the WeatherProbabilistic
    /// sibling repo (Phase 5a — hierarchical Bayesian logistic regression
    /// with lead-as-feature, partial-pooling across stations, 5 NWP precip
    /// features) — distinct model from 3a's LightGBM, so the median line
    /// is a real independent second opinion, not a re-rendering of the
    /// headline. What we use it for is the WIDTH (CI80) per row: the
    /// 2026-04 Phase 4 bake-off found narrow-CI rows have ~5x lower Brier
    /// than wide-CI rows, so CI80 transfers as a forecast-skill flag
    /// downstream.
    ///
    /// Silent skip if (a) no rows for this (station, lead) pair (e.g.
    /// before the first predict-5a.yml fire, or for stations the 5a
    /// model hasn't been retrained for yet), or (b) the lead isn't in
    /// the trained set.
    /// </summary>
    /// <summary>
    /// Standalone 4a panel — pulled out of the main P(wet) chart 2026-05-09
    /// because adding 4a's median + dashed Q05/Q95 on top of 3a/3c/3d + 4
    /// per-NWP PoP overlays + climatology made the chart visually
    /// unreadable. Mirrors <see cref="RenderBayesianCiPanel"/>'s shape.
    /// Silent skip if no 4a rows for this (station, lead) pair (e.g. before
    /// the first predict-4a.yml fire).
    /// </summary>
    /// <summary>
    /// Leads where the 4a (BART) panel should render. Mirrors LEADS in
    /// WeatherProbabilistic/scripts/{train_4a,predict_4a}.py — 12 dropped
    /// 2026-05-10 (offset_day archive coverage gap; TestRows=0 in training
    /// metadata, predictions were extrapolation along the lead axis).
    /// Historical lead-12 rows still exist on R2 from before the predict
    /// scope was tightened, so the renderer needs its own gate; otherwise
    /// the +12h tab keeps showing the 4a panel from stale data.
    /// </summary>
    private static readonly HashSet<int> Phase4aDisplayLeads = new() { 24, 48, 72, 96, 120 };

    private static string RenderPhase4aPanel(
        SiteInputs input, string stationSlug, int lead, double xMin, double xMax)
    {
        if (!Phase4aDisplayLeads.Contains(lead)) return "";

        // Drop rows whose NWP source location doesn't match this station's
        // home location — same rule as the main P(wet) chart above. Predictions
        // for Membury stations made from Bonehill NWP are a category error
        // and would skew the panel even though they happen to be present in
        // the predictions tree.
        var stationHomeLocation = BuildStationHomeLocation(input);
        var rows = input.PrecipPredictions
            .Where(r => RowMatchesStationHomeLocation(r, stationHomeLocation))
            .Where(r => string.Equals(r.Station, stationSlug, StringComparison.OrdinalIgnoreCase)
                        && r.LeadHours == lead
                        && input.PhaseByVersion.TryGetValue(r.Version, out var ph)
                        && string.Equals(ph, "4a", StringComparison.Ordinal))
            .GroupBy(r => r.ValidTimeUtc)
            .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
            .OrderBy(r => r.ValidTimeUtc)
            .ToList();
        if (rows.Count == 0) return "";

        // Visual consistency with 5a: CI brackets render in
        // PrecipPhases.Phase4aBand (lighter amber), main line in
        // PrecipPhases.Phase4a.Color (full amber). Series order
        // q95 → main → q05 mirrors 5a so the chart tooltip reads
        // top-to-bottom 95% / centre / 5%. NB the "centre line" here is
        // ProbWet (posterior MEAN), not Q50 (median) — 4a's parquet
        // exposes both but the panel has always plotted the mean as the
        // headline. 5a plots Q50 by convention; we don't unify the
        // headline source because they're meaningfully different
        // summaries and the existing labels reflect that.
        var color = PrecipPhases.Phase4a.Color;
        var bandColor = PrecipPhases.Phase4aBand;
        var meanPts = rows.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet)).ToList();
        var series = new List<LineSeries>();
        var hasBand = rows.Any(r => r.ProbWetQ05.HasValue && r.ProbWetQ95.HasValue);
        if (hasBand)
        {
            var q95Pts = rows.Where(r => r.ProbWetQ95.HasValue)
                             .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWetQ95!.Value))
                             .ToList();
            series.Add(new LineSeries("q95 (upper)", bandColor, q95Pts, Dashed: true));
        }
        series.Add(new LineSeries("P(wet) (4a)", color, meanPts));
        if (hasBand)
        {
            var q05Pts = rows.Where(r => r.ProbWetQ05.HasValue)
                             .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWetQ05!.Value))
                             .ToList();
            series.Add(new LineSeries("q05 (lower)", bandColor, q05Pts, Dashed: true));
        }

        var s = new StringBuilder();
        s.Append("<h4>Phase 4a — BART posterior P(wet) + 90% band</h4>");
        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Phase 4a P(wet) — {PrettyStation(stationSlug)} — +{lead}h",
            XLabel = "Valid time (UTC)",
            YLabel = "Probability",
            Series = series,
            Height = 180,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.00", Ci),
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        }));
        return s.ToString();
    }

    private static string RenderBayesianCiPanel(
        SiteInputs input, string stationSlug, int lead,
        double xMin, double xMax)
    {
        if (input.BayesianCi.Count == 0) return "";

        // Match Phase 5a CI rows to the precip station via slug — both
        // sides share the same key (BayesianCiPoint.StationSlug is parsed
        // off the predictions parquet path partition; stationSlug here is
        // the page's station). Stations the 5a model hasn't been retrained
        // for yet just produce 0 matches and skip silently.
        //
        // No now-1h floor: showing past hours alongside future gives eye-
        // context ("model said X for last day, says Y now") and the
        // renderer's outer windowStart still bounds the dataset.
        //
        // Filter on a ±12h band around the page's nominal lead rather than
        // strict equality. Phase 5a (lead-as-feature) emits one row per
        // (cycle, lead) pair with the ACTUAL run-to-valid offset; strict
        // `LeadHours == 24` would only catch cycles at HH ∈ {0,6,12,18}
        // landing on 00/06/12/18Z valid_times — same cycle-grid bottleneck
        // we hit before. The band keeps the per-lead-page distinction
        // (lead-24 page = "predictions made roughly 24h ahead") while
        // letting hourly cycle output fill the chart densely. Bands tile
        // cleanly: [12,36], [36,60], [60,84] for the three pages.
        var rows = input.BayesianCi
            .Where(p => string.Equals(p.StationSlug, stationSlug, StringComparison.OrdinalIgnoreCase)
                        && p.LeadHours >= lead - 12 && p.LeadHours < lead + 12)
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g.OrderByDescending(p => p.PredictedAtUtc)
                          .ThenByDescending(p => p.LeadHours)
                          .First())
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();
        if (rows.Count == 0) return "";

        var medianPts = rows.Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.PWetQ50)).ToList();
        var q05Pts = rows.Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.PWetQ05)).ToList();
        var q95Pts = rows.Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.PWetQ95)).ToList();

        var s = new StringBuilder();
        s.Append("<h4>Bayesian credible interval — independent confidence signal</h4>");
        // No summary text — the chart is self-explanatory: tight band =
        // confident, wide band = uncertain. An averaged tier was tried
        // 2026-05-06 and reverted on user feedback ("why a single value?")
        // because it smeared per-hour variability the chart already shows.
        // Band aligned to 4a's 90% (q05/q95) on 2026-05-10 — the prior
        // 80% was an inheritance from the bayesian_ci era; matching 4a
        // makes the two confidence panels visually comparable.
        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Bayesian P(wet) median + 90% CI — {PrettyStation(stationSlug)} — +{lead}h",
            XLabel = "Valid time (UTC)",
            YLabel = "Probability",
            // Median solid; q05/q95 dashed so the legend reads them as
            // "band edges" not as independent forecast lines, even though
            // they share the BayesianBand swatch (semantically symmetric —
            // both are the same posterior's quantile bracket).
            Series = new List<LineSeries>
            {
                new("q95 (upper)", NwpPalette.BayesianBand, q95Pts, Dashed: true),
                new("Median",      NwpPalette.BayesianMedian, medianPts),
                new("q05 (lower)", NwpPalette.BayesianBand, q05Pts, Dashed: true),
            },
            Height = 180,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.00", Ci),
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        }));
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
        // Same collapsible wrapper as the hourly P(wet) + agreement table
        // below — daily totals were eating chart's worth of vertical space
        // by default. Closed by default so the chart stays the focal point;
        // tap the summary to drill in. Class re-used so styling stays
        // consistent across both tables.
        return $"""
            <details class="hourly-detail">
              <summary>Daily P(wet) summary</summary>
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
            {rows}      </tbody>
                </table>
              </figure>
            </details>
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
            // P(wet) cell colour-graded green-to-red by value — same
            // PrecipProbColor scale used on the home page summary line and
            // every other P(wet) chip on the site, so a 70% wet hour reads
            // the same colour wherever it appears. Confidence is already
            // surfaced in its own column to the right (NWPs wet + the
            // confidence chip), so the cell can lose the dual-encoding
            // (purple-by-confidence) and just colour-grade by P(wet).
            var pwetStyle = $"color: {PrecipProbColor(r.ProbWet)}; font-weight: 600";
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
