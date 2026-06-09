using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Temperature skill page: eyeball plot of each active blender vs ERA5 truth,
    /// followed by rolling MAE per lead. No station axis — temperature is a
    /// single-location quantity, so this page is the same whatever station the
    /// reader is interested in.
    /// </summary>
    public static string RenderTempSkill(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("<section>");
        // Sub-nav first so the tab bar's Y position is fixed across the
        // three Skill pages — flicking between sub-tabs doesn't jolt the
        // page as content height varies.
        content.Append(RenderSkillSubNav("temp", input.RenderingFor));
        content.Append("""
              <hgroup>
                <h2>Skill — temperature</h2>
                <p>Blenders vs ERA5 truth + EGTE METAR and Taw Green obs (both lower than the 393 m tor → expect a warm bias).</p>
              </hgroup>
            """);

        content.Append("<h3>Vs truth</h3>");
        content.Append(RenderTempVsTruthBlock(input));

        content.Append("<h3>Phase comparison — +24h lead</h3>");
        content.Append(RenderTempPhaseComparisonBlock(input));

        content.Append("<hr/><h3>Rolling MAE</h3>");
        content.Append(RenderRollingMaeBlock(input));

        content.Append("</section>");
        return WrapPage(input, "Skill — temperature", "skill", content.ToString());
    }

    /// <summary>
    /// Mirror of the rain skill page's +24h phase-comparison chart, for
    /// temperature blenders. One series per active temp phase
    /// (<see cref="ActivePhasePolicy"/> ByTarget["temperature"]), each in
    /// its <see cref="TempPhases"/> colour so a phase reads the same hue
    /// on this chart and on the rolling-MAE panel below. No truth line —
    /// matches the rain version's shape; the per-phase eyeball charts
    /// above show vs-truth context. Rendered only when at least two
    /// phases have data at +24h, otherwise the chart says nothing the
    /// vs-truth panels don't.
    /// </summary>
    private static string RenderTempPhaseComparisonBlock(SiteInputs input)
    {
        const int leadHours = 24;
        var (xMin, xMax) = TempSectionRange(input);

        var series = new List<LineSeries>();
        foreach (var phaseKey in ActivePhasePolicy.ByTarget["temperature"])
        {
            var versions = input.PhaseByVersion
                .Where(kv => kv.Value == phaseKey)
                .Select(kv => kv.Key)
                .ToHashSet(StringComparer.Ordinal);
            if (versions.Count == 0) continue;

            var pts = input.Predictions
                .Where(p => p.LeadHours == leadHours
                            && versions.Contains(p.ModelVersion)
                            && p.ValidTimeUtc >= input.WindowStartUtc)
                .GroupBy(p => p.ValidTimeUtc)
                .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
                .OrderBy(p => p.ValidTimeUtc)
                .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.BlendTemperature))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries($"Phase {phaseKey}", TempPhases.ColorFor(phaseKey), pts));
        }

        if (series.Count < 2)
            return "<p class=\"skill-line\"><em>Phase comparison needs ≥ 2 active phases with predictions at +24h — currently fewer.</em></p>";

        return LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Temperature blender phase comparison — +{leadHours}h",
            XLabel = "Time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 280,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        });
    }

    /// <summary>
    /// Per-target sub-nav for the Skill pages. Same shape as
    /// <see cref="RenderModelsSubNav"/>: three pill links sitting under the
    /// page heading. The active variant is plain (not a link) so the eye lands
    /// on it.
    /// </summary>
    /// <summary>
    /// Per-target sub-nav for the Skill pages. Phase D — filters entries
    /// by <see cref="LocationDescriptor.Tabs"/>: a location without
    /// <c>dry_window</c> in its tabs gets a 2-button sub-nav (Temperature /
    /// Rain), and similarly for any other missing target. When
    /// <paramref name="loc"/> is null (test fixtures / pre-Phase-D
    /// callers) all three entries render to preserve legacy behaviour.
    /// </summary>
    private static string RenderSkillSubNav(string activeSlug, LocationDescriptor? loc = null)
    {
        var entries = new (string Slug, string File, string Label, string TabId)[]
        {
            ("temp",       "skill-temperature.html", "Temperature", "temperature"),
            ("rain",       "skill-rainfall.html",    "Rain",        "rain"),
            ("wind",       "skill-wind.html",        "Wind",        "wind"),
            ("dry-window", "skill-dry-window.html",  "Dry window",  "dry_window"),
        };
        var s = new StringBuilder();
        s.Append("<nav class=\"lead-nav\"><ul>");
        foreach (var (slug, file, label, tabId) in entries)
        {
            if (loc is not null && !loc.HasTab(tabId)) continue;
            var cls = slug == activeSlug ? " class=\"active\"" : "";
            s.Append(Ci, $"<li><a href=\"{file}\"{cls}>{Escape(label)}</a></li>");
        }
        s.Append("</ul></nav>");
        return s.ToString();
    }

    /// <summary>
    /// Skill — Wind. Rolling MAE for blended wind speed (Dunkeswell SYNOP truth)
    /// + circular MAE for wind direction (Dunkeswell SYNOP truth) once those
    /// verify trees exist. Today's MVP renders the structure with empty-state
    /// placeholders — verify data for the new wind_speed_lgb / wind_blend /
    /// wind_mvn phases accumulates from the first Sunday retrain after
    /// 2026-05-31. Layout decisions in
    /// <c>project_wind_tab_design_2026-05-28</c>.
    /// </summary>
    public static string RenderWindSkill(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("<section>");
        content.Append(RenderSkillSubNav("wind", input.RenderingFor));
        content.Append("""
              <hgroup>
                <h2>Skill — wind</h2>
                <p>Rolling MAE for blended wind speed (scored vs ERA5 by element-verify)
                   and circular MAE for direction (wind_mvn, vs Dunkeswell SYNOP).</p>
              </hgroup>
            """);

        content.Append("<h3>Speed MAE (rolling, ERA5 truth)</h3>");
        content.Append(RenderWindSpeedSkillBlock(input));

        content.Append("<hr/><h3>Direction MAE (circular, rolling, Dunkeswell truth)</h3>");
        content.Append(RenderWindDirectionSkillBlock(input));

        content.Append("</section>");
        return WrapPage(input, "Skill — wind", "skill", content.ToString());
    }

    /// <summary>Rolling MAE chart(s) for wind speed — per lead, one line per
    /// phase (wind champion + wind_speed_lgb + wind_blend + wind_gust_lgb gust).
    /// Driven from <see cref="SiteInputs.VerifyHistory"/>: each element-verify run
    /// writes an <c>element_wind</c> / <c>element_wind_gust</c> file (AsOfUtc +
    /// per-(phase, lead) BlendMetric in m/s), so plotting BlendMetric over AsOfUtc
    /// is the rolling-MAE-over-time series. Renders the empty-state per lead until
    /// element-verify rows accumulate. MAE shown in mph to match the forecast page.</summary>
    private static string RenderWindSpeedSkillBlock(SiteInputs input)
    {
        // Flatten every element_wind / element_wind_gust verify row to a point.
        var pts = input.VerifyHistory
            .Where(f => f.Target is "element_wind" or "element_wind_gust")
            .SelectMany(f => f.Rows.Select(r => (
                AsOf: f.AsOfUtc,
                Phase: string.IsNullOrEmpty(r.Phase) ? r.ModelVersion : r.Phase,
                Lead: r.LeadHours,
                MaeMph: r.BlendMetric * MsToMph)))
            .ToList();

        if (pts.Count == 0)
            return RenderEmptyChart(
                "Wind speed MAE (mph) — rolling",
                "No element-verify rows yet. Lines (wind champion / wind_speed_lgb / wind_blend / "
                + "wind_gust_lgb gust) populate per lead once element-verify scores these phases against ERA5.");

        // Phase → (label, colour). Unknown phases fall through to a neutral grey.
        static (string Label, string Color) PhaseStyle(string phase) => phase switch
        {
            "wind"           => ("wind (champion)", "#6a1b9a"),
            "wind_speed_lgb" => ("wind_speed_lgb",  "#1976d2"),
            "wind_blend"     => ("wind_blend",      "#2e7d32"),
            "wind_gust_lgb"  => ("wind_gust_lgb gust", "#ff9800"),
            _                 => (phase,            "#9e9e9e"),
        };

        double xMax = pts.Max(p => p.AsOf).ToOADate();
        double xMin = xMax - 30.0;

        var content = new StringBuilder();
        foreach (var lead in new[] { 24, 48, 72 })
        {
            var leadPts = pts.Where(p => p.Lead == lead).ToList();
            content.Append(Ci, $"<h4>Lead +{lead}h</h4>");
            if (leadPts.Count == 0)
            {
                content.Append(RenderEmptyChart($"Wind speed MAE — lead {lead}h", "No scored rows at this lead yet."));
                continue;
            }
            var series = new List<LineSeries>();
            foreach (var phase in leadPts.Select(p => p.Phase).Distinct().OrderBy(p => p, StringComparer.Ordinal))
            {
                var (label, color) = PhaseStyle(phase);
                var line = leadPts.Where(p => p.Phase == phase)
                    .OrderBy(p => p.AsOf)
                    .Select(p => (X: p.AsOf.ToOADate(), Y: p.MaeMph))
                    .ToList();
                if (line.Count > 0) series.Add(new LineSeries(label, color, line));
            }
            content.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"Wind speed MAE — lead +{lead}h (mph)",
                XLabel = "Verify date (UTC)",
                YLabel = "MAE (mph)",
                Series = series,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                FormatY = v => v.ToString("0.00", Ci),
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }
        return content.ToString();
    }

    /// <summary>Rolling circular MAE chart for wind direction —
    /// wind_mvn (champion-eventual) vs NWP-mean baseline.</summary>
    private static string RenderWindDirectionSkillBlock(SiteInputs input)
    {
        return RenderEmptyChart(
            "Wind direction MAE (°) — rolling 30-day, circular",
            "Direction verification arrives once wind_mvn predictions land on R2 " +
            "(first Sunday retrain after 2026-05-31). Circular MAE is min(|err|, 360 - |err|) " +
            "averaged over the rolling window.");
    }

    /// <summary>
    /// Skill — Dry window. Per-station predicted-vs-observed tables for the
    /// dry-window blenders (3b champion + active challengers). Was previously the
    /// last section of the Rain skill page; lifted to its own page in the
    /// 2026-05-04 site rework so each variable's skill content has a focused
    /// home. Uses the same per-station sub-nav (station list union of precip
    /// + dry-window stations, filtered to active) as Rain skill.
    /// </summary>
    public static string RenderDryWindowSkill(SiteInputs input, string? stationSlug = null)
    {
        // Was GetRainSkillStations (union of precip + dry-window stations) —
        // that surfaced a Membury tab even though Membury has no dry-window
        // predictions. Use the dry-window-only set instead.
        var stations = GetDryWindowSkillStations(input);
        var currentStation = ResolveStationFromSlug(stations, stationSlug);

        var content = new StringBuilder();
        content.Append("<section>");
        content.Append(RenderSkillSubNav("dry-window", input.RenderingFor));
        content.Append("""
              <hgroup>
                <h2>Skill — dry window</h2>
                <p>Predicted vs observed dry-block verdict per (day × window). Observed blank for dates past the last full gauge day.</p>
              </hgroup>
            """);

        if (currentStation is not null)
            content.Append(RenderStationSubNav("skill-dry-window", stations, currentStation));

        content.Append(RenderDryWindowVsTruthTable(input, currentStation));

        content.Append("</section>");
        return WrapPage(input, "Skill — dry window", "skill", content.ToString());
    }

    /// <summary>
    /// Rainfall skill page: per-station eyeball plots of P(wet) vs observed wet-hour,
    /// followed by the dry-window prediction vs observed-verdict table. Station is
    /// the primary axis here — stations live on separate gauges with wildly different
    /// climatologies, so each gets its own HTML variant.
    ///
    /// <paramref name="stationSlug"/> picks which station to render. <c>null</c>
    /// means the canonical first station — this variant ships as
    /// <c>skill-rainfall.html</c>; the others ship as
    /// <c>skill-rainfall-{slug}.html</c>.
    /// </summary>
    /// <summary>
    /// Rain skill page. Phase D — the page lives at
    /// <c>/{slug}/skill-rainfall.html</c> per location; no locationName
    /// parameter, no per-page loc-switcher (global chrome handles location
    /// switching). <paramref name="stationSlug"/> picks which station to
    /// render (null = first).
    /// </summary>
    public static string RenderRainSkill(SiteInputs input, string? stationSlug = null)
        => RenderRainSkill(input, stationSlug, locationName: null);

    /// <summary>
    /// Two-arity overload retained briefly so the auto-retrain workflows /
    /// in-flight callers don't break on the parameter change. Phase D
    /// renders ignore <paramref name="locationName"/> — pick the location
    /// by building <paramref name="input"/> with the right RenderingFor.
    /// </summary>
    public static string RenderRainSkill(SiteInputs input, string? stationSlug, string? locationName)
    {
        _ = locationName; // intentionally unused in Phase D
        var active = input.RenderingFor;
        var stations = GetRainSkillStations(input, active);
        var currentStation = ResolveStationFromSlug(stations, stationSlug);

        var content = new StringBuilder();
        content.Append("<section>");
        content.Append(RenderSkillSubNav("rain", active));

        var intro = "P(wet) vs observed wet hours, then rolling Brier per phase.";
        content.Append(Ci, $"""
              <hgroup>
                <h2>Skill — rain</h2>
                <p>{Escape(intro)}</p>
              </hgroup>
            """);

        if (currentStation is not null)
            content.Append(RenderStationSubNav("skill-rainfall", stations, currentStation));

        content.Append("<h3>P(wet) vs observed wet-hour</h3>");
        content.Append(RenderPrecipVsTruthBlock(input, currentStation));
        content.Append("<hr/><h3>Rolling Brier (P(wet))</h3>");
        content.Append(RenderRollingBrierBlock(input, currentStation));

        // Phase 3f distributional skill — silently empty for stations that
        // don't have rainfall_amount verify history yet (everywhere except
        // Membury today). Renders the CRPS / coverage / PIT / exceedance
        // Brier widgets when rows exist.
        var rainfallAmountHasRows = input.VerifyHistory.Any(f =>
            string.Equals(f.Target, "rainfall_amount", StringComparison.Ordinal) &&
            f.Rows.Any(r => string.Equals(r.Station, currentStation, StringComparison.Ordinal)));
        if (rainfallAmountHasRows)
        {
            content.Append("<hr/><h3>Rainfall amount (3f) — distributional skill</h3>");
            content.Append(RenderRainfallAmountSkillBlock(input, currentStation));
        }

        content.Append("</section>");
        return WrapPage(input, "Skill — rain", "skill", content.ToString());
    }

    /// <summary>
    /// Station set used by the rainfall-skill sub-nav. Union of precip and dry-window
    /// stations so the sub-nav shows every station the reader might care about, even
    /// if one of the two sections has no data for it.
    /// </summary>
    internal static IReadOnlyList<string> GetRainSkillStations(SiteInputs input)
        => GetRainSkillStations(input, activeLocation: null);

    /// <summary>
    /// Station set for the DRY-WINDOW skill sub-nav. Unlike
    /// <see cref="GetRainSkillStations"/> this is NOT a union with precip
    /// stations — Membury (precip-only) would otherwise get a dry-window
    /// tab that renders an empty table. Only stations with actual
    /// dry-window predictions appear, intersected with ActiveStationSlugs
    /// so a demoted station's stale predictions don't surface a tab.
    /// </summary>
    internal static IReadOnlyList<string> GetDryWindowSkillStations(SiteInputs input)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (var d in input.DryWindowPredictions) set.Add(d.Station);
        if (input.ActiveStationSlugs.Count > 0)
            set.IntersectWith(input.ActiveStationSlugs);
        return set.OrderBy(s => s, StringComparer.Ordinal).ToList();
    }

    /// <summary>
    /// <inheritdoc cref="GetRainSkillStations(SiteInputs)"/>
    /// When <paramref name="activeLocation"/> is non-null, the result is
    /// further intersected with that location's <c>RainStationSlugs</c> so
    /// the Membury skill page only lists Membury stations and vice versa.
    /// </summary>
    internal static IReadOnlyList<string> GetRainSkillStations(SiteInputs input, LocationDescriptor? activeLocation)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in input.PrecipPredictions) set.Add(p.Station);
        foreach (var d in input.DryWindowPredictions) set.Add(d.Station);
        // Filter to currently-active stations from config (post-swap behaviour).
        // Empty ActiveStationSlugs preserves old "render whatever's on disk"
        // behaviour for legacy callers / tests that don't populate it.
        if (input.ActiveStationSlugs.Count > 0)
            set.IntersectWith(input.ActiveStationSlugs);
        if (activeLocation is not null)
            set.IntersectWith(activeLocation.RainStationSlugs);
        return set.OrderBy(s => s, StringComparer.Ordinal).ToList();
    }

    /// <summary>
    /// Resolve a URL slug to a station id. Unknown or null slug picks the first
    /// station (canonical landing). Empty station list returns <c>null</c> — callers
    /// skip the per-station sub-nav entirely in that case.
    /// </summary>
    internal static string? ResolveStationFromSlug(IReadOnlyList<string> stations, string? slug)
    {
        if (stations.Count == 0) return null;
        if (string.IsNullOrEmpty(slug)) return stations[0];
        foreach (var s in stations)
            if (string.Equals(StationSlug(s), slug, StringComparison.OrdinalIgnoreCase))
                return s;
        return stations[0];
    }

    // Single source of truth for leads + colours on the per-phase eyeball
    // charts (temp vs truth, P(wet) vs observed). Both charts iterate this
    // list; the per-lead `if (pts.Count > 0)` guard inside each loop drops
    // empty series so legacy phases without lead-12 rows still render
    // cleanly. When a new exact-runtime lead ships, update this one place
    // instead of hunting the same array down in two methods — duplicating
    // it is what silently dropped 2d's +12h line for a day after the skill
    // chart fix on 2026-05-06.
    private static readonly (int Lead, string Color)[] EyeballLeadSpecs =
    {
        (12, "#ce93d8"),
        (24, "#b39ddb"),
        (48, "#7c4dff"),
        (72, "#4527a0"),
    };

    // The lead set actually plotted on the eyeball vs-truth charts, derived
    // from EyeballLeadSpecs. Those charts cap their X axis at the furthest
    // valid_time among THESE leads — predictions also exist at 96/120h
    // (drawn on the rolling-MAE/Brier panels, not the eyeball charts), and
    // letting those drive the axis stretched it ~2-3 days past where any
    // drawn line reached, leaving blank space on the right.
    private static readonly HashSet<int> EyeballLeads =
        EyeballLeadSpecs.Select(s => s.Lead).ToHashSet();

    // -------------------------------------------------------------------------------
    // Eyeball: temperature vs truth, grouped by phase so champion/challenger lines at
    // different leads don't pile up in one unreadable chart.
    // -------------------------------------------------------------------------------
    private static string RenderTempVsTruthBlock(SiteInputs input)
    {
        var content = new StringBuilder();

        // Drive the panel set off ActivePhasePolicy.ByTarget["temperature"]
        // so a new phase joining the shipping lineup automatically gets a
        // panel here without touching this file. The earlier hardcoded
        // {2b, 2c} array silently dropped 2d for a day after it shipped on
        // 2026-05-05; building from the same source-of-truth ActivePhase
        // Policy uses for everywhere else (manifests, rolling-MAE, models
        // page) eliminates that drift class.
        //
        // Per-phase (Title, Description) text stays in PhaseDescriptions
        // because it's display-specific copy; an unrecognised phase shows
        // a fallback heading instead of being silently dropped.
        var versionsByPhase = input.PhaseByVersion
            .GroupBy(kv => kv.Value, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Select(kv => kv.Key).ToHashSet(StringComparer.Ordinal),
                StringComparer.Ordinal);

        var phaseSpecs = ActivePhasePolicy.ByTarget["temperature"]
            .Select(p => (
                Key: p,
                Title: PhaseDescriptions.GetValueOrDefault((p, "title"), $"Phase {p}"),
                Description: PhaseDescriptions.GetValueOrDefault((p, "desc"),
                    "(no description registered — add to SitePages.Skill.PhaseDescriptions)")))
            .ToArray();

        // Section-wide X range so 2b and 2c read as aligned panels — past extent
        // pinned to WindowStartUtc, forward to the latest prediction valid_time
        // across either phase. Truth (ERA5 + METAR) doesn't extend past now, so
        // the right edge always belongs to a prediction; left edge to truth.
        var (xMin, xMax) = TempSectionRange(input);

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
            content.Append(RenderTempVsTruthChart(input, filtered, spec.Key, xMin, xMax));
            content.Append("</article>");
        }

        if (!anyDrawn)
            content.Append("<p><em>No temperature predictions in window — run <code>predict</code>.</em></p>");

        return content.ToString();
    }

    /// <summary>
    /// 14-day rolling window ending at the latest <em>plotted</em> valid_time
    /// on the eyeball charts. Anchoring on the data's right edge instead of
    /// the input rolling-window-start keeps the chart visually consistent
    /// across renders (a sparse-data day doesn't widen the window). The right
    /// edge is the furthest of the things actually drawn: a blend line at a
    /// plotted lead (<see cref="EyeballLeads"/>, ≤72h) or the Met Office Spot
    /// reference line. 96/120h predictions are excluded — they aren't drawn
    /// here (only on the rolling panels), and letting them set the edge
    /// stretched the axis ~2-3 days past every drawn line. Falls back to
    /// truth extent when nothing plotted exists; both nulls → caller passes
    /// through to Chart.js auto-scaling.
    /// Tightened 30 → 14 days 2026-05-04; lead-capped 2026-05-22.
    /// </summary>
    private static (double? Min, double? Max) TempSectionRange(SiteInputs input)
    {
        double? max = null;
        void Bump(DateTime t)
        {
            var x = t.ToOADate();
            if (max is null || x > max) max = x;
        }

        // Blend lines — only the leads actually drawn on the eyeball charts.
        foreach (var p in input.Predictions)
            if (EyeballLeads.Contains(p.LeadHours)) Bump(p.ValidTimeUtc);

        // The MO Spot reference line is drawn too (its +24h lead bucket) —
        // count it so a predict-lag day can't clip it off the right edge.
        foreach (var m in input.MetOfficeSpotForecasts)
            if (m.LeadHours >= 24 && m.LeadHours < 48 && m.Temperature2m.HasValue)
                Bump(m.ValidTimeUtc);

        if (max is null)
            foreach (var kv in input.TruthByTime) Bump(kv.Key);

        if (max is null) return (null, null);
        return (max - 14.0, max);
    }

    private static string RenderTempVsTruthChart(
        SiteInputs input,
        IReadOnlyList<TempPredictionRow> filtered,
        string phaseKey,
        double? xMin,
        double? xMax)
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

        // Met Office DataHub Land Obs at the configured geohash. Closer
        // to Bonehill than EGTE METAR (~22 km NNW vs ~30 km E) and on
        // the same prevailing-weather side of the moor; ~250m below
        // Bonehill vs ~360m for EGTE so the systematic warm bias is
        // smaller. Read it as a second cross-check, not truth.
        var moObsPts = input.MetOfficeObsByTime
            .Where(m => m.ObservedTimeUtc >= input.WindowStartUtc)
            .OrderBy(m => m.ObservedTimeUtc)
            .Select(m => (X: m.ObservedTimeUtc.ToOADate(), Y: m.Temperature2m))
            .ToList();
        // Brown 600 for MO obs — was deep orange #fb8c00 originally but it
        // sat too close to METAR's orange #ffa726 on the chart, so the eye
        // read them as one colour family. Brown is well off-axis from every
        // other temp-skill series (red ERA5, orange METAR, purples Blend,
        // green MO Spot) so it stays visually distinct.
        if (moObsPts.Count > 0)
            series.Add(new LineSeries("Met Office obs (Taw Green)", "#6d4c41", moObsPts));

        foreach (var (lead, color) in EyeballLeadSpecs)
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

        // Met Office Spot forecast as a third reference line — same shape as
        // ERA5/METAR truth, but it's a *forecast* so we label it explicitly
        // and use the Met Office brand navy so the eye reads it as another
        // forecaster's view, not a truth source.
        // MO Spot temp filtered to the +24h lead bucket [24, 47] — matches the
        // rain skill chart's convention and our blender's lead-band training.
        // input.MetOfficeSpotForecasts is already scoped to this page's
        // location by RenderSiteCommand (single-location SiteInputs invariant).
        var moSpotPts = input.MetOfficeSpotForecasts
            .Where(m => m.ValidTimeUtc >= input.WindowStartUtc
                        && m.Temperature2m.HasValue
                        && m.LeadHours >= 24 && m.LeadHours < 48)
            .GroupBy(m => m.ValidTimeUtc)
            .Select(g => g.OrderBy(m => m.LeadHours).First())
            .OrderBy(m => m.ValidTimeUtc)
            .Select(m => (X: m.ValidTimeUtc.ToOADate(), Y: m.Temperature2m!.Value))
            .ToList();
        if (moSpotPts.Count > 0)
            series.Add(new LineSeries("Met Office Spot (+24h lead)", NwpPalette.MetOfficeSpot, moSpotPts));

        if (series.Count == 0)
            return RenderEmptyChart($"Temperature — phase {phaseKey}", "No overlap between predictions and truth in window.");

        return LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Temperature vs truth — phase {phaseKey}",
            XLabel = "Time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 360,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        });
    }

    // -------------------------------------------------------------------------------
    // Rolling Brier per (version, lead) for the *current* rain station — same shape
    // as the temp page's rolling-MAE panel but on a binary classification metric.
    // Truth conversion uses the 0.1 mm/h threshold the blender was trained on.
    // 30-day rolling window matches the precip verify command's default (wet hours
    // are sparser than temp signal so a 14-day window over-shoots).
    // -------------------------------------------------------------------------------
    private static string RenderRollingBrierBlock(SiteInputs input, string? currentStation)
    {
        if (currentStation is null || input.RollingBrier.Count == 0)
            return "<p class=\"skill-line\"><em>No rolling Brier points yet — predict tree hasn't aged into rainfall truth, or no points at the current station.</em></p>";

        var stationRows = input.RollingBrier.Where(r => r.Station == currentStation).ToList();
        if (stationRows.Count == 0)
            return "<p class=\"skill-line\"><em>No rolling Brier points for the selected station.</em></p>";

        // 30-day x-axis ending at the rightmost point so all per-lead panels
        // share the same time window. Same convention as RenderRollingMaeBlock.
        double? xMax = null;
        foreach (var r in stationRows)
        {
            var x = r.WindowEndUtc.ToOADate();
            if (xMax is null || x > xMax) xMax = x;
        }
        double? xMin = xMax is { } m ? m - 30.0 : null;

        var content = new StringBuilder();
        content.Append(Ci, $"<p class=\"skill-line\">{Escape(PrettyStation(currentStation))} — 30-day rolling Brier per (version, lead). Lower better.</p>");

        // Rolling-Brier is for precipitation (3a/3c/3d), not dry-window — so
        // ForecastsTempRain (= Leads.Full + lead 12) is the right set:
        //   - 3a/3c emit + verify at {24, 48, 72, 96, 120} so 96/120 panels
        //     surface (previously hidden by the Leads.Short copy-paste).
        //   - 3d emits at {12, 24} so the +12h panel surfaces once verify
        //     rows land (~5d ERA5 latency).
        foreach (var lead in Leads.ForecastsTempRain)
        {
            var phases = stationRows.Where(r => r.LeadHours == lead)
                .Select(r => r.Phase).Distinct().OrderBy(p => p, StringComparer.Ordinal).ToList();

            var series = new List<LineSeries>();
            for (int i = 0; i < phases.Count; i++)
            {
                var p = phases[i];
                var pts = stationRows
                    .Where(r => r.LeadHours == lead && r.Phase == p)
                    .OrderBy(r => r.WindowEndUtc)
                    .Select(r => (X: r.WindowEndUtc.ToOADate(), Y: r.BlendBrier))
                    .ToList();
                if (pts.Count > 0)
                    series.Add(new LineSeries($"Phase {p}", PrecipPhases.ColorFor(p), pts));
            }

            content.Append(Ci, $"<h4>Lead +{lead}h</h4>");
            if (series.Count == 0)
            {
                content.Append(RenderEmptyChart($"Rolling Brier — lead {lead}h", "No scored predictions at this lead."));
                continue;
            }

            content.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"Rolling Brier — {PrettyStation(currentStation)} — lead +{lead}h",
                XLabel = "Window end (UTC)",
                YLabel = "Brier",
                Series = series,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                FormatY = v => v.ToString("0.000", Ci),
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }
        return content.ToString();
    }

    // -------------------------------------------------------------------------------
    // Rolling quantitative MAE per (version, lead). One chart per lead — one line per
    // version. This is the "does the eyeball match the numbers?" cross-check.
    // -------------------------------------------------------------------------------
    private static string RenderRollingMaeBlock(SiteInputs input)
    {
        if (input.RollingMae.Count == 0)
            return "<p><em>No rolling MAE points computed — the window is too short or there's no matching ERA5 truth yet.</em></p>";

        // 30-day rolling window ending at the latest WindowEndUtc across every
        // lead × version. All per-lead panels share the same right edge and
        // back 30 days, so sparser leads no longer crop tighter than the dense
        // ones and the panel reads as one strip.
        double? xMax = null;
        foreach (var r in input.RollingMae)
        {
            var x = r.WindowEndUtc.ToOADate();
            if (xMax is null || x > xMax) xMax = x;
        }
        double? xMin = xMax is { } m ? m - 30.0 : null;

        var content = new StringBuilder();
        // ForecastsTempRain (= Leads.Full + lead 12) so the +12h panel
        // surfaces for 2d. Legacy phases (2b/2c) have no lead-12 verify
        // rows, so the +12h panel will sit on the empty-chart fallback
        // until 2d's first verify cycle lands (~5d ERA5 latency).
        foreach (var lead in Leads.ForecastsTempRain)
        {
            var phases = input.RollingMae.Where(r => r.LeadHours == lead)
                .Select(r => r.Phase).Distinct().OrderBy(p => p, StringComparer.Ordinal).ToList();

            var series = new List<LineSeries>();
            for (int i = 0; i < phases.Count; i++)
            {
                var p = phases[i];
                var pts = input.RollingMae
                    .Where(r => r.LeadHours == lead && r.Phase == p)
                    .OrderBy(r => r.WindowEndUtc)
                    .Select(r => (X: r.WindowEndUtc.ToOADate(), Y: r.BlendMae))
                    .ToList();
                if (pts.Count > 0)
                    series.Add(new LineSeries($"Phase {p}", TempPhases.ColorFor(p), pts));
            }

            content.Append(Ci, $"<h4>Lead +{lead}h</h4>");
            if (series.Count == 0)
            {
                content.Append(RenderEmptyChart($"Rolling MAE — lead {lead}h", "No scored predictions at this lead."));
                continue;
            }

            content.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"Rolling MAE — lead +{lead}h",
                XLabel = "Window end (UTC)",
                YLabel = "MAE (°C)",
                Series = series,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                FormatY = v => v.ToString("0.00", Ci),
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }
        return content.ToString();
    }

    // -------------------------------------------------------------------------------
    // Eyeball: precipitation vs observed rainfall. P(wet) lines per phase / lead,
    // with observed wet hours rendered as light-blue background bands instead of
    // discrete dots — gives a continuous "this was a wet period" stripe behind the
    // forecast lines. A dashed vertical "today" line marks where the future starts.
    // -------------------------------------------------------------------------------
    private static string RenderPrecipVsTruthBlock(SiteInputs input, string? currentStation)
    {
        var content = new StringBuilder();
        content.Append("""
            <p class="skill-line">Blue bands = observed wet hours (≥ 0.1 mm). MO Spot PoP uses a looser "any measurable precip" threshold — read it as direction-of-effect.</p>
            """);

        if (currentStation is null)
        {
            content.Append("<p><em>No precipitation predictions in window.</em></p>");
            return content.ToString();
        }

        // Sub-nav above the section selects which station we render for.
        content.Append(RenderPrecipVsTruthChart(input, currentStation));
        return content.ToString();
    }

    private static string RenderPrecipVsTruthChart(SiteInputs input, string station)
    {
        var content = new StringBuilder();
        content.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

        var stationPredictions = input.PrecipPredictions.Where(p => p.Station == station).ToList();

        // Wet-period strips replace the previous 0/1 truth-dot series. Same
        // ≥ 0.1 mm threshold the blender was trained on.
        var wetBands = input.RainfallTruth.TryGetValue(station, out var truth)
            ? ComputeWetBands(truth, input.WindowStartUtc)
            : new List<(double, double)>();

        // 14-day rolling window. Per-phase charts plus the champion-vs-
        // challenger overlay all share this time axis and stack as one panel.
        // Right edge = furthest thing actually drawn: a P(wet) line at a
        // plotted lead (EyeballLeads, ≤72h) or the MO Spot PoP reference
        // line. 96/120h predictions exist but aren't drawn here, so they're
        // excluded — letting them set xMax stretched the axis ~2-3 days past
        // every line. Tightened 30 → 14 days 2026-05-04; lead-capped 2026-05-22.
        var plottedValidTimes = stationPredictions
            .Where(p => EyeballLeads.Contains(p.LeadHours))
            .Select(p => p.ValidTimeUtc)
            .Concat(input.MetOfficeSpotForecasts
                .Where(m => m.LeadHours >= 24 && m.LeadHours < 48
                            && m.PrecipitationProbabilityPercent.HasValue)
                .Select(m => m.ValidTimeUtc))
            .ToList();
        double? xMax = plottedValidTimes.Count > 0
            ? plottedValidTimes.Max().ToOADate()
            : null;
        double? xMin = xMax is { } m ? m - 14.0 : null;

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
            foreach (var (lead, color) in EyeballLeadSpecs)
            {
                var pts = latestPerLead
                    .Where(r => r.LeadHours == lead)
                    .OrderBy(r => r.ValidTimeUtc)
                    .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                    .ToList();
                if (pts.Count > 0)
                    series.Add(new LineSeries($"P(wet) +{lead}h", color, pts));
            }

            // Met Office Spot precipitation probability as a single comparison
            // line per chart, filtered to the +24h lead bucket [24, 47].
            // Matches how our blender is trained (24-hour lead band per bucket)
            // — so the Spot line represents "what Spot would have said 24-47 h
            // before the ValidTime". Bucket [24, 47] is fully covered by Spot's
            // hourly endpoint (0-48h horizon), giving dense data; longer-lead
            // buckets would lean on the three-hourly endpoint and be sparse.
            //
            // PoP comes in percent on 0-100 → divide by 100 to share the
            // chart's [0, 1] Y-axis. NB the Met Office threshold is "any
            // measurable precip", a looser bound than our 0.1 mm/h training
            // label — the skill-line block above the chart calls this out.
            //
            // input.MetOfficeSpotForecasts is already scoped to this page's
            // location by RenderSiteCommand (single-location SiteInputs
            // invariant) — every row is this location's Spot forecast.
            var moSpotPts = input.MetOfficeSpotForecasts
                .Where(m => m.ValidTimeUtc >= input.WindowStartUtc
                            && m.PrecipitationProbabilityPercent.HasValue
                            && m.LeadHours >= 24 && m.LeadHours < 48)
                .GroupBy(m => m.ValidTimeUtc)
                .Select(g => g.OrderBy(m => m.LeadHours).First())
                .OrderBy(m => m.ValidTimeUtc)
                .Select(m => (X: m.ValidTimeUtc.ToOADate(), Y: m.PrecipitationProbabilityPercent!.Value / 100.0))
                .ToList();
            if (moSpotPts.Count > 0)
                series.Add(new LineSeries("Met Office Spot PoP (+24h lead)", NwpPalette.MetOfficeSpot, moSpotPts));

            if (series.Count == 0)
            {
                content.Append(RenderEmptyChart(
                    $"P(wet) vs observed — {station} — {spec.Key}",
                    "No forecast or truth in window."));
                continue;
            }

            content.Append(Ci, $"<h5>{Escape(spec.ShortTitle)}</h5>");
            content.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"P(wet) vs observed wet hours — {PrettyStation(station)} — Phase {spec.Key}",
                XLabel = "Time (UTC)",
                YLabel = "P(wet)",
                Series = series,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
                Bands = wetBands,
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }

        var cvc = BuildChampionVsChallengerSeries(stationPredictions, input.PhaseByVersion, input.WindowStartUtc, leadHours: 24);
        if (cvc.Count >= 2)
        {
            content.Append("<h5>Phase comparison — +24h lead</h5>");
            content.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title = $"3a vs 3c vs observed — {PrettyStation(station)} — +24h",
                XLabel = "Time (UTC)",
                YLabel = "P(wet)",
                Series = cvc,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
                Bands = wetBands,
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }

        if (!anyRendered)
            content.Append(RenderEmptyChart($"P(wet) vs observed — {station}", "No forecast or truth in window."));

        return content.ToString();
    }

    // LookupNwpLabel moved to WeatherBlend.Train.Common.Nwp.DisplayLabel
    // 2026-05-20 — same constant, three previous duplicates collapsed.

    // -------------------------------------------------------------------------------
    // Dry-window vs observed. One table per station × window length showing each
    // lead's predicted probability alongside the observed verdict (dry / wet / —).
    // -------------------------------------------------------------------------------
    private static string RenderDryWindowVsTruthTable(SiteInputs input, string? currentStation)
    {
        if (input.DryWindowPredictions.Count == 0)
            return "<p><em>No dry-window predictions in window.</em></p>";
        if (currentStation is null || !input.DryWindowPredictions.Any(d => d.Station == currentStation))
            return $"<p><em>No dry-window predictions for {Escape(PrettyStation(currentStation ?? ""))} in window.</em></p>";

        // Same station sub-nav as precip selects which station we render for here.
        var observed = ComputeObservedDryWindows(input);
        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = Leads.Short;

        var content = new StringBuilder();
        var station = currentStation;

        foreach (var window in windows)
        {
            var cutoff = input.GeneratedAtUtc.Date.AddDays(-7);
            var windowRows = input.DryWindowPredictions
                .Where(d => d.Station == station && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                .ToList();
            if (windowRows.Count == 0) continue;

            content.Append(Ci, $"<h4>{window}-hour dry window</h4>");

            // Bucket by phase so the champion + each active challenger
            // render as separate tables under the same window heading.
            // Mirrors the dry-window page's per-phase grouping. Phases
            // not in DryWindowPhases.All silently drop (they're retired).
            bool anyPhaseRendered = false;
            foreach (var phase in DryWindowPhases.All)
            {
                var phaseRows = windowRows
                    .Where(d => DryWindowPhases.Bucket(input.PhaseByVersion, d.Version) == phase)
                    .ToList();
                if (phaseRows.Count == 0) continue;
                anyPhaseRendered = true;

                // Latest prediction per (target_date, lead) within this phase bucket.
                var latest = phaseRows
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

                // Drop rows for today and future target-dates: the Observed
                // column is computed from completed-day rainfall truth, so
                // today's row would always show "—" until the day finishes
                // and EA gauges land (~3-5d ingest latency). User-set
                // 2026-05-10 — the dangling "—" rows added noise without
                // information; predictions for today/future are visible on
                // the dry-window forecasts page where they belong.
                var todayUtc = input.GeneratedAtUtc.Date;
                var dates = latest.Select(d => d.TargetDateUtc).Distinct()
                    .Where(d => d < todayUtc)
                    .OrderBy(d => d).ToList();
                if (dates.Count == 0) continue;

                var tbody = new StringBuilder();
                foreach (var date in dates)
                {
                    var byLead = latest.Where(d => d.TargetDateUtc == date).ToDictionary(d => d.LeadHours);

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                        {
                            var color = ProbabilityColor(d.ProbHasDryWindow);
                            leadCells.Append(Ci, $"<td class=\"num\" style=\"color: {color}; font-weight: 600\">{(d.ProbHasDryWindow * 100).ToString("0", Ci)}%</td>");
                        }
                        else
                        {
                            leadCells.Append("<td class=\"num\">—</td>");
                        }
                    }

                    // Tick / cross matches the prob-cell gradient endpoints so the
                    // verdict reads as "did the prediction pay off?" at a glance.
                    var observedCell = observed.TryGetValue((station, window, date), out var obs)
                        ? (obs
                            ? "<span style=\"color: #43a047; font-weight: 700\">&#x2713;</span>"
                            : "<span style=\"color: #e53935; font-weight: 700\">&#x2717;</span>")
                        : "—";

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time datetime="{date:yyyy-MM-dd}">{date:ddd} {date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          <td>{observedCell}</td>
                        </tr>
                        """);
                }

                // Only emit the phase sub-heading when more than one phase
                // is shipping for this window — single-phase windows stay
                // clean; multi-phase windows get headed for clarity.
                if (DryWindowPhases.All.Count > 1)
                {
                    content.Append(Ci, $"<h5>{Escape(phase.ShortTitle)}</h5>");
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
                            <th>Observed</th>
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);
            }

            if (!anyPhaseRendered)
            {
                content.Append("<p><em>No predictions in known phase buckets for this window.</em></p>");
            }
        }

        return content.ToString();
    }

    // -------------------------------------------------------------------------------
    // Shared helpers — used to live on the old forecast-vs-truth / precipitation
    // pages. Kept as internal statics because SitePagesTests reaches for them.
    // -------------------------------------------------------------------------------

    /// <summary>
    /// Display-text registry for temperature-phase eyeball panels on the
    /// skill page. Keyed by (phase, slot) where slot ∈ {"title", "desc"}.
    /// Unknown phases fall back to "Phase {p}" + a sentinel description in
    /// <see cref="RenderTempVsTruthBlock"/> so a newly-added phase still
    /// surfaces (with a "no description registered" hint) until copy is
    /// added here. That's the explicit failure mode replacing the silent
    /// drop we hit when 2d landed.
    ///
    /// Add a new phase: append its (key, "title") and (key, "desc") rows
    /// here AND add the phase to <c>ActivePhasePolicy.ByTarget["temperature"]</c>.
    /// </summary>
    private static readonly Dictionary<(string Phase, string Slot), string> PhaseDescriptions = new()
    {
        [("2b", "title")] = "Phase 2b lean (13 features)",
        [("2b", "desc")]  = "Six per-model temperatures, their mean/std/range, and cyclical hour/day-of-year encodings. The original champion.",
        [("2c", "title")] = "Phase 2c rich (88 features)",
        [("2c", "desc")]  = "Adds per-model dew point, RH, cloud {total/low/mid/high}, wind speed/dir/gusts, surface pressure, plus cross-model aggregates. Challenger.",
        [("2d", "title")] = "Phase 2d exact-runtime (T2, leads 12+24)",
        [("2d", "desc")]  = "Per-cycle exact init time + lead, GFS+AIFS required, IFS+MO Global optional. Champion at lead 12; challenger at lead 24. Sparse panel until verify catches up (~5d ERA5 latency).",
    };

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
                series.Add(new LineSeries(phase.ChampionVsChallengerLabel, phase.Color, pts));
        }
        return series;
    }

    /// <summary>
    /// For every (station, window, target-date) triple referenced by a prediction,
    /// delegate to <see cref="DryWindowLabelBuilder.HasDryWindow"/> using the same
    /// daytime window the trainer's labels use (<see cref="SiteInputs.DryWindowDaytime"/>).
    /// Days lacking a truth reading at any hour inside the daytime range are skipped
    /// — they'd be dropped by the labeller too. Sharing the labeller's logic means
    /// the "Observed" column can never disagree with the truth the model was scored
    /// against (the 2026-05-04 bug had it scanning the full UTC day with the wrong
    /// threshold, so wet daytimes with dry overnights got false ✓ ticks).
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

            var (startHour, endHour) = input.DryWindowDaytime.UtcHourRangeFor(DateOnly.FromDateTime(date));
            if (startHour >= endHour) continue;

            var hours = new double?[24];
            bool complete = true;
            for (int h = startHour; h < endHour; h++)
            {
                if (hourly.TryGetValue(date.AddHours(h), out var mm)) hours[h] = mm;
                else { complete = false; break; }
            }
            if (!complete) continue;

            result[(station, window, date)] =
                DryWindowLabelBuilder.HasDryWindow(hours, window, startHour, endHour);
        }
        return result;
    }
}
