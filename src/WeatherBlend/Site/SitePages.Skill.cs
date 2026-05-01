using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

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
        content.Append("""
            <section>
              <hgroup>
                <h2>Temperature skill</h2>
                <p>Eyeball comparison first — each active blender's temperature trajectory plotted against
                   ERA5 + METAR truth. Rolling quantitative MAE by lead follows, so you can see whether the
                   visual impression holds up under aggregation.</p>
              </hgroup>
            """);

        content.Append("<h3>Vs truth</h3>");
        content.Append(RenderTempVsTruthBlock(input));

        content.Append("<hr/><h3>Rolling MAE</h3>");
        content.Append(RenderRollingMaeBlock(input));

        content.Append("</section>");
        return WrapPage(input, "Temperature skill", "skill-temperature", content.ToString());
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
    public static string RenderRainSkill(SiteInputs input, string? stationSlug = null)
    {
        var stations = GetRainSkillStations(input);
        var currentStation = ResolveStationFromSlug(stations, stationSlug);

        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Rainfall skill</h2>
                <p>Eyeball first: P(wet) trajectories against a 0/1 wet-hour indicator from the same
                   ≥ 0.1 mm threshold the blender was trained on. Dry-window predicted vs observed verdict
                   follows. Stations sit on separate EA gauges, so flip between them via the sub-nav.</p>
              </hgroup>
            """);

        if (currentStation is not null)
            content.Append(RenderStationSubNav("skill-rainfall", stations, currentStation));

        content.Append("<h3>P(wet) vs observed wet-hour</h3>");
        content.Append(RenderPrecipVsTruthBlock(input, currentStation));

        content.Append("<hr/><h3>Dry window — predicted vs observed</h3>");
        content.Append(Ci, $"""
            <p class="skill-line">One row per target UTC day × window length. Both prediction and observed verdict are scoped
               to the 09–18 local-time daytime window (Europe/London, DST-aware). The "observed" column is blank for dates
               beyond the last full rainfall day.</p>
            """);
        content.Append(RenderDryWindowVsTruthTable(input, currentStation));

        content.Append("</section>");
        return WrapPage(input, "Rainfall skill", "skill-rainfall", content.ToString());
    }

    /// <summary>
    /// Station set used by the rainfall-skill sub-nav. Union of precip and dry-window
    /// stations so the sub-nav shows every station the reader might care about, even
    /// if one of the two sections has no data for it.
    /// </summary>
    internal static IReadOnlyList<string> GetRainSkillStations(SiteInputs input)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in input.PrecipPredictions) set.Add(p.Station);
        foreach (var d in input.DryWindowPredictions) set.Add(d.Station);
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

        (string Key, string Title, string Description)[] phaseSpecs =
        {
            ("2b", "Phase 2b lean (13 features)",
                "Six per-model temperatures, their mean/std/range, and cyclical hour/day-of-year encodings. The original champion."),
            ("2c", "Phase 2c rich (88 features)",
                "Adds per-model dew point, RH, cloud {total/low/mid/high}, wind speed/dir/gusts, surface pressure, plus cross-model aggregates. Challenger."),
        };

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
    /// 30-day rolling window ending at the latest prediction valid_time across
    /// the section. Anchoring on the data's right edge instead of the input
    /// rolling-window-start keeps the chart visually consistent across renders
    /// (a sparse-data day doesn't widen the window) and tracks the forward
    /// forecast horizon — when predictions extend +120h the window slides
    /// with them. Falls back to truth extent when no predictions exist; both
    /// nulls → caller passes through to Chart.js auto-scaling.
    /// </summary>
    private static (double? Min, double? Max) TempSectionRange(SiteInputs input)
    {
        double? max = null;
        foreach (var p in input.Predictions)
        {
            var x = p.ValidTimeUtc.ToOADate();
            if (max is null || x > max) max = x;
        }
        if (max is null)
        {
            foreach (var kv in input.TruthByTime)
            {
                var x = kv.Key.ToOADate();
                if (max is null || x > max) max = x;
            }
        }
        if (max is null) return (null, null);
        return (max - 30.0, max);
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

        // Met Office Spot forecast as a third reference line — same shape as
        // ERA5/METAR truth, but it's a *forecast* so we label it explicitly
        // and use the Met Office brand navy so the eye reads it as another
        // forecaster's view, not a truth source.
        var moSpotPts = input.MetOfficeSpotForecasts
            .Where(m => m.ValidTimeUtc >= input.WindowStartUtc && m.Temperature2m.HasValue)
            .OrderBy(m => m.ValidTimeUtc)
            .Select(m => (X: m.ValidTimeUtc.ToOADate(), Y: m.Temperature2m!.Value))
            .ToList();
        if (moSpotPts.Count > 0)
            series.Add(new LineSeries("Met Office Spot", NwpPalette.MetOfficeSpot, moSpotPts));

        if (series.Count == 0)
            return RenderEmptyChart($"Temperature — phase {phaseKey}", "No overlap between predictions and truth in window.");

        return LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title = $"Temperature vs truth — phase {phaseKey}",
            XLabel = "Time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 360,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
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
        foreach (var lead in Leads.Full)
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
            <p class="skill-line">P(wet) is the blender's probability that the next hour sees ≥ 0.1 mm. Light-blue vertical bands
               behind the lines mark hours where the gauge actually recorded ≥ 0.1 mm — so reading the chart is "during the
               blue stripe, did our P(wet) lines climb?". A dashed vertical line marks today; everything to the right is
               forecast-only. Hours with fewer than 4 of 4 15-min readings are dropped to avoid flipping wet↔dry at the
               boundary. The dark-navy "Met Office Spot PoP" line, when present, is the Met Office DataHub Spot product's
               own probability for the Bonehill point — its threshold is "any measurable precip", a slightly looser bound
               than our 0.1 mm/h, so read it as direction-of-effect alongside our blender rather than a like-for-like overlay.</p>
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

        // 30-day rolling window ending at the latest prediction valid_time for
        // this station. Per-phase charts plus the champion-vs-challenger
        // overlay all share the same time axis and stack as one panel.
        double? xMax = stationPredictions.Count > 0
            ? stationPredictions.Max(p => p.ValidTimeUtc).ToOADate()
            : null;
        double? xMin = xMax is { } m ? m - 30.0 : null;

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

            // Met Office Spot precipitation probability as a comparison line.
            // The DataHub Spot product emits PoP in percent on a 0–100 scale —
            // divide by 100 here so it shares the chart's [0, 1] Y-axis with
            // P(wet). NB the Met Office threshold is "any measurable precip",
            // a slightly looser bound than our 0.1 mm/h training label, so the
            // comparison is direction-of-effect rather than apples-to-apples.
            // The skill-line block above the chart calls this out.
            var moSpotPts = input.MetOfficeSpotForecasts
                .Where(m => m.ValidTimeUtc >= input.WindowStartUtc && m.PrecipitationProbabilityPercent.HasValue)
                .OrderBy(m => m.ValidTimeUtc)
                .Select(m => (X: m.ValidTimeUtc.ToOADate(), Y: m.PrecipitationProbabilityPercent!.Value / 100.0))
                .ToList();
            if (moSpotPts.Count > 0)
                series.Add(new LineSeries("Met Office Spot PoP", NwpPalette.MetOfficeSpot, moSpotPts));

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

    /// <summary>Map an Open-Meteo model id to the short label used in
    /// <see cref="NwpsForPrecipitation"/> ("gfs_seamless" → "GFS", etc.).
    /// Falls back to the raw model id when unrecognised. Only ~4 NWPs
    /// publish <c>precipitation_probability</c> via Open-Meteo (GFS / ECMWF
    /// / ICON / GEM); the others have no PoP rows in the forecasts tree
    /// and silently drop out of any legend that filters by "has PoP rows".</summary>
    private static string LookupNwpLabel(string modelId) => modelId switch
    {
        "gfs_seamless"          => "GFS",
        "ecmwf_ifs025"          => "ECMWF",
        "icon_seamless"         => "ICON",
        "meteofrance_seamless"  => "MF",
        "ukmo_seamless"         => "UKMO",
        "gem_seamless"          => "GEM",
        "ecmwf_aifs025_single"  => "AIFS",
        "jma_seamless"          => "JMA",
        _ => modelId,
    };

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

        {
            var station = currentStation;

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
                series.Add(new LineSeries(phase.ChampionVsChallengerLabel, phase.Color, pts));
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
