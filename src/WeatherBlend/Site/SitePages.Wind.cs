using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Per-lead wind forecast page (rewritten 2026-05-28 from the
    /// snapshot-style first cut). Lead-axis sub-nav mirrors the temp + rain
    /// pages; each tab renders a stacked pair of speed charts (multi-phase
    /// blender on top with the wind_mvn 95% CI ribbon, raw per-NWP overlay
    /// below) plus a row of per-hour direction circles for the latest day
    /// this lead predicts.
    ///
    /// Locked decisions (Harry, 2026-05-28):
    ///   * Direction shown using the "to" convention — arrow points the
    ///     way the wind is blowing. Wedge fill spans the CI95 angular
    ///     spread; ≥270° spread or sub-2 mph speed snaps to VARIABLE.
    ///   * mph everywhere (m/s × 2.23694).
    ///   * UTC everywhere, no local-time conversion.
    ///   * Direction-circle hours match the home-overview window
    ///     [<see cref="HomeFirstVisibleHourUtc"/>,
    ///      <see cref="HomeLastVisibleHourUtcExclusive"/>) so the day's
    ///     story reads consistently across pages.
    /// </summary>
    public static string RenderForecastsWind(SiteInputs input, int lead)
    {
        var body = new StringBuilder();
        body.Append("<section>");
        body.Append(RenderWindLeadSubNav(lead));

        body.Append(Ci, $"""
              <hgroup>
                <h2>Wind +{lead}h</h2>
                <p>Blender speed forecasts on top (with the wind_mvn 95% CI band shaded);
                   raw NWP inputs below. Per-hour direction circles at the bottom show
                   the latest day this lead predicts.</p>
              </hgroup>
            """);

        body.Append(RenderWindSpeedSection(input, lead));
        body.Append(RenderWindDirectionCirclesSection(input, lead));

        body.Append("</section>");
        return WrapPage(input, $"Wind forecast +{lead}h", "wind", body.ToString());
    }

    private const double MsToMph = 2.23694;

    private static string RenderWindLeadSubNav(int current)
    {
        var items = new StringBuilder();
        foreach (var lead in Leads.Short)
        {
            var cls = lead == current ? " class=\"active\"" : "";
            items.Append(Ci, $"""<li><a href="forecasts-wind-{lead}h.html"{cls}>+{lead}h</a></li>""");
        }
        return $"""<nav class="lead-nav"><ul>{items}</ul></nav>""";
    }

    // ----------------------------------------------------------------
    // Speed charts — blender (multi-phase) + MVN CI band + per-NWP
    // ----------------------------------------------------------------

    private static string RenderWindSpeedSection(SiteInputs input, int lead)
    {
        var s = new StringBuilder();

        var atLead = input.WindPredictions
            .Where(p => p.LeadHours == lead)
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();
        var mvnAtLead = input.WindDirectionPredictions
            .Where(p => p.LeadHours == lead)
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        if (atLead.Count == 0 && mvnAtLead.Count == 0)
        {
            s.Append("<h3>Wind speed — our blend</h3>");
            s.Append(Ci, $"<p><em>No +{lead}h wind speed forecast available yet.</em></p>");
            return s.ToString();
        }

        var allXs = atLead.Select(p => p.ValidTimeUtc)
            .Concat(mvnAtLead.Select(p => p.ValidTimeUtc))
            .ToList();
        var maxValid = allXs.Count == 0 ? input.GeneratedAtUtc : allXs.Max();
        if (maxValid < input.GeneratedAtUtc) maxValid = input.GeneratedAtUtc;
        var xMin = ForecastChartXMin(input);
        var xMax = maxValid.ToOADate();

        // ---- Chart 1: our blenders + MVN ribbon ----
        // One line per phase. Phase is read from input.PhaseByVersion (the
        // standard mapping populated by ModelMetadataRepository); fall back
        // to the version literal so a fresh bundle without a phase entry
        // still renders.
        var blendSeries = new List<LineSeries>();
        var orderedPhases = ActivePhasePolicy.ByTarget.TryGetValue("wind", out var phaseList)
            ? phaseList
            : new List<string> { "wind", "wind_speed_lgb", "wind_blend" };

        // Group at-lead rows by phase, keep freshest PredictedAtUtc per
        // ValidTime (covers the case where two versions of the same phase
        // are both Active — newer cycle wins).
        //
        // input.PhaseByVersion only carries entries for temperature /
        // precipitation / dry_window bundles (see ModelMetadataRepository.
        // GetPhaseByVersion) — it never reads element (wind / humidity /
        // …) training_metadata. So for wind versions the dict lookup
        // misses and we have to derive the phase from the version-string
        // convention the trainers stamp:
        //   * "v_wind_blend_live"            → wind_blend (minted, no bundle)
        //   * "...{_wind_speed_lgb}"         → wind_speed_lgb
        //   * "...{_wind_mvn}"               → wind_mvn
        //   * bare "vYYYY-MM-DD_hhmmss"      → wind (champion, pre-VersionSuffix)
        // Long-term we'd want GetPhaseByVersion to also read element
        // bundle metadata.
        static string DeriveWindPhase(string version)
        {
            if (version.StartsWith("v_wind_blend", StringComparison.Ordinal)
                || version.Contains("_wind_blend", StringComparison.Ordinal))
                return "wind_blend";
            if (version.Contains("_wind_speed_lgb", StringComparison.Ordinal))
                return "wind_speed_lgb";
            if (version.Contains("_wind_mvn", StringComparison.Ordinal))
                return "wind_mvn";
            return "wind";
        }
        string PhaseOf(string version)
            => input.PhaseByVersion.TryGetValue(version, out var ph) && !string.IsNullOrEmpty(ph)
               ? ph : DeriveWindPhase(version);

        var byPhase = atLead
            .GroupBy(p => PhaseOf(p.Version))
            .ToDictionary(g => g.Key, g => g
                .GroupBy(r => r.ValidTimeUtc)
                .Select(gv => gv.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .ToList());

        for (int i = 0; i < orderedPhases.Count; i++)
        {
            var phase = orderedPhases[i];
            if (!byPhase.TryGetValue(phase, out var rows) || rows.Count == 0) continue;
            var color = i == 0 ? NwpPalette.Blend
                : i == 1 ? NwpPalette.BlendChallenger
                : NwpPalette.BlendExactChallenger;
            var label = i == 0 ? $"Blend ({phase} champion)" : $"Blend ({phase} challenger)";
            blendSeries.Add(new LineSeries(label, color,
                rows.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.SpeedMs * MsToMph)).ToList()));
        }

        // wind_mvn: speed magnitude line + CI95 ribbon. The ribbon's low +
        // high series both need to be in Series so the renderer can match
        // them by name. Boundary lines stay in the legend (mirrors the
        // rainfall_amount P10/P90 pattern in SitePages.RainfallAmount.cs)
        // — light grey so the eye reads the ribbon as the "envelope"
        // rather than two competing lines, while still leaving them as
        // identifiable reference values.
        if (mvnAtLead.Count > 0)
        {
            blendSeries.Add(new LineSeries("MVN speed", "#1976d2",
                mvnAtLead.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.SpeedMs * MsToMph)).ToList()));
            blendSeries.Add(new LineSeries("MVN CI95 low", "#bbdefb",
                mvnAtLead.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.SpeedCi95LoMs * MsToMph)).ToList()));
            blendSeries.Add(new LineSeries("MVN CI95 high", "#bbdefb",
                mvnAtLead.Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.SpeedCi95HiMs * MsToMph)).ToList()));
        }

        var ribbons = mvnAtLead.Count > 0
            ? new[] { new RibbonSpec("MVN CI95 low", "MVN CI95 high", "rgba(25, 118, 210, 0.18)") }
            : Array.Empty<RibbonSpec>();

        // wind_gust_lgb gust line — pulled from input.WindGustByValidMs
        // (already collapsed to freshest PredictionMadeAtUtc per ValidTime
        // by the SQL window function). Plotted at the SAME ValidTime grid
        // as the champion blend rows so the chart isn't haloed with
        // off-grid points; one row per ValidTime in the at-lead set.
        var gustPts = atLead
            .Select(r => (
                X: r.ValidTimeUtc.ToOADate(),
                G: input.WindGustByValidMs.TryGetValue(r.ValidTimeUtc, out var g) ? (double?)g : null))
            .Where(p => p.G.HasValue)
            .Select(p => (p.X, Y: p.G!.Value * MsToMph))
            .ToList();
        if (gustPts.Count > 0)
        {
            // Amber so it pops against the purple blend lines + blue MVN
            // ribbon without competing with the per-NWP chart's palette.
            blendSeries.Add(new LineSeries("wind_gust_lgb gust", "#ff9800", gustPts));
        }

        s.Append("<h3>Wind speed — our blend</h3>");
        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title  = $"Blend wind speed — +{lead}h (mph)",
            XLabel = "Valid time (UTC)",
            YLabel = "Wind speed (mph)",
            Series = blendSeries,
            Ribbons = ribbons,
            Height = 280,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0", Ci),
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        }));

        // ---- Chart 2: raw per-NWP overlay ----
        // We need per-NWP values for the same set of rows. Use the champion
        // wind phase rows (one per ValidTime) as the source so each NWP
        // gets a single value per hour.
        var championPhase = orderedPhases.Count > 0 ? orderedPhases[0] : "wind";
        if (!byPhase.TryGetValue(championPhase, out var nwpRows))
        {
            nwpRows = atLead.Count > 0 ? atLead : new List<WindForecastPoint>();
        }

        var nwpSeries = new List<LineSeries>();
        void AddNwp(string label, string color, Func<WindForecastPoint, double?> selector)
        {
            var pts = nwpRows
                .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: selector(r) ?? double.NaN, Has: selector(r).HasValue))
                .Where(p => p.Has)
                .Select(p => (p.X, Y: p.Y * MsToMph))
                .ToList();
            if (pts.Count > 0) nwpSeries.Add(new LineSeries(label, color, pts));
        }
        AddNwp(Nwp.DisplayLabel("gfs_seamless"),          NwpPalette.Gfs,   r => r.Gfs);
        AddNwp(Nwp.DisplayLabel("ecmwf_ifs025"),          NwpPalette.Ecmwf, r => r.Ecmwf);
        AddNwp(Nwp.DisplayLabel("icon_seamless"),         NwpPalette.Icon,  r => r.Icon);
        AddNwp(Nwp.DisplayLabel("meteofrance_seamless"),  NwpPalette.Mf,    r => r.Mf);
        AddNwp(Nwp.DisplayLabel("ukmo_seamless"),         NwpPalette.Ukmo,  r => r.Ukmo);
        AddNwp(Nwp.DisplayLabel("gem_seamless"),          NwpPalette.Gem,   r => r.Gem);
        AddNwp(Nwp.DisplayLabel("ecmwf_aifs025_single"),  NwpPalette.Aifs,  r => r.Aifs);

        s.Append("<h3>NWP wind speed forecasts</h3>");
        if (nwpSeries.Count == 0)
        {
            s.Append("<p><em>No per-NWP wind speed values in this window.</em></p>");
        }
        else
        {
            s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
            {
                Title  = $"NWP wind speed — +{lead}h (mph)",
                XLabel = "Valid time (UTC)",
                YLabel = "Wind speed (mph)",
                Series = nwpSeries,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0", Ci),
                TodayLineX = input.GeneratedAtUtc.ToOADate(),
                XMin = xMin,
                XMax = xMax,
            }));
        }

        return s.ToString();
    }

    // ----------------------------------------------------------------
    // Direction circles — per-hour row for the latest day at this lead
    // ----------------------------------------------------------------

    private static string RenderWindDirectionCirclesSection(SiteInputs input, int lead)
    {
        var s = new StringBuilder();
        s.Append("<h3>Wind direction</h3>");

        var mvnAtLead = input.WindDirectionPredictions
            .Where(p => p.LeadHours == lead)
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        if (mvnAtLead.Count == 0)
        {
            s.Append(Ci, $"<p><em>No +{lead}h wind_mvn direction predictions available yet. The MVN bundle is trained on Dunkeswell SYNOP truth; circles will populate once predict-wind-direction has run for this cycle.</em></p>");
            return s.ToString();
        }

        // Latest day this lead predicts (group by UTC date, take the latest).
        var latestDate = mvnAtLead
            .GroupBy(p => p.ValidTimeUtc.Date)
            .Select(g => g.Key)
            .OrderByDescending(d => d)
            .First();
        var firstHour = HomeFirstVisibleHourUtc;
        var lastHourExcl = HomeLastVisibleHourUtcExclusive;

        var byHour = mvnAtLead
            .Where(p => p.ValidTimeUtc.Date == latestDate)
            .ToDictionary(p => p.ValidTimeUtc.Hour, p => p);

        s.Append(Ci, $"""<p class="wind-day-label">Latest day at +{lead}h: <strong>{latestDate:ddd dd MMM yyyy}</strong> (UTC; hours {firstHour:00}–{lastHourExcl - 1:00})</p>""");

        var grid = new StringBuilder();
        grid.Append("<div class=\"wind-dir-row\">");
        for (int h = firstHour; h < lastHourExcl; h++)
        {
            grid.Append("<div class=\"wind-dir-cell\">");
            grid.Append(Ci, $"<div class=\"wind-dir-hour\">{h:00}Z</div>");
            if (byHour.TryGetValue(h, out var row))
            {
                grid.Append(RenderWindDirectionCircle(row));
                var speedMph = row.SpeedMs * MsToMph;
                grid.Append(Ci, $"<div class=\"wind-dir-speed\">{speedMph:0} mph</div>");
            }
            else
            {
                grid.Append(RenderEmptyDirectionCircle());
                grid.Append("<div class=\"wind-dir-speed\">—</div>");
            }
            grid.Append("</div>");
        }
        grid.Append("</div>");
        s.Append(grid);

        return s.ToString();
    }

    /// <summary>One wind-direction circle. Convention: meteorological
    /// <c>BlendDirection</c> ("from") is converted to "to" (the arrow
    /// points the way the wind is blowing) by adding 180°. The wedge fill
    /// spans the CI95 range mapped to "to" space too. ≥270° spread or
    /// sub-2 mph speed renders the VARIABLE indicator instead.</summary>
    private static string RenderWindDirectionCircle(WindDirectionForecastPoint p)
    {
        var speedMph = p.SpeedMs * MsToMph;
        var lo = p.DirectionCi95LoDeg;
        var hi = p.DirectionCi95HiDeg;
        // Wrap-around CI: if hi < lo, the interval crosses 360 → unwrap.
        var span = (hi - lo + 360.0) % 360.0;
        if (span >= 270.0 || speedMph < 2.0) return RenderVariableWindGlyph();

        var toDir = (p.DirectionDeg + 180.0) % 360.0;
        var toLo = (lo + 180.0) % 360.0;
        var toHi = (hi + 180.0) % 360.0;

        // SVG: 60×60. Centre (30,30). Outer circle r=24. Arrow + wedge.
        var arrow = ArrowSvg(toDir);
        var wedge = WedgeSvg(toLo, toHi);
        return $"""
            <svg viewBox="0 0 60 60" width="60" height="60" class="wind-dir-svg" aria-label="wind to {toDir:0}° at {speedMph:0} mph">
              <circle cx="30" cy="30" r="24" fill="none" stroke="#ccd" stroke-width="1"/>
              {wedge}
              {arrow}
              <text x="30" y="6" text-anchor="middle" font-size="6" fill="#888">N</text>
            </svg>
            """;
    }

    /// <summary>Empty-state direction circle drawn when no MVN row matches
    /// this hour (gaps in the per-hour mapping).</summary>
    private static string RenderEmptyDirectionCircle()
        => """
        <svg viewBox="0 0 60 60" width="60" height="60" class="wind-dir-svg wind-dir-empty" aria-label="no direction data">
          <circle cx="30" cy="30" r="24" fill="none" stroke="#eee" stroke-width="1"/>
          <text x="30" y="34" text-anchor="middle" font-size="9" fill="#bbb">—</text>
        </svg>
        """;

    /// <summary>Variable-wind glyph: dashed circle, no arrow, "VAR" label.</summary>
    private static string RenderVariableWindGlyph()
        => """
        <svg viewBox="0 0 60 60" width="60" height="60" class="wind-dir-svg" aria-label="variable wind direction">
          <circle cx="30" cy="30" r="24" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="3 4"/>
          <text x="30" y="34" text-anchor="middle" font-size="9" fill="#666">VAR</text>
        </svg>
        """;

    /// <summary>Arrow inside a 60×60 SVG centred at (30,30), pointing at
    /// <paramref name="toDirDeg"/> (0=N, 90=E, clockwise). Length 18,
    /// arrowhead 5×4.</summary>
    private static string ArrowSvg(double toDirDeg)
    {
        // Maths convention: x-axis right (E=0°), y-axis up. We have
        // compass-to (N=0°, clockwise). dx = sin(θ), dy = -cos(θ).
        var rad = toDirDeg * Math.PI / 180.0;
        var dx = Math.Sin(rad);
        var dy = -Math.Cos(rad);
        var tipX = 30 + 18 * dx;
        var tipY = 30 + 18 * dy;
        var tailX = 30 - 14 * dx;
        var tailY = 30 - 14 * dy;
        // Arrowhead: two short barbs ±150° off the shaft direction.
        var barbAng1 = (toDirDeg + 150.0) * Math.PI / 180.0;
        var barbAng2 = (toDirDeg - 150.0) * Math.PI / 180.0;
        var bx1 = tipX + 6 * Math.Sin(barbAng1);
        var by1 = tipY - 6 * Math.Cos(barbAng1);
        var bx2 = tipX + 6 * Math.Sin(barbAng2);
        var by2 = tipY - 6 * Math.Cos(barbAng2);
        return string.Format(Ci, """
            <line x1="{0:0.#}" y1="{1:0.#}" x2="{2:0.#}" y2="{3:0.#}" stroke="#7c4dff" stroke-width="2" stroke-linecap="round"/>
            <polyline points="{4:0.#},{5:0.#} {6:0.#},{7:0.#} {8:0.#},{9:0.#}" fill="none" stroke="#7c4dff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            """, tailX, tailY, tipX, tipY, bx1, by1, tipX, tipY, bx2, by2);
    }

    /// <summary>SVG arc wedge from <paramref name="loDeg"/> to
    /// <paramref name="hiDeg"/> compass-to degrees, filled with a faint
    /// brand colour so the CI is visible but not noisy.</summary>
    private static string WedgeSvg(double loDeg, double hiDeg)
    {
        // SVG coords: 0° points up (N), clockwise. We use the same
        // convention so no flip needed beyond dy negation in trig.
        var span = (hiDeg - loDeg + 360.0) % 360.0;
        if (span < 1.0) return "";
        var loRad = loDeg * Math.PI / 180.0;
        var hiRad = hiDeg * Math.PI / 180.0;
        var r = 22.0;
        var x1 = 30 + r * Math.Sin(loRad);
        var y1 = 30 - r * Math.Cos(loRad);
        var x2 = 30 + r * Math.Sin(hiRad);
        var y2 = 30 - r * Math.Cos(hiRad);
        var largeArc = span > 180.0 ? 1 : 0;
        // Pie-slice: M centre → L p1 → A radius … → L centre Z
        return string.Format(Ci,
            "<path d=\"M 30 30 L {0:0.#} {1:0.#} A {2:0.#} {2:0.#} 0 {3} 1 {4:0.#} {5:0.#} Z\" fill=\"rgba(124, 77, 255, 0.18)\" stroke=\"none\"/>",
            x1, y1, r, largeArc, x2, y2);
    }
}
