using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Wind tab — top-level page introduced 2026-05-28 alongside Slice 3.A/B/C
    /// of the WIND_BLENDER_PLAN. Layout is "Option A" from the design memo
    /// (project_wind_tab_design_2026-05-28): a single hero showing the
    /// champion's speed + gust + direction, an inline model-comparison
    /// strip just below (champion vs challengers), then a 24-hour forecast
    /// strip + 3-day summary.
    ///
    /// Decisions locked 2026-05-28:
    ///   * mph everywhere (m/s × 2.23694)
    ///   * UTC everywhere (no local-time conversion anywhere on the site)
    ///   * Direction shown using the intuitive "to" convention (arrow
    ///     points the way the wind is blowing, not the way it comes from);
    ///     label carries the heading-towards cardinal/degree.
    ///   * Gust source is wind_gust_lgb directly (already shipped 2026-05-27)
    ///   * Below 2 mph the direction collapses to a "VARIABLE" indicator
    ///   * Wedge half-angle from per-row Ci95 spread; ≥270° spread also
    ///     snaps to VARIABLE.
    ///
    /// Today's MVP renders what's available from <see cref="SiteInputs"/>:
    ///   * Speed from <see cref="SiteInputs.FeelsLikePredictions"/>
    ///     (carries <c>WindSpeed10mMs</c> per ValidTimeUtc).
    ///   * Gust from <see cref="SiteInputs.WindGustByValidMs"/>.
    ///   * Direction: empty-state placeholder until wind_mvn predictions
    ///     ship (first Sunday retrain post-2026-05-31).
    ///   * Challenger strip: only the champion row populated until
    ///     wind_speed_lgb / wind_blend predict trees exist on R2.
    ///
    /// Future work hooked from this file:
    ///   * Data plumbing for wind_mvn direction + Ci95 bands → real wedge
    ///   * Data plumbing for wind_speed_lgb + wind_blend speeds → live
    ///     challenger rows in the model strip
    ///   * Per-NWP speed overlay below the hero (mirroring rain page)
    /// </summary>
    public static string RenderForecastsWind(SiteInputs input)
    {
        var body = new StringBuilder();
        body.Append("<section>");
        // RenderingFor is nullable on multi-location aggregate pages; the
        // per-location render path always populates it.
        var displayName = input.RenderingFor?.DisplayName ?? "this location";
        body.Append(Ci, $"""
              <hgroup>
                <h2>Wind</h2>
                <p>Direction (arrow shows where the wind is blowing <em>to</em>), speed,
                   and gust at {Escape(displayName)}.</p>
              </hgroup>
            """);

        // -- Hero card: current/imminent wind cell ---------------------
        body.Append(RenderWindHero(input));

        // -- Inline model-comparison strip (Option A) ------------------
        body.Append(RenderWindModelStrip(input));

        // -- 24-hour forecast strip ------------------------------------
        body.Append(RenderWindHourlyStrip(input));

        // -- 3-day summary ---------------------------------------------
        body.Append(RenderWindThreeDaySummary(input));

        body.Append("</section>");
        return WrapPage(input, "Wind", "wind", body.ToString());
    }

    private const double MsToMph = 2.23694;

    /// <summary>Pick the FeelsLike row closest to <c>GeneratedAtUtc</c>
    /// (forwards-biased — the imminent hour rather than the just-past
    /// one). Returns null when the page has no FeelsLikePredictions for
    /// this location (e.g. element blenders not trained yet at this
    /// location).</summary>
    private static (DateTime ValidTimeUtc, double? SpeedMs, double? GustMs)? CurrentWindCell(SiteInputs input)
    {
        var anchor = input.GeneratedAtUtc;
        var rows = input.FeelsLikePredictions
            .Where(p => p.WindSpeed10mMs.HasValue && p.ValidTimeUtc >= anchor.AddHours(-1))
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();
        if (rows.Count == 0)
        {
            // Fallback: latest available point even if it's in the past
            // (better than nothing for the hero — the timestamp surfaces
            // how stale the value is).
            var fallback = input.FeelsLikePredictions
                .Where(p => p.WindSpeed10mMs.HasValue)
                .OrderByDescending(p => p.ValidTimeUtc)
                .FirstOrDefault();
            if (fallback is null) return null;
            input.WindGustByValidMs.TryGetValue(fallback.ValidTimeUtc, out var fallbackGust);
            return (fallback.ValidTimeUtc, fallback.WindSpeed10mMs, fallbackGust > 0 ? fallbackGust : null);
        }
        var pick = rows[0];
        input.WindGustByValidMs.TryGetValue(pick.ValidTimeUtc, out var gust);
        return (pick.ValidTimeUtc, pick.WindSpeed10mMs, gust > 0 ? gust : null);
    }

    private static string RenderWindHero(SiteInputs input)
    {
        var cell = CurrentWindCell(input);
        if (cell is null)
        {
            return """
                <div class="wind-hero wind-hero-empty">
                  <p><em>No wind forecast available yet for this location — element blenders haven't produced rows for the current cycle.</em></p>
                </div>
                """;
        }
        var (valid, speedMs, gustMs) = cell.Value;
        var speedMph = (speedMs ?? 0) * MsToMph;
        var gustMph  = (gustMs  ?? 0) * MsToMph;
        var isVariable = speedMph < 2.0;

        // Direction: placeholder until wind_mvn predictions are wired.
        // When direction lands, swap this block for an inline SVG arrow +
        // wedge (see project_wind_tab_design_2026-05-28 for the math).
        var directionBlock = isVariable
            ? RenderVariableWindGlyph()
            : RenderDirectionPlaceholder();

        var gustLine = gustMs.HasValue
            ? $"<p class=\"wind-gust\">gust {gustMph:0} mph</p>"
            : "";

        return string.Format(Ci, """
            <div class="wind-hero">
              <div class="wind-hero-arrow">{0}</div>
              <div class="wind-hero-readout">
                <p class="wind-speed">{1:0} mph</p>
                {2}
                <p class="wind-validtime">{3:yyyy-MM-dd HH:mm} UTC · model wind (champion)</p>
              </div>
            </div>
            """, directionBlock, speedMph, gustLine, valid);
    }

    /// <summary>Variable-wind glyph: full circle, no arrow, "VARIABLE"
    /// label. Used when speed &lt; 2 mph (direction is meaningless at low
    /// wind) and when the wind_mvn CI spread exceeds ~270°.</summary>
    private static string RenderVariableWindGlyph()
        => """
        <svg viewBox="0 0 80 80" width="80" height="80" aria-label="variable wind direction">
          <circle cx="40" cy="40" r="32" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="3 4"/>
          <text x="40" y="46" text-anchor="middle" font-size="11" fill="#666">VAR</text>
        </svg>
        """;

    /// <summary>Empty-state direction glyph shown until wind_mvn predict
    /// data is plumbed into SiteInputs. Once wind_mvn lands, swap to an
    /// arrow + wedge driven by BlendDirection + Ci95Lo/Hi.</summary>
    private static string RenderDirectionPlaceholder()
        => """
        <svg viewBox="0 0 80 80" width="80" height="80" aria-label="direction model not yet trained">
          <circle cx="40" cy="40" r="32" fill="none" stroke="#ccc" stroke-width="2"/>
          <text x="40" y="44" text-anchor="middle" font-size="9" fill="#999">dir model</text>
          <text x="40" y="56" text-anchor="middle" font-size="9" fill="#999">pending</text>
        </svg>
        """;

    private static string RenderWindModelStrip(SiteInputs input)
    {
        // Option A: one row per phase + Δ vs champion. Until challenger
        // predict trees exist, only the champion row populates with a
        // live value; the others sit as "pending" placeholders so the
        // visual structure ships intact.
        var cell = CurrentWindCell(input);
        var championMph = cell?.SpeedMs * MsToMph;

        var sb = new StringBuilder();
        sb.Append("""
            <details class="wind-models" open>
              <summary>Compare model forecasts</summary>
              <table class="wind-models-table">
                <thead><tr>
                  <th>Model</th><th class="num">Speed (mph)</th>
                  <th class="num">Δ vs champion</th><th>Role</th>
                </tr></thead><tbody>
            """);

        // Champion: wind (ERA5-truth LightGBM). Currently the headline.
        // Δ vs champion is by construction 0.0.
        var champText = championMph.HasValue
            ? string.Create(Ci, $"<td class=\"num\">{championMph:0.0}</td><td class=\"num\">—</td>")
            : "<td class=\"num\">—</td><td class=\"num\">—</td>";
        sb.Append(Ci, $"<tr><td>★ wind</td>{champText}<td>champion</td></tr>");

        // Challengers: pending until predict trees land on R2 post-
        // Sunday retrain. The Δ column lights up once we have a live
        // value for each row.
        sb.Append("""
            <tr>
              <td>wind_speed_lgb</td>
              <td class="num">—</td>
              <td class="num">—</td>
              <td>challenger · LightGBM, Dunkeswell-truth · pending first Sunday retrain</td>
            </tr>
            <tr>
              <td>wind_blend</td>
              <td class="num">—</td>
              <td class="num">—</td>
              <td>challenger · sigmoid(wind_speed_lgb, wind_mvn) · pending</td>
            </tr>
            """);

        sb.Append("""
              </tbody></table>
              <p class="wind-models-footnote">
                Speed and skill compared on the
                <a href="skill-wind.html">wind Skill page</a> (rolling MAE
                vs Dunkeswell SYNOP truth). Δ shows the per-cell deviation
                from the champion model at the current forecast hour.
              </p>
            </details>
            """);
        return sb.ToString();
    }

    private static string RenderWindHourlyStrip(SiteInputs input)
    {
        // Pull the next 24 forward-looking hourly cells with a wind
        // value. Rendered as two rows of 12 columns to fit a mobile
        // viewport; hours read in UTC throughout.
        var anchor = input.GeneratedAtUtc;
        var rows = input.FeelsLikePredictions
            .Where(p => p.WindSpeed10mMs.HasValue && p.ValidTimeUtc >= anchor)
            .OrderBy(p => p.ValidTimeUtc)
            .Take(24)
            .ToList();
        if (rows.Count == 0)
        {
            return "<div class=\"wind-hourly wind-hourly-empty\"><p><em>No hourly wind forecast available yet.</em></p></div>";
        }
        var sb = new StringBuilder();
        sb.Append("<div class=\"wind-hourly\">");
        sb.Append("<h3>Next 24 hours</h3>");
        sb.Append("<table class=\"wind-hourly-table\"><thead><tr>");
        foreach (var p in rows) sb.Append(Ci, $"<th>{p.ValidTimeUtc:HH}</th>");
        sb.Append("</tr></thead><tbody><tr>");
        foreach (var p in rows)
        {
            var mph = (p.WindSpeed10mMs ?? 0) * MsToMph;
            sb.Append(Ci, $"<td class=\"num\">{mph:0}</td>");
        }
        sb.Append("</tr><tr>");
        foreach (var p in rows)
        {
            // Gust where present; em-dash otherwise so the column stays
            // aligned at calm rows.
            if (input.WindGustByValidMs.TryGetValue(p.ValidTimeUtc, out var gustMs))
            {
                var gustMph = gustMs * MsToMph;
                sb.Append(Ci, $"<td class=\"num gust\">{gustMph:0}</td>");
            }
            else
            {
                sb.Append("<td class=\"num gust\">—</td>");
            }
        }
        sb.Append("</tr></tbody></table>");
        sb.Append("<p class=\"wind-hourly-key\">Top: speed (mph). Bottom: gust (mph). Hours UTC.</p>");
        sb.Append("</div>");
        return sb.ToString();
    }

    private static string RenderWindThreeDaySummary(SiteInputs input)
    {
        // Group forward forecast by valid-day UTC and summarise per day:
        // mph range (min, max), max gust. Limited to 3 days for tile fit.
        var anchor = input.GeneratedAtUtc;
        var byDay = input.FeelsLikePredictions
            .Where(p => p.WindSpeed10mMs.HasValue && p.ValidTimeUtc >= anchor)
            .GroupBy(p => p.ValidTimeUtc.Date)
            .OrderBy(g => g.Key)
            .Take(3)
            .ToList();
        if (byDay.Count == 0) return "";
        var sb = new StringBuilder();
        sb.Append("<div class=\"wind-days\">");
        sb.Append("<h3>Next 3 days</h3>");
        sb.Append("<div class=\"wind-day-cards\">");
        foreach (var day in byDay)
        {
            var speeds = day.Select(p => p.WindSpeed10mMs!.Value * MsToMph).ToList();
            var min = speeds.Min();
            var max = speeds.Max();
            double? maxGust = null;
            foreach (var p in day)
            {
                if (input.WindGustByValidMs.TryGetValue(p.ValidTimeUtc, out var g))
                {
                    var gMph = g * MsToMph;
                    if (maxGust is null || gMph > maxGust) maxGust = gMph;
                }
            }
            var gustLine = maxGust.HasValue
                ? string.Create(Ci, $"gust to {maxGust:0} mph")
                : "no gust data";
            sb.Append(Ci, $"""
                <div class="wind-day-card">
                  <p class="wind-day-date">{day.Key:ddd dd MMM}</p>
                  <p class="wind-day-range">{min:0}–{max:0} mph</p>
                  <p class="wind-day-gust">{gustLine}</p>
                </div>
                """);
        }
        sb.Append("</div></div>");
        return sb.ToString();
    }
}
