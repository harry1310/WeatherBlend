using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Sea state page (Sennen sea-state Phase 2, 2026-06-11 — Harry's ask:
    /// "wave, tide and wind details on the site for Sennen"). Single page,
    /// no lead sub-nav: wave_height_lgb is ANY-LEAD (one continuous hourly
    /// forecast per cycle, 0–146 h) so a per-lead axis would slice the same
    /// curve into arbitrary pieces. Three sections:
    ///
    ///   1. Significant wave height — q50 line + shaded CQR 90% band
    ///      (same band-ribbon idiom as the wind page's wind_speed_lgb chart).
    ///   2. Tide height vs MSL — best_match sea_level_height_msl pass-through,
    ///      with a dashed y=0 MSL reference line.
    ///   3. Swell — primary + secondary swell-height chart and a collapsible
    ///      3-hourly detail table (period / "from" direction with compass
    ///      point / SST).
    ///
    /// Wind lived here for one day (2026-06-11→12) and moved to Sennen's own
    /// wind tab — "just like wind for bonehill" — the morning after.
    ///
    /// Wave rows arrive already collapsed to one row per ValidTimeUtc by
    /// RenderSiteCommand's SQL, but the page re-applies the same rule
    /// defensively (newest ModelVersion, then freshest PredictedAtUtc) —
    /// the second-ModelVersion-straddling-a-retrain case has broken charts
    /// twice before (see SeriesDedup's header).
    /// </summary>
    public static string RenderSeaState(SiteInputs input)
    {
        var body = new StringBuilder();
        body.Append("<section>");

        body.Append("""
              <hgroup>
                <h2>Sea state</h2>
                <p>Significant wave height from the wave_height_lgb blender (quantile-LGB
                   over the five wave models, with its shaded 90% prediction band —
                   conformal-calibrated quantile regression), plus the tide curve, swell
                   detail and 10 m wind for the cliff. Heights in metres, times UTC.
                   Directions are "coming from".</p>
              </hgroup>
            """);

        // Defensive version-dedup (see summary). SQL already collapsed; a
        // straddling retrain between the query and a future code change must
        // not re-introduce two rows per hour.
        var waves = input.WavePredictions
            .GroupBy(w => w.ValidTimeUtc)
            .Select(g => g
                .OrderByDescending(w => w.Version, StringComparer.Ordinal)
                .ThenByDescending(w => w.PredictedAtUtc)
                .First())
            .OrderBy(w => w.ValidTimeUtc)
            .ToList();

        if (waves.Count == 0)
        {
            body.Append("<h3>Wave height</h3>");
            body.Append("<p><em>No wave-height predictions available yet. The chart will populate once predict-wave has run for this cycle.</em></p>");
        }
        else
        {
            var maxValid = waves.Max(w => w.ValidTimeUtc);
            if (maxValid < input.GeneratedAtUtc) maxValid = input.GeneratedAtUtc;
            var xMin = ForecastChartXMin(input);
            var xMax = maxValid.ToOADate();

            body.Append(RenderWaveHeightSection(input, waves, xMin, xMax));
            body.Append(RenderTideSection(input, waves, xMin, xMax));
            body.Append(RenderSwellSection(input, waves, xMin, xMax));
        }

        // Wind moved to Sennen's own wind tab 2026-06-12 (Harry: "just like
        // wind for bonehill") — the sea-state combiner still reads wind data
        // directly; this page stays waves + tide + swell.

        body.Append("</section>");
        return WrapPage(input, "Sea state", "sea-state", body.ToString());
    }

    /// <summary>Brand colour for the wave q50 line — matches the
    /// wave_height_lgb display colour in phases.yaml so the Models page
    /// and this chart read as the same product.</summary>
    private const string WaveColor = "#0277bd";

    // ----------------------------------------------------------------
    // Significant wave height — q50 line + CQR 90% band ribbon
    // ----------------------------------------------------------------

    private static string RenderWaveHeightSection(
        SiteInputs input, IReadOnlyList<WaveForecastPoint> waves, double xMin, double xMax)
    {
        var s = new StringBuilder();
        s.Append("<h3>Wave height</h3>");

        var series = new List<LineSeries>
        {
            new("wave_height_lgb q50", WaveColor,
                waves.Select(w => (X: w.ValidTimeUtc.ToOADate(), Y: w.WaveHeightM)).ToList()),
        };

        // Band ribbon — same fallback contract as the wind page's CQR band:
        // rows without band columns contribute no band points; the q50 line
        // renders unchanged, the chart never breaks.
        var ribbons = Array.Empty<RibbonSpec>();
        var bandPts = waves.Where(w => w.BandLoM.HasValue && w.BandHiM.HasValue).ToList();
        if (bandPts.Count > 0)
        {
            series.Add(new LineSeries("90% lo", "rgba(2, 119, 189, 0.35)",
                bandPts.Select(w => (X: w.ValidTimeUtc.ToOADate(), Y: w.BandLoM!.Value)).ToList()));
            series.Add(new LineSeries("90% hi", "rgba(2, 119, 189, 0.35)",
                bandPts.Select(w => (X: w.ValidTimeUtc.ToOADate(), Y: w.BandHiM!.Value)).ToList()));
            ribbons = new[] { new RibbonSpec("90% lo", "90% hi", "rgba(2, 119, 189, 0.14)") };
        }

        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title  = "Significant wave height (m)",
            XLabel = "Valid time (UTC)",
            YLabel = "Wave height (m)",
            Series = series,
            Ribbons = ribbons,
            Height = 280,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.0", Ci),
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
            // Wave height is a physical magnitude — the axis must read from
            // 0 so a calm week doesn't auto-scale into looking dramatic.
            YMin = 0,
        }));
        return s.ToString();
    }

    // ----------------------------------------------------------------
    // Tide — best_match sea-level curve with the y=0 MSL reference
    // ----------------------------------------------------------------

    private static string RenderTideSection(
        SiteInputs input, IReadOnlyList<WaveForecastPoint> waves, double xMin, double xMax)
    {
        var s = new StringBuilder();
        s.Append("<h3>Tide</h3>");

        var tidePts = waves
            .Where(w => w.TideHeightMsl.HasValue)
            .Select(w => (X: w.ValidTimeUtc.ToOADate(), Y: w.TideHeightMsl!.Value))
            .ToList();
        if (tidePts.Count == 0)
        {
            s.Append("<p><em>No tide-height values in this window (the sea_level_height_msl pass-through hasn't landed).</em></p>");
            return s.ToString();
        }

        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title  = "Tide height vs mean sea level (m)",
            XLabel = "Valid time (UTC)",
            YLabel = "Tide height (m, MSL = 0)",
            Series = new List<LineSeries>
            {
                new("Tide height", "#0288d1", tidePts),
                // Constant y=0 reference so the reader sees where MSL sits in
                // the ±3 m oscillation — drawn as a dashed series (the chart
                // annotation plugin payload only knows vertical lines/bands).
                new("MSL (0 m)", "#9e9e9e",
                    new[] { (X: xMin, Y: 0.0), (X: xMax, Y: 0.0) }, Dashed: true),
            },
            Height = 240,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.0", Ci),
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
        }));
        return s.ToString();
    }

    // ----------------------------------------------------------------
    // Swell — primary + secondary height chart, 3-hourly detail table
    // ----------------------------------------------------------------

    private static string RenderSwellSection(
        SiteInputs input, IReadOnlyList<WaveForecastPoint> waves, double xMin, double xMax)
    {
        var s = new StringBuilder();
        s.Append("<h3>Swell</h3>");

        var primaryPts = waves
            .Where(w => w.SwellHeightM.HasValue)
            .Select(w => (X: w.ValidTimeUtc.ToOADate(), Y: w.SwellHeightM!.Value))
            .ToList();
        var secondaryPts = waves
            .Where(w => w.SecondarySwellHeightM.HasValue)
            .Select(w => (X: w.ValidTimeUtc.ToOADate(), Y: w.SecondarySwellHeightM!.Value))
            .ToList();

        if (primaryPts.Count == 0 && secondaryPts.Count == 0)
        {
            s.Append("<p><em>No swell values in this window.</em></p>");
            return s.ToString();
        }

        var series = new List<LineSeries>();
        if (primaryPts.Count > 0) series.Add(new LineSeries("Primary swell", "#00897b", primaryPts));
        if (secondaryPts.Count > 0) series.Add(new LineSeries("Secondary swell", "#80cbc4", secondaryPts, Dashed: true));

        s.Append(LineChartRenderer.RenderChartJs(new LineChartSpec
        {
            Title  = "Swell height (m)",
            XLabel = "Valid time (UTC)",
            YLabel = "Swell height (m)",
            Series = series,
            Height = 240,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.0", Ci),
            TodayLineX = input.GeneratedAtUtc.ToOADate(),
            XMin = xMin,
            XMax = xMax,
            YMin = 0,
        }));

        // 3-hourly detail table — hourly would be ~146 rows of near-identical
        // values (swell evolves slowly); 3-hourly keeps the table scannable
        // while still resolving the tide-cycle-scale changes that matter.
        // Collapsed by default via the same hourly-detail idiom the rain
        // pages use. Future rows only — the table answers "what's coming",
        // the charts carry the recent past.
        var rows = waves
            .Where(w => w.ValidTimeUtc >= input.GeneratedAtUtc && w.ValidTimeUtc.Hour % 3 == 0)
            .ToList();
        if (rows.Count == 0) return s.ToString();

        var t = new StringBuilder();
        t.Append("""<details class="hourly-detail"><summary>3-hourly swell detail</summary><figure><table>""");
        t.Append("<thead><tr><th>Time (UTC)</th><th class=\"num\">Swell (m)</th><th class=\"num\">Period (s)</th><th>From</th><th class=\"num\">2nd swell (m)</th><th class=\"num\">2nd period (s)</th><th class=\"num\">Sea temp (°C)</th></tr></thead><tbody>");
        foreach (var w in rows)
        {
            var from = w.SwellDirectionDeg is { } deg
                ? $"{CompassPoint(deg)} ({deg:0}°)"
                : "—";
            t.Append(Ci, $"""
                <tr>
                  <td>{w.ValidTimeUtc:ddd dd MMM HH}Z</td>
                  <td class="num">{FmtNullable(w.SwellHeightM, "0.0")}</td>
                  <td class="num">{FmtNullable(w.SwellPeriodS, "0.0")}</td>
                  <td>{Escape(from)}</td>
                  <td class="num">{FmtNullable(w.SecondarySwellHeightM, "0.0")}</td>
                  <td class="num">{FmtNullable(w.SecondarySwellPeriodS, "0.0")}</td>
                  <td class="num">{FmtNullable(w.SeaSurfaceTempC, "0.0")}</td>
                </tr>
                """);
        }
        t.Append("</tbody></table></figure></details>");
        s.Append(t);

        return s.ToString();
    }

    /// <summary>16-point compass label for a "coming from" bearing in
    /// degrees (0 = N, clockwise). Inputs outside [0, 360) wrap.</summary>
    internal static string CompassPoint(double degrees)
    {
        string[] points =
        {
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
        };
        var normalized = ((degrees % 360.0) + 360.0) % 360.0;
        var idx = (int)Math.Round(normalized / 22.5) % 16;
        return points[idx];
    }

}
