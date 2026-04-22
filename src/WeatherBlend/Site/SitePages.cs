using System.Globalization;
using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// Pure HTML renderers for the static site. No I/O, no DuckDB — takes already-loaded
/// model objects and returns strings. Callers (see <c>RenderSiteCommand</c>) own the
/// query-and-write orchestration.
/// </summary>
public static class SitePages
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

    public sealed record SiteInputs
    {
        public required string LocationDisplay { get; init; }
        public required double Latitude { get; init; }
        public required double Longitude { get; init; }
        public required double ElevationMeters { get; init; }

        /// <summary>ICAO of the primary METAR station (for the chart legend). Empty → no METAR.</summary>
        public required string MetarStation { get; init; }

        public required DateTime GeneratedAtUtc { get; init; }

        /// <summary>Earliest timestamp rendered in charts. Truth series are clamped to this.</summary>
        public required DateTime WindowStartUtc { get; init; }

        /// <summary>All prediction rows in the reporting window (typically last 30d).</summary>
        public required IReadOnlyList<PredictionRow> Predictions { get; init; }

        /// <summary>ERA5 truth by ValidTime for the same window.</summary>
        public required IReadOnlyDictionary<DateTime, double> TruthByTime { get; init; }

        /// <summary>METAR observations (time-sorted) from the primary station over the window.</summary>
        public required IReadOnlyList<(DateTime ObservedTimeUtc, double Temperature2m)> MetarByTime { get; init; }

        /// <summary>Rolling MAE per (version, lead) for the verify chart.</summary>
        public required IReadOnlyList<RollingMaePoint> RollingMae { get; init; }
    }

    public sealed record RollingMaePoint(string ModelVersion, int LeadHours, DateTime WindowEndUtc, double BlendMae, int N);

    public static string RenderIndex(SiteInputs input)
    {
        // Latest prediction per lead: most-recent PredictionMadeAt wins.
        var latestByLead = input.Predictions
            .GroupBy(p => p.LeadHours)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First());

        var cards = new StringBuilder();
        foreach (var lead in new[] { 24, 48, 72 })
        {
            if (latestByLead.TryGetValue(lead, out var p))
            {
                cards.Append(Ci, $"""
                    <article class="forecast-card">
                      <header><h3>+{lead}h</h3><small>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</small></header>
                      <div class="temp">{p.BlendTemperature.ToString("0.0", Ci)}°C</div>
                      <footer>
                        <small>Made {p.PredictionMadeAtUtc:yyyy-MM-dd HH:mm}Z</small><br/>
                        <small>Model: <code>{Escape(p.ModelVersion)}</code></small>
                      </footer>
                    </article>
                    """);
            }
            else
            {
                cards.Append(Ci, $"""
                    <article class="forecast-card forecast-card-empty">
                      <header><h3>+{lead}h</h3></header>
                      <div class="temp">—</div>
                      <footer><small>No prediction available</small></footer>
                    </article>
                    """);
            }
        }

        var chart = RenderOverviewChart(input);

        // Headline skill line (blend MAE vs best single over the window).
        var skill = ComputeHeadlineSkill(input);

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Latest blended forecast</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m</p>
              </hgroup>
              <div class="forecast-grid">
            {cards}  </div>
            </section>

            <section>
              <h2>Recent conditions & forecast</h2>
              <p class="skill-line">Truth lines (ERA5, METAR) run through the past; blend & per-model forecasts mark the next 24/48/72h from the most recent predict run.</p>
              {chart}
              <p class="skill-line">{Escape(skill)}</p>
            </section>
            """);

        return WrapPage(input, "Home", "index", body.ToString());
    }

    /// <summary>
    /// The headline chart. Two truth series (ERA5, METAR from the primary station) plus
    /// one series per lead-hour bucket — the 24h, 48h, and 72h blend forecasts are
    /// produced by separate per-lead blenders, so each gets its own line. Over time each
    /// lead-line traces that horizon's forecast trajectory against the truth.
    /// </summary>
    private static string RenderOverviewChart(SiteInputs input)
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
            var metarLabel = string.IsNullOrWhiteSpace(input.MetarStation) ? "METAR" : $"METAR {input.MetarStation}";
            series.Add(new LineSeries(metarLabel, "#ffa726", metarPts));
        }

        // One line per lead bucket — each lead is a separate blender, and plotting a
        // single line across the three would imply a forecast trajectory that doesn't
        // exist. Colours are a purple gradient so they read visually as "same family".
        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),  // light
            (48, "#7c4dff"),  // mid
            (72, "#4527a0"),  // dark
        };
        foreach (var (lead, color) in leadSpecs)
        {
            var pts = input.Predictions
                .Where(p => p.LeadHours == lead)
                .OrderBy(p => p.ValidTimeUtc)
                .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.BlendTemperature))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries($"Blend +{lead}h", color, pts));
        }

        if (series.Count == 0)
        {
            return RenderEmptyChart(
                "Recent conditions & forecast",
                "No predictions, ERA5, or METAR data in window.");
        }

        return LineChartRenderer.Render(new LineChartSpec
        {
            Title = "Recent conditions (truth) & blend forecasts by lead",
            XLabel = "Time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 380,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
        });
    }

    public static string RenderPredictions(SiteInputs input)
    {
        var rows = input.Predictions
            .OrderByDescending(p => p.PredictionMadeAtUtc)
            .ThenBy(p => p.LeadHours)
            .ToList();

        var tableBody = new StringBuilder();
        foreach (var p in rows)
        {
            var truth = input.TruthByTime.TryGetValue(p.ValidTimeUtc, out var t) ? t : (double?)null;
            var err = truth.HasValue ? p.BlendTemperature - truth.Value : (double?)null;
            tableBody.Append(Ci, $"""
                <tr>
                  <td><time>{p.PredictionMadeAtUtc:yyyy-MM-dd HH:mm}Z</time></td>
                  <td><time>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</time></td>
                  <td>{p.LeadHours}h</td>
                  <td class="num"><strong>{p.BlendTemperature.ToString("0.00", Ci)}</strong></td>
                  <td class="num">{FmtNullable(truth)}</td>
                  <td class="num">{FmtNullable(err, "+0.00;-0.00;0.00")}</td>
                  <td class="num">{FmtNullable(p.TempGfs)}</td>
                  <td class="num">{FmtNullable(p.TempEcmwf)}</td>
                  <td class="num">{FmtNullable(p.TempIcon)}</td>
                  <td class="num">{FmtNullable(p.TempMf)}</td>
                  <td class="num">{FmtNullable(p.TempUkmo)}</td>
                  <td class="num">{FmtNullable(p.TempGem)}</td>
                  <td><small><code>{Escape(p.ModelVersion)}</code></small></td>
                </tr>
                """);
        }

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Predictions</h2>
                <p>{rows.Count} rows, sorted by prediction time (newest first).</p>
              </hgroup>
              <figure>
                <table>
                  <thead>
                    <tr>
                      <th>Made at</th>
                      <th>Valid time</th>
                      <th>Lead</th>
                      <th class="num">Blend</th>
                      <th class="num">ERA5</th>
                      <th class="num">Err</th>
                      <th class="num">GFS</th>
                      <th class="num">ECMWF</th>
                      <th class="num">ICON</th>
                      <th class="num">MF</th>
                      <th class="num">UKMO</th>
                      <th class="num">GEM</th>
                      <th>Version</th>
                    </tr>
                  </thead>
                  <tbody>
            {tableBody}      </tbody>
                </table>
              </figure>
            </section>
            """);

        return WrapPage(input, "Predictions", "predictions", body.ToString());
    }

    public static string RenderVerify(SiteInputs input)
    {
        var content = new StringBuilder();

        content.Append("""
            <section>
              <hgroup>
                <h2>Verification</h2>
                <p>Rolling blend-vs-ERA5 MAE by lead time, stratified by model version.</p>
              </hgroup>
            """);

        if (input.RollingMae.Count == 0)
        {
            content.Append("<p><em>No rolling MAE points computed — the window is too short or there's no matching ERA5 truth yet.</em></p>");
        }
        else
        {
            // One chart per lead, with one line per model version.
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
                    {
                        series.Add(new LineSeries(v, palette[i % palette.Length], pts));
                    }
                }

                if (series.Count == 0)
                {
                    content.Append(Ci, $"<h3>Lead {lead}h</h3>");
                    content.Append(RenderEmptyChart($"Rolling MAE — lead {lead}h", "No scored predictions at this lead."));
                    continue;
                }

                content.Append(Ci, $"<h3>Lead {lead}h</h3>");
                content.Append(LineChartRenderer.Render(new LineChartSpec
                {
                    Title = $"Rolling MAE — lead {lead}h",
                    XLabel = "Window end (UTC)",
                    YLabel = "MAE (°C)",
                    Series = series,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                    FormatY = v => v.ToString("0.00", Ci),
                }));
            }
        }

        content.Append("</section>");
        return WrapPage(input, "Verification", "verify", content.ToString());
    }

    public static string RenderAbout(SiteInputs input)
    {
        var body = $"""
            <section>
              <h2>About WeatherBlend</h2>
              <p>
                WeatherBlend is a multi-model weather-forecast blending proof of concept for
                <strong>{Escape(input.LocationDisplay)}</strong>
                ({input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m).
                It combines forecasts from six numerical weather-prediction (NWP) models —
                GFS, ECMWF IFS, DWD ICON, Météo-France, UK Met Office, and Environment Canada GEM —
                via a LightGBM blender trained against ERA5 reanalysis.
              </p>

              <h3>Scope</h3>
              <p>
                This PoC targets temperature at lead times <strong>24h, 48h and 72h</strong>.
                Shorter and longer horizons are out of scope. Precipitation is a future phase.
              </p>

              <h3>Data sources</h3>
              <ul>
                <li><strong>Forecasts:</strong> Open-Meteo (live + historical-forecast API).</li>
                <li><strong>Training truth:</strong> ERA5 reanalysis via Open-Meteo (gapless, quantitative).</li>
                <li><strong>Verification truth:</strong> METAR from aviationweather.gov and OGIMET, used as a real-observation sanity check.</li>
              </ul>

              <h3>Caveats</h3>
              <ul>
                <li>ERA5 is a 0.25° gridded reanalysis; it represents a grid-cell average near Bonehill Rocks, not the tor itself.</li>
                <li>Lowland METAR (Exeter, Yeovilton) has systematic biases against a 393m moorland tor — useful as a cross-check, not ground truth.</li>
                <li>Blender predictions currently cover one model version and a short verification window.</li>
              </ul>

              <p><small>Source: <a href="https://github.com/">WeatherBlend repo</a>. Rendered {input.GeneratedAtUtc:yyyy-MM-dd HH:mm}Z.</small></p>
            </section>
            """;

        return WrapPage(input, "About", "about", body);
    }

    public static string Stylesheet() => """
        :root { --brand: #7c4dff; }
        body > main { padding-top: 1rem; padding-bottom: 3rem; }
        nav.site-nav { padding: 0.5rem 0 1rem; border-bottom: 1px solid var(--pico-muted-border-color); margin-bottom: 1.5rem; }
        nav.site-nav ul { display: flex; gap: 1rem; list-style: none; padding: 0; margin: 0; }
        nav.site-nav a { text-decoration: none; }
        nav.site-nav a.active { font-weight: 600; color: var(--brand); }

        .forecast-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .forecast-card { padding: 1rem; }
        .forecast-card h3 { margin: 0 0 0.25rem; }
        .forecast-card .temp { font-size: 2.5rem; font-weight: 700; color: var(--brand); line-height: 1.1; margin: 0.5rem 0; }
        .forecast-card-empty .temp { color: var(--pico-muted-color); }

        .skill-line { font-style: italic; color: var(--pico-muted-color); }

        table td.num, table th.num { text-align: right; font-variant-numeric: tabular-nums; }
        table small { color: var(--pico-muted-color); }

        svg.chart { width: 100%; height: auto; max-width: 100%; background: var(--pico-card-background-color); border-radius: 4px; margin: 0.5rem 0 1.5rem; }
        .chart-title { font-size: 14px; font-weight: 600; fill: var(--pico-color); }
        .chart-grid { stroke: var(--pico-muted-border-color); stroke-width: 0.5; }
        .chart-frame { fill: none; stroke: var(--pico-muted-border-color); stroke-width: 1; }
        .chart-tick { font-size: 10px; fill: var(--pico-muted-color); font-family: ui-monospace, monospace; }
        .chart-tick-mark { stroke: var(--pico-muted-border-color); stroke-width: 1; }
        .chart-axis-label { font-size: 11px; fill: var(--pico-muted-color); }
        .chart-legend { font-size: 11px; fill: var(--pico-color); font-family: ui-monospace, monospace; }
        .chart-empty { font-size: 12px; fill: var(--pico-muted-color); font-style: italic; }

        footer.site-footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--pico-muted-border-color); color: var(--pico-muted-color); font-size: 0.875rem; }
        """;

    private static string WrapPage(SiteInputs input, string pageTitle, string pageId, string bodyHtml)
    {
        return $$"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>{{Escape(pageTitle)}} — WeatherBlend</title>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
              <link rel="stylesheet" href="styles.css">
            </head>
            <body>
              <main class="container">
                <header>
                  <hgroup>
                    <h1>WeatherBlend</h1>
                    <p>Multi-model forecast blending for {{Escape(input.LocationDisplay)}}</p>
                  </hgroup>
                  <nav class="site-nav">
                    <ul>
                      <li><a href="index.html"{{NavActive(pageId, "index")}}>Home</a></li>
                      <li><a href="predictions.html"{{NavActive(pageId, "predictions")}}>Predictions</a></li>
                      <li><a href="verify.html"{{NavActive(pageId, "verify")}}>Verification</a></li>
                      <li><a href="about.html"{{NavActive(pageId, "about")}}>About</a></li>
                    </ul>
                  </nav>
                </header>

                {{bodyHtml}}

                <footer class="site-footer">
                  Rendered {{input.GeneratedAtUtc.ToString("yyyy-MM-dd HH:mm", Ci)}}Z ·
                  Training truth: ERA5 · Verification: ERA5 + METAR ·
                  <a href="about.html">About this site</a>
                </footer>
              </main>
            </body>
            </html>
            """;
    }

    private static string NavActive(string current, string target)
        => current == target ? " class=\"active\"" : "";

    private static string ComputeHeadlineSkill(SiteInputs input)
    {
        var scored = input.Predictions
            .Where(p => p.LeadHours == 24 && input.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();
        if (scored.Count == 0) return "No scored predictions yet — skill headline will appear once ERA5 catches up.";

        var truth = scored.Select(p => input.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend = scored.Select(p => p.BlendTemperature).ToArray();
        var blendMae = Mae(blend, truth);

        // Best single across the six per-model columns.
        (string name, double mae)[] singles =
        {
            ("GFS",   MaeWithGaps(scored.Select(p => p.TempGfs).ToArray(),   truth)),
            ("ECMWF", MaeWithGaps(scored.Select(p => p.TempEcmwf).ToArray(), truth)),
            ("ICON",  MaeWithGaps(scored.Select(p => p.TempIcon).ToArray(),  truth)),
            ("MF",    MaeWithGaps(scored.Select(p => p.TempMf).ToArray(),    truth)),
            ("UKMO",  MaeWithGaps(scored.Select(p => p.TempUkmo).ToArray(),  truth)),
            ("GEM",   MaeWithGaps(scored.Select(p => p.TempGem).ToArray(),   truth)),
        };
        var best = singles.Where(s => !double.IsNaN(s.mae)).OrderBy(s => s.mae).FirstOrDefault();
        if (best.name is null || double.IsNaN(best.mae))
        {
            return $"Over {scored.Count} scored 24h predictions, blend MAE is {blendMae.ToString("0.00", Ci)}°C.";
        }

        var pct = (best.mae - blendMae) / best.mae * 100.0;
        var verb = blendMae < best.mae ? "beat" : "trailed";
        return $"Over {scored.Count} scored 24h predictions, blend MAE is {blendMae.ToString("0.00", Ci)}°C — {verb} best single ({best.name}, {best.mae.ToString("0.00", Ci)}°C) by {Math.Abs(pct).ToString("0.0", Ci)}%.";
    }

    private static double Mae(double[] pred, double[] truth)
    {
        double sum = 0;
        int n = Math.Min(pred.Length, truth.Length);
        for (int i = 0; i < n; i++) sum += Math.Abs(pred[i] - truth[i]);
        return n == 0 ? double.NaN : sum / n;
    }

    private static double MaeWithGaps(double?[] pred, double[] truth)
    {
        double sum = 0;
        int n = 0;
        for (int i = 0; i < pred.Length && i < truth.Length; i++)
        {
            if (pred[i] is double v) { sum += Math.Abs(v - truth[i]); n++; }
        }
        return n == 0 ? double.NaN : sum / n;
    }

    private static string FmtNullable(double? v, string fmt = "0.00")
        => v.HasValue ? v.Value.ToString(fmt, Ci) : "—";

    private static string RenderEmptyChart(string title, string message) =>
        $"""<div class="chart-empty-box"><strong>{Escape(title)}</strong><p><em>{Escape(message)}</em></p></div>""";

    private static string Escape(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
}
