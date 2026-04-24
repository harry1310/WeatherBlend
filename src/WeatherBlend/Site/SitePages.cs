using System.Globalization;
using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// Pure HTML renderers for the static site. No I/O, no DuckDB — takes already-loaded
/// model objects and returns strings. Callers (see <c>RenderSiteCommand</c>) own the
/// query-and-write orchestration.
///
/// Split across partial files by page: index, forecasts (per lead), dry-window,
/// skill (merged eyeball + rolling), models, about. This file holds the shared input
/// contract, page-chrome (<see cref="WrapPage"/>, <see cref="Stylesheet"/>) and small helpers.
/// </summary>
public static partial class SitePages
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

        /// <summary>Phase 3a P(wet) predictions across all stations / versions in the window.</summary>
        public required IReadOnlyList<PrecipForecastPoint> PrecipPredictions { get; init; }

        /// <summary>Phase 3b P(dry-window) predictions across all (station, window) blenders.</summary>
        public required IReadOnlyList<DryWindowForecastPoint> DryWindowPredictions { get; init; }

        /// <summary>
        /// training_metadata.Phase per ModelVersion that appears in Predictions. Used by
        /// the forecast-vs-truth page to bucket lines into "2b lean" vs "2c rich" groups.
        /// Missing entries (older versions, missing metadata) are treated as an empty phase
        /// by callers — those versions end up in a "other" group if present.
        /// </summary>
        public IReadOnlyDictionary<string, string> PhaseByVersion { get; init; }
            = new Dictionary<string, string>();

        /// <summary>
        /// Hourly observed rainfall (mm/h) per EA rainfall station slug. Aggregated from
        /// the 15-minute EA gauge parquet tree with the same 4-of-4 rule used by
        /// PrecipVerify — partial hours dropped so a half-observed hour can't flip wet↔dry.
        /// Key is the EA station slug (e.g. <c>ea_bellever_dartmoor</c>), matching the
        /// prediction <c>TruthStation</c> column.
        /// </summary>
        public IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> RainfallTruth { get; init; }
            = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>();

        /// <summary>
        /// Temperature <c>Manifest.Current</c> — the official champion. When set, the home
        /// page filters its forecast cards to this version so results are deterministic
        /// even while a challenger is also active. Empty string → no filter (fall back to
        /// "latest by PredictionMadeAt" across all versions).
        /// </summary>
        public string CurrentVersion { get; init; } = "";

        /// <summary>
        /// Per-station precipitation champion (<c>StationEntry.Current</c>) keyed by EA
        /// station slug (e.g. <c>ea_bellever_dartmoor</c>). The home page pulls the P(wet)
        /// chip from this version only, so a challenger that happens to write the same
        /// (lead, valid_time) never leaks onto the headline card.
        /// </summary>
        public IReadOnlyDictionary<string, string> PrecipCurrentByStation { get; init; }
            = new Dictionary<string, string>();

        /// <summary>
        /// Per-model held-out test scores (Blend vs ERA5 / EA rainfall) for every active
        /// blender, grouped by composite. One entry per (target, optional station, optional
        /// window) × version. Drives the Models page; empty when no training metadata is
        /// on disk (first-render, missing rclone sync, etc.).
        /// </summary>
        public IReadOnlyList<ModelSummary> ModelSummaries { get; init; }
            = Array.Empty<ModelSummary>();
    }

    /// <summary>
    /// Composite-neutral blender metric snapshot for the Models page. Pulls the subset
    /// of fields from <c>training_metadata.json</c> that actually appear in the site; the
    /// rest (hyperparameters, deviations-from-brief) are skipped to keep the Site layer
    /// from depending on <c>WeatherBlend.Train</c>.
    ///
    /// <c>Composite</c> names the (target, station?, window?) triple in a format the
    /// Models page can sort by: "temperature", "precipitation / bellever_dartmoor",
    /// "dry_window / bellever_dartmoor / 6h". Scoring metric per target: temperature is
    /// MAE (°C); precipitation is Brier score; dry-window is Brier score. The
    /// <see cref="MetricLabel"/> string identifies which.
    /// </summary>
    public sealed record ModelSummary(
        string Composite,
        string Version,
        string Phase,
        string DataSource,
        DateTime TrainedAtUtc,
        string MetricLabel,
        IReadOnlyDictionary<int, PerLeadMetric> PerLead);

    /// <summary>One (lead → metric) entry, kept neutral so the Site layer doesn't import Train types.</summary>
    public sealed record PerLeadMetric(
        int LeadHours,
        string BestSingle,
        double BestSingleValMae,
        double BlendTestScore,
        double BlendTestRmse,
        double BlendTestBias,
        int TestRows,
        int TestCalendarMonths);

    public sealed record RollingMaePoint(string ModelVersion, int LeadHours, DateTime WindowEndUtc, double BlendMae, int N);

    public sealed record PrecipForecastPoint(
        string Station,
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double ProbWet,
        double ClimatologyPWet,
        double? PrecipGfs,
        double? PrecipEcmwf,
        double? PrecipIcon,
        double? PrecipMf,
        double? PrecipUkmo,
        double? PrecipGem);

    public sealed record DryWindowForecastPoint(
        string Station,
        int WindowHours,
        string Version,
        DateTime PredictedAtUtc,
        DateTime TargetDateUtc,
        int LeadHours,
        double ProbHasDryWindow,
        double ClimatologyProbHasDryWindow,
        double? AgreementHasDryWindow);

    public static string Stylesheet() => """
        :root { --brand: #7c4dff; --pwet: #0288d1; }
        body > main { padding-top: 1rem; padding-bottom: 3rem; }
        nav.site-nav { padding: 0.5rem 0 1rem; border-bottom: 1px solid var(--pico-muted-border-color); margin-bottom: 1.5rem; }
        nav.site-nav ul { display: flex; gap: 1rem; list-style: none; padding: 0; margin: 0; }
        nav.site-nav a { text-decoration: none; }
        nav.site-nav a.active { font-weight: 600; color: var(--brand); }

        .forecast-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .forecast-card { padding: 1rem; }
        .forecast-card h3 { margin: 0 0 0.25rem; }
        .forecast-card .temp { font-size: 2.5rem; font-weight: 700; color: var(--temp-color, var(--brand)); line-height: 1.1; margin: 0.5rem 0 0.25rem; }
        .forecast-card-empty .temp { color: var(--pico-muted-color); }
        .forecast-card .pwet { font-size: 0.95rem; color: var(--pwet); margin: 0 0 0.5rem; font-variant-numeric: tabular-nums; }
        .forecast-card .pwet strong { font-weight: 700; }
        .forecast-card .pwet small { color: var(--pico-muted-color); margin-left: 0.35rem; }

        nav.lead-nav { margin: 0 0 1.25rem; padding: 0; }
        nav.lead-nav ul { display: flex; gap: 0.75rem; list-style: none; padding: 0; margin: 0; }
        nav.lead-nav a { text-decoration: none; padding: 0.25rem 0.75rem; border-radius: 4px; background: var(--pico-card-background-color); }
        nav.lead-nav a.active { background: var(--brand); color: white; font-weight: 600; }

        .skill-line { font-style: italic; color: var(--pico-muted-color); }

        table td.num, table th.num { text-align: right; font-variant-numeric: tabular-nums; }
        table td.num.strong { font-weight: 700; color: var(--brand); }
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
                      <li><a href="forecasts-24h.html"{{NavActive(pageId, "forecasts")}}>Forecasts</a></li>
                      <li><a href="dry-window.html"{{NavActive(pageId, "dry-window")}}>Dry window</a></li>
                      <li><a href="skill-temperature.html"{{NavActive(pageId, "skill-temperature")}}>Temp skill</a></li>
                      <li><a href="skill-rainfall.html"{{NavActive(pageId, "skill-rainfall")}}>Rain skill</a></li>
                      <li><a href="models.html"{{NavActive(pageId, "models")}}>Models</a></li>
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

    private static string FmtNullable(double? v, string fmt = "0.00")
        => v.HasValue ? v.Value.ToString(fmt, Ci) : "—";

    /// <summary>
    /// Map a station slug (e.g. <c>ea_bellever_dartmoor</c>) to a display name.
    /// Strips the data-source prefix and title-cases the rest.
    /// </summary>
    private static string PrettyStation(string slug)
    {
        if (string.IsNullOrWhiteSpace(slug)) return slug;
        var trimmed = slug.StartsWith("ea_", StringComparison.Ordinal) ? slug[3..] : slug;
        var parts = trimmed.Split('_', StringSplitOptions.RemoveEmptyEntries);
        return string.Join(' ', parts.Select(p => p.Length == 0 ? p : char.ToUpperInvariant(p[0]) + p[1..]));
    }

    private static string RenderEmptyChart(string title, string message) =>
        $"""<div class="chart-empty-box"><strong>{Escape(title)}</strong><p><em>{Escape(message)}</em></p></div>""";

    /// <summary>
    /// Short URL-safe slug for a station. Used to route <c>skill.html</c> /
    /// <c>skill-{slug}.html</c> and the dry-window equivalents. Returns the station
    /// id unchanged for anything we don't recognise — links would still work but
    /// look less tidy.
    /// </summary>
    internal static string StationSlug(string station) => station switch
    {
        "ea_bellever_dartmoor" => "bellever",
        "ea_princetown" => "princetown",
        "ea_dartmoor_nr_hexworthy" => "hexworthy",
        _ => station,
    };

    /// <summary>
    /// Render a per-station sub-nav for a section. The first station is the canonical
    /// one (its link is <c>{pageBase}.html</c>, matching the top-nav entry); the rest
    /// live at <c>{pageBase}-{slug}.html</c>. The current station is marked active.
    /// </summary>
    internal static string RenderStationSubNav(string pageBase, IReadOnlyList<string> stations, string currentStation)
    {
        if (stations.Count <= 1) return "";   // one station → sub-nav adds nothing

        var items = new StringBuilder();
        for (int i = 0; i < stations.Count; i++)
        {
            var s = stations[i];
            var href = i == 0 ? $"{pageBase}.html" : $"{pageBase}-{StationSlug(s)}.html";
            var cls = s == currentStation ? " class=\"active\"" : "";
            items.Append(Ci, $"""<li><a href="{href}"{cls}>{Escape(PrettyStation(s))}</a></li>""");
        }
        return $"""<nav class="lead-nav"><ul>{items}</ul></nav>""";
    }

    private static string Escape(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    /// <summary>
    /// CSS colour for a displayed temperature. RGB-interpolated between explicit
    /// anchors so the gradient walks directly from indigo (cold) through the brand
    /// purple (mild) to orange and red (warm/hot), without the HSL shortest-arc
    /// trap of passing through magenta at 14°C. Anchors chosen by eye for the
    /// Dartmoor climate: sub-zero is rare but plausible, 12°C is a typical spring
    /// afternoon (the brand colour), and values above 25°C are heatwave territory.
    /// </summary>
    internal static string TemperatureColor(double celsius)
    {
        // (t, r, g, b) anchors — values chosen so cold feels cold without going
        // washed-out and warm feels warm without going crimson.
        (double t, int r, int g, int b)[] anchors =
        {
            (-5,  57,  73, 171),  // indigo #3949ab
            ( 5,  98,  85, 255),  // blue-purple #6255ff
            (12, 124,  77, 255),  // brand purple #7c4dff
            (18, 255, 143,   0),  // orange #ff8f00
            (25, 229,  57,  53),  // red #e53935
            (32, 183,  28,  28),  // deep red #b71c1c
        };
        if (double.IsNaN(celsius)) return "var(--pico-muted-color)";
        if (celsius <= anchors[0].t) return FormatRgb(anchors[0].r, anchors[0].g, anchors[0].b);
        if (celsius >= anchors[^1].t) return FormatRgb(anchors[^1].r, anchors[^1].g, anchors[^1].b);
        for (int i = 0; i < anchors.Length - 1; i++)
        {
            var a = anchors[i];
            var b = anchors[i + 1];
            if (celsius >= a.t && celsius <= b.t)
            {
                var k = (celsius - a.t) / (b.t - a.t);
                int r = (int)Math.Round(a.r + (b.r - a.r) * k);
                int g = (int)Math.Round(a.g + (b.g - a.g) * k);
                int bl = (int)Math.Round(a.b + (b.b - a.b) * k);
                return FormatRgb(r, g, bl);
            }
        }
        var last = anchors[^1];
        return FormatRgb(last.r, last.g, last.b);
    }

    private static string FormatRgb(int r, int g, int b)
        => $"rgb({r.ToString(Ci)} {g.ToString(Ci)} {b.ToString(Ci)})";
}
