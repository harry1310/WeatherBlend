using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;

namespace WeatherBlend.Site;

/// <summary>
/// One series on a <see cref="LineChartSpec"/>. Points are plotted in the order given;
/// X values should be monotonically increasing for sensible line rendering.
///
/// <paramref name="PointsOnly"/> renders markers only, no connecting polyline — used
/// for discrete-valued truth series (e.g. the 0/1 wet-hour indicator) where a line
/// between 0 and 1 would imply non-existent intermediate values.
/// </summary>
public sealed record LineSeries(
    string Name,
    string Color,
    IReadOnlyList<(double X, double Y)> Points,
    bool PointsOnly = false);

public sealed record LineChartSpec
{
    public required string Title { get; init; }
    public required string XLabel { get; init; }
    public required string YLabel { get; init; }
    public required IReadOnlyList<LineSeries> Series { get; init; }

    public int Width { get; init; } = 720;
    public int Height { get; init; } = 320;
    public int PadLeft { get; init; } = 56;
    public int PadRight { get; init; } = 16;
    public int PadTop { get; init; } = 32;
    public int PadBottom { get; init; } = 48;

    /// <summary>Formats an X value for the axis label (e.g. DateTime.FromOADate(...)).</summary>
    public Func<double, string> FormatX { get; init; } = v => v.ToString("0.##", CultureInfo.InvariantCulture);

    /// <summary>Formats a Y value for the axis label.</summary>
    public Func<double, string> FormatY { get; init; } = v => v.ToString("0.##", CultureInfo.InvariantCulture);

    /// <summary>
    /// Vertical background bands, each (X start, X end) in OADate. Rendered behind
    /// the data via the Chart.js annotation plugin — used on rainfall skill charts
    /// to highlight observed wet hours instead of plotting them as 0/1 dots.
    /// X values are OADate to match <see cref="LineSeries.Points"/>; the JS layer
    /// converts both to Unix ms in the same step.
    /// </summary>
    public IReadOnlyList<(double XStart, double XEnd)> Bands { get; init; }
        = Array.Empty<(double, double)>();

    /// <summary>CSS colour string used to fill <see cref="Bands"/>. Defaults to a
    /// pale blue that reads as "wet" without competing with the line series.</summary>
    public string BandColor { get; init; } = "rgba(33, 150, 243, 0.18)";

    /// <summary>Optional vertical reference line at this X (OADate). Rendered
    /// dashed via the annotation plugin. Used to mark "today" on skill charts so
    /// a reader can see at a glance which side of the chart is the future.</summary>
    public double? TodayLineX { get; init; }
}

public static class LineChartRenderer
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

    public static string Render(LineChartSpec spec)
    {
        var allPoints = spec.Series.SelectMany(s => s.Points).ToList();
        if (allPoints.Count == 0)
        {
            return EmptyChart(spec);
        }

        double xMin = allPoints.Min(p => p.X);
        double xMax = allPoints.Max(p => p.X);
        double yMin = allPoints.Min(p => p.Y);
        double yMax = allPoints.Max(p => p.Y);

        if (xMin == xMax) { xMax = xMin + 1; }
        if (yMin == yMax) { yMax = yMin + 1; }

        // 5% padding on Y axis so points don't kiss the frame.
        var yPad = (yMax - yMin) * 0.05;
        yMin -= yPad;
        yMax += yPad;

        int plotX = spec.PadLeft;
        int plotY = spec.PadTop;
        int plotW = spec.Width - spec.PadLeft - spec.PadRight;
        int plotH = spec.Height - spec.PadTop - spec.PadBottom;

        double ScaleX(double x) => plotX + (x - xMin) / (xMax - xMin) * plotW;
        double ScaleY(double y) => plotY + plotH - (y - yMin) / (yMax - yMin) * plotH;

        var sb = new StringBuilder();
        sb.Append(Ci, $"<svg viewBox=\"0 0 {spec.Width} {spec.Height}\" xmlns=\"http://www.w3.org/2000/svg\" class=\"chart\" role=\"img\" aria-label=\"{Escape(spec.Title)}\">");

        // Title
        sb.Append(Ci, $"<text x=\"{spec.Width / 2}\" y=\"20\" text-anchor=\"middle\" class=\"chart-title\">{Escape(spec.Title)}</text>");

        // Gridlines + Y ticks (5 gridlines).
        for (int i = 0; i <= 5; i++)
        {
            double y = plotY + plotH * i / 5.0;
            double yVal = yMax - (yMax - yMin) * i / 5.0;
            sb.Append(Ci, $"<line x1=\"{plotX:0.#}\" y1=\"{y:0.#}\" x2=\"{plotX + plotW:0.#}\" y2=\"{y:0.#}\" class=\"chart-grid\" />");
            sb.Append(Ci, $"<text x=\"{plotX - 6:0.#}\" y=\"{y + 4:0.#}\" text-anchor=\"end\" class=\"chart-tick\">{Escape(spec.FormatY(yVal))}</text>");
        }

        // X ticks (6 evenly spaced).
        for (int i = 0; i <= 5; i++)
        {
            double x = plotX + plotW * i / 5.0;
            double xVal = xMin + (xMax - xMin) * i / 5.0;
            sb.Append(Ci, $"<line x1=\"{x:0.#}\" y1=\"{plotY + plotH:0.#}\" x2=\"{x:0.#}\" y2=\"{plotY + plotH + 4:0.#}\" class=\"chart-tick-mark\" />");
            sb.Append(Ci, $"<text x=\"{x:0.#}\" y=\"{plotY + plotH + 18:0.#}\" text-anchor=\"middle\" class=\"chart-tick\">{Escape(spec.FormatX(xVal))}</text>");
        }

        // Frame
        sb.Append(Ci, $"<rect x=\"{plotX}\" y=\"{plotY}\" width=\"{plotW}\" height=\"{plotH}\" class=\"chart-frame\" />");

        // Axis labels
        sb.Append(Ci, $"<text x=\"{spec.Width / 2}\" y=\"{spec.Height - 6}\" text-anchor=\"middle\" class=\"chart-axis-label\">{Escape(spec.XLabel)}</text>");
        sb.Append(Ci, $"<text x=\"14\" y=\"{spec.PadTop + plotH / 2}\" text-anchor=\"middle\" class=\"chart-axis-label\" transform=\"rotate(-90, 14, {spec.PadTop + plotH / 2})\">{Escape(spec.YLabel)}</text>");

        // Series — polylines, plus point markers on sparse series only. Dense
        // series (e.g. hourly truth over 30 days) get just the line; the dots
        // become a smear at that density and ruin readability. Points-only
        // series render as dots at every point regardless of density, with a
        // smaller radius so they stay legible when the series is hourly.
        const int MarkerThreshold = 30;
        foreach (var s in spec.Series)
        {
            if (s.Points.Count == 0) continue;

            if (s.PointsOnly)
            {
                foreach (var p in s.Points)
                {
                    sb.Append(Ci, $"<circle cx=\"{ScaleX(p.X):0.#}\" cy=\"{ScaleY(p.Y):0.#}\" r=\"2\" fill=\"{s.Color}\" />");
                }
                continue;
            }

            var pathPoints = string.Join(" ", s.Points.Select(p => string.Create(Ci, $"{ScaleX(p.X):0.#},{ScaleY(p.Y):0.#}")));
            sb.Append(Ci, $"<polyline points=\"{pathPoints}\" fill=\"none\" stroke=\"{s.Color}\" stroke-width=\"1.75\" class=\"chart-line\" />");

            if (s.Points.Count <= MarkerThreshold)
            {
                foreach (var p in s.Points)
                {
                    sb.Append(Ci, $"<circle cx=\"{ScaleX(p.X):0.#}\" cy=\"{ScaleY(p.Y):0.#}\" r=\"3\" fill=\"{s.Color}\" />");
                }
            }
        }

        // Legend (top-right, inside plot area). Wraps to two columns once there are
        // too many series to stack vertically without eating the chart.
        const int LegendRowHeight = 14;
        const int LegendColWidth = 110;
        int legendCols = spec.Series.Count > 5 ? 2 : 1;
        int legendRows = (spec.Series.Count + legendCols - 1) / legendCols;
        int legendRight = plotX + plotW - 6;
        int legendTop = plotY + 8;
        for (int i = 0; i < spec.Series.Count; i++)
        {
            var s = spec.Series[i];
            int col = i / legendRows;
            int row = i % legendRows;
            int cellRight = legendRight - (legendCols - 1 - col) * LegendColWidth;
            int y = legendTop + row * LegendRowHeight;
            sb.Append(Ci, $"<rect x=\"{cellRight - LegendColWidth + 4}\" y=\"{y - 8}\" width=\"10\" height=\"10\" fill=\"{s.Color}\" />");
            sb.Append(Ci, $"<text x=\"{cellRight - LegendColWidth + 18}\" y=\"{y}\" class=\"chart-legend\">{Escape(s.Name)}</text>");
        }

        sb.Append("</svg>");
        return sb.ToString();
    }

    /// <summary>
    /// Render a chart as a Chart.js v4 canvas element with an embedded JSON config.
    /// The runtime in <c>chart.js</c> (see <see cref="SitePages.ChartScript"/>) reads
    /// <c>data-cjs</c>, converts OADate X values to Unix ms, and builds a Chart with
    /// pre-baked tick formatters driven by the metadata we emit here. Falls back to
    /// the static <see cref="EmptyChart"/> SVG when there are no points.
    ///
    /// We probe the spec's <see cref="LineChartSpec.FormatX"/> and
    /// <see cref="LineChartSpec.FormatY"/> lambdas with representative values so the
    /// JS side can replicate them without parsing C# format strings — keeps the
    /// JS layer free of culture / format-pattern logic.
    /// </summary>
    public static string RenderChartJs(LineChartSpec spec)
    {
        if (spec.Series.All(s => s.Points.Count == 0))
            return EmptyChart(spec);

        var xKind = ProbeXKind(spec.FormatX);
        var (yDec, ySuffix, yTrim) = ProbeYFormat(spec.FormatY);

        var datasets = new List<object>();
        foreach (var s in spec.Series)
        {
            if (s.Points.Count == 0) continue;
            // Compact wire format — [x, y] pairs only. The JS side converts X (OADate)
            // to Unix ms; pre-formatted labels aren't needed because Chart.js calls
            // our tick-format callback for tooltip values too (via the same yFormat
            // metadata we pass below). Non-finite values (NaN / ±Infinity) get
            // dropped here — JSON has no representation for them, so leaving them
            // in throws at JsonSerializer time and crashes the whole render.
            // Chart.js handles the resulting gaps via spanGaps:true.
            var pts = s.Points
                .Where(p => double.IsFinite(p.X) && double.IsFinite(p.Y))
                .Select(p => new[] { p.X, p.Y })
                .ToArray();
            if (pts.Length == 0) continue;
            datasets.Add(new
            {
                label = s.Name,
                color = s.Color,
                discrete = s.PointsOnly,
                points = pts,
            });
        }

        // Annotation plugin payload — only emitted when non-empty so charts that
        // don't need bands or a "today" line ship a tighter JSON config. Same
        // finite-only guard on band edges and the today line.
        object? annotations = null;
        var finiteBands = spec.Bands
            .Where(b => double.IsFinite(b.XStart) && double.IsFinite(b.XEnd))
            .Select(b => new[] { b.XStart, b.XEnd })
            .ToArray();
        var finiteTodayX = (spec.TodayLineX is { } t && double.IsFinite(t)) ? (double?)t : null;
        if (finiteBands.Length > 0 || finiteTodayX.HasValue)
        {
            annotations = new
            {
                bands = finiteBands,
                bandColor = spec.BandColor,
                todayX = finiteTodayX,
            };
        }

        var cfg = new
        {
            title = spec.Title,
            xLabel = spec.XLabel,
            yLabel = spec.YLabel,
            xKind,
            yDec,
            ySuffix,
            yTrim,
            datasets,
            annotations,
        };

        var json = JsonSerializer.Serialize(cfg);
        var encoded = WebUtility.HtmlEncode(json);
        return string.Create(Ci, $"""
            <div class="chart-cjs" style="height: {spec.Height}px;">
              <canvas data-cjs="{encoded}"></canvas>
            </div>
            """);
    }

    /// <summary>
    /// Map the C# X-axis formatter to one of the two date layouts the JS side knows
    /// about: <c>"date"</c> ("MM-dd") or <c>"datetime"</c> ("MM-dd HHZ"). The probe
    /// uses an OADate at 12:00 UTC; presence of <c>'Z'</c> in the formatted string
    /// is the discriminator. If the formatter is something else entirely (e.g. a
    /// raw numeric format), we still return <c>"date"</c> — the JS side renders a
    /// reasonable fallback rather than failing.
    /// </summary>
    internal static string ProbeXKind(Func<double, string> formatX)
    {
        var probe = new DateTime(2024, 4, 26, 12, 0, 0, DateTimeKind.Utc).ToOADate();
        var s = formatX(probe);
        return s.Contains('Z') ? "datetime" : "date";
    }

    /// <summary>
    /// Probe the Y-axis formatter for the metadata the JS side needs:
    /// number of decimal places, optional suffix (e.g. <c>"°"</c>, <c>"%"</c>),
    /// and whether trailing zeros should be trimmed (the C# <c>"0.#"</c> idiom).
    /// </summary>
    internal static (int Dec, string Suffix, bool Trim) ProbeYFormat(Func<double, string> formatY)
    {
        // Two probes: an integer-valued one (12.0) tells us whether trailing zeros
        // get trimmed; a fractional one (3.456) tells us how many decimals are kept.
        var s12 = formatY(12.0);
        var sFrac = formatY(3.456);

        static string TrailingNonNumeric(string v)
        {
            // Walk left over trailing non-numeric chars (e.g. "°", "°C", "%", " mm")
            // and stop at the first digit/dot/sign — that boundary marks where the
            // numeric body ends. Substring(i) is then the suffix.
            int i = v.Length;
            while (i > 0 && !(char.IsDigit(v[i - 1]) || v[i - 1] == '.' || v[i - 1] == '-' || v[i - 1] == '+'))
                i--;
            return v.Substring(i);
        }

        var suffix = TrailingNonNumeric(s12);
        var body12 = s12.Substring(0, s12.Length - suffix.Length);
        var bodyFrac = sFrac.Substring(0, sFrac.Length - suffix.Length);

        var dot = bodyFrac.IndexOf('.');
        var dec = dot < 0 ? 0 : (bodyFrac.Length - dot - 1);
        var trim = !body12.Contains('.');

        return (dec, suffix, trim);
    }

    private static string EmptyChart(LineChartSpec spec)
    {
        return string.Create(Ci, $"""
            <svg viewBox="0 0 {spec.Width} {spec.Height}" xmlns="http://www.w3.org/2000/svg" class="chart" role="img" aria-label="{Escape(spec.Title)}">
              <text x="{spec.Width / 2}" y="{spec.Height / 2}" text-anchor="middle" class="chart-empty">No data in window</text>
            </svg>
            """);
    }

    private static string Escape(string s)
    {
        return s.Replace("&", "&amp;")
                .Replace("<", "&lt;")
                .Replace(">", "&gt;")
                .Replace("\"", "&quot;");
    }
}
