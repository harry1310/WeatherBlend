using System.Globalization;
using System.Text;

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
