using System.Globalization;
using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins the SVG → Chart.js bridging logic in <see cref="LineChartRenderer"/>:
/// the X/Y format probes (which had a sign-inverted bug shipping wrong tick
/// labels until 2026-04-28), the JSON payload shape Chart.js consumes, and
/// the empty-series fallback to a static SVG. Tests exercise both the probes
/// directly and the rendered JSON so we'd catch a regression in either layer.
/// </summary>
public class SvgChartTests
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

    // -------- ProbeXKind --------

    [Fact]
    public void ProbeXKind_returns_date_for_MM_dd_format()
    {
        // MM-dd is the default temperature-skill X formatter — date only, no time.
        Func<double, string> fmt = v => DateTime.FromOADate(v).ToString("MM-dd", Ci);
        LineChartRenderer.ProbeXKind(fmt).Should().Be("date");
    }

    [Fact]
    public void ProbeXKind_returns_datetime_for_format_with_Z_suffix()
    {
        // The forecasts page uses "MM-dd HH'Z'" for hourly grids — must round-trip
        // to "datetime" so the JS-side formatter emits the hour component.
        Func<double, string> fmt = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci);
        LineChartRenderer.ProbeXKind(fmt).Should().Be("datetime");
    }

    // -------- ProbeYFormat --------

    [Theory]
    [InlineData("0.#",  1, true,  "")]   // up-to-1-decimal, trim trailing zeros
    [InlineData("0.0",  1, false, "")]   // always 1 decimal
    [InlineData("0.00", 2, false, "")]   // always 2 decimals
    [InlineData("0",    0, true,  "")]   // integer, no decimal point at all
    public void ProbeYFormat_extracts_decimals_and_trim_flag(string fmt, int expectDec, bool expectTrim, string expectSuffix)
    {
        Func<double, string> formatter = v => v.ToString(fmt, Ci);
        var (dec, suffix, trim) = LineChartRenderer.ProbeYFormat(formatter);

        dec.Should().Be(expectDec);
        trim.Should().Be(expectTrim);
        suffix.Should().Be(expectSuffix);
    }

    [Fact]
    public void ProbeYFormat_extracts_degree_suffix()
    {
        // Pre-fix bug: TrailingNonNumeric walked the wrong direction and returned
        // an empty suffix here. Tick labels rendered as "5.40" instead of "5.4°".
        Func<double, string> fmt = v => v.ToString("0.#", Ci) + "°";
        var (dec, suffix, trim) = LineChartRenderer.ProbeYFormat(fmt);

        dec.Should().Be(1);
        suffix.Should().Be("°");
        trim.Should().BeTrue();
    }

    [Fact]
    public void ProbeYFormat_extracts_multichar_suffix()
    {
        // Suffix can be more than one char — e.g. "°C" or " mm". The probe walks
        // backwards over every trailing non-numeric char, not just the last one.
        Func<double, string> fmt = v => v.ToString("0.0", Ci) + " mm";
        var (_, suffix, _) = LineChartRenderer.ProbeYFormat(fmt);

        suffix.Should().Be(" mm");
    }

    [Fact]
    public void ProbeYFormat_handles_signed_format_without_collapsing_to_suffix()
    {
        // "+0.00;-0.00;0.00" produces "+12.00" / "-3.46" — leading sign mustn't be
        // mistaken for a trailing suffix (TrailingNonNumeric counts +/- as numeric).
        Func<double, string> fmt = v => v.ToString("+0.00;-0.00;0.00", Ci);
        var (dec, suffix, _) = LineChartRenderer.ProbeYFormat(fmt);

        dec.Should().Be(2);
        suffix.Should().Be("");
    }

    // -------- RenderChartJs JSON shape --------

    [Fact]
    public void RenderChartJs_emits_data_cjs_attribute_on_canvas()
    {
        var spec = SimpleSpec();
        var html = LineChartRenderer.RenderChartJs(spec);

        html.Should().Contain("<canvas data-cjs=\"");
        html.Should().Contain("class=\"chart-cjs\"");
    }

    [Fact]
    public void RenderChartJs_returns_static_SVG_when_every_series_is_empty()
    {
        var spec = new LineChartSpec
        {
            Title = "Empty",
            XLabel = "x",
            YLabel = "y",
            Series = new[] { new LineSeries("nothing", "#000", Array.Empty<(double, double)>()) },
        };
        var html = LineChartRenderer.RenderChartJs(spec);

        // Empty payload would crash the JS bootstrap; fall back to the SVG empty
        // state so the page still renders something readable.
        html.Should().Contain("<svg");
        html.Should().Contain("No data in window");
        html.Should().NotContain("data-cjs");
    }

    [Fact]
    public void RenderChartJs_omits_annotations_block_when_no_bands_or_today_set()
    {
        // Most charts pass no annotations — the JSON payload must reflect that
        // (annotations: null) so the JS bootstrap's null check can short-circuit.
        var spec = SimpleSpec();
        var html = LineChartRenderer.RenderChartJs(spec);

        // Decode the data-cjs payload and inspect.
        var cfg = ExtractCjsConfig(html);
        cfg.GetProperty("annotations").ValueKind.Should().Be(System.Text.Json.JsonValueKind.Null);
    }

    [Fact]
    public void RenderChartJs_emits_bands_in_annotation_payload()
    {
        var spec = SimpleSpec() with
        {
            Bands = new[]
            {
                (45_000.0, 45_000.5),
                (45_001.0, 45_001.25),
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        var bands = cfg.GetProperty("annotations").GetProperty("bands");
        bands.GetArrayLength().Should().Be(2);
        bands[0][0].GetDouble().Should().Be(45_000.0);
        bands[0][1].GetDouble().Should().Be(45_000.5);
    }

    [Fact]
    public void RenderChartJs_emits_today_line_in_annotation_payload()
    {
        var todayOa = new DateTime(2026, 4, 28, 12, 0, 0, DateTimeKind.Utc).ToOADate();
        var spec = SimpleSpec() with { TodayLineX = todayOa };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        cfg.GetProperty("annotations").GetProperty("todayX").GetDouble()
            .Should().BeApproximately(todayOa, 1e-9);
    }

    // -------- Non-finite-value filtering --------
    //
    // JSON has no representation for NaN / ±Infinity, so any non-finite value
    // reaching JsonSerializer crashes the entire render. RenderChartJs must
    // strip those before serialisation; Chart.js spans the resulting gap.

    [Fact]
    public void RenderChartJs_drops_NaN_y_value_from_a_point_array()
    {
        var spec = new LineChartSpec
        {
            Title = "with-nan", XLabel = "x", YLabel = "y",
            Series = new[]
            {
                new LineSeries("a", "#000",
                    new (double, double)[] { (1, 1.0), (2, double.NaN), (3, 3.0) }),
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        var pts = cfg.GetProperty("datasets")[0].GetProperty("points");
        pts.GetArrayLength().Should().Be(2,
            "NaN-y point must be dropped; remaining x=1 and x=3 points kept");
    }

    [Fact]
    public void RenderChartJs_drops_PositiveInfinity_y_value()
    {
        var spec = new LineChartSpec
        {
            Title = "inf", XLabel = "x", YLabel = "y",
            Series = new[]
            {
                new LineSeries("a", "#000",
                    new (double, double)[] { (1, double.PositiveInfinity), (2, 5.0) }),
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        cfg.GetProperty("datasets")[0].GetProperty("points").GetArrayLength().Should().Be(1);
    }

    [Fact]
    public void RenderChartJs_drops_NaN_x_value_from_a_point_array()
    {
        // X-axis non-finite is just as serialiser-fatal as Y-axis.
        var spec = new LineChartSpec
        {
            Title = "nan-x", XLabel = "x", YLabel = "y",
            Series = new[]
            {
                new LineSeries("a", "#000",
                    new (double, double)[] { (double.NaN, 1.0), (2.0, 2.0) }),
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        cfg.GetProperty("datasets")[0].GetProperty("points").GetArrayLength().Should().Be(1);
    }

    [Fact]
    public void RenderChartJs_skips_dataset_entirely_when_every_point_is_non_finite()
    {
        // If filtering wipes every point in a series, drop the dataset itself —
        // an empty `points` array is still a valid JSON payload but Chart.js
        // would render an empty legend entry.
        var spec = new LineChartSpec
        {
            Title = "all-nan", XLabel = "x", YLabel = "y",
            Series = new[]
            {
                new LineSeries("doomed", "#000",
                    new (double, double)[] { (1, double.NaN), (2, double.NaN) }),
                new LineSeries("good", "#fff",
                    new (double, double)[] { (1, 1.0), (2, 2.0) }),
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        var ds = cfg.GetProperty("datasets");
        ds.GetArrayLength().Should().Be(1);
        ds[0].GetProperty("label").GetString().Should().Be("good");
    }

    [Fact]
    public void RenderChartJs_drops_non_finite_band_edges_but_keeps_finite_ones()
    {
        var spec = new LineChartSpec
        {
            Title = "bands", XLabel = "x", YLabel = "y",
            Series = new[] { new LineSeries("a", "#000", new (double, double)[] { (1, 1) }) },
            Bands = new[]
            {
                (45_000.0, 45_000.5),                          // finite — keep
                (double.NaN, 45_001.0),                        // start NaN — drop
                (45_002.0, double.PositiveInfinity),           // end inf — drop
                (45_003.0, 45_003.5),                          // finite — keep
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        cfg.GetProperty("annotations").GetProperty("bands").GetArrayLength().Should().Be(2);
    }

    [Fact]
    public void RenderChartJs_omits_today_line_when_value_is_non_finite()
    {
        // A NaN today-line (computed somewhere upstream from a divide-by-zero
        // or similar) must NOT crash the serialiser. Treat it as "no today line"
        // and skip the annotation.
        var spec = new LineChartSpec
        {
            Title = "today-nan", XLabel = "x", YLabel = "y",
            Series = new[] { new LineSeries("a", "#000", new (double, double)[] { (1, 1) }) },
            TodayLineX = double.NaN,
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        // No bands, no finite todayX → annotations block omitted entirely.
        cfg.GetProperty("annotations").ValueKind.Should().Be(System.Text.Json.JsonValueKind.Null);
    }

    [Fact]
    public void RenderChartJs_packs_every_dataset_with_color_and_discrete_flag()
    {
        var spec = new LineChartSpec
        {
            Title = "Mix",
            XLabel = "x",
            YLabel = "y",
            Series = new[]
            {
                new LineSeries("line", "#aabbcc",
                    new (double, double)[] { (1, 1.5), (2, 2.5) }),
                new LineSeries("dots", "#112233",
                    new (double, double)[] { (1, 0), (2, 1) }, PointsOnly: true),
            },
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        var cfg = ExtractCjsConfig(html);

        var ds = cfg.GetProperty("datasets");
        ds.GetArrayLength().Should().Be(2);
        ds[0].GetProperty("label").GetString().Should().Be("line");
        ds[0].GetProperty("color").GetString().Should().Be("#aabbcc");
        ds[0].GetProperty("discrete").GetBoolean().Should().BeFalse();
        ds[1].GetProperty("discrete").GetBoolean().Should().BeTrue();
    }

    private static LineChartSpec SimpleSpec() => new()
    {
        Title = "test",
        XLabel = "t",
        YLabel = "y",
        Series = new[]
        {
            new LineSeries("a", "#7c4dff",
                new (double, double)[] { (45_000.0, 1.0), (45_000.5, 2.0) }),
        },
        FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
        FormatY = v => v.ToString("0.0", Ci),
    };

    /// <summary>
    /// Pull the JSON config out of the <c>data-cjs</c> attribute on the rendered
    /// canvas. The attribute value is HTML-encoded so we decode it before parsing.
    /// </summary>
    private static System.Text.Json.JsonElement ExtractCjsConfig(string html)
    {
        var marker = "data-cjs=\"";
        var i = html.IndexOf(marker, StringComparison.Ordinal);
        if (i < 0) throw new InvalidOperationException("no data-cjs attribute in html");
        i += marker.Length;
        var end = html.IndexOf('"', i);
        var raw = html[i..end];
        var json = System.Net.WebUtility.HtmlDecode(raw);
        return System.Text.Json.JsonDocument.Parse(json).RootElement.Clone();
    }
}
