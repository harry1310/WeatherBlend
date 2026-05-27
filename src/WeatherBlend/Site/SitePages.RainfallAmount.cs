using System.Globalization;
using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// Phase 3f rainfall_amount surfacing on the rain tab. One per-station
/// section per location where 3f is active (Membury today, possibly
/// Bonehill later). The card sits below the P(wet) section because
/// "is it raining?" (3a / 3c) is the broader question and "how much?"
/// (3f) refines it once the answer to the first one is yes.
///
/// Plan reference: <c>docs/RAINFALL_AMOUNT_3F_PLAN.md</c> §4.1 — headline
/// strip per lead bucket, hourly intensity ribbon, exceedance grid. The
/// optional distribution-explorer modal is deferred.
/// </summary>
public static partial class SitePages
{
    /// <summary>
    /// Render the rainfall_amount section for one station, at the
    /// supplied lead bucket. Returns an empty string when 3f has no
    /// rows for the (station, lead) pair on the current cycle's
    /// freshest predict — keeps a stale prediction tree from
    /// rendering a misleading card.
    /// </summary>
    internal static string RenderRainfallAmountSection(
        SiteInputs input, string stationSlug, int lead)
    {
        // Today's anchor — predict_3f writes one parquet per cycle, but
        // SiteInputs has them already joined; the per-(lead, valid_time)
        // freshest-PredictionMadeAt row wins to mirror the precip card's
        // same-cycle freshness rule.
        var rows = input.RainfallAmountPredictions
            .Where(r => string.Equals(r.TruthStation, stationSlug, StringComparison.Ordinal))
            .Where(r => r.LeadHours == lead)
            .GroupBy(r => r.ValidTimeUtc)
            .Select(g => g.OrderByDescending(r => r.PredictionMadeAtUtc).First())
            .OrderBy(r => r.ValidTimeUtc)
            .ToList();
        if (rows.Count == 0) return "";

        // Restrict to "future of now" hours — the chart wants tomorrow's
        // intensity, not last week's predictions. WindowStartUtc gives
        // historical context for skill; for the predictions card we
        // want anchor-onwards.
        var futureRows = rows.Where(r => r.ValidTimeUtc >= input.GeneratedAtUtc).ToList();
        if (futureRows.Count == 0) return "";

        var headline = RenderRainfallAmountHeadline(futureRows);
        var ribbon = RenderRainfallAmountRibbon(futureRows, lead);
        var exceedance = RenderRainfallAmountExceedance(futureRows);
        return $"""
            <section class="rainfall-amount" data-station="{Escape(stationSlug)}" data-lead="{lead}">
              <h3>How much rain? <small>(Phase 3f, NGBoost-LogNormal · Stage 1 = Phase 3a)</small></h3>
              {headline}
              {ribbon}
              {exceedance}
            </section>
            """;
    }

    /// <summary>
    /// "Median X mm/h • 80% interval A–B • Z% chance of any rain". One
    /// line summarising the day — uses the median (not the mean) for the
    /// point estimate because LogNormal mean is skew-sensitive at high
    /// σ_log; median is what a human reads as "typical". Aggregates
    /// across the lead's 24 hourly rows by taking:
    ///   * sum-mm reading from per-hour medians (≈ daily expected mm,
    ///     conservative — sum-of-medians ≤ median-of-sum on skewed dists)
    ///   * P10 / P90 of the per-hour P(wet ≥ 0.1mm)-weighted mean for the
    ///     "80% interval" rough bound. Honest caveat: this is a
    ///     coarse summary; the chart below shows the true per-hour band.
    ///   * max per-hour PExceed0_1 as "chance of any rain in window".
    /// </summary>
    private static string RenderRainfallAmountHeadline(IReadOnlyList<RainfallAmountPredictionRow> rows)
    {
        var dailyMedianMm = rows.Sum(r => r.MedianMmPerHr);
        var dailyP10Mm    = rows.Sum(r => r.P10MmPerHr);
        var dailyP90Mm    = rows.Sum(r => r.P90MmPerHr);
        var chanceAnyRain = rows.Max(r => r.PExceed0_1);
        return $"""
            <p class="rainfall-headline">
              <strong>Median {dailyMedianMm.ToString("0.0", Ci)} mm</strong> over the day
              · 80% interval {dailyP10Mm.ToString("0.0", Ci)}&ndash;{dailyP90Mm.ToString("0.0", Ci)} mm
              · {(chanceAnyRain * 100).ToString("0", Ci)}% chance of any rain (peak hour)
              <br><small>One-line summary; see the chart for the hour-by-hour distribution.</small>
            </p>
            """;
    }

    /// <summary>
    /// Hourly intensity chart — P50 line, P10–P90 ribbon, P2.5–P97.5
    /// outer ribbon. The ribbons reuse the LineChartSpec.Ribbons field
    /// added 2026-05-27 alongside this card; future distributional
    /// phases (Bonehill 3f later, 4a's predictive distribution) can
    /// reuse the same primitive.
    /// </summary>
    private static string RenderRainfallAmountRibbon(
        IReadOnlyList<RainfallAmountPredictionRow> rows, int lead)
    {
        var p50 = rows.Select(r => (r.ValidTimeUtc.ToOADate(), r.P50MmPerHr)).ToList();
        var p10 = rows.Select(r => (r.ValidTimeUtc.ToOADate(), r.P10MmPerHr)).ToList();
        var p90 = rows.Select(r => (r.ValidTimeUtc.ToOADate(), r.P90MmPerHr)).ToList();
        var p2  = rows.Select(r => (r.ValidTimeUtc.ToOADate(), r.P2_5MmPerHr)).ToList();
        var p97 = rows.Select(r => (r.ValidTimeUtc.ToOADate(), r.P97_5MmPerHr)).ToList();

        var spec = new LineChartSpec
        {
            Title = $"Hourly rainfall intensity — +{lead}h",
            XLabel = "Valid time (UTC)",
            YLabel = "mm / h",
            Height = 220,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
            FormatY = v => v.ToString("0.0", Ci),
            Series = new[]
            {
                new LineSeries("P2.5",  "#bbdefb", p2),
                new LineSeries("P10",   "#90caf9", p10),
                new LineSeries("P50",   "#1565c0", p50),
                new LineSeries("P90",   "#90caf9", p90),
                new LineSeries("P97.5", "#bbdefb", p97),
            },
            Ribbons = new[]
            {
                // Outer band first so the inner band renders on top of it.
                new RibbonSpec("P2.5", "P97.5", "rgba(33, 150, 243, 0.10)"),
                new RibbonSpec("P10",  "P90",   "rgba(33, 150, 243, 0.22)"),
            },
        };
        return $"<figure>{LineChartRenderer.Render(spec)}</figure>";
    }

    /// <summary>
    /// Exceedance grid — compact "P(&gt;1mm), P(&gt;5mm), P(&gt;10mm)" per
    /// hour. Reads each row's PExceed* directly (computed at predict
    /// time so we don't re-run the LogNormal math here).
    /// </summary>
    private static string RenderRainfallAmountExceedance(IReadOnlyList<RainfallAmountPredictionRow> rows)
    {
        var sb = new StringBuilder();
        sb.Append(Ci, $"""
            <details class="hourly-detail rainfall-exceedance">
              <summary>Exceedance probabilities — hourly P(rain &ge; threshold)</summary>
              <figure>
                <table>
                  <thead>
                    <tr>
                      <th>Valid time</th>
                      <th class="num" title="P(rain ≥ 0.1 mm/h)">P&ge;0.1</th>
                      <th class="num" title="P(rain ≥ 1 mm/h) — meaningful rain">P&ge;1</th>
                      <th class="num" title="P(rain ≥ 5 mm/h) — heavy rain">P&ge;5</th>
                      <th class="num" title="P(rain ≥ 10 mm/h) — torrential">P&ge;10</th>
                    </tr>
                  </thead>
                  <tbody>
            """);
        foreach (var r in rows)
        {
            sb.Append(Ci, $"""
                    <tr>
                      <td><time datetime="{r.ValidTimeUtc:yyyy-MM-ddTHH:mm}Z">{r.ValidTimeUtc:MM-dd HH'Z'}</time></td>
                      <td class="num" style="color: {ExceedanceColor(r.PExceed0_1)}">{(r.PExceed0_1 * 100).ToString("0", Ci)}%</td>
                      <td class="num" style="color: {ExceedanceColor(r.PExceed1)}">{(r.PExceed1 * 100).ToString("0", Ci)}%</td>
                      <td class="num" style="color: {ExceedanceColor(r.PExceed5)}">{(r.PExceed5 * 100).ToString("0", Ci)}%</td>
                      <td class="num" style="color: {ExceedanceColor(r.PExceed10)}">{(r.PExceed10 * 100).ToString("0", Ci)}%</td>
                    </tr>
                """);
        }
        sb.Append("</tbody></table></figure></details>");
        return sb.ToString();
    }

    /// <summary>Green → orange → red as the probability of a given mm/h
    /// threshold climbs. Matches the PrecipProbColor scale used on the
    /// home P(wet) chip — palette stays consistent across the rain tab.</summary>
    private static string ExceedanceColor(double p) =>
        p >= 0.50 ? "#c62828"
      : p >= 0.30 ? "#ef6c00"
      : p >= 0.15 ? "#f9a825"
      :             "#2e7d32";
}
