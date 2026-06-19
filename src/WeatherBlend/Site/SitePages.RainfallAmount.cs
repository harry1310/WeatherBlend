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
        SiteInputs input, string stationSlug, int lead, double xMin, double xMax)
    {
        // Today's anchor — predict_3f writes one parquet per cycle, but
        // SiteInputs has them already joined; the per-(lead, valid_time)
        // freshest-PredictionMadeAt row wins to mirror the precip card's
        // same-cycle freshness rule.
        var rows = input.RainfallAmountPredictions
            .Where(r => string.Equals(r.TruthStation, stationSlug, StringComparison.Ordinal))
            .Where(r => r.LeadHours == lead)
            .FreshestPerValid(r => r.ValidTimeUtc, r => r.PredictionMadeAtUtc)
            .OrderBy(r => r.ValidTimeUtc)
            .ToList();
        if (rows.Count == 0) return "";

        // The HEADLINE + exceedance table summarise what's COMING — they
        // stay future-of-now ("0.4mm total" must not count this morning's
        // already-fallen rain). The section also stays absent when a stale
        // prediction tree has no future rows at all.
        var futureRows = rows.Where(r => r.ValidTimeUtc >= input.GeneratedAtUtc).ToList();
        if (futureRows.Count == 0) return "";

        // The CHART shows the page's shared window INCLUDING the recent
        // past (2026-06-11, Harry): every other chart on the rain page
        // starts at xMin (now − ForecastChartHistoryDays) and this one
        // started at "now", leaving its left third blank and the day's
        // story truncated. Same x-window, freshest-cycle rows.
        var chartRows = rows.Where(r => r.ValidTimeUtc.ToOADate() >= xMin).ToList();

        var headline = RenderRainfallAmountHeadline(futureRows);
        var ribbon = RenderRainfallAmountRibbon(chartRows, lead, xMin, xMax, input.GeneratedAtUtc.ToOADate());
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
    /// "Village" composite rainfall section — the location's per-gauge 3f
    /// forecasts combined with the on-site calibration weights
    /// (<see cref="VillageRainSpec"/>; Membury: NNLS vs 42 months of the
    /// manual village gauge, true-rainfall study 2026-06-10). Rendered
    /// once per rain page, after the per-station sections.
    ///
    /// Compositing: per common valid hour, every per-hour quantity
    /// (median + quantile band edges) is the weighted sum across gauges.
    /// Weighted-summing QUANTILES is exact only when the gauges co-vary
    /// perfectly (comonotonicity); at 6-7 km apart their daily totals
    /// correlate at r ≈ 0.95+, so the approximation is good and honest
    /// for a display band. The exceedance grid is deliberately NOT
    /// rendered here — P(weighted sum ≥ T) does not combine linearly
    /// from per-gauge exceedance probabilities, and an almost-right
    /// table is worse than none. The headline's "chance of any rain"
    /// uses the weight-normalised average of the gauges' P≥0.1 — rain
    /// OCCURRENCE is effectively common across a 7 km triangle, so a
    /// consensus probability is fair where an amount-threshold one isn't.
    ///
    /// ALL weighted gauges must have rows at this lead — a composite
    /// missing a 0.5-weight member would silently read ~half the real
    /// rate. Absent gauge = no section (same empty-string contract as
    /// the per-station card).
    /// </summary>
    internal static string RenderVillageRainfallSection(
        SiteInputs input, LocationDescriptor location, int lead, double xMin, double xMax)
    {
        var cal = location.VillageRain;
        if (cal is null || cal.WeightsBySlug.Count == 0) return "";

        // Same freshest-per-(lead, valid) selection as the per-station
        // section, per weighted gauge. Window = the page's shared x-axis
        // (xMin includes the recent past — 2026-06-11, same fix as the
        // per-station chart); the headline below re-restricts to
        // future-of-now, and the all-gauges-required guard keys on having
        // FUTURE rows so a stale tree still renders nothing.
        var byStation = new Dictionary<string, Dictionary<DateTime, RainfallAmountPredictionRow>>(StringComparer.Ordinal);
        foreach (var slug in cal.WeightsBySlug.Keys)
        {
            var rows = input.RainfallAmountPredictions
                .Where(r => string.Equals(r.TruthStation, slug, StringComparison.Ordinal))
                .Where(r => r.LeadHours == lead)
                .FreshestPerValid(r => r.ValidTimeUtc, r => r.PredictionMadeAtUtc)
                .Where(r => r.ValidTimeUtc.ToOADate() >= xMin)
                .ToDictionary(r => r.ValidTimeUtc);
            if (!rows.Keys.Any(v => v >= input.GeneratedAtUtc)) return "";
            byStation[slug] = rows;
        }

        var commonValids = byStation.Values
            .Select(d => (IEnumerable<DateTime>)d.Keys)
            .Aggregate((a, b) => a.Intersect(b))
            .OrderBy(v => v)
            .ToList();
        if (commonValids.Count == 0) return "";

        var weightSum = cal.WeightsBySlug.Values.Sum();
        var template = byStation.First().Value[commonValids[0]];
        var composite = commonValids.Select(v =>
        {
            double W(Func<RainfallAmountPredictionRow, double> f) =>
                cal.WeightsBySlug.Sum(kv => kv.Value * f(byStation[kv.Key][v]));
            // Occurrence consensus (NOT a weighted amount — see doc note).
            double Consensus(Func<RainfallAmountPredictionRow, double> f) => W(f) / weightSum;
            return new RainfallAmountPredictionRow
            {
                LocationName = template.LocationName,
                TruthStation = location.Name + "_village",
                ModelVersion = template.ModelVersion,
                Precip3aVersion = template.Precip3aVersion,
                PredictionMadeAtUtc = byStation.Values.Min(d => d[v].PredictionMadeAtUtc),
                ValidTimeUtc = v,
                LeadHours = lead,
                // Distribution parameters don't compose across gauges —
                // the composite is quantile-level only. Carry the
                // occurrence consensus as Pi; NaN the per-gauge LogNormal
                // params so any future consumer that needs them fails
                // loudly rather than reading one gauge's values.
                Pi = Consensus(r => r.Pi),
                MuLog = double.NaN,
                SigmaLog = double.NaN,
                MeanMmPerHr   = W(r => r.MeanMmPerHr),
                MedianMmPerHr = W(r => r.MedianMmPerHr),
                P2_5MmPerHr   = W(r => r.P2_5MmPerHr),
                P10MmPerHr    = W(r => r.P10MmPerHr),
                P50MmPerHr    = W(r => r.P50MmPerHr),
                P90MmPerHr    = W(r => r.P90MmPerHr),
                P97_5MmPerHr  = W(r => r.P97_5MmPerHr),
                PExceed0_1 = Consensus(r => r.PExceed0_1),
                PExceed1   = Consensus(r => r.PExceed1),
                PExceed5   = Consensus(r => r.PExceed5),
                PExceed10  = Consensus(r => r.PExceed10),
            };
        }).ToList();

        var weightsLine = string.Join(" + ", cal.WeightsBySlug
            .OrderByDescending(kv => kv.Value)
            .Select(kv => $"{kv.Value.ToString("0.00", Ci)}·{Escape(PrettyStation(kv.Key))}"));
        // Headline summarises what's COMING; the ribbon shows the full
        // shared window incl. the recent past (mirrors the per-station
        // section's futureRows/chartRows split).
        var futureComposite = composite.Where(r => r.ValidTimeUtc >= input.GeneratedAtUtc).ToList();
        if (futureComposite.Count == 0) return "";
        var headline = RenderRainfallAmountHeadline(futureComposite);
        var ribbon = RenderRainfallAmountRibbon(composite, lead, xMin, xMax, input.GeneratedAtUtc.ToOADate());
        // Per-day 09Z→09Z village totals (Harry 2026-06-18): the village's manual
        // gauge is read as a 09:00Z rain-day, so this sums the composite hourly
        // medians into the same 09Z→09Z windows the truth log uses — the at-this-
        // lead predicted daily total to set beside each recorded day.
        var dailyTable = RenderVillage9to9DailyTable(composite, lead);
        return $"""
            <section class="rainfall-amount village" data-station="{Escape(location.Name)}_village" data-lead="{lead}">
              <h3>{Escape(cal.Label)} — best estimate of rain at the village
                <small>(EA gauge forecasts × on-site calibration)</small></h3>
              <p><small>Weighted composite of the three gauge forecasts above:
                {weightsLine}, fitted {Escape(cal.FittedOn)} against the village's own
                daily gauge (42 months). The village receives ~7% less rain than the
                raw gauge average. Band edges assume the gauges co-vary
                (they correlate at r&nbsp;≈&nbsp;0.95); no exceedance table here because
                threshold probabilities don't combine across gauges.</small></p>
              {headline}
              {ribbon}
              {dailyTable}
            </section>
            """;
    }

    /// <summary>Minimum of the 24 hourly predictions a 09Z→09Z window must have
    /// before its total is shown — a part-covered edge day would under-read and
    /// read as a falsely dry day. 22–23h days still show, flagged with the count.</summary>
    private const int MinHoursForDay = 22;

    /// <summary>
    /// Per-day 09Z→09Z predicted village rain totals — the table beside each
    /// lead's village chart that lines up with the manual gauge log (read as a
    /// 09:00Z rain-day). Each hour belongs to the rain-day that STARTS at the
    /// most recent 09:00Z at/before it (key = <c>(valid − 9h).Date</c>), and the
    /// day's total is the sum of the composite hourly medians — the same
    /// "sum of hourly medians" convention the headline uses (it deliberately is
    /// NOT the median of the daily sum; that needs MC convolution, deferred).
    /// Window spans the chart's range (incl. the recent past), so completed
    /// rain-days the user already has truth for sit next to the upcoming forecast
    /// totals — every row at this lead's horizon.
    /// </summary>
    internal static string RenderVillage9to9DailyTable(IReadOnlyList<RainfallAmountPredictionRow> composite, int lead)
    {
        var days = composite
            .GroupBy(r => r.ValidTimeUtc.AddHours(-9).Date)   // 09Z→09Z rain-day start date
            .Select(g => (Start: g.Key.AddHours(9), Hours: g.Count(), TotalMm: g.Sum(r => r.MedianMmPerHr)))
            .Where(d => d.Hours >= MinHoursForDay)
            .OrderBy(d => d.Start)
            .ToList();
        if (days.Count == 0) return "";

        var sb = new StringBuilder();
        sb.Append(Ci, $"""
            <details class="hourly-detail village-daily" open>
              <summary>Predicted 09Z&rarr;09Z daily totals <small>(sum of hourly village medians, at +{lead}h)</small></summary>
              <figure>
                <table>
                  <thead><tr><th>Rain-day (09Z&rarr;09Z)</th><th class="num">Predicted total</th></tr></thead>
                  <tbody>
            """);
        foreach (var d in days)
        {
            var label = d.Start.ToString("ddd dd MMM", Ci);
            var part = d.Hours < 24 ? $" <small>({d.Hours}h)</small>" : "";
            sb.Append(Ci, $"""
                    <tr><td><time datetime="{d.Start:yyyy-MM-ddTHH:mm}Z">{label}</time>{part}</td>
                      <td class="num">{d.TotalMm.ToString("0.0", Ci)} mm</td></tr>
                """);
        }
        sb.Append("</tbody></table></figure></details>");
        return sb.ToString();
    }

    /// <summary>
    /// "Total median X mm • Wettest hour: Y mm/h (Z%) • W% chance of any rain".
    /// One line summarising the day. Two reasons we don't surface a
    /// "daily 80% interval" here:
    ///   * Quantile of a sum ≠ sum of quantiles, so naive Σ P10 / Σ P90
    ///     across hours actually produces an interval below the median
    ///     (each per-hour P10 sits below the median, so the sum stays
    ///     below the sum-of-medians). The honest fix is Monte Carlo
    ///     convolution of the per-hour distributions, deferred.
    ///   * The chart below already shows the per-hour 80% band — the
    ///     useful uncertainty signal is "which hour" plus "how much
    ///     spread on that hour", not a daily-integral interval.
    /// So the headline picks the wettest expected hour and shows its
    /// (median, P90, P&ge;1mm) directly — the at-a-glance number a
    /// reader uses to plan ("rain expected around 14Z, ~2 mm/h, 60%
    /// chance of meaningful rain").
    /// </summary>
    private static string RenderRainfallAmountHeadline(IReadOnlyList<RainfallAmountPredictionRow> rows)
    {
        var dailyMedianMm = rows.Sum(r => r.MedianMmPerHr);
        var wettest = rows.OrderByDescending(r => r.MedianMmPerHr).First();
        var chanceAnyRain = rows.Max(r => r.PExceed0_1);
        // Dry-day variant: when the wettest hour's median is sub-threshold
        // (< 0.1mm/h, the WET_THRESHOLD_MM convention), the "wettest hour
        // = 0.0 mm/h" clause reads as nonsense. Surface the peak chance-
        // of-rain probability instead so the headline is honest about
        // what the model IS predicting (low intensity everywhere) rather
        // than picking an arbitrary "wettest" hour.
        const double WetThresholdMmHr = 0.1;
        if (wettest.MedianMmPerHr < WetThresholdMmHr)
        {
            return $"""
                <p class="rainfall-headline">
                  <strong>Dry day expected.</strong>
                  Peak chance of any rain: {(chanceAnyRain * 100).ToString("0", Ci)}%
                  at <time datetime="{wettest.ValidTimeUtc:yyyy-MM-ddTHH:mm}Z">{wettest.ValidTimeUtc:HH'Z'}</time>.
                  <br><small>Hour-by-hour distribution below.</small>
                </p>
                """;
        }
        return $"""
            <p class="rainfall-headline">
              <strong>{dailyMedianMm.ToString("0.0", Ci)} mm total</strong> (sum of hourly medians)
              · wettest hour <time datetime="{wettest.ValidTimeUtc:yyyy-MM-ddTHH:mm}Z">{wettest.ValidTimeUtc:HH'Z'}</time>:
                {wettest.MedianMmPerHr.ToString("0.0", Ci)} mm/h
                (80% band {wettest.P10MmPerHr.ToString("0.0", Ci)}&ndash;{wettest.P90MmPerHr.ToString("0.0", Ci)})
              · {(chanceAnyRain * 100).ToString("0", Ci)}% chance of any rain in the peak hour
              <br><small>Hour-by-hour distribution below.</small>
            </p>
            """;
    }

    /// <summary>
    /// Hourly intensity chart — "Median" line (MedianMmPerHr — π·exp(μ_log)),
    /// P10–P90 ribbon, P2.5–P97.5 outer ribbon. The ribbons reuse the
    /// LineChartSpec.Ribbons field added 2026-05-27 alongside this card;
    /// future distributional phases can reuse the same primitive.
    ///
    /// Central line is MedianMmPerHr — not P50MmPerHr — because P50 of
    /// the mixed mass-at-zero + LogNormal distribution is identically 0
    /// whenever P(wet) ≤ 50% (i.e. the dry-mass already covers the 50th
    /// percentile). MedianMmPerHr = π · exp(μ_log) is the schema's
    /// "preferred for headline display" value (RainfallAmountPredictionRow.cs
    /// line 95-97) and is the same number the headline strip above the
    /// chart shows for the wettest hour.
    ///
    /// The ribbon edges remain MIXED quantiles (P10MmPerHr etc.) — they
    /// genuinely sit at 0 for most hours, which is the truthful "10% of
    /// the time, no rain" reading. Switching to conditional-given-wet
    /// quantiles would change the chart's semantics without an
    /// in-frame label explaining the conditioning; better to keep the
    /// flat-zero floor honest than to silently swap meanings.
    /// </summary>
    private static string RenderRainfallAmountRibbon(
        IReadOnlyList<RainfallAmountPredictionRow> rows, int lead,
        double xMin, double xMax, double todayLineX)
    {
        var median = rows.Select(r => (r.ValidTimeUtc.ToOADate(), r.MedianMmPerHr)).ToList();
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
            // Share the rain page's X axis so this chart lines up with the
            // P(wet) / NWP charts above it (was auto-scaling to just the 3f
            // rows' narrow domain — Harry 2026-06-05). XMin/XMax + the
            // "now" marker mirror the other LineChartSpec callers on the page.
            XMin = xMin,
            XMax = xMax,
            TodayLineX = todayLineX,
            Series = new[]
            {
                new LineSeries("P2.5",   "#bbdefb", p2),
                new LineSeries("P10",    "#90caf9", p10),
                new LineSeries("Median", "#1565c0", median),
                new LineSeries("P90",    "#90caf9", p90),
                new LineSeries("P97.5",  "#bbdefb", p97),
            },
            Ribbons = new[]
            {
                // Outer band first so the inner band renders on top of it.
                new RibbonSpec("P2.5", "P97.5", "rgba(33, 150, 243, 0.10)"),
                new RibbonSpec("P10",  "P90",   "rgba(33, 150, 243, 0.22)"),
            },
        };
        // Chart.js rendering (not the static SVG) — gives tooltips,
        // hover-anywhere read-out, and the zoom/pan that the rest of the
        // forecast pages use. Ribbons are now supported on this path too
        // (see SvgChart.RenderChartJs's fillTo + fillColor wire).
        return $"<figure>{LineChartRenderer.RenderChartJs(spec)}</figure>";
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

    // -----------------------------------------------------------------
    // SKILL WIDGETS (plan §4.2)
    // -----------------------------------------------------------------

    /// <summary>
    /// Render the rainfall_amount skill block on the rain skill page —
    /// four widgets per the plan: CRPS rolling time-series, coverage
    /// rolling time-series, PIT histogram (most-recent aggregated),
    /// per-threshold Brier scores (most-recent aggregated).
    ///
    /// Data source is <see cref="SiteInputs.VerifyHistory"/> filtered to
    /// <c>Target == "rainfall_amount"</c>. The verify sidecars carry the
    /// rainfall_amount distributional metrics (BlendMetric = CRPS,
    /// Coverage80, PitBins, ExceedanceBriers) per (asOf, station, lead),
    /// so the rolling charts are simply asOfUtc on the X axis.
    ///
    /// Sample-size caveat — when fewer than 14 days of verify history
    /// have accumulated for a (station, lead) pair, the block renders a
    /// "n=N cycles — stabilises after ~14 days" note above the chart so
    /// the reader doesn't over-interpret early numbers. Caveat
    /// self-clears once n ≥ 14.
    /// </summary>
    internal static string RenderRainfallAmountSkillBlock(SiteInputs input, string? currentStation)
    {
        if (currentStation is null)
            return "<p class=\"skill-line\"><em>Select a station to see rainfall amount skill.</em></p>";

        // Flatten verify history → individual rows for this target + station.
        var rows = input.VerifyHistory
            .Where(f => string.Equals(f.Target, "rainfall_amount", StringComparison.Ordinal))
            .SelectMany(f => f.Rows.Select(r => (File: f, Row: r)))
            .Where(t => string.Equals(t.Row.Station, currentStation, StringComparison.Ordinal))
            .ToList();

        // Pool per (asOf, lead) BEFORE charting (2026-06-11, Harry): during a
        // version transition verify scores BOTH the outgoing and incoming 3f
        // bundles for the same window, so a sidecar carries two rows per
        // (station, lead) and the rolling lines grew vertical cliffs. The
        // n-weighted pooling (and the full WHY) lives in
        // SeriesDedup.PoolVerifyRowsPerKey, shared by every rolling verify
        // chart. Rows are already filtered to one target + station, so
        // (asOf, lead) is the full key here.
        rows = rows.PoolVerifyRowsPerKey(t => (t.File.AsOfUtc, t.Row.LeadHours));
        if (rows.Count == 0)
            return "<p class=\"skill-line\"><em>No rainfall amount verify history yet — Phase 3f isn't active for this station, or no asOf weeks have aged into truth.</em></p>";

        var leads = rows.Select(t => t.Row.LeadHours).Distinct().OrderBy(l => l).ToList();

        var content = new StringBuilder();
        content.Append(Ci, $"""
            <p class="skill-line">
              Phase 3f distributional skill — {Escape(PrettyStation(currentStation))}.
              CRPS (lower better) is the primary skill metric. Coverage
              should sit at 0.80 for a well-calibrated 80% interval; PIT
              should be flat. Exceedance Brier is per-threshold P(rain ≥
              T) — same units as P(wet) Brier above. Sample size shown
              per row.
            </p>
            """);

        // ---- CRPS rolling time-series (one line per lead) ----
        content.Append("<h4>Rolling CRPS</h4>");
        content.Append(RenderRollingTimeSeries(
            rows, leads,
            "CRPS — " + PrettyStation(currentStation),
            "CRPS",
            r => r.BlendMetric,
            input.GeneratedAtUtc));

        // ---- Coverage rolling time-series (one line per lead) ----
        content.Append("<h4>Rolling 80% coverage</h4>");
        content.Append("<p class=\"skill-line\"><small>Target = 0.80. Below = under-spread (intervals too narrow); above = over-spread.</small></p>");
        content.Append(RenderRollingTimeSeries(
            rows, leads,
            "80% coverage — " + PrettyStation(currentStation),
            "Coverage",
            r => r.Coverage80,
            input.GeneratedAtUtc,
            referenceLineY: 0.80));

        // ---- PIT histogram (most-recent aggregated across leads) ----
        var latestPerLead = leads
            .Select(l => rows.Where(t => t.Row.LeadHours == l)
                .OrderByDescending(t => t.File.AsOfUtc).FirstOrDefault())
            .Where(t => t.Row is not null)
            .ToList();
        content.Append("<h4>PIT histogram <small>(most recent verify week, all leads pooled)</small></h4>");
        content.Append(RenderPitHistogram(latestPerLead));

        // ---- Exceedance Brier table (latest per lead) ----
        content.Append("<h4>Exceedance Brier <small>(latest verify week, per lead)</small></h4>");
        content.Append(RenderExceedanceBrierTable(latestPerLead));

        // ---- Sample-size caveat (plan §6 — "stabilises after ~14 days") ----
        // Plan reads "n=N cycles — stabilises after ~14 days of VERIFY
        // HISTORY", so the gate is wall-clock weeks of asOf dates
        // accumulated, not pair count within one week. Verify runs
        // weekly → 14 days ≈ 2 distinct asOf weeks before the
        // rolling-chart shape is meaningful.
        var asOfDates = rows.Select(t => t.File.AsOfUtc.Date).Distinct().ToList();
        if (asOfDates.Count < 2)
        {
            var sinceFirst = (input.GeneratedAtUtc.Date - asOfDates.Min()).Days;
            content.Append(Ci, $"""
                <aside class="skill-caveat">
                  <strong>{asOfDates.Count} verify week so far</strong>
                  (first {sinceFirst} day{(sinceFirst == 1 ? "" : "s")} ago) —
                  skill metrics stabilise after ~14 days of verify history. Treat
                  the headline numbers with caution until then.
                </aside>
                """);
        }

        return content.ToString();
    }

    // PoolVerifyRows moved to SeriesDedup (2026-06-12) — the same
    // version-transition pooling applies to every rolling verify chart
    // (3f CRPS/coverage here, wind speed MAE on the wind skill page), so
    // the n-weighted-mean semantics live in one place.

    /// <summary>
    /// Generic rolling-metric chart for a single value per
    /// VerifyHistoryRow. One line per lead; X = AsOfUtc, Y = the
    /// metric. Returns empty-state HTML when every selector yields no
    /// finite values.
    /// </summary>
    private static string RenderRollingTimeSeries(
        IReadOnlyList<(VerifyHistoryFile File, VerifyHistoryRow Row)> rows,
        IReadOnlyList<int> leads,
        string title,
        string yLabel,
        Func<VerifyHistoryRow, double?> selector,
        DateTime nowUtc,
        double? referenceLineY = null)
    {
        var series = new List<LineSeries>();
        var palette = new[] { "#1565c0", "#388e3c", "#f57c00", "#8e24aa", "#00838f" };
        for (int i = 0; i < leads.Count; i++)
        {
            var lead = leads[i];
            var pts = rows
                .Where(t => t.Row.LeadHours == lead)
                .OrderBy(t => t.File.AsOfUtc)
                .Select(t => (X: t.File.AsOfUtc.ToOADate(), Y: selector(t.Row)))
                .Where(p => p.Y.HasValue && double.IsFinite(p.Y.Value))
                .Select(p => (p.X, p.Y!.Value))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries($"+{lead}h", palette[i % palette.Length], pts));
        }
        if (series.Count == 0)
            return "<p class=\"skill-line\"><em>No verify points yet at any lead.</em></p>";

        var spec = new LineChartSpec
        {
            Title = title,
            XLabel = "Verify week (UTC)",
            YLabel = yLabel,
            Series = series,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
            FormatY = v => v.ToString("0.000", Ci),
            TodayLineX = nowUtc.ToOADate(),
        };
        var html = LineChartRenderer.RenderChartJs(spec);
        // Hint the reader where the target line should sit. The renderer
        // doesn't have a horizontal-reference primitive yet; the prose
        // line above the chart already calls out "Target = 0.80" so we
        // skip drawing it inline.
        _ = referenceLineY;
        return $"<figure>{html}</figure>";
    }

    /// <summary>
    /// Inline-SVG bar histogram for the 10 PIT bins. Aggregates across
    /// every lead row in <paramref name="rows"/> by summing per-bin
    /// counts, then renders as 10 equal-width bars. A perfectly
    /// calibrated forecast is flat at <c>N/10</c>; bumps at 0 / 1 flag
    /// under-spread, tilt flags mean bias.
    /// </summary>
    private static string RenderPitHistogram(
        IReadOnlyList<(VerifyHistoryFile File, VerifyHistoryRow Row)> rows)
    {
        var bins = new int[10];
        var totalN = 0;
        foreach (var (_, r) in rows)
        {
            if (r.PitBins is null || r.PitBins.Count != 10) continue;
            for (int i = 0; i < 10; i++) bins[i] += r.PitBins[i];
            totalN += r.N;
        }
        if (totalN == 0)
            return "<p class=\"skill-line\"><em>No PIT samples yet.</em></p>";

        const int W = 480, H = 180, padL = 40, padR = 16, padT = 24, padB = 28;
        var plotW = W - padL - padR;
        var plotH = H - padT - padB;
        var barW = plotW / 10.0;
        var maxBin = Math.Max(1, bins.Max());
        var expected = totalN / 10.0;

        var sb = new StringBuilder();
        sb.Append(Ci, $"<figure><svg viewBox=\"0 0 {W} {H}\" xmlns=\"http://www.w3.org/2000/svg\" class=\"chart pit-hist\" role=\"img\" aria-label=\"PIT histogram\">");
        sb.Append(Ci, $"<text x=\"{W / 2}\" y=\"16\" text-anchor=\"middle\" class=\"chart-title\">PIT histogram — N = {totalN}</text>");
        // Bars.
        for (int i = 0; i < 10; i++)
        {
            var h = plotH * bins[i] / maxBin;
            var x = padL + i * barW;
            var y = padT + plotH - h;
            sb.Append(Ci, $"<rect x=\"{x:0.#}\" y=\"{y:0.#}\" width=\"{barW - 2:0.#}\" height=\"{h:0.#}\" fill=\"#1565c0\" />");
            // Bin label below each bar — "0.0", "0.1", ... "0.9".
            sb.Append(Ci, $"<text x=\"{x + barW / 2:0.#}\" y=\"{padT + plotH + 14}\" text-anchor=\"middle\" class=\"chart-tick\">{(i / 10.0).ToString("0.0", Ci)}</text>");
        }
        // Expected-flat reference line.
        var refY = padT + plotH - plotH * expected / maxBin;
        sb.Append(Ci, $"<line x1=\"{padL}\" y1=\"{refY:0.#}\" x2=\"{padL + plotW}\" y2=\"{refY:0.#}\" stroke=\"#c62828\" stroke-dasharray=\"4 3\" stroke-width=\"1\" />");
        // Frame.
        sb.Append(Ci, $"<rect x=\"{padL}\" y=\"{padT}\" width=\"{plotW}\" height=\"{plotH}\" class=\"chart-frame\" />");
        sb.Append("</svg></figure>");
        sb.Append("<p class=\"skill-line\"><small>Red dashed line = perfectly flat (N / 10 per bin). Bumps at 0.0 / 0.9 flag under-spread; tilt flags mean bias.</small></p>");
        return sb.ToString();
    }

    /// <summary>
    /// Per-threshold Brier table. One row per lead in
    /// <paramref name="rows"/>; one column per threshold key found in
    /// any row's <see cref="VerifyHistoryRow.ExceedanceBriers"/>. Lower
    /// Brier = better calibration of the exceedance probability at that
    /// threshold.
    /// </summary>
    private static string RenderExceedanceBrierTable(
        IReadOnlyList<(VerifyHistoryFile File, VerifyHistoryRow Row)> rows)
    {
        var allThresholds = rows
            .Where(t => t.Row.ExceedanceBriers is not null)
            .SelectMany(t => t.Row.ExceedanceBriers!.Keys)
            .Distinct()
            .OrderBy(k => double.TryParse(k, NumberStyles.Float, Ci, out var d) ? d : double.MaxValue)
            .ToList();
        if (allThresholds.Count == 0)
            return "<p class=\"skill-line\"><em>No exceedance Brier scores in the latest verify week.</em></p>";

        var sb = new StringBuilder();
        sb.Append("<table class=\"skill-table\"><thead><tr><th>Lead</th>");
        foreach (var t in allThresholds)
            sb.Append(Ci, $"<th class=\"num\" title=\"P(rain ≥ {Escape(t)} mm/h)\">≥ {Escape(t)} mm</th>");
        sb.Append("<th class=\"num\">N</th></tr></thead><tbody>");
        foreach (var (_, r) in rows.OrderBy(t => t.Row.LeadHours))
        {
            sb.Append(Ci, $"<tr><td>+{r.LeadHours}h</td>");
            foreach (var t in allThresholds)
            {
                var v = (r.ExceedanceBriers is not null && r.ExceedanceBriers.TryGetValue(t, out var b))
                    ? b.ToString("0.000", Ci)
                    : "—";
                sb.Append(Ci, $"<td class=\"num\">{v}</td>");
            }
            sb.Append(Ci, $"<td class=\"num\">{r.N}</td></tr>");
        }
        sb.Append("</tbody></table>");
        return sb.ToString();
    }
}
