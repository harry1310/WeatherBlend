using System.Text;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Pick the start hour with the highest P(an N-hour dry block runs from
    /// it) — <see cref="StartHourForecastPoint.RawProduct"/> — from a curve,
    /// or return <c>null</c> when the row carries no useful signal at all
    /// (daily P(any block) is so low that picking ANY start hour would
    /// mislead the reader into chasing a near-certainly-not-happening
    /// block). For curves where daily P is meaningful but the curve itself
    /// is near-uniform we still surface the argmax because the <c>(NN%)</c>
    /// printed alongside it is itself the signal: a poor day renders as
    /// "10:00Z (24%)", a good one as "10:00Z (58%)", and the reader judges
    /// from the number how likely a dry walk actually is. Earlier we also
    /// gated on a 10pp peak−trough range and
    /// it suppressed lead-48 / lead-72 cells where the daily P was 90%+ but
    /// the within-day shape was flat — the user couldn't tell the
    /// difference between "model has weak opinion on when" and "no curve
    /// available", which was the wrong trade-off.
    /// </summary>
    internal const double StartHourMinDailyProb = 0.10;

    internal static StartHourForecastPoint? PickBestStart(
        IEnumerable<StartHourForecastPoint> curveForOneCell)
    {
        StartHourForecastPoint? best = null;
        double maxProb = double.NegativeInfinity;
        foreach (var r in curveForOneCell)
        {
            // Best = the start hour with the highest P(an N-hour dry block
            // runs from it) — RawProduct, the per-hour marginal.
            if (r.RawProduct > maxProb) { maxProb = r.RawProduct; best = r; }
        }
        if (best is null) return null;
        if (best.DailyProbAnyBlock < StartHourMinDailyProb) return null;
        return best;
    }

    /// <summary>
    /// Dry window page. <paramref name="stationSlug"/> picks which station
    /// to render; <c>null</c> means the first station, which ships as
    /// <c>dry-window.html</c> (filename preserved so existing links don't break).
    /// The other stations ship as <c>dry-window-{slug}.html</c>. Each variant
    /// shows the active phases × all windows for the one station.
    /// </summary>
    public static string RenderDryWindow(SiteInputs input, string? stationSlug = null)
    {
        var content = new StringBuilder();
        content.Append("<section>");
        // Phase D: the cross-variable Forecasts sub-nav is gone — each
        // variable (Temperature / Rain / Dry window) is its own top-level
        // per-loc nav button now.
        content.Append("""
              <hgroup>
                <h2>Dry-window forecast</h2>
                <p>P(at least one N-hour dry block in 09:00–18:00 local). 3b champion plus Monte-Carlo challengers.</p>
              </hgroup>
            """);

        if (input.DryWindowPredictions.Count == 0)
        {
            content.Append("<p><em>No dry-window predictions in window. Run <code>predict --target dry-window --truth-station all --window all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry-window forecast", "dry-window", content.ToString());
        }

        // Filter station list to currently-active stations (from config) so a
        // demoted station whose historical predictions are still on disk doesn't
        // get a per-station tab. Empty ActiveStationSlugs preserves the legacy
        // "render whatever's on disk" behaviour for tests that don't populate it.
        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct()
            .Where(s => input.ActiveStationSlugs.Count == 0 || input.ActiveStationSlugs.Contains(s))
            .OrderBy(s => s, StringComparer.Ordinal).ToList();
        var currentStation = ResolveStationFromSlug(stations, stationSlug);

        if (currentStation is not null)
            content.Append(RenderStationSubNav("forecasts-dry-window", stations, currentStation));

        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = Leads.Short;
        // Today forward only (was yesterday-onward) — reading the page mid-
        // morning, the "yesterday" tile was always already-known noise. Drop
        // it so the view starts at "right now / today's outlook".
        var cutoff = input.GeneratedAtUtc.Date;

        if (currentStation is null)
        {
            content.Append("<p><em>No dry-window predictions for the selected station.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry-window forecast", "dry-window", content.ToString());
        }

        content.Append(Ci, $"<h3>{Escape(PrettyStation(currentStation))}</h3>");

        // Index the start-hour curves once per (station, window, lead,
        // target_date, version) so the inner loop is O(1) per row. Version is
        // part of the key so phases writing curves to the same cell under
        // different model_versions don't collide in the ToDictionary call.
        // Each phase reads only its own curve via DryWindowPhase.StartHourCurveVersion.
        // Empty dictionary when no curves on disk — renderer falls back
        // gracefully to the pre-curve table layout.
        // input.StartHourPredictions is already scoped to this page's location
        // by RenderSiteCommand (single-location SiteInputs invariant) — key by
        // the cell tuple directly.
        var curvesByCell = input.StartHourPredictions
            .Where(s => s.Station == currentStation)
            .GroupBy(s => (s.Station, s.WindowHours, s.LeadHours, s.TargetDateUtc, s.Version))
            .ToDictionary(g => g.Key, g => (IReadOnlyList<StartHourForecastPoint>)g.ToList());
        var hasCurves = curvesByCell.Count > 0;

        foreach (var window in windows)
        {
            var windowRows = input.DryWindowPredictions
                .Where(d => d.Station == currentStation && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                .ToList();

            if (windowRows.Count == 0)
            {
                content.Append(Ci, $"<h4>{window}-hour dry window</h4><p><em>No predictions on or after {cutoff:yyyy-MM-dd}.</em></p>");
                continue;
            }

            content.Append(Ci, $"<h4>{window}-hour dry window</h4>");

            bool anyRendered = false;
            foreach (var phase in DryWindowPhases.All)
            {
                var phaseRows = windowRows
                    .Where(d => DryWindowPhases.Bucket(input.PhaseByVersion, d.Version) == phase)
                    .ToList();
                if (phaseRows.Count == 0) continue;
                anyRendered = true;

                // Per-phase column policy. Each phase contributes the columns
                // its model actually generates — sharing one mega-table forces
                // some phases to display empty cells, so we gate per phase.
                //   3b — LightGBM marginal blender. Carries cross-NWP
                //        agreement (the BSS-weighted vote of its component
                //        models) and a conformal calibrator τ on the marginal.
                //   3p — Gaussian copula MC over 3o; carries the MC band +
                //        start-hour curve columns.
                bool showAgreement  = phase.Key == DryWindowPhases.Phase3b.Key;
                bool showMcBand     = false;
                // Keys off the phase's own curve version, so any future iid-MC
                // phase that declares a StartHourCurveVersion opts in for free.
                bool showBestStart  = phase.StartHourCurveVersion is not null && hasCurves;

                // Latest prediction per (target_date, lead) within this phase bucket.
                var latest = phaseRows
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

                var dates = latest.Select(d => d.TargetDateUtc).Distinct().OrderBy(d => d).ToList();

                var tbody = new StringBuilder();
                foreach (var date in dates)
                {
                    var byLead = latest.Where(d => d.TargetDateUtc == date).ToDictionary(d => d.LeadHours);

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                        {
                            // Cell text colour walks the green→red gradient so the eye
                            // can scan the column without reading every digit. Values
                            // render as integer percentages — same scale as the Home
                            // P(wet) chip, less cognitive load than a 0..1 fraction.
                            //
                            // Optional epistemic band suffix: phases that
                            // populate EpistemicProbDryWindowQ10/Q90 get a
                            // small "(LO–HI%)" line under the headline. No
                            // active phase populates these today; left in
                            // place for future producers.
                            var color = ProbabilityColor(d.ProbHasDryWindow);
                            string bandSuffix = "";
                            if (d.EpistemicProbDryWindowQ10.HasValue && d.EpistemicProbDryWindowQ90.HasValue)
                            {
                                var lo = (d.EpistemicProbDryWindowQ10.Value * 100).ToString("0", Ci);
                                var hi = (d.EpistemicProbDryWindowQ90.Value * 100).ToString("0", Ci);
                                bandSuffix = $"<br><small style=\"opacity:0.7;font-weight:400\">{lo}–{hi}%</small>";
                            }
                            leadCells.Append(Ci, $"<td class=\"num\" style=\"color: {color}; font-weight: 600\">{(d.ProbHasDryWindow * 100).ToString("0", Ci)}%{bandSuffix}</td>");
                        }
                        else
                        {
                            leadCells.Append("<td class=\"num\">—</td>");
                        }
                    }

                    string agreementTd = "";
                    if (showAgreement)
                    {
                        var agreementCell = byLead.Values
                            .Select(d => d.AgreementHasDryWindow)
                            .FirstOrDefault(a => a.HasValue);
                        var agreement = agreementCell.HasValue
                            ? (agreementCell.Value * 100).ToString("0", Ci) + "%"
                            : "—";
                        agreementTd = $"<td class=\"num\">{agreement}</td>";
                    }

                    string mcTd = "";
                    if (showMcBand)
                    {
                        // Read off the smallest available lead so the user sees
                        // the freshest forecast.
                        string mcCell = "—";
                        foreach (var lead in leadOrder)
                        {
                            if (!byLead.TryGetValue(lead, out var d)) continue;
                            if (!d.McP50LongestDryRunHours.HasValue) continue;
                            var p50 = d.McP50LongestDryRunHours.Value;
                            var p10 = d.McP10LongestDryRunHours ?? 0;
                            var p90 = d.McP90LongestDryRunHours ?? 0;
                            // "median 5h, 80%CI 2-8h" — short to fit inline.
                            mcCell = $"med {p50:0}h <small>(80%: {p10:0}-{p90:0}h)</small>";
                            break;
                        }
                        mcTd = $"<td>{mcCell}</td>";
                    }

                    // Confidence chip. Two routes depending on what's on the row:
                    //
                    // (a) ConformalSetTag populated (3b — bundle ships a fitted
                    //     90%-coverage calibrator) → conformal chip + τ. The
                    //     calibrator's "positive class" for dry-window IS "dry
                    //     block exists", so its "Wet" → confident DRY day and
                    //     "Dry" → confident WET day (this inversion is local to
                    //     the dry-window page, not precip's).
                    //
                    // (b) No ConformalSetTag (3p — parameter-free copula MC
                    //     over 3o; no analogous val-slice replay tree, see
                    //     2026-05-26 thread) → derive chip from probability
                    //     proximity to 0.5 (heuristic, no coverage guarantee)
                    //     and append the per-MC-sample longest-dry-run band
                    //     as the inline detail. Narrow P10–P90 = robust;
                    //     wide = fragile.
                    //
                    // We pick the smallest available lead's row so the user
                    // sees the freshest forecast — same as the lead pick in
                    // the agreement / mc / best-start cells above.
                    string conformalCell = "—";
                    foreach (var lead in leadOrder)
                    {
                        if (!byLead.TryGetValue(lead, out var d)) continue;
                        if (!string.IsNullOrEmpty(d.ConformalSetTag))
                        {
                            var (label, cls) = d.ConformalSetTag switch
                            {
                                "Ambiguous" => ("ambiguous", "low"),
                                "Wet"       => ("confident dry", "high"),
                                "Dry"       => ("confident wet", "high"),
                                _           => (d.ConformalSetTag.ToLowerInvariant(), "unknown"),
                            };
                            var tauPart = input.DryWindowConformalTau.TryGetValue((d.Version, d.LeadHours), out var tau)
                                ? string.Create(Ci, $" · τ={(tau * 100):0}%")
                                : "";
                            conformalCell = string.Create(Ci, $"<span class=\"conf conf-{cls}\">{label}</span> <small>(P={(d.ProbHasDryWindow * 100):0}%{tauPart})</small>");
                            break;
                        }
                        // Heuristic fallback for MC phases (3p) — chip from
                        // ProbHasDryWindow proximity to 0.5 with ±0.20 band.
                        // This intentionally mirrors the conformal chip's
                        // shape ("confident dry" / "ambiguous" / "confident
                        // wet") without claiming a 90% coverage guarantee.
                        var (hLabel, hCls) = d.ProbHasDryWindow switch
                        {
                            >= 0.80 => ("confident dry", "high"),
                            <= 0.20 => ("confident wet", "high"),
                            _       => ("ambiguous",     "low"),
                        };
                        var bandPart = (d.McP10LongestDryRunHours, d.McP90LongestDryRunHours) switch
                        {
                            ({ } p10, { } p90) =>
                                string.Create(Ci, $" · run {p10:0}–{p90:0}h"),
                            _ => "",
                        };
                        conformalCell = string.Create(Ci, $"<span class=\"conf conf-{hCls}\">{hLabel}</span> <small>(P={(d.ProbHasDryWindow * 100):0}%{bandPart})</small>");
                        break;
                    }

                    string bestStartTd = "";
                    if (showBestStart)
                    {
                        // Argmax of the curve at the smallest available lead.
                        // "—" when no curve, when daily P(any) is too low to
                        // bother, or when the curve is too uniform to peak —
                        // see PickBestStart for the suppression rules.
                        string bestStartCell = "—";
                        foreach (var lead in leadOrder)
                        {
                            if (!byLead.ContainsKey(lead)) continue;
                            if (!curvesByCell.TryGetValue((currentStation, window, lead, date, phase.StartHourCurveVersion!), out var curve)) continue;
                            var best = PickBestStart(curve);
                            if (best is null) continue;
                            // "10:00Z (58%)" — UTC start hour + P(an N-hour dry
                            // block runs from it), so the reader sees both when
                            // to set off and how likely it stays dry.
                            bestStartCell = $"{best.StartHourUtc:00}:00Z <small>({(best.RawProduct * 100).ToString("0", Ci)}%)</small>";
                            break;
                        }
                        bestStartTd = $"<td>{bestStartCell}</td>";
                    }

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time datetime="{date:yyyy-MM-dd}">{date:ddd} {date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          {agreementTd}
                          {mcTd}
                          <td>{conformalCell}</td>
                          {bestStartTd}
                        </tr>
                        """);
                }

                // Always emit the per-phase heading now that each phase has its
                // own column shape — the table alone doesn't tell the reader
                // which model produced the row. (Previously skipped when only
                // one phase shipped, but with two distinct tables the heading
                // is genuinely informative regardless.)
                content.Append(Ci, $"""
                    <h5>{Escape(phase.LongTitle)}</h5>
                    <p class="skill-line">{Escape(phase.Description)}</p>
                    """);

                var agreementHeader = showAgreement ? "<th class=\"num\">Model agreement</th>" : "";
                var mcHeader        = showMcBand    ? "<th>MC longest dry run</th>"           : "";
                var bestStartHeader = showBestStart ? "<th>Best start <small>(UTC, P dry block)</small></th>" : "";

                content.Append(Ci, $"""
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
                            {agreementHeader}
                            {mcHeader}
                            <th>Confidence</th>
                            {bestStartHeader}
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);

                // Start-hour curve panel — only MC phases produce one
                // (declared via DryWindowPhase.StartHourCurveVersion). Plots
                // the within-day shape the "Best start" column reads its
                // argmax from, so the reader sees the whole curve, not just
                // the peak. Returns "" when this phase has no curve on disk.
                if (showBestStart)
                    content.Append(RenderStartHourCurveChart(phase, currentStation, window, cutoff, curvesByCell));
            }

            if (!anyRendered)
            {
                content.Append("<p><em>No predictions in known phase buckets for this window.</em></p>");
            }
        }

        content.Append("""
            <p class="skill-line">Dry hour = all four 15-min readings ≤ 0.1 mm. Search bounded to 09:00–18:00 local.</p>
            </section>
            """);
        // Phase D: "dry-window" is a top-level per-loc tab.
        return WrapPage(input, "Dry window", "dry-window", content.ToString());
    }

    /// <summary>One distinct line colour per forecast horizon on the
    /// start-hour curve chart, indexed by position in <see cref="Leads.Short"/>
    /// (24h / 48h / 72h).</summary>
    private static readonly string[] StartHourLeadColors = { "#1e88e5", "#fb8c00", "#8e24aa" };

    /// <summary>
    /// Render the start-hour probability curve for one (phase, station,
    /// window) as a line chart: x = daytime block-start hour (UTC), y =
    /// P(an N-hour dry block runs from this hour) — the per-start-hour
    /// marginal (RawProduct), one line per forecast horizon. Only the
    /// iid-MC phases carry a start-hour curve — the phase identifies its
    /// own curve via <see cref="DryWindowPhase.StartHourCurveVersion"/>,
    /// so each phase reads its own curve with no cross-talk.
    /// For each horizon the curve from the freshest prediction run (latest
    /// <c>PredictedAtUtc</c>) on or after <paramref name="cutoff"/> is drawn,
    /// so the panel reads as "the next forecastable day at +24h / +48h / +72h"
    /// and never surfaces a stale curve from an older run still on disk.
    /// Returns an empty string when the phase has no curve version or no
    /// curve rows on disk for this cell, so the caller can append it blind.
    /// </summary>
    private static string RenderStartHourCurveChart(
        DryWindowPhase phase,
        string station,
        int window,
        DateTime cutoff,
        IReadOnlyDictionary<(string Station, int WindowHours, int LeadHours, DateTime TargetDateUtc, string Version),
            IReadOnlyList<StartHourForecastPoint>> curvesByCell)
    {
        if (phase.StartHourCurveVersion is not { } version) return "";

        var series = new List<LineSeries>();
        int colorIdx = 0;
        foreach (var lead in Leads.Short)
        {
            // Curve from the freshest prediction run for this horizon. A target
            // date can have curves from several runs on disk (today's run plus
            // older ones whose horizon happens to land on the same day); the
            // cutoff filter alone isn't enough — ordering by TargetDateUtc would
            // still surface a 3-day-old run's curve. Ordering by PredictedAtUtc
            // picks the latest forecast, and its target is lead hours ahead.
            var cell = curvesByCell
                .Where(kv => kv.Key.Station == station
                          && kv.Key.WindowHours == window
                          && kv.Key.LeadHours == lead
                          && kv.Key.Version == version
                          && kv.Key.TargetDateUtc >= cutoff)
                .Select(kv => (kv.Key.TargetDateUtc, Curve: kv.Value))
                .OrderByDescending(c => c.Curve.Max(p => p.PredictedAtUtc))
                .ThenBy(c => c.TargetDateUtc)
                .FirstOrDefault();
            if (cell.Curve is null || cell.Curve.Count == 0) continue;

            // Dedupe to the latest run per start hour in case the freshest
            // anchor was predicted more than once into the same parquet.
            var points = cell.Curve
                .GroupBy(p => p.StartHourUtc)
                .Select(g => g.OrderByDescending(p => p.PredictedAtUtc).First())
                .OrderBy(p => p.StartHourUtc)
                // Y pre-scaled to 0-100: the Chart.js path formats the raw
                // value (no *100), so the points carry the percentage and
                // FormatY is a bare "{v}%".
                .Select(p => ((double)p.StartHourUtc, p.RawProduct * 100.0))
                .ToList();
            series.Add(new LineSeries(
                Name: string.Create(Ci, $"+{lead}h · {cell.TargetDateUtc:ddd dd MMM}"),
                Color: StartHourLeadColors[colorIdx % StartHourLeadColors.Length],
                Points: points));
            colorIdx++;
        }

        if (series.Count == 0) return "";

        var spec = new LineChartSpec
        {
            Title = string.Create(Ci, $"{phase.ShortTitle} — dry-block chance by start hour, {window}h window"),
            XLabel = "Block start hour (UTC)",
            YLabel = "P(dry block starts here)",
            Series = series,
            Height = 300,
            // X is a plain hour-of-day (0-23), shared across forecast-horizon
            // lines — NOT a datetime. ClockHour keeps the JS from running the
            // OADate conversion and labels ticks/tooltips as "HH:00Z".
            XAxis = ChartXAxis.ClockHour,
            FormatX = h => string.Create(Ci, $"{Math.Round(h):00}:00Z"),
            FormatY = v => string.Create(Ci, $"{v:0}%"),
        };
        return string.Create(Ci, $"""
            <figure>
              {LineChartRenderer.RenderChartJs(spec)}
              <figcaption class="skill-line">Monte-Carlo P(an {window}-hour dry block runs from each start hour) — one line per forecast horizon, soonest target day shown. The peak is the best time to set off; the height is how likely it stays dry. Each hour is its own probability, so the lines need not sum to the daily "any dry window" figure.</figcaption>
            </figure>
            """);
    }
}
