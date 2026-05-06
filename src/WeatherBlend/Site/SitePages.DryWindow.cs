using System.Text;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Pick the highest-calibrated-probability start hour from a curve, or
    /// return <c>null</c> when the row carries no useful signal at all
    /// (daily P(any block) is so low that picking ANY start hour would
    /// mislead the reader into chasing a near-certainly-not-happening
    /// block). For curves where daily P is meaningful but the curve itself
    /// is near-uniform — high daily P with little variation across start
    /// hours — we still surface the argmax because the calibrated <c>(NN%)</c>
    /// printed alongside it is itself the curve-sharpness signal: a flat
    /// curve renders as "10:00Z (24%)", a peaked one as "10:00Z (45%)", and
    /// the reader gets to judge from the number whether the model has a
    /// real opinion. Earlier we also gated on a 10pp peak−trough range and
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
            if (r.ConditionalProb > maxProb) { maxProb = r.ConditionalProb; best = r; }
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
        // Forecasts variable sub-nav at the very top so its Y position is
        // fixed across all three forecasts pages (Temperature / Rain / Dry
        // window). Header copy below.
        content.Append(RenderForecastsSubNav("dry-window"));
        content.Append("""
              <hgroup>
                <h2>Dry-window forecast</h2>
                <p>P(at least one N-hour dry block in 09:00–18:00 local). 3b champion + 3g challenger.</p>
              </hgroup>
            """);

        if (input.DryWindowPredictions.Count == 0)
        {
            content.Append("<p><em>No dry-window predictions in window. Run <code>predict --target dry-window --truth-station all --window all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry-window forecast", "forecasts", content.ToString());
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
            return WrapPage(input, "Dry-window forecast", "forecasts", content.ToString());
        }

        content.Append(Ci, $"<h3>{Escape(PrettyStation(currentStation))}</h3>");

        // Index the start-hour curves once per (station, window, lead,
        // target_date) so the inner loop is O(1) per row. Empty dictionary
        // when no curves on disk — renderer falls back gracefully to the
        // pre-curve table layout.
        var curvesByCell = input.StartHourPredictions
            .Where(s => s.Station == currentStation)
            .GroupBy(s => (s.Station, s.WindowHours, s.LeadHours, s.TargetDateUtc))
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
                // its model actually generates — sharing one mega-table forced
                // 3b rows to display an empty MC column and 3g rows to display
                // a "Model agreement" that's the 3b ensemble's, not 3g's.
                //   3b — LightGBM marginal blender. Carries cross-NWP
                //        agreement (the BSS-weighted vote of its component
                //        models) and a conformal calibrator τ on the marginal.
                //   3g — Monte Carlo over Phase 3a hourly P(wet). Doesn't have
                //        cross-NWP agreement (single MC ensemble); does have
                //        the longest-dry-run quantile band and the start-hour
                //        curves that PickBestStart consumes. Conformal τ is a
                //        separate fit against the MC marginal.
                bool showAgreement  = phase.Key == DryWindowPhases.Phase3b.Key;
                bool showMcBand     = phase.Key == DryWindowPhases.Phase3g.Key;
                bool showBestStart  = phase.Key == DryWindowPhases.Phase3g.Key && hasCurves;

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
                            var color = ProbabilityColor(d.ProbHasDryWindow);
                            leadCells.Append(Ci, $"<td class=\"num\" style=\"color: {color}; font-weight: 600\">{(d.ProbHasDryWindow * 100).ToString("0", Ci)}%</td>");
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

                    // Conformal-set chip: "Confident" when the prediction set
                    // is a singleton, "Ambiguous" when both classes are in
                    // the 90% set. We pick the smallest available lead's tag;
                    // if no row has a fitted conformal calibrator, "—". τ is
                    // looked up by (version, lead) so each phase reads its
                    // own calibrator without cross-talk.
                    //
                    // Tag-to-label INVERSION vs precip: ConformalSetTag stores
                    // the calibrator's "positive class" semantics — for
                    // dry-window the positive class IS "dry block exists",
                    // so "Wet" → confident DRY day, "Dry" → confident WET day.
                    string conformalCell = "—";
                    foreach (var lead in leadOrder)
                    {
                        if (!byLead.TryGetValue(lead, out var d)) continue;
                        if (string.IsNullOrEmpty(d.ConformalSetTag)) continue;
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
                            if (!curvesByCell.TryGetValue((currentStation, window, lead, date), out var curve)) continue;
                            var best = PickBestStart(curve);
                            if (best is null) continue;
                            // "10:00Z (32%)" — UTC start hour + the calibrated
                            // marginal so the reader sees both location and
                            // confidence.
                            bestStartCell = $"{best.StartHourUtc:00}:00Z <small>({(best.CalibratedProb * 100).ToString("0", Ci)}%)</small>";
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
                var bestStartHeader = showBestStart ? "<th>Best start <small>(UTC, calibrated %)</small></th>" : "";

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
                            <th>Conformal <small>(90% set)</small></th>
                            {bestStartHeader}
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);
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
        // pageId="forecasts" so the top-nav "Forecasts" stays highlighted
        // when viewing dry-window — it's a sub-page reached via the
        // Forecasts variable sub-nav. (The empty-state branches above
        // already pass "forecasts" for the same reason.)
        return WrapPage(input, "Dry window", "forecasts", content.ToString());
    }
}
