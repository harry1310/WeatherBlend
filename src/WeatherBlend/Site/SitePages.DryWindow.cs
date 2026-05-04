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

                    var agreementCell = byLead.Values
                        .Select(d => d.AgreementHasDryWindow)
                        .FirstOrDefault(a => a.HasValue);
                    var agreement = agreementCell.HasValue
                        ? (agreementCell.Value * 100).ToString("0", Ci) + "%"
                        : "—";

                    // MC interval column — populated only for Phase 3g rows
                    // (the parameter-free MC predictor that captures longest-
                    // dry-run quantiles in the same pass as the headline).
                    // Read off the SMALLEST available lead so the user sees
                    // the freshest forecast; "—" if no 3g row exists for this
                    // (station, window, date) cell.
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

                    // Conformal-set chip: "Confident" when the prediction set
                    // is a singleton, "Ambiguous" when both classes are in
                    // the 90% set. We pick the smallest available lead's tag;
                    // if no row has a fitted conformal calibrator, "—".
                    //
                    // Tag-to-label INVERSION vs precip: ConformalSetTag stores
                    // the calibrator's "positive class" semantics — for
                    // dry-window the positive class IS "dry block exists"
                    // (matches DryWindowTrainingRow.Label), so a "Wet" tag
                    // (high P(positive)) means "confident a dry block
                    // exists" → confident DRY DAY. Conversely a "Dry" tag
                    // (low P(positive)) means "confident NO dry block" →
                    // confident WET DAY. Precip uses the same enum but the
                    // positive class there IS wet, so its labels stay
                    // "Wet"→"confident wet" / "Dry"→"confident dry". A
                    // future cleanup could rename the enum to
                    // {HighProb, LowProb, Ambiguous} to remove this
                    // per-page rendering subtlety.
                    string conformalCell = "—";
                    foreach (var lead in leadOrder)
                    {
                        if (!byLead.TryGetValue(lead, out var d)) continue;
                        if (string.IsNullOrEmpty(d.ConformalSetTag)) continue;
                        var (label, cls) = d.ConformalSetTag switch
                        {
                            "Ambiguous" => ("ambiguous", "low"),
                            "Wet"       => ("confident dry day", "high"),
                            "Dry"       => ("confident wet day", "high"),
                            _           => (d.ConformalSetTag.ToLowerInvariant(), "unknown"),
                        };
                        conformalCell = $"<span class=\"conf conf-{cls}\">{label}</span>";
                        break;
                    }

                    // Best-start cell: argmax of the curve at the smallest
                    // available lead bucket for this (station, window, date).
                    // Falls back to "—" when no curve, when the daily P(any) is
                    // too low for a "best start" to mean anything, or when the
                    // curve is too uniform to peak. See PickBestStart for the
                    // suppression rules.
                    string bestStartCell = "—";
                    if (hasCurves)
                    {
                        foreach (var lead in leadOrder)
                        {
                            if (!byLead.ContainsKey(lead)) continue;
                            if (!curvesByCell.TryGetValue((currentStation, window, lead, date), out var curve)) continue;
                            var best = PickBestStart(curve);
                            if (best is null) continue;
                            // "10:00Z (32%)" — UTC start hour + the
                            // calibrated marginal so the reader sees both
                            // location and confidence.
                            bestStartCell = $"{best.StartHourUtc:00}:00Z <small>({(best.CalibratedProb * 100).ToString("0", Ci)}%)</small>";
                            break;
                        }
                    }

                    var bestStartTd = hasCurves ? $"<td>{bestStartCell}</td>" : "";

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time datetime="{date:yyyy-MM-dd}">{date:ddd} {date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          <td class="num">{agreement}</td>
                          <td>{mcCell}</td>
                          <td>{conformalCell}</td>
                          {bestStartTd}
                        </tr>
                        """);
                }

                // When only one phase is active, the per-phase header is just
                // noise — the section already names the window. Skip it and let
                // the table sit directly under the <h4>. The header returns
                // automatically the moment a second phase joins All.
                if (DryWindowPhases.All.Count > 1)
                {
                    content.Append(Ci, $"""
                        <h5>{Escape(phase.LongTitle)}</h5>
                        <p class="skill-line">{Escape(phase.Description)}</p>
                        """);
                }
                var bestStartHeader = hasCurves
                    ? "<th>Best start <small>(UTC, calibrated %)</small></th>"
                    : "";
                content.Append(Ci, $"""
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
                            <th class="num">Model agreement</th>
                            <th>MC longest dry run <small>(3g only)</small></th>
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
        return WrapPage(input, "Dry window", "dry-window", content.ToString());
    }
}
