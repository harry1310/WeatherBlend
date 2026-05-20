using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Held-out test performance for the latest active blender per (composite × phase).
    /// One small card per (target × phase) with a short prose description plus a
    /// tight per-lead table comparing the blend's test score to the best single NWP
    /// scored on the same test slice (apples-to-apples).
    ///
    /// Latest-only: the <see cref="SiteInputs.ModelSummaries"/> list grew unwieldy
    /// once each composite had multiple historical artefacts. This page now keeps
    /// only the most recently trained version per (composite, phase) so the eye
    /// can scan it. Older versions still live on disk; visit the artefact files
    /// directly if you need the history.
    /// </summary>
    /// <summary>
    /// Per-target Models page: one of "temperature" / "precipitation" /
    /// "dry_window". Filters <see cref="SiteInputs.ModelSummaries"/> to
    /// composites whose target prefix matches and renders the same per-phase
    /// blender cards. Emitted to <c>models-temp.html</c> /
    /// <c>models-rain.html</c> / <c>models-dry-window.html</c> by
    /// <see cref="WeatherBlend.Commands.RenderSiteCommand"/>; the legacy
    /// <c>models.html</c> route was retired with the 2026-05-04 site rework.
    /// </summary>
    public static string RenderModels(SiteInputs input, string target)
    {
        var (subNavSlug, prettyTitle, intro) = target switch
        {
            "temperature" => ("temp",
                "Temperature models",
                "2b (lean) + 2c (rich). MAE °C, lower better."),
            "precipitation" => ("rain",
                "Rain models",
                "Per-station P(wet ≥ 0.1 mm/h). 3a (lean) + 3c (rich). Brier, lower better."),
            "dry_window" => ("dry-window",
                "Dry-window models",
                "Per-(station, window) P(N-hour dry block in 09–18 local). 3b LightGBM + 3g MC. Brier, lower better."),
            _ => throw new ArgumentException($"Unknown target '{target}'.", nameof(target)),
        };

        // Phase C commit 4: primary location slug threads down to the
        // verify-history filters so legacy sidecars (which had Station=null
        // because they predate the per-loc composite key) still surface on
        // the primary card. Membury / future-loc cards do strict equality.
        var primaryLocSlug = PrimaryLocation(input.Locations)?.Name
            ?? input.ActiveLocationName
            ?? "";

        var content = new StringBuilder();
        content.Append("<section>");
        // Sub-nav goes first so its vertical position is fixed across the
        // three Models pages — flicking between Temp / Rain / Dry-window
        // doesn't make the tab bar jump around as content height varies.
        content.Append(RenderModelsSubNav(subNavSlug));
        content.Append(Ci, $"""
              <hgroup>
                <h2>{Escape(prettyTitle)}</h2>
                <p>{Escape(intro)} Δ vs best single NWP — negative = blend wins.</p>
              </hgroup>
            """);

        if (input.ModelSummaries.Count == 0)
        {
            content.Append("<p><em>No training metadata on disk. Run <code>train</code> for each target, then rclone <code>data/models</code> from R2.</em></p>");
            content.Append("</section>");
            return WrapPage(input, prettyTitle, "models", content.ToString());
        }

        // Filter to this target's composites only. Composite keys: "temperature",
        // "precipitation/{station}", "dry_window/{station}/{N}h" — match by
        // target prefix.
        var targetSummaries = input.ModelSummaries
            .Where(m => m.Composite == target || m.Composite.StartsWith(target + "/", StringComparison.Ordinal))
            .Where(m => IsActivePhase(m.Composite, m.Phase))
            .ToList();

        if (targetSummaries.Count == 0)
        {
            content.Append("<p><em>No active blenders for this target in the current manifest.</em></p>");
            content.Append("</section>");
            return WrapPage(input, prettyTitle, "models", content.ToString());
        }

        // One card per (composite, phase). Phases like 2d and 3d are
        // commonly split across multiple ModelSummaries because the
        // short-leads and long-leads halves are trained as separate
        // versions (e.g. 2d short → leads {12,24,48} in version A,
        // 2d long → leads {72,96,120} in version B). Merge their PerLead
        // dicts so the card shows the full lead range; the header carries
        // the latest version's metadata (newest training wins for the
        // version tag and TrainedAt label). Per lead, the freshest version
        // containing that lead supplies the metric.
        //
        // Also tracks the previous training CYCLE per family for the
        // "Δ vs previous train" badge (Phase 4 of AUTO_RETRAIN_PLAN.md) —
        // see MergeTrainingCycle and the per-cycle merge in the Select below.
        var latestPerFamily = targetSummaries
            .GroupBy(m => (m.Composite, m.Phase), comparer: null!)
            .Select(g =>
            {
                var ordered = g.OrderByDescending(m => m.TrainedAtUtc).ToList();
                var latest = ordered[0];
                var mergedPerLead = ordered
                    .SelectMany(s => s.PerLead)
                    .GroupBy(kv => kv.Key)
                    .ToDictionary(grp => grp.Key, grp => grp.First().Value);
                // The "Δ vs previous train" badge compares two TRAINING
                // CYCLES, not two single versions. A cycle = every version of
                // this (composite, phase) sharing a calendar date. 2d/3d
                // routinely split a cycle into a short-leads version and a
                // long-leads version on the same day, so any single version
                // covers only part of the cycle's lead set. Merge each cycle
                // independently so the badge sees its full lead coverage —
                // without this, a latest {12,24,48} vs a previous long-half
                // {72,96,120} are disjoint and the badge silently never
                // renders. Per-cycle (not the all-versions mergedPerLead
                // above), so a lead's score is never backfilled across
                // cycles; a genuine cross-cycle lead-set change still yields
                // an empty intersect and correctly skips the badge.
                var latestDate = latest.TrainedAtUtc.Date;
                var latestCycle = MergeTrainingCycle(
                    ordered.Where(m => m.TrainedAtUtc.Date == latestDate).ToList());
                var olderCycles = ordered
                    .Where(m => m.TrainedAtUtc.Date != latestDate)
                    .ToList();
                ModelSummary? previousCycle = olderCycles.Count == 0
                    ? null
                    : MergeTrainingCycle(olderCycles
                        .Where(m => m.TrainedAtUtc.Date == olderCycles[0].TrainedAtUtc.Date)
                        .ToList());
                return (
                    Composite: latest.Composite,
                    LatestForDisplay: latest with { PerLead = mergedPerLead },
                    LatestCycle: latestCycle,
                    Previous: previousCycle);
            })
            .GroupBy(t => t.Composite, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal);

        foreach (var group in latestPerFamily)
        {
            content.Append(Ci, $"""<h3>{Escape(PrettyComposite(group.Key))}</h3>""");
            // Within a composite (e.g. one temperature block), order phase-cards
            // by champion → challenger so a glance at the page reads "lean first,
            // rich underneath".
            var ordered = group
                .OrderBy(t => PhasePriority(group.Key, t.LatestForDisplay.Phase))
                .ThenByDescending(t => t.LatestForDisplay.TrainedAtUtc)
                .ToList();
            foreach (var t in ordered)
                content.Append(RenderBlenderCard(group.Key, t.LatestForDisplay,
                    t.LatestCycle, t.Previous, input.VerifyHistory, primaryLocSlug));
        }

        content.Append("</section>");
        return WrapPage(input, prettyTitle, "models", content.ToString());
    }

    /// <summary>
    /// Per-target sub-nav for the Models page. Same shape as the per-station
    /// sub-nav on Dry-window / Skill-rainfall: three pill links sitting under
    /// the page heading. The active variant is plain (not a link) so the
    /// reader's eye lands on it.
    /// </summary>
    private static string RenderModelsSubNav(string activeSlug)
    {
        var entries = new (string Slug, string File, string Label)[]
        {
            ("temp",       "models-temp.html",       "Temperature"),
            ("rain",       "models-rain.html",       "Rain"),
            ("dry-window", "models-dry-window.html", "Dry window"),
            ("spec",       "models-spec.html",       "Spec"),
        };
        var s = new StringBuilder();
        s.Append("<nav class=\"lead-nav\"><ul>");
        foreach (var (slug, file, label) in entries)
        {
            var cls = slug == activeSlug ? " class=\"active\"" : "";
            s.Append(Ci, $"<li><a href=\"{file}\"{cls}>{Escape(label)}</a></li>");
        }
        s.Append("</ul></nav>");
        return s.ToString();
    }

    /// <summary>
    /// Maps a Models-page composite key
    /// (<c>"temperature"</c> / <c>"precipitation/{station}"</c> /
    /// <c>"dry_window/{station}/{N}h"</c>) to the
    /// <see cref="WeatherBlend.Models.VerifyHistoryFile.Target"/> +
    /// <see cref="WeatherBlend.Models.VerifyHistoryRow.Station"/> +
    /// <see cref="WeatherBlend.Models.VerifyHistoryRow.WindowHours"/> tuple a
    /// verify-history file uses, so the per-card history filter can pick out
    /// rows matching this exact card.
    /// </summary>
    /// <summary>
    /// Matches a <see cref="WeatherBlend.Models.VerifyHistoryRow.Station"/>
    /// against a card's expected station key (from
    /// <see cref="ParseCompositeForHistory"/>), with back-compat for
    /// pre-Phase-C-commit-4 sidecars where temp + element rows had
    /// <c>Station=null</c> (all such rows were primary-location at write
    /// time, so they match the primary card only). Non-primary cards do
    /// strict equality — a Membury card never picks up a legacy null-station
    /// row.
    /// </summary>
    private static bool StationMatchesCard(string? rowStation, string? cardStation, string primaryLocSlug)
    {
        if (rowStation == cardStation) return true;
        if (string.IsNullOrEmpty(rowStation)
            && !string.IsNullOrEmpty(cardStation)
            && !string.IsNullOrEmpty(primaryLocSlug)
            && string.Equals(cardStation, primaryLocSlug, StringComparison.Ordinal))
            return true;
        return false;
    }

    private static (string Target, string? Station, int? WindowHours) ParseCompositeForHistory(string composite)
    {
        var parts = composite.Split('/');
        if (parts.Length == 1) return (parts[0], null, null);
        if (parts.Length == 2) return (parts[0], parts[1], null);
        if (parts.Length == 3 && parts[2].EndsWith("h", StringComparison.Ordinal)
            && int.TryParse(parts[2][..^1], out var w))
            return (parts[0], parts[1], w);
        return (composite, null, null);
    }

    private static string RenderBlenderCard(string composite, ModelSummary m,
        ModelSummary latestCycle,
        ModelSummary? previous,
        IReadOnlyList<WeatherBlend.Models.VerifyHistoryFile> verifyHistory,
        string primaryLocSlug)
    {
        var description = PhaseDescription(composite, m.Phase);
        var phaseLabel = string.IsNullOrEmpty(m.Phase) ? "—" : m.Phase;
        // Badge compares the latest training cycle against the previous one
        // — each merged across its same-day split versions (see
        // MergeTrainingCycle). The merged ModelSummary `m` continues to
        // drive the metrics table below.
        var deltaBadge = RenderTrainingDeltaBadge(latestCycle, previous);

        var tbody = new StringBuilder();
        // ForecastsTempRain (= Leads.Full + lead 12) so 2d/3d's lead-12 row
        // surfaces in their card. Legacy phases (2b/2c/3a/3c) continue on
        // missing lead 12 — same effective render as before for them.
        foreach (var lead in Leads.ForecastsTempRain)
        {
            if (!m.PerLead.TryGetValue(lead, out var s)) continue;

            // Δ% vs best single, computed on the SAME test slice (apples-to-apples).
            // BestSingleTestMae populated from 2026-04-29+; older artefacts fall back
            // to the val number with a "(val)" tag so the mismatch is visible.
            var bestRef = s.BestSingleTestMae > 0 && !double.IsNaN(s.BestSingleTestMae)
                ? s.BestSingleTestMae : s.BestSingleValMae;
            var bestRefIsTest = s.BestSingleTestMae > 0 && !double.IsNaN(s.BestSingleTestMae);

            string deltaCell;
            if (bestRef > 0 && !double.IsNaN(bestRef) && !double.IsNaN(s.BlendTestScore))
            {
                var pct = (s.BlendTestScore - bestRef) / bestRef * 100.0;
                var cls = pct < 0 ? "delta-good" : "delta-bad";
                deltaCell = $"""<td class="num {cls}">{pct.ToString("+0.0;-0.0;0.0", Ci)}%</td>""";
            }
            else
            {
                deltaCell = """<td class="num">—</td>""";
            }

            var bestLabel = bestRefIsTest ? "" : " val";
            tbody.Append(Ci, $"""
                <tr>
                  <td>+{lead}h</td>
                  <td class="num"><strong>{s.BlendTestScore.ToString("0.000", Ci)}</strong></td>
                  <td class="num">{Escape(string.IsNullOrEmpty(s.BestSingle) ? "—" : s.BestSingle)} <small>({bestRef.ToString("0.000", Ci)}{bestLabel})</small></td>
                  {deltaCell}
                </tr>
                """);
        }

        var historyHtml = RenderVerifyHistorySection(composite, m.Phase, verifyHistory, m, primaryLocSlug);

        return $"""
            <article class="blender-card">
              <header>
                <hgroup>
                  <h4>Phase {Escape(phaseLabel)} <small>· <code>{Escape(m.Version)}</code></small> {deltaBadge}</h4>
                  <p class="skill-line">{Escape(description)} Trained {m.TrainedAtUtc:yyyy-MM-dd}. Metric: {Escape(m.MetricLabel)}.</p>
                </hgroup>
              </header>
              <table>
                <thead>
                  <tr>
                    <th>Lead</th>
                    <th class="num">Blend</th>
                    <th class="num">Best single</th>
                    <th class="num">Δ vs best</th>
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
              {historyHtml}
            </article>
            """;
    }

    /// <summary>
    /// "Δ vs previous train" badge for the blender card header. Compares
    /// the latest training run's average BlendTestScore against the
    /// previous training run's, on the SAME set of leads (intersection of
    /// PerLead keys present in both versions). Renders empty when:
    ///
    /// - <paramref name="previous"/> is null (first-ever train of this
    ///   phase, or only one version on disk).
    /// - The lead-sets are disjoint — guard against split-version phases
    ///   like 2d/3d where short and long halves train independently and
    ///   the previous-version snapshot might cover only one half.
    /// - Either side has a NaN BlendTestScore (legacy artefact prior to
    ///   the metric being persisted).
    ///
    /// Tooltip warns the reader the comparison is on the training-time
    /// test slice (which shifts each retrain) — verify history is the
    /// place to read real-world skill drift.
    ///
    /// Phase 4 of AUTO_RETRAIN_PLAN.md.
    /// </summary>
    private static string RenderTrainingDeltaBadge(ModelSummary latest, ModelSummary? previous)
    {
        if (previous is null) return "";

        var sharedLeads = latest.PerLead.Keys
            .Intersect(previous.PerLead.Keys)
            .Where(L => !double.IsNaN(latest.PerLead[L].BlendTestScore)
                     && !double.IsNaN(previous.PerLead[L].BlendTestScore))
            .ToList();
        if (sharedLeads.Count == 0) return "";

        var latestAvg = sharedLeads.Average(L => latest.PerLead[L].BlendTestScore);
        var previousAvg = sharedLeads.Average(L => previous.PerLead[L].BlendTestScore);
        var delta = latestAvg - previousAvg;

        // Lower metric = better (MAE for temp, Brier for precip, both are
        // 0-up scales). delta < 0 → latest improved → green; delta > 0 →
        // regressed → red.
        var cls = delta < 0 ? "delta-good" : "delta-bad";
        var sign = delta >= 0 ? "+" : "";
        var tooltip = "Training-slice delta — test window shifts each retrain; "
                      + "see verify history for real-world skill trend.";

        return $"""<span class="delta-badge {cls}" title="{Escape(tooltip)}">Δ {sign}{delta.ToString("0.000", Ci)} vs prev train</span>""";
    }

    /// <summary>
    /// Merge every <see cref="ModelSummary"/> of one training cycle — the
    /// versions of a (composite, phase) that share a calendar date — into a
    /// single summary whose <c>PerLead</c> spans the cycle's full lead
    /// coverage. 2d/3d routinely split a cycle into a short-leads version and
    /// a long-leads version; this stitches them back so the "Δ vs previous
    /// train" badge compares whole cycles rather than half a cycle. Header
    /// fields (version tag, TrainedAt) come from the newest version in the
    /// cycle; per lead, the newest version carrying that lead supplies the
    /// score. <paramref name="cycleVersions"/> must be non-empty and ordered
    /// newest-first.
    /// </summary>
    private static ModelSummary MergeTrainingCycle(IReadOnlyList<ModelSummary> cycleVersions)
    {
        var perLead = cycleVersions
            .SelectMany(s => s.PerLead)
            .GroupBy(kv => kv.Key)
            .ToDictionary(grp => grp.Key, grp => grp.First().Value);
        return cycleVersions[0] with { PerLead = perLead };
    }

    /// <summary>
    /// Per-card "Verify history" sub-section: one row per past weekly verify
    /// run for this card's (target, station, windowHours, phase) lineage,
    /// with per-lead blend metric columns and a drift indicator. Filters
    /// <paramref name="verifyHistory"/> to runs whose <c>Target</c>,
    /// <c>Station</c>, <c>WindowHours</c> and <c>Phase</c> match the card,
    /// sorted newest-first.
    ///
    /// Phase-based (not version-based) match is deliberate: the card always
    /// represents the current Active version of a phase, but verify can only
    /// score versions whose paired truth window has fully landed (the
    /// 5-day ERA5 latency etc.). A freshly retrained champion therefore
    /// has no exact-version verify rows for ~5 days. Matching by phase
    /// keeps the same lineage's history visible across retrains.
    ///
    /// Renders nothing when no matching history is on disk — keeps the card
    /// clean for fresh-deploy / pre-first-verify states.
    /// </summary>
    private static string RenderVerifyHistorySection(string composite, string phase,
        IReadOnlyList<WeatherBlend.Models.VerifyHistoryFile> verifyHistory,
        ModelSummary trainedSummary,
        string primaryLocSlug)
    {
        if (verifyHistory.Count == 0 || string.IsNullOrEmpty(phase)) return "";
        var (target, station, windowHours) = ParseCompositeForHistory(composite);

        // Find every (file, row) pair that matches this card. Each matching
        // file contributes one row to the history table; the per-lead values
        // come from rows in that file. Filter to the trained-lead view (the
        // historical row set) so the actual-lead-bucket rows added 2026-05-05
        // don't double-count into the per-lead columns. Bucket rows render
        // in a separate sibling table below.
        static bool IsTrainedView(WeatherBlend.Models.VerifyHistoryRow r)
            => r.ViewKind is null or "trained";

        var matchingFiles = verifyHistory
            .Where(f => f.Target == target)
            .Select(f =>
            {
                var rows = f.Rows.Where(r =>
                        IsTrainedView(r)
                        && string.Equals(r.Phase, phase, StringComparison.Ordinal)
                        && StationMatchesCard(r.Station, station, primaryLocSlug)
                        && r.WindowHours == windowHours)
                    .ToList();
                return (File: f, Rows: rows);
            })
            .Where(x => x.Rows.Count > 0)
            .OrderByDescending(x => x.File.AsOfUtc)
            .ToList();

        if (matchingFiles.Count == 0)
        {
            // Empty-state explanation rather than silently omitting the section.
            // Two ways a card lands here:
            //   1. The phase is genuinely brand-new and verify hasn't run since
            //      training (will populate at the next weekly cycle + 5d ERA5 lag).
            //   2. The phase tag drifted between trainings (e.g. an old "2b_redo"
            //      version's verify rows can't match a current "2b" card). The
            //      message names the phase being filtered for so the cause is
            //      visible without diff'ing JSON files.
            return $"""
                <div class="verify-history">
                  <h5>Verify history <small>(no runs yet)</small></h5>
                  <p class="skill-line">No verify rows yet for phase <code>{Escape(phase)}</code>. Next cycle: Mon/Thu 09:30 UTC, then 5d ERA5 latency.</p>
                </div>
                """;
        }

        // Lead set for the columns: canonical Leads.Full (24/48/72/96/120) plus
        // any extra leads present in the data (so 2d/3d's +12h column appears
        // for those phases). Using Leads.Full as a floor — rather than a pure
        // data-driven union — keeps the canonical long-lead columns visible
        // even when this week's runs happen not to have 96/120 rows (e.g.
        // because a freshly-promoted champion hasn't accumulated long-lead
        // history yet). Missing cells render as an em-dash.
        var dataLeads = matchingFiles
            .SelectMany(x => x.Rows.Select(r => r.LeadHours))
            .ToHashSet();
        var allLeads = WeatherBlend.Train.Common.Leads.Full
            .Concat(dataLeads)
            .Distinct()
            .OrderBy(l => l)
            .ToList();
        var metricLabel = matchingFiles[0].File.MetricLabel;

        var tbody = new StringBuilder();
        foreach (var (file, rows) in matchingFiles)
        {
            // A single verify file can contain multiple versions of the same
            // phase (e.g. v..._phase3a from Apr 23 + v..._phase3a from Apr 26),
            // so per-lead values are picked from the latest version (lexicographic
            // sort works because every version starts with `vYYYY-MM-DD_HHmmss`).
            var byLead = rows
                .GroupBy(r => r.LeadHours)
                .ToDictionary(
                    g => g.Key,
                    g => g.OrderByDescending(r => r.ModelVersion, StringComparer.Ordinal).First());
            // The right-hand "Drift" column (with ⚠/✓ icons stacked) was
            // removed 2026-05-11 per user feedback: it duplicated the per-
            // lead red-text highlight without adding information. The
            // driftAnyBlend / driftAnyBest aggregations that fed it are
            // gone too. Per-lead red text remains as the only drift
            // indicator (driftCls below), and that's the source of truth.

            // Distinct versions scored across all leads in this run. Usually
            // one (the active version of this phase at verify-time), but a
            // recent retrain leaves the predict tree carrying both old + new
            // for a few days so verify scores both. Showing them all keeps
            // the reader honest about which version the row's numbers came
            // from — matters when the row's drift is on the OLD version.
            var versionsScored = byLead.Values
                .Select(r => r.ModelVersion)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(v => v, StringComparer.Ordinal)
                .ToList();
            var versionCellHtml = string.Join(
                "<br>",
                versionsScored.Select(v => $"<code>{Escape(v)}</code>"));

            // Sample size column — surfaces "this row scored on what?" right
            // next to the version. Max N across leads is the most informative
            // single number: it's the count for the shortest lead (24h) where
            // most predictions land. A drift signal on n=53 is real; on n=2 it
            // isn't. Verifier's MinDriftN gate already uses this, but having
            // the user see it explicitly avoids surprise about borderline cells.
            var maxN = byLead.Values.Max(r => r.N);
            var nCellHtml = $"""<td class="num"><span title="Max predictions per lead in this run">{maxN}</span></td>""";

            var leadCells = new StringBuilder();
            foreach (var lead in allLeads)
            {
                if (byLead.TryGetValue(lead, out var r))
                {
                    var driftCls = r.DriftFlag ? " delta-bad" : "";
                    // Best-NWP realtime metric stacked under the blend so the
                    // reader can compare drift in like units. Sourced from the
                    // verify command's per-row BestSingleName/Metric (live
                    // truth window, same NWPs the blender feeds on). Drift
                    // ratio uses the active phase's training baseline from
                    // ModelSummary.PerLead[lead].BestSingleTestMae — same
                    // 1.5× threshold as the blend's own DriftFlag, so a lit
                    // best-NWP cell means "the best raw NWP is also
                    // struggling vs. its training-time baseline" (i.e. weather
                    // is hard, not just the blender degrading).
                    string bestLine;
                    if (r.BestSingleMetric.HasValue && !string.IsNullOrEmpty(r.BestSingleName))
                    {
                        var trainedBaseline = trainedSummary.PerLead.TryGetValue(lead, out var pl)
                            ? pl.BestSingleTestMae : double.NaN;
                        var bestDrifted = !double.IsNaN(trainedBaseline) && trainedBaseline > 0
                            && r.BestSingleMetric.Value > 1.5 * trainedBaseline;
                        var bestCls = bestDrifted ? " delta-bad" : "";
                        bestLine = $"<small class=\"{bestCls}\">{Escape(r.BestSingleName!)}: {r.BestSingleMetric.Value.ToString("0.000", Ci)}</small>";
                    }
                    else
                    {
                        bestLine = "<small>—</small>";
                    }
                    leadCells.Append(Ci, $"""<td class="num"><span class="{driftCls.Trim()}">{r.BlendMetric.ToString("0.000", Ci)}</span><br>{bestLine}</td>""");
                }
                else
                {
                    leadCells.Append("""<td class="num">—</td>""");
                }
            }

            tbody.Append(Ci, $"""
                <tr>
                  <td><time datetime="{file.AsOfUtc:yyyy-MM-dd}">{file.AsOfUtc:ddd yyyy-MM-dd}</time></td>
                  <td><small>{versionCellHtml}</small></td>
                  {nCellHtml}
                  {leadCells}
                </tr>
                """);
        }

        var leadHeaders = new StringBuilder();
        foreach (var lead in allLeads)
            leadHeaders.Append(Ci, $"""<th class="num">+{lead}h</th>""");

        var bucketTable = RenderActualLeadBucketTable(target, station, windowHours, phase, verifyHistory, primaryLocSlug);

        return $"""
            <div class="verify-history">
              <h5>Verify history <small>({matchingFiles.Count} run{(matchingFiles.Count == 1 ? "" : "s")})</small></h5>
              <p class="skill-line">Mon + Thu rolling {Escape(metricLabel)}. Per-lead cells turn red when the rolling metric breaches the lead-specific drift threshold; check the verify report for the per-cell breakdown. Version column names the trained model — a fresh champion takes ~5-9d to show.</p>
              <table>
                <thead>
                  <tr>
                    <th>Run (UTC)</th>
                    <th>Version</th>
                    <th class="num" title="Max predictions per lead in this run">N</th>
                    {leadHeaders}
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
            {bucketTable}</div>
            """;
    }

    /// <summary>
    /// Sibling sub-table to the verify-history view, rendered below the
    /// trained-lead table. Buckets predictions by their ACTUAL lead at
    /// prediction time (<c>ValidTime − PredictionMadeAt</c>) in 6h slices,
    /// so the reader can see whether MAE varies within a trained-lead bucket
    /// (relevant after the 2026-05-04 hourly-temp-predict change which
    /// spread each trained bucket across actual leads L .. L+23). Empty
    /// string when no bucket rows are present in the verify sidecars
    /// (older sidecars written before the 2026-05-05 schema bump).
    /// </summary>
    private static string RenderActualLeadBucketTable(
        string target, string? station, int? windowHours, string phase,
        IReadOnlyList<WeatherBlend.Models.VerifyHistoryFile> verifyHistory,
        string primaryLocSlug)
    {
        var matching = verifyHistory
            .Where(f => f.Target == target)
            .Select(f =>
            {
                var rows = f.Rows.Where(r =>
                        r.ViewKind == "actual_6h"
                        && string.Equals(r.Phase, phase, StringComparison.Ordinal)
                        && StationMatchesCard(r.Station, station, primaryLocSlug)
                        && r.WindowHours == windowHours)
                    .ToList();
                return (File: f, Rows: rows);
            })
            .Where(x => x.Rows.Count > 0)
            .OrderByDescending(x => x.File.AsOfUtc)
            .ToList();
        if (matching.Count == 0) return "";

        // Bucket lows union across runs, sorted ascending. Each bucket spans
        // [low, low+6) hours of actual lead.
        var allBuckets = matching
            .SelectMany(x => x.Rows
                .Where(r => r.ActualLeadBucketLowH.HasValue)
                .Select(r => r.ActualLeadBucketLowH!.Value))
            .Distinct()
            .OrderBy(b => b)
            .ToList();
        if (allBuckets.Count == 0) return "";

        var tbody = new StringBuilder();
        foreach (var (file, rows) in matching)
        {
            // Latest version per bucket within this run (same dedupe rule
            // as the trained-lead table — lex-sort works because every
            // version starts with vYYYY-MM-DD_HHmmss).
            var byBucket = rows
                .Where(r => r.ActualLeadBucketLowH.HasValue)
                .GroupBy(r => r.ActualLeadBucketLowH!.Value)
                .ToDictionary(
                    g => g.Key,
                    g => g.OrderByDescending(r => r.ModelVersion, StringComparer.Ordinal).First());

            var bucketCells = new StringBuilder();
            foreach (var bucket in allBuckets)
            {
                if (byBucket.TryGetValue(bucket, out var r))
                    bucketCells.Append(Ci, $"""<td class="num">{r.BlendMetric.ToString("0.000", Ci)}</td>""");
                else
                    bucketCells.Append("""<td class="num">—</td>""");
            }
            tbody.Append(Ci, $"""
                <tr>
                  <td><time datetime="{file.AsOfUtc:yyyy-MM-dd}">{file.AsOfUtc:ddd yyyy-MM-dd}</time></td>
                  {bucketCells}
                </tr>
                """);
        }

        var bucketHeaders = new StringBuilder();
        foreach (var bucket in allBuckets)
            bucketHeaders.Append(Ci, $"""<th class="num">{bucket}-{bucket + 5}h</th>""");

        return $"""
            <h6 style="margin-top:0.6rem">By actual lead at prediction time <small>(6h buckets)</small></h6>
            <p class="skill-line">Same data grouped by <code>ValidTime − PredictionMadeAt</code> (6h buckets) instead of trained-lead label. Reveals MAE structure within a trained bucket once predict spread to hourly outputs (2026-05-04+).</p>
            <table>
              <thead>
                <tr>
                  <th>Run (UTC)</th>
                  {bucketHeaders}
                </tr>
              </thead>
              <tbody>
            {tbody}    </tbody>
            </table>
            """;
    }

    /// <summary>
    /// True iff the (target, phase) pair should render on the Models page.
    /// Thin adapter over <see cref="ActivePhasePolicy.IsActive"/> that splits
    /// the composite key into its target prefix — the page filters every
    /// version emitted into the rolling window through this so retired
    /// phases (still on disk) don't leak onto the cards.
    /// </summary>
    private static bool IsActivePhase(string composite, string phase)
        => ActivePhasePolicy.IsActive(composite.Split('/')[0], phase);

    /// <summary>
    /// Sort key for a phase within a composite — index in the
    /// champion-first <see cref="ActivePhasePolicy"/> list, so lean ranks
    /// above rich. Unknown phases sort last.
    /// </summary>
    private static int PhasePriority(string composite, string phase)
        => ActivePhasePolicy.Priority(composite.Split('/')[0], phase);

    /// <summary>
    /// One-sentence prose summary of what each phase actually does — composed for
    /// the Models page so the reader doesn't need to know the codebase to read the
    /// table. Keeps the per-blender card honest without trying to recompute feature
    /// counts at render time. Falls back to "Phase {x} blender." for unknown phases.
    /// </summary>
    private static string PhaseDescription(string composite, string phase)
    {
        var target = composite.Split('/')[0];
        return (target, phase) switch
        {
            ("temperature", "2b")  => "Lean blender, 13 features.",
            ("temperature", "2c")  => "Rich blender, 88 features (adds dew/RH/cloud/wind/pressure).",
            // 2d ships once UKV inclusion is decided (2026-05-05+). Description
            // here so the card renders correctly the moment ActivePhasePolicy
            // gets "2d" added to the temperature list.
            ("temperature", "2d")  => "Exact-runtime blender. Trains on raw S3 cycles (GFS + AIFS required, IFS oper + MO Global + UKV optional) instead of Open-Meteo offset_day, with rigorous (RunTime, ValidTime, Lead) provenance per row. UKV pulled per-V-hour from 03Z + 15Z cycles.",
            ("precipitation", "3a") => "Lean P(wet) classifier, 27 features.",
            ("precipitation", "3c") => "Rich P(wet) classifier, 55 features.",
            // 3d ships once UKV's lead-aware backfill (leads {9,15,21,27}) lands
            // on R2 and a real artefact has been trained. Description here so
            // the card renders correctly the moment ActivePhasePolicy gets "3d"
            // added to the precipitation list.
            ("precipitation", "3d") => "Exact-runtime P(wet) classifier. Trains on raw S3 cycles (GFS + IFS oper + AIFS required, MO Global + UKV optional) instead of Open-Meteo offset_day, with rigorous (RunTime, ValidTime, Lead) provenance per row. UKV pulled per-V-hour with target-lead-aware tuples.",
            ("dry_window", "3b")   => "53-feature LightGBM per-(station, window).",
            ("dry_window", "3g")   => "Parameter-free MC over 3a hourly P(wet). Monotonic by construction.",
            ("dry_window", "3j")   => "Gaussian copula MC over 3a hourly P(wet) with a per-(station, lead) 9×9 Σ fit on observed daytime binary sequences. Captures within-day wet/dry autocorrelation that 3g's iid sampler misses.",
            ("dry_window", "3n")   => "Regime-conditioned copula MC. Two Σs per (station, lead): Σ_settled (train days where NWPs agreed on hourly wet/dry) and Σ_unsettled (disagreement days). Per-day live NWP agreement picks which Σ to apply.",
            _ => $"Phase {phase} blender.",
        };
    }

    /// <summary>
    /// Pretty-print a composite key. Examples:
    ///   "temperature"                              → "Temperature"
    ///   "precipitation/ea_bellever_dartmoor"       → "Precipitation — Bellever Dartmoor"
    ///   "dry_window/ea_bellever_dartmoor/6h"       → "Dry window — Bellever Dartmoor — 6-hour"
    /// </summary>
    private static string PrettyComposite(string key)
    {
        var parts = key.Split('/');
        if (parts.Length == 0) return key;

        var target = parts[0] switch
        {
            "temperature" => "Temperature",
            "precipitation" => "Precipitation",
            "dry_window" => "Dry window",
            _ => parts[0],
        };
        if (parts.Length == 1) return target;

        var station = PrettyStation(parts[1]);
        if (parts.Length == 2) return $"{target} — {station}";

        // dry-window: third part is "{N}h"
        var window = parts[2].TrimEnd('h');
        return $"{target} — {station} — {window}-hour";
    }
}
