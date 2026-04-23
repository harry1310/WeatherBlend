using System.Globalization;
using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// Pure HTML renderers for the static site. No I/O, no DuckDB — takes already-loaded
/// model objects and returns strings. Callers (see <c>RenderSiteCommand</c>) own the
/// query-and-write orchestration.
/// </summary>
public static class SitePages
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
    }

    public sealed record RollingMaePoint(string ModelVersion, int LeadHours, DateTime WindowEndUtc, double BlendMae, int N);

    public sealed record PrecipForecastPoint(
        string Station,
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double ProbWet,
        double ClimatologyPWet);

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

    public static string RenderIndex(SiteInputs input)
    {
        // Cards are "what's the blender saying now?" — always the champion, never a
        // challenger that happens to have also written the same (lead, valid_time).
        // Empty CurrentVersion means no manifest read (legacy), so fall back to "any".
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion)
            ? input.Predictions
            : input.Predictions.Where(p => p.ModelVersion == input.CurrentVersion).ToList();

        var latestByLead = cardSource
            .GroupBy(p => p.LeadHours)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First());

        var cards = new StringBuilder();
        foreach (var lead in new[] { 24, 48, 72 })
        {
            if (latestByLead.TryGetValue(lead, out var p))
            {
                cards.Append(Ci, $"""
                    <article class="forecast-card">
                      <header><h3>+{lead}h</h3><small>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</small></header>
                      <div class="temp">{p.BlendTemperature.ToString("0.0", Ci)}°C</div>
                      <footer>
                        <small>Made {p.PredictionMadeAtUtc:yyyy-MM-dd HH:mm}Z</small><br/>
                        <small>Model: <code>{Escape(p.ModelVersion)}</code></small>
                      </footer>
                    </article>
                    """);
            }
            else
            {
                cards.Append(Ci, $"""
                    <article class="forecast-card forecast-card-empty">
                      <header><h3>+{lead}h</h3></header>
                      <div class="temp">—</div>
                      <footer><small>No prediction available</small></footer>
                    </article>
                    """);
            }
        }

        var skill = ComputeHeadlineSkill(input);
        var versionNote = string.IsNullOrEmpty(input.CurrentVersion)
            ? "No champion pinned in MANIFEST — cards may drift between active versions."
            : $"Champion version: <code>{Escape(input.CurrentVersion)}</code>. Charts comparing every active version against truth live on the <a href=\"forecast-vs-truth.html\">forecast vs truth page</a>.";

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Latest blended forecast</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m</p>
              </hgroup>
              <div class="forecast-grid">
            {cards}  </div>
              <p class="skill-line">{versionNote}</p>
              <p class="skill-line">{Escape(skill)}</p>
            </section>
            """);

        return WrapPage(input, "Home", "index", body.ToString());
    }

    public static string RenderPredictions(SiteInputs input)
    {
        var rows = input.Predictions
            .OrderByDescending(p => p.PredictionMadeAtUtc)
            .ThenBy(p => p.LeadHours)
            .ToList();

        var tableBody = new StringBuilder();
        foreach (var p in rows)
        {
            var truth = input.TruthByTime.TryGetValue(p.ValidTimeUtc, out var t) ? t : (double?)null;
            var err = truth.HasValue ? p.BlendTemperature - truth.Value : (double?)null;
            tableBody.Append(Ci, $"""
                <tr>
                  <td><time>{p.PredictionMadeAtUtc:yyyy-MM-dd HH:mm}Z</time></td>
                  <td><time>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</time></td>
                  <td>{p.LeadHours}h</td>
                  <td class="num"><strong>{p.BlendTemperature.ToString("0.00", Ci)}</strong></td>
                  <td class="num">{FmtNullable(truth)}</td>
                  <td class="num">{FmtNullable(err, "+0.00;-0.00;0.00")}</td>
                  <td class="num">{FmtNullable(p.TempGfs)}</td>
                  <td class="num">{FmtNullable(p.TempEcmwf)}</td>
                  <td class="num">{FmtNullable(p.TempIcon)}</td>
                  <td class="num">{FmtNullable(p.TempMf)}</td>
                  <td class="num">{FmtNullable(p.TempUkmo)}</td>
                  <td class="num">{FmtNullable(p.TempGem)}</td>
                  <td><small><code>{Escape(p.ModelVersion)}</code></small></td>
                </tr>
                """);
        }

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Predictions</h2>
                <p>{rows.Count} rows, sorted by prediction time (newest first).</p>
              </hgroup>
              <figure>
                <table>
                  <thead>
                    <tr>
                      <th>Made at</th>
                      <th>Valid time</th>
                      <th>Lead</th>
                      <th class="num">Blend</th>
                      <th class="num">ERA5</th>
                      <th class="num">Err</th>
                      <th class="num">GFS</th>
                      <th class="num">ECMWF</th>
                      <th class="num">ICON</th>
                      <th class="num">MF</th>
                      <th class="num">UKMO</th>
                      <th class="num">GEM</th>
                      <th>Version</th>
                    </tr>
                  </thead>
                  <tbody>
            {tableBody}      </tbody>
                </table>
              </figure>
            </section>
            """);

        return WrapPage(input, "Predictions", "predictions", body.ToString());
    }

    public static string RenderPrecipitation(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Precipitation — P(wet ≥ 0.1 mm/h)</h2>
                <p>Per-station hourly occurrence blender. EA Hydrology gauges as truth; LightGBM per lead-time bucket.
                   Three blenders are active per station — <strong>Phase 3a (lean, 27 features)</strong> as the original champion,
                   <strong>Phase 3a_isotonic</strong> (3a's output re-mapped by a per-lead pool-adjacent-violators isotonic regression
                   fit on the validation slice), and <strong>Phase 3c (rich, 55 features)</strong> as the feature-richness challenger
                   with per-model humidity, surface pressure, and EA trailing-rainfall persistence. All three produce predictions
                   every cycle; charts below are grouped by phase so differences between feature richness and post-hoc calibration
                   are visible at a glance.</p>
              </hgroup>
            """);

        if (input.PrecipPredictions.Count == 0)
        {
            content.Append("<p><em>No precipitation predictions in window. Run <code>predict --target precipitation --truth-station all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Precipitation", "precipitation", content.ToString());
        }

        var stations = input.PrecipPredictions.Select(p => p.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();

        // Lead-bucket palette matches the temperature overview chart.
        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),
            (48, "#7c4dff"),
            (72, "#4527a0"),
        };

        // Phase specs mirror the temperature forecast-vs-truth page: one chart per
        // phase, with leads as separate series inside. "other" catches pre-3a rows
        // or versions with no training_metadata.json on disk.
        (string Key, string Title, string Description)[] phaseSpecs =
        {
            ("3a", "Phase 3a — lean (27 features)",
                "Per-model precip + precip-prob, spread stats, covariate means, calendar encodings. Original champion."),
            ("3a_isotonic", "Phase 3a_isotonic — lean + post-hoc calibration",
                "Phase 3a's model unchanged; its output is remapped through a per-lead pool-adjacent-violators isotonic regression fit on the validation slice. Tests whether calibration post-processing alone delivers what Phase 3c's extra features deliver."),
            ("3c", "Phase 3c — rich (55 features)",
                "Lean + 18 per-model humidity (dew/RH/depression) + 6 per-model surface pressure + 4 EA trailing-rainfall persistence features. Challenger."),
            ("other", "Other versions",
                "Versions with no phase tag on disk — typically pre-3a experiments left in the manifest."),
        };

        foreach (var station in stations)
        {
            var stationRows = input.PrecipPredictions.Where(p => p.Station == station).ToList();

            // Headline: peak P(wet) at lead 24h over the next 72h, drawn from 3c if
            // available (the challenger is better-calibrated at +24h), otherwise 3a.
            var phaseForHeadline = stationRows.Any(r => BucketPrecipPhase(input.PhaseByVersion, r.Version) == "3c")
                ? "3c" : "3a";
            var headlineRows = stationRows
                .Where(r => BucketPrecipPhase(input.PhaseByVersion, r.Version) == phaseForHeadline
                         && r.LeadHours == 24
                         && r.ValidTimeUtc >= input.GeneratedAtUtc)
                .GroupBy(r => r.ValidTimeUtc)
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .ToList();
            string headline;
            if (headlineRows.Count > 0)
            {
                var peak = headlineRows.OrderByDescending(r => r.ProbWet).First();
                var meanP = headlineRows.Average(r => r.ProbWet);
                headline = $"Next {headlineRows.Count}h at +24h (Phase {phaseForHeadline}): mean P(wet) = {meanP.ToString("0.00", Ci)}, peak {peak.ProbWet.ToString("0.00", Ci)} at {peak.ValidTimeUtc:yyyy-MM-dd HH:mm}Z.";
            }
            else
            {
                headline = "No +24h-lead forecast in the future window.";
            }

            content.Append(Ci, $"""
                <article>
                  <hgroup>
                    <h3>{Escape(PrettyStation(station))}</h3>
                    <p class="skill-line">{Escape(headline)}</p>
                  </hgroup>
                """);

            // One chart per phase present for this station.
            bool anyRendered = false;
            foreach (var spec in phaseSpecs)
            {
                var phaseRows = stationRows
                    .Where(r => BucketPrecipPhase(input.PhaseByVersion, r.Version) == spec.Key)
                    .ToList();
                if (phaseRows.Count == 0) continue;
                anyRendered = true;

                var latestPerLead = phaseRows
                    .GroupBy(r => (r.LeadHours, r.ValidTimeUtc))
                    .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                    .ToList();

                var series = new List<LineSeries>();
                foreach (var (lead, color) in leadSpecs)
                {
                    var pts = latestPerLead
                        .Where(r => r.LeadHours == lead && r.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
                        .OrderBy(r => r.ValidTimeUtc)
                        .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                        .ToList();
                    if (pts.Count > 0)
                        series.Add(new LineSeries($"Blend +{lead}h", color, pts));
                }

                // Climatology baseline — identical per valid time, pull it from whichever
                // lead has data and overlay as a reference line.
                var climPts = latestPerLead
                    .Where(r => r.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1))
                    .GroupBy(r => r.ValidTimeUtc)
                    .Select(g => g.First())
                    .OrderBy(r => r.ValidTimeUtc)
                    .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ClimatologyPWet))
                    .ToList();
                if (climPts.Count > 0)
                    series.Add(new LineSeries("Climatology", "#9e9e9e", climPts));

                content.Append(Ci, $"""
                    <h4>{Escape(spec.Title)}</h4>
                    <p class="skill-line">{Escape(spec.Description)}</p>
                    """);

                if (series.Count == 0)
                {
                    content.Append(RenderEmptyChart(
                        $"P(wet) — {PrettyStation(station)} — {spec.Key}",
                        "No forecast valid times in the future window."));
                }
                else
                {
                    content.Append(LineChartRenderer.Render(new LineChartSpec
                    {
                        Title = $"P(wet ≥ 0.1 mm/h) — {PrettyStation(station)} — Phase {spec.Key}",
                        XLabel = "Valid time (UTC)",
                        YLabel = "Probability",
                        Series = series,
                        Height = 300,
                        FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                        FormatY = v => v.ToString("0.00", Ci),
                    }));
                }
            }

            // Champion-vs-challenger overlay at +24h — the most skillful lead and
            // the one the dog-walk user actually cares about. Drops in if both
            // phases produced 24h rows for this station. "Now minus 1h" to anchor
            // at the forecast trajectory rather than full history.
            var cvc24 = BuildChampionVsChallengerSeries(stationRows, input.PhaseByVersion, input.GeneratedAtUtc.AddHours(-1), leadHours: 24);
            if (cvc24.Count >= 2)
            {
                content.Append(Ci, $"""
                    <h4>Three-way comparison — +24h lead</h4>
                    <p class="skill-line">3a (raw), 3a_isotonic (calibrated), and 3c (rich) on the same axis. If 3a_isotonic tracks 3c, calibration alone did the work the extra features were supposed to do.</p>
                    """);
                content.Append(LineChartRenderer.Render(new LineChartSpec
                {
                    Title = $"3a vs 3a_isotonic vs 3c — {PrettyStation(station)} — +24h",
                    XLabel = "Valid time (UTC)",
                    YLabel = "Probability",
                    Series = cvc24,
                    Height = 280,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                    FormatY = v => v.ToString("0.00", Ci),
                }));
            }

            if (!anyRendered)
            {
                content.Append(RenderEmptyChart($"P(wet) — {station}", "No forecast valid times in the future window."));
            }

            content.Append("</article>");
        }

        // Latest-per-(station, phase, lead, valid_time) detail table, capped to the
        // next 96h forward of generation to keep page weight reasonable.
        var future = input.PrecipPredictions
            .Where(p => p.ValidTimeUtc >= input.GeneratedAtUtc.AddHours(-1) && p.ValidTimeUtc <= input.GeneratedAtUtc.AddHours(96))
            .GroupBy(p => (p.Station, p.Version, p.LeadHours, p.ValidTimeUtc))
            .Select(g => g.OrderByDescending(p => p.PredictedAtUtc).First())
            .OrderBy(p => p.Station, StringComparer.Ordinal)
            .ThenBy(p => p.ValidTimeUtc)
            .ThenBy(p => p.LeadHours)
            .ThenBy(p => BucketPrecipPhase(input.PhaseByVersion, p.Version), StringComparer.Ordinal)
            .ToList();

        if (future.Count > 0)
        {
            var tbody = new StringBuilder();
            foreach (var p in future)
            {
                var phase = BucketPrecipPhase(input.PhaseByVersion, p.Version);
                tbody.Append(Ci, $"""
                    <tr>
                      <td>{Escape(PrettyStation(p.Station))}</td>
                      <td>{Escape(phase)}</td>
                      <td><time>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</time></td>
                      <td>{p.LeadHours}h</td>
                      <td class="num"><strong>{p.ProbWet.ToString("0.00", Ci)}</strong></td>
                      <td class="num">{p.ClimatologyPWet.ToString("0.00", Ci)}</td>
                      <td><small><code>{Escape(p.Version)}</code></small></td>
                    </tr>
                    """);
            }

            content.Append(Ci, $"""
                <h3>Hourly detail (next 96h)</h3>
                <figure>
                  <table>
                    <thead>
                      <tr>
                        <th>Station</th>
                        <th>Phase</th>
                        <th>Valid time</th>
                        <th>Lead</th>
                        <th class="num">P(wet)</th>
                        <th class="num">Climatology</th>
                        <th>Version</th>
                      </tr>
                    </thead>
                    <tbody>
                {tbody}    </tbody>
                  </table>
                </figure>
                """);
        }

        content.Append("</section>");
        return WrapPage(input, "Precipitation", "precipitation", content.ToString());
    }

    /// <summary>
    /// Bucket a precipitation version into its phase tag. Precip phases are exact
    /// strings — "3a", "3a_isotonic", or "3c" — so anything else lands in "other".
    /// </summary>
    private static string BucketPrecipPhase(IReadOnlyDictionary<string, string> phaseByVersion, string version)
    {
        if (!phaseByVersion.TryGetValue(version, out var phase) || string.IsNullOrWhiteSpace(phase))
            return "other";
        if (phase.Equals("3a", StringComparison.OrdinalIgnoreCase)) return "3a";
        if (phase.Equals("3a_isotonic", StringComparison.OrdinalIgnoreCase)) return "3a_isotonic";
        if (phase.Equals("3c", StringComparison.OrdinalIgnoreCase)) return "3c";
        return "other";
    }

    /// <summary>
    /// Build "3a vs 3c" line series at a single lead for one station. Caller sets the
    /// earliest valid time to include — the precipitation page uses "now minus 1h" to
    /// get the forecast trajectory, while forecast-vs-truth uses <c>WindowStartUtc</c>
    /// to see historical predictions alongside the observed rainfall line.
    /// </summary>
    private static List<LineSeries> BuildChampionVsChallengerSeries(
        IReadOnlyList<PrecipForecastPoint> stationRows,
        IReadOnlyDictionary<string, string> phaseByVersion,
        DateTime minValidTimeUtc,
        int leadHours)
    {
        var series = new List<LineSeries>();
        (string phase, string label, string color)[] spec =
        {
            ("3a",          "Phase 3a (champion)",            "#90a4ae"),
            ("3a_isotonic", "Phase 3a_isotonic (calibrated)", "#26a69a"),
            ("3c",          "Phase 3c (challenger)",          "#7c4dff"),
        };
        foreach (var (phase, label, color) in spec)
        {
            var pts = stationRows
                .Where(r => r.LeadHours == leadHours
                         && BucketPrecipPhase(phaseByVersion, r.Version) == phase
                         && r.ValidTimeUtc >= minValidTimeUtc)
                .GroupBy(r => r.ValidTimeUtc)
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .OrderBy(r => r.ValidTimeUtc)
                .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries(label, color, pts));
        }
        return series;
    }

    public static string RenderDryWindow(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("""
            <section>
              <hgroup>
                <h2>Dry window — P(∃ N-hour dry block in target UTC day)</h2>
                <p>Phase 3b per-station, per-window blender. 18 LightGBM classifiers (Bellever + Princetown × {3, 4, 6}h × leads 24/48/72h). Truth from EA Hydrology gauges with a 4-of-4 hourly gate.</p>
              </hgroup>
            """);

        if (input.DryWindowPredictions.Count == 0)
        {
            content.Append("<p><em>No dry-window predictions in window. Run <code>predict --target dry-window --truth-station all --window all</code>.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Dry window", "dry-window", content.ToString());
        }

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = new[] { 24, 48, 72 };

        foreach (var station in stations)
        {
            content.Append(Ci, $"<h3>{Escape(PrettyStation(station))}</h3>");

            foreach (var window in windows)
            {
                // Latest prediction per (target_date, lead). Keep target dates strictly
                // forward of "yesterday" to avoid stale rows dominating the view.
                var cutoff = input.GeneratedAtUtc.Date.AddDays(-1);
                var latest = input.DryWindowPredictions
                    .Where(d => d.Station == station && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

                if (latest.Count == 0)
                {
                    content.Append(Ci, $"<h4>{window}-hour dry window</h4><p><em>No predictions on or after {cutoff:yyyy-MM-dd}.</em></p>");
                    continue;
                }

                var dates = latest.Select(d => d.TargetDateUtc).Distinct().OrderBy(d => d).ToList();

                var tbody = new StringBuilder();
                foreach (var date in dates)
                {
                    var byLead = latest.Where(d => d.TargetDateUtc == date).ToDictionary(d => d.LeadHours);
                    var anyClim = byLead.Values.FirstOrDefault();
                    var clim = anyClim is null ? "—" : anyClim.ClimatologyProbHasDryWindow.ToString("0.00", Ci);

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                        {
                            // Bold cell when the blender meaningfully diverges from climatology.
                            var diff = d.ProbHasDryWindow - d.ClimatologyProbHasDryWindow;
                            var cls = Math.Abs(diff) >= 0.10 ? " class=\"num strong\"" : " class=\"num\"";
                            leadCells.Append(Ci, $"<td{cls}>{d.ProbHasDryWindow.ToString("0.00", Ci)}</td>");
                        }
                        else
                        {
                            leadCells.Append("<td class=\"num\">—</td>");
                        }
                    }

                    var agreementCell = byLead.Values
                        .Select(d => d.AgreementHasDryWindow)
                        .FirstOrDefault(a => a.HasValue);
                    var agreement = agreementCell.HasValue ? agreementCell.Value.ToString("0.00", Ci) : "—";

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time>{date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          <td class="num">{clim}</td>
                          <td class="num">{agreement}</td>
                        </tr>
                        """);
                }

                content.Append(Ci, $"""
                    <h4>{window}-hour dry window</h4>
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
                            <th class="num">Climatology</th>
                            <th class="num">Model agreement</th>
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);
            }
        }

        content.Append("""
            <p class="skill-line">A dry "hour" requires all four 15-min EA gauge readings to be ≤ 0.1 mm. Cross-midnight dry stretches are not credited (UTC-day boundary). Daylight filtering is deferred to the application layer.</p>
            </section>
            """);
        return WrapPage(input, "Dry window", "dry-window", content.ToString());
    }

    public static string RenderVerify(SiteInputs input)
    {
        var content = new StringBuilder();

        content.Append("""
            <section>
              <hgroup>
                <h2>Verification</h2>
                <p>Rolling blend-vs-ERA5 MAE by lead time, stratified by model version.</p>
              </hgroup>
            """);

        if (input.RollingMae.Count == 0)
        {
            content.Append("<p><em>No rolling MAE points computed — the window is too short or there's no matching ERA5 truth yet.</em></p>");
        }
        else
        {
            // One chart per lead, with one line per model version.
            foreach (var lead in new[] { 24, 48, 72 })
            {
                var versions = input.RollingMae.Where(r => r.LeadHours == lead)
                    .Select(r => r.ModelVersion).Distinct().OrderBy(v => v, StringComparer.Ordinal).ToList();

                var series = new List<LineSeries>();
                var palette = new[] { "#7c4dff", "#26a69a", "#ef5350", "#ffa726", "#42a5f5" };
                for (int i = 0; i < versions.Count; i++)
                {
                    var v = versions[i];
                    var pts = input.RollingMae
                        .Where(r => r.LeadHours == lead && r.ModelVersion == v)
                        .OrderBy(r => r.WindowEndUtc)
                        .Select(r => (X: r.WindowEndUtc.ToOADate(), Y: r.BlendMae))
                        .ToList();
                    if (pts.Count > 0)
                    {
                        series.Add(new LineSeries(v, palette[i % palette.Length], pts));
                    }
                }

                if (series.Count == 0)
                {
                    content.Append(Ci, $"<h3>Lead {lead}h</h3>");
                    content.Append(RenderEmptyChart($"Rolling MAE — lead {lead}h", "No scored predictions at this lead."));
                    continue;
                }

                content.Append(Ci, $"<h3>Lead {lead}h</h3>");
                content.Append(LineChartRenderer.Render(new LineChartSpec
                {
                    Title = $"Rolling MAE — lead {lead}h",
                    XLabel = "Window end (UTC)",
                    YLabel = "MAE (°C)",
                    Series = series,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                    FormatY = v => v.ToString("0.00", Ci),
                }));
            }
        }

        content.Append("</section>");
        return WrapPage(input, "Verification", "verify", content.ToString());
    }

    /// <summary>
    /// Dedicated page that plots blend forecasts against observed truth, so the user can
    /// eyeball whether a given model version is actually tracking reality. Temperature is
    /// split into two charts (one per phase: 2b lean vs 2c rich) so champion/challenger
    /// lines don't overlap and the user can compare at a glance. Precipitation gets one
    /// chart per station with observed hourly rainfall overlaid on the P(wet) forecasts.
    /// Dry-window is a grid of observed-vs-predicted for completed target days.
    /// </summary>
    public static string RenderForecastVsTruth(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Forecast vs truth</h2>
                <p>Side-by-side view of what each active blender predicted vs what actually happened.
                   Temperature lines run forward past "now" because the blend projects 24/48/72h ahead;
                   truth (ERA5, METAR, rainfall) stops at the last observation.</p>
              </hgroup>
            """);

        // --- Temperature by phase --------------------------------------------------
        var versionsByPhase = input.PhaseByVersion
            .GroupBy(kv => BucketPhase(kv.Value))
            .ToDictionary(g => g.Key, g => g.Select(kv => kv.Key).ToHashSet(StringComparer.Ordinal));
        // Versions in predictions but with no metadata phase go to "other" — at least
        // they still get rendered somewhere rather than being silently dropped.
        var untagged = input.Predictions
            .Select(p => p.ModelVersion).Distinct()
            .Where(v => !input.PhaseByVersion.ContainsKey(v))
            .ToHashSet(StringComparer.Ordinal);
        if (untagged.Count > 0)
        {
            if (!versionsByPhase.TryGetValue("other", out var others))
                versionsByPhase["other"] = new HashSet<string>(untagged, StringComparer.Ordinal);
            else
                foreach (var v in untagged) others.Add(v);
        }

        (string Key, string Title, string Description)[] phaseSpecs =
        {
            ("2b", "Temperature — Phase 2b lean (13 features)",
                "Six per-model temperatures, their mean/std/range, and cyclical hour/day-of-year encodings. The original champion."),
            ("2c", "Temperature — Phase 2c rich (88 features)",
                "Adds per-model dew point, RH, cloud {total/low/mid/high}, wind speed/dir/gusts, surface pressure, plus cross-model aggregates. Trained to challenge 2b."),
            ("other", "Temperature — other versions",
                "Versions with no training metadata on disk — typically pre-2b experiments left in the manifest."),
        };
        foreach (var spec in phaseSpecs)
        {
            if (!versionsByPhase.TryGetValue(spec.Key, out var versions) || versions.Count == 0)
                continue;

            var filtered = input.Predictions.Where(p => versions.Contains(p.ModelVersion)).ToList();
            content.Append(Ci, $"""
                <article>
                  <hgroup>
                    <h3>{Escape(spec.Title)}</h3>
                    <p class="skill-line">{Escape(spec.Description)}</p>
                  </hgroup>
                """);
            content.Append(RenderTempVsTruthChart(input, filtered, spec.Key));
            content.Append("</article>");
        }

        if (versionsByPhase.Count == 0)
        {
            content.Append("<p><em>No temperature predictions with metadata — run <code>train</code> and <code>predict</code>.</em></p>");
        }

        // --- Precipitation per station ---------------------------------------------
        content.Append("<hr/>");
        content.Append(Ci, $"""
            <h3>Precipitation — P(wet) vs observed rainfall</h3>
            <p class="skill-line">P(wet) is the blender's probability that the next hour sees ≥ 0.1 mm. The rainfall line is the same 4-of-4 hourly aggregation used for verification — partial hours are dropped to avoid flipping wet↔dry at the boundary.</p>
            """);
        var precipStations = input.PrecipPredictions.Select(p => p.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        if (precipStations.Count == 0)
        {
            content.Append("<p><em>No precipitation predictions in window.</em></p>");
        }
        else
        {
            foreach (var station in precipStations)
                content.Append(RenderPrecipVsTruthChart(input, station));
        }

        // --- Dry-window per station × window --------------------------------------
        content.Append("<hr/>");
        content.Append(Ci, $"""
            <h3>Dry window — predicted probability vs (eventual) observation</h3>
            <p class="skill-line">One row per target UTC day × window length. The "observed" column is blank for dates beyond the last full rainfall day. Rolling Brier skill scores are on the <a href="verify.html">verification page</a>.</p>
            """);
        content.Append(RenderDryWindowVsTruthTable(input));

        content.Append("</section>");
        return WrapPage(input, "Forecast vs truth", "forecast-vs-truth", content.ToString());
    }

    private static string BucketPhase(string? phase)
    {
        if (string.IsNullOrWhiteSpace(phase)) return "other";
        // "2b" and "2b_redo" both belong in the lean bucket. "2c" is rich. Anything else
        // that isn't one of those two is shown in the "other" bucket.
        if (phase.StartsWith("2b", StringComparison.OrdinalIgnoreCase)) return "2b";
        if (phase.Equals("2c", StringComparison.OrdinalIgnoreCase)) return "2c";
        return "other";
    }

    private static string RenderTempVsTruthChart(
        SiteInputs input,
        IReadOnlyList<PredictionRow> filtered,
        string phaseKey)
    {
        var series = new List<LineSeries>();

        var era5Pts = input.TruthByTime
            .Where(kv => kv.Key >= input.WindowStartUtc)
            .OrderBy(kv => kv.Key)
            .Select(kv => (X: kv.Key.ToOADate(), Y: kv.Value))
            .ToList();
        if (era5Pts.Count > 0)
            series.Add(new LineSeries("ERA5 (truth)", "#ef5350", era5Pts));

        var metarPts = input.MetarByTime
            .Where(m => m.ObservedTimeUtc >= input.WindowStartUtc)
            .OrderBy(m => m.ObservedTimeUtc)
            .Select(m => (X: m.ObservedTimeUtc.ToOADate(), Y: m.Temperature2m))
            .ToList();
        if (metarPts.Count > 0)
        {
            var label = string.IsNullOrWhiteSpace(input.MetarStation) ? "METAR" : $"METAR {input.MetarStation}";
            series.Add(new LineSeries(label, "#ffa726", metarPts));
        }

        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),
            (48, "#7c4dff"),
            (72, "#4527a0"),
        };
        foreach (var (lead, color) in leadSpecs)
        {
            // Latest prediction per valid-time — if the same (lead, valid_time) was
            // produced by two same-phase versions (rare, but possible across a retrain)
            // the newer PredictionMadeAtUtc wins.
            var pts = filtered
                .Where(p => p.LeadHours == lead)
                .GroupBy(p => p.ValidTimeUtc)
                .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
                .OrderBy(p => p.ValidTimeUtc)
                .Select(p => (X: p.ValidTimeUtc.ToOADate(), Y: p.BlendTemperature))
                .ToList();
            if (pts.Count > 0)
                series.Add(new LineSeries($"Blend +{lead}h", color, pts));
        }

        if (series.Count == 0)
            return RenderEmptyChart($"Temperature — phase {phaseKey}", "No overlap between predictions and truth in window.");

        return LineChartRenderer.Render(new LineChartSpec
        {
            Title = $"Temperature vs truth — phase {phaseKey}",
            XLabel = "Time (UTC)",
            YLabel = "Temperature (°C)",
            Series = series,
            Height = 360,
            FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
            FormatY = v => v.ToString("0.#", Ci) + "°",
        });
    }

    private static string RenderPrecipVsTruthChart(SiteInputs input, string station)
    {
        var content = new StringBuilder();
        content.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

        var stationPredictions = input.PrecipPredictions.Where(p => p.Station == station).ToList();

        // Precompute the observed-rainfall series once; it overlays on every phase chart.
        List<(double X, double Y)> truthPts = new();
        if (input.RainfallTruth.TryGetValue(station, out var truth) && truth.Count > 0)
        {
            truthPts = truth
                .Where(kv => kv.Key >= input.WindowStartUtc)
                .OrderBy(kv => kv.Key)
                .Select(kv => (X: kv.Key.ToOADate(), Y: Math.Min(1.0, kv.Value)))
                .ToList();
        }

        (int lead, string color)[] leadSpecs =
        {
            (24, "#b39ddb"),
            (48, "#7c4dff"),
            (72, "#4527a0"),
        };
        (string Key, string Title)[] phaseSpecs =
        {
            ("3a", "Phase 3a (lean)"),
            ("3a_isotonic", "Phase 3a_isotonic (lean + PAV calibration)"),
            ("3c", "Phase 3c (rich)"),
            ("other", "Other versions"),
        };

        bool anyRendered = false;
        foreach (var spec in phaseSpecs)
        {
            var phaseRows = stationPredictions
                .Where(p => BucketPrecipPhase(input.PhaseByVersion, p.Version) == spec.Key)
                .ToList();
            if (phaseRows.Count == 0) continue;
            anyRendered = true;

            var latestPerLead = phaseRows
                .GroupBy(r => (r.LeadHours, r.ValidTimeUtc))
                .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
                .ToList();

            var series = new List<LineSeries>();
            foreach (var (lead, color) in leadSpecs)
            {
                var pts = latestPerLead
                    .Where(r => r.LeadHours == lead)
                    .OrderBy(r => r.ValidTimeUtc)
                    .Select(r => (X: r.ValidTimeUtc.ToOADate(), Y: r.ProbWet))
                    .ToList();
                if (pts.Count > 0)
                    series.Add(new LineSeries($"P(wet) +{lead}h", color, pts));
            }

            // Rainfall truth: same 0–1 axis by capping at 1. The capped line functions
            // as "was it meaningfully raining?" without a second Y-axis.
            if (truthPts.Count > 0)
                series.Add(new LineSeries("Observed mm/h (capped at 1)", "#ef5350", truthPts));

            if (series.Count == 0)
            {
                content.Append(RenderEmptyChart(
                    $"P(wet) vs observed — {station} — {spec.Key}",
                    "No forecast or truth in window."));
                continue;
            }

            content.Append(Ci, $"<h5>{Escape(spec.Title)}</h5>");
            content.Append(LineChartRenderer.Render(new LineChartSpec
            {
                Title = $"P(wet) vs observed rainfall — {PrettyStation(station)} — Phase {spec.Key}",
                XLabel = "Time (UTC)",
                YLabel = "Probability / mm·h⁻¹ (capped)",
                Series = series,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));
        }

        // Champion-vs-challenger overlay at +24h with observed truth so the reader can
        // see which phase tracked reality better for the lead that matters most.
        var cvc = BuildChampionVsChallengerSeries(stationPredictions, input.PhaseByVersion, input.WindowStartUtc, leadHours: 24);
        // Drop the generatedAt clamp above — for forecast-vs-truth we want history too.
        if (cvc.Count >= 2)
        {
            if (truthPts.Count > 0)
                cvc.Add(new LineSeries("Observed mm/h (capped at 1)", "#ef5350", truthPts));
            content.Append("<h5>Three-way comparison — +24h lead</h5>");
            content.Append(LineChartRenderer.Render(new LineChartSpec
            {
                Title = $"3a vs 3a_isotonic vs 3c vs observed — {PrettyStation(station)} — +24h",
                XLabel = "Time (UTC)",
                YLabel = "Probability / mm·h⁻¹ (capped)",
                Series = cvc,
                Height = 280,
                FormatX = v => DateTime.FromOADate(v).ToString("MM-dd HH'Z'", Ci),
                FormatY = v => v.ToString("0.00", Ci),
            }));
        }

        if (!anyRendered)
            content.Append(RenderEmptyChart($"P(wet) vs observed — {station}", "No forecast or truth in window."));

        return content.ToString();
    }

    private static string RenderDryWindowVsTruthTable(SiteInputs input)
    {
        if (input.DryWindowPredictions.Count == 0)
            return "<p><em>No dry-window predictions in window.</em></p>";

        // "Observed dry window existed?" per (station, date, window). Use the same
        // rainfall truth we already loaded for the precip chart — a dry hour is one
        // where the 4-of-4 hourly SUM is ≤ 0.1 mm, and a dry window is N consecutive
        // such hours all within the same UTC day.
        var observed = ComputeObservedDryWindows(input);

        var stations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var windows = input.DryWindowPredictions.Select(d => d.WindowHours).Distinct().OrderBy(w => w).ToList();
        var leadOrder = new[] { 24, 48, 72 };

        var content = new StringBuilder();

        foreach (var station in stations)
        {
            content.Append(Ci, $"<h4>{Escape(PrettyStation(station))}</h4>");

            foreach (var window in windows)
            {
                var cutoff = input.GeneratedAtUtc.Date.AddDays(-7);
                var latest = input.DryWindowPredictions
                    .Where(d => d.Station == station && d.WindowHours == window && d.TargetDateUtc >= cutoff)
                    .GroupBy(d => (d.TargetDateUtc, d.LeadHours))
                    .Select(g => g.OrderByDescending(d => d.PredictedAtUtc).First())
                    .ToList();

                if (latest.Count == 0) continue;

                var dates = latest.Select(d => d.TargetDateUtc).Distinct().OrderBy(d => d).ToList();

                var tbody = new StringBuilder();
                foreach (var date in dates)
                {
                    var byLead = latest.Where(d => d.TargetDateUtc == date).ToDictionary(d => d.LeadHours);

                    var leadCells = new StringBuilder();
                    foreach (var lead in leadOrder)
                    {
                        if (byLead.TryGetValue(lead, out var d))
                            leadCells.Append(Ci, $"<td class=\"num\">{d.ProbHasDryWindow.ToString("0.00", Ci)}</td>");
                        else
                            leadCells.Append("<td class=\"num\">—</td>");
                    }

                    var observedCell = observed.TryGetValue((station, window, date), out var obs)
                        ? (obs ? "<strong>dry</strong>" : "wet")
                        : "—";

                    tbody.Append(Ci, $"""
                        <tr>
                          <td><time>{date:yyyy-MM-dd}</time></td>
                          {leadCells}
                          <td>{observedCell}</td>
                        </tr>
                        """);
                }

                content.Append(Ci, $"""
                    <h5>{window}-hour dry window</h5>
                    <figure>
                      <table>
                        <thead>
                          <tr>
                            <th>Target date (UTC)</th>
                            <th class="num">+24h</th>
                            <th class="num">+48h</th>
                            <th class="num">+72h</th>
                            <th>Observed</th>
                          </tr>
                        </thead>
                        <tbody>
                    {tbody}    </tbody>
                      </table>
                    </figure>
                    """);
            }
        }

        return content.ToString();
    }

    /// <summary>
    /// For every (station, window, target-date) triple referenced by a prediction, walk
    /// that day's hourly rainfall and check whether at least <c>window</c> consecutive
    /// hours are ≤ 0.1 mm within the UTC day. Returns null for days with no complete
    /// 24-hour coverage.
    /// </summary>
    internal static IReadOnlyDictionary<(string Station, int WindowHours, DateTime TargetDate), bool>
        ComputeObservedDryWindows(SiteInputs input)
    {
        var result = new Dictionary<(string, int, DateTime), bool>();
        // Group by (station, date) so we only walk each day once.
        var triples = input.DryWindowPredictions
            .Select(d => (d.Station, d.WindowHours, d.TargetDateUtc.Date))
            .Distinct();

        foreach (var (station, window, date) in triples)
        {
            if (!input.RainfallTruth.TryGetValue(station, out var hourly) || hourly.Count == 0)
                continue;

            // Need all 24 hourly buckets for a fair verdict. If any are missing we
            // don't emit a row — this leaves the table cell as "—".
            var hours = new double?[24];
            bool complete = true;
            for (int h = 0; h < 24; h++)
            {
                var ts = date.AddHours(h);
                if (hourly.TryGetValue(ts, out var mm)) hours[h] = mm;
                else { complete = false; break; }
            }
            if (!complete) continue;

            int run = 0;
            bool found = false;
            for (int h = 0; h < 24; h++)
            {
                if ((hours[h] ?? double.NaN) <= 0.1) { run++; if (run >= window) { found = true; break; } }
                else run = 0;
            }
            result[(station, window, date)] = found;
        }
        return result;
    }

    public static string RenderAbout(SiteInputs input)
    {
        var body = $"""
            <section>
              <h2>About WeatherBlend</h2>
              <p>
                WeatherBlend is a multi-model weather-forecast blending proof of concept for
                <strong>{Escape(input.LocationDisplay)}</strong>
                ({input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m).
                It combines forecasts from six numerical weather-prediction (NWP) models —
                GFS, ECMWF IFS, DWD ICON, Météo-France, UK Met Office, and Environment Canada GEM —
                via a LightGBM blender trained against ERA5 reanalysis.
              </p>

              <h3>Scope</h3>
              <p>
                This PoC targets <strong>temperature</strong>, <strong>hourly precipitation occurrence
                P(wet ≥ 0.1 mm/h)</strong>, and <strong>per-day dry-window probability</strong>
                (P(∃ contiguous N-hour dry block in target UTC day) for N = 3, 4, or 6 hours) at lead times
                <strong>24h, 48h, and 72h</strong>. Shorter and longer horizons are out of scope.
                Quantitative precipitation intensity (mm) is deferred indefinitely — the dry-window
                framing covers the user-facing question without the calibration headaches of
                conditional intensity regression.
              </p>

              <h3>Data sources</h3>
              <ul>
                <li><strong>Forecasts:</strong> Open-Meteo (live + historical-forecast API).</li>
                <li><strong>Training truth (temperature):</strong> ERA5 reanalysis via Open-Meteo (gapless, quantitative).</li>
                <li><strong>Training + verification truth (precipitation, dry window):</strong> Environment Agency Hydrology rainfall gauges (Bellever Dartmoor, Princetown), 15-min tips aggregated to hourly with a 4-of-4 reading gate.</li>
                <li><strong>Verification truth (temperature):</strong> METAR from aviationweather.gov and OGIMET, used as a real-observation sanity check.</li>
              </ul>

              <h3>Caveats</h3>
              <ul>
                <li>ERA5 is a 0.25° gridded reanalysis; it represents a grid-cell average near Bonehill Rocks, not the tor itself.</li>
                <li>Lowland METAR (Exeter, Yeovilton) has systematic biases against a 393m moorland tor — useful as a cross-check, not ground truth.</li>
                <li>Blender predictions currently cover one model version and a short verification window.</li>
              </ul>

              <p><small>Source: <a href="https://github.com/">WeatherBlend repo</a>. Rendered {input.GeneratedAtUtc:yyyy-MM-dd HH:mm}Z.</small></p>
            </section>
            """;

        return WrapPage(input, "About", "about", body);
    }

    public static string Stylesheet() => """
        :root { --brand: #7c4dff; }
        body > main { padding-top: 1rem; padding-bottom: 3rem; }
        nav.site-nav { padding: 0.5rem 0 1rem; border-bottom: 1px solid var(--pico-muted-border-color); margin-bottom: 1.5rem; }
        nav.site-nav ul { display: flex; gap: 1rem; list-style: none; padding: 0; margin: 0; }
        nav.site-nav a { text-decoration: none; }
        nav.site-nav a.active { font-weight: 600; color: var(--brand); }

        .forecast-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .forecast-card { padding: 1rem; }
        .forecast-card h3 { margin: 0 0 0.25rem; }
        .forecast-card .temp { font-size: 2.5rem; font-weight: 700; color: var(--brand); line-height: 1.1; margin: 0.5rem 0; }
        .forecast-card-empty .temp { color: var(--pico-muted-color); }

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
                      <li><a href="predictions.html"{{NavActive(pageId, "predictions")}}>Temperature</a></li>
                      <li><a href="precipitation.html"{{NavActive(pageId, "precipitation")}}>Precipitation</a></li>
                      <li><a href="dry-window.html"{{NavActive(pageId, "dry-window")}}>Dry window</a></li>
                      <li><a href="forecast-vs-truth.html"{{NavActive(pageId, "forecast-vs-truth")}}>Forecast vs truth</a></li>
                      <li><a href="verify.html"{{NavActive(pageId, "verify")}}>Verification</a></li>
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

    private static string ComputeHeadlineSkill(SiteInputs input)
    {
        var scored = input.Predictions
            .Where(p => p.LeadHours == 24 && input.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();
        if (scored.Count == 0) return "No scored predictions yet — skill headline will appear once ERA5 catches up.";

        var truth = scored.Select(p => input.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend = scored.Select(p => p.BlendTemperature).ToArray();
        var blendMae = Mae(blend, truth);

        // Best single across the six per-model columns.
        (string name, double mae)[] singles =
        {
            ("GFS",   MaeWithGaps(scored.Select(p => p.TempGfs).ToArray(),   truth)),
            ("ECMWF", MaeWithGaps(scored.Select(p => p.TempEcmwf).ToArray(), truth)),
            ("ICON",  MaeWithGaps(scored.Select(p => p.TempIcon).ToArray(),  truth)),
            ("MF",    MaeWithGaps(scored.Select(p => p.TempMf).ToArray(),    truth)),
            ("UKMO",  MaeWithGaps(scored.Select(p => p.TempUkmo).ToArray(),  truth)),
            ("GEM",   MaeWithGaps(scored.Select(p => p.TempGem).ToArray(),   truth)),
        };
        var best = singles.Where(s => !double.IsNaN(s.mae)).OrderBy(s => s.mae).FirstOrDefault();
        if (best.name is null || double.IsNaN(best.mae))
        {
            return $"Over {scored.Count} scored 24h predictions, blend MAE is {blendMae.ToString("0.00", Ci)}°C.";
        }

        var pct = (best.mae - blendMae) / best.mae * 100.0;
        var verb = blendMae < best.mae ? "beat" : "trailed";
        return $"Over {scored.Count} scored 24h predictions, blend MAE is {blendMae.ToString("0.00", Ci)}°C — {verb} best single ({best.name}, {best.mae.ToString("0.00", Ci)}°C) by {Math.Abs(pct).ToString("0.0", Ci)}%.";
    }

    private static double Mae(double[] pred, double[] truth)
    {
        double sum = 0;
        int n = Math.Min(pred.Length, truth.Length);
        for (int i = 0; i < n; i++) sum += Math.Abs(pred[i] - truth[i]);
        return n == 0 ? double.NaN : sum / n;
    }

    private static double MaeWithGaps(double?[] pred, double[] truth)
    {
        double sum = 0;
        int n = 0;
        for (int i = 0; i < pred.Length && i < truth.Length; i++)
        {
            if (pred[i] is double v) { sum += Math.Abs(v - truth[i]); n++; }
        }
        return n == 0 ? double.NaN : sum / n;
    }

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

    private static string Escape(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
}
