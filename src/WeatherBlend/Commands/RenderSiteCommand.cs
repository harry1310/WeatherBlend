using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// Renders a self-contained static site to <c>data/site/</c>: home, predictions table,
/// verification charts, and an about page. Reads prediction and ERA5 parquet via DuckDB
/// using the same glob pattern as <see cref="TempVerifyCommand"/> and the same
/// <c>hive_partitioning=false</c> rule (hive keys collide with in-file column names).
///
/// Runs entirely offline — output is a directory of plain HTML, CSS, and inline SVG.
/// For production, the CI workflow should rclone-sync the R2 prediction/truth trees
/// locally before invoking this command, then push the resulting <c>data/site/</c>
/// tree to the Cloudflare Pages repo (or via the direct-upload API).
/// </summary>
public sealed class RenderSiteCommand
{
    private readonly ILogger<RenderSiteCommand> _log;
    private readonly AppConfig _cfg;
    private readonly TruthRepository _truth;
    private readonly ModelMetadataRepository _metadata;
    private readonly PredictionsRepository _predictions;

    public RenderSiteCommand(ILogger<RenderSiteCommand> log, AppConfig cfg,
        TruthRepository truth, ModelMetadataRepository metadata, PredictionsRepository predictions)
    {
        _log = log;
        _cfg = cfg;
        _truth = truth;
        _metadata = metadata;
        _predictions = predictions;
    }

    public async Task<int> RunAsync(
        string outputDir,
        int windowDays,
        int rollingWindowDays,
        CancellationToken ct)
    {
        if (windowDays < 1) throw new ArgumentOutOfRangeException(nameof(windowDays));
        if (rollingWindowDays < 1) throw new ArgumentOutOfRangeException(nameof(rollingWindowDays));

        var now = DateTime.UtcNow;
        var windowStart = now.AddDays(-windowDays);
        // Forecast cards need valid-times that are still in the future at render time,
        // so include the full forecast horizon (leads go to 120h) plus a day of headroom.
        var predictionEnd = now.AddDays(6);

        _log.LogInformation(
            "Render site: predictions [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], rolling {Rolling}d, output → {Dir}",
            windowStart, predictionEnd, rollingWindowDays, outputDir);

        var predictions = _predictions.GetTemperaturePredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} temperature prediction rows.", predictions.Count);

        // Precip + dry-window come back as the canonical domain rows from the
        // repo; the renderer projects to its lighter SitePages records below.
        // Pass null for the station/cell filter — render scans the whole subtree
        // unlike verify, which limits to its requested stations.
        var precipRows = _predictions.GetPrecipitationPredictions(stations: null, windowStart, predictionEnd, ct);
        var precip = precipRows.Select(r => new SitePages.PrecipForecastPoint(
            Station:         r.TruthStation,
            Version:         r.ModelVersion,
            PredictedAtUtc:  r.PredictionMadeAtUtc,
            ValidTimeUtc:    r.ValidTimeUtc,
            LeadHours:       r.LeadHours,
            ProbWet:         r.ProbWet,
            ClimatologyPWet: r.ClimatologyPWet,
            PrecipGfs:       r.PrecipGfs,
            PrecipEcmwf:     r.PrecipEcmwf,
            PrecipIcon:      r.PrecipIcon,
            PrecipMf:        r.PrecipMf,
            PrecipUkmo:      r.PrecipUkmo,
            PrecipGem:       r.PrecipGem,
            PrecipAifs:      r.PrecipAifs,
            PrecipJma:       r.PrecipJma,
            AgreementWet01:  r.PrecipAgreementWet01,
            ConformalSetTag: r.ConformalSetTag)).ToList();
        _log.LogInformation("Loaded {N} precipitation prediction rows.", precip.Count);

        var feelsLike = QueryFeelsLikePredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} feels-like prediction rows.", feelsLike.Count);

        var dryWindowRows = _predictions.GetDryWindowPredictions(cells: null, windowStart, predictionEnd, ct);
        var dryWindow = dryWindowRows.Select(r => new SitePages.DryWindowForecastPoint(
            Station:                     r.TruthStation,
            WindowHours:                 r.WindowHours,
            Version:                     r.ModelVersion,
            PredictedAtUtc:              r.PredictionMadeAtUtc,
            TargetDateUtc:               r.TargetDateUtc,
            LeadHours:                   r.LeadHours,
            ProbHasDryWindow:            r.ProbHasDryWindow,
            ClimatologyProbHasDryWindow: r.ClimatologyProbHasDryWindow,
            AgreementHasDryWindow:       r.AgreementHasDryWindow,
            McMeanLongestDryRunHours:    r.McMeanLongestDryRunHours,
            McP10LongestDryRunHours:     r.McP10LongestDryRunHours,
            McP50LongestDryRunHours:     r.McP50LongestDryRunHours,
            McP90LongestDryRunHours:     r.McP90LongestDryRunHours,
            ConformalSetTag:             r.ConformalSetTag)).ToList();
        _log.LogInformation("Loaded {N} dry-window prediction rows.", dryWindow.Count);

        var startHour = QueryStartHourPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} start-hour curve rows.", startHour.Count);

        var metOfficeSpot = QueryMetOfficeSpotForecasts(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} Met Office Spot forecast rows.", metOfficeSpot.Count);

        var nwpPop = QueryNwpPrecipProbabilities(windowStart, predictionEnd, ct);
        var bayesianCi = QueryBayesianCi(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} Bayesian CI rows ({S} stations).",
            bayesianCi.Count, bayesianCi.Select(p => p.StationCode).Distinct().Count());
        var lowCloudByValid = QueryLowCloudSignals(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded low-cloud / mist signal for {N} valid hours.", lowCloudByValid.Count);
        _log.LogInformation("Loaded {N} per-NWP precipitation-probability rows ({M} models).",
            nwpPop.Count, nwpPop.Select(p => p.Model).Distinct().Count());

        var verifyHistory = LoadVerifyHistory(_cfg.Storage.ReportsPath);
        _log.LogInformation("Loaded {N} verify-history JSON sidecars.", verifyHistory.Count);

        var dryWindowConformalTau = LoadDryWindowConformalTau();
        _log.LogInformation("Loaded {N} dry-window conformal τ values across (version, lead).",
            dryWindowConformalTau.Count);

        var precipConformalTau = LoadPrecipConformalTau();
        _log.LogInformation("Loaded {N} precip conformal τ values across (station, version, lead).",
            precipConformalTau.Count);

        // ERA5 is gapless-for-past, but only past — query up to the clock time.
        // Persistence-lookback headroom (up to 72h before the earliest prediction)
        // keeps rolling-MAE truth pairs matchable at the start of the window.
        var truth = _truth.GetEra5Hourly(windowStart.AddDays(-3), now, ct);
        _log.LogInformation("Loaded {N} ERA5 truth points.", truth.Count);

        var metar = _truth.GetMetarTemperature(_cfg.Location.Metar.Primary, windowStart, now, ct);
        _log.LogInformation("Loaded {N} METAR observations ({Station}).",
            metar.Count, string.IsNullOrWhiteSpace(_cfg.Location.Metar.Primary) ? "none" : _cfg.Location.Metar.Primary);

        var moObs = _truth.GetMetOfficeObsTemperature(windowStart, now, ct);
        _log.LogInformation("Loaded {N} Met Office obs (geohash {GH}).",
            moObs.Count, _cfg.MetOffice.ObsGeohash);

        // PhaseByVersion has to be computed before the rolling functions so
        // they can group by phase rather than version (a retrain would
        // otherwise fragment the chart into stubs).
        var phaseByVersion = _metadata.GetPhaseByVersion(
            predictions.Select(p => p.ModelVersion),
            precip.Select(p => (p.Station, p.Version)),
            dryWindow.Select(d => (d.Station, d.WindowHours, d.Version)));
        var rolling = ComputeRollingMae(predictions, truth, phaseByVersion, rollingWindowDays);
        // Precip uses a longer 30-day rolling window than temp's because wet
        // hours are sparser — the verify command's defaults pin this rule
        // (PrecipVerifyCommand: 30d window vs TempVerifyCommand: 14d).
        const int precipRollingWindowDays = 30;

        _log.LogInformation("Phase map: {Entries}",
            string.Join(", ", phaseByVersion.Select(kv => $"{kv.Key}→{kv.Value}")));

        var rainfall = LoadRainfallTruth(windowStart, now, precip, ct);
        _log.LogInformation("Rainfall truth: {N} stations loaded.", rainfall.Count);

        var currentVersion = _metadata.GetChampion("temperature");
        var championByLead = _metadata.GetChampionByLead("temperature");
        _log.LogInformation("Champion (temperature): {Version}",
            string.IsNullOrEmpty(currentVersion) ? "(none)" : currentVersion);
        if (championByLead.Count > 0)
            _log.LogInformation("Per-lead champion overrides: {Entries}",
                string.Join(", ", championByLead.OrderBy(kv => kv.Key).Select(kv => $"+{kv.Key}h→{kv.Value}")));

        var precipCurrentByStation = _metadata.GetChampionsByStation("precipitation");
        var precipChampionByStationLead = _metadata.GetChampionByStationLead("precipitation");
        _log.LogInformation("Champion (precipitation): {Entries}",
            precipCurrentByStation.Count == 0
                ? "(none)"
                : string.Join(", ", precipCurrentByStation.Select(kv => $"{kv.Key}→{kv.Value}")));
        if (precipChampionByStationLead.Count > 0)
            _log.LogInformation("Per-(station, lead) precipitation champion overrides: {Entries}",
                string.Join(", ", precipChampionByStationLead
                    .OrderBy(kv => kv.Key.Station, StringComparer.Ordinal)
                    .ThenBy(kv => kv.Key.LeadHours)
                    .Select(kv => $"{kv.Key.Station}@+{kv.Key.LeadHours}h→{kv.Value}")));

        var activeStationSlugs = new HashSet<string>(
            _cfg.Location.Rainfall.Stations.Select(s => StationSlug.WithEaPrefix(s.Name)),
            StringComparer.Ordinal);
        var modelSummaries = LoadModelSummaries(predictions, precip, dryWindow, activeStationSlugs);
        _log.LogInformation("Loaded {N} model summaries for Models page.", modelSummaries.Count);

        var input = new SitePages.SiteInputs
        {
            LocationDisplay = string.IsNullOrWhiteSpace(_cfg.Location.DisplayName) ? _cfg.Location.Name : _cfg.Location.DisplayName,
            Latitude = _cfg.Location.Latitude,
            Longitude = _cfg.Location.Longitude,
            ElevationMeters = _cfg.Location.ElevationMeters,
            MetarStation = _cfg.Location.Metar.Primary,
            GeneratedAtUtc = now,
            WindowStartUtc = windowStart,
            Predictions = predictions,
            TruthByTime = truth,
            MetarByTime = metar,
            MetOfficeObsByTime = moObs,
            LowCloudByValid = lowCloudByValid,
            RollingMae = rolling,
            RollingBrier = ComputeRollingBrier(precip, rainfall, phaseByVersion, precipRollingWindowDays),
            PrecipPredictions = precip,
            DryWindowPredictions = dryWindow,
            DryWindowConformalTau = dryWindowConformalTau,
            PrecipConformalTau = precipConformalTau,
            DryWindowDaytime = _cfg.DryWindow.BuildDaytimeWindow(),
            PhaseByVersion = phaseByVersion,
            RainfallTruth = rainfall,
            CurrentVersion = currentVersion,
            ChampionByLead = championByLead,
            PrecipCurrentByStation = precipCurrentByStation,
            PrecipChampionByStationLead = precipChampionByStationLead,
            ActiveStationSlugs = activeStationSlugs,
            ModelSummaries = modelSummaries,
            FeelsLikePredictions = feelsLike,
            StartHourPredictions = startHour,
            MetOfficeSpotForecasts = metOfficeSpot,
            NwpPrecipProbabilities = nwpPop,
            BayesianCi = bayesianCi,
            VerifyHistory = verifyHistory,
        };

        Directory.CreateDirectory(outputDir);
        // Clean every .html in outputDir before writing the fresh set. Stale
        // per-station files (dry-window-princetown.html etc. left over from a
        // station swap) would otherwise stick around forever — Cloudflare
        // pages deploy is additive on what's in the source dir, not a wipe.
        // Non-HTML files (chart.js, styles.css) are overwritten in place by
        // the writers below; only the per-page HTML can become orphaned.
        foreach (var stale in Directory.EnumerateFiles(outputDir, "*.html"))
            File.Delete(stale);
        // Home page is now per-day — six files (today + five forward) so
        // each day is a real sub-tab not just a section. index.html stays
        // canonical for offset=0; forward days are index-{n}.html.
        for (int n = 0; n <= SitePages.MaxHomeDayOffset; n++)
        {
            var file = n == 0 ? "index.html" : $"index-{n}.html";
            await File.WriteAllTextAsync(Path.Combine(outputDir, file),
                SitePages.RenderIndex(input, n), ct);
        }
        // Forecasts split per variable per lead on 2026-05-04. Temperature +
        // rain are per-lead — uses Leads.ForecastsTempRain (= Leads.Full plus
        // +12h for 2d / 3d exact-runtime). Dry-window is per-day so emits
        // once + one per non-first station.
        foreach (var lead in Leads.ForecastsTempRain)
        {
            await File.WriteAllTextAsync(Path.Combine(outputDir, $"forecasts-temp-{lead}h.html"),
                SitePages.RenderForecastsTemp(input, lead), ct);
            await File.WriteAllTextAsync(Path.Combine(outputDir, $"forecasts-rain-{lead}h.html"),
                SitePages.RenderForecastsRain(input, lead), ct);
        }
        // Models page split into 3 (per target) on 2026-05-04 — see
        // SitePages.RenderModels for the per-target intro copy. The legacy
        // models.html route was retired in the same change; the nav points
        // at models-temp.html as the canonical landing for "Models".
        await File.WriteAllTextAsync(Path.Combine(outputDir, "models-temp.html"),       SitePages.RenderModels(input, "temperature"),   ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "models-rain.html"),       SitePages.RenderModels(input, "precipitation"), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "models-dry-window.html"), SitePages.RenderModels(input, "dry_window"),    ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "about.html"),         SitePages.RenderAbout(input),          ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "styles.css"),         SitePages.Stylesheet(),                ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "chart.js"),           SitePages.ChartScript(),               ct);

        // Temp skill is single-file — temperature has no station axis.
        await File.WriteAllTextAsync(Path.Combine(outputDir, "skill-temperature.html"),
            SitePages.RenderTempSkill(input), ct);

        // Rain skill is per-station — one canonical file plus one per non-first station.
        // Stations set is the union of precip + dry-window stations, so even if a station
        // has only one of the two, it gets its own tab.
        var rainStations = SitePages.GetRainSkillStations(input);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "skill-rainfall.html"),
            SitePages.RenderRainSkill(input, null), ct);
        for (int i = 1; i < rainStations.Count; i++)
        {
            var slug = SitePages.StationSlug(rainStations[i]);
            await File.WriteAllTextAsync(
                Path.Combine(outputDir, $"skill-rainfall-{slug}.html"),
                SitePages.RenderRainSkill(input, slug), ct);
        }

        // Dry-window skill — split out from rain-skill on 2026-05-04 so each
        // variable's skill content has its own page. Per-station like rain.
        await File.WriteAllTextAsync(Path.Combine(outputDir, "skill-dry-window.html"),
            SitePages.RenderDryWindowSkill(input, null), ct);
        for (int i = 1; i < rainStations.Count; i++)
        {
            var slug = SitePages.StationSlug(rainStations[i]);
            await File.WriteAllTextAsync(
                Path.Combine(outputDir, $"skill-dry-window-{slug}.html"),
                SitePages.RenderDryWindowSkill(input, slug), ct);
        }

        // Dry-window page is per-station too. Same active-station filter as
        // rain-skill — a station that was demoted from config (Princetown
        // post 2026-05-04) shouldn't render just because its historical
        // dry-window predictions are still on disk.
        var activeSet = input.ActiveStationSlugs;
        var dryStations = input.DryWindowPredictions
            .Select(d => d.Station)
            .Where(s => activeSet.Count == 0 || activeSet.Contains(s))
            .Distinct()
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToList();
        // Dry-window forecasts page is now part of the Forecasts family
        // (filename forecasts-dry-window.html, sub-nav links to it from
        // every Forecasts page). Per-station variants use the same
        // station-loop shape as skill-rainfall.
        await File.WriteAllTextAsync(Path.Combine(outputDir, "forecasts-dry-window.html"),
            SitePages.RenderDryWindow(input, null), ct);
        for (int i = 1; i < dryStations.Count; i++)
        {
            var slug = SitePages.StationSlug(dryStations[i]);
            await File.WriteAllTextAsync(
                Path.Combine(outputDir, $"forecasts-dry-window-{slug}.html"),
                SitePages.RenderDryWindow(input, slug), ct);
        }

        // index × (1 + MaxHomeDayOffset) + (forecasts-temp × leads) +
        // (forecasts-rain × leads) + models × 3 + about + styles + chart.js +
        // skill-temperature + skill-rainfall × stations +
        // skill-dry-window × stations + forecasts-dry-window × stations.
        var totalFiles = 7 + (SitePages.MaxHomeDayOffset + 1)
            + (Leads.ForecastsTempRain.Length * 2)
            + Math.Max(1, rainStations.Count)        // skill-rainfall variants
            + Math.Max(1, rainStations.Count)        // skill-dry-window variants
            + Math.Max(1, dryStations.Count);        // forecasts-dry-window variants
        _log.LogInformation("Site rendered → {Dir} ({Files} files)", outputDir, totalFiles);
        return 0;
    }

    private IReadOnlyList<SitePages.ModelSummary> LoadModelSummaries(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        IReadOnlyList<SitePages.DryWindowForecastPoint> dryWindow,
        IReadOnlySet<string> activeStationSlugs)
    {
        // Only load metadata for versions that actually emitted predictions in the
        // window. Anything else is stale / experimental / deleted and the Models page
        // would list it with no corresponding forecast activity.
        // Per-station targets (precip + dry-window) are also filtered through
        // activeStationSlugs so a demoted station (Princetown post 2026-05-04)
        // doesn't get a Models card just because its historical predictions are
        // still on disk.
        var modelsRoot = _cfg.Storage.ModelsPath;
        var summaries = new List<SitePages.ModelSummary>();

        foreach (var version in predictions.Select(p => p.ModelVersion).Distinct())
        {
            var dir = Path.Combine(modelsRoot, "temperature", version);
            var summary = TryLoadSummary(dir, composite: "temperature", version, metricLabel: "Test MAE (°C)");
            if (summary is not null) summaries.Add(summary);
        }

        foreach (var (station, version) in precip.Select(p => (p.Station, p.Version)).Distinct())
        {
            if (activeStationSlugs.Count > 0 && !activeStationSlugs.Contains(station)) continue;
            var dir = Path.Combine(modelsRoot, "precipitation", station, version);
            var composite = $"precipitation/{station}";
            var summary = TryLoadSummary(dir, composite, version, metricLabel: "Test Brier");
            if (summary is not null) summaries.Add(summary);
        }

        foreach (var (station, window, version) in dryWindow.Select(d => (d.Station, d.WindowHours, d.Version)).Distinct())
        {
            if (activeStationSlugs.Count > 0 && !activeStationSlugs.Contains(station)) continue;
            var dir = Path.Combine(modelsRoot, "dry_window", station, $"window_{window}h", version);
            var composite = $"dry_window/{station}/{window}h";
            var summary = TryLoadSummary(dir, composite, version, metricLabel: "Test Brier");
            if (summary is not null) summaries.Add(summary);
        }

        return summaries;
    }

    private SitePages.ModelSummary? TryLoadSummary(string versionDir, string composite, string version, string metricLabel)
    {
        try
        {
            if (!Directory.Exists(versionDir)) return null;
            var metadataPath = Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(metadataPath)) return null;

            var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
            var perLead = metadata.PerLead
                .Where(kv => int.TryParse(kv.Key, out _))
                .ToDictionary(
                    kv => int.Parse(kv.Key),
                    kv => new SitePages.PerLeadMetric(
                        LeadHours:          kv.Value.LeadHours,
                        BestSingle:         kv.Value.BestSingle,
                        BestSingleValMae:   kv.Value.BestSingleValMae,
                        BestSingleTestMae:  kv.Value.BestSingleTestMae,
                        BlendTestScore:     kv.Value.BlendTestMae,
                        BlendTestRmse:      kv.Value.BlendTestRmse,
                        BlendTestBias:      kv.Value.BlendTestBias,
                        TestRows:           kv.Value.TestRows,
                        TestCalendarMonths: kv.Value.TestCalendarMonths));

            return new SitePages.ModelSummary(
                Composite:     composite,
                Version:       version,
                Phase:         metadata.Phase,
                DataSource:    metadata.DataSource,
                TrainedAtUtc:  metadata.TrainedAtUtc,
                MetricLabel:   metricLabel,
                PerLead:       perLead);
        }
        catch (Exception ex)
        {
            _log.LogWarning("Failed to load model summary for {Composite}/{Version}: {Msg}", composite, version, ex.Message);
            return null;
        }
    }


    /// <summary>
    /// Resolve EA rainfall truth for the stations that actually produced
    /// precip predictions in the window. Pure delegate to
    /// <see cref="TruthRepository.GetEaHourlyRainfallByStation"/>; this stub
    /// only exists because we want the per-station list to be derived from
    /// the predictions (so empty windows skip the I/O entirely).
    /// </summary>
    private IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> LoadRainfallTruth(
        DateTime start, DateTime end,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        CancellationToken ct)
    {
        var stations = precip.Select(p => p.Station).Distinct().ToList();
        return _truth.GetEaHourlyRainfallByStation(stations, start, end, ct);
    }

    /// <summary>
    /// Rolling MAE per (Phase, LeadHours, daily window end) across the last
    /// <paramref name="windowDays"/>. One point per day from the earliest paired
    /// date through the latest. Days with fewer than <paramref name="windowDays"/>
    /// of data behind them emit a point computed over whatever's available in
    /// <c>[d − windowDays + 1, d]</c> — partial-window points show up at the
    /// start of the data, which is what users expect when the prediction tree
    /// is younger than the rolling window. Without that, a 14-day window over
    /// 3 days of data renders an empty chart even though the underlying pairs
    /// are right there in the per-lead chart above.
    ///
    /// Grouping rule (changed 2026-05-02): predictions are deduped by
    /// (Phase, Lead, ValidTime) — the freshest <c>PredictionMadeAtUtc</c>
    /// wins per cell — and then aggregated by Phase. Previously this method
    /// emitted one series per <c>ModelVersion</c>, which meant a retrain
    /// fragmented the chart into a long line for the old version and a
    /// short stub for the new one. Predictions whose version is missing
    /// from <paramref name="phaseByVersion"/> (retired phases without
    /// metadata on disk) are dropped; the renderer treats them as
    /// off-roadmap.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingMaePoint> ComputeRollingMae(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyDictionary<DateTime, double> truthByTime,
        IReadOnlyDictionary<string, string> phaseByVersion,
        int windowDays)
    {
        // Pair, attach phase, drop versions that don't map to a known phase.
        var paired = predictions
            .Where(p => truthByTime.ContainsKey(p.ValidTimeUtc))
            // Two filters: version → phase known, AND phase is in the
            // shipping lineup. The latter drops retired-but-still-on-disk
            // versions (e.g. v..._phase2redo → "2b_redo") so the chart
            // shows only what the codebase considers live.
            .Where(p => phaseByVersion.TryGetValue(p.ModelVersion, out var ph)
                        && ActivePhasePolicy.IsActive("temperature", ph))
            .Select(p => (
                Phase: phaseByVersion[p.ModelVersion],
                p.LeadHours,
                p.ValidTimeUtc,
                p.PredictionMadeAtUtc,
                Pred: p.BlendTemperature,
                Truth: truthByTime[p.ValidTimeUtc]))
            .GroupBy(p => (p.Phase, p.LeadHours, p.ValidTimeUtc))
            .Select(g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First())
            .ToList();

        if (paired.Count == 0) return Array.Empty<SitePages.RollingMaePoint>();

        var minDate = paired.Min(r => r.ValidTimeUtc).Date;
        var maxDate = paired.Max(r => r.ValidTimeUtc).Date;

        var points = new List<SitePages.RollingMaePoint>();

        foreach (var (phase, lead) in paired.Select(p => (p.Phase, p.LeadHours)).Distinct())
        {
            var subset = paired.Where(p => p.Phase == phase && p.LeadHours == lead).ToList();
            if (subset.Count == 0) continue;

            // Emit one rolling-MAE point per calendar day end across the
            // available paired range. Window slides backwards by windowDays
            // from each emitted day; partial windows at the start are fine —
            // the N field on each point tells the reader how many pairs went
            // in so they can judge stability.
            for (var d = minDate; d <= maxDate; d = d.AddDays(1))
            {
                var windowEnd = d.AddDays(1).AddTicks(-1);
                var windowStart = windowEnd.AddDays(-windowDays);

                var inWindow = subset.Where(r => r.ValidTimeUtc >= windowStart && r.ValidTimeUtc <= windowEnd).ToList();
                if (inWindow.Count == 0) continue;

                var mae = inWindow.Sum(r => Math.Abs(r.Pred - r.Truth)) / inWindow.Count;
                points.Add(new SitePages.RollingMaePoint(phase, lead, windowEnd, mae, inWindow.Count));
            }
        }

        return points;
    }

    /// <summary>
    /// Rolling Brier per (Station, Phase, LeadHours, day-end) across
    /// the last <paramref name="windowDays"/>. Mirror of
    /// <see cref="ComputeRollingMae"/> for binary classification: pairs each
    /// prediction's <c>ProbWet</c> with the EA gauge's wet/dry indicator at
    /// the same hour, then computes the squared-error mean over each rolling
    /// window. Truth conversion uses the same 0.1 mm/h threshold the
    /// blender was trained on. Partial-window points emit at the start of
    /// the data — same rule the temp rolling-MAE uses (commit 4028ba5),
    /// readers gauge stability from the per-point <c>N</c>.
    ///
    /// Phase grouping (changed 2026-05-02): same rule as
    /// <see cref="ComputeRollingMae"/> — dedup by (Station, Phase, Lead,
    /// ValidTime) preferring the freshest <c>PredictedAtUtc</c>, then group
    /// by Phase. Predictions whose version isn't in
    /// <paramref name="phaseByVersion"/> are dropped.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingBrierPoint> ComputeRollingBrier(
        IReadOnlyList<SitePages.PrecipForecastPoint> predictions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> rainfallByStation,
        IReadOnlyDictionary<string, string> phaseByVersion,
        int windowDays)
    {
        const double WetThresholdMm = 0.1;

        // Pair each prediction with its station's truth (drop unpaired —
        // either station has no truth dict yet, or that hour didn't pass
        // the 4-of-4 quarter-hour gate upstream). Drop versions without a
        // known phase. Then dedup (Station, Phase, Lead, ValidTime) to the
        // freshest prediction so two versions of the same phase don't
        // double-count the same hour in the rolling window.
        var paired = predictions
            .Where(p => rainfallByStation.TryGetValue(p.Station, out var byTime)
                        && byTime.ContainsKey(p.ValidTimeUtc))
            // Same active-phase filter as ComputeRollingMae — drops retired
            // phases like "3a_isotonic" so the rain-skill chart matches the
            // Models page's allowlist.
            .Where(p => phaseByVersion.TryGetValue(p.Version, out var ph)
                        && ActivePhasePolicy.IsActive("precipitation", ph))
            .Select(p =>
            {
                var byTime = rainfallByStation[p.Station];
                var truthMm = byTime[p.ValidTimeUtc];
                return (p.Station,
                        Phase: phaseByVersion[p.Version],
                        p.LeadHours,
                        p.ValidTimeUtc,
                        p.PredictedAtUtc,
                        Pred: p.ProbWet,
                        Truth: truthMm >= WetThresholdMm ? 1.0 : 0.0);
            })
            .GroupBy(p => (p.Station, p.Phase, p.LeadHours, p.ValidTimeUtc))
            .Select(g => g.OrderByDescending(p => p.PredictedAtUtc).First())
            .ToList();

        if (paired.Count == 0) return Array.Empty<SitePages.RollingBrierPoint>();

        var minDate = paired.Min(r => r.ValidTimeUtc).Date;
        var maxDate = paired.Max(r => r.ValidTimeUtc).Date;

        var points = new List<SitePages.RollingBrierPoint>();
        foreach (var (station, phase, lead) in paired
            .Select(p => (p.Station, p.Phase, p.LeadHours)).Distinct())
        {
            var subset = paired.Where(p => p.Station == station
                                        && p.Phase == phase
                                        && p.LeadHours == lead).ToList();
            if (subset.Count == 0) continue;

            for (var d = minDate; d <= maxDate; d = d.AddDays(1))
            {
                var windowEnd = d.AddDays(1).AddTicks(-1);
                var windowStart = windowEnd.AddDays(-windowDays);
                var inWindow = subset.Where(r => r.ValidTimeUtc >= windowStart
                                              && r.ValidTimeUtc <= windowEnd).ToList();
                if (inWindow.Count == 0) continue;
                var brier = inWindow.Sum(r => (r.Pred - r.Truth) * (r.Pred - r.Truth)) / inWindow.Count;
                points.Add(new SitePages.RollingBrierPoint(
                    station, phase, lead, windowEnd, brier, inWindow.Count));
            }
        }
        return points;
    }

    // (QueryPredictions / QueryPrecipPredictions / QueryDryWindowPredictions
    // moved to PredictionsRepository on 2026-05-02; the projections from the
    // domain row types into SitePages.* records happen at the call site
    // above so the storage layer doesn't import the Site namespace.)

    private IReadOnlyList<SitePages.FeelsLikeForecastPoint> QueryFeelsLikePredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
                                                   WeatherBlend.Commands.FeelsLikePredictCommand.PredictionsSubdir,
                                                   "**", "*.parquet"));

        // Both UtciC and ApparentTemperatureC are required on every row (Steadman
        // 1994 added alongside UTCI as part of the predictUTCI → predictFeelsLike
        // refactor). Probe the schema first so a pre-Steadman parquet doesn't
        // surface as a typed-mapping crash; reuse the conn for the main query.
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        if (!ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC"))
        {
            _log.LogWarning(
                "Feels-like predictions tree empty or pre-Steadman — chip will be absent on home cards until predict runs once.");
            return Array.Empty<SitePages.FeelsLikeForecastPoint>();
        }

        // Pull element inputs alongside UTCI + Apparent so the home-page tile
        // pop-out can surface "why is UTCI this value?" without a second read.
        // All five are required-non-null at write time so no NULL defending.
        var sql = $@"
SELECT ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours, UtciC, Band,
       ApparentTemperatureC,
       TemperatureC, RelativeHumidityPct, WindSpeed10mMs,
       ShortwaveDownWm2, CloudCoverPct
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND UtciC IS NOT NULL
  AND ApparentTemperatureC IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY LeadHours, ValidTimeUtc";

        return ParquetReader.Query(conn, sql, r => new SitePages.FeelsLikeForecastPoint(
            Version:             r.GetString(0),
            PredictedAtUtc:      r.GetDateTime(1),
            ValidTimeUtc:        r.GetDateTime(2),
            LeadHours:           r.GetInt32(3),
            UtciC:               r.GetDouble(4),
            Band:                r.IsDBNull(5) ? "" : r.GetString(5),
            ApparentC:           r.GetDouble(6),
            TemperatureC:        r.IsDBNull(7)  ? null : (double?)r.GetDouble(7),
            RelativeHumidityPct: r.IsDBNull(8)  ? null : (double?)r.GetDouble(8),
            WindSpeed10mMs:      r.IsDBNull(9)  ? null : (double?)r.GetDouble(9),
            ShortwaveDownWm2:    r.IsDBNull(10) ? null : (double?)r.GetDouble(10),
            CloudCoverPct:       r.IsDBNull(11) ? null : (double?)r.GetDouble(11)),
            _log, "Feels-like predictions tree empty — chip will be absent on home cards.", ct);
    }

    /// <summary>
    /// Scans <paramref name="reportsDir"/> for <c>verify_*_*.json</c> sidecar
    /// files written by the weekly verify commands. Each file is one verify
    /// run for one target; Models page renderer filters per-card on (target,
    /// station, version, windowHours) to surface a per-card history table.
    /// Missing dir → empty list (fresh deploy / R2 pull skipped); malformed
    /// files are logged and skipped rather than failing the whole render.
    /// </summary>
    private List<WeatherBlend.Models.VerifyHistoryFile> LoadVerifyHistory(string reportsDir)
    {
        var result = new List<WeatherBlend.Models.VerifyHistoryFile>();
        if (!Directory.Exists(reportsDir)) return result;
        var jsonOptions = new System.Text.Json.JsonSerializerOptions
        {
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            // Mirror the writer (VerifyHistoryWriter.JsonOptions) so non-finite
            // metrics serialised as "Infinity"/"NaN" round-trip cleanly.
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        };
        foreach (var path in Directory.EnumerateFiles(reportsDir, "verify_*.json"))
        {
            try
            {
                var json = File.ReadAllText(path);
                var file = System.Text.Json.JsonSerializer.Deserialize<WeatherBlend.Models.VerifyHistoryFile>(json, jsonOptions);
                if (file is not null && !string.IsNullOrEmpty(file.Target))
                    result.Add(file);
            }
            catch (Exception ex)
            {
                _log.LogWarning("Skipping malformed verify-history JSON {Path}: {Msg}", path, ex.Message);
            }
        }
        return result;
    }

    /// <summary>
    /// Per-NWP precipitation_probability rows for the configured location, one
    /// per (Model, ValidTime) at the freshest RunTime that's not in the future
    /// at render time. Filters to the canonical 8 blender NWPs to avoid stray
    /// experimental partitions (mirrors LiveCycleAsOf's defensive Model IN list).
    /// Only ~4 of those publish PoP via Open-Meteo (GFS / ECMWF / ICON / GEM);
    /// the others return zero rows and silently drop out of the chart.
    /// </summary>
    /// <summary>
    /// Per-valid-hour low-cloud / mist signal aggregated across NWP forecasts.
    /// For each valid hour in [start, end], take each model's most-recent
    /// (smallest-lead) forecast and count: how many publish vis &lt; 1km
    /// (mist / fog signal — only 6 NWPs publish Visibility), and how many
    /// have T-Td &lt; 1.5°C (Espy proxy for cloud base below 393m at the
    /// typical SW Devon NWP grid elevation of 50-200m).
    ///
    /// Both counts persist on the row regardless of whether they fired; the
    /// renderer applies the firing thresholds (3 of 6 vis, 6 of 11 cloud-
    /// base) and decides whether to draw the badge. Keeping the raw counts
    /// lets the hover-tooltip read "5 of 6 NWPs forecast mist" rather than
    /// just "warning".
    /// </summary>
    /// <summary>
    /// Load conformal τ values for every Active dry-window version × lead.
    /// Walks the dry-window manifest, opens each version's per-lead
    /// <c>conformal_calibrator_{L}h.json</c>, and dictionaries the resulting
    /// τ by (version, lead). Surfaced on the dry-window page so tags read as
    /// "P=62% · τ=30% · confident dry day" rather than just "confident dry
    /// day". Missing JSONs (a fresh version that hasn't had auto-conformal
    /// fit yet) silently drop — the cell falls back to the tag-only view.
    /// </summary>
    /// <summary>
    /// Load conformal τ values for every Active precipitation version × lead.
    /// Per-(station, version, lead) because precip artefact paths are per
    /// station so versions aren't unique across stations. Surfaced on the
    /// precip forecasts page beside the conformal chip ("Confident wet ·
    /// τ=70% · P=85%"). Missing JSONs (3d versions don't fit conformal yet,
    /// or fresh 3a/3c that haven't had auto-conformal run) drop silently —
    /// the cell falls back to chip-only.
    /// </summary>
    private IReadOnlyDictionary<(string Station, string Version, int LeadHours), double> LoadPrecipConformalTau()
    {
        var dict = new Dictionary<(string, string, int), double>();
        var manifest = _metadata.TryGetManifest("precipitation");
        if (manifest?.Stations is null) return dict;
        var modelsRoot = _cfg.Storage.ModelsPath;

        foreach (var (stationSlug, entry) in manifest.Stations)
        {
            foreach (var versionName in entry.Active)
            {
                var versionDir = Path.Combine(modelsRoot, "precipitation", stationSlug, versionName);
                if (!Directory.Exists(versionDir)) continue;
                // Try both lead sets — Full covers 3a/3c at 24-120, the
                // smaller {12, 24} covers 3d. Cheap to over-iterate; a
                // missing calibrator file just returns null silently.
                foreach (var lead in WeatherBlend.Train.Common.Leads.ForecastsTempRain)
                {
                    var cal = ModelArtifact.TryLoadLeadConformalCalibrator(versionDir, lead);
                    if (cal is null) continue;
                    dict[(stationSlug, versionName, lead)] = cal.Tau;
                }
            }
        }
        return dict;
    }

    private IReadOnlyDictionary<(string Version, int LeadHours), double> LoadDryWindowConformalTau()
    {
        var dict = new Dictionary<(string, int), double>();
        var manifest = _metadata.TryGetManifest("dry_window");
        if (manifest?.Stations is null) return dict;
        var modelsRoot = _cfg.Storage.ModelsPath;

        foreach (var (compositeKey, entry) in manifest.Stations)
        {
            foreach (var versionName in entry.Active)
            {
                var versionDir = Path.Combine(modelsRoot, "dry_window", compositeKey, versionName);
                if (!Directory.Exists(versionDir)) continue;
                foreach (var lead in WeatherBlend.Train.Common.Leads.Short)
                {
                    var cal = ModelArtifact.TryLoadLeadConformalCalibrator(versionDir, lead);
                    if (cal is null) continue;
                    // Same versionName is shared across (station, window)
                    // composites for 3b/3g — but conformal artefacts live
                    // under each composite's version dir so the τ values
                    // genuinely differ per (station, window). We key only on
                    // (version, lead) here because the dry-window page already
                    // resolved (station, window) before reading this dict. If
                    // two composites collide on (version, lead) we keep the
                    // first; in practice 3b/3g use suffixed version names per
                    // composite + the suffixes only appear once per composite.
                    dict.TryAdd((versionName, lead), cal.Tau);
                }
            }
        }
        return dict;
    }

    private IReadOnlyDictionary<DateTime, SitePages.LowCloudSignal> QueryLowCloudSignals(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.ForecastsPath, "**", "*.parquet"));

        // Two thresholds:
        //   Vis < 1000 m → mist (Met Office definition: vis 1-5km = mist;
        //     <1km = fog; we lump <1km as the badge-firing threshold so a
        //     thin mist day where vis stays 1.5-2km doesn't pop a warning).
        //   T - Td < 1.5°C → near-saturated air, Espy cloud base below ~190m
        //     AGL — at typical Open-Meteo grid elevations of 50-200m for SW
        //     Devon that puts cloud base AMSL below Bonehill's 393m on most
        //     NWPs. Conservative single threshold pending an Elevation column
        //     on ForecastRow that would let us thread per-model grid elev.
        const double VisFireThresholdM = 1000.0;
        const double TmTdFireThresholdC = 1.5;

        var sql = $@"
WITH latest AS (
  SELECT Model, ValidTimeUtc,
         Visibility, Temperature2m, DewPoint2m,
         row_number() OVER (PARTITION BY Model, ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
    AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT ValidTimeUtc,
       SUM(CASE WHEN Visibility IS NOT NULL AND Visibility < {VisFireThresholdM.ToString(System.Globalization.CultureInfo.InvariantCulture)} THEN 1 ELSE 0 END)::INTEGER AS vis_fired,
       SUM(CASE WHEN Visibility IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS vis_total,
       SUM(CASE WHEN Temperature2m IS NOT NULL AND DewPoint2m IS NOT NULL
                     AND (Temperature2m - DewPoint2m) < {TmTdFireThresholdC.ToString(System.Globalization.CultureInfo.InvariantCulture)}
                THEN 1 ELSE 0 END)::INTEGER AS tdspread_fired,
       SUM(CASE WHEN Temperature2m IS NOT NULL AND DewPoint2m IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS tdspread_total
FROM latest
WHERE rn = 1
GROUP BY ValidTimeUtc";

        var rows = ParquetReader.Query(sql, r => (
            Time: r.GetDateTime(0),
            VisFired: r.GetInt32(1),
            VisTotal: r.GetInt32(2),
            TdFired: r.GetInt32(3),
            TdTotal: r.GetInt32(4)),
            _log, "Forecast tree empty — low-cloud / mist warning will be absent.", ct);

        return rows.ToDictionary(
            x => x.Time,
            x => new SitePages.LowCloudSignal(
                VisFiredCount: x.VisFired, VisTotalCount: x.VisTotal,
                CloudBaseFiredCount: x.TdFired, CloudBaseTotalCount: x.TdTotal));
    }

    /// <summary>
    /// Phase 2b live Bayesian P(wet) credible-interval rows from
    /// WeatherProbabilistic. Hive layout
    ///   data/predictions/precipitation_bayesian_ci/location=*/station=*/
    ///   anchor=*/widths.parquet
    /// Picks the LATEST anchor per (station, valid_time, lead) — multiple
    /// anchors may have predicted the same (valid, lead) tuple if the
    /// cron has run more than once for it; the freshest anchor is the
    /// one whose forecast inputs are most recent.
    /// </summary>
    private IReadOnlyList<SitePages.BayesianCiPoint> QueryBayesianCi(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var subdir = Path.Combine(_cfg.Storage.PredictionsPath, "precipitation_bayesian_ci");
        if (!Directory.Exists(subdir))
        {
            _log.LogInformation(
                "Bayesian CI tree at {Dir} not present yet — confidence panel will be absent until WeatherProbabilistic's predict-bayesian.yml has fired at least once.",
                subdir);
            return Array.Empty<SitePages.BayesianCiPoint>();
        }
        var glob = ParquetReader.Glob(Path.Combine(subdir, "**", "*.parquet"));
        // Latest anchor per (station, valid_time, lead). Quote dotted column
        // names ("p_wet_q0.1" etc) — DuckDB supports double-quoted identifiers.
        var sql = $@"
WITH ranked AS (
  SELECT station, valid_time, lead, anchor,
         p_wet_mean, p_wet_std,
         ""p_wet_q0.05"" AS q05,
         ""p_wet_q0.1""  AS q10,
         ""p_wet_q0.5""  AS q50,
         ""p_wet_q0.9""  AS q90,
         ""p_wet_q0.95"" AS q95,
         ci80_width, ci90_width,
         row_number() OVER (PARTITION BY station, valid_time, lead ORDER BY anchor DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE valid_time >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND valid_time <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT station, valid_time, lead, anchor, p_wet_mean, p_wet_std,
       q05, q10, q50, q90, q95, ci80_width, ci90_width
FROM ranked WHERE rn = 1 ORDER BY station, valid_time, lead";

        return ParquetReader.Query(sql, r => new SitePages.BayesianCiPoint(
            StationCode:   r.GetString(0),
            ValidTimeUtc:  r.GetDateTime(1),
            LeadHours:     (int)r.GetInt64(2),
            AnchorDate:    r.GetDateTime(3),
            PWetMean:      r.GetDouble(4),
            PWetStd:       r.GetDouble(5),
            PWetQ05:       r.GetDouble(6),
            PWetQ10:       r.GetDouble(7),
            PWetQ50:       r.GetDouble(8),
            PWetQ90:       r.GetDouble(9),
            PWetQ95:       r.GetDouble(10),
            Ci80Width:     r.GetDouble(11),
            Ci90Width:     r.GetDouble(12)),
            _log, "Bayesian CI tree empty — confidence panel will be absent.", ct);
    }

    private IReadOnlyList<SitePages.NwpPrecipProbForecastPoint> QueryNwpPrecipProbabilities(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.ForecastsPath, "**", "*.parquet"));

        var nwpFilter = "('gfs_seamless','ecmwf_ifs025','icon_seamless','meteofrance_seamless'," +
                        "'ukmo_seamless','gem_seamless','ecmwf_aifs025_single','jma_seamless')";

        var sql = $@"
WITH ranked AS (
  SELECT Model, RunTimeUtc, ValidTimeUtc, PrecipitationProbability,
         row_number() OVER (PARTITION BY Model, ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
    AND Model IN {nwpFilter}
    AND PrecipitationProbability IS NOT NULL
    AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT Model, ValidTimeUtc, PrecipitationProbability
FROM ranked WHERE rn = 1 ORDER BY Model, ValidTimeUtc";

        return ParquetReader.Query(sql, r => new SitePages.NwpPrecipProbForecastPoint(
            Model:               r.GetString(0),
            ValidTimeUtc:        r.GetDateTime(1),
            ProbabilityPercent:  r.GetDouble(2)),
            _log, "Per-NWP precipitation_probability tree empty — overlay panel will be absent.", ct);
    }

    /// <summary>
    /// Met Office DataHub Spot forecasts for the configured location. Pre-
    /// filtered to "latest RunTime per ValidTime" so the renderer doesn't have
    /// to dedupe; bounded by the rendering window. Returns empty when the
    /// model partition isn't on disk yet (fresh deploy or capture not started).
    /// </summary>
    private IReadOnlyList<SitePages.MetOfficeSpotForecastPoint> QueryMetOfficeSpotForecasts(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.ForecastsPath, "**", "*.parquet"));

        var sql = $@"
WITH ranked AS (
  SELECT RunTimeUtc, ValidTimeUtc, Temperature2m, PrecipitationProbability,
         row_number() OVER (PARTITION BY ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
    AND Model = 'met_office_spot'
    AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT RunTimeUtc, ValidTimeUtc, Temperature2m, PrecipitationProbability
FROM ranked WHERE rn = 1 ORDER BY ValidTimeUtc";

        return ParquetReader.Query(sql, r => new SitePages.MetOfficeSpotForecastPoint(
            RunTimeUtc:                      r.GetDateTime(0),
            ValidTimeUtc:                    r.GetDateTime(1),
            Temperature2m:                   r.IsDBNull(2) ? null : r.GetDouble(2),
            PrecipitationProbabilityPercent: r.IsDBNull(3) ? null : r.GetDouble(3)),
            _log, "Met Office Spot tree empty — comparison line will be absent.", ct);
    }

    private IReadOnlyList<SitePages.StartHourForecastPoint> QueryStartHourPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
            WeatherBlend.Commands.StartHourPredictCommand.PredictionsSubdir, "**", "*.parquet"));

        // Probe in case no curves have been written yet (fresh deploy, first
        // run before the predict-and-render cycle has produced anything).
        // Without this the read_parquet would throw on a missing tree.
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "ConditionalProb"))
        {
            _log.LogWarning("Start-hour curves tree empty — best-start column will be absent until predict runs once.");
            return Array.Empty<SitePages.StartHourForecastPoint>();
        }

        // Bounded by TargetDateUtc so we read the same forward window the
        // dry-window page draws, no further. ProbHasDryWindow / Daily prob is
        // mirrored on every row so renderer / verify can read it without
        // re-joining to the dry-window tree.
        var sql = $@"
SELECT TruthStation, WindowHours, ModelVersion, PredictionMadeAtUtc, TargetDateUtc, LeadHours,
       StartHourUtc, ConditionalProb, CalibratedProb, DailyProbAnyBlock
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND TargetDateUtc >= TIMESTAMP '{start.Date:yyyy-MM-dd HH:mm:ss}'
  AND TargetDateUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, WindowHours, LeadHours, TargetDateUtc, StartHourUtc";

        return ParquetReader.Query(conn, sql, r => new SitePages.StartHourForecastPoint(
            Station:           r.GetString(0),
            WindowHours:       r.GetInt32(1),
            Version:           r.GetString(2),
            PredictedAtUtc:    r.GetDateTime(3),
            TargetDateUtc:     r.GetDateTime(4),
            LeadHours:         r.GetInt32(5),
            StartHourUtc:      r.GetInt32(6),
            ConditionalProb:   r.GetDouble(7),
            CalibratedProb:    r.GetDouble(8),
            DailyProbAnyBlock: r.GetDouble(9)),
            _log, "Start-hour curves tree empty — best-start column will be absent.", ct);
    }

}
