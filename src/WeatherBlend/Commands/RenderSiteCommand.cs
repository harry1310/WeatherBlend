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

        if (_cfg.Locations.Count == 0)
            throw new InvalidOperationException("No locations configured — site cannot render.");

        _log.LogInformation(
            "Render site: predictions [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], rolling {Rolling}d, output → {Dir}, locations={N}",
            windowStart, predictionEnd, rollingWindowDays, outputDir, _cfg.Locations.Count);

        // Phase D (2026-05-20): output is now /<slug>/... per location plus
        // a tiny set of top-level pages (picker, specs, about, styles, chart).
        // DuckDB scans fetch unfiltered once below; the per-loc loop slices
        // each list to the loop iteration's LocationName.
        var predictionsAllLocs = _predictions.GetTemperaturePredictions(
            locations: null, windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} temperature prediction rows (all locations).",
            predictionsAllLocs.Count);

        // Precip + dry-window come back as the canonical domain rows from the
        // repo; the renderer projects to its lighter SitePages records below.
        // Pass null for the station/cell filter — render scans the whole subtree
        // unlike verify, which limits to its requested stations.
        var precipRows = _predictions.GetPrecipitationPredictions(stations: null, windowStart, predictionEnd, ct);

        // Phase 3f rainfall_amount predictions — same window as P(wet), scoped
        // per-loc below. Empty list when the rainfall_amount tree hasn't been
        // synced from R2 or no location has 3f trained yet (Membury-only today).
        var rainfallAmountAllLocs = _predictions.GetRainfallAmountPredictions(
            stations: null, windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} rainfall_amount prediction rows (all locations).",
            rainfallAmountAllLocs.Count);
        var precipAllLocs = precipRows.Select(r => new SitePages.PrecipForecastPoint(
            Station:         r.TruthStation,
            Version:         r.ModelVersion,
            LocationName:    r.LocationName,
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
            ConformalSetTag: r.ConformalSetTag,
            ProbWetQ05:      r.ProbWetQ05,
            ProbWetQ95:      r.ProbWetQ95,
            Ci80Width:       r.Ci80Width,
            Ci90Width:       r.Ci90Width,
            UpperAirIncluded: r.UpperAirIncluded,
            UpperAirAgeHours: r.UpperAirAgeHours)).ToList();
        _log.LogInformation("Loaded {N} precipitation prediction rows (all locations).", precipAllLocs.Count);

        var feelsLikeAllLocs = QueryFeelsLikePredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} feels-like prediction rows (all locations).", feelsLikeAllLocs.Count);

        var rockSurfaceAllLocs = QueryRockSurfacePredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} rock-surface prediction rows (all locations).", rockSurfaceAllLocs.Count);

        var windGustAllLocs = QueryWindGustByValidTime(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} wind-gust prediction rows (all locations).", windGustAllLocs.Count);

        var windAllLocs = QueryWindPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} wind prediction rows ({M} versions, all locations).",
            windAllLocs.Count, windAllLocs.Select(p => p.Version).Distinct().Count());

        var windDirAllLocs = QueryWindDirectionPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} wind_mvn direction prediction rows (all locations).", windDirAllLocs.Count);

        var waveAllLocs = QueryWavePredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} wave_height prediction rows (all locations).", waveAllLocs.Count);

        var dryWindowRows = _predictions.GetDryWindowPredictions(cells: null, windowStart, predictionEnd, ct);
        var dryWindowAllLocs = dryWindowRows.Select(r => new SitePages.DryWindowForecastPoint(
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
            EpistemicProbDryWindowMean:  r.EpistemicProbDryWindowMean,
            EpistemicProbDryWindowQ10:   r.EpistemicProbDryWindowQ10,
            EpistemicProbDryWindowQ90:   r.EpistemicProbDryWindowQ90,
            EpistemicSigmaUsed:          r.EpistemicSigmaUsed,
            ConformalSetTag:             r.ConformalSetTag)).ToList();
        _log.LogInformation("Loaded {N} dry-window prediction rows (all locations).", dryWindowAllLocs.Count);

        var startHourAllLocs = QueryStartHourPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} start-hour curve rows (all locations).", startHourAllLocs.Count);

        // Spot loads with extra lookback beyond the render window: the wind
        // skill chart's Spot comparison line replays each element_wind
        // verify sidecar's (asOf − windowDays) scoring window, and a sidecar
        // sitting at the left edge of the chart's 30-day axis reaches ~14+5
        // days further back than windowStart. 21 days of margin covers the
        // element verify defaults (window 14d, latency 5d) comfortably; the
        // eyeball + rolling charts ignore the extra rows via their own
        // window filters.
        var spotLookbackStart = windowStart.AddDays(-SpotVerifyLookbackExtraDays);
        var metOfficeSpotAllLocs = QueryMetOfficeSpotForecasts(spotLookbackStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} Met Office Spot forecast rows (all locations).", metOfficeSpotAllLocs.Count);

        var nwpPopAllLocs = QueryNwpPrecipProbabilities(windowStart, predictionEnd, ct);
        var nwpPrecipRatesAllLocs = QueryNwpPrecipRates(windowStart, predictionEnd, ct);
        var nwpTemperaturesAllLocs = QueryNwpTemperatures(windowStart, predictionEnd, ct);
        var lowCloudByValidAllLocs = QueryLowCloudSignals(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded low-cloud / mist signal for {N} location-keyed inner dicts.", lowCloudByValidAllLocs.Count);
        _log.LogInformation("Loaded {N} per-NWP precipitation-probability rows ({M} models).",
            nwpPopAllLocs.Count, nwpPopAllLocs.Select(p => p.Model).Distinct().Count());

        var verifyHistory = LoadVerifyHistory(_cfg.Storage.ReportsPath);
        _log.LogInformation("Loaded {N} verify-history JSON sidecars.", verifyHistory.Count);

        var dryWindowConformalTau = LoadDryWindowConformalTau();
        _log.LogInformation("Loaded {N} dry-window conformal τ values across (version, lead).",
            dryWindowConformalTau.Count);

        var precipConformalTau = LoadPrecipConformalTau();
        _log.LogInformation("Loaded {N} precip conformal τ values across (station, version, lead).",
            precipConformalTau.Count);

        // Champion versions are global (keyed by station = location-name for
        // temp / EA slug for precip). Resolved BEFORE the phase map so the
        // CURRENT champion's phase is always in it — see the union below.
        var tempChampions = _metadata.GetChampionsByStation("temperature");
        var precipCurrentByStation = _metadata.GetChampionsByStation("precipitation");

        // PhaseByVersion is computed across ALL locations (a single dict
        // keyed by version string); rolling-MAE/Brier compute per-loc inside
        // the foreach below.
        //
        // Union the prediction-derived versions with the current champion
        // versions. A champion minted by today's retrain has ZERO predictions
        // until the next predict cycle runs (the retrain→predict race — e.g.
        // Membury retrained after the predict step on 2026-06-14), so it never
        // appears in the prediction keys. Its phase would then be absent here,
        // and ChampionMatcher's phase fallback — which needs BOTH the row's and
        // the champion's phase — silently fails to match the PREVIOUS champion's
        // still-valid rows, blanking the whole overview. Feeding the champion
        // versions in resolves their phase off disk (the bundle's
        // training_metadata.json is synced even when no predictions exist yet),
        // so the overview keeps rendering on the prior champion's phase until
        // the new one has predictions. Match on PHASE, never a specific version.
        var tempPhaseKeys = WithChampionVersions(
            predictionsAllLocs.Select(p => (Station: p.LocationName, Version: p.ModelVersion)),
            tempChampions);
        var precipPhaseKeys = WithChampionVersions(
            precipAllLocs.Select(p => (Station: p.Station, Version: p.Version)),
            precipCurrentByStation);
        var phaseByVersion = _metadata.GetPhaseByVersion(
            tempPhaseKeys,
            precipPhaseKeys,
            dryWindowAllLocs.Select(d => (d.Station, d.WindowHours, d.Version)),
            // Element keys are LOCATION-keyed (element bundles live under
            // data/models/{element}/{location}/{version}). Wind is the only
            // element tree whose page groups by phase today; pass more trees
            // here if another element page grows phase lines.
            windAllLocs.Select(p => (Station: p.LocationName, p.Version)));
        // Precip uses a longer 30-day rolling window than temp's because wet
        // hours are sparser — the verify command's defaults pin this rule
        // (PrecipVerifyCommand: 30d window vs TempVerifyCommand: 14d).
        const int precipRollingWindowDays = 30;

        _log.LogInformation("Phase map: {Entries}",
            string.Join(", ", phaseByVersion.Select(kv => $"{kv.Key}→{kv.Value}")));
        _log.LogInformation("Champion (temperature) by station: {Entries}",
            tempChampions.Count == 0 ? "(none)" : string.Join(", ", tempChampions.Select(kv => $"{kv.Key}→{kv.Value}")));
        _log.LogInformation("Champion (precipitation): {Entries}",
            precipCurrentByStation.Count == 0
                ? "(none)"
                : string.Join(", ", precipCurrentByStation.Select(kv => $"{kv.Key}→{kv.Value}")));

        // All-locations station slug union — needed by the cross-location
        // Specs page's feature spec rows. Per-loc loops below scope this down
        // to one location's stations. s.Slug is ea_* for EA gauges, wl_* for
        // WeatherLink cove gauges (Sennen → Lands End).
        var allStationSlugs = new HashSet<string>(
            _cfg.Locations.SelectMany(loc => loc.Rainfall.Stations.Select(s => s.Slug)),
            StringComparer.Ordinal);

        var locationDescriptors = _cfg.Locations.Select((loc, idx) => new SitePages.LocationDescriptor(
            Name: loc.Name,
            DisplayName: string.IsNullOrWhiteSpace(loc.DisplayName) ? loc.Name : loc.DisplayName,
            // Pool-only gauges (e.g. Princetown) feed pooled 3o training only —
            // never a per-station product, so they never appear in any rendered
            // station list (forecasts / skill / dry-window / index all read this).
            RainStationSlugs: loc.ProductRainfallStations.Select(s => s.Slug).ToList(),
            IsPrimary: idx == 0,
            Tabs: loc.Tabs.ToList(),
            // The climbing range (05:00–20:00 UTC) is defined once in
            // SitePages.ClimbingRange*; a non-climbing location (Membury) opts
            // into the full 24h day via overview.allHours.
            OverviewFirstVisibleHourUtc: loc.Overview.AllHours ? 0 : SitePages.ClimbingRangeFirstHourUtc,
            OverviewLastVisibleHourUtcExclusive: loc.Overview.AllHours ? 24 : SitePages.ClimbingRangeLastHourExclusiveUtc,
            VillageRain: loc.VillageRain is { Weights.Count: > 0 } vr
                ? new SitePages.VillageRainSpec(
                    Label: string.IsNullOrWhiteSpace(vr.Label) ? $"{loc.DisplayName} (on-site estimate)" : vr.Label,
                    FittedOn: vr.FittedOn,
                    WeightsBySlug: vr.Weights.ToDictionary(
                        kv => _cfg.ResolveStationSlug(kv.Key), kv => kv.Value, StringComparer.Ordinal))
                : null,
            // Sea-state badge thresholds — marine locations with a
            // seaStateBadge config block only (Sennen). No block = null =
            // no badge on that location's overview tiles.
            SeaStateBadge: loc.Marine?.SeaStateBadge is { } ssb
                ? new SeaStateBadgeSpec(
                    TideHighMsl: ssb.TideHighMsl,
                    RunUpProxy: ssb.RunUpProxy,
                    WindMinMph: ssb.WindMinMph,
                    WindSectorFromDeg: ssb.WindSectorFromDeg,
                    WindSectorToDeg: ssb.WindSectorToDeg)
                : null,
            ShowClimbingConditions: loc.ShowClimbingConditions,
            // Low-cloud alert noun ("cloud base below ___"): "tor" only for an
            // INLAND climbing crag (Bonehill); every other location — marine
            // crags (Sennen: Harry prefers "hilltops" over "cliff" for the
            // inland high ground the cloud actually sits on, 2026-06-20) and
            // lowland sites (Membury) — uses the default "the hilltops".
            CloudFeatureNoun: loc.ShowClimbingConditions && loc.Marine?.SeaStateBadge is null
                            ? "tor"
                            : "the hilltops",
            // Drying-model gate, resolved per location (global block + this
            // location's rockSurface override). Default false keeps the verdict
            // on the pre-drying behaviour until a location is calibrated + flipped.
            SurfaceWaterEnabled: _cfg.RockSurface.ResolveFor(loc.RockSurface).SurfaceWaterEnabled,
            // Greasy-margin width for the climbing index's continuous greasiness
            // ramp, resolved per location (global block + this location's override).
            GreasyMarginC: _cfg.RockSurface.ResolveFor(loc.RockSurface).GreasyMarginC,
            // Wind-exposure multiplier for the climbing index's wind factor
            // (1.0 = exposed-tor curve as-is; sheltered crags scale down).
            WindExposure: loc.WindExposure)).ToList();

        // Top-level model summaries / feature spec rows surface ALL locations'
        // active versions on /specs.html. Per-loc Models page recomputes with
        // a tighter station filter so a demoted station can't appear.
        var modelSummariesAll = LoadModelSummaries(predictionsAllLocs, precipAllLocs, dryWindowAllLocs, rainfallAmountAllLocs, allStationSlugs);
        var featureSpecRowsAll = LoadFeatureSpecRows(predictionsAllLocs, precipAllLocs, dryWindowAllLocs, rainfallAmountAllLocs, allStationSlugs, modelSummariesAll);
        _log.LogInformation("Loaded {N} model summaries / {M} feature-spec rows (all locations).",
            modelSummariesAll.Count, featureSpecRowsAll.Count);

        // === Output directory hygiene ===
        Directory.CreateDirectory(outputDir);
        // Clean top-level HTML files only — per-loc subdirs get cleaned at
        // the start of their own iteration. Non-HTML (styles.css, chart.js)
        // overwrites in place, no stale-file risk.
        foreach (var stale in Directory.EnumerateFiles(outputDir, "*.html"))
            File.Delete(stale);

        // Top-level inputs for the picker / specs / about pages. No
        // RenderingFor (top-level chrome) and empty AssetPrefix (relative
        // hrefs from outputDir).
        var topLevelInputs = BuildTopLevelInputs(
            now, windowStart, locationDescriptors, modelSummariesAll, featureSpecRowsAll,
            phaseByVersion, verifyHistory);

        var totalFiles = 0;

        // === Per-location render ===
        foreach (var loc in _cfg.Locations)
        {
            var descriptor = locationDescriptors.First(l => l.Name == loc.Name);
            var locDir = Path.Combine(outputDir, loc.Name);
            Directory.CreateDirectory(locDir);
            foreach (var stale in Directory.EnumerateFiles(locDir, "*.html"))
                File.Delete(stale);

            _log.LogInformation("Render {Loc} (tabs: {Tabs})", loc.Name, string.Join(",", loc.Tabs));

            // === Scope EVERY all-locations collection to this location ===
            // The invariant: the SiteInputs built below is single-location by
            // construction — every list/dict it carries holds only this loc's
            // rows. No page does its own location filtering (Phase D, the
            // 2026-05-21 single-location-SiteInputs refactor). A page that
            // forgets to filter can't leak another location's data because
            // there's nothing else in the object to leak.
            var locStationSlugs = new HashSet<string>(
                loc.Rainfall.Stations.Select(s => s.Slug),
                StringComparer.Ordinal);

            bool IsThisLoc(string locationName) =>
                string.Equals(locationName, loc.Name, StringComparison.Ordinal);

            var predictions = predictionsAllLocs.Where(p => IsThisLoc(p.LocationName)).ToList();
            // Precip + dry-window rows are keyed by EA station, not LocationName
            // — scope by this loc's configured station slugs.
            var precip = precipAllLocs
                .Where(p => locStationSlugs.Count == 0 || locStationSlugs.Contains(p.Station))
                .ToList();
            // Rainfall_amount rows carry both LocationName + TruthStation; scope
            // by location since 3f is per-location-active per phases.yaml.
            var rainfallAmount = rainfallAmountAllLocs
                .Where(r => IsThisLoc(r.LocationName))
                .ToList();
            var dryWindow = dryWindowAllLocs
                .Where(d => locStationSlugs.Count == 0 || locStationSlugs.Contains(d.Station))
                .ToList();
            var feelsLike = feelsLikeAllLocs.Where(f => IsThisLoc(f.LocationName)).ToList();
            var rockSurface = rockSurfaceAllLocs.Where(r => IsThisLoc(r.LocationName)).ToList();
            // Wind-gust: per-location dict, latest PredictedAtUtc per ValidTime
            // already collapsed by the SQL window function. Empty dict when the
            // wind_gust tree is unsynced — UTCI pop-out falls back to wind-only.
            var windGustByValid = windGustAllLocs
                .Where(g => IsThisLoc(g.LocationName))
                .ToDictionary(g => g.ValidTimeUtc, g => g.GustMs);
            var windPredictions = windAllLocs.Where(w => IsThisLoc(w.LocationName)).ToList();
            var windDirectionPredictions = windDirAllLocs.Where(w => IsThisLoc(w.LocationName)).ToList();
            var wavePredictions = waveAllLocs.Where(w => IsThisLoc(w.LocationName)).ToList();
            var startHour = startHourAllLocs.Where(s => IsThisLoc(s.LocationName)).ToList();
            var metOfficeSpot = metOfficeSpotAllLocs.Where(m => IsThisLoc(m.LocationName)).ToList();
            var nwpPop = nwpPopAllLocs.Where(p => IsThisLoc(p.LocationName)).ToList();
            var nwpPrecipRates = nwpPrecipRatesAllLocs.Where(r => IsThisLoc(r.LocationName)).ToList();
            var nwpTemperatures = nwpTemperaturesAllLocs.Where(t => IsThisLoc(t.LocationName)).ToList();
            var lowCloud = lowCloudByValidAllLocs.TryGetValue(loc.Name, out var lcInner)
                ? lcInner
                : (IReadOnlyDictionary<DateTime, SitePages.LowCloudSignal>)
                    new Dictionary<DateTime, SitePages.LowCloudSignal>();

            // Per-loc fetches (truth varies by location — different gauge / cell).
            var truth = _truth.GetEra5Hourly(loc.Name, windowStart.AddDays(-3), now, ct);
            var metar = _truth.GetMetarTemperature(loc.Metar.Primary, windowStart, now, ct);
            var moObs = _truth.GetMetOfficeObsTemperature(loc.Name, windowStart, now, ct);
            var rainfall = LoadRainfallTruth(windowStart, now, precip, ct);

            // Per-loc champion (one version per (target, station) — the per-lead
            // ChampionByLead override was removed 2026-06-15).
            var currentVersion = tempChampions.TryGetValue(loc.Name, out var tcv) ? tcv : "";

            // Per-loc rolling metrics.
            var rolling = ComputeRollingMae(predictions, truth, phaseByVersion, rollingWindowDays);
            var rollingBrier = ComputeRollingBrier(precip, rainfall, phaseByVersion, precipRollingWindowDays);

            // Met Office Spot comparison series (computed fresh each render,
            // like the resident rolling metrics above, so the Spot lines get
            // full history automatically). Same truth, same windows, same
            // per-day cadence; Spot rows bucket per panel lead — see the
            // Compute* docs. Never feeds champion/lead-policy/drift logic.
            var spotRollingMae = ComputeSpotRollingMae(
                metOfficeSpot, truth, Leads.ForecastsTempRain, rollingWindowDays);
            var spotRollingBrier = ComputeSpotRollingBrier(
                metOfficeSpot, rainfall, Leads.ForecastsTempRain, precipRollingWindowDays);

            // Spot wind MAE replays this location's element_wind verify
            // sidecar windows against ERA5 WindSpeed10m truth, so each Spot
            // point shares its X (= sidecar AsOfUtc) and scoring window with
            // the resident verify points on the wind skill chart. ERA5 wind
            // is only loaded when there's something to replay.
            var isPrimaryLoc = string.Equals(loc.Name, _cfg.Locations[0].Name, StringComparison.Ordinal);
            var spotWindWindows = verifyHistory
                .Where(f => string.Equals(f.Target, "element_wind", StringComparison.Ordinal))
                .Where(f => f.Rows.Any(r => r.Station is null
                    ? isPrimaryLoc   // pre-Phase-C sidecars carried no station — primary-loc fallback
                    : string.Equals(r.Station, loc.Name, StringComparison.Ordinal)))
                .Select(f => (f.AsOfUtc, f.WindowDays, f.LatencyDays))
                .Distinct()
                .ToList();
            IReadOnlyList<SitePages.RollingMaePoint> spotWindRollingMae = Array.Empty<SitePages.RollingMaePoint>();
            if (spotWindWindows.Count > 0 && metOfficeSpot.Any(m => m.WindSpeed10mMs.HasValue))
            {
                var windTruthStart = spotWindWindows.Min(w => w.AsOfUtc.AddDays(-w.WindowDays));
                var windTruthEnd = spotWindWindows.Max(w => w.AsOfUtc);
                var windTruth = _truth.GetEra5Hourly(loc.Name, windTruthStart, windTruthEnd, ct, "WindSpeed10m");
                spotWindRollingMae = ComputeSpotWindVerifyMae(metOfficeSpot, spotWindWindows, windTruth, Leads.Short);
            }

            // Per-loc model summaries (scoped to loc's stations).
            var locModelSummaries = LoadModelSummaries(predictions, precip, dryWindow, rainfallAmount, locStationSlugs, elementLocationFilter: loc.Name);
            var locFeatureSpecRows = LoadFeatureSpecRows(predictions, precip, dryWindow, rainfallAmount, locStationSlugs, locModelSummaries);

            // Active wave_height bundle snapshot for the Sea-state Models
            // page (Sennen today). Null for non-marine locations / unsynced
            // model trees — the page renders its empty state.
            var waveModelCard = LoadWaveModelCard(loc.Name);

            var locInputs = new SitePages.SiteInputs
            {
                ActiveLocationName = loc.Name,
                RenderingFor = descriptor,
                AssetPrefix = "../",
                LocationDisplay = string.IsNullOrWhiteSpace(loc.DisplayName) ? loc.Name : loc.DisplayName,
                Latitude = loc.Latitude,
                Longitude = loc.Longitude,
                ElevationMeters = loc.ElevationMeters,
                MetarStation = loc.Metar.Primary,
                GeneratedAtUtc = now,
                WindowStartUtc = windowStart,
                Predictions = predictions,
                TruthByTime = truth,
                MetarByTime = metar,
                MetOfficeObsByTime = moObs,
                LowCloudByValid = lowCloud,
                RollingMae = rolling,
                RollingBrier = rollingBrier,
                SpotRollingMae = spotRollingMae,
                SpotRollingBrier = spotRollingBrier,
                SpotWindRollingMae = spotWindRollingMae,
                PrecipPredictions = precip,
                DryWindowPredictions = dryWindow,
                RainfallAmountPredictions = rainfallAmount,
                DryWindowConformalTau = dryWindowConformalTau,
                PrecipConformalTau = precipConformalTau,
                DryWindowDaytime = _cfg.DryWindow.BuildDaytimeWindow(),
                PhaseByVersion = phaseByVersion,
                RainfallTruth = rainfall,
                CurrentVersion = currentVersion,
                PrecipCurrentByStation = precipCurrentByStation,
                ActiveStationSlugs = locStationSlugs,
                Locations = locationDescriptors,
                ModelSummaries = locModelSummaries,
                FeatureSpecRows = locFeatureSpecRows,
                FeelsLikePredictions = feelsLike,
                RockSurfacePredictions = rockSurface,
                WindGustByValidMs = windGustByValid,
                WindPredictions = windPredictions,
                WindDirectionPredictions = windDirectionPredictions,
                WavePredictions = wavePredictions,
                WaveModelCard = waveModelCard,
                StartHourPredictions = startHour,
                MetOfficeSpotForecasts = metOfficeSpot,
                NwpPrecipProbabilities = nwpPop,
                NwpPrecipRates = nwpPrecipRates,
                NwpTemperatures = nwpTemperatures,
                VerifyHistory = verifyHistory,
            };

            totalFiles += await RenderLocationPagesAsync(locDir, locInputs, descriptor, ct);
        }

        // === Top-level pages ===
        await File.WriteAllTextAsync(Path.Combine(outputDir, "index.html"),
            SitePages.RenderHomePicker(topLevelInputs), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "specs.html"),
            SitePages.RenderModelsSpec(topLevelInputs), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "about.html"),
            SitePages.RenderAbout(topLevelInputs), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "styles.css"),
            SitePages.Stylesheet(), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "chart.js"),
            SitePages.ChartScript(), ct);
        totalFiles += 5;

        _log.LogInformation("Site rendered → {Dir} ({Files} files across {Locs} locations + top-level)",
            outputDir, totalFiles, _cfg.Locations.Count);
        return 0;
    }

    /// <summary>
    /// Render one location's per-tab pages into <paramref name="locDir"/>.
    /// Tab-gated: each block runs only when its tab id is in
    /// <paramref name="loc"/>.<see cref="SitePages.LocationDescriptor.Tabs"/>.
    /// Returns the number of files written.
    /// </summary>
    private async Task<int> RenderLocationPagesAsync(
        string locDir, SitePages.SiteInputs locInputs, SitePages.LocationDescriptor loc, CancellationToken ct)
    {
        int count = 0;

        async Task Write(string fileName, string html)
        {
            await File.WriteAllTextAsync(Path.Combine(locDir, fileName), html, ct);
            count++;
        }

        if (loc.HasTab("overview"))
        {
            for (int n = 0; n <= SitePages.MaxHomeDayOffset; n++)
            {
                var file = n == 0 ? "index.html" : $"index-{n}.html";
                await Write(file, SitePages.RenderIndex(locInputs, n));
            }
        }

        if (loc.HasTab("temperature"))
        {
            foreach (var lead in Leads.ForecastsTempRain)
                await Write($"forecasts-temp-{lead}h.html", SitePages.RenderForecastsTemp(locInputs, lead));
        }

        if (loc.HasTab("rain"))
        {
            foreach (var lead in Leads.ForecastsTempRain)
                await Write($"forecasts-rain-{lead}h.html", SitePages.RenderForecastsRain(locInputs, lead));
        }

        if (loc.HasTab("sea_state"))
        {
            // Single page — wave_height_lgb is ANY-LEAD (one continuous
            // hourly forecast per cycle), so there's no lead axis to sub-nav.
            await Write("forecasts-sea-state.html", SitePages.RenderSeaState(locInputs));
        }

        if (loc.HasTab("wind"))
        {
            // Per-lead pages, one per trained wind lead (24 / 48 / 72 h).
            // Mirrors temp/rain so the lead sub-nav reads consistently.
            foreach (var lead in Leads.Short)
                await Write($"forecasts-wind-{lead}h.html", SitePages.RenderForecastsWind(locInputs, lead));
        }

        if (loc.HasTab("dry_window"))
        {
            var activeSet = locInputs.ActiveStationSlugs;
            var dryStations = locInputs.DryWindowPredictions
                .Select(d => d.Station)
                .Where(s => activeSet.Count == 0 || activeSet.Contains(s))
                .Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
            await Write("forecasts-dry-window.html", SitePages.RenderDryWindow(locInputs, null));
            for (int i = 1; i < dryStations.Count; i++)
            {
                var slug = SitePages.StationSlug(dryStations[i]);
                await Write($"forecasts-dry-window-{slug}.html",
                    SitePages.RenderDryWindow(locInputs, slug));
            }
        }

        if (loc.HasTab("skill"))
        {
            if (loc.HasTab("temperature"))
                await Write("skill-temperature.html", SitePages.RenderTempSkill(locInputs));

            if (loc.HasTab("rain"))
            {
                var rainStations = SitePages.GetRainSkillStations(locInputs, loc);
                await Write("skill-rainfall.html", SitePages.RenderRainSkill(locInputs, null, locationName: null));
                for (int i = 1; i < rainStations.Count; i++)
                {
                    var slug = SitePages.StationSlug(rainStations[i]);
                    await Write($"skill-rainfall-{slug}.html",
                        SitePages.RenderRainSkill(locInputs, slug, locationName: null));
                }
            }

            if (loc.HasTab("dry_window"))
            {
                var dwStations = SitePages.GetDryWindowSkillStations(locInputs);
                await Write("skill-dry-window.html", SitePages.RenderDryWindowSkill(locInputs, null));
                for (int i = 1; i < dwStations.Count; i++)
                {
                    var slug = SitePages.StationSlug(dwStations[i]);
                    await Write($"skill-dry-window-{slug}.html",
                        SitePages.RenderDryWindowSkill(locInputs, slug));
                }
            }

            if (loc.HasTab("wind"))
            {
                // Single page for now — no station axis (wind is a single-
                // location quantity, like temperature). Per-phase rolling MAE
                // lines arrive once verify scores wind_speed_lgb / wind_blend
                // against Dunkeswell SYNOP truth post-Sunday-retrain.
                await Write("skill-wind.html", SitePages.RenderWindSkill(locInputs));
            }
        }

        if (loc.HasTab("models"))
        {
            // Three per-target files with a sub-nav between them — same shape
            // as Skill. The top-nav "Models" button points at models-temp.html
            // for locations with temperature; falls through to rain when temp
            // isn't a tab.
            if (loc.HasTab("temperature"))
                await Write("models-temp.html", SitePages.RenderModels(locInputs, "temperature"));
            if (loc.HasTab("rain"))
                await Write("models-rain.html", SitePages.RenderModels(locInputs, "precipitation"));
            if (loc.HasTab("dry_window"))
                await Write("models-dry-window.html", SitePages.RenderModels(locInputs, "dry_window"));
            if (loc.HasTab("wind"))
                await Write("models-wind.html", SitePages.RenderModels(locInputs, "wind"));
            if (loc.HasTab("sea_state"))
                // Bespoke page (any-lead wave bundle has no per-lead card) —
                // see SitePages.ModelsSeaState.cs.
                await Write("models-sea-state.html", SitePages.RenderModelsSeaState(locInputs));
            if (loc.HasTab("element"))
                await Write("models-elements.html", SitePages.RenderModels(locInputs, "element"));
        }

        return count;
    }

    /// <summary>
    /// SiteInputs shell for top-level pages (picker / specs / about). Carries
    /// the full cross-location model summaries + feature-spec rows so Specs
    /// can render every location's deployment. No <c>RenderingFor</c>, empty
    /// <c>AssetPrefix</c>.
    /// </summary>
    private static SitePages.SiteInputs BuildTopLevelInputs(
        DateTime now, DateTime windowStart,
        IReadOnlyList<SitePages.LocationDescriptor> locations,
        IReadOnlyList<SitePages.ModelSummary> modelSummaries,
        IReadOnlyList<SitePages.FeatureSpecRow> featureSpecRows,
        IReadOnlyDictionary<string, string> phaseByVersion,
        IReadOnlyList<Models.VerifyHistoryFile> verifyHistory)
    {
        return new SitePages.SiteInputs
        {
            ActiveLocationName = "",
            RenderingFor = null,
            AssetPrefix = "",
            LocationDisplay = "",
            Latitude = 0,
            Longitude = 0,
            ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = now,
            WindowStartUtc = windowStart,
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            MetOfficeObsByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
            PhaseByVersion = phaseByVersion,
            Locations = locations,
            ModelSummaries = modelSummaries,
            FeatureSpecRows = featureSpecRows,
            VerifyHistory = verifyHistory,
        };
    }

    private IReadOnlyList<SitePages.ModelSummary> LoadModelSummaries(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        IReadOnlyList<SitePages.DryWindowForecastPoint> dryWindow,
        IReadOnlyList<RainfallAmountPredictionRow> rainfallAmount,
        IReadOnlySet<string> activeStationSlugs,
        string? elementLocationFilter = null)
    {
        // Only load metadata for versions that actually emitted predictions in the
        // window. Anything else is stale / experimental / deleted and the Models page
        // would list it with no corresponding forecast activity.
        // Per-station targets (precip + dry-window) are also filtered through
        // activeStationSlugs so a demoted station doesn't get a Models card
        // just because its historical predictions are still on disk.
        var modelsRoot = _cfg.Storage.ModelsPath;
        var summaries = new List<SitePages.ModelSummary>();

        // Temperature is per-station (per-location) on disk since 2026-05-14.
        // Derive the (station, version) pair from each prediction's LocationName
        // — same shape as the precip block immediately below.
        foreach (var (station, version) in predictions
            .Select(p => (p.LocationName, p.ModelVersion))
            .Distinct())
        {
            var dir = Path.Combine(modelsRoot, "temperature", station, version);
            // Phase C commit 4 (2026-05-20): composite now includes the
            // station/location segment so Bonehill + Membury surface as
            // separate Models-page sections (was a collision-on-"temperature"
            // bug while only the primary's bundles were loaded; now both
            // would write to the same key). Mirrors how precipitation already
            // encodes the per-station axis (precipitation/ea_bellever_dartmoor).
            var composite = $"temperature/{station}";
            var summary = TryLoadSummary(dir, composite, version, metricLabel: "Test MAE (°C)");
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

        // Phase 3f rainfall_amount — per-station versioning under
        // data/models/rainfall_amount/{station}/{version}/. metricLabel
        // is "Test CRPS" because 3f's primary skill metric is mean CRPS
        // (not Brier / MAE — those PerLeadStats fields land null on
        // 3f bundles, per train_3f.py's DeviationsFromBrief note); the
        // verify-history block inside each card pulls the live rolling
        // CRPS from VerifyHistoryRow.BlendMetric (= mean CRPS for
        // rainfall_amount rows) and renders alongside.
        foreach (var (station, version) in rainfallAmount.Select(r => (r.TruthStation, r.ModelVersion)).Distinct())
        {
            if (activeStationSlugs.Count > 0 && !activeStationSlugs.Contains(station)) continue;
            var dir = Path.Combine(modelsRoot, "rainfall_amount", station, version);
            var composite = $"rainfall_amount/{station}";
            var summary = TryLoadSummary(dir, composite, version, metricLabel: "Test CRPS");
            if (summary is not null) summaries.Add(summary);
        }

        // Element blenders — wind speed (ERA5 `wind` + Dunkeswell `wind_speed_lgb`,
        // both under the `wind` dir), gust, direction (mvn), cloud, humidity,
        // radiation. Location-keyed manifests (Stations key = location name), so
        // enumerate Active versions per configured location via ResolveStationActive
        // rather than a prediction list. Composite = "{modelDir}/{location}" so wind's
        // two speed phases surface as champion/challenger cards (like temp 2b/2c).
        // Metric label carries the per-target units. elementLocationFilter scopes the
        // per-location Models page; null (all-locs / Specs) loads every location.
        var elementDirs = new (string Dir, string Metric)[]
        {
            ("wind",                "Test MAE (m/s)"),
            ("wind_gust",           "Test MAE (m/s)"),
            ("wind_direction",      "Test MAE (°)"),
            ("cloud_cover",         "Test MAE (%)"),
            ("humidity",            "Test MAE (%)"),
            ("shortwave_radiation", "Test MAE (W/m²)"),
        };
        foreach (var (dir, metric) in elementDirs)
        {
            if (!Directory.Exists(Path.Combine(modelsRoot, dir))) continue;
            foreach (var loc in _cfg.Locations)
            {
                if (elementLocationFilter is not null && loc.Name != elementLocationFilter) continue;
                IReadOnlyList<string> active;
                try { active = ModelArtifact.ResolveStationActive(modelsRoot, dir, loc.Name); }
                catch { continue; }
                foreach (var version in active)
                {
                    var vdir = Path.Combine(modelsRoot, dir, loc.Name, version);
                    var summary = TryLoadSummary(vdir, $"{dir}/{loc.Name}", version, metric);
                    if (summary is not null) summaries.Add(summary);
                }
            }
        }

        return summaries;
    }

    /// <summary>
    /// Walk every (composite, version) that emitted predictions in the window
    /// and load each version's <c>feature_schema.json</c> via
    /// <see cref="ModelArtifact.LoadBlenderSpecs"/>. Joins each row with the
    /// <see cref="SitePages.ModelSummary"/> already-loaded to pull
    /// <c>Phase</c> + <c>TrainedAtUtc</c> without re-reading
    /// <c>training_metadata.json</c>. Same active-station gating as
    /// <see cref="LoadModelSummaries"/> so a demoted station doesn't appear
    /// on the Spec page either. Schema-load failures are logged and skipped —
    /// a missing schema disqualifies just that one (composite, version) row,
    /// not the whole page.
    /// </summary>
    private IReadOnlyList<SitePages.FeatureSpecRow> LoadFeatureSpecRows(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        IReadOnlyList<SitePages.DryWindowForecastPoint> dryWindow,
        IReadOnlyList<RainfallAmountPredictionRow> rainfallAmount,
        IReadOnlySet<string> activeStationSlugs,
        IReadOnlyList<SitePages.ModelSummary> summaries)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var rows = new List<SitePages.FeatureSpecRow>();
        // Index summaries by (composite, version) so we can carry phase +
        // trained-at without re-parsing training_metadata.json. Versions
        // that emitted predictions but have no metadata on disk are skipped
        // here too — the Spec table needs the phase string to group by.
        var byKey = summaries.ToDictionary(s => (s.Composite, s.Version));

        void Add(string composite, string version)
        {
            if (!byKey.TryGetValue((composite, version), out var summary)) return;
            var dir = composite switch
            {
                // Phase C commit 4 (2026-05-20): temperature composite is now
                // "temperature/{station}" (where station = location slug) so
                // Bonehill + Membury surface as separate Models-page sections.
                // Directory layout is unchanged: {modelsRoot}/temperature/{station}/{version}/.
                _ when composite.StartsWith("temperature/", StringComparison.Ordinal) =>
                    Path.Combine(modelsRoot, "temperature",
                        composite["temperature/".Length..], version),
                _ when composite.StartsWith("precipitation/", StringComparison.Ordinal) =>
                    Path.Combine(modelsRoot, "precipitation",
                        composite["precipitation/".Length..], version),
                _ when composite.StartsWith("dry_window/", StringComparison.Ordinal) =>
                    // dry_window/{station}/{N}h
                    BuildDryWindowDir(modelsRoot, composite, version),
                _ when composite.StartsWith("rainfall_amount/", StringComparison.Ordinal) =>
                    Path.Combine(modelsRoot, "rainfall_amount",
                        composite["rainfall_amount/".Length..], version),
                _ => null,
            };
            if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;
            try
            {
                var specs = ModelArtifact.LoadBlenderSpecs(dir);
                foreach (var (lead, spec) in specs)
                {
                    rows.Add(new SitePages.FeatureSpecRow(
                        Composite: composite,
                        Phase: summary.Phase,
                        Version: version,
                        TrainedAtUtc: summary.TrainedAtUtc,
                        LeadHours: lead,
                        FeatureSet: spec.FeatureSet,
                        RequiredModels: spec.RequiredModels.ToArray(),
                        OptionalModels: spec.OptionalModels.ToArray(),
                        FeatureNames: spec.FeatureNames.ToArray(),
                        // Structured-spec fields from the BlenderSpec (Phase 1
                        // shipped 2026-05-07). Empty strings / null on legacy
                        // schemas predating the migration; the renderer's
                        // InterpretFeatureSet falls back to FeatureNames-based
                        // inference in that case so old + new schemas render
                        // identically.
                        DataSource: spec.DataSource,
                        Tier: spec.Tier,
                        UkvStrategy: spec.UkvStrategy?.ToString()));
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning("Skipping feature spec for {Composite}/{Version}: {Msg}",
                    composite, version, ex.Message);
            }
        }

        // Temperature is per-station/per-location on disk; the composite must
        // carry the station segment (Phase C commit 4, 2026-05-20) so it
        // matches the key LoadModelSummaries writes into `byKey` and the
        // directory switch in Add(). Calling Add("temperature", ..) with the
        // bare composite missed both — the byKey lookup and the switch's
        // StartsWith("temperature/") case — so no temperature rows reached
        // the Spec page at all. Mirror the precip block immediately below.
        foreach (var (station, version) in predictions
            .Select(p => (p.LocationName, p.ModelVersion))
            .Distinct())
            Add($"temperature/{station}", version);

        foreach (var (station, version) in precip.Select(p => (p.Station, p.Version)).Distinct())
        {
            if (activeStationSlugs.Count > 0 && !activeStationSlugs.Contains(station)) continue;
            Add($"precipitation/{station}", version);
        }

        foreach (var (station, window, version) in dryWindow.Select(d => (d.Station, d.WindowHours, d.Version)).Distinct())
        {
            if (activeStationSlugs.Count > 0 && !activeStationSlugs.Contains(station)) continue;
            Add($"dry_window/{station}/{window}h", version);
        }

        foreach (var (station, version) in rainfallAmount.Select(r => (r.TruthStation, r.ModelVersion)).Distinct())
        {
            if (activeStationSlugs.Count > 0 && !activeStationSlugs.Contains(station)) continue;
            Add($"rainfall_amount/{station}", version);
        }

        return rows;
    }

    private static string BuildDryWindowDir(string modelsRoot, string composite, string version)
    {
        // "dry_window/{station}/{N}h" → modelsRoot/dry_window/{station}/window_{N}h/{version}
        var parts = composite.Split('/');
        if (parts.Length != 3) return string.Empty;
        var station = parts[1];
        var window = parts[2]; // e.g. "6h"
        return Path.Combine(modelsRoot, "dry_window", station, $"window_{window}", version);
    }

    /// <summary>
    /// Find a temperature version's on-disk directory by scanning each
    /// station entry under <c>{modelsRoot}/temperature/</c>. Returns the
    /// first match — version-dir names are timestamped and effectively
    /// unique, but if two locations ever shared a version name (e.g. the
    /// retrain workflow ran at the same UTC second), the first match still
    /// loads the right BlenderSpec since both bundles would have been
    /// trained against the same blender config. Returns null when no
    /// matching dir exists (the version was deleted or predates the
    /// per-station migration).
    /// </summary>

    /// <summary>
    /// Snapshot of the location's active <c>wave_height</c> bundle for the
    /// Sea-state Models page: <c>training_metadata.json</c> (version /
    /// trained-at / row counts / window / truth source — the wave bundle is
    /// any-lead so it has no PerLead dict and can't go through
    /// <see cref="TryLoadSummary"/>) merged with <c>calibration.json</c>
    /// (cal-slice MAE + band coverage, Python snake_case). Both parsed
    /// tolerantly via JsonDocument; null when the location has no wave
    /// manifest entry or the metadata file is missing/corrupt. A missing
    /// calibration.json degrades to a card without the cal-slice rows.
    /// </summary>
    private SitePages.WaveModelCard? LoadWaveModelCard(string locationName)
    {
        try
        {
            IReadOnlyList<string> active;
            try { active = ModelArtifact.ResolveStationActive(_cfg.Storage.ModelsPath, "wave_height", locationName); }
            catch { return null; }
            if (active.Count == 0) return null;
            // Newest active version (vYYYY-MM-DD_HHmmss prefix → ordinal max).
            var version = active.OrderByDescending(v => v, StringComparer.Ordinal).First();
            var dir = Path.Combine(_cfg.Storage.ModelsPath, "wave_height", locationName, version);
            var metaPath = Path.Combine(dir, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(metaPath)) return null;

            using var meta = System.Text.Json.JsonDocument.Parse(File.ReadAllText(metaPath));
            var m = meta.RootElement;
            static string MStr(System.Text.Json.JsonElement root, string name) =>
                root.TryGetProperty(name, out var el) && el.ValueKind == System.Text.Json.JsonValueKind.String
                    ? el.GetString() ?? "" : "";
            static int MInt(System.Text.Json.JsonElement root, string name) =>
                root.TryGetProperty(name, out var el) && el.ValueKind == System.Text.Json.JsonValueKind.Number
                    ? el.GetInt32() : 0;
            var trainedAt = m.TryGetProperty("TrainedAtUtc", out var ta) && ta.TryGetDateTime(out var dt)
                ? dt.ToUniversalTime() : DateTime.MinValue;

            double? calMae = null, calCoverage = null, nominal = null;
            string? leadMode = null, note = null;
            var calPath = Path.Combine(dir, "calibration.json");
            if (File.Exists(calPath))
            {
                try
                {
                    using var cal = System.Text.Json.JsonDocument.Parse(File.ReadAllText(calPath));
                    var c = cal.RootElement;
                    static double? CNum(System.Text.Json.JsonElement root, string name) =>
                        root.TryGetProperty(name, out var el) && el.ValueKind == System.Text.Json.JsonValueKind.Number
                            ? el.GetDouble() : null;
                    static string? CStr(System.Text.Json.JsonElement root, string name) =>
                        root.TryGetProperty(name, out var el) && el.ValueKind == System.Text.Json.JsonValueKind.String
                            ? el.GetString() : null;
                    calMae = CNum(c, "cal_mae");
                    calCoverage = CNum(c, "cal_band_coverage");
                    nominal = CNum(c, "coverage");
                    leadMode = CStr(c, "lead_mode");
                    note = CStr(c, "note");
                }
                catch (Exception ex)
                {
                    _log.LogWarning("Failed to parse wave calibration.json for {Loc}/{Version}: {Msg}",
                        locationName, version, ex.Message);
                }
            }

            return new SitePages.WaveModelCard(
                Version: version,
                Phase: MStr(m, "Phase"),
                TrainedAtUtc: trainedAt,
                RowsTrain: MInt(m, "RowsTrain"),
                RowsVal: MInt(m, "RowsVal"),
                RowsCal: MInt(m, "RowsCal"),
                WindowStart: MStr(m, "WindowStart"),
                WindowEnd: MStr(m, "WindowEnd"),
                FeatureCount: MInt(m, "FeatureCount"),
                TruthSource: MStr(m, "TruthSource"),
                CalMae: calMae,
                CalBandCoverage: calCoverage,
                NominalCoverage: nominal,
                LeadMode: leadMode,
                CalNote: note);
        }
        catch (Exception ex)
        {
            _log.LogWarning("Failed to load wave model card for {Loc}: {Msg}", locationName, ex.Message);
            return null;
        }
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
                        // 4a (Python-trained) emits JSON null here — surface
                        // as NaN. Models page's bestRef computation already
                        // treats NaN as "missing" and renders the em-dash
                        // branch (SitePages.Models.cs:216-218).
                        BestSingleValMae:   kv.Value.BestSingleValMae  ?? double.NaN,
                        BestSingleTestMae:  kv.Value.BestSingleTestMae ?? double.NaN,
                        // Same null → NaN coercion as BestSingleVal/TestMae
                        // — 3f's distributional bundles write JSON null for
                        // these (mean CRPS is the headline; point MAE has
                        // no meaning on a mixed mass-at-zero distribution).
                        // Renderers test double.IsNaN and render "—" for
                        // the missing branch.
                        BlendTestScore:     kv.Value.BlendTestMae  ?? double.NaN,
                        BlendTestRmse:      kv.Value.BlendTestRmse ?? double.NaN,
                        BlendTestBias:      kv.Value.BlendTestBias ?? double.NaN,
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
    /// Resolve rainfall truth for the stations that actually produced
    /// precip predictions in the window — EA gauges and WeatherLink cove
    /// gauges (e.g. Lands End) alike. Pure delegate to
    /// <see cref="TruthRepository.GetHourlyRainfallByStation"/>; this stub
    /// only exists because we want the per-station list to be derived from
    /// the predictions (so empty windows skip the I/O entirely).
    /// </summary>
    private IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> LoadRainfallTruth(
        DateTime start, DateTime end,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        CancellationToken ct)
    {
        var stations = precip.Select(p => p.Station).Distinct().ToList();
        return _truth.GetHourlyRainfallByStation(stations, start, end, ct);
    }

    /// <summary>
    /// Prediction-derived (station, version) keys unioned with the CURRENT
    /// champion versions for the phase map. The champion that today's retrain
    /// minted has no predictions until the next predict cycle runs (the
    /// retrain→predict race), so it's absent from <paramref name="predictionKeys"/>.
    /// If its phase never enters the phase map, <see cref="ChampionMatcher"/>'s
    /// phase fallback — which needs the champion's phase as well as the row's —
    /// can't match the PREVIOUS champion's still-valid rows and the overview
    /// blanks (the recurring "no tiles after a Sunday retrain" bug; 2026-05-26
    /// then again 2026-06-14 for Membury). Adding the champion keys lets
    /// <see cref="ModelMetadataRepository.GetPhaseByVersion"/> resolve their
    /// phase off the on-disk bundle, which is synced even with zero predictions.
    /// Duplicates are harmless — GetPhaseByVersion de-dupes.
    /// </summary>
    internal static IEnumerable<(string Station, string Version)> WithChampionVersions(
        IEnumerable<(string Station, string Version)> predictionKeys,
        IReadOnlyDictionary<string, string> championByStation)
        => predictionKeys
            .Concat(championByStation.Select(kv => (Station: kv.Key, Version: kv.Value)));

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

    // ------------------------------------------------------------------
    // Met Office Spot comparison-line scoring (2026-06-12). Spot is a
    // reference product on the skill charts, NEVER a blender input or a
    // champion/challenger — so its scoring lives here next to the resident
    // rolling metrics (same truth, same windows, same cadence) and its
    // output rides dedicated SiteInputs fields rather than joining the
    // phase-keyed lists.
    // ------------------------------------------------------------------

    /// <summary>Extra days of Spot history loaded beyond the render window
    /// so the wind chart's per-sidecar window replay (asOf − windowDays −
    /// latency, reaching ~19d past the 30-day axis edge) has rows to score.
    /// See the QueryMetOfficeSpotForecasts call site.</summary>
    internal const int SpotVerifyLookbackExtraDays = 21;

    /// <summary>Phase tag stamped on Spot comparison points. Display-only —
    /// the renderers label these series "Met Office Spot" and no
    /// phase-policy lookup ever sees the tag.</summary>
    internal const string SpotPhaseTag = "met_office_spot";

    /// <summary>
    /// Lead bucket for the Spot comparison series on a per-lead rolling
    /// chart: panel lead L covers Spot rows with LeadHours ∈ [L,
    /// nextPanelLead) (the final panel takes [L, L+24)). Mirrors the
    /// eyeball charts' "+24h lead bucket [24, 48)" convention and the
    /// trained-lead bucket semantics of the resident series (a lead-24
    /// prediction row's actual lead spreads 24..47h across the day's
    /// hourly rows; Spot's own lead is synthesised per pull, so an exact
    /// match would be dishonest precision). Each Spot forecast hour lands
    /// in exactly one panel.
    /// </summary>
    internal static (int Lo, int Hi) SpotLeadBucket(IReadOnlyList<int> panelLeads, int index)
        => (panelLeads[index],
            index + 1 < panelLeads.Count ? panelLeads[index + 1] : panelLeads[index] + 24);

    /// <summary>
    /// Rolling MAE for the Met Office Spot temperature forecast — the
    /// comparison line on the temp rolling-MAE chart. Pairs Spot rows
    /// (deduped to the smallest lead per valid hour inside each panel's
    /// bucket) with the SAME ERA5 truth dict the resident
    /// <see cref="ComputeRollingMae"/> uses, then emits one point per
    /// calendar day over the same backwards-sliding <paramref name="windowDays"/>
    /// window. Leads with no Spot coverage emit nothing — the chart omits
    /// the line there rather than faking it.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingMaePoint> ComputeSpotRollingMae(
        IReadOnlyList<SitePages.MetOfficeSpotForecastPoint> spot,
        IReadOnlyDictionary<DateTime, double> truthByTime,
        IReadOnlyList<int> panelLeads,
        int windowDays)
    {
        var points = new List<SitePages.RollingMaePoint>();
        for (int i = 0; i < panelLeads.Count; i++)
        {
            var (lo, hi) = SpotLeadBucket(panelLeads, i);
            var paired = spot
                .Where(m => m.Temperature2m.HasValue
                            && m.LeadHours >= lo && m.LeadHours < hi
                            && truthByTime.ContainsKey(m.ValidTimeUtc))
                .LatestPerValid(m => m.ValidTimeUtc, m => m.LeadHours)
                .Select(m => (m.ValidTimeUtc,
                              Err: Math.Abs(m.Temperature2m!.Value - truthByTime[m.ValidTimeUtc])))
                .ToList();
            if (paired.Count == 0) continue;

            var minDate = paired.Min(r => r.ValidTimeUtc).Date;
            var maxDate = paired.Max(r => r.ValidTimeUtc).Date;
            for (var d = minDate; d <= maxDate; d = d.AddDays(1))
            {
                var windowEnd = d.AddDays(1).AddTicks(-1);
                var windowStart = windowEnd.AddDays(-windowDays);
                var inWindow = paired.Where(r => r.ValidTimeUtc >= windowStart && r.ValidTimeUtc <= windowEnd).ToList();
                if (inWindow.Count == 0) continue;
                points.Add(new SitePages.RollingMaePoint(
                    SpotPhaseTag, panelLeads[i], windowEnd, inWindow.Average(r => r.Err), inWindow.Count));
            }
        }
        return points;
    }

    /// <summary>
    /// Rolling Brier for the Met Office Spot PoP — comparison line on the
    /// rain skill chart. PoP arrives in percent [0, 100] → /100, scored
    /// against the SAME per-station EA wet/dry labels (≥ 0.1 mm/h) and
    /// the same 30-day rolling windows the resident
    /// <see cref="ComputeRollingBrier"/> series use. NB the Met Office PoP
    /// threshold is "any measurable precip" — slightly looser than the
    /// 0.1 mm/h label; the chart copy already flags this on the eyeball
    /// panel and the same caveat applies here.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingBrierPoint> ComputeSpotRollingBrier(
        IReadOnlyList<SitePages.MetOfficeSpotForecastPoint> spot,
        IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> rainfallByStation,
        IReadOnlyList<int> panelLeads,
        int windowDays)
    {
        const double WetThresholdMm = 0.1;
        var points = new List<SitePages.RollingBrierPoint>();
        foreach (var (station, byTime) in rainfallByStation)
        {
            for (int i = 0; i < panelLeads.Count; i++)
            {
                var (lo, hi) = SpotLeadBucket(panelLeads, i);
                var paired = spot
                    .Where(m => m.PrecipitationProbabilityPercent.HasValue
                                && m.LeadHours >= lo && m.LeadHours < hi
                                && byTime.ContainsKey(m.ValidTimeUtc))
                    .LatestPerValid(m => m.ValidTimeUtc, m => m.LeadHours)
                    .Select(m =>
                    {
                        var p = m.PrecipitationProbabilityPercent!.Value / 100.0;
                        var t = byTime[m.ValidTimeUtc] >= WetThresholdMm ? 1.0 : 0.0;
                        return (m.ValidTimeUtc, Sq: (p - t) * (p - t));
                    })
                    .ToList();
                if (paired.Count == 0) continue;

                var minDate = paired.Min(r => r.ValidTimeUtc).Date;
                var maxDate = paired.Max(r => r.ValidTimeUtc).Date;
                for (var d = minDate; d <= maxDate; d = d.AddDays(1))
                {
                    var windowEnd = d.AddDays(1).AddTicks(-1);
                    var windowStart = windowEnd.AddDays(-windowDays);
                    var inWindow = paired.Where(r => r.ValidTimeUtc >= windowStart && r.ValidTimeUtc <= windowEnd).ToList();
                    if (inWindow.Count == 0) continue;
                    points.Add(new SitePages.RollingBrierPoint(
                        station, SpotPhaseTag, panelLeads[i], windowEnd, inWindow.Average(r => r.Sq), inWindow.Count));
                }
            }
        }
        return points;
    }

    /// <summary>
    /// Met Office Spot wind-speed MAE replayed over each element_wind
    /// verify sidecar's exact (asOf, windowDays, latencyDays) scoring
    /// window, against the same ERA5 WindSpeed10m truth element-verify
    /// scores the blenders on. The wind skill chart's resident points come
    /// from those append-only weekly sidecars, so recomputing Spot here per
    /// sidecar window gives it the same X positions (WindowEndUtc = AsOfUtc),
    /// the same cadence, AND full history — without touching the verify
    /// command or its drift logic. Sidecar windows that pre-date the loaded
    /// Spot capture (started 2026-04-24) score on whatever rows exist; N on
    /// the point carries the pair count as everywhere else.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingMaePoint> ComputeSpotWindVerifyMae(
        IReadOnlyList<SitePages.MetOfficeSpotForecastPoint> spot,
        IReadOnlyList<(DateTime AsOfUtc, int WindowDays, int LatencyDays)> verifyWindows,
        IReadOnlyDictionary<DateTime, double> windTruthByTime,
        IReadOnlyList<int> panelLeads)
    {
        var points = new List<SitePages.RollingMaePoint>();
        foreach (var (asOf, windowDays, latencyDays) in verifyWindows.OrderBy(w => w.AsOfUtc))
        {
            // Same window arithmetic as ElementVerifyCommand.RunAsync.
            var windowStart = asOf.AddDays(-windowDays);
            var windowEnd = asOf.AddDays(-latencyDays);
            for (int i = 0; i < panelLeads.Count; i++)
            {
                var (lo, hi) = SpotLeadBucket(panelLeads, i);
                var errs = spot
                    .Where(m => m.WindSpeed10mMs.HasValue
                                && m.LeadHours >= lo && m.LeadHours < hi
                                && m.ValidTimeUtc >= windowStart && m.ValidTimeUtc <= windowEnd
                                && windTruthByTime.ContainsKey(m.ValidTimeUtc))
                    .LatestPerValid(m => m.ValidTimeUtc, m => m.LeadHours)
                    .Select(m => Math.Abs(m.WindSpeed10mMs!.Value - windTruthByTime[m.ValidTimeUtc]))
                    .ToList();
                if (errs.Count == 0) continue;
                points.Add(new SitePages.RollingMaePoint(
                    SpotPhaseTag, panelLeads[i], asOf, errs.Average(), errs.Count));
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
        // Phase C commit 3: multi-loc IN-list + LocationName projection;
        // home renderer filters by active location at consume time.
        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        var sql = $@"
SELECT ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours, UtciC, Band,
       ApparentTemperatureC,
       TemperatureC, RelativeHumidityPct, WindSpeed10mMs,
       ShortwaveDownWm2, CloudCoverPct, LocationName
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName IN ({locFilter})
  AND UtciC IS NOT NULL
  AND ApparentTemperatureC IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY LocationName, LeadHours, ValidTimeUtc";

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
            CloudCoverPct:       r.IsDBNull(11) ? null : (double?)r.GetDouble(11),
            LocationName:        r.GetString(12)),
            _log, "Feels-like predictions tree empty — chip will be absent on home cards.", ct);
    }

    /// <summary>
    /// Pulls rock surface temperature + condensation predictions (Phase P1) for
    /// every configured location in the window. Flat tree (LocationName in-column);
    /// the per-loc render filters at consume time. Empty when the rock_surface
    /// tree hasn't been synced / predict hasn't run — the overview chip + temp-tab
    /// chart fall back silently.
    /// </summary>
    private IReadOnlyList<SitePages.RockSurfaceForecastPoint> QueryRockSurfacePredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
                                                   WeatherBlend.Commands.RockSurfacePredictCommand.PredictionsSubdir,
                                                   "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "GreasinessStatus"))
        {
            _log.LogWarning("Rock-surface predictions tree empty — chip + chart absent until predict runs once.");
            return Array.Empty<SitePages.RockSurfaceForecastPoint>();
        }

        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        // Face arrived with cliff-face mode (SENNEN_ROCK_TEMP_PLAN.md S3) — a
        // tree of purely pre-face files has no such column, so guard the SELECT.
        var faceCol = ParquetReader.HasColumn(conn, glob, "Face") ? "COALESCE(Face, '')" : "''";
        // Surface-water columns arrived with the Phase A drying model — a tree
        // predicted before then has neither, so guard each and read 0 (bone-dry).
        var surfaceWaterCol = ParquetReader.HasColumn(conn, glob, "SurfaceWaterMm") ? "COALESCE(SurfaceWaterMm, 0.0)" : "0.0";
        var rainWaterCol    = ParquetReader.HasColumn(conn, glob, "RainWaterMm")    ? "COALESCE(RainWaterMm, 0.0)"    : "0.0";
        // LastRainAtUtc arrived with the "wet from rain since HHZ" display — guard
        // it too; a pre-column tree yields NULL (no wet-event timestamp shown).
        var lastRainCol     = ParquetReader.HasColumn(conn, glob, "LastRainAtUtc")  ? "LastRainAtUtc" : "CAST(NULL AS TIMESTAMP)";
        // ForcingTier arrived with the near-term emit — guard it; a pre-column tree
        // yields '' (treated as blend, no "<24h" marker).
        var forcingTierCol  = ParquetReader.HasColumn(conn, glob, "ForcingTier")    ? "COALESCE(ForcingTier, '')" : "''";
        // AmbientRhPct arrived with the sea-cliff salt greasiness — guard it; a
        // pre-column tree yields NaN (salt factor skipped).
        var ambientRhCol    = ParquetReader.HasColumn(conn, glob, "AmbientRhPct")   ? "AmbientRhPct" : "CAST('NaN' AS DOUBLE)";
        var sql = $@"
SELECT ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       RockSurfaceTempC, AirTempC, DewPointC, CondensationMarginC, GreasinessStatus, LocationName,
       {faceCol} AS Face,
       {surfaceWaterCol} AS SurfaceWaterMm,
       {rainWaterCol} AS RainWaterMm,
       {lastRainCol} AS LastRainAtUtc,
       {forcingTierCol} AS ForcingTier,
       {ambientRhCol} AS AmbientRhPct
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName IN ({locFilter})
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY LocationName, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(conn, sql, r => new SitePages.RockSurfaceForecastPoint(
            Version:             r.GetString(0),
            PredictedAtUtc:      r.GetDateTime(1),
            ValidTimeUtc:        r.GetDateTime(2),
            LeadHours:           r.GetInt32(3),
            RockSurfaceTempC:    r.GetDouble(4),
            AirTempC:            r.GetDouble(5),
            DewPointC:           r.GetDouble(6),
            CondensationMarginC: r.GetDouble(7),
            GreasinessStatus:    r.IsDBNull(8) ? "" : r.GetString(8),
            SurfaceWaterMm:      r.IsDBNull(11) ? 0.0 : r.GetDouble(11),
            RainWaterMm:         r.IsDBNull(12) ? 0.0 : r.GetDouble(12),
            LastRainAtUtc:       r.IsDBNull(13) ? (DateTime?)null : r.GetDateTime(13),
            LocationName:        r.GetString(9),
            Face:                r.IsDBNull(10) ? "" : r.GetString(10),
            ForcingTier:         r.IsDBNull(14) ? "" : r.GetString(14),
            AmbientRhPct:        r.IsDBNull(15) ? double.NaN : r.GetDouble(15)),
            _log, "Rock-surface predictions tree empty — chip + chart absent.", ct);
    }

    /// <summary>
    /// Pulls wind-gust blender predictions (m/s) for every configured
    /// location in the rendering window, collapsed to one row per
    /// (LocationName, ValidTimeUtc) at the latest <c>PredictionMadeAtUtc</c>.
    /// Surfaced inline on the UTCI pop-out next to the 10 m wind row when
    /// gust is materially above wind (gust mph &gt; wind mph + 1).
    /// </summary>
    private IReadOnlyList<(string LocationName, DateTime ValidTimeUtc, double GustMs)>
        QueryWindGustByValidTime(DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
                                                   "wind_gust", "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "BlendValue"))
        {
            _log.LogInformation("Wind-gust predictions tree empty — pop-out gust suffix absent until predict runs.");
            return Array.Empty<(string, DateTime, double)>();
        }
        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        var sql = $@"
WITH ranked AS (
    SELECT LocationName, ValidTimeUtc, BlendValue,
           ROW_NUMBER() OVER (PARTITION BY LocationName, ValidTimeUtc
                              ORDER BY PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE Element = 'wind-gust'
      AND LocationName IN ({locFilter})
      AND BlendValue IS NOT NULL
      AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT LocationName, ValidTimeUtc, BlendValue
FROM ranked
WHERE rn = 1
ORDER BY LocationName, ValidTimeUtc";
        return ParquetReader.Query(conn, sql, r => (
            LocationName: r.GetString(0),
            ValidTimeUtc: r.GetDateTime(1),
            GustMs:       r.GetDouble(2)),
            _log, "Wind-gust predictions tree empty — pop-out gust suffix absent until predict runs.", ct);
    }

    /// <summary>Pulls every wind <see cref="WeatherBlend.Models.ElementPredictionRow"/>
    /// in the rendering window (champion <c>wind</c> phase + challenger
    /// <c>wind_speed_lgb</c> / <c>wind_blend</c> phases). One row per
    /// (Version, ValidTime), collapsed to freshest <c>PredictionMadeAtUtc</c>
    /// by SQL window function. Per-NWP columns are carried through nullable
    /// so the page can render the per-NWP overlay without a second read.</summary>
    private IReadOnlyList<SitePages.WindForecastPoint> QueryWindPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
                                                   "wind", "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "BlendValue"))
        {
            _log.LogInformation("Wind predictions tree empty — per-lead wind page will empty-state until predict runs.");
            return Array.Empty<SitePages.WindForecastPoint>();
        }
        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        // CQR band sidecar (BandLoMs/BandHiMs) — written only by the Python
        // wind_speed_lgb predictor (2026-06-10 cutover). union_by_name fills
        // NULLs on the champion/blend parquets that lack the columns, but a
        // tree with NO band-carrying file at all (e.g. before the first
        // Python predict cycle) would fail the SELECT — so probe first and
        // select typed NULLs in that case. Never break the chart for a
        // missing band: the page falls back to the plain point line.
        var hasBand = ParquetReader.HasColumn(conn, glob, "BandLoMs");
        var bandCols = hasBand
            ? "BandLoMs, BandHiMs"
            : "CAST(NULL AS DOUBLE) AS BandLoMs, CAST(NULL AS DOUBLE) AS BandHiMs";
        var sql = $@"
WITH ranked AS (
    SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
           BlendValue, ModelGfs, ModelEcmwf, ModelIcon, ModelMf, ModelUkmo, ModelGem, ModelAifs,
           {bandCols},
           ROW_NUMBER() OVER (PARTITION BY LocationName, ModelVersion, LeadHours, ValidTimeUtc
                              ORDER BY PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE Element = 'wind'
      AND LocationName IN ({locFilter})
      AND BlendValue IS NOT NULL
      AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendValue, ModelGfs, ModelEcmwf, ModelIcon, ModelMf, ModelUkmo, ModelGem, ModelAifs,
       BandLoMs, BandHiMs
FROM ranked
WHERE rn = 1
ORDER BY LocationName, ModelVersion, LeadHours, ValidTimeUtc";
        return ParquetReader.Query(conn, sql, r => new SitePages.WindForecastPoint(
            Version:        r.GetString(1),
            PredictedAtUtc: r.GetDateTime(2),
            ValidTimeUtc:   r.GetDateTime(3),
            LeadHours:      r.GetInt32(4),
            SpeedMs:        r.GetDouble(5),
            Gfs:            r.IsDBNull(6)  ? null : (double?)r.GetDouble(6),
            Ecmwf:          r.IsDBNull(7)  ? null : (double?)r.GetDouble(7),
            Icon:           r.IsDBNull(8)  ? null : (double?)r.GetDouble(8),
            Mf:             r.IsDBNull(9)  ? null : (double?)r.GetDouble(9),
            Ukmo:           r.IsDBNull(10) ? null : (double?)r.GetDouble(10),
            Gem:            r.IsDBNull(11) ? null : (double?)r.GetDouble(11),
            Aifs:           r.IsDBNull(12) ? null : (double?)r.GetDouble(12),
            LocationName:   r.GetString(0),
            BandLoMs:       r.IsDBNull(13) ? null : (double?)r.GetDouble(13),
            BandHiMs:       r.IsDBNull(14) ? null : (double?)r.GetDouble(14)),
            _log, "Wind predictions tree empty.", ct);
    }

    /// <summary>Pulls every <c>wind_mvn</c> direction prediction row in the
    /// rendering window. WP's predict-wind.yml writes them to
    /// <c>data/predictions/wind_direction/{location}/model_version=*_wind_mvn/date=*/predictions.parquet</c>
    /// — note the location segment in the path differs from the WB element
    /// predictions tree.</summary>
    private IReadOnlyList<SitePages.WindDirectionForecastPoint> QueryWindDirectionPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
                                                   "wind_direction", "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "BlendDirection"))
        {
            _log.LogInformation("wind_mvn (wind_direction) predictions tree empty — direction circles + speed CI ribbon will empty-state until predict-wind runs.");
            return Array.Empty<SitePages.WindDirectionForecastPoint>();
        }
        // CI80 columns added 2026-05-28; parquets predating that revision
        // don't carry them. Probe before SELECTing so the query stays
        // backwards-compatible — older parquets return null and the
        // renderer falls back to CI95.
        var hasCi80 = ParquetReader.HasColumn(conn, glob, "BlendSpeedCi80Lo");
        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        var ci80Cols = hasCi80
            ? ", BlendDirectionCi80Lo, BlendDirectionCi80Hi, BlendSpeedCi80Lo, BlendSpeedCi80Hi"
            : ", NULL::DOUBLE AS BlendDirectionCi80Lo, NULL::DOUBLE AS BlendDirectionCi80Hi, NULL::DOUBLE AS BlendSpeedCi80Lo, NULL::DOUBLE AS BlendSpeedCi80Hi";
        var sql = $@"
WITH ranked AS (
    SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
           BlendDirection, BlendDirectionCi95Lo, BlendDirectionCi95Hi,
           BlendSpeedMagnitude, BlendSpeedCi95Lo, BlendSpeedCi95Hi
           {ci80Cols},
           ROW_NUMBER() OVER (PARTITION BY LocationName, ModelVersion, LeadHours, ValidTimeUtc
                              ORDER BY PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName IN ({locFilter})
      AND BlendDirection IS NOT NULL
      AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendDirection, BlendDirectionCi95Lo, BlendDirectionCi95Hi,
       BlendSpeedMagnitude, BlendSpeedCi95Lo, BlendSpeedCi95Hi,
       BlendDirectionCi80Lo, BlendDirectionCi80Hi, BlendSpeedCi80Lo, BlendSpeedCi80Hi
FROM ranked
WHERE rn = 1
ORDER BY LocationName, LeadHours, ValidTimeUtc";
        return ParquetReader.Query(conn, sql, r => new SitePages.WindDirectionForecastPoint(
            Version:             r.GetString(1),
            PredictedAtUtc:      r.GetDateTime(2),
            ValidTimeUtc:        r.GetDateTime(3),
            LeadHours:           r.GetInt32(4),
            DirectionDeg:        r.GetDouble(5),
            DirectionCi95LoDeg:  r.GetDouble(6),
            DirectionCi95HiDeg:  r.GetDouble(7),
            SpeedMs:             r.GetDouble(8),
            SpeedCi95LoMs:       r.GetDouble(9),
            SpeedCi95HiMs:       r.GetDouble(10),
            LocationName:        r.GetString(0),
            DirectionCi80LoDeg:  r.IsDBNull(11) ? null : (double?)r.GetDouble(11),
            DirectionCi80HiDeg:  r.IsDBNull(12) ? null : (double?)r.GetDouble(12),
            SpeedCi80LoMs:       r.IsDBNull(13) ? null : (double?)r.GetDouble(13),
            SpeedCi80HiMs:       r.IsDBNull(14) ? null : (double?)r.GetDouble(14)),
            _log, "wind_mvn predictions tree empty.", ct);
    }

    /// <summary>Pulls every <c>wave_height</c> prediction row in the rendering
    /// window for the Sea state page. WP's predict-wave workflow writes
    /// <c>data/predictions/wave_height/model_version=*_wave_height_lgb/date=*/predictions.parquet</c>
    /// — one continuous hourly forecast per cycle (LeadHours 0–146), so unlike
    /// the per-lead element queries the collapse here is to ONE row per
    /// (LocationName, ValidTimeUtc): newest ModelVersion first (version strings
    /// are v{timestamp}, so ordinal DESC = newest — the rule that keeps a
    /// second ModelVersion straddling the Sunday retrain from double-plotting),
    /// then freshest PredictionMadeAtUtc. Pass-through marine columns ride
    /// along so the page needs no second read.</summary>
    private IReadOnlyList<SitePages.WaveForecastPoint> QueryWavePredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
                                                   "wave_height", "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "BlendValue"))
        {
            _log.LogInformation("wave_height predictions tree empty — Sea state page will empty-state until predict-wave runs.");
            return Array.Empty<SitePages.WaveForecastPoint>();
        }
        // The Python writer stamps PredictionMadeAtUtc as TIMESTAMPTZ (the
        // wave tree's only timezone-aware column). Pin the session timezone
        // to UTC so the CAST below yields the UTC wall time on any runner —
        // without it DuckDB casts into the host's local zone and the made-at
        // would drift an hour during BST (the same class of bug the removed
        // Met Office Python writer caused in the forecast tree).
        using (var tz = conn.CreateCommand())
        {
            tz.CommandText = "SET TimeZone = 'UTC'";
            tz.ExecuteNonQuery();
        }
        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        var sql = $@"
WITH ranked AS (
    SELECT LocationName, ModelVersion,
           CAST(PredictionMadeAtUtc AS TIMESTAMP) AS PredictionMadeAtUtc,
           ValidTimeUtc, CAST(LeadHours AS INTEGER) AS LeadHours,
           BlendValue, BandLoM, BandHiM,
           WavePeriodS, WaveDirectionDeg,
           SwellHeightM, SwellPeriodS, SwellDirectionDeg,
           SecondarySwellHeightM, SecondarySwellPeriodS,
           TideHeightMsl, SeaSurfaceTempC,
           ROW_NUMBER() OVER (PARTITION BY LocationName, ValidTimeUtc
                              ORDER BY ModelVersion DESC, PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE Element = 'wave_height'
      AND LocationName IN ({locFilter})
      AND BlendValue IS NOT NULL
      AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendValue, BandLoM, BandHiM,
       WavePeriodS, WaveDirectionDeg,
       SwellHeightM, SwellPeriodS, SwellDirectionDeg,
       SecondarySwellHeightM, SecondarySwellPeriodS,
       TideHeightMsl, SeaSurfaceTempC
FROM ranked
WHERE rn = 1
ORDER BY LocationName, ValidTimeUtc";
        return ParquetReader.Query(conn, sql, r => new SitePages.WaveForecastPoint(
            Version:               r.GetString(1),
            PredictedAtUtc:        r.GetDateTime(2),
            ValidTimeUtc:          r.GetDateTime(3),
            LeadHours:             r.GetInt32(4),
            WaveHeightM:           r.GetDouble(5),
            BandLoM:               r.IsDBNull(6)  ? null : (double?)r.GetDouble(6),
            BandHiM:               r.IsDBNull(7)  ? null : (double?)r.GetDouble(7),
            WavePeriodS:           r.IsDBNull(8)  ? null : (double?)r.GetDouble(8),
            WaveDirectionDeg:      r.IsDBNull(9)  ? null : (double?)r.GetDouble(9),
            SwellHeightM:          r.IsDBNull(10) ? null : (double?)r.GetDouble(10),
            SwellPeriodS:          r.IsDBNull(11) ? null : (double?)r.GetDouble(11),
            SwellDirectionDeg:     r.IsDBNull(12) ? null : (double?)r.GetDouble(12),
            SecondarySwellHeightM: r.IsDBNull(13) ? null : (double?)r.GetDouble(13),
            SecondarySwellPeriodS: r.IsDBNull(14) ? null : (double?)r.GetDouble(14),
            TideHeightMsl:         r.IsDBNull(15) ? null : (double?)r.GetDouble(15),
            SeaSurfaceTempC:       r.IsDBNull(16) ? null : (double?)r.GetDouble(16),
            LocationName:          r.GetString(0)),
            _log, "wave_height predictions tree empty.", ct);
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
    /// have T-Td &lt; 1.5°C (Espy proxy for near-surface saturation → cloud
    /// base at/below the location's own grid cell; each location is evaluated
    /// at its own configured elevation, NOT a fixed altitude).
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
                    // Same versionName can be shared across (station, window)
                    // composites — but conformal artefacts live under each
                    // composite's version dir so the τ values genuinely differ
                    // per (station, window). We key only on (version, lead)
                    // here because the dry-window page already resolved
                    // (station, window) before reading this dict. If two
                    // composites collide on (version, lead) we keep the first;
                    // in practice version names are suffixed per composite so
                    // suffixes only appear once per composite.
                    dict.TryAdd((versionName, lead), cal.Tau);
                }
            }
        }
        return dict;
    }

    private IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, SitePages.LowCloudSignal>>
        QueryLowCloudSignals(DateTime start, DateTime end, CancellationToken ct)
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

        // Phase C commit 3: multi-loc IN-list + LocationName in GROUP BY;
        // returns Dict<location, Dict<time, signal>> so the renderer can
        // pick the active loc's badge per page without re-querying.
        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        var sql = $@"
WITH latest AS (
  SELECT Model, LocationName, ValidTimeUtc,
         Visibility, Temperature2m, DewPoint2m,
         row_number() OVER (PARTITION BY Model, LocationName, ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE LocationName IN ({locFilter})
    AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT LocationName, ValidTimeUtc,
       SUM(CASE WHEN Visibility IS NOT NULL AND Visibility < {VisFireThresholdM.ToString(System.Globalization.CultureInfo.InvariantCulture)} THEN 1 ELSE 0 END)::INTEGER AS vis_fired,
       SUM(CASE WHEN Visibility IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS vis_total,
       SUM(CASE WHEN Temperature2m IS NOT NULL AND DewPoint2m IS NOT NULL
                     AND (Temperature2m - DewPoint2m) < {TmTdFireThresholdC.ToString(System.Globalization.CultureInfo.InvariantCulture)}
                THEN 1 ELSE 0 END)::INTEGER AS tdspread_fired,
       SUM(CASE WHEN Temperature2m IS NOT NULL AND DewPoint2m IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS tdspread_total
FROM latest
WHERE rn = 1
GROUP BY LocationName, ValidTimeUtc";

        var rows = ParquetReader.Query(sql, r => (
            Loc: r.GetString(0),
            Time: r.GetDateTime(1),
            VisFired: r.GetInt32(2),
            VisTotal: r.GetInt32(3),
            TdFired: r.GetInt32(4),
            TdTotal: r.GetInt32(5)),
            _log, "Forecast tree empty — low-cloud / mist warning will be absent.", ct);

        var result = new Dictionary<string, IReadOnlyDictionary<DateTime, SitePages.LowCloudSignal>>(StringComparer.Ordinal);
        foreach (var grp in rows.GroupBy(x => x.Loc, StringComparer.Ordinal))
        {
            result[grp.Key] = grp.ToDictionary(
                x => x.Time,
                x => new SitePages.LowCloudSignal(
                    VisFiredCount: x.VisFired, VisTotalCount: x.VisTotal,
                    CloudBaseFiredCount: x.TdFired, CloudBaseTotalCount: x.TdTotal));
        }
        return result;
    }

    /// <summary>Per-NWP hourly precipitation probability (%) from the raw
    /// forecast tree, deduped to freshest RunTime per (Model, ValidTime).
    /// Feeds the rain-page PoP overlay panel.</summary>
    private IReadOnlyList<SitePages.NwpPrecipProbForecastPoint> QueryNwpPrecipProbabilities(
        DateTime start, DateTime end, CancellationToken ct)
        => QueryNwpColumn(start, end, "PrecipitationProbability",
            r => new SitePages.NwpPrecipProbForecastPoint(
                Model:              r.GetString(0),
                ValidTimeUtc:       r.GetDateTime(1),
                ProbabilityPercent: r.GetDouble(2),
                LocationName:       r.GetString(3)),
            "Per-NWP precipitation_probability tree empty — overlay panel will be absent.", ct);

    /// <summary>Per-NWP precipitation rate (mm/h) from the raw forecast tree —
    /// hourly granularity, deduped to freshest RunTime per (Model, ValidTime).
    /// Used by the lead-12 rain page's mm/h chart where the
    /// prediction-row-based overlay is too sparse.</summary>
    private IReadOnlyList<SitePages.NwpPrecipRateForecastPoint> QueryNwpPrecipRates(
        DateTime start, DateTime end, CancellationToken ct)
        => QueryNwpColumn(start, end, "Precipitation",
            r => new SitePages.NwpPrecipRateForecastPoint(
                Model:           r.GetString(0),
                ValidTimeUtc:    r.GetDateTime(1),
                PrecipMmPerHour: r.GetDouble(2),
                LocationName:    r.GetString(3)),
            "Per-NWP precipitation rate tree empty — overlay panel will be absent.", ct);

    /// <summary>Per-NWP 2m temperature (°C) from the raw forecast tree —
    /// same shape as the precip rate / PoP queries above. Used on the
    /// lead-12 temperature page where the prediction-row-based overlay
    /// is too sparse (2b/2c don't emit at lead 12; 2d only emits at
    /// {0,6,12,18}Z valid hours).</summary>
    private IReadOnlyList<SitePages.NwpTemperatureForecastPoint> QueryNwpTemperatures(
        DateTime start, DateTime end, CancellationToken ct)
        => QueryNwpColumn(start, end, "Temperature2m",
            r => new SitePages.NwpTemperatureForecastPoint(
                Model:         r.GetString(0),
                ValidTimeUtc:  r.GetDateTime(1),
                Temperature2m: r.GetDouble(2),
                LocationName:  r.GetString(3)),
            "Per-NWP temperature tree empty — overlay panel will be absent.", ct);

    /// <summary>
    /// Shared body for the three per-NWP overlay queries above — they were
    /// textually identical except the projected column. One row per
    /// (Location, Model, ValidTime) inside the window: blender-input NWPs
    /// only, NULL values dropped, freshest RunTime wins. The mapper reads
    /// columns positionally: 0=Model, 1=ValidTimeUtc, 2=the projected value,
    /// 3=LocationName.
    /// </summary>
    private IReadOnlyList<T> QueryNwpColumn<T>(
        DateTime start, DateTime end, string column,
        Func<System.Data.IDataReader, T> map, string emptyMessage, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.ForecastsPath, "**", "*.parquet"));

        var nwpFilter = WeatherBlend.Train.Common.Nwp.SqlInList();

        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));

        var sql = $@"
WITH ranked AS (
  SELECT Model, LocationName, RunTimeUtc, ValidTimeUtc, {column},
         row_number() OVER (PARTITION BY Model, LocationName, ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE LocationName IN ({locFilter})
    AND Model IN {nwpFilter}
    AND {column} IS NOT NULL
    AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT Model, ValidTimeUtc, {column}, LocationName
FROM ranked WHERE rn = 1 ORDER BY LocationName, Model, ValidTimeUtc";

        return ParquetReader.Query(sql, map, _log, emptyMessage, ct);
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
        // Glob the model=met_office_spot subtree only — the hive path encodes
        // the model, and a whole-tree glob makes DuckDB open every NWP file's
        // footer just to discard its rows (~7 min on a local full-archive
        // tree vs seconds; CI never noticed because its pull is date-windowed).
        var glob = ParquetReader.Glob(Path.Combine(
            _cfg.Storage.ForecastsPath, "location=*", $"model={_cfg.MetOffice.SpotModelTag}", "**", "*.parquet"));

        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));

        // Partition by (Location, ValidTime, LeadHours) — keeps one row per
        // (Loc, Valid, Lead) so the renderer can bucket Spot rows into the
        // chart's per-lead traces ([L, L+24) windows). The freshest RunTime
        // within each (Loc, Valid, Lead) triple wins — typically the only
        // one (a 6-hourly Spot cycle emits exactly one row per ValidTime per
        // unique lead) so the dedup is mostly a no-op but defends against
        // an unusual collect cycle.
        var sql = $@"
WITH ranked AS (
  SELECT RunTimeUtc, ValidTimeUtc, LocationName, LeadHours, Temperature2m, PrecipitationProbability, WindSpeed10m,
         row_number() OVER (PARTITION BY LocationName, ValidTimeUtc, LeadHours ORDER BY RunTimeUtc DESC) AS rn
  FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
  WHERE LocationName IN ({locFilter})
    AND Model = 'met_office_spot'
    AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
    AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
)
SELECT RunTimeUtc, ValidTimeUtc, Temperature2m, PrecipitationProbability, LocationName, LeadHours, WindSpeed10m
FROM ranked WHERE rn = 1 ORDER BY LocationName, ValidTimeUtc, LeadHours";

        return ParquetReader.Query(sql, r => new SitePages.MetOfficeSpotForecastPoint(
            RunTimeUtc:                      r.GetDateTime(0),
            ValidTimeUtc:                    r.GetDateTime(1),
            Temperature2m:                   r.IsDBNull(2) ? null : r.GetDouble(2),
            PrecipitationProbabilityPercent: r.IsDBNull(3) ? null : r.GetDouble(3),
            LocationName:                    r.GetString(4),
            LeadHours:                       r.GetInt32(5),
            WindSpeed10mMs:                  r.IsDBNull(6) ? null : r.GetDouble(6)),
            _log, "Met Office Spot tree empty — comparison line will be absent.", ct);
    }

    /// <summary>
    /// Dry-window start-hour curves: one row per (Station, WindowHours,
    /// LeadHours, TargetDateUtc, StartHourUtc), produced today by Phase 3p
    /// as a byproduct of its copula-MC pass. Read off
    /// <c>data/predictions/dry_window_start_hour/{station}/window_{N}h/
    /// model_version={v}/date={anchor}/predictions.parquet</c>. Returns an
    /// empty list on first-ever-deploy (no curves on disk yet) so the
    /// dry-window page degrades silently to "no chart".
    /// </summary>
    private IReadOnlyList<SitePages.StartHourForecastPoint> QueryStartHourPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var subdir = Path.Combine(_cfg.Storage.PredictionsPath, "dry_window_start_hour");
        if (!Directory.Exists(subdir))
        {
            _log.LogInformation(
                "Start-hour curves tree at {Dir} not present — chart will be absent until the next 3p predict cycle lands.",
                subdir);
            return Array.Empty<SitePages.StartHourForecastPoint>();
        }

        var glob = ParquetReader.Glob(Path.Combine(subdir, "**", "*.parquet"));
        // Probe: a fresh deploy may have the dir but no parquet yet, or
        // historical parquets predating ConditionalProb. read_parquet
        // would throw on the missing column.
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "ConditionalProb"))
        {
            _log.LogWarning("Start-hour curves tree has no ConditionalProb-shaped rows yet — chart will be absent.");
            return Array.Empty<SitePages.StartHourForecastPoint>();
        }

        var locFilter = string.Join(",", _cfg.Locations.Select(l => $"'{l.Name.Replace("'", "''")}'"));
        if (string.IsNullOrEmpty(locFilter)) locFilter = "''";

        var sql = $@"
SELECT TruthStation, WindowHours, ModelVersion, PredictionMadeAtUtc, TargetDateUtc, LeadHours,
       StartHourUtc, RawProduct, ConditionalProb, CalibratedProb, DailyProbAnyBlock, LocationName
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName IN ({locFilter})
  AND TargetDateUtc >= TIMESTAMP '{start.Date:yyyy-MM-dd HH:mm:ss}'
  AND TargetDateUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY LocationName, TruthStation, WindowHours, LeadHours, TargetDateUtc, StartHourUtc";

        return ParquetReader.Query(conn, sql, r => new SitePages.StartHourForecastPoint(
            Station:           r.GetString(0),
            WindowHours:       r.GetInt32(1),
            Version:           r.GetString(2),
            PredictedAtUtc:    r.GetDateTime(3),
            TargetDateUtc:     r.GetDateTime(4),
            LeadHours:         r.GetInt32(5),
            StartHourUtc:      r.GetInt32(6),
            RawProduct:        r.GetDouble(7),
            ConditionalProb:   r.GetDouble(8),
            CalibratedProb:    r.GetDouble(9),
            DailyProbAnyBlock: r.GetDouble(10),
            LocationName:      r.GetString(11)),
            _log, "Start-hour curves tree empty — chart will be absent.", ct);
    }

}
