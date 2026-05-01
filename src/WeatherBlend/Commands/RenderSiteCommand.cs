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

    public RenderSiteCommand(ILogger<RenderSiteCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
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

        var predictions = QueryPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} temperature prediction rows.", predictions.Count);

        var precip = QueryPrecipPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} precipitation prediction rows.", precip.Count);

        var feelsLike = QueryFeelsLikePredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} feels-like prediction rows.", feelsLike.Count);

        var dryWindow = QueryDryWindowPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} dry-window prediction rows.", dryWindow.Count);

        var startHour = QueryStartHourPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} start-hour curve rows.", startHour.Count);

        var metOfficeSpot = QueryMetOfficeSpotForecasts(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} Met Office Spot forecast rows.", metOfficeSpot.Count);

        var nwpPop = QueryNwpPrecipProbabilities(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} per-NWP precipitation-probability rows ({M} models).",
            nwpPop.Count, nwpPop.Select(p => p.Model).Distinct().Count());

        var verifyHistory = LoadVerifyHistory(_cfg.Storage.ReportsPath);
        _log.LogInformation("Loaded {N} verify-history JSON sidecars.", verifyHistory.Count);

        // ERA5 is gapless-for-past, but only past — query up to the clock time.
        // Persistence-lookback headroom (up to 72h before the earliest prediction)
        // keeps rolling-MAE truth pairs matchable at the start of the window.
        var truth = QueryTruth(windowStart.AddDays(-3), now, ct);
        _log.LogInformation("Loaded {N} ERA5 truth points.", truth.Count);

        var metar = QueryMetar(windowStart, now, ct);
        _log.LogInformation("Loaded {N} METAR observations ({Station}).",
            metar.Count, string.IsNullOrWhiteSpace(_cfg.Location.Metar.Primary) ? "none" : _cfg.Location.Metar.Primary);

        var rolling = ComputeRollingMae(predictions, truth, rollingWindowDays);
        // Precip uses a longer 30-day rolling window than temp's because wet
        // hours are sparser — the verify command's defaults pin this rule
        // (PrecipVerifyCommand: 30d window vs TempVerifyCommand: 14d).
        const int precipRollingWindowDays = 30;

        var phaseByVersion = LoadPhaseByVersion(predictions, precip, dryWindow);
        _log.LogInformation("Phase map: {Entries}",
            string.Join(", ", phaseByVersion.Select(kv => $"{kv.Key}→{kv.Value}")));

        var rainfall = LoadRainfallTruth(windowStart, now, precip, ct);
        _log.LogInformation("Rainfall truth: {N} stations loaded.", rainfall.Count);

        var currentVersion = LoadCurrentTemperatureVersion();
        _log.LogInformation("Champion (temperature): {Version}",
            string.IsNullOrEmpty(currentVersion) ? "(none)" : currentVersion);

        var precipCurrentByStation = LoadCurrentPrecipByStation();
        _log.LogInformation("Champion (precipitation): {Entries}",
            precipCurrentByStation.Count == 0
                ? "(none)"
                : string.Join(", ", precipCurrentByStation.Select(kv => $"{kv.Key}→{kv.Value}")));

        var modelSummaries = LoadModelSummaries(predictions, precip, dryWindow);
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
            RollingMae = rolling,
            RollingBrier = ComputeRollingBrier(precip, rainfall, precipRollingWindowDays),
            PrecipPredictions = precip,
            DryWindowPredictions = dryWindow,
            PhaseByVersion = phaseByVersion,
            RainfallTruth = rainfall,
            CurrentVersion = currentVersion,
            PrecipCurrentByStation = precipCurrentByStation,
            ModelSummaries = modelSummaries,
            FeelsLikePredictions = feelsLike,
            StartHourPredictions = startHour,
            MetOfficeSpotForecasts = metOfficeSpot,
            NwpPrecipProbabilities = nwpPop,
            VerifyHistory = verifyHistory,
        };

        Directory.CreateDirectory(outputDir);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "index.html"),         SitePages.RenderIndex(input),          ct);
        foreach (var lead in Leads.Full)
            await File.WriteAllTextAsync(Path.Combine(outputDir, $"forecasts-{lead}h.html"),
                SitePages.RenderForecasts(input, lead), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "models.html"),        SitePages.RenderModels(input),         ct);
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

        // Dry-window page is per-station too. Dry-window-only stations are a subset
        // of rain-skill stations (it's currently Bellever + Princetown, no Hexworthy).
        var dryStations = input.DryWindowPredictions.Select(d => d.Station).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        await File.WriteAllTextAsync(Path.Combine(outputDir, "dry-window.html"), SitePages.RenderDryWindow(input, null), ct);
        for (int i = 1; i < dryStations.Count; i++)
        {
            var slug = SitePages.StationSlug(dryStations[i]);
            await File.WriteAllTextAsync(
                Path.Combine(outputDir, $"dry-window-{slug}.html"),
                SitePages.RenderDryWindow(input, slug), ct);
        }

        // index + per-lead forecasts + models + about + styles + chart.js
        // + skill-temperature + skill-rainfall × stations + dry-window × stations.
        var totalFiles = 6 + Leads.Full.Length
            + Math.Max(1, rainStations.Count) + Math.Max(1, dryStations.Count);
        _log.LogInformation("Site rendered → {Dir} ({Files} files)", outputDir, totalFiles);
        return 0;
    }

    private IReadOnlyList<SitePages.ModelSummary> LoadModelSummaries(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        IReadOnlyList<SitePages.DryWindowForecastPoint> dryWindow)
    {
        // Only load metadata for versions that actually emitted predictions in the
        // window. Anything else is stale / experimental / deleted and the Models page
        // would list it with no corresponding forecast activity.
        var modelsRoot = Path.Combine("data", "models");
        var summaries = new List<SitePages.ModelSummary>();

        foreach (var version in predictions.Select(p => p.ModelVersion).Distinct())
        {
            var dir = Path.Combine(modelsRoot, "temperature", version);
            var summary = TryLoadSummary(dir, composite: "temperature", version, metricLabel: "Test MAE (°C)");
            if (summary is not null) summaries.Add(summary);
        }

        foreach (var (station, version) in precip.Select(p => (p.Station, p.Version)).Distinct())
        {
            var dir = Path.Combine(modelsRoot, "precipitation", station, version);
            var composite = $"precipitation/{station}";
            var summary = TryLoadSummary(dir, composite, version, metricLabel: "Test Brier");
            if (summary is not null) summaries.Add(summary);
        }

        foreach (var (station, window, version) in dryWindow.Select(d => (d.Station, d.WindowHours, d.Version)).Distinct())
        {
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

    private string LoadCurrentTemperatureVersion()
    {
        // Reads MANIFEST.json directly rather than calling ResolveVersionDir("current"),
        // which throws when the manifest is absent — on a fresh checkout the home page
        // should still render, just without a champion filter.
        var manifestPath = Path.Combine("data", "models", "temperature", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath)) return "";
        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = System.Text.Json.JsonSerializer.Deserialize<ModelArtifact.Manifest>(json);
            return manifest?.Current ?? "";
        }
        catch (Exception ex)
        {
            _log.LogWarning("Failed to read temperature manifest: {Msg}", ex.Message);
            return "";
        }
    }

    private Dictionary<string, string> LoadCurrentPrecipByStation()
    {
        // Precipitation uses the per-station manifest layout: one StationEntry per EA
        // station slug, each with its own Current. Missing manifest → no champion,
        // home page skips the P(wet) chip rather than falling back to "latest anywhere".
        var manifestPath = Path.Combine("data", "models", "precipitation", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath)) return new(StringComparer.Ordinal);
        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = System.Text.Json.JsonSerializer.Deserialize<ModelArtifact.Manifest>(json);
            if (manifest?.Stations is null) return new(StringComparer.Ordinal);
            return manifest.Stations
                .Where(kv => !string.IsNullOrEmpty(kv.Value.Current))
                .ToDictionary(kv => kv.Key, kv => kv.Value.Current, StringComparer.Ordinal);
        }
        catch (Exception ex)
        {
            _log.LogWarning("Failed to read precipitation manifest: {Msg}", ex.Message);
            return new(StringComparer.Ordinal);
        }
    }

    private Dictionary<string, string> LoadPhaseByVersion(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        IReadOnlyList<SitePages.DryWindowForecastPoint> dryWindow)
    {
        // Only the versions that actually produced predictions are worth reading.
        // Missing training_metadata.json is non-fatal — the caller treats empty phase
        // as "other" and renders the row outside the 2b/2c or 3a/3c charts.
        var modelsRoot = Path.Combine("data", "models");
        var phases = new Dictionary<string, string>(StringComparer.Ordinal);

        // Temperature: flat tree data/models/temperature/{version}/.
        foreach (var v in predictions.Select(p => p.ModelVersion).Distinct())
        {
            try
            {
                var dir = ModelArtifact.ResolveVersionDir(modelsRoot, "temperature", v);
                if (!Directory.Exists(dir)) continue;
                var metadataPath = Path.Combine(dir, ModelArtifact.TrainingMetadataFileName);
                if (!File.Exists(metadataPath)) continue;
                var metadata = ModelArtifact.LoadTrainingMetadata(dir);
                if (!string.IsNullOrWhiteSpace(metadata.Phase))
                    phases[v] = metadata.Phase;
            }
            catch (Exception ex)
            {
                _log.LogWarning("Phase lookup failed for temperature {Version}: {Msg}", v, ex.Message);
            }
        }

        // Precipitation: per-station tree data/models/precipitation/{station}/{version}/.
        // Same version string can recur across stations (e.g. v..._phase3c at all three
        // stations), but the Phase is the same — probing any one station that produced
        // the version is enough, so we iterate distinct (station, version) pairs and the
        // later write just overwrites with the identical value.
        foreach (var (station, version) in precip.Select(p => (p.Station, p.Version)).Distinct())
        {
            try
            {
                var dir = Path.Combine(modelsRoot, "precipitation", station, version);
                if (!Directory.Exists(dir)) continue;
                var metadataPath = Path.Combine(dir, ModelArtifact.TrainingMetadataFileName);
                if (!File.Exists(metadataPath)) continue;
                var metadata = ModelArtifact.LoadTrainingMetadata(dir);
                if (!string.IsNullOrWhiteSpace(metadata.Phase))
                    phases[version] = metadata.Phase;
            }
            catch (Exception ex)
            {
                _log.LogWarning("Phase lookup failed for precipitation {Station}/{Version}: {Msg}", station, version, ex.Message);
            }
        }

        // Dry-window: per-(station, window) tree data/models/dry_window/{station}/window_{N}h/{version}/.
        // Phase 3d artefacts have phase-suffixed version dirs ("..._phase3d_shape", "..._phase3d_calibrated"),
        // so version strings don't collide across composites — a flat version→phase map is safe.
        foreach (var (station, window, version) in dryWindow.Select(d => (d.Station, d.WindowHours, d.Version)).Distinct())
        {
            try
            {
                var dir = Path.Combine(modelsRoot, "dry_window", station, $"window_{window}h", version);
                if (!Directory.Exists(dir)) continue;
                var metadataPath = Path.Combine(dir, ModelArtifact.TrainingMetadataFileName);
                if (!File.Exists(metadataPath)) continue;
                var metadata = ModelArtifact.LoadTrainingMetadata(dir);
                if (!string.IsNullOrWhiteSpace(metadata.Phase))
                    phases[version] = metadata.Phase;
            }
            catch (Exception ex)
            {
                _log.LogWarning("Phase lookup failed for dry-window {Station}/{Window}h/{Version}: {Msg}",
                    station, window, version, ex.Message);
            }
        }

        return phases;
    }

    private IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> LoadRainfallTruth(
        DateTime start, DateTime end,
        IReadOnlyList<SitePages.PrecipForecastPoint> precip,
        CancellationToken ct)
    {
        // Only load truth for stations that actually produced precip predictions in the
        // window. Mirrors PrecipVerifyCommand's slug → StationName mapping so the filter
        // resolves to the exact StationName stored in the truth parquet.
        var stations = precip.Select(p => p.Station).Distinct().ToList();
        if (stations.Count == 0)
            return new Dictionary<string, IReadOnlyDictionary<DateTime, double>>();

        var stationNamesBySlug = _cfg.Location.Rainfall.Stations.ToDictionary(
            s => "ea_" + Slugify(s.Name),
            s => s.Name,
            StringComparer.OrdinalIgnoreCase);

        var result = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase);
        foreach (var slug in stations)
        {
            if (!stationNamesBySlug.TryGetValue(slug, out var stationName))
            {
                _log.LogWarning("Rainfall truth: no config station for slug {Slug}.", slug);
                result[slug] = new Dictionary<DateTime, double>();
                continue;
            }
            result[slug] = QueryHourlyRainfall(stationName, start, end, ct);
        }
        return result;
    }

    private IReadOnlyDictionary<DateTime, double> QueryHourlyRainfall(
        string stationName, DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet"));

        // Same 4-of-4 aggregation as PrecipVerify so a partial hour can't flip wet↔dry.
        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND StationName  = '{stationName.Replace("'", "''")}'
  AND Value15MinMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end.AddHours(1):yyyy-MM-dd HH:mm:ss}'
GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1";

        var rows = ParquetReader.Query(
            sql,
            r => (Hour: r.GetDateTime(0), Mm: r.GetDouble(1)),
            _log,
            $"Rainfall tree empty for {stationName}.",
            ct);
        return rows.ToDictionary(x => x.Hour, x => x.Mm);
    }

    // Kept local (rather than importing Slugify from PrecipVerifyCommand) so this file
    // doesn't depend on that command's internals. The rule is simple: lowercase,
    // non-alphanumeric → underscore, collapse repeats.
    private static string Slugify(string input)
    {
        var sb = new System.Text.StringBuilder(input.Length);
        var lastWasUnderscore = false;
        foreach (var c in input.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(c);
                lastWasUnderscore = false;
            }
            else if (!lastWasUnderscore)
            {
                sb.Append('_');
                lastWasUnderscore = true;
            }
        }
        return sb.ToString().Trim('_');
    }

    /// <summary>
    /// Rolling MAE per (ModelVersion, LeadHours, daily window end) across the last
    /// <paramref name="windowDays"/>. One point per day from the earliest paired
    /// date through the latest. Days with fewer than <paramref name="windowDays"/>
    /// of data behind them emit a point computed over whatever's available in
    /// <c>[d − windowDays + 1, d]</c> — partial-window points show up at the
    /// start of the data, which is what users expect when the prediction tree
    /// is younger than the rolling window. Without that, a 14-day window over
    /// 3 days of data renders an empty chart even though the underlying pairs
    /// are right there in the per-lead chart above.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingMaePoint> ComputeRollingMae(
        IReadOnlyList<TempPredictionRow> predictions,
        IReadOnlyDictionary<DateTime, double> truthByTime,
        int windowDays)
    {
        // Pair each prediction with its ERA5 truth (drop unpaired).
        var paired = predictions
            .Where(p => truthByTime.ContainsKey(p.ValidTimeUtc))
            .Select(p => (p.ModelVersion, p.LeadHours, p.ValidTimeUtc, Pred: p.BlendTemperature, Truth: truthByTime[p.ValidTimeUtc]))
            .ToList();

        if (paired.Count == 0) return Array.Empty<SitePages.RollingMaePoint>();

        var minDate = paired.Min(r => r.ValidTimeUtc).Date;
        var maxDate = paired.Max(r => r.ValidTimeUtc).Date;

        var points = new List<SitePages.RollingMaePoint>();

        foreach (var (version, lead) in paired.Select(p => (p.ModelVersion, p.LeadHours)).Distinct())
        {
            var subset = paired.Where(p => p.ModelVersion == version && p.LeadHours == lead).ToList();
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
                points.Add(new SitePages.RollingMaePoint(version, lead, windowEnd, mae, inWindow.Count));
            }
        }

        return points;
    }

    /// <summary>
    /// Rolling Brier per (Station, ModelVersion, LeadHours, day-end) across
    /// the last <paramref name="windowDays"/>. Mirror of
    /// <see cref="ComputeRollingMae"/> for binary classification: pairs each
    /// prediction's <c>ProbWet</c> with the EA gauge's wet/dry indicator at
    /// the same hour, then computes the squared-error mean over each rolling
    /// window. Truth conversion uses the same 0.1 mm/h threshold the
    /// blender was trained on. Partial-window points emit at the start of
    /// the data — same rule the temp rolling-MAE uses (commit 4028ba5),
    /// readers gauge stability from the per-point <c>N</c>.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingBrierPoint> ComputeRollingBrier(
        IReadOnlyList<SitePages.PrecipForecastPoint> predictions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> rainfallByStation,
        int windowDays)
    {
        const double WetThresholdMm = 0.1;

        // Pair each prediction with its station's truth (drop unpaired —
        // either station has no truth dict yet, or that hour didn't pass
        // the 4-of-4 quarter-hour gate upstream).
        var paired = predictions
            .Where(p => rainfallByStation.TryGetValue(p.Station, out var byTime)
                        && byTime.ContainsKey(p.ValidTimeUtc))
            .Select(p =>
            {
                var byTime = rainfallByStation[p.Station];
                var truthMm = byTime[p.ValidTimeUtc];
                return (p.Station, p.Version, p.LeadHours, p.ValidTimeUtc,
                        Pred: p.ProbWet,
                        Truth: truthMm >= WetThresholdMm ? 1.0 : 0.0);
            })
            .ToList();

        if (paired.Count == 0) return Array.Empty<SitePages.RollingBrierPoint>();

        var minDate = paired.Min(r => r.ValidTimeUtc).Date;
        var maxDate = paired.Max(r => r.ValidTimeUtc).Date;

        var points = new List<SitePages.RollingBrierPoint>();
        foreach (var (station, version, lead) in paired
            .Select(p => (p.Station, p.Version, p.LeadHours)).Distinct())
        {
            var subset = paired.Where(p => p.Station == station
                                        && p.Version == version
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
                    station, version, lead, windowEnd, brier, inWindow.Count));
            }
        }
        return points;
    }

    private IReadOnlyList<TempPredictionRow> QueryPredictions(DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Scope to the temperature subtree only — sibling subtrees (precipitation,
        // dry_window) have different schemas and would surface as nulls or wrong
        // columns under union_by_name.
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "temperature", "**", "*.parquet"));

        var sql = $@"
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendTemperature,
       TempGfs, TempEcmwf, TempIcon, TempMf, TempUkmo, TempGem, TempAifs,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem, RunTimeAifs,
       TempMean, TempStd, TempRange,
       FeatureVectorHash
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND BlendTemperature IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY PredictionMadeAtUtc DESC, LeadHours";

        return ParquetReader.Query(sql, r => new TempPredictionRow
        {
            LocationName        = r.GetString(0),
            ModelVersion        = r.GetString(1),
            PredictionMadeAtUtc = r.GetDateTime(2),
            ValidTimeUtc        = r.GetDateTime(3),
            LeadHours           = r.GetInt32(4),
            BlendTemperature    = r.GetDouble(5),
            TempGfs   = NullableDouble(r,  6),
            TempEcmwf = NullableDouble(r,  7),
            TempIcon  = NullableDouble(r,  8),
            TempMf    = NullableDouble(r,  9),
            TempUkmo  = NullableDouble(r, 10),
            TempGem   = NullableDouble(r, 11),
            TempAifs  = NullableDouble(r, 12),
            RunTimeGfs   = NullableDate(r, 13),
            RunTimeEcmwf = NullableDate(r, 14),
            RunTimeIcon  = NullableDate(r, 15),
            RunTimeMf    = NullableDate(r, 16),
            RunTimeUkmo  = NullableDate(r, 17),
            RunTimeGem   = NullableDate(r, 18),
            RunTimeAifs  = NullableDate(r, 19),
            TempMean  = NullableDouble(r, 20),
            TempStd   = NullableDouble(r, 21),
            TempRange = NullableDouble(r, 22),
            FeatureVectorHash = r.IsDBNull(23) ? "" : r.GetString(23),
        }, _log, "Predictions tree empty — rendering an empty-state site.", ct);
    }

    private IReadOnlyList<SitePages.PrecipForecastPoint> QueryPrecipPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "precipitation", "**", "*.parquet"));

        var sql = $@"
SELECT TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet,
       PrecipGfs, PrecipEcmwf, PrecipIcon, PrecipMf, PrecipUkmo, PrecipGem, PrecipAifs, PrecipJma
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(sql, r => new SitePages.PrecipForecastPoint(
            Station:         r.GetString(0),
            Version:         r.GetString(1),
            PredictedAtUtc:  r.GetDateTime(2),
            ValidTimeUtc:    r.GetDateTime(3),
            LeadHours:       r.GetInt32(4),
            ProbWet:         r.GetDouble(5),
            ClimatologyPWet: r.GetDouble(6),
            PrecipGfs:       r.IsDBNull(7)  ? null : r.GetDouble(7),
            PrecipEcmwf:     r.IsDBNull(8)  ? null : r.GetDouble(8),
            PrecipIcon:      r.IsDBNull(9)  ? null : r.GetDouble(9),
            PrecipMf:        r.IsDBNull(10) ? null : r.GetDouble(10),
            PrecipUkmo:      r.IsDBNull(11) ? null : r.GetDouble(11),
            PrecipGem:       r.IsDBNull(12) ? null : r.GetDouble(12),
            PrecipAifs:      r.IsDBNull(13) ? null : r.GetDouble(13),
            PrecipJma:       r.IsDBNull(14) ? null : r.GetDouble(14)),
            _log, "Precipitation predictions tree empty — precip page will render an empty state.", ct);
    }

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

        var sql = $@"
SELECT ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours, UtciC, Band,
       ApparentTemperatureC
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND UtciC IS NOT NULL
  AND ApparentTemperatureC IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY LeadHours, ValidTimeUtc";

        return ParquetReader.Query(conn, sql, r => new SitePages.FeelsLikeForecastPoint(
            Version:        r.GetString(0),
            PredictedAtUtc: r.GetDateTime(1),
            ValidTimeUtc:   r.GetDateTime(2),
            LeadHours:      r.GetInt32(3),
            UtciC:          r.GetDouble(4),
            Band:           r.IsDBNull(5) ? "" : r.GetString(5),
            ApparentC:      r.GetDouble(6)),
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

    private IReadOnlyList<SitePages.DryWindowForecastPoint> QueryDryWindowPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "dry_window", "**", "*.parquet"));

        // Dry-window predictions are anchored on TargetDateUtc (UTC midnight of the
        // labelled day), not ValidTimeUtc. Bound on TargetDate instead.
        var sql = $@"
SELECT TruthStation, WindowHours, ModelVersion, PredictionMadeAtUtc, TargetDateUtc, LeadHours,
       ProbHasDryWindow, ClimatologyProbHasDryWindow, AgreementHasDryWindow
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND TargetDateUtc >= TIMESTAMP '{start.Date:yyyy-MM-dd HH:mm:ss}'
  AND TargetDateUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, WindowHours, LeadHours, TargetDateUtc";

        return ParquetReader.Query(sql, r => new SitePages.DryWindowForecastPoint(
            Station:                     r.GetString(0),
            WindowHours:                 r.GetInt32(1),
            Version:                     r.GetString(2),
            PredictedAtUtc:              r.GetDateTime(3),
            TargetDateUtc:               r.GetDateTime(4),
            LeadHours:                   r.GetInt32(5),
            ProbHasDryWindow:            r.GetDouble(6),
            ClimatologyProbHasDryWindow: r.GetDouble(7),
            AgreementHasDryWindow:       r.IsDBNull(8) ? null : r.GetDouble(8)),
            _log, "Dry-window predictions tree empty — dry-window page will render an empty state.", ct);
    }

    private IReadOnlyList<(DateTime ObservedTimeUtc, double Temperature2m)> QueryMetar(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var station = _cfg.Location.Metar.Primary;
        if (string.IsNullOrWhiteSpace(station))
        {
            _log.LogWarning("No primary METAR station configured — METAR chart series will be empty.");
            return Array.Empty<(DateTime, double)>();
        }

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.ObservationsPath, "**", "*.parquet"));

        // Filter on in-file Station column (authoritative per CLAUDE.md gotcha about
        // hive-key vs column collision). Primary station only — secondary fallback
        // would confuse the chart legend.
        var sql = $@"
SELECT ObservedTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND Station = '{station.Replace("'", "''")}'
  AND Temperature2m IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ObservedTimeUtc";

        return ParquetReader.Query(sql, r => (r.GetDateTime(0), r.GetDouble(1)),
            _log, "METAR tree empty — METAR chart series will be empty.", ct);
    }

    private IReadOnlyDictionary<DateTime, double> QueryTruth(DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet"));

        var sql = $@"
SELECT ValidTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND Temperature2m IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'";

        var rows = ParquetReader.Query(sql, r => (Time: r.GetDateTime(0), Temp: r.GetDouble(1)),
            _log, "ERA5 tree empty — skill charts will be empty.", ct);
        return rows.ToDictionary(x => x.Time, x => x.Temp);
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static DateTime? NullableDate(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
