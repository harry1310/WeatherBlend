using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Renders a self-contained static site to <c>data/site/</c>: home, predictions table,
/// verification charts, and an about page. Reads prediction and ERA5 parquet via DuckDB
/// using the same glob pattern as <see cref="VerifyCommand"/> and the same
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
        // so include the full forecast horizon (leads go to 72h) plus a day of headroom.
        var predictionEnd = now.AddDays(4);

        _log.LogInformation(
            "Render site: predictions [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], rolling {Rolling}d, output → {Dir}",
            windowStart, predictionEnd, rollingWindowDays, outputDir);

        var predictions = QueryPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} temperature prediction rows.", predictions.Count);

        var precip = QueryPrecipPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} precipitation prediction rows.", precip.Count);

        var dryWindow = QueryDryWindowPredictions(windowStart, predictionEnd, ct);
        _log.LogInformation("Loaded {N} dry-window prediction rows.", dryWindow.Count);

        // ERA5 is gapless-for-past, but only past — query up to the clock time.
        // Persistence-lookback headroom (up to 72h before the earliest prediction)
        // keeps rolling-MAE truth pairs matchable at the start of the window.
        var truth = QueryTruth(windowStart.AddDays(-3), now, ct);
        _log.LogInformation("Loaded {N} ERA5 truth points.", truth.Count);

        var metar = QueryMetar(windowStart, now, ct);
        _log.LogInformation("Loaded {N} METAR observations ({Station}).",
            metar.Count, string.IsNullOrWhiteSpace(_cfg.Location.Metar.Primary) ? "none" : _cfg.Location.Metar.Primary);

        var rolling = ComputeRollingMae(predictions, truth, rollingWindowDays);

        var phaseByVersion = LoadPhaseByVersion(predictions);
        _log.LogInformation("Phase map: {Entries}",
            string.Join(", ", phaseByVersion.Select(kv => $"{kv.Key}→{kv.Value}")));

        var rainfall = LoadRainfallTruth(windowStart, now, precip, ct);
        _log.LogInformation("Rainfall truth: {N} stations loaded.", rainfall.Count);

        var currentVersion = LoadCurrentTemperatureVersion();
        _log.LogInformation("Champion (temperature): {Version}",
            string.IsNullOrEmpty(currentVersion) ? "(none)" : currentVersion);

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
            PrecipPredictions = precip,
            DryWindowPredictions = dryWindow,
            PhaseByVersion = phaseByVersion,
            RainfallTruth = rainfall,
            CurrentVersion = currentVersion,
        };

        Directory.CreateDirectory(outputDir);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "index.html"),        SitePages.RenderIndex(input),         ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "predictions.html"),  SitePages.RenderPredictions(input),   ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "precipitation.html"),SitePages.RenderPrecipitation(input), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "dry-window.html"),   SitePages.RenderDryWindow(input),     ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "verify.html"),       SitePages.RenderVerify(input),        ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "forecast-vs-truth.html"), SitePages.RenderForecastVsTruth(input), ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "about.html"),        SitePages.RenderAbout(input),         ct);
        await File.WriteAllTextAsync(Path.Combine(outputDir, "styles.css"),        SitePages.Stylesheet(),               ct);

        _log.LogInformation("Site rendered → {Dir} ({Files} files)", outputDir, 8);
        return 0;
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

    private Dictionary<string, string> LoadPhaseByVersion(IReadOnlyList<PredictionRow> predictions)
    {
        // Only the versions that actually produced predictions are worth reading.
        // Missing training_metadata.json is non-fatal — the caller treats empty phase
        // as "other" and renders the row outside the 2b/2c charts.
        var modelsRoot = Path.Combine("data", "models");
        var phases = new Dictionary<string, string>(StringComparer.Ordinal);
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
                _log.LogWarning("Phase lookup failed for {Version}: {Msg}", v, ex.Message);
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

        var glob = Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

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

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var hourly = new Dictionary<DateTime, double>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                hourly[r.GetDateTime(0)] = r.GetDouble(1);
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Rainfall tree empty for {Station}.", stationName);
        }
        return hourly;
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
    /// <paramref name="windowDays"/>. One point per day; the window ending at that
    /// day-end uses the prior <paramref name="windowDays"/>.
    /// </summary>
    internal static IReadOnlyList<SitePages.RollingMaePoint> ComputeRollingMae(
        IReadOnlyList<PredictionRow> predictions,
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

            // Emit one rolling-MAE point per calendar day end.
            for (var d = minDate.AddDays(windowDays - 1); d <= maxDate; d = d.AddDays(1))
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

    private IReadOnlyList<PredictionRow> QueryPredictions(DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Scope to the temperature subtree only — sibling subtrees (precipitation,
        // dry_window) have different schemas and would surface as nulls or wrong
        // columns under union_by_name.
        var glob = Path.Combine(_cfg.Storage.PredictionsPath, "temperature", "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        var sql = $@"
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendTemperature,
       TempGfs, TempEcmwf, TempIcon, TempMf, TempUkmo, TempGem,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem,
       TempMean, TempStd, TempRange,
       FeatureVectorHash
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY PredictionMadeAtUtc DESC, LeadHours";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<PredictionRow>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add(new PredictionRow
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
                    RunTimeGfs   = NullableDate(r, 12),
                    RunTimeEcmwf = NullableDate(r, 13),
                    RunTimeIcon  = NullableDate(r, 14),
                    RunTimeMf    = NullableDate(r, 15),
                    RunTimeUkmo  = NullableDate(r, 16),
                    RunTimeGem   = NullableDate(r, 17),
                    TempMean  = NullableDouble(r, 18),
                    TempStd   = NullableDouble(r, 19),
                    TempRange = NullableDouble(r, 20),
                    FeatureVectorHash = r.IsDBNull(21) ? "" : r.GetString(21),
                });
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Predictions tree empty — rendering an empty-state site.");
            return Array.Empty<PredictionRow>();
        }
        return rows;
    }

    private IReadOnlyList<SitePages.PrecipForecastPoint> QueryPrecipPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = Path.Combine(_cfg.Storage.PredictionsPath, "precipitation", "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        var sql = $@"
SELECT TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, LeadHours, ValidTimeUtc";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<SitePages.PrecipForecastPoint>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add(new SitePages.PrecipForecastPoint(
                    Station:        r.GetString(0),
                    Version:        r.GetString(1),
                    PredictedAtUtc: r.GetDateTime(2),
                    ValidTimeUtc:   r.GetDateTime(3),
                    LeadHours:      r.GetInt32(4),
                    ProbWet:        r.GetDouble(5),
                    ClimatologyPWet: r.GetDouble(6)));
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Precipitation predictions tree empty — precip page will render an empty state.");
            return Array.Empty<SitePages.PrecipForecastPoint>();
        }
        return rows;
    }

    private IReadOnlyList<SitePages.DryWindowForecastPoint> QueryDryWindowPredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = Path.Combine(_cfg.Storage.PredictionsPath, "dry_window", "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

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

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<SitePages.DryWindowForecastPoint>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add(new SitePages.DryWindowForecastPoint(
                    Station:                     r.GetString(0),
                    WindowHours:                 r.GetInt32(1),
                    Version:                     r.GetString(2),
                    PredictedAtUtc:              r.GetDateTime(3),
                    TargetDateUtc:               r.GetDateTime(4),
                    LeadHours:                   r.GetInt32(5),
                    ProbHasDryWindow:            r.GetDouble(6),
                    ClimatologyProbHasDryWindow: r.GetDouble(7),
                    AgreementHasDryWindow:       r.IsDBNull(8) ? null : r.GetDouble(8)));
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Dry-window predictions tree empty — dry-window page will render an empty state.");
            return Array.Empty<SitePages.DryWindowForecastPoint>();
        }
        return rows;
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

        var glob = Path.Combine(_cfg.Storage.ObservationsPath, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

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

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<(DateTime, double)>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add((r.GetDateTime(0), r.GetDouble(1)));
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("METAR tree empty — METAR chart series will be empty.");
        }
        return rows;
    }

    private IReadOnlyDictionary<DateTime, double> QueryTruth(DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var glob = Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        var sql = $@"
SELECT ValidTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND Temperature2m IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var truth = new Dictionary<DateTime, double>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                truth[r.GetDateTime(0)] = r.GetDouble(1);
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("ERA5 tree empty — skill charts will be empty.");
        }
        return truth;
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static DateTime? NullableDate(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
