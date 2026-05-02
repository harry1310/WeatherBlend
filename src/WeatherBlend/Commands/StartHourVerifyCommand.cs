using System.Globalization;
using System.Text;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.StartHour;
using WeatherBlend.Models;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Weekly rolling verification for the dry-window start-hour curve. Reads
/// the start-hour predict tree + EA rainfall truth, derives the truth
/// start-hour set per (station, window, target_date), and aggregates
/// per-(station, window, lead) skill metrics:
///
///   - top-1 accuracy: argmax_s π_s ∈ truth_starts
///   - Brier (mean): Σ (π_s − τ_s)² averaged across informative days
///   - log-loss skill: 1 − ll(π) / ll(uniform), positive = beats uniform
///
/// "Informative" = some-but-not-all candidate starts were valid in truth;
/// fully-dry and no-block days are excluded because the curve has no shape
/// signal to score on those.
///
/// Drift flag isn't wired here: the curve is a derivation, not a model,
/// so there's no training-time skill to compare against. A negative
/// log-loss skill score is the "should I worry" signal — flagged in the
/// markdown report when present.
/// </summary>
public sealed class StartHourVerifyCommand
{
    private readonly ILogger<StartHourVerifyCommand> _log;
    private readonly AppConfig _cfg;

    public StartHourVerifyCommand(ILogger<StartHourVerifyCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(
        DateOnly? asOf,
        int windowDays,
        int latencyDays,
        CancellationToken ct)
    {
        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd = asOfUtc.AddDays(-latencyDays);

        _log.LogInformation(
            "StartHourVerify: as-of {AsOf:yyyy-MM-dd}Z, window [{S:yyyy-MM-dd}..{E:yyyy-MM-dd}], latency {L}d",
            asOfUtc, windowStart, windowEnd, latencyDays);

        var curveRows = QueryCurves(windowStart, windowEnd, ct);
        if (curveRows.Count == 0)
        {
            _log.LogWarning("No start-hour curves in window — nothing to verify.");
            return 0;
        }
        _log.LogInformation("Loaded {N} curve rows.", curveRows.Count);

        // Hourly truth keyed by (station-friendly-name, target_date_utc, hour-of-day).
        var rainfall = QueryHourlyRainfall(windowStart.AddDays(-1), windowEnd.AddDays(1), ct);
        _log.LogInformation("Loaded {N} hourly rainfall rows across {S} stations.",
            rainfall.Sum(kv => kv.Value.Count), rainfall.Count);

        var daytime = _cfg.DryWindow.BuildDaytimeWindow();

        // Group curves by (station, window, lead, target_date). For each
        // group: derive truth_starts from rainfall, score the curve, accumulate.
        var byCell = curveRows
            .GroupBy(r => (r.Station, r.WindowHours, r.LeadHours, r.TargetDateUtc));

        var aggregator = new Dictionary<(string Station, int Window, int Lead), Aggregator>();
        int dropped_partial = 0, dropped_uninformative = 0, scored = 0;

        foreach (var group in byCell)
        {
            ct.ThrowIfCancellationRequested();
            var (station, window, lead, targetDate) = group.Key;
            var (startUtc, endUtc) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(targetDate));

            var stationFriendly = ResolveFriendlyStationName(station);
            if (stationFriendly is null
                || !rainfall.TryGetValue(stationFriendly, out var rainByDate)
                || !rainByDate.TryGetValue(targetDate.Date, out var hourlyMm))
            {
                dropped_partial++;
                continue;
            }

            var truthStarts = StartHourTruth.ValidStartsFor(hourlyMm, startUtc, endUtc, window);
            if (truthStarts is null) { dropped_partial++; continue; }

            var totalStarts = endUtc - startUtc - window + 1;
            if (totalStarts <= 0) { dropped_partial++; continue; }
            if (!StartHourMetrics.IsInformative(truthStarts.Count, totalStarts))
            {
                dropped_uninformative++;
                continue;
            }

            // Latest PMT per StartHour for this cell — multiple cycles could
            // have written competing rows; freshest forecast wins.
            var curve = group
                .GroupBy(r => r.StartHourUtc)
                .Select(g => g.OrderByDescending(r => r.PredictionMadeAtUtc).First())
                .OrderBy(r => r.StartHourUtc)
                .Select(r => (r.StartHourUtc, r.ConditionalProb))
                .ToList();

            var top1 = StartHourMetrics.Top1Hit(curve, truthStarts) ? 1 : 0;
            var brier = StartHourMetrics.Brier(curve, truthStarts);
            var ll = StartHourMetrics.LogLoss(curve, truthStarts);
            var llU = StartHourMetrics.LogLossUniform(totalStarts, truthStarts.Count);

            var aggKey = (station, window, lead);
            if (!aggregator.TryGetValue(aggKey, out var agg))
            {
                agg = new Aggregator();
                aggregator[aggKey] = agg;
            }
            agg.Add(top1, brier, ll, llU);
            scored++;
        }

        _log.LogInformation(
            "Scored {S} (station, window, lead, day) cells. Dropped {P} for partial / unobservable truth, {U} as uninformative.",
            scored, dropped_partial, dropped_uninformative);

        var md = BuildMarkdown(aggregator, asOfUtc, windowDays, latencyDays);
        var reportPath = Path.Combine(
            _cfg.Storage.ReportsPath,
            $"verify_start_hour_{asOfUtc:yyyy-MM-dd}.md");
        Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
        await File.WriteAllTextAsync(reportPath, md, ct);
        _log.LogInformation("Wrote report → {Path}", reportPath);

        Console.Write(md);
        return 0;
    }

    private sealed class Aggregator
    {
        public int Days;
        public int Top1Hits;
        public double BrierSum;
        public double LogLossSum;
        public double LogLossUniformSum;

        public void Add(int top1, double brier, double ll, double llU)
        {
            Days++;
            Top1Hits += top1;
            BrierSum += brier;
            LogLossSum += ll;
            LogLossUniformSum += llU;
        }

        public double Top1Rate => Days == 0 ? 0 : (double)Top1Hits / Days;
        public double MeanBrier => Days == 0 ? 0 : BrierSum / Days;
        public double Skill => LogLossUniformSum > 0
            ? 1 - LogLossSum / LogLossUniformSum
            : 0;
    }

    private static string BuildMarkdown(
        Dictionary<(string Station, int Window, int Lead), Aggregator> agg,
        DateTime asOfUtc, int windowDays, int latencyDays)
    {
        var ci = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.AppendLine($"# Start-hour curve verification — {asOfUtc:yyyy-MM-dd}Z");
        sb.AppendLine();
        sb.AppendLine($"Rolling window {windowDays}d ending {asOfUtc.AddDays(-latencyDays):yyyy-MM-dd}Z; latency {latencyDays}d.");
        sb.AppendLine();
        sb.AppendLine("Skill score = 1 − log-loss(curve) / log-loss(uniform). Positive = curve beats uniform on informative days. Top-1 = % of informative days where the argmax start hour was actually a valid dry start in EA truth.");
        sb.AppendLine();
        sb.AppendLine("| Station | Window | Lead | Days | Top-1 | Mean Brier | Skill |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|");
        foreach (var ((station, window, lead), a) in agg
            .OrderBy(kv => kv.Key.Station, StringComparer.Ordinal)
            .ThenBy(kv => kv.Key.Window)
            .ThenBy(kv => kv.Key.Lead))
        {
            var flag = a.Skill < 0 ? " ⚠️" : "";
            sb.AppendLine(ci, $"| {station} | {window}h | {lead}h | {a.Days} | {a.Top1Rate:P1} | {a.MeanBrier:F4} | {a.Skill:+0.000;-0.000;+0.000}{flag} |");
        }
        if (agg.Count == 0)
            sb.AppendLine("| — | — | — | 0 | — | — | — |");
        sb.AppendLine();
        return sb.ToString();
    }

    // ------------------------------------------------------------------------
    // I/O — DuckDB queries for the predict + truth trees. Same patterns as
    // the existing verify commands; bounded by the rolling window so the
    // partition scans stay cheap.
    // ------------------------------------------------------------------------

    private record CurveRow(string Station, int WindowHours, int LeadHours,
                            DateTime TargetDateUtc, int StartHourUtc,
                            double ConditionalProb, DateTime PredictionMadeAtUtc);

    private IReadOnlyList<CurveRow> QueryCurves(DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath,
            StartHourPredictCommand.PredictionsSubdir, "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "ConditionalProb"))
            return Array.Empty<CurveRow>();

        var sql = $@"
SELECT TruthStation, WindowHours, LeadHours, TargetDateUtc, StartHourUtc,
       ConditionalProb, PredictionMadeAtUtc
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND TargetDateUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND TargetDateUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'";

        return ParquetReader.Query(conn, sql, r => new CurveRow(
            Station:             r.GetString(0),
            WindowHours:         r.GetInt32(1),
            LeadHours:           r.GetInt32(2),
            TargetDateUtc:       r.GetDateTime(3),
            StartHourUtc:        r.GetInt32(4),
            ConditionalProb:     r.GetDouble(5),
            PredictionMadeAtUtc: r.GetDateTime(6)),
            _log, "Start-hour predict tree empty — verify reports zero rows.", ct);
    }

    /// <summary>Hourly EA rainfall keyed by station-friendly-name → date →
    /// hour → mm. 4-of-4 gate so partial hours don't quietly become "dry".</summary>
    private Dictionary<string, Dictionary<DateTime, Dictionary<int, double>>>
        QueryHourlyRainfall(DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet"));
        var sql = $@"
SELECT StationName,
       date_trunc('day', date_trunc('hour', ObservedTimeUtc))::DATE AS day,
       extract(hour from ObservedTimeUtc) AS h,
       SUM(Value15MinMm) AS mm
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND Value15MinMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end.AddHours(1):yyyy-MM-dd HH:mm:ss}'
GROUP BY 1, 2, 3
HAVING COUNT(*) = 4";

        var rows = ParquetReader.Query(sql, r => (
            Station: r.GetString(0),
            Day:     DateTime.SpecifyKind(r.GetDateTime(1), DateTimeKind.Utc),
            Hour:    Convert.ToInt32(r.GetValue(2)),
            Mm:      r.GetDouble(3)),
            _log, "Rainfall tree empty — verify cannot score.", ct);

        var result = new Dictionary<string, Dictionary<DateTime, Dictionary<int, double>>>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            if (!result.TryGetValue(row.Station, out var byDate))
            {
                byDate = new Dictionary<DateTime, Dictionary<int, double>>();
                result[row.Station] = byDate;
            }
            if (!byDate.TryGetValue(row.Day, out var byHour))
            {
                byHour = new Dictionary<int, double>();
                byDate[row.Day] = byHour;
            }
            byHour[row.Hour] = row.Mm;
        }
        return result;
    }

    /// <summary>Map "ea_bellever_dartmoor" → "Bellever Dartmoor" via the
    /// configured rainfall stations. Returns null if no match.</summary>
    private string? ResolveFriendlyStationName(string slug)
    {
        var bare = slug.StartsWith("ea_", StringComparison.Ordinal) ? slug[3..] : slug;
        foreach (var s in _cfg.Location.Rainfall.Stations)
            if (StationSlug.Of(s.Name).Equals(bare, StringComparison.Ordinal)) return s.Name;
        return null;
    }
}
