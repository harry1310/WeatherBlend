using System.Data;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Weekly rolling dry-window verification. Reads <see cref="DryWindowPredictionRow"/>
/// parquet + EA rainfall truth via DuckDB, aggregates truth into hourly bins with
/// the training 4-of-4 gate, derives daily dry-window labels, and scores each
/// (station, window, version, lead) slice against blend/mean-of-models/climatology.
///
/// Drift flag fires when rolling blend Brier exceeds
/// <paramref name="driftThreshold"/>× the per-lead training Brier recorded in
/// <see cref="ModelArtifact.TrainingMetadata"/>. Defaults: 30-day window, 5-day
/// latency, 1.5× drift — matching the precip verify semantics.
/// </summary>
public sealed class DryWindowVerifyCommand
{
    private readonly ILogger<DryWindowVerifyCommand> _log;
    private readonly AppConfig _cfg;

    public DryWindowVerifyCommand(ILogger<DryWindowVerifyCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(
        string stationArg,
        string windowArg,
        DateOnly? asOf,
        int windowDays,
        int latencyDays,
        double driftThreshold,
        CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var manifestPath = Path.Combine(modelsRoot, "dry_window", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            _log.LogError("No dry_window manifest at {Path}. Train first.", manifestPath);
            return 2;
        }
        var manifest = System.Text.Json.JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(manifestPath))
            ?? throw new InvalidOperationException("Failed to parse dry_window manifest.");

        var entries = FilterEntries(manifest, stationArg, windowArg);
        if (entries.Count == 0)
        {
            _log.LogError("No manifest entries match station='{S}' window='{W}'.", stationArg, windowArg);
            return 2;
        }

        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-latencyDays);

        _log.LogInformation(
            "DryWindowVerify: as-of {AsOf:yyyy-MM-dd}Z, window [{S:yyyy-MM-dd}..{E:yyyy-MM-dd}], latency {L}d, drift {D:0.00}×",
            asOfUtc, windowStart, windowEnd, latencyDays, driftThreshold);

        // Fetch predictions from each (station-slug, window) subtree under the prediction root.
        var predictions = QueryPredictions(entries, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);
        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — nothing to verify.");
            return 0;
        }

        // Per-station dry-window truth, covering every (window, date) we might score.
        var windowsUsed = predictions.Select(p => p.WindowHours).Distinct().ToArray();
        var truthByStationWindow = QueryTruthLabels(
            entries.Select(kv => kv.Key).Select(k => ParseSlug(k)).Where(s => s is not null).Cast<string>().Distinct().ToArray(),
            windowsUsed, windowStart.AddDays(-2), windowEnd.AddDays(2), ct);

        // Group predictions by (station, window, version, lead) and score.
        var grouped = predictions
            .GroupBy(p => (p.TruthStation, p.WindowHours, p.ModelVersion, p.LeadHours))
            .OrderBy(g => g.Key.TruthStation, StringComparer.Ordinal)
            .ThenBy(g => g.Key.WindowHours)
            .ThenBy(g => g.Key.LeadHours);

        var resultRows = new List<VerifyRow>();
        foreach (var g in grouped)
        {
            ct.ThrowIfCancellationRequested();
            var key = (g.Key.TruthStation, g.Key.WindowHours);
            if (!truthByStationWindow.TryGetValue(key, out var truthByDate))
            {
                _log.LogWarning("No truth for station '{S}' window {W}h — skipping.", g.Key.TruthStation, g.Key.WindowHours);
                continue;
            }

            var paired = g
                .Select(p =>
                {
                    var d = DateOnly.FromDateTime(p.TargetDateUtc);
                    return (Pred: p, HasTruth: truthByDate.TryGetValue(d, out var t), Truth: t);
                })
                .Where(x => x.HasTruth)
                .ToList();
            if (paired.Count == 0) continue;

            var prob = paired.Select(x => x.Pred.ProbHasDryWindow).ToArray();
            var clim = paired.Select(x => x.Pred.ClimatologyProbHasDryWindow).ToArray();
            var y    = paired.Select(x => x.Truth ? 1.0 : 0.0).ToArray();
            var mean = paired.Select(x => MeanOfModels(x.Pred)).ToArray();
            var binary = prob.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();

            var blendBrier = PrecipMetrics.Brier(prob, y);
            var climBrier  = PrecipMetrics.Brier(clim, y);
            var meanBrier  = PrecipMetrics.Brier(mean.Where(v => !double.IsNaN(v)).ToArray(),
                                                 mean.Zip(y).Where(t => !double.IsNaN(t.First)).Select(t => t.Second).ToArray());
            var bss = PrecipMetrics.BrierSkillScore(blendBrier, climBrier);
            var fbias = PrecipMetrics.FrequencyBias(binary, y);

            // Drift baseline: blend Brier on the training test partition, stored
            // in PerLeadStats.BlendTestMae (reused field — see training metadata).
            // Phase tag from the same metadata so the headline can group 3b vs 3d-shape.
            double? trainingBrier = null;
            string phase = "";
            try
            {
                var versionDir = ModelArtifact.ResolveStationVersionDir(
                    modelsRoot, "dry_window",
                    $"{g.Key.TruthStation}/window_{g.Key.WindowHours}h",
                    g.Key.ModelVersion);
                var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
                if (metadata.PerLead.TryGetValue(g.Key.LeadHours.ToString(CultureInfo.InvariantCulture), out var pl))
                    trainingBrier = pl.BlendTestMae;
                phase = metadata.Phase ?? "";
            }
            catch (Exception ex)
            {
                _log.LogWarning("Metadata load failed for {Key}/window_{W}h v{V}: {Msg}",
                    g.Key.TruthStation, g.Key.WindowHours, g.Key.ModelVersion, ex.Message);
            }

            var drift = trainingBrier.HasValue && trainingBrier.Value > 0
                && blendBrier > trainingBrier.Value * driftThreshold;

            resultRows.Add(new VerifyRow(
                g.Key.TruthStation, g.Key.WindowHours, g.Key.ModelVersion, phase, g.Key.LeadHours,
                paired.Count, paired.Count(x => x.Truth),
                blendBrier, climBrier, meanBrier, bss, fbias, trainingBrier, drift));
        }

        var md = BuildMarkdown(asOfUtc, windowDays, latencyDays, driftThreshold, resultRows);
        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath, $"verify_dry_window_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        foreach (var r in resultRows)
        {
            _log.LogInformation(
                "{S} / window_{W}h / {V} lead {L}h — n={N} (pos={P}), Brier {B:0.0000} (clim {C:0.0000}, BSS {Bss:+0.000;-0.000;0.000}){Drift}",
                r.Station, r.WindowHours, r.Version, r.LeadHours, r.N, r.Positives,
                r.BlendBrier, r.ClimBrier, r.Bss, r.Drift ? "  [DRIFT]" : "");
        }

        return resultRows.Any(r => r.Drift) ? 4 : 0;
    }

    private sealed record VerifyRow(
        string Station, int WindowHours, string Version, string Phase, int LeadHours,
        int N, int Positives,
        double BlendBrier, double ClimBrier, double MeanBrier,
        double Bss, double FreqBias,
        double? TrainingBrier, bool Drift);

    private List<KeyValuePair<string, ModelArtifact.StationEntry>> FilterEntries(
        ModelArtifact.Manifest manifest, string stationArg, string windowArg)
    {
        var result = new List<KeyValuePair<string, ModelArtifact.StationEntry>>();
        foreach (var kv in manifest.Stations)
        {
            var m = Regex.Match(kv.Key, @"^(?<slug>[^/]+)/window_(?<w>\d+)h$");
            if (!m.Success) continue;
            var slug = m.Groups["slug"].Value;
            var window = int.Parse(m.Groups["w"].Value, CultureInfo.InvariantCulture);

            if (!string.Equals(stationArg, "all", StringComparison.OrdinalIgnoreCase))
            {
                var stripped = slug.StartsWith("ea_") ? slug[3..] : slug;
                if (!slug.Equals(stationArg, StringComparison.OrdinalIgnoreCase)
                    && !stripped.Equals(stationArg, StringComparison.OrdinalIgnoreCase)
                    && !slug.Equals("ea_" + Slugify(stationArg), StringComparison.OrdinalIgnoreCase))
                    continue;
            }
            if (!string.Equals(windowArg, "all", StringComparison.OrdinalIgnoreCase))
            {
                if (!int.TryParse(windowArg, out var w) || w != window) continue;
            }
            result.Add(kv);
        }
        return result;
    }

    private static string? ParseSlug(string compositeKey)
    {
        var m = Regex.Match(compositeKey, @"^(?<slug>[^/]+)/window_\d+h$");
        return m.Success ? m.Groups["slug"].Value : null;
    }

    private IReadOnlyList<DryWindowPredictionRow> QueryPredictions(
        IReadOnlyList<KeyValuePair<string, ModelArtifact.StationEntry>> entries,
        DateTime windowStart, DateTime windowEnd, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var globs = entries
            .Select(kv =>
            {
                var slug = ParseSlug(kv.Key)!;
                var window = int.Parse(kv.Key.Split("/window_")[1].TrimEnd('h'), CultureInfo.InvariantCulture);
                return "'" + ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "dry_window", slug,
                    $"window_{window}h", "**", "*.parquet")) + "'";
            })
            .ToArray();
        if (globs.Length == 0) return Array.Empty<DryWindowPredictionRow>();

        var sql = $@"
SELECT LocationName, TruthStation, WindowHours, ModelVersion,
       PredictionMadeAtUtc, TargetDateUtc, LeadHours,
       ProbHasDryWindow, ClimatologyProbHasDryWindow,
       AgreementHasDryWindow, PrecipSumMean, LongestDryRunMean, WetHourCountMean,
       HasDryWindowGfs, HasDryWindowEcmwf, HasDryWindowIcon,
       HasDryWindowMf,  HasDryWindowUkmo,  HasDryWindowGem,  HasDryWindowAifs,  HasDryWindowJma,
       PrecipSumGfs, PrecipSumEcmwf, PrecipSumIcon, PrecipSumMf, PrecipSumUkmo, PrecipSumGem,
       PrecipSumAifs, PrecipSumJma,
       FeatureVectorHash
FROM read_parquet([{string.Join(", ", globs)}], hive_partitioning = false, union_by_name = true)
WHERE TargetDateUtc >= TIMESTAMP '{windowStart:yyyy-MM-dd HH:mm:ss}'
  AND TargetDateUtc <= TIMESTAMP '{windowEnd:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, WindowHours, ModelVersion, LeadHours, TargetDateUtc";

        // ParquetReader.Query catches DuckDB's "No files found" signal — note
        // that the list-form throws "IO Error: No files found that match the
        // pattern ..." (verified in DuckDbReadParquetProbeTests), so the
        // standard substring match handles it. If ANY (station, window)
        // glob is missing, the entire query fails — keeping today's
        // "missing → empty result" semantics is intentional.
        return ParquetReader.Query(sql, r => new DryWindowPredictionRow
        {
            LocationName = r.GetString(0),
            TruthStation = r.GetString(1),
            WindowHours  = r.GetInt32(2),
            ModelVersion = r.GetString(3),
            PredictionMadeAtUtc = r.GetDateTime(4),
            TargetDateUtc       = r.GetDateTime(5),
            LeadHours           = r.GetInt32(6),
            ProbHasDryWindow    = r.GetDouble(7),
            ClimatologyProbHasDryWindow = r.GetDouble(8),
            AgreementHasDryWindow = Nullable(r,  9),
            PrecipSumMean         = Nullable(r, 10),
            LongestDryRunMean     = Nullable(r, 11),
            WetHourCountMean      = Nullable(r, 12),
            HasDryWindowGfs   = Nullable(r, 13),
            HasDryWindowEcmwf = Nullable(r, 14),
            HasDryWindowIcon  = Nullable(r, 15),
            HasDryWindowMf    = Nullable(r, 16),
            HasDryWindowUkmo  = Nullable(r, 17),
            HasDryWindowGem   = Nullable(r, 18),
            HasDryWindowAifs  = Nullable(r, 19),
            HasDryWindowJma   = Nullable(r, 20),
            PrecipSumGfs   = Nullable(r, 21),
            PrecipSumEcmwf = Nullable(r, 22),
            PrecipSumIcon  = Nullable(r, 23),
            PrecipSumMf    = Nullable(r, 24),
            PrecipSumUkmo  = Nullable(r, 25),
            PrecipSumGem   = Nullable(r, 26),
            PrecipSumAifs  = Nullable(r, 27),
            PrecipSumJma   = Nullable(r, 28),
            FeatureVectorHash = r.IsDBNull(29) ? "" : r.GetString(29),
        }, _log, "Prediction tree empty for requested (station, window) — nothing to verify.", ct);
    }

    private IReadOnlyDictionary<(string Station, int WindowHours), IReadOnlyDictionary<DateOnly, bool>>
        QueryTruthLabels(string[] stationSlugs, int[] windowLengths, DateTime start, DateTime end, CancellationToken ct)
    {
        var slugToName = _cfg.Location.Rainfall.Stations.ToDictionary(
            s => "ea_" + Slugify(s.Name),
            s => s.Name,
            StringComparer.OrdinalIgnoreCase);

        var result = new Dictionary<(string, int), IReadOnlyDictionary<DateOnly, bool>>();
        foreach (var slug in stationSlugs)
        {
            if (!slugToName.TryGetValue(slug, out var stationName))
            {
                _log.LogWarning("Slug '{Slug}' doesn't map to a configured rainfall station.", slug);
                continue;
            }
            var hourly = QueryHourlyTruth(stationName, start, end, ct);
            if (hourly.Count == 0) continue;

            var labels = DryWindowLabelBuilder.Build(hourly, windowLengths, _cfg.DryWindow.BuildDaytimeWindow());
            foreach (var w in windowLengths)
            {
                var byDate = labels.Labels
                    .Where(l => l.WindowHours == w)
                    .ToDictionary(l => l.Date, l => l.HasDryWindow);
                result[(slug, w)] = byDate;
            }
        }
        return result;
    }

    private IReadOnlyList<DryWindowLabelBuilder.HourlyTruth> QueryHourlyTruth(
        string stationName, DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet"));

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

        return ParquetReader.Query(sql, r =>
            new DryWindowLabelBuilder.HourlyTruth(
                DateTime.SpecifyKind(r.GetDateTime(0), DateTimeKind.Utc),
                r.GetDouble(1)),
            _log, $"No rainfall parquet for station {stationName}.", ct);
    }

    private static double MeanOfModels(DryWindowPredictionRow p)
    {
        double sum = 0; int n = 0;
        void Add(double? v) { if (v.HasValue) { sum += v.Value; n++; } }
        Add(p.HasDryWindowGfs); Add(p.HasDryWindowEcmwf); Add(p.HasDryWindowIcon);
        Add(p.HasDryWindowMf);  Add(p.HasDryWindowUkmo);  Add(p.HasDryWindowGem);
        Add(p.HasDryWindowAifs); Add(p.HasDryWindowJma);
        return n == 0 ? double.NaN : sum / n;
    }

    private static double? Nullable(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static string Slugify(string s) => s.ToLowerInvariant()
        .Replace(' ', '_').Replace('-', '_').Replace(',', '_');

    private static string BuildMarkdown(DateTime asOfUtc, int windowDays, int latencyDays, double driftThreshold,
        IReadOnlyList<VerifyRow> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Dry-window verification — {asOfUtc:yyyy-MM-dd}");
        sb.AppendLine();
        sb.AppendLine($"*As-of {asOfUtc:yyyy-MM-dd HH:mm}Z · window {windowDays}d · latency {latencyDays}d · drift threshold {driftThreshold:0.00}×.*");
        sb.AppendLine();

        if (rows.Count == 0)
        {
            sb.AppendLine("No (station, window, version, lead) slices had paired predictions + truth in the window.");
            return sb.ToString();
        }

        AppendPhaseComparison(sb, rows);

        foreach (var stationGroup in rows.GroupBy(r => r.Station).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            sb.AppendLine($"## {stationGroup.Key}");
            sb.AppendLine();
            foreach (var winGroup in stationGroup.GroupBy(r => r.WindowHours).OrderBy(g => g.Key))
            {
                sb.AppendLine($"### Window {winGroup.Key}h");
                sb.AppendLine();
                sb.AppendLine("| Version | Phase | Lead | n | pos | Blend Brier | Clim Brier | Mean-of-models Brier | BSS | Freq bias | Training Brier | Drift |");
                sb.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|");
                foreach (var r in winGroup.OrderBy(r => r.Version).ThenBy(r => r.LeadHours))
                {
                    sb.AppendLine($"| `{r.Version}` | {(string.IsNullOrEmpty(r.Phase) ? "—" : r.Phase)} | {r.LeadHours}h | {r.N} | {r.Positives} | "
                        + $"{r.BlendBrier:0.0000} | {r.ClimBrier:0.0000} | "
                        + $"{(double.IsNaN(r.MeanBrier) ? "—" : r.MeanBrier.ToString("0.0000"))} | "
                        + $"{r.Bss:+0.0000;-0.0000;0.0000} | "
                        + $"{(double.IsNaN(r.FreqBias) ? "—" : r.FreqBias.ToString("0.00"))} | "
                        + $"{(r.TrainingBrier.HasValue ? r.TrainingBrier.Value.ToString("0.0000") : "—")} | "
                        + $"{(r.Drift ? "**⚠ yes**" : "ok")} |");
                }
                sb.AppendLine();
            }
        }

        var drifting = rows.Count(r => r.Drift);
        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine($"- Slices scored: **{rows.Count}**");
        sb.AppendLine($"- Drifting slices: **{drifting}** (threshold = rolling Brier > {driftThreshold:0.00}× training Brier)");
        sb.AppendLine();

        return sb.ToString();
    }

    /// <summary>
    /// Side-by-side Brier comparison across phases (3b vs 3d-shape) for every
    /// (station, window, lead) slice that has more than one phase represented
    /// in the verification window. Champion/challenger pattern: only phases
    /// present at the slice are shown.
    /// </summary>
    private static void AppendPhaseComparison(StringBuilder sb, IReadOnlyList<VerifyRow> rows)
    {
        var multiPhase = rows
            .Where(r => !string.IsNullOrEmpty(r.Phase))
            .GroupBy(r => (r.Station, r.WindowHours, r.LeadHours))
            .Where(g => g.Select(r => r.Phase).Distinct(StringComparer.Ordinal).Count() > 1)
            .ToList();
        if (multiPhase.Count == 0) return;

        var phases = multiPhase
            .SelectMany(g => g.Select(r => r.Phase))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToList();

        sb.AppendLine("## Phase comparison");
        sb.AppendLine();
        sb.Append("| Station | Window | Lead | n |");
        foreach (var p in phases) sb.Append($" {p} Brier | {p} BSS |");
        sb.AppendLine();
        sb.Append("|---|---:|---:|---:|");
        foreach (var _ in phases) sb.Append("---:|---:|");
        sb.AppendLine();

        foreach (var g in multiPhase
            .OrderBy(g => g.Key.Station, StringComparer.Ordinal)
            .ThenBy(g => g.Key.WindowHours)
            .ThenBy(g => g.Key.LeadHours))
        {
            // n may differ across phases when a model is missing predictions on
            // some target dates — show the median so the headline stays sane.
            var nMedian = (int)g.Select(r => (double)r.N).OrderBy(v => v).Skip(g.Count() / 2).First();
            sb.Append($"| {g.Key.Station} | {g.Key.WindowHours}h | {g.Key.LeadHours}h | ~{nMedian} |");
            foreach (var p in phases)
            {
                var row = g.FirstOrDefault(r => string.Equals(r.Phase, p, StringComparison.Ordinal));
                if (row is null) sb.Append(" — | — |");
                else sb.Append($" {row.BlendBrier:0.0000} | {row.Bss:+0.0000;-0.0000;0.0000} |");
            }
            sb.AppendLine();
        }
        sb.AppendLine();
    }
}
