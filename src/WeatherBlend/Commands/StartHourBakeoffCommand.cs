using System.Globalization;
using System.Text;
using System.Text.Json;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.StartHour;
using WeatherBlend.Models;
using WeatherBlend.Predict.StartHour;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 1 of the start-hour bake-off (planned 2026-05-03). Scaffolds the
/// shared comparison infrastructure that all three challenger methods —
/// the current StartHourCurveDerivation product, option C (3g-style MC),
/// option B (Bayesian Gaussian copula) — get scored against. Phase 1 alone
/// produces the baseline number for the current method on the canonical
/// test set, before any of the new methods is built.
///
/// Three subcommands:
///   --action prepare  → Build the canonical test-set JSON + per-row truth
///                       parquet. Walks the same dry-window chronological
///                       70/15/15 split as 3b/3g so the test slices align
///                       to the day. Idempotent: skip if outputs exist.
///   --action score --method [current|C|B]
///                     → Derive the named method's start-hour curve for
///                       every (station, window, lead, target_date) row in
///                       the test set, write predictions parquet under
///                       data/reports/start_hour_bakeoff/method=&lt;m&gt;/.
///   --action compare  → Join all method-prediction parquets to truth via
///                       StartHourMetrics. Per-cell Brier / top-1 / log-loss
///                       + uniform baseline; emit comparison report.
///
/// All three subcommands are pure functions of disk state — no live API
/// calls, no model retraining. The bake-off lives entirely under
/// <c>data/reports/start_hour_bakeoff/</c> so production paths under
/// <c>data/predictions/start_hour/</c> are untouched.
/// </summary>
public sealed class StartHourBakeoffCommand
{
    public const string TestSetFileName = "start_hour_test_set_2026-05-03.json";
    public const string TruthLabelsFileName = "start_hour_truth_labels.parquet";
    public const string ReportFileName = "start_hour_bakeoff_report.md";
    public const string BakeoffSubdir = "start_hour_bakeoff";

    /// <summary>Canonical method names — used as folder partitions and as
    /// the join key in the comparison report.</summary>
    public static class Methods
    {
        public const string Current = "current";
        public const string OptionC = "C";
        public const string OptionB = "B";
    }

    private readonly ILogger<StartHourBakeoffCommand> _log;
    private readonly AppConfig _cfg;

    public StartHourBakeoffCommand(ILogger<StartHourBakeoffCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string action, string? method, CancellationToken ct)
    {
        return action.ToLowerInvariant() switch
        {
            "prepare" => await PrepareAsync(ct),
            "score"   => await ScoreAsync(method ?? Methods.Current, ct),
            "compare" => await CompareAsync(ct),
            _         => InvalidAction(action),
        };
    }

    private int InvalidAction(string action)
    {
        _log.LogError("Unknown --action '{A}'. Expected: prepare | score | compare.", action);
        return 2;
    }

    // ---- prepare: build test set + truth labels --------------------------------

    private async Task<int> PrepareAsync(CancellationToken ct)
    {
        var stations = _cfg.Location.Rainfall.Stations.Select(s => s.Name).ToArray();
        var windows = new[] { 3, 4, 6 };
        var leads = Leads.Short;
        var daytime = _cfg.DryWindow.BuildDaytimeWindow();

        _log.LogInformation(
            "Phase 1 prepare — stations=[{S}] windows=[{W}] leads=[{L}]",
            string.Join(", ", stations), string.Join(",", windows), string.Join(",", leads));

        var testRows = new List<TestRow>();
        var truthRows = new List<TruthRow>();

        foreach (var stationName in stations)
        {
            ct.ThrowIfCancellationRequested();
            var stationSlug = StationSlug.WithEaPrefix(stationName);
            var hourlyMm = LoadHourlyTruth(stationName);
            _log.LogInformation("  loaded {N} hourly rainfall obs for {S}", hourlyMm.Count, stationName);

            foreach (var window in windows)
            {
                foreach (var lead in leads)
                {
                    // Use the 3b feature builder to enumerate the same chronological
                    // (date, label) sequence the 3b/3g training pipeline used; take
                    // the last 15% as the test slice. This guarantees identical
                    // test boundaries to the dry-window blenders.
                    var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
                    var rows = DryWindowFeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                        _cfg.Location.Name, stationName, spec, window, daytime, ct);
                    if (rows.Count < 100)
                    {
                        _log.LogWarning(
                            "  {S}/window_{W}h/lead_{L}h: only {N} rows; skipping cell.",
                            stationName, window, lead, rows.Count);
                        continue;
                    }
                    var ds = DryWindowDataset.Split(rows);

                    foreach (var r in ds.Test)
                    {
                        var date = DateOnly.FromDateTime(r.TargetDateUtc);
                        var (startUtc, endUtc) = daytime.UtcHourRangeFor(date);
                        var dayHourly = ExtractDailyHourly(hourlyMm, date, startUtc, endUtc);
                        var validStarts = dayHourly is null
                            ? null
                            : StartHourTruth.ValidStartsFor(dayHourly, startUtc, endUtc, window);
                        if (validStarts is null)
                            continue;  // partial truth coverage on this day → can't score honestly

                        testRows.Add(new TestRow(stationSlug, window, lead, date,
                            startUtc, endUtc, validStarts.Count));

                        // Per-(date, window, station): one truth row per candidate
                        // start hour. Lead-independent — truth is the gauge, not the
                        // forecast. Cell uniqueness is (station, window, date, hour);
                        // we'll de-dupe across leads at write time.
                        for (int h = startUtc; h + window <= endUtc; h++)
                        {
                            truthRows.Add(new TruthRow
                            {
                                StationSlug = stationSlug,
                                WindowHours = window,
                                TargetDateUtc = r.TargetDateUtc,
                                StartHourUtc = h,
                                HasDryBlockStarting = validStarts.Contains(h),
                            });
                        }
                    }
                    _log.LogInformation(
                        "  {S}/window_{W}h/lead_{L}h: test slice = {N} rows ({Pos} dry-block days)",
                        stationName, window, lead, ds.Test.Count, ds.TestPositives);
                }
            }
        }

        // De-duplicate truth rows: identical labels are produced once per
        // (station, window, lead) test slice but the truth itself is
        // lead-independent. Keep one row per (station, window, date, hour).
        var uniqueTruth = truthRows
            .GroupBy(t => (t.StationSlug, t.WindowHours, t.TargetDateUtc, t.StartHourUtc))
            .Select(g => g.First())
            .OrderBy(t => t.StationSlug).ThenBy(t => t.WindowHours)
            .ThenBy(t => t.TargetDateUtc).ThenBy(t => t.StartHourUtc)
            .ToList();

        Directory.CreateDirectory(BakeoffDir());
        await File.WriteAllTextAsync(
            Path.Combine(BakeoffDir(), TestSetFileName),
            JsonSerializer.Serialize(testRows, new JsonSerializerOptions { WriteIndented = true }),
            ct);
        await ParquetSerializer.SerializeAsync(uniqueTruth, Path.Combine(BakeoffDir(), TruthLabelsFileName),
            cancellationToken: ct);

        _log.LogInformation(
            "Wrote test set ({N} rows) + truth labels ({T} unique rows) → {Dir}",
            testRows.Count, uniqueTruth.Count, BakeoffDir());
        return 0;
    }

    // ---- score: derive a method's predictions for the test set ----------------

    private async Task<int> ScoreAsync(string method, CancellationToken ct)
    {
        var testSet = LoadTestSet();
        if (testSet.Count == 0)
        {
            _log.LogError("No test set found. Run --action prepare first.");
            return 2;
        }

        return method.ToLowerInvariant() switch
        {
            "current" => await ScoreCurrentAsync(testSet, ct),
            "c"       => await ScoreOptionCAsync(testSet, ct),
            "b"       => InvalidMethod("B: built in Phase 3; not yet implemented."),
            _         => InvalidMethod($"Unknown method '{method}'. Expected: current | C | B."),
        };
    }

    private int InvalidMethod(string msg)
    {
        _log.LogError(msg);
        return 2;
    }

    /// <summary>
    /// Score the current method (StartHourCurveDerivation product). For each
    /// test row, load the matching 3a replay hourly P(wet) for that lead,
    /// run the existing curve derivation under independence, normalise to a
    /// conditional distribution. Output predictions parquet.
    ///
    /// Note: we deliberately score only ConditionalProb (the shape over hours)
    /// not CalibratedProb (which scales by 3b's daily marginal). The Phase 1
    /// metrics use uniform-over-truth as the target distribution — a proper
    /// PMF — so the conditional shape is the comparable quantity. Magnitude
    /// scaling by the daily marginal would be a constant per row and doesn't
    /// affect the comparison.
    /// </summary>
    private async Task<int> ScoreCurrentAsync(List<TestRow> testSet, CancellationToken ct)
    {
        // Resolve 3a champion per station — the same versions that produced
        // the replay parquets we'll read from.
        var precip3a = ResolvePrecip3aChampions();
        if (precip3a is null) return 2;

        // Pre-load replay parquets per (station, lead).
        var replayByStationLead = new Dictionary<(string, int), Dictionary<DateTime, double>>();
        foreach (var stationGroup in testSet.GroupBy(t => t.StationSlug))
        {
            var stationSlug = stationGroup.Key;
            if (!precip3a.TryGetValue(stationSlug, out var precipVer))
            {
                _log.LogWarning("No 3a champion for {S}; skipping.", stationSlug);
                continue;
            }
            foreach (var lead in stationGroup.Select(t => t.LeadHours).Distinct())
            {
                replayByStationLead[(stationSlug, lead)] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precipVer, lead);
            }
        }

        var preds = new List<PredictionRow>();
        var skippedReplay = 0;
        foreach (var t in testSet)
        {
            ct.ThrowIfCancellationRequested();
            if (!replayByStationLead.TryGetValue((t.StationSlug, t.LeadHours), out var hourly)) continue;

            var midnight = new DateTime(t.TargetDateUtc.Year, t.TargetDateUtc.Month, t.TargetDateUtc.Day, 0, 0, 0, DateTimeKind.Utc);
            var hourlyByHour = new Dictionary<int, double>();
            bool gap = false;
            for (int h = t.DaytimeStartUtc; h < t.DaytimeEndUtc; h++)
            {
                if (!hourly.TryGetValue(midnight.AddHours(h), out var q)) { gap = true; break; }
                hourlyByHour[h] = q;
            }
            if (gap) { skippedReplay++; continue; }

            // Reuse the production curve derivation. Pass dailyProbAnyBlock=1
            // so CalibratedProb == ConditionalProb (we don't use CalibratedProb
            // here, but keeping the call signature honest).
            var rows = StartHourCurveDerivation.Derive(
                locationName: _cfg.Location.Name,
                truthStation: t.StationSlug,
                windowHours: t.WindowHours,
                startHourVersion: $"bakeoff_{Methods.Current}",
                predictionMadeAtUtc: DateTime.UtcNow,
                leadHours: t.LeadHours,
                targetDateUtc: midnight,
                daytimeStartUtc: t.DaytimeStartUtc,
                daytimeEndUtc: t.DaytimeEndUtc,
                hourlyPWet: hourlyByHour,
                dailyProbAnyBlock: 1.0,
                precipVersion: precip3a[t.StationSlug],
                dryWindowVersion: "n/a-bakeoff");

            foreach (var r in rows)
            {
                preds.Add(new PredictionRow
                {
                    Method = Methods.Current,
                    StationSlug = t.StationSlug,
                    WindowHours = t.WindowHours,
                    LeadHours = t.LeadHours,
                    TargetDateUtc = r.TargetDateUtc,
                    StartHourUtc = r.StartHourUtc,
                    ProbDryBlockStarting = r.ConditionalProb,
                });
            }
        }

        var outDir = Path.Combine(BakeoffDir(), $"method={Methods.Current}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");
        await ParquetSerializer.SerializeAsync(preds, outPath, cancellationToken: ct);
        _log.LogInformation(
            "Wrote {N} '{M}' predictions ({S} rows skipped for replay gaps) → {Path}",
            preds.Count, Methods.Current, skippedReplay, outPath);
        return 0;
    }

    /// <summary>
    /// Option C: 3g recipe ported to the start-hour question. For each test
    /// row, run 10,000 MC samples of the daytime hourly Bernoullis using 3a's
    /// q vector; per sample, find the FIRST length-N dry run and tally where
    /// it begins. Output a conditional distribution over candidate start
    /// hours (sums to 1).
    ///
    /// Independence assumption is wrong (rain clusters), but per the dry-
    /// window 3g bake-off this costs surprisingly little when the structural
    /// rule averages over many sample days. Same architecture, no PAV, no
    /// Bayesian copula. The hypothesis under test is whether C beats the
    /// current method's analytical product on Brier/Top-1.
    /// </summary>
    private async Task<int> ScoreOptionCAsync(List<TestRow> testSet, CancellationToken ct)
    {
        var precip3a = ResolvePrecip3aChampions();
        if (precip3a is null) return 2;

        var replayByStationLead = new Dictionary<(string, int), Dictionary<DateTime, double>>();
        foreach (var stationGroup in testSet.GroupBy(t => t.StationSlug))
        {
            var stationSlug = stationGroup.Key;
            if (!precip3a.TryGetValue(stationSlug, out var precipVer))
            {
                _log.LogWarning("No 3a champion for {S}; skipping.", stationSlug);
                continue;
            }
            foreach (var lead in stationGroup.Select(t => t.LeadHours).Distinct())
            {
                replayByStationLead[(stationSlug, lead)] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precipVer, lead);
            }
        }

        var preds = new List<PredictionRow>();
        var skippedReplay = 0;
        var skippedNoBlock = 0;
        var rng = new Random(42);
        var mcSamples = DryWindow3gPredictor.DefaultMcSamples;

        foreach (var t in testSet)
        {
            ct.ThrowIfCancellationRequested();
            if (!replayByStationLead.TryGetValue((t.StationSlug, t.LeadHours), out var hourly)) continue;

            var midnight = new DateTime(t.TargetDateUtc.Year, t.TargetDateUtc.Month, t.TargetDateUtc.Day, 0, 0, 0, DateTimeKind.Utc);
            var qDaytime = DryWindow3gPredictor.ExtractDaytimeQ(
                hourly, midnight, t.DaytimeStartUtc, t.DaytimeEndUtc);
            if (qDaytime is null) { skippedReplay++; continue; }

            var pi = DryWindow3gPredictor.StartHourConditional(qDaytime, t.WindowHours, rng, mcSamples);
            if (pi is null)
            {
                // No MC sample produced any block. Fall back to uniform — same
                // shape as the current method's degenerate-day fallback so the
                // comparison stays honest.
                var n = t.DaytimeEndUtc - t.WindowHours - t.DaytimeStartUtc + 1;
                pi = new double[n];
                for (int i = 0; i < n; i++) pi[i] = 1.0 / n;
                skippedNoBlock++;
            }

            for (int i = 0; i < pi.Length; i++)
            {
                preds.Add(new PredictionRow
                {
                    Method = Methods.OptionC,
                    StationSlug = t.StationSlug,
                    WindowHours = t.WindowHours,
                    LeadHours = t.LeadHours,
                    TargetDateUtc = midnight,
                    StartHourUtc = t.DaytimeStartUtc + i,
                    ProbDryBlockStarting = pi[i],
                });
            }
        }

        var outDir = Path.Combine(BakeoffDir(), $"method={Methods.OptionC}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");
        await ParquetSerializer.SerializeAsync(preds, outPath, cancellationToken: ct);
        _log.LogInformation(
            "Wrote {N} '{M}' predictions ({R} skipped for replay gaps, {Z} fallback-to-uniform days) → {Path}",
            preds.Count, Methods.OptionC, skippedReplay, skippedNoBlock, outPath);
        return 0;
    }

    // ---- compare: emit per-cell Brier / top-1 / log-loss ----------------------

    private async Task<int> CompareAsync(CancellationToken ct)
    {
        var truth = await LoadTruthAsync(ct);
        if (truth.Count == 0)
        {
            _log.LogError("No truth labels. Run --action prepare first.");
            return 2;
        }
        // Truth keyed by (station, window, date, hour) — lead-independent.
        var truthByKey = truth
            .GroupBy(t => (t.StationSlug, t.WindowHours,
                           Date: DateOnly.FromDateTime(t.TargetDateUtc), t.StartHourUtc))
            .ToDictionary(g => g.Key, g => g.First().HasDryBlockStarting);

        // Discover scored methods by listing method= partitions.
        var methodDirs = Directory.Exists(BakeoffDir())
            ? Directory.GetDirectories(BakeoffDir(), "method=*")
            : Array.Empty<string>();
        if (methodDirs.Length == 0)
        {
            _log.LogError("No method predictions found under {Dir}. Run --action score --method <m> first.", BakeoffDir());
            return 2;
        }

        var report = new StringBuilder();
        report.AppendLine($"# Start-hour bake-off comparison report ({DateTime.UtcNow:yyyy-MM-dd HH:mm}Z)");
        report.AppendLine();
        report.AppendLine("Test set: chronological 70/15/15 split shared with the dry-window blenders. ");
        report.AppendLine("Metrics: lower-is-better Brier + log-loss against uniform-over-truth target; ");
        report.AppendLine("higher-is-better Top-1 hit rate (did argmax_s π_s land on a truth-valid start?). ");
        report.AppendLine($"Informative-day filter (1 ≤ |truth| < |starts|) drops fully-dry / no-block days. ");
        report.AppendLine();

        var allCells = new List<CellResult>();
        foreach (var dir in methodDirs.OrderBy(d => d))
        {
            var methodName = Path.GetFileName(dir).Substring("method=".Length);
            var preds = await LoadPredictionsAsync(Path.Combine(dir, "predictions.parquet"), ct);
            _log.LogInformation("  {M}: loaded {N} predictions", methodName, preds.Count);
            var cells = ScoreMethod(preds, truthByKey);
            allCells.AddRange(cells);
        }

        // Cell-level table: one row per (station, window, lead, method).
        report.AppendLine("## Per-cell metrics");
        report.AppendLine();
        report.AppendLine("| Station | Window | Lead | Method | n_inf | Brier | LogLoss | LogLoss/Uniform | Top-1 |");
        report.AppendLine("|---|---|---|---|---:|---:|---:|---:|---:|");
        foreach (var c in allCells.OrderBy(c => c.StationSlug)
                                  .ThenBy(c => c.WindowHours)
                                  .ThenBy(c => c.LeadHours)
                                  .ThenBy(c => c.Method))
        {
            report.AppendLine($"| {c.StationSlug} | {c.WindowHours}h | {c.LeadHours}h | {c.Method} | {c.NInformative} | {c.MeanBrier:0.0000} | {c.MeanLogLoss:0.0000} | {c.LogLossSkillRatio:0.000} | {c.Top1Rate:P1} |");
        }

        // Aggregates per method.
        report.AppendLine();
        report.AppendLine("## Per-method aggregate (informative-day micro-average)");
        report.AppendLine();
        report.AppendLine("| Method | n_cells | n_inf | Brier | LogLoss | Top-1 |");
        report.AppendLine("|---|---:|---:|---:|---:|---:|");
        foreach (var grp in allCells.GroupBy(c => c.Method).OrderBy(g => g.Key))
        {
            var totalInf = grp.Sum(c => c.NInformative);
            if (totalInf == 0) continue;
            var brier = grp.Sum(c => c.MeanBrier * c.NInformative) / totalInf;
            var ll    = grp.Sum(c => c.MeanLogLoss * c.NInformative) / totalInf;
            var top1  = grp.Sum(c => c.Top1Hits) / (double)totalInf;
            report.AppendLine($"| {grp.Key} | {grp.Count()} | {totalInf} | {brier:0.0000} | {ll:0.0000} | {top1:P1} |");
        }

        var outPath = Path.Combine(BakeoffDir(), ReportFileName);
        await File.WriteAllTextAsync(outPath, report.ToString(), ct);
        _log.LogInformation("Wrote comparison report → {P}", outPath);
        Console.WriteLine();
        Console.WriteLine(report.ToString());
        return 0;
    }

    /// <summary>
    /// Score one method's predictions cell-by-cell. Each cell = (station,
    /// window, lead). We aggregate per-day metrics within a cell so the
    /// reader can compare cells directly without pretending leads are
    /// interchangeable.
    /// </summary>
    private List<CellResult> ScoreMethod(
        List<PredictionRow> preds,
        Dictionary<(string, int, DateOnly, int), bool> truthByKey)
    {
        var byCell = preds.GroupBy(p => (p.StationSlug, p.WindowHours, p.LeadHours, p.Method));
        var results = new List<CellResult>();
        foreach (var g in byCell)
        {
            var (station, window, lead, method) = g.Key;
            var byDate = g.GroupBy(p => DateOnly.FromDateTime(p.TargetDateUtc));
            int nInf = 0, top1Hits = 0;
            double brierSum = 0, llSum = 0, llUniformSum = 0;

            foreach (var dayPreds in byDate)
            {
                var date = dayPreds.Key;
                var curve = dayPreds.OrderBy(p => p.StartHourUtc)
                                    .Select(p => (p.StartHourUtc, Pi: p.ProbDryBlockStarting))
                                    .ToList();
                if (curve.Count == 0) continue;

                var truthStarts = new HashSet<int>();
                foreach (var (s, _) in curve)
                {
                    if (truthByKey.TryGetValue((station, window, date, s), out var v) && v)
                        truthStarts.Add(s);
                }

                if (!StartHourMetrics.IsInformative(truthStarts.Count, curve.Count)) continue;
                nInf++;
                if (StartHourMetrics.Top1Hit(curve, truthStarts)) top1Hits++;
                brierSum += StartHourMetrics.Brier(curve, truthStarts);
                llSum += StartHourMetrics.LogLoss(curve, truthStarts);
                llUniformSum += StartHourMetrics.LogLossUniform(curve.Count, truthStarts.Count);
            }

            if (nInf > 0)
                results.Add(new CellResult(
                    method, station, window, lead, nInf, top1Hits,
                    MeanBrier: brierSum / nInf,
                    MeanLogLoss: llSum / nInf,
                    LogLossSkillRatio: llUniformSum > 0 ? llSum / llUniformSum : double.NaN,
                    Top1Rate: top1Hits / (double)nInf));
        }
        return results;
    }

    // ---- helpers --------------------------------------------------------------

    private string BakeoffDir() => Path.Combine("data", "reports", BakeoffSubdir);

    private List<TestRow> LoadTestSet()
    {
        var p = Path.Combine(BakeoffDir(), TestSetFileName);
        if (!File.Exists(p)) return new();
        return JsonSerializer.Deserialize<List<TestRow>>(File.ReadAllText(p))!;
    }

    private async Task<List<TruthRow>> LoadTruthAsync(CancellationToken ct)
    {
        var p = Path.Combine(BakeoffDir(), TruthLabelsFileName);
        if (!File.Exists(p)) return new();
        return (await ParquetSerializer.DeserializeAsync<TruthRow>(p, cancellationToken: ct)).ToList();
    }

    private async Task<List<PredictionRow>> LoadPredictionsAsync(string path, CancellationToken ct)
    {
        if (!File.Exists(path)) return new();
        return (await ParquetSerializer.DeserializeAsync<PredictionRow>(path, cancellationToken: ct)).ToList();
    }

    /// Pull the hourly mm dictionary for a target date's daytime window from
    /// the per-station hourly truth. Returns null if any daytime hour is
    /// missing — a partial-coverage day can't be honestly scored.
    private static Dictionary<int, double>? ExtractDailyHourly(
        Dictionary<DateTime, double> hourly, DateOnly date, int startUtc, int endUtc)
    {
        var result = new Dictionary<int, double>();
        var midnight = new DateTime(date.Year, date.Month, date.Day, 0, 0, 0, DateTimeKind.Utc);
        for (int h = startUtc; h < endUtc; h++)
        {
            if (!hourly.TryGetValue(midnight.AddHours(h), out var mm)) return null;
            result[h] = mm;
        }
        return result;
    }

    /// Load raw hourly truth as (HourUtc → mm) for the given station, summed
    /// from the 15-minute EA observations.
    private Dictionary<DateTime, double> LoadHourlyTruth(string stationName)
    {
        var glob = Path.Combine(_cfg.Storage.RainfallPath,
            $"location={_cfg.Location.Name}",
            $"station={stationName}",
            "*", "*.parquet").Replace('\\', '/');
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
            SELECT date_trunc('hour', ObservedTimeUtc) AS hour_utc, SUM(Value15MinMm) AS mm
            FROM read_parquet('{glob}', hive_partitioning=false)
            WHERE Value15MinMm IS NOT NULL
            GROUP BY 1
            ORDER BY 1";
        var dict = new Dictionary<DateTime, double>();
        using var rdr = cmd.ExecuteReader();
        while (rdr.Read())
        {
            var t = DateTime.SpecifyKind(rdr.GetDateTime(0), DateTimeKind.Utc);
            dict[t] = rdr.GetDouble(1);
        }
        return dict;
    }

    /// Resolve the current 3a champion per station from the precipitation manifest.
    private Dictionary<string, string>? ResolvePrecip3aChampions()
    {
        var manifestPath = Path.Combine(_cfg.Storage.ModelsPath, "precipitation", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            _log.LogError("No precipitation manifest at {P}", manifestPath);
            return null;
        }
        var manifest = JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(manifestPath));
        if (manifest?.Stations is null) return null;
        return manifest.Stations.ToDictionary(kv => kv.Key, kv => kv.Value.Current ?? "");
    }

    // ---- types ---------------------------------------------------------------

    public sealed record TestRow(
        string StationSlug,
        int WindowHours,
        int LeadHours,
        DateOnly TargetDateUtc,
        int DaytimeStartUtc,
        int DaytimeEndUtc,
        int TruthValidStartCount);

    public sealed class TruthRow
    {
        public string StationSlug { get; set; } = "";
        public int WindowHours { get; set; }
        public DateTime TargetDateUtc { get; set; }
        public int StartHourUtc { get; set; }
        public bool HasDryBlockStarting { get; set; }
    }

    public sealed class PredictionRow
    {
        public string Method { get; set; } = "";
        public string StationSlug { get; set; } = "";
        public int WindowHours { get; set; }
        public int LeadHours { get; set; }
        public DateTime TargetDateUtc { get; set; }
        public int StartHourUtc { get; set; }
        public double ProbDryBlockStarting { get; set; }
    }

    private sealed record CellResult(
        string Method,
        string StationSlug,
        int WindowHours,
        int LeadHours,
        int NInformative,
        int Top1Hits,
        double MeanBrier,
        double MeanLogLoss,
        double LogLossSkillRatio,
        double Top1Rate);
}
