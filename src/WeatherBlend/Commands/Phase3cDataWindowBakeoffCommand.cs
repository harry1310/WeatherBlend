using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// A vs B data-window diagnostic — does the 2022-2023 JMA-only training data
/// help or hurt 3c per-station performance?
///
/// **Variant A**: train per-station 3c on the full available row set (current
/// behaviour — includes 2022-2023 rows where only JMA has precip and most
/// rich features are NaN).
///
/// **Variant B**: train per-station 3c on only rows where
/// <c>ValidTimeUtc &gt;= 2024-01-01</c> (when 5+ models reliably had data).
///
/// Both score on the SAME test set (the last 15% of A's chronological split,
/// which is entirely in the 2025+ "dense-feature" era). Identical hyperparams,
/// identical seed, identical features.
///
/// If B wins systematically → the JMA 2022-2023 backfill is net-negative and
/// the v1-v6 plus 3e-oro bake-offs from 2026-05-24/25 are running on
/// contaminated data. Filter back to 2024+ and re-run the comparison.
/// If A wins → backfill is net-positive despite sparsity, current results stand.
/// If tied → sparse rows are neutral; original 3e advantage was sample-size
/// dependent, not data-quality dependent.
/// </summary>
public sealed class Phase3cDataWindowBakeoffCommand
{
    private static readonly int[] Leads = { 24, 48, 72 };
    private static readonly DateTime DENSE_ERA_START =
        new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private readonly AppConfig _cfg;
    private readonly ILogger<Phase3cDataWindowBakeoffCommand> _log;

    public Phase3cDataWindowBakeoffCommand(AppConfig cfg, ILogger<Phase3cDataWindowBakeoffCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var stations = _cfg.Locations
            .SelectMany(loc => loc.Rainfall.Stations.Select(s => (Loc: loc, Name: s.Name)))
            .ToList();
        _log.LogInformation("Data-window diagnostic across {N} stations × {L} leads", stations.Count, Leads.Length);

        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        var results = new List<DiagResult>();

        var runningJsonl = Path.Combine(_cfg.Storage.ReportsPath,
            "phase3c_data_window_running.jsonl");
        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        await File.WriteAllTextAsync(runningJsonl, "", ct);

        foreach (var lead in Leads)
        {
            _log.LogInformation("===== Lead {Lead}h =====", lead);
            var spec = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            _log.LogInformation("Rich spec: {N} features", spec.FeatureCount);

            foreach (var (loc, name) in stations)
            {
                ct.ThrowIfCancellationRequested();
                var slug = StationSlug.WithEaPrefix(name);
                var rows = PrecipRichFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    loc.Name, name, spec, minValidTime: null, ct);
                if (rows.Count < 200)
                {
                    _log.LogWarning("  {Slug}: only {N} rows — skipping", slug, rows.Count);
                    continue;
                }

                // Use BinaryDataset.Split to get the canonical chronological
                // split on the FULL row set. Variant A uses this train as-is.
                var dsA = BinaryDataset.Split(rows);
                var trainA = dsA.Train;
                var valA = dsA.Val;
                var test = dsA.Test;  // shared between A and B by design

                // Variant B: filter A's train (and val, for like-for-like
                // early stopping) to rows where ValidTimeUtc >= 2024-01-01.
                // Same chronological order — just dropping older rows.
                var trainB = trainA.Where(r => r.ValidTimeUtc >= DENSE_ERA_START).ToList();
                var valB   = valA.Where(r => r.ValidTimeUtc >= DENSE_ERA_START).ToList();

                if (trainB.Count < 100 || valB.Count < 20)
                {
                    _log.LogWarning("  {Slug} lead {Lead}h: too few 2024+ rows (train={TB} val={VB}) — skipping",
                        slug, lead, trainB.Count, valB.Count);
                    continue;
                }

                var preCount2022_23 = trainA.Count - trainB.Count;
                _log.LogInformation(
                    "  {Slug}: A train={TA} (of which {Pre} are pre-2024), B train={TB}; val A/B={VA}/{VB}; test={E} (shared, time range {T0:yyyy-MM-dd}..{T1:yyyy-MM-dd})",
                    slug, trainA.Count, preCount2022_23, trainB.Count, valA.Count, valB.Count, test.Count,
                    test[0].ValidTimeUtc, test[^1].ValidTimeUtc);

                // Sanity log: wet rate in train slices.
                var wetA = trainA.Count(r => r.Label) / (double)trainA.Count;
                var wetB = trainB.Count(r => r.Label) / (double)trainB.Count;
                _log.LogInformation(
                    "    wet-rate: A train={WA:P1}, B train={WB:P1}, test={WT:P1}",
                    wetA, wetB, test.Count(r => r.Label) / (double)test.Count);

                // Train both variants. Same hp, same seed (in hp.Seed).
                var trainedA = PrecipOccurrenceTrainer.TrainVector(trainA, valA, spec, hp);
                var trainedB = PrecipOccurrenceTrainer.TrainVector(trainB, valB, spec, hp);

                var truth = test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                var probA = PrecipOccurrenceTrainer.PredictVectorProbability(
                    trainedA.Ml, trainedA.Model, spec, test);
                var probB = PrecipOccurrenceTrainer.PredictVectorProbability(
                    trainedB.Ml, trainedB.Model, spec, test);
                var brierA = PrecipMetrics.Brier(probA, truth);
                var brierB = PrecipMetrics.Brier(probB, truth);
                var delta = brierA > 0 ? (brierB - brierA) / brierA * 100 : double.NaN;

                _log.LogInformation(
                    "  {Slug} lead {Lead}h: A Brier={A:0.0000} (n_train={NA}), B Brier={B:0.0000} (n_train={NB}), Δ B vs A: {D:+0.0;-0.0;0.0}%",
                    slug, lead, brierA, trainA.Count, brierB, trainB.Count, delta);

                var row = new DiagResult(
                    Lead: lead, Slug: slug, LocationName: loc.Name,
                    NTrainA: trainA.Count, NTrainB: trainB.Count, NTest: test.Count,
                    PreDenseRows: preCount2022_23,
                    BrierA: brierA, BrierB: brierB,
                    WetRateA: wetA, WetRateB: wetB,
                    TestStart: test[0].ValidTimeUtc, TestEnd: test[^1].ValidTimeUtc);
                results.Add(row);
                await File.AppendAllTextAsync(runningJsonl, JsonSerializer.Serialize(row) + "\n", ct);
            }
        }

        if (results.Count == 0)
        {
            _log.LogError("No results produced.");
            return 3;
        }

        var path = WriteReport(results);
        _log.LogInformation("Wrote {Path}", path);
        return 0;
    }

    private string WriteReport(IReadOnlyList<DiagResult> results)
    {
        var dir = _cfg.Storage.ReportsPath;
        var path = Path.Combine(dir, $"phase3c_data_window_diagnostic_{DateTime.UtcNow:yyyy-MM-dd}.md");

        var sb = new StringBuilder();
        sb.AppendLine("# Phase 3c data-window diagnostic — does the 2022-2023 JMA backfill help or hurt?");
        sb.AppendLine();
        sb.AppendLine($"Generated {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC.");
        sb.AppendLine();
        sb.AppendLine("Per-station 3c (LightGBM, rich (59) features) trained on two data windows,");
        sb.AppendLine("scored on **identical** test rows (the last 15% of the full chronological split).");
        sb.AppendLine();
        sb.AppendLine("- **Variant A**: train on ALL rows the rich SQL returns (current behaviour — includes");
        sb.AppendLine("  2022-2023 rows where only JMA reliably contributes precipitation and most rich");
        sb.AppendLine("  features are NaN).");
        sb.AppendLine("- **Variant B**: train on rows with `ValidTimeUtc >= 2024-01-01` only (dense-feature era,");
        sb.AppendLine("  5+ models contributing).");
        sb.AppendLine();
        sb.AppendLine("Same hyperparameters, same seed (42), same features, same test rows.");
        sb.AppendLine();

        sb.AppendLine("## Per-station results");
        sb.AppendLine();
        sb.AppendLine("| Station | Lead | A n_train | B n_train | pre-2024 dropped | test_n | A Brier | B Brier | Δ B vs A |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        foreach (var r in results.OrderBy(r => r.Slug).ThenBy(r => r.Lead))
        {
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1} | {2} | {3} | {4} | {5} | {6:0.0000} | {7:0.0000} | {8:+0.0;-0.0;0.0}% |",
                r.Slug, r.Lead, r.NTrainA, r.NTrainB, r.PreDenseRows, r.NTest,
                r.BrierA, r.BrierB,
                r.BrierA > 0 ? (r.BrierB - r.BrierA) / r.BrierA * 100 : double.NaN));
        }

        sb.AppendLine();
        sb.AppendLine("## Aggregate per lead (mean across stations)");
        sb.AppendLine();
        sb.AppendLine("| Lead | n stations | mean A Brier | mean B Brier | mean Δ |");
        sb.AppendLine("|---:|---:|---:|---:|---:|");
        foreach (var lead in results.Select(r => r.Lead).Distinct().OrderBy(l => l))
        {
            var slice = results.Where(r => r.Lead == lead).ToList();
            var ma = slice.Average(r => r.BrierA);
            var mb = slice.Average(r => r.BrierB);
            var d = ma > 0 ? (mb - ma) / ma * 100 : double.NaN;
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1} | {2:0.0000} | {3:0.0000} | {4:+0.0;-0.0;0.0}% |",
                lead, slice.Count, ma, mb, d));
        }

        sb.AppendLine();
        sb.AppendLine("## Aggregate per Bonehill cell (Bellever / Bovey / Hexworthy / Princetown)");
        sb.AppendLine();
        var bonehill = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy", "ea_princetown",
        };
        sb.AppendLine("| Lead | n stations | mean A Brier | mean B Brier | mean Δ |");
        sb.AppendLine("|---:|---:|---:|---:|---:|");
        foreach (var lead in results.Select(r => r.Lead).Distinct().OrderBy(l => l))
        {
            var slice = results.Where(r => r.Lead == lead && bonehill.Contains(r.Slug)).ToList();
            if (slice.Count == 0) continue;
            var ma = slice.Average(r => r.BrierA);
            var mb = slice.Average(r => r.BrierB);
            var d = ma > 0 ? (mb - ma) / ma * 100 : double.NaN;
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1} | {2:0.0000} | {3:0.0000} | {4:+0.0;-0.0;0.0}% |",
                lead, slice.Count, ma, mb, d));
        }

        sb.AppendLine();
        sb.AppendLine("## Verdict guide");
        sb.AppendLine();
        sb.AppendLine("- **B clearly wins** (Δ B vs A < -1% across most cells): JMA 2022-2023 backfill is net-negative.");
        sb.AppendLine("  Filter rich SQL back to ValidTimeUtc >= 2024-01-01. The v1-v6 + 3e-oro results from");
        sb.AppendLine("  2026-05-24/25 ran on contaminated data — original 3e-vs-3c story (3e wins) likely still holds.");
        sb.AppendLine("- **A clearly wins** (Δ B vs A > +1%): backfill is net-positive despite sparsity. Current results stand.");
        sb.AppendLine("- **Tied** (|Δ| < 0.5%): backfill is neutral; original 3e win was sample-size-dependent.");
        sb.AppendLine();
        sb.AppendLine("Bonehill stations (with the JMA backfill) drive the answer; Membury stations (which always had");
        sb.AppendLine("dense 2022+ JMA from its 2026-05-11 backfill) provide a control — if Δ is ~0 for Membury but");
        sb.AppendLine("non-zero for Bonehill, that confirms the issue is sparse-row contamination, not sample size.");

        File.WriteAllText(path, sb.ToString());
        return path;
    }

    private sealed record DiagResult(
        int Lead, string Slug, string LocationName,
        int NTrainA, int NTrainB, int NTest, int PreDenseRows,
        double BrierA, double BrierB,
        double WetRateA, double WetRateB,
        DateTime TestStart, DateTime TestEnd);
}
