using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.PrecipExact;

namespace WeatherBlend.Commands;

/// <summary>
/// 4-way IFS-cycle bake-off for the exact-runtime P(wet) blender (phase 3d).
/// Answers "is the ECMWF IFS feature — and specifically its 06/18Z scda
/// cycles — helping or hurting 3d?" by comparing what IFS contributes when
/// sourced from different cycle subsets, holding the row/eval set CONSTANT:
///   * Both — IFS from all four cycles {00, 06, 12, 18}
///   * Oper — IFS from the 00/12Z cycles only
///   * Scda — IFS from the 06/18Z (short cut-off) cycles only
///   * None — IFS dropped entirely
///
/// Design — why the comparison is valid. Production 3d is tier P1, which
/// hard-REQUIRES IFS, so a P1 `--cycles 0,12` cut would drop every
/// 06/18Z-valid row and a `--cycles 6,18` cut every 00/12Z-valid row —
/// oper vs scda would then be scored on disjoint, non-comparable test
/// sets. To avoid that, all four variants here carry IFS as an OPTIONAL
/// feature (Required = GFS + AIFS): masking the IFS column by cycle keeps
/// the row, so every variant trains and scores on the identical
/// (station, lead) row set. Because IFS archive coverage is ~100%, the
/// "Both" variant equals production 3d (P1) to within a handful of rows.
///
/// Efficiency. The forecast archive is scanned ONCE per (station, lead) —
/// the costly DuckDB pass — and the Oper/Scda/None variants are derived
/// from that single pull in memory by NaN-masking the IFS column and
/// re-running <see cref="PrecipExactFeatureBuilder.ComposeRow"/> (which
/// recomputes the mean/std/range spread features over the surviving
/// models). That is exactly equivalent to four separate cycle-filtered
/// builds but at a quarter of the scans — the first cut of this command
/// scanned per-variant and exhausted memory on a 7.8 GB box.
///
/// Crash-safe. Every cell's result is appended to a JSONL sidecar the
/// instant it finishes and the markdown report is rebuilt from that
/// sidecar after every cell, so an interrupted run keeps all completed
/// cells. Re-running a station (via --station) refreshes its rows
/// (dedupe is last-write-wins on (station, lead, variant)).
///
/// Exploratory: trains in memory, persists no model artefact, touches no
/// manifest. Metric: Brier at the 0.1 mm/h wet threshold, BSS vs
/// train-set climatology. Outputs data/reports/ifs_cycle_bakeoff_3d.md.
/// </summary>
public sealed class PrecipIfsCycleBakeoffCommand
{
    public const int HoldoutDays = 90;
    public const int ValidationDays = 30;
    private static readonly int[] DefaultLeads = { 12, 24, 48, 72 };

    // Matches the P1/P2 tier StartDate — MO Global archive start, the
    // binding constraint on the exact-runtime row window.
    private static readonly DateOnly TierStart = new(2024, 5, 4);

    private static readonly string ReportDir = Path.Combine("data", "reports");
    private static readonly string JsonlPath = Path.Combine(ReportDir, "ifs_cycle_bakeoff_3d.jsonl");
    private static readonly string MdPath = Path.Combine(ReportDir, "ifs_cycle_bakeoff_3d.md");

    private static readonly string[] VariantOrder = { "Both", "Oper", "Scda", "None" };

    // The IFS NaN-rate is genuinely NaN for the None variant (no IFS
    // column) and BSS can be too — allow named float literals so the
    // JSONL sidecar round-trips instead of throwing on serialize.
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    private readonly AppConfig _cfg;
    private readonly ILogger<PrecipIfsCycleBakeoffCommand> _log;

    public PrecipIfsCycleBakeoffCommand(AppConfig cfg, ILogger<PrecipIfsCycleBakeoffCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    private sealed record CellResult(
        string Station, int Lead, string Variant,
        int TrainN, int ValN, int TestN, double TestWetRate,
        double Brier, double Bss, double IfsNanPct);

    public async Task<int> RunAsync(int leadOverride, string? stationOverride, string asOfStr, CancellationToken ct)
    {
        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured — cannot bake off precip-exact.");
            return 2;
        }
        if (!DateOnly.TryParse(asOfStr, CultureInfo.InvariantCulture, out var asOf))
        {
            _log.LogError("Could not parse --as-of '{AsOf}' (expected yyyy-MM-dd).", asOfStr);
            return 2;
        }

        IReadOnlyList<string> stations;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            stations = _cfg.Location.Rainfall.Stations.Select(s => s.Name).ToList();
        }
        else
        {
            var match = _cfg.Location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found. Available: {Available}",
                    stationOverride, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
                return 2;
            }
            stations = new[] { match.Name };
        }
        var leads = leadOverride > 0 ? new[] { leadOverride } : DefaultLeads;

        // Split boundaries derived from --as-of so all variants share a
        // byte-identical train/val/test partition.
        var asOfEnd = new DateTime(asOf.Year, asOf.Month, asOf.Day, 0, 0, 0, DateTimeKind.Utc).AddDays(1);
        var testStart = asOfEnd.AddDays(-HoldoutDays);
        var valStart = testStart.AddDays(-ValidationDays);

        // Required = GFS + AIFS for every variant → constant row set.
        var ifsTier = new PrecipExactFeatureBuilder.TierSpec(
            Name: "ifsopt",
            Required: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
            Optional: new[] { "ecmwf_ifs_oper", "met_office_global", "gefs_ncep_mean" },
            StartDate: TierStart,
            Description: "GFS+AIFS required; IFS+MOGlobal+GEFS optional (IFS-cycle bake-off)");
        var noneTier = new PrecipExactFeatureBuilder.TierSpec(
            Name: "noifs",
            Required: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
            Optional: new[] { "met_office_global", "gefs_ncep_mean" },
            StartDate: TierStart,
            Description: "GFS+AIFS required; MOGlobal+GEFS optional; IFS dropped (IFS-cycle bake-off)");

        Directory.CreateDirectory(ReportDir);
        // A full run (no --station) starts a fresh sidecar; a single-station
        // run appends so it can backfill a station a previous run missed.
        if (string.IsNullOrWhiteSpace(stationOverride) && File.Exists(JsonlPath))
            File.Delete(JsonlPath);

        foreach (var station in stations)
        {
            foreach (var lead in leads)
            {
                ct.ThrowIfCancellationRequested();
                _log.LogInformation("Scan: station='{Station}' lead={Lead}h", station, lead);

                var ifsSpec = PrecipExactFeatureBuilder.BuildSpec(ifsTier, lead, includeUkv: true);
                var noneSpec = PrecipExactFeatureBuilder.BuildSpec(noneTier, lead, includeUkv: true);

                // ONE forecast scan per (station, lead) — the "Both" rows.
                List<BinaryTrainingRow> bothRows;
                try
                {
                    var built = await Task.Run(() => PrecipExactFeatureBuilder.Build(
                        _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath, _cfg.Location.Name, station,
                        ifsTier, ifsSpec, lead, includeUkv: true, runCycleHoursFilter: null, ct: ct), ct);
                    bothRows = built
                        .Where(r => r.ValidTimeUtc < asOfEnd)
                        .OrderBy(r => r.ValidTimeUtc)
                        .ToList();
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "Build failed for station='{Station}' lead={Lead}h — skipped", station, lead);
                    continue;
                }
                if (bothRows.Count == 0)
                {
                    _log.LogWarning("  no rows for station='{Station}' lead={Lead}h", station, lead);
                    continue;
                }

                var variants = new (string Name, BlenderSpec Spec, List<BinaryTrainingRow> Rows)[]
                {
                    ("Both", ifsSpec, bothRows),
                    ("Oper", ifsSpec, DeriveIfsCycle(bothRows, ifsSpec, lead, new[] { 0, 12 })),
                    ("Scda", ifsSpec, DeriveIfsCycle(bothRows, ifsSpec, lead, new[] { 6, 18 })),
                    ("None", noneSpec, DeriveNoIfs(bothRows, noneSpec)),
                };

                foreach (var (name, spec, rows) in variants)
                {
                    try
                    {
                        var cell = TrainAndScore(station, lead, name, spec, rows, valStart, testStart);
                        if (cell is null)
                        {
                            _log.LogWarning("  {Variant}: empty train/val/test split — skipped", name);
                            continue;
                        }
                        AppendCell(cell);
                        _log.LogInformation("  {Variant,-4}: Brier={Brier:F4} BSS={Bss:+0.000;-0.000;0.000} "
                                            + "test={TestN}", name, cell.Brier, cell.Bss, cell.TestN);
                    }
                    catch (Exception ex)
                    {
                        _log.LogError(ex, "  cell failed: {Station}/{Lead}h/{Variant}", station, lead, name);
                    }
                }

                // Rebuild the markdown report from the sidecar after every
                // (station, lead) so a kill never costs more than the cell
                // in flight.
                RebuildReport(asOf);
                GC.Collect();
            }
        }

        _log.LogInformation("Bake-off complete. Report → {Path}", Path.GetFullPath(MdPath));
        return 0;
    }

    /// <summary>
    /// Derive an IFS-cycle-restricted variant from the all-cycles "Both"
    /// rows: NaN-mask the IFS column on any row whose IFS cycle hour
    /// (ValidTime − lead) is not in <paramref name="keepCycleHours"/>, then
    /// re-compose so the mean/std/range spread features drop IFS too.
    /// </summary>
    private static List<BinaryTrainingRow> DeriveIfsCycle(
        IReadOnlyList<BinaryTrainingRow> bothRows, BlenderSpec ifsSpec, int lead, int[] keepCycleHours)
    {
        var keep = keepCycleHours.ToHashSet();
        var outRows = new List<BinaryTrainingRow>(bothRows.Count);
        foreach (var row in bothRows)
        {
            // Both spec model order: [gfs, ifs, aifs, moglobal, gefs], then ukv.
            var perModel = new double[5];
            for (int i = 0; i < 5; i++) perModel[i] = row.Features[i];
            var cycleHour = row.ValidTimeUtc.AddHours(-lead).Hour;
            if (!keep.Contains(cycleHour)) perModel[1] = double.NaN; // mask IFS
            outRows.Add(PrecipExactFeatureBuilder.ComposeRow(
                ifsSpec, row.ValidTimeUtc, perModel, row.TruthMmHour, row.Features[5]));
        }
        return outRows;
    }

    /// <summary>Derive the IFS-free variant: keep GFS/AIFS/MOGlobal/GEFS, drop IFS.</summary>
    private static List<BinaryTrainingRow> DeriveNoIfs(
        IReadOnlyList<BinaryTrainingRow> bothRows, BlenderSpec noneSpec)
    {
        var outRows = new List<BinaryTrainingRow>(bothRows.Count);
        foreach (var row in bothRows)
        {
            // None spec model order: [gfs, aifs, moglobal, gefs] — skip index 1 (ifs).
            var perModel = new[]
            {
                (double)row.Features[0], (double)row.Features[2],
                (double)row.Features[3], (double)row.Features[4],
            };
            outRows.Add(PrecipExactFeatureBuilder.ComposeRow(
                noneSpec, row.ValidTimeUtc, perModel, row.TruthMmHour, row.Features[5]));
        }
        return outRows;
    }

    private static CellResult? TrainAndScore(
        string station, int lead, string variant, BlenderSpec spec,
        IReadOnlyList<BinaryTrainingRow> rows, DateTime valStart, DateTime testStart)
    {
        var train = rows.Where(r => r.ValidTimeUtc < valStart).ToList();
        var val = rows.Where(r => r.ValidTimeUtc >= valStart && r.ValidTimeUtc < testStart).ToList();
        var test = rows.Where(r => r.ValidTimeUtc >= testStart).ToList();
        if (train.Count == 0 || val.Count == 0 || test.Count == 0) return null;

        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        var trained = PrecipOccurrenceTrainer.TrainVector(train, val, spec, hp);
        var probs = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, test);

        var labels = test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
        var brier = Brier(probs, labels);
        var clim = train.Count(r => r.Label) / (double)train.Count;
        var climBrier = Brier(Enumerable.Repeat(clim, test.Count).ToArray(), labels);
        var bss = climBrier > 0 ? (climBrier - brier) / climBrier : double.NaN;

        double ifsNanPct = double.NaN;
        var ifsIdx = spec.FeatureNames.ToList().IndexOf("precip_ifs");
        if (ifsIdx >= 0 && rows.Count > 0)
            ifsNanPct = 100.0 * rows.Count(r => float.IsNaN(r.Features[ifsIdx])) / rows.Count;

        var wetRate = test.Count(r => r.Label) / (double)test.Count;
        return new CellResult(station, lead, variant,
            train.Count, val.Count, test.Count, wetRate, brier, bss, ifsNanPct);
    }

    private static void AppendCell(CellResult cell)
        => File.AppendAllText(JsonlPath, JsonSerializer.Serialize(cell, JsonOpts) + Environment.NewLine);

    private static List<CellResult> ReadCells()
    {
        var cells = new List<CellResult>();
        if (!File.Exists(JsonlPath)) return cells;
        foreach (var line in File.ReadAllLines(JsonlPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                var c = JsonSerializer.Deserialize<CellResult>(line, JsonOpts);
                if (c is not null) cells.Add(c);
            }
            catch { /* skip a malformed line rather than abort the report */ }
        }
        // Dedupe last-write-wins per (station, lead, variant).
        return cells
            .GroupBy(c => (c.Station, c.Lead, c.Variant))
            .Select(g => g.Last())
            .ToList();
    }

    private void RebuildReport(DateOnly asOf)
    {
        try { File.WriteAllText(MdPath, BuildReport(ReadCells(), asOf)); }
        catch (Exception ex) { _log.LogWarning("Could not write report: {Msg}", ex.Message); }
    }

    private static string BuildReport(IReadOnlyList<CellResult> results, DateOnly asOf)
    {
        var sb = new StringBuilder();
        var stations = results.Select(r => r.Station).Distinct().ToList();
        var leads = results.Select(r => r.Lead).Distinct().OrderBy(l => l).ToList();

        sb.AppendLine("# Phase 3d — IFS-cycle bake-off");
        sb.AppendLine();
        sb.AppendLine($"_Generated {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC. "
                      + $"{results.Count} cells._");
        sb.AppendLine();
        sb.AppendLine($"Training/eval data capped at ValidTime ≤ **{asOf:yyyy-MM-dd}**. "
                      + $"Test window = last {HoldoutDays} days before the cap; "
                      + $"validation = the {ValidationDays} days before that; train = everything earlier.");
        sb.AppendLine("Metric: Brier at the 0.1 mm/h wet threshold (lower is better). "
                      + "BSS = Brier skill score vs train-set climatology (higher is better).");
        sb.AppendLine();
        sb.AppendLine("Four variants, **constant row/eval set** (IFS optional, Required = GFS + AIFS):");
        sb.AppendLine();
        sb.AppendLine("| Variant | IFS feature |");
        sb.AppendLine("|---|---|");
        sb.AppendLine("| Both | IFS from all four cycles {00,06,12,18} |");
        sb.AppendLine("| Oper | IFS from the 00/12Z cycles only (06/18Z NaN-masked) |");
        sb.AppendLine("| Scda | IFS from the 06/18Z short-cutoff cycles only (00/12Z NaN-masked) |");
        sb.AppendLine("| None | IFS dropped entirely |");
        sb.AppendLine();

        foreach (var station in stations)
        {
            var stationCells = results.Where(r => r.Station == station).ToList();
            if (stationCells.Count == 0) continue;

            sb.AppendLine($"## {station}");
            sb.AppendLine();
            sb.AppendLine("| Lead | Test rows | Wet % | Both | Oper | Scda | None | Best |");
            sb.AppendLine("|------|-----------|-------|------|------|------|------|------|");
            foreach (var lead in leads)
            {
                var cells = stationCells.Where(r => r.Lead == lead).ToList();
                if (cells.Count == 0) continue;
                var byVariant = cells.ToDictionary(c => c.Variant);
                var testN = cells.Select(c => c.TestN).Distinct().ToList();
                var testNStr = testN.Count == 1 ? testN[0].ToString() : string.Join("/", testN) + " ⚠";
                var wet = cells[0].TestWetRate;
                var best = cells.OrderBy(c => c.Brier).First().Variant;
                string Cell(string v) => byVariant.TryGetValue(v, out var c)
                    ? $"{c.Brier:F4} ({c.Bss:+0.000;-0.000;0.000})"
                    : "—";
                sb.AppendLine($"| {lead}h | {testNStr} | {wet:P0} "
                              + $"| {Cell("Both")} | {Cell("Oper")} | {Cell("Scda")} | {Cell("None")} "
                              + $"| **{best}** |");
            }
            sb.AppendLine();
        }

        sb.AppendLine("## Aggregate");
        sb.AppendLine();
        sb.AppendLine("| Variant | Mean Brier | Mean BSS | Cells won |");
        sb.AppendLine("|---------|------------|----------|-----------|");
        var cellKeys = results.Select(r => (r.Station, r.Lead)).Distinct().ToList();
        var winCount = new Dictionary<string, int>();
        foreach (var (st, ld) in cellKeys)
        {
            var cell = results.Where(r => r.Station == st && r.Lead == ld).ToList();
            if (cell.Count == 0) continue;
            var w = cell.OrderBy(c => c.Brier).First().Variant;
            winCount[w] = winCount.GetValueOrDefault(w) + 1;
        }
        foreach (var v in VariantOrder)
        {
            var cells = results.Where(r => r.Variant == v).ToList();
            if (cells.Count == 0) continue;
            sb.AppendLine($"| {v} | {cells.Average(c => c.Brier):F4} "
                          + $"| {cells.Average(c => c.Bss):+0.000;-0.000;0.000} "
                          + $"| {winCount.GetValueOrDefault(v)} / {cellKeys.Count} |");
        }
        sb.AppendLine();

        sb.AppendLine("## Cycle-filter sanity check (IFS feature NaN-rate)");
        sb.AppendLine();
        sb.AppendLine("Expect ~0% for Both, ~50% for Oper and Scda. (None has no IFS column.)");
        sb.AppendLine();
        sb.AppendLine("| Variant | Mean IFS NaN % |");
        sb.AppendLine("|---------|----------------|");
        foreach (var v in new[] { "Both", "Oper", "Scda" })
        {
            var cells = results.Where(r => r.Variant == v && !double.IsNaN(r.IfsNanPct)).ToList();
            if (cells.Count == 0) continue;
            sb.AppendLine($"| {v} | {cells.Average(c => c.IfsNanPct):F1}% |");
        }
        sb.AppendLine();
        return sb.ToString();
    }

    private static double Brier(IReadOnlyList<double> probs, IReadOnlyList<double> labels)
    {
        if (probs.Count != labels.Count) throw new ArgumentException("length mismatch");
        double sumSq = 0;
        for (int i = 0; i < probs.Count; i++)
        {
            var d = probs[i] - labels[i];
            sumSq += d * d;
        }
        return sumSq / probs.Count;
    }
}
