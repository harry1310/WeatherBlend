using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3b label diagnostic (pre-training checkpoint). Reads hourly EA rainfall
/// from the truth tree, runs <see cref="DryWindowLabelBuilder"/> for each station
/// and each window length, and prints:
///   - per-month label rate per (station, window)
///   - unusable-day rate
///   - window-length correlation (6h ⊂ 4h ⊂ 3h sanity check)
///   - Bellever vs Princetown agreement
///
/// Output goes to <c>data/reports/phase3b_diagnostic_{yyyy-MM-dd_HHmmss}.md</c>.
/// No model training, no artefacts — this command is purely for the human-in-the-
/// loop sanity check before spending compute on 18 blenders.
/// </summary>
public sealed class DryWindowDiagnosticCommand
{
    private readonly ILogger<DryWindowDiagnosticCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] Windows = { 3, 4, 6 };

    // Bellever primary + Princetown secondary per the phase-3b brief.
    private static readonly string[] Phase3bStations = { "Bellever Dartmoor", "Princetown" };

    public DryWindowDiagnosticCommand(ILogger<DryWindowDiagnosticCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var stationNames = Phase3bStations
            .Where(target => _cfg.Location.Rainfall.Stations
                .Any(s => s.Name.Equals(target, StringComparison.OrdinalIgnoreCase)))
            .ToArray();
        if (stationNames.Length == 0)
        {
            _log.LogError("No phase-3b rainfall stations configured. Expected one of: [{Stations}]",
                string.Join(", ", Phase3bStations));
            return 2;
        }

        _log.LogInformation("Running dry-window label diagnostic for stations=[{Stations}], windows=[{W}]",
            string.Join(", ", stationNames), string.Join(", ", Windows));

        var perStation = new Dictionary<string, DryWindowLabelBuilder.BuildResult>();
        foreach (var station in stationNames)
        {
            ct.ThrowIfCancellationRequested();
            var hourly = QueryHourlyTruth(station, ct);
            _log.LogInformation("Station {Station}: {N} quality-passed hourly rows loaded.", station, hourly.Count);
            if (hourly.Count == 0)
            {
                _log.LogWarning("No truth rows for {Station}; skipping.", station);
                continue;
            }

            var result = DryWindowLabelBuilder.Build(hourly, Windows);
            perStation[station] = result;

            _log.LogInformation(
                "Station {Station}: candidate={Cand} usable={Use} dropped={Drop} ({Pct:P1} dropped)",
                station, result.CandidateDays, result.UsableDays, result.DroppedDays, result.DroppedFraction);
        }

        if (perStation.Count == 0)
        {
            _log.LogError("No stations produced labels. Aborting.");
            return 3;
        }

        var md = BuildReport(perStation);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"phase3b_diagnostic_{DateTime.UtcNow:yyyy-MM-dd_HHmmss}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Diagnostic report → {Path}", outPath);

        // Echo headline numbers so the user gets the gist without opening the file.
        foreach (var (station, result) in perStation)
        {
            var labelsByWindow = result.Labels
                .GroupBy(l => l.WindowHours)
                .ToDictionary(g => g.Key, g => g.Count(l => l.HasDryWindow) / (double)Math.Max(1, g.Count()));
            _log.LogInformation(
                "HEADLINE {Station}: P(3h)={P3:P1} P(4h)={P4:P1} P(6h)={P6:P1} (usable days={U}/{C})",
                station, labelsByWindow[3], labelsByWindow[4], labelsByWindow[6],
                result.UsableDays, result.CandidateDays);
        }

        return 0;
    }

    private IReadOnlyList<DryWindowLabelBuilder.HourlyTruth> QueryHourlyTruth(
        string stationName, CancellationToken ct)
    {
        var glob = Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");
        var escStation = stationName.Replace("'", "''");
        var escLocation = _cfg.Location.Name.Replace("'", "''");

        // Same aggregation as training/verify: four readings per hour required.
        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLocation}'
  AND StationName  = '{escStation}'
  AND Value15MinMm IS NOT NULL
GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<DryWindowLabelBuilder.HourlyTruth>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                var hourUtc = DateTime.SpecifyKind(r.GetDateTime(0), DateTimeKind.Utc);
                rows.Add(new DryWindowLabelBuilder.HourlyTruth(hourUtc, r.GetDouble(1)));
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Rainfall tree empty for {Station}.", stationName);
        }
        return rows;
    }

    private static string BuildReport(
        IReadOnlyDictionary<string, DryWindowLabelBuilder.BuildResult> perStation)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("# Phase 3b — label diagnostic");
        sb.AppendLine();
        sb.AppendLine($"Generated {DateTime.UtcNow:yyyy-MM-dd HH:mm}Z. " +
                      "Pre-training sanity check: is there signal to learn?");
        sb.AppendLine();

        // ---- Section 1: coverage + unusable-day rate ----
        sb.AppendLine("## Coverage");
        sb.AppendLine();
        sb.AppendLine("| Station | First usable | Last usable | Candidate days | Usable | Dropped | % dropped |");
        sb.AppendLine("|---|---|---|---:|---:|---:|---:|");
        foreach (var (station, r) in perStation)
        {
            var dates = r.Labels.Select(l => l.Date).Distinct().OrderBy(d => d).ToArray();
            var first = dates.Length == 0 ? "—" : dates[0].ToString("yyyy-MM-dd");
            var last  = dates.Length == 0 ? "—" : dates[^1].ToString("yyyy-MM-dd");
            sb.AppendLine($"| {station} | {first} | {last} | {r.CandidateDays} | {r.UsableDays} | {r.DroppedDays} | {r.DroppedFraction:P1} |");
        }
        sb.AppendLine();

        // ---- Section 2: per-month label rate per (station, window) ----
        sb.AppendLine("## Label distribution — fraction of usable days where `has_dry_window=1`");
        sb.AppendLine();
        foreach (var (station, r) in perStation)
        {
            sb.AppendLine($"### {station}");
            sb.AppendLine();
            sb.AppendLine("| Month | n_usable | 3h | 4h | 6h |");
            sb.AppendLine("|---|---:|---:|---:|---:|");
            var byMonth = r.Labels
                .GroupBy(l => l.Date.Month)
                .OrderBy(g => g.Key);
            foreach (var g in byMonth)
            {
                var ndays = g.Select(l => l.Date).Distinct().Count();
                var p3 = g.Where(l => l.WindowHours == 3).Count(l => l.HasDryWindow) / (double)Math.Max(1, g.Count(l => l.WindowHours == 3));
                var p4 = g.Where(l => l.WindowHours == 4).Count(l => l.HasDryWindow) / (double)Math.Max(1, g.Count(l => l.WindowHours == 4));
                var p6 = g.Where(l => l.WindowHours == 6).Count(l => l.HasDryWindow) / (double)Math.Max(1, g.Count(l => l.WindowHours == 6));
                sb.AppendLine($"| {MonthName(g.Key)} | {ndays} | {p3:P1} | {p4:P1} | {p6:P1} |");
            }

            // Overall rates.
            var overall3 = r.Labels.Where(l => l.WindowHours == 3).Count(l => l.HasDryWindow) / (double)r.UsableDays;
            var overall4 = r.Labels.Where(l => l.WindowHours == 4).Count(l => l.HasDryWindow) / (double)r.UsableDays;
            var overall6 = r.Labels.Where(l => l.WindowHours == 6).Count(l => l.HasDryWindow) / (double)r.UsableDays;
            sb.AppendLine($"| **overall** | **{r.UsableDays}** | **{overall3:P1}** | **{overall4:P1}** | **{overall6:P1}** |");
            sb.AppendLine();
        }

        // ---- Section 3: window-length containment sanity check ----
        sb.AppendLine("## Window-length sanity check");
        sb.AppendLine();
        sb.AppendLine("By construction, a 6h dry window contains a 4h and 3h one. If any row " +
                      "fails containment, the builder has a bug.");
        sb.AppendLine();
        sb.AppendLine("| Station | days with 6h | also ≥4h | also ≥3h | days with 3h | of which ≥4h | of which ≥6h |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|");
        foreach (var (station, r) in perStation)
        {
            var byDate = r.Labels
                .GroupBy(l => l.Date)
                .ToDictionary(
                    g => g.Key,
                    g => g.ToDictionary(l => l.WindowHours, l => l.HasDryWindow));

            int n6 = byDate.Count(d => d.Value[6]);
            int n6_and_4 = byDate.Count(d => d.Value[6] && d.Value[4]);
            int n6_and_3 = byDate.Count(d => d.Value[6] && d.Value[3]);
            int n3 = byDate.Count(d => d.Value[3]);
            int n3_and_4 = byDate.Count(d => d.Value[3] && d.Value[4]);
            int n3_and_6 = byDate.Count(d => d.Value[3] && d.Value[6]);

            sb.AppendLine($"| {station} | {n6} | {n6_and_4}/{n6} | {n6_and_3}/{n6} | {n3} | {n3_and_4}/{n3} ({SafeFrac(n3_and_4, n3):P1}) | {n3_and_6}/{n3} ({SafeFrac(n3_and_6, n3):P1}) |");
        }
        sb.AppendLine();

        // ---- Section 4: Bellever vs Princetown agreement ----
        if (perStation.Count >= 2)
        {
            sb.AppendLine("## Bellever vs Princetown agreement");
            sb.AppendLine();
            sb.AppendLine("Fraction of shared usable days where both stations give the same label.");
            sb.AppendLine();
            sb.AppendLine("| Window | shared days | both=1 | both=0 | disagree | agreement |");
            sb.AppendLine("|---|---:|---:|---:|---:|---:|");

            var keys = perStation.Keys.OrderBy(k => k).ToArray();
            var a = keys[0];
            var b = keys[1];
            var byDateA = perStation[a].Labels.ToLookup(l => (l.Date, l.WindowHours));
            var byDateB = perStation[b].Labels.ToLookup(l => (l.Date, l.WindowHours));

            foreach (var w in Windows)
            {
                var labelsA = perStation[a].Labels
                    .Where(l => l.WindowHours == w)
                    .ToDictionary(l => l.Date, l => l.HasDryWindow);
                var labelsB = perStation[b].Labels
                    .Where(l => l.WindowHours == w)
                    .ToDictionary(l => l.Date, l => l.HasDryWindow);
                var sharedDates = labelsA.Keys.Intersect(labelsB.Keys).ToArray();

                int both1 = sharedDates.Count(d => labelsA[d] && labelsB[d]);
                int both0 = sharedDates.Count(d => !labelsA[d] && !labelsB[d]);
                int disagree = sharedDates.Length - both1 - both0;
                var agree = sharedDates.Length == 0 ? double.NaN : (both1 + both0) / (double)sharedDates.Length;

                sb.AppendLine($"| {w}h | {sharedDates.Length} | {both1} | {both0} | {disagree} | {agree:P1} |");
            }
            sb.AppendLine();
        }

        sb.AppendLine("## Interpretation");
        sb.AppendLine();
        sb.AppendLine("- **> 95% positive rate** → extreme imbalance, little skill headroom over climatology.");
        sb.AppendLine("- **< 10% positive rate** → sparse positives; val/test sets will be noisy.");
        sb.AppendLine("- **Winter vs summer spread** in the 6h column is the strongest signal for where blending can help.");
        sb.AppendLine("- **Station disagreement > 20%** on the 3h window would be surprising and worth probing.");

        return sb.ToString();
    }

    private static double SafeFrac(int num, int den) => den == 0 ? 0.0 : (double)num / den;

    private static string MonthName(int m) => m switch
    {
        1 => "Jan", 2 => "Feb", 3 => "Mar", 4 => "Apr",
        5 => "May", 6 => "Jun", 7 => "Jul", 8 => "Aug",
        9 => "Sep", 10 => "Oct", 11 => "Nov", 12 => "Dec",
        _ => m.ToString(),
    };
}
