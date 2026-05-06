using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3g rho-sweep bake-off — research probe, NOT a trainer.
///
/// For each (station, window, lead) cell, builds the SAME chronological
/// 70/15/15 split that 3g training uses, then scores the test split under
/// multiple AR(1) Gaussian copula correlation values rho. Output is a
/// markdown table comparing test Brier across rho values per cell, plus a
/// summary line of "best rho per cell" and aggregate deltas vs rho=0
/// (current 3g production).
///
/// No artefacts are written. No model registry mutations. Re-runs are cheap
/// and idempotent. Use when deciding whether to productionise a non-zero
/// rho into the 3g sampler.
/// </summary>
public sealed class DryWindowRhoBakeoffCommand
{
    private readonly ILogger<DryWindowRhoBakeoffCommand> _log;
    private readonly AppConfig _cfg;
    private readonly ModelMetadataRepository _metadata;

    private static readonly int[] DefaultLeads = Leads.Short;
    private static readonly int[] DefaultWindows = { 3, 4, 6 };
    private static readonly double[] DefaultRhos = { 0.0, 0.3, 0.5, 0.7, 0.9 };

    public DryWindowRhoBakeoffCommand(
        ILogger<DryWindowRhoBakeoffCommand> log,
        AppConfig cfg,
        ModelMetadataRepository metadata)
    {
        _log = log;
        _cfg = cfg;
        _metadata = metadata;
    }

    public Task<int> RunAsync(string? stationArg, string? windowArg, int[]? leads, double[]? rhos, CancellationToken ct)
    {
        var stations = ResolveStations(stationArg ?? "all");
        if (stations is null) return Task.FromResult(2);

        var windows = ParseWindows(windowArg ?? "all");
        if (windows is null)
        {
            _log.LogError("Invalid window arg '{W}'. Expected 'all', '3', '4', or '6'.", windowArg);
            return Task.FromResult(2);
        }

        var leadSet = leads is { Length: > 0 } ? leads : DefaultLeads;
        var rhoSet = rhos is { Length: > 0 } ? rhos : DefaultRhos;
        foreach (var r in rhoSet)
        {
            if (r < 0 || r >= 1)
            {
                _log.LogError("rho values must be in [0, 1); got {R}", r);
                return Task.FromResult(2);
            }
        }

        var precipManifest = _metadata.TryGetManifest("precipitation");
        if (precipManifest?.Stations is null)
        {
            _log.LogError("rho-bakeoff needs the precipitation manifest to resolve 3a champions; not found.");
            return Task.FromResult(2);
        }

        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        const int rngSeed = 42;
        var mcSamples = DryWindow3gPredictor.DefaultMcSamples;

        _log.LogInformation(
            "rho-bakeoff — stations=[{S}] windows=[{W}] leads=[{L}] rhos=[{R}] MC samples={Mc}",
            string.Join(", ", stations), string.Join(",", windows),
            string.Join(",", leadSet), string.Join(",", rhoSet.Select(r => r.ToString("0.00"))),
            mcSamples);

        // Cell results: key = (station, window, lead) → rho → Brier
        var results = new List<CellResult>();

        foreach (var stationName in stations)
        {
            ct.ThrowIfCancellationRequested();
            var stationSlug = StationSlug.WithEaPrefix(stationName);
            if (!precipManifest.Stations.TryGetValue(stationSlug, out var precipEntry)
                || string.IsNullOrEmpty(precipEntry.Current))
            {
                _log.LogWarning("{S}: no 3a champion in precipitation manifest; skipping.", stationName);
                continue;
            }
            var precip3aVersion = precipEntry.Current;
            _log.LogInformation("=== Station '{S}' (3a champion {V}) ===", stationName, precip3aVersion);

            // Pre-load replay parquets once per station (one per lead).
            var replayByLead = new Dictionary<int, Dictionary<DateTime, double>>();
            foreach (var lead in leadSet)
            {
                replayByLead[lead] = DryWindow3gPredictor.LoadReplayHourly(
                    _cfg.Storage.PredictionsPath, stationSlug, precip3aVersion, lead);
                _log.LogInformation("  loaded {N} 3a replay hourly P(wet) for lead {L}h",
                    replayByLead[lead].Count, lead);
            }

            foreach (var window in windows)
            {
                foreach (var lead in leadSet)
                {
                    ct.ThrowIfCancellationRequested();
                    _log.LogInformation("--- {S} window={W}h lead={L}h ---", stationName, window, lead);

                    // Same row construction as 3g training so the chronological
                    // split lands on the same dates — apples-to-apples vs 3g
                    // and 3b benchmarks.
                    var spec = DryWindowFeatureBuilder.BuildSpec(_cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
                    var rows = DryWindowFeatureBuilder.BuildForLead(
                        _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                        _cfg.Location.Name, stationName,
                        spec, window, daytime, ct);

                    if (rows.Count < 100)
                    {
                        _log.LogWarning("  only {N} rows; skipping this cell.", rows.Count);
                        continue;
                    }
                    var ds = DryWindowDataset.Split(rows);
                    var hourly = replayByLead[lead];

                    // Build the q-vector + label sequence ONCE per cell, then
                    // score each rho on the same sequence with a fresh RNG
                    // seed so different rhos don't see different random streams.
                    var qVectors = new List<double[]>(ds.Test.Count);
                    var labels = new List<bool>(ds.Test.Count);
                    foreach (var row in ds.Test)
                    {
                        var (s, e) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(row.TargetDateUtc));
                        var q = DryWindow3gPredictor.ExtractDaytimeQ(hourly, row.TargetDateUtc, s, e);
                        if (q is null) continue;
                        qVectors.Add(q);
                        labels.Add(row.Label);
                    }
                    if (qVectors.Count < 10)
                    {
                        _log.LogWarning("  only {N} test rows after replay-gap filter; skipping.", qVectors.Count);
                        continue;
                    }

                    var brierByRho = new Dictionary<double, double>();
                    foreach (var rho in rhoSet)
                    {
                        var rng = new Random(rngSeed);
                        var probs = new double[qVectors.Count];
                        for (int i = 0; i < qVectors.Count; i++)
                        {
                            probs[i] = rho == 0.0
                                ? DryWindow3gPredictor.ProbDryWindow(qVectors[i], window, rng, mcSamples)
                                : DryWindow3gPredictor.ProbDryWindow(qVectors[i], window, rho, rng, mcSamples);
                        }
                        var brier = Brier(probs, labels);
                        brierByRho[rho] = brier;
                        _log.LogInformation("  rho={Rho:0.00} Brier={B:0.0000}", rho, brier);
                    }

                    results.Add(new CellResult(
                        Station: stationName,
                        Window: window,
                        Lead: lead,
                        TestRows: qVectors.Count,
                        BrierByRho: brierByRho));
                }
            }
        }

        if (results.Count == 0)
        {
            _log.LogWarning("No cells scored.");
            return Task.FromResult(3);
        }

        EmitMarkdownReport(results, rhoSet);
        return Task.FromResult(0);
    }

    private void EmitMarkdownReport(List<CellResult> results, double[] rhoSet)
    {
        var rhoCols = rhoSet.OrderBy(r => r).ToArray();
        var sb = new System.Text.StringBuilder();
        sb.AppendLine();
        sb.AppendLine("=== rho-bakeoff results ===");
        sb.Append("| Station | Window | Lead | n_test |");
        foreach (var r in rhoCols) sb.Append($" rho={r:0.00} |");
        sb.AppendLine(" best | Δ vs rho=0 |");
        sb.Append("|---|---|---|---|");
        foreach (var _ in rhoCols) sb.Append("---|");
        sb.AppendLine("---|---|");

        foreach (var c in results.OrderBy(r => r.Station).ThenBy(r => r.Window).ThenBy(r => r.Lead))
        {
            sb.Append($"| {c.Station} | {c.Window}h | {c.Lead}h | {c.TestRows} |");
            foreach (var r in rhoCols)
            {
                var b = c.BrierByRho[r];
                sb.Append($" {b:0.0000} |");
            }
            var (bestRho, bestBrier) = c.BrierByRho.OrderBy(kv => kv.Value).First();
            var baseline = c.BrierByRho.GetValueOrDefault(0.0, double.NaN);
            var delta = double.IsNaN(baseline) ? 0.0 : bestBrier - baseline;
            sb.Append($" rho={bestRho:0.00} |");
            sb.Append($" {delta:+0.0000;-0.0000;+0.0000} |");
            sb.AppendLine();
        }

        // Aggregate: how often is each rho the cell winner?
        var winCounts = rhoCols.ToDictionary(r => r, _ => 0);
        foreach (var c in results)
        {
            var (bestRho, _) = c.BrierByRho.OrderBy(kv => kv.Value).First();
            if (winCounts.ContainsKey(bestRho)) winCounts[bestRho]++;
        }
        sb.AppendLine();
        sb.AppendLine("=== Aggregate ===");
        sb.Append("Wins per rho: ");
        sb.AppendLine(string.Join(", ", rhoCols.Select(r => $"rho={r:0.00}: {winCounts[r]}/{results.Count}")));

        // Mean Brier per rho across all cells.
        sb.Append("Mean Brier across cells: ");
        sb.AppendLine(string.Join(", ", rhoCols.Select(r =>
            $"rho={r:0.00}: {results.Average(c => c.BrierByRho[r]):0.0000}")));

        // Mean delta vs rho=0.
        if (rhoCols.Contains(0.0))
        {
            sb.Append("Mean Δ Brier vs rho=0: ");
            sb.AppendLine(string.Join(", ", rhoCols.Where(r => r != 0.0).Select(r =>
                $"rho={r:0.00}: {results.Average(c => c.BrierByRho[r] - c.BrierByRho[0.0]):+0.0000;-0.0000;+0.0000}")));
        }

        _log.LogInformation("{Report}", sb.ToString());
    }

    private string[]? ResolveStations(string stationArg)
    {
        if (string.IsNullOrWhiteSpace(stationArg) || stationArg.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            var found = _cfg.Location.Rainfall.Stations.Select(s => s.Name).ToArray();
            if (found.Length == 0)
            {
                _log.LogError("No rainfall stations in config.");
                return null;
            }
            return found;
        }

        var match = _cfg.Location.Rainfall.Stations
            .FirstOrDefault(s => s.Name.Equals(stationArg, StringComparison.OrdinalIgnoreCase)
                              || SlugMatch(s.Name, stationArg));
        if (match is null)
        {
            _log.LogError("Station '{Arg}' not in config. Available: [{Names}]",
                stationArg, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
            return null;
        }
        return new[] { match.Name };
    }

    private static bool SlugMatch(string configName, string arg)
    {
        var slug = StationSlug.Of(configName);
        return slug.Equals(arg, StringComparison.OrdinalIgnoreCase)
            || ($"ea_{slug}").Equals(arg, StringComparison.OrdinalIgnoreCase);
    }

    private static int[]? ParseWindows(string w) => w.ToLowerInvariant() switch
    {
        "all" => DefaultWindows,
        "3"   => new[] { 3 },
        "4"   => new[] { 4 },
        "6"   => new[] { 6 },
        _     => null,
    };

    private static double Brier(IReadOnlyList<double> probs, IReadOnlyList<bool> labels)
    {
        var sum = 0.0;
        for (int i = 0; i < probs.Count; i++)
        {
            var y = labels[i] ? 1.0 : 0.0;
            sum += (probs[i] - y) * (probs[i] - y);
        }
        return sum / probs.Count;
    }

    private sealed record CellResult(
        string Station,
        int Window,
        int Lead,
        int TestRows,
        Dictionary<double, double> BrierByRho);
}
