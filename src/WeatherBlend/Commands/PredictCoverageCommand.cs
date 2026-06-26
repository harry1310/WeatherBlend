using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Predict.Coverage;

namespace WeatherBlend.Commands;

/// <summary>
/// Post-predict coverage guard. Asserts that every active (target × location ×
/// phase) cell that has a trained bundle actually produced fresh predictions
/// for this cycle's anchor — the positive coverage assertion that was missing
/// when the 2026-06-05 GEM outage silently zeroed Membury temperature + the
/// Bonehill element blenders for days without anything going red.
///
/// Run as the final, HARD-FAIL step after predict (and after render/deploy in
/// the fused workflow, so a degraded cycle still publishes what it has while
/// still alerting). Exit codes:
///   0 — every active+bundled cell produced fresh rows (warnings allowed).
///   5 — ≥1 breach (an active+bundled cell produced nothing). Distinct code so
///       it can never be confused with predict's soft-skip 2/3/4; any non-zero
///       reds the job → the GH App files a [ci-fail] issue (existing path).
/// See <see cref="CoverageGuard"/> for the rationale + granularity.
/// </summary>
public sealed class PredictCoverageCommand
{
    private readonly ILogger<PredictCoverageCommand> _log;
    private readonly AppConfig _cfg;

    public PredictCoverageCommand(ILogger<PredictCoverageCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
        => RunAsync(forDate, locationOverride: null, ct);

    public Task<int> RunAsync(DateOnly? forDate, string? locationOverride, CancellationToken ct)
        => RunAsync(forDate, locationOverride, PhaseRegistry.Default, ct);

    /// <summary>Registry-injectable overload — tests pass a controlled
    /// <see cref="PhaseRegistry"/> so coverage assertions don't depend on the
    /// live phases.yaml.</summary>
    internal Task<int> RunAsync(DateOnly? forDate, string? locationOverride, PhaseRegistry registry, CancellationToken ct)
    {
        var anchor = PredictAnchor.Compute(DateTime.UtcNow, forDate);
        var anchorStr = anchor.ToString("yyyy-MM-dd");

        var locations = _cfg.Locations
            .Where(l => string.IsNullOrWhiteSpace(locationOverride)
                        || string.Equals(l.Name, locationOverride, StringComparison.OrdinalIgnoreCase))
            .Select(l => new CoverageGuard.LocationSpec(
                l.Name,
                // poolOnly gauges (Princetown, Manaton) train the 3o pool but are never
                // predicted, so the 3o/3oni bundles re-saved under their slug must NOT be
                // coverage-checked — they'd always breach ("active bundle, no predictions").
                // Use the real slug (wl_* for WeatherLink) for the rest.
                l.Rainfall.Stations.Where(s => !s.PoolOnly).Select(s => s.Slug).ToList()))
            .ToList();

        if (locations.Count == 0)
        {
            _log.LogError("Coverage guard: no locations matched (override='{Loc}').", locationOverride);
            return Task.FromResult(2);
        }

        _log.LogInformation("Coverage guard — anchor {Anchor:yyyy-MM-dd} (for-date={ForDate}), locations=[{Locs}]",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live", string.Join(", ", locations.Select(l => l.Name)));

        var predictionsRoot = _cfg.Storage.PredictionsPath;

        bool Produced(CoverageGuard.CoverageCell c)
        {
            var partition = c.Layout == CoverageGuard.PredLayout.PerKeyDir
                ? Path.Combine(predictionsRoot, c.Target, c.StationKey, $"model_version={c.Version}", $"date={anchorStr}", "predictions.parquet")
                : Path.Combine(predictionsRoot, c.Target, $"model_version={c.Version}", $"date={anchorStr}", "predictions.parquet");

            if (!File.Exists(partition)) return false;
            // PerKeyDir: predict only writes the parquet when it has ≥1 row AND the
            // directory already scopes the cell → existence == produced.
            if (c.Layout == CoverageGuard.PredLayout.PerKeyDir) return true;
            // Flat: one parquet holds every location's rows — count this location's.
            return FlatPartitionRowsForLocation(partition, c.Location) > 0;
        }

        // Scope to dotnet phases: this guard runs inside predict / predict-and-
        // render, which produce ONLY the dotnet phases (predict-all + predict-
        // tail). The python phases (4a / 3f / wind_mvn) come from separate
        // workflows on their own cadences (predict-3f.yml even runs AFTER this),
        // so checking them here would false-positive before their partition
        // exists. Guarding the python phases in their own workflows is a
        // follow-up. (Verified 2026-06-05: the incident was 100% dotnet phases.)
        var result = CoverageGuard.Run(_cfg.Storage.ModelsPath, registry, locations, Produced,
            includePhase: p => p.Impl == PhaseImpl.Dotnet);

        foreach (var w in result.Warnings)
            _log.LogWarning("::warning::Coverage: {Target}/{Station}{Phase} — {Reason}",
                w.Target, string.IsNullOrEmpty(w.StationKey) ? "" : w.StationKey + "/", w.Phase, w.Reason);

        if (result.Passed)
        {
            _log.LogInformation("Coverage guard PASS — {N} active+bundled cell(s) all produced fresh predictions for {Anchor}. ({W} warning(s).)",
                result.CellsChecked, anchorStr, result.Warnings.Count);
            return Task.FromResult(0);
        }

        foreach (var b in result.Breaches)
            _log.LogError("::error::Coverage breach: {Target}/{Station}/{Phase} (versions {Versions}) — {Reason}",
                b.Target, b.StationKey, b.Phase, b.Versions, b.Reason);

        _log.LogError("Coverage guard FAIL — {N} breach(es) of {Checked} checked cell(s) for anchor {Anchor}. " +
                      "An active, bundled phase produced no predictions this cycle (the 2026-06-05 GEM-outage signature).",
            result.Breaches.Count, result.CellsChecked, anchorStr);
        return Task.FromResult(5);
    }

    /// <summary>Count rows for <paramref name="location"/> in a flat element-tree
    /// partition (LocationName is in-column, not in the path). DuckDB, in-memory,
    /// same read pattern the predict/verify commands use.</summary>
    private static long FlatPartitionRowsForLocation(string parquetPath, string location)
    {
        var p = parquetPath.Replace('\\', '/').Replace("'", "''");
        var loc = location.Replace("'", "''");
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT count(*) FROM read_parquet('{p}', union_by_name = true) WHERE LocationName = '{loc}'";
        var scalar = cmd.ExecuteScalar();
        return scalar is null or DBNull ? 0 : Convert.ToInt64(scalar);
    }
}
