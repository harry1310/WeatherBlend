using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the Met Office UK Deterministic 2km (UKV) archive
/// from AWS Open Data. Wraps <see cref="MetOfficeArchiveBackfillClient"/>
/// with collect-friendly defaults, mirroring
/// <see cref="MetOfficeGlobalArchiveCollector"/> for the global 10km dataset.
///
/// UKV is an optional column for the 2d temperature blender AND the 3d
/// precipitation blender (both productionised 2026-05-05+). The two
/// blenders use DIFFERENT picker strategies after the 2026-05-06 bake-off:
///   * Temp 2d → Strict: cycles {0, 6, 12, 18} × leads {12, 24}.
///   * Precip 3d → Averaging: cycles {3, 15} × leads {9, 15, 21, 27},
///     averaged across two leads per V-hour.
/// See <see cref="WeatherBlend.Train.Exact12h.Exact12hFeatureBuilder.UkvPickStrategy"/>
/// for the per-target rationale (temp gains 0.017°C MAE with Strict;
/// precip loses up to 0.09 BSS at Hexworthy with Strict, so it sticks
/// with Averaging).
///
/// To support both pickers live, the collector pulls the UNION of both
/// (cycle, lead) sets each tick. Cross-product is 6 cycles × 6 leads = 36
/// (cycle, lead) tuples; ~16 of those are actually consumed by the two
/// pickers, the rest are extra rows in R2 that nothing reads. Storage
/// cost is trivial; pulling a superset means a future picker change
/// doesn't trigger a backfill.
///
/// Defaults (overridable for one-off backfills via the CLI command):
///   * Lookback = 3 days. AWS publishes ~3-6h after each cycle's run-time;
///     a 3-day window catches both today and yesterday's cycles plus any
///     that landed late.
///   * Cycles = 0, 3, 6, 12, 15, 18 — superset of both pickers' needs.
///   * Leads = 9, 12, 15, 21, 24, 27 — superset.
///
/// The script is idempotent — re-pulling an existing date overwrites the
/// parquet with whatever AWS now has.
/// </summary>
public sealed class MetOfficeUkvArchiveCollector
{
    private const string ScriptName = "met_office_ukv_archive_backfill.py";
    // Superset of both picker strategies' cycles. Strict (temp 2d) needs
    // {0,6,12,18}; Averaging (precip 3d) needs {3,15}. Union = 6 cycles.
    private static readonly int[] DefaultCycles = { 0, 3, 6, 12, 15, 18 };
    // Superset of both picker strategies' leads. Strict (temp 2d) needs
    // {12, 24, 48}; Averaging (precip 3d) needs {9, 15, 21, 27, 45, 51,
    // 69, 75}. Union = 9 leads. Lead 48 added 2026-05-07 alongside the
    // historical backfill so 2d's Strict picker can reach 48h. Long
    // leads (45/51/69/75) are only published from 03Z + 15Z cycles —
    // including them in the cycle 0/6/12/18 fetches would 404 silently
    // since those cycles cap at ~T+54h, but that's harmless overhead.
    // Filtering per-cycle to "leads this cycle can publish" is a further
    // optimisation if the live collect runtime ever bites.
    private static readonly int[] DefaultLeads = { 9, 12, 15, 21, 24, 27, 45, 48, 51, 69, 72, 75 };
    private const int DefaultLookbackDays = 3;
    private const int DefaultParallelism = 8;
    // No min-cycle-age filter (was 7h until 2026-05-06). AWS publishes a cycle's
    // NetCDFs ~3-6h after run time; the script handles per-variable 404s
    // gracefully (missing vars become null columns, leads with zero vars
    // skipped silently), and re-pulls overwrite partial parquets when more
    // vars land. Trade-off: noisier logs for fresher data — UKV becomes
    // available ~6h earlier per cycle.
    private const int MinCycleAgeHours = 0;

    private readonly MetOfficeArchiveBackfillClient _client;
    private readonly ILogger<MetOfficeUkvArchiveCollector> _log;

    public MetOfficeUkvArchiveCollector(
        MetOfficeArchiveBackfillClient client,
        ILogger<MetOfficeUkvArchiveCollector> log)
    {
        _client = client;
        _log = log;
    }

    public async Task<int> CollectAsync(CancellationToken ct)
    {
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        var start = today.AddDays(-DefaultLookbackDays);
        _log.LogInformation(
            "Met Office UKV 2km archive — refreshing {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] min-age={Age}h",
            start, today, string.Join(',', DefaultCycles), string.Join(',', DefaultLeads), MinCycleAgeHours);
        return await _client.RunAsync(
            ScriptName, start, today, DefaultCycles, DefaultLeads, DefaultParallelism, MinCycleAgeHours, ct);
    }
}
