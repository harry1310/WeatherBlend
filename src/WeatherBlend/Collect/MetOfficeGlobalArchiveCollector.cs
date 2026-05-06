using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the Met Office Global Deterministic 10km archive
/// from AWS Open Data. Wraps <see cref="MetOfficeArchiveBackfillClient"/>
/// with collect-friendly defaults so the live <c>s3-collect</c> command can
/// keep this source current.
///
/// Status changed 2026-05-05: Met Office Global is now a BLENDER INPUT for
/// the exact-runtime 2d temperature blender, not just a comparison baseline.
/// (Earlier rationale in memory/project_met_office_global_role.md remains
/// valid for the offset_day-based 2b/2c blenders, which still don't use it.)
///
/// Defaults:
///   * Lookback = 3 days — AWS publishes ~3-6h after run, 3-day window catches
///     today + yesterday's cycles + any late landings.
///   * Cycles = 0, 6, 12, 18 — all four daily cycles. 06Z/18Z cap at ~66h
///     horizon so leads >66h (i.e. 72/96/120) will 404 silently for those
///     cycles; the python script drops them at fetch time and the cycle
///     parquet just lacks those rows. 2d/3d only need lead 12 and 24, both
///     well within range, so the partial coverage is fine. Restricted to
///     {0,12} until 2026-05-06 — that was a leak from the older multi-lead
///     world where 96/120h leads mattered.
///   * Leads = 1,3,6,12,24,36,48,72,96,120 — matches the GFS / ECMWF backfill
///     lead set so 2d sees the same column shape across every input source
///     for the 00Z/12Z cycles. 06Z/18Z cycles return rows only for leads
///     ≤66h (see cycle note above).
///
/// Idempotent — re-pulling an existing date overwrites the parquet with
/// whatever AWS now has.
/// </summary>
public sealed class MetOfficeGlobalArchiveCollector
{
    private const string ScriptName = "met_office_archive_backfill.py";
    private static readonly int[] DefaultCycles = { 0, 6, 12, 18 };
    private static readonly int[] DefaultLeads = { 1, 3, 6, 12, 24, 36, 48, 72, 96, 120 };
    private const int DefaultLookbackDays = 3;
    private const int DefaultParallelism = 8;
    // No min-cycle-age filter (was 7h until 2026-05-06). AWS publishes a cycle's
    // NetCDFs ~3-6h after run time; the script handles per-variable 404s
    // gracefully (missing vars become null columns, leads with zero vars
    // skipped silently), and re-pulls overwrite partial parquets when more
    // vars land. Trade-off: noisier logs for fresher data — MO Global becomes
    // available ~6h earlier per cycle.
    private const int MinCycleAgeHours = 0;

    private readonly MetOfficeArchiveBackfillClient _client;
    private readonly ILogger<MetOfficeGlobalArchiveCollector> _log;

    public MetOfficeGlobalArchiveCollector(
        MetOfficeArchiveBackfillClient client,
        ILogger<MetOfficeGlobalArchiveCollector> log)
    {
        _client = client;
        _log = log;
    }

    public async Task<int> CollectAsync(CancellationToken ct)
    {
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        var start = today.AddDays(-DefaultLookbackDays);
        _log.LogInformation(
            "Met Office Global Det archive — refreshing {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] min-age={Age}h",
            start, today, string.Join(',', DefaultCycles), string.Join(',', DefaultLeads), MinCycleAgeHours);
        return await _client.RunAsync(
            ScriptName, start, today, DefaultCycles, DefaultLeads, DefaultParallelism, MinCycleAgeHours, ct);
    }
}
