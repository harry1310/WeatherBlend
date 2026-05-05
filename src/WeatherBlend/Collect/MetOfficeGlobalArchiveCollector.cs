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
///   * Cycles = 0,12 — 06Z/18Z cap at ~66h horizon and don't publish to long
///     enough leads for our blender bucket.
///   * Leads = 1,3,6,12,24,36,48,72,96,120 — matches the GFS / ECMWF backfill
///     lead set so 2d sees the same column shape across every input source.
///     Bumped from {24,48,72,120} on 2026-05-05 along with the bake-off win.
///
/// Idempotent — re-pulling an existing date overwrites the parquet with
/// whatever AWS now has.
/// </summary>
public sealed class MetOfficeGlobalArchiveCollector
{
    private const string ScriptName = "met_office_archive_backfill.py";
    private static readonly int[] DefaultCycles = { 0, 12 };
    private static readonly int[] DefaultLeads = { 1, 3, 6, 12, 24, 36, 48, 72, 96, 120 };
    private const int DefaultLookbackDays = 3;
    private const int DefaultParallelism = 8;
    // AWS publishes a cycle's NetCDFs ~3-6h after run time; a 7h floor avoids the
    // same-day "all 14 vars 404" noise on whichever 12Z run hasn't fully landed yet.
    private const int MinCycleAgeHours = 7;

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
