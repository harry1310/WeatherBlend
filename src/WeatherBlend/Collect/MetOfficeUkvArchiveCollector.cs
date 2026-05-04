using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the Met Office UK Deterministic 2km (UKV) archive
/// from AWS Open Data. Wraps <see cref="MetOfficeArchiveBackfillClient"/>
/// with collect-friendly defaults, mirroring
/// <see cref="MetOfficeGlobalArchiveCollector"/> for the global 10km dataset.
///
/// Unlike Global Det (comparison-only), UKV is intended as a blender input —
/// see <c>memory/project_met_office_4way_bakeoff_prep.md</c> for the planned
/// 4-way bake-off introducing it as a 7th NWP alongside the existing 6 + AIFS.
///
/// Defaults (overridable for one-off backfills via the CLI command):
///   * Lookback = 3 days. AWS publishes ~3-6h after each cycle's run-time;
///     a 3-day window catches both today and yesterday's cycles plus any
///     that landed late.
///   * Cycles = 3,15 only. UKV runs hourly, but only the 03Z and 15Z runs
///     extend to 120h leads — every other cycle caps at ~24-48h and is
///     useless for our 24/48/72/120h blender lead buckets. Empirically
///     verified 2026-04-28 against the bucket listing.
///   * Leads = 24,48,72,120 — UKV publishes exactly to 120h, perfectly
///     covering our blender lead set.
///
/// The script is idempotent — re-pulling an existing date overwrites the
/// parquet with whatever AWS now has.
/// </summary>
public sealed class MetOfficeUkvArchiveCollector
{
    private const string ScriptName = "met_office_ukv_archive_backfill.py";
    private static readonly int[] DefaultCycles = { 3, 15 };
    private static readonly int[] DefaultLeads = { 24, 48, 72, 120 };
    private const int DefaultLookbackDays = 3;
    private const int DefaultParallelism = 8;
    // AWS publishes a cycle's NetCDFs ~3-6h after run time; a 7h floor avoids the
    // same-day "all 16 vars 404" noise on whichever 15Z run hasn't fully landed yet.
    private const int MinCycleAgeHours = 7;

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
