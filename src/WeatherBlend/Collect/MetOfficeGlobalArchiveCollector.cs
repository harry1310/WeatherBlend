using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the Met Office Global Deterministic 10km archive
/// from AWS Open Data. Wraps <see cref="MetOfficeArchiveBackfillClient"/>
/// with collect-friendly defaults so <c>CollectCommand</c> can call it like
/// any other collector.
///
/// Met Office Global Det is a COMPARISON BASELINE
/// (memory/project_met_office_global_role.md), not a blender input. The
/// blenders never query <c>model=met_office_global</c>. Refreshing it daily
/// keeps the comparison current as the blenders' test windows advance.
///
/// Defaults (overridable for one-off backfills via the CLI command):
///   * Lookback = 3 days. AWS publishes ~3-6h after each cycle's run-time;
///     a 3-day window catches both today and yesterday's cycles plus any
///     that landed late.
///   * Cycles = 0,12 only. 06Z and 18Z run a 20h horizon and are useless
///     for our 24/48/72h leads.
///   * Leads = 24,48,72 — same as the rest of WeatherBlend.
///
/// The script is idempotent — re-pulling an existing date overwrites the
/// parquet with whatever AWS now has. Net cost per collect run is small
/// (under 100 NetCDF GETs even on day-one of a fresh tree).
/// </summary>
public sealed class MetOfficeGlobalArchiveCollector
{
    private static readonly int[] DefaultCycles = { 0, 12 };
    private static readonly int[] DefaultLeads = { 24, 48, 72 };
    private const int DefaultLookbackDays = 3;
    private const int DefaultParallelism = 8;

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
            "Met Office Global Det archive — refreshing {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}]",
            start, today, string.Join(',', DefaultCycles), string.Join(',', DefaultLeads));
        return await _client.RunAsync(
            start, today, DefaultCycles, DefaultLeads, DefaultParallelism, ct);
    }
}
