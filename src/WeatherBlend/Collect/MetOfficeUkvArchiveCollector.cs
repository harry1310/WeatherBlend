using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the Met Office UK Deterministic 2km (UKV) archive
/// from AWS Open Data. Wraps <see cref="MetOfficeArchiveBackfillClient"/>
/// with collect-friendly defaults, mirroring
/// <see cref="MetOfficeGlobalArchiveCollector"/> for the global 10km dataset.
///
/// UKV is the 5th input to the 2d temperature blender (productionised
/// 2026-05-05+). It enters as an always-optional column: per ValidTime
/// hour, <see cref="WeatherBlend.Train.Exact12h.Exact12hFeatureBuilder"/>
/// pulls a single (cycle, lead) tuple that lands ~12h-ahead of that
/// ValidTime. UKV runs at 03Z + 15Z only, so the rule is:
///   V=00 → 15Z prev day + lead 9     V=12 → 03Z same day + lead 9
///   V=06 → 15Z prev day + lead 15    V=18 → 03Z same day + lead 15
/// targetLead = 24 (avg = 24h ahead):
///   V=00 → 03Z prev day + lead 21    V=12 → 15Z prev day + lead 21
///   V=06 → 03Z prev day + lead 27    V=18 → 15Z prev day + lead 27
///
/// Hence the live collector pulls leads 9 + 15 + 21 + 27 — exactly what
/// the per-V-hour rule reads at both blender leads. Earlier defaults of
/// {24, 48, 72, 120} were aimed at the rejected raw-MO direct-blender
/// bake-off (memory/project_met_office_raw_negative_result.md); narrowed
/// 2026-05-05 once UKV's only live consumer was 2d. Lead-24 picks added
/// 2026-05-05 — see Exact12hFeatureBuilder.UkvPicksForLead docstring for
/// the leak fix.
///
/// Defaults (overridable for one-off backfills via the CLI command):
///   * Lookback = 3 days. AWS publishes ~3-6h after each cycle's run-time;
///     a 3-day window catches both today and yesterday's cycles plus any
///     that landed late.
///   * Cycles = 3,15 only. UKV runs hourly, but only the 03Z and 15Z runs
///     extend past ~24h leads. The conditional rule above only ever needs
///     these two cycles per ValidTime — every other cycle is dropped.
///   * Leads = 9, 15, 21, 27 — exactly the leads 2d's per-V-hour
///     conditional pull reads from at lead-12 (9, 15) and lead-24 (21, 27).
///     9 + 15 average to 12h-ahead; 21 + 27 average to 24h-ahead. Each
///     pair matches the strictness of the other inputs at the same blender
///     lead. Adding lead-24 picks (21, 27) was the 2026-05-05 fix to a
///     leak where lead-24 training reused 12h-ahead UKV regardless.
///
/// The script is idempotent — re-pulling an existing date overwrites the
/// parquet with whatever AWS now has.
/// </summary>
public sealed class MetOfficeUkvArchiveCollector
{
    private const string ScriptName = "met_office_ukv_archive_backfill.py";
    private static readonly int[] DefaultCycles = { 3, 15 };
    // Leads 9 + 15 (lead-12 picks) + 21 + 27 (lead-24 picks) are exactly
    // what Exact12hFeatureBuilder.Build's UKV CTE reads under the
    // per-targetLead V=00/06/12/18 → (cycle, lead) mapping. Pulling more
    // wastes bandwidth (and AWS GET requests) on data nothing reads.
    private static readonly int[] DefaultLeads = { 9, 15, 21, 27 };
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
