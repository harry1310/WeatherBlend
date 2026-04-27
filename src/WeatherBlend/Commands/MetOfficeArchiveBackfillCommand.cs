using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;

namespace WeatherBlend.Commands;

/// <summary>
/// Backfill historical Met Office Global Deterministic 10km forecasts from
/// AWS Open Data into <c>data/forecasts/.../model=met_office_global/</c>.
/// Thin wrapper around <see cref="MetOfficeArchiveBackfillClient"/> which
/// shells out to the Python script.
///
/// Cycle defaults to 0,12 only — 06Z and 18Z runs go out 20h, useless for
/// our 24/48/72h leads. AWS keeps a 2-year rolling window so the practical
/// useful start date is two years ago.
/// </summary>
public sealed class MetOfficeArchiveBackfillCommand
{
    private readonly MetOfficeArchiveBackfillClient _client;
    private readonly ILogger<MetOfficeArchiveBackfillCommand> _log;

    public MetOfficeArchiveBackfillCommand(
        MetOfficeArchiveBackfillClient client,
        ILogger<MetOfficeArchiveBackfillCommand> log)
    {
        _client = client;
        _log = log;
    }

    public async Task<int> RunAsync(
        DateOnly start, DateOnly end,
        IReadOnlyList<int> cycles, IReadOnlyList<int> leads,
        int parallelism, CancellationToken ct)
    {
        _log.LogInformation(
            "Met Office archive backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] parallelism={N}",
            start, end, string.Join(',', cycles), string.Join(',', leads), parallelism);
        // Historical backfill — every cycle in range is far older than any sensible
        // publication delay, so the min-age floor is 0 (no skip).
        return await _client.RunAsync(start, end, cycles, leads, parallelism, minCycleAgeHours: 0, ct);
    }
}
