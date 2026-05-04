using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;

namespace WeatherBlend.Commands;

/// <summary>
/// Backfill historical Met Office forecasts from AWS Open Data — either the
/// Global Deterministic 10km dataset (<c>met_office_global</c>) or the UK
/// Deterministic 2km dataset (<c>met_office_ukv</c>). Thin wrapper around
/// <see cref="MetOfficeArchiveBackfillClient"/>; both share the same Python
/// script CLI surface so the wrapper only switches the script path.
///
/// AWS publishes from 2024-05-04 onwards (~12 months rolling window confirmed
/// 2026-05-05). Historical backfill always uses min-age=0 — every cycle in
/// range is far older than the ~3-6h publication lag.
/// </summary>
public sealed class MetOfficeArchiveBackfillCommand
{
    public static class Sources
    {
        public const string Global = "global";
        public const string Ukv    = "ukv";
    }

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
        string source,
        DateOnly start, DateOnly end,
        IReadOnlyList<int> cycles, IReadOnlyList<int> leads,
        int parallelism, CancellationToken ct)
    {
        var script = source switch
        {
            Sources.Global => "met_office_archive_backfill.py",
            Sources.Ukv    => "met_office_ukv_archive_backfill.py",
            _              => throw new ArgumentException(
                $"Unknown Met Office source '{source}'. Expected: global | ukv.", nameof(source)),
        };
        _log.LogInformation(
            "Met Office {Source} archive backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] parallelism={N}",
            source, start, end, string.Join(',', cycles), string.Join(',', leads), parallelism);
        return await _client.RunAsync(
            script, start, end, cycles, leads, parallelism, minCycleAgeHours: 0, ct);
    }
}
