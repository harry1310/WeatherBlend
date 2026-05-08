using Microsoft.Extensions.Logging;
using WeatherBlend.Commands;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the GEFS ensemble-mean archive from NOAA's S3
/// bucket (<c>noaa-gefs-pds</c>). Thin wrapper around
/// <see cref="GefsBackfillCommand"/> with a 3-day rolling lookback so the
/// scheduled cron picks up newly-published cycles plus any that landed late
/// without re-fetching the whole archive.
///
/// **Not yet wired into <c>s3-collect</c>.** The collector class exists so
/// the live cron can pick up GEFS once we've validated the historical
/// backfill produces useful 2d/3d feature columns. Wire-up is a one-liner
/// in <c>S3CollectCommand</c> + <c>Program.cs</c> when ready.
///
/// NOAA publishes GEFS cycles ~3-6h after run time. 3-day lookback
/// comfortably covers this; 404s on not-yet-published cycles are skipped
/// silently by <see cref="GefsClient"/> per the existing backfill.
/// </summary>
public sealed class GefsArchiveCollector
{
    private const int DefaultLookbackDays = 3;

    private readonly GefsBackfillCommand _backfill;
    private readonly ILogger<GefsArchiveCollector> _log;

    public GefsArchiveCollector(
        GefsBackfillCommand backfill,
        ILogger<GefsArchiveCollector> log)
    {
        _backfill = backfill;
        _log = log;
    }

    public async Task<int> CollectAsync(CancellationToken ct)
    {
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        var start = today.AddDays(-DefaultLookbackDays);
        _log.LogInformation(
            "GEFS exact-runtime archive — refreshing {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} (all 4 cycles)",
            start, today);
        return await _backfill.RunAsync(start, today, cycles: null, ct);
    }
}
