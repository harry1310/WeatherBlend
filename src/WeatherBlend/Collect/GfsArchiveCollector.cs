using Microsoft.Extensions.Logging;
using WeatherBlend.Commands;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the GFS exact-runtime archive from NOAA's S3
/// bucket (<c>noaa-gfs-bdp-pds</c>). Thin wrapper around
/// <see cref="GfsBackfillCommand"/> with a 3-day rolling lookback so the
/// scheduled cron picks up newly-published cycles + any that landed late
/// without re-fetching the entire archive.
///
/// Wired into the live <c>s3-collect</c> command alongside the ECMWF and
/// Met Office equivalents — these are the four exact-runtime sources the
/// 2d temperature blender (productionised 2026-05-05+) consumes. The
/// existing live collect via Open-Meteo (<c>collect</c>) is unrelated and
/// keeps running for the 2b/2c blenders.
///
/// Defaults — same lead set as the GFS backfill (1..120h on the 6h-aligned
/// grid) so the collected cycles can serve any blender lead bucket. All
/// four cycles per day (00/06/12/18Z) since GFS publishes them all.
///
/// NOAA publishes ~3-4h after run time. The 3-day lookback comfortably
/// covers this; 404s on not-yet-published cycles are skipped silently by
/// <see cref="GfsClient"/> per the existing backfill behaviour.
/// </summary>
public sealed class GfsArchiveCollector
{
    private const int DefaultLookbackDays = 3;

    private readonly GfsBackfillCommand _backfill;
    private readonly ILogger<GfsArchiveCollector> _log;

    public GfsArchiveCollector(
        GfsBackfillCommand backfill,
        ILogger<GfsArchiveCollector> log)
    {
        _backfill = backfill;
        _log = log;
    }

    public async Task<int> CollectAsync(CancellationToken ct)
    {
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        var start = today.AddDays(-DefaultLookbackDays);
        _log.LogInformation(
            "GFS exact-runtime archive — refreshing {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} (all 4 cycles)",
            start, today);
        // cycles=null → backfill defaults to all four (00/06/12/18Z).
        // Lead set is hardcoded inside GfsBackfillCommand (1..120h, 6h-aligned).
        return await _backfill.RunAsync(start, today, cycles: null, ct);
    }
}
