using Microsoft.Extensions.Logging;
using WeatherBlend.Commands;

namespace WeatherBlend.Collect;

/// <summary>
/// Collect-time refresh of the ECMWF exact-runtime archive. Wraps
/// <see cref="EcmwfBackfillCommand"/> for both streams (IFS oper + AIFS
/// oper) with a 3-day rolling lookback. Sources from
/// <see cref="EcmwfClient.LiveBaseUrl"/> (ECMWF's own server) — the AWS
/// mirror's persistent S3 SlowDown throttling made it unusable for live
/// collection 2026-05-20; the 3-day lookback fits inside the live
/// endpoint's rolling window.
///
/// One <see cref="CollectAsync"/> per stream so the live <c>s3-collect</c>
/// command can report each one's success/failure independently — useful
/// because IFS and AIFS publish on different schedules (~7-8h and ~5-7h
/// after run time respectively) and one occasionally lags the other by
/// hours.
///
/// Lead set hardcoded inside the backfill command (1..120h, 6h-aligned).
/// Cycles default to all four (00/06/12/18Z) — IFS oper only publishes
/// 00/12Z to AWS Open Data so 06/18Z 404s are skipped silently; AIFS
/// publishes all four.
/// </summary>
public sealed class EcmwfArchiveCollector
{
    private const int DefaultLookbackDays = 3;

    private readonly EcmwfBackfillCommand _backfill;
    private readonly ILogger<EcmwfArchiveCollector> _log;

    public EcmwfArchiveCollector(
        EcmwfBackfillCommand backfill,
        ILogger<EcmwfArchiveCollector> log)
    {
        _backfill = backfill;
        _log = log;
    }

    public async Task<int> CollectIfsAsync(CancellationToken ct) => await CollectStreamAsync(EcmwfClient.Streams.IfsOper, ct);
    public async Task<int> CollectAifsAsync(CancellationToken ct) => await CollectStreamAsync(EcmwfClient.Streams.AifsOper, ct);

    private async Task<int> CollectStreamAsync(string stream, CancellationToken ct)
    {
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        var start = today.AddDays(-DefaultLookbackDays);
        _log.LogInformation(
            "ECMWF {Stream} exact-runtime archive — refreshing {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}",
            stream, start, today);
        return await _backfill.RunAsync(
            stream, start, today, cycles: null, EcmwfClient.LiveBaseUrl, ct);
    }
}
