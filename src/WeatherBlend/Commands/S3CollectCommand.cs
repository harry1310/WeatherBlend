using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;

namespace WeatherBlend.Commands;

/// <summary>
/// Live collect of the five exact-runtime forecast sources used by the 2d
/// temperature blender (productionised 2026-05-05+):
///   - GFS              (NOAA S3, ~3-4h publication latency)
///   - ECMWF IFS oper   (AWS Open Data, ~7-8h ← slowest publisher)
///   - ECMWF AIFS oper  (AWS Open Data, ~5-7h)
///   - Met Office Global (Met Office AWS, ~3-6h, NetCDF via Python subprocess)
///   - Met Office UKV    (Met Office AWS, ~3-6h, NetCDF via Python subprocess)
///
/// Each source pulls a 3-day rolling window — comfortably covers today's
/// cycles + late landings. Any not-yet-published cycles 404 silently per the
/// underlying backfill clients' existing behaviour, so the collect is safe
/// to run as often as the cron fires (typically every 6h, paired with the
/// cycle landing cadence).
///
/// Per-source success/failure tracked separately so a single source's
/// outage doesn't cascade — exit code is the OR of failures (non-zero if
/// any source produced no rows or errored, but the others still get to run).
///
/// CLI: <c>s3-collect [--sources gfs,ifs,aifs,mo-global,ukv]</c>. Default =
/// all five. Useful subset for dev: <c>--sources gfs</c> (cheapest to test).
/// </summary>
public sealed class S3CollectCommand
{
    public static class Sources
    {
        public const string Gfs      = "gfs";
        public const string Ifs      = "ifs";
        public const string Aifs     = "aifs";
        public const string MoGlobal = "mo-global";
        public const string Ukv      = "ukv";
        public static readonly string[] All = { Gfs, Ifs, Aifs, MoGlobal, Ukv };
    }

    private readonly GfsArchiveCollector _gfs;
    private readonly EcmwfArchiveCollector _ecmwf;
    private readonly MetOfficeGlobalArchiveCollector _moGlobal;
    private readonly MetOfficeUkvArchiveCollector _moUkv;
    private readonly ILogger<S3CollectCommand> _log;

    public S3CollectCommand(
        GfsArchiveCollector gfs,
        EcmwfArchiveCollector ecmwf,
        MetOfficeGlobalArchiveCollector moGlobal,
        MetOfficeUkvArchiveCollector moUkv,
        ILogger<S3CollectCommand> log)
    {
        _gfs = gfs;
        _ecmwf = ecmwf;
        _moGlobal = moGlobal;
        _moUkv = moUkv;
        _log = log;
    }

    public async Task<int> RunAsync(IReadOnlyList<string>? sources, CancellationToken ct)
    {
        sources ??= Sources.All;
        // Validate up-front so a typo in the workflow input fails fast
        // rather than silently skipping a source.
        var unknown = sources.Where(s => !Sources.All.Contains(s, StringComparer.OrdinalIgnoreCase)).ToList();
        if (unknown.Count > 0)
        {
            _log.LogError(
                "Unknown source(s) [{Unknown}]. Expected any of: {Valid}.",
                string.Join(",", unknown), string.Join(",", Sources.All));
            return 2;
        }

        var distinct = sources.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        _log.LogInformation(
            "S3 live collect — sources=[{Sources}] running in parallel (sequential per source)",
            string.Join(",", distinct));
        var t0 = DateTime.UtcNow;

        // Each source writes to its own model= partition (gfs_ncep,
        // ecmwf_ifs_oper, ecmwf_aifs_oper, met_office_global,
        // met_office_ukv) so there's no filesystem contention across sources.
        // Within a source the per-cycle iteration stays sequential — the
        // upstream collectors haven't been audited for thread safety on
        // their internal HTTP/wgrib2/python state. Net effect: wall-clock
        // ≈ slowest source instead of sum-of-all-sources. Each source is
        // wrapped in its own try/catch so one source's outage doesn't fail
        // the others (preserves the original per-source isolation).
        var tasks = distinct.Select(src => Task.Run(async () =>
        {
            var srcStart = DateTime.UtcNow;
            try
            {
                ct.ThrowIfCancellationRequested();
                var exit = await CollectOneAsync(src, ct);
                var elapsed = DateTime.UtcNow - srcStart;
                if (exit != 0)
                {
                    _log.LogWarning(
                        "Source {Src} returned non-zero exit {Exit} after {Elapsed}",
                        src, exit, elapsed);
                    return 1;
                }
                _log.LogInformation("Source {Src} OK after {Elapsed}", src, elapsed);
                return 0;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                _log.LogError(ex,
                    "Source {Src} threw after {Elapsed} — other sources continuing.",
                    src, DateTime.UtcNow - srcStart);
                return 1;
            }
        }, ct)).ToArray();

        var results = await Task.WhenAll(tasks);
        var failures = results.Sum();

        _log.LogInformation(
            "S3 live collect done in {Elapsed} — {N} source(s), {F} failure(s)",
            DateTime.UtcNow - t0, distinct.Count, failures);
        return failures == 0 ? 0 : 1;
    }

    private Task<int> CollectOneAsync(string source, CancellationToken ct) => source.ToLowerInvariant() switch
    {
        Sources.Gfs      => _gfs.CollectAsync(ct),
        Sources.Ifs      => _ecmwf.CollectIfsAsync(ct),
        Sources.Aifs     => _ecmwf.CollectAifsAsync(ct),
        Sources.MoGlobal => _moGlobal.CollectAsync(ct),
        Sources.Ukv      => _moUkv.CollectAsync(ct),
        _ => throw new ArgumentException($"Unknown source '{source}'.", nameof(source)),
    };
}
