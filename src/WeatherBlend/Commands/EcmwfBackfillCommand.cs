using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Date-range fetch of ECMWF IFS/AIFS oper cycles. Same shape as
/// <see cref="GfsBackfillCommand"/> but with ECMWF's per-(cycle, lead)
/// GRIB2 files + JSON-Lines index sidecar via <see cref="EcmwfClient"/>.
/// Shared by the live s3-collect refresh (via <see cref="EcmwfArchiveCollector"/>,
/// recent window, <see cref="EcmwfClient.LiveBaseUrl"/>) and the historical
/// <c>ecmwf-backfill</c> CLI (deep archive, <see cref="EcmwfClient.ArchiveBaseUrl"/>) —
/// the caller passes the endpoint to <see cref="RunAsync"/>.
///
/// Two streams supported, sharing all the parsing infrastructure:
///   * <c>ifs</c>  — IFS oper deterministic forecast (~2y 4m archive,
///                   2023-01-18+). Model id stamped as <c>ecmwf_ifs_oper</c>
///                   (distinct from Open-Meteo's <c>ecmwf_ifs025</c>).
///   * <c>aifs</c> — AIFS deterministic AI forecast (~1y 2m archive,
///                   2024-02-29+). Model id stamped as <c>ecmwf_aifs_oper</c>
///                   (distinct from Open-Meteo's <c>ecmwf_aifs025_single</c>).
///
/// Every row carries <see cref="WeatherBlend.Models.RunTimeSources.Exact"/>
/// — RunTime + ValidTime + LeadHours come from the file path / filename,
/// not from a "reported by API" approximation.
/// </summary>
public sealed class EcmwfBackfillCommand
{
    private readonly AppConfig _cfg;
    private readonly EcmwfClient _ecmwf;
    private readonly ILogger<EcmwfBackfillCommand> _log;

    public EcmwfBackfillCommand(AppConfig cfg, EcmwfClient ecmwf, ILogger<EcmwfBackfillCommand> log)
    {
        _cfg = cfg;
        _ecmwf = ecmwf;
        _log = log;
    }

    /// <param name="apiBaseUrl">
    /// ECMWF endpoint — <see cref="EcmwfClient.LiveBaseUrl"/> for the
    /// recent-window s3-collect refresh, <see cref="EcmwfClient.ArchiveBaseUrl"/>
    /// for historical backfill. See <see cref="EcmwfClient"/> for why the two
    /// endpoints exist.
    /// </param>
    public async Task<int> RunAsync(
        string stream,
        DateOnly start,
        DateOnly end,
        IReadOnlyList<int>? cycles,
        string apiBaseUrl,
        CancellationToken ct)
    {
        if (stream != EcmwfClient.Streams.IfsOper && stream != EcmwfClient.Streams.AifsOper)
        {
            _log.LogError("Unknown stream '{Stream}'. Expected: ifs | aifs.", stream);
            return 2;
        }
        cycles ??= EcmwfClient.CycleHours;
        // Capture set capped at 120h (5 days) — see GfsBackfillCommand for
        // rationale. Both IFS oper (3h step to 144, 6h step to 240) and AIFS
        // oper (6h step to 360) align cleanly to a 6h grid, so the same set
        // works for both streams.
        var leadHours = new[]
        {
            6, 12, 24, 36, 48, 72, 96, 120,
        };

        var scratchDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "tmp", $"ecmwf_{stream}");
        scratchDir = Path.GetFullPath(scratchDir);

        _log.LogInformation(
            "ECMWF {Stream} backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] endpoint={Endpoint} scratch={Scratch}",
            stream, start, end, string.Join(',', cycles), string.Join(',', leadHours), apiBaseUrl, scratchDir);

        var totalRows = 0;
        var errors = 0;

        for (var date = start; date <= end; date = date.AddDays(1))
        {
            foreach (var cc in cycles)
            {
                ct.ThrowIfCancellationRequested();
                try
                {
                    var rows = await _ecmwf.FetchCycleAsync(
                        _cfg.Location, date, cc, stream, leadHours, scratchDir, apiBaseUrl, ct);
                    if (rows.Count > 0)
                    {
                        await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                        totalRows += rows.Count;
                        _log.LogInformation("  ecmwf_{Stream} {Date:yyyy-MM-dd} {CC:00}z: {Rows} rows",
                            stream, date, cc, rows.Count);
                    }
                    else
                    {
                        _log.LogWarning("  ecmwf_{Stream} {Date:yyyy-MM-dd} {CC:00}z: no rows",
                            stream, date, cc);
                    }
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  ecmwf_{Stream} {Date:yyyy-MM-dd} {CC:00}z FAILED", stream, date, cc);
                }
            }
        }

        _log.LogInformation("ECMWF {Stream} backfill done: {Total} rows, {Errors} errors",
            stream, totalRows, errors);
        return errors == 0 ? 0 : 1;
    }
}
