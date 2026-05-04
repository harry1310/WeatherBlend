using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Historical backfill from the ECMWF Open Data AWS bucket
/// (<c>s3://ecmwf-forecasts/</c>). Same shape as
/// <see cref="GfsBackfillCommand"/> but with ECMWF's per-(cycle, lead)
/// GRIB2 files + JSON-Lines index sidecar via <see cref="EcmwfClient"/>.
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

    public async Task<int> RunAsync(
        string stream,
        DateOnly start,
        DateOnly end,
        IReadOnlyList<int>? cycles,
        IReadOnlyList<int>? leads,
        CancellationToken ct)
    {
        if (stream != EcmwfClient.Streams.IfsOper && stream != EcmwfClient.Streams.AifsOper)
        {
            _log.LogError("Unknown stream '{Stream}'. Expected: ifs | aifs.", stream);
            return 2;
        }
        cycles ??= EcmwfClient.CycleHours;
        var leadHours = (leads is { Count: > 0 } l)
            ? l
            : (_cfg.LeadHours.Count > 0
                ? _cfg.LeadHours
                : new List<int> { 6, 12, 24, 36, 48, 72, 96, 120, 144 });

        var scratchDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "tmp", $"ecmwf_{stream}");
        scratchDir = Path.GetFullPath(scratchDir);

        _log.LogInformation(
            "ECMWF {Stream} backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] scratch={Scratch}",
            stream, start, end, string.Join(',', cycles), string.Join(',', leadHours), scratchDir);

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
                        _cfg.Location, date, cc, stream, leadHours, scratchDir, ct);
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
