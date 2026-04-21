using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3 backfill from NOAA's GFS archive on S3. Unlike the Open-Meteo
/// historical-forecast API, this archive is keyed by real cycle (00/06/12/18Z) and
/// lead hour, so every row carries an exact <c>RunTimeUtc</c>/<c>LeadHours</c> pair
/// tagged <c>RunTimeSource = "exact"</c> — the whole point of the exercise.
///
/// Output path matches the live collector: one parquet per (location, model, date, run).
/// Model id is <c>gfs_ncep</c> to distinguish these rows from Open-Meteo's
/// <c>gfs_seamless</c> in downstream joins.
/// </summary>
public sealed class GfsBackfillCommand
{
    private readonly AppConfig _cfg;
    private readonly GfsClient _gfs;
    private readonly ILogger<GfsBackfillCommand> _log;

    public GfsBackfillCommand(AppConfig cfg, GfsClient gfs, ILogger<GfsBackfillCommand> log)
    {
        _cfg = cfg;
        _gfs = gfs;
        _log = log;
    }

    public async Task<int> RunAsync(
        DateOnly start,
        DateOnly end,
        IReadOnlyList<int>? cycles,
        CancellationToken ct)
    {
        cycles ??= GfsClient.CycleHours;
        var leadHours = _cfg.LeadHours.Count > 0
            ? _cfg.LeadHours
            : new List<int> { 1, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168 };

        var scratchDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "tmp", "gfs");
        scratchDir = Path.GetFullPath(scratchDir);

        _log.LogInformation(
            "GFS backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] scratch={Scratch}",
            start, end, string.Join(',', cycles), string.Join(',', leadHours), scratchDir);

        var totalRows = 0;
        var errors = 0;

        for (var date = start; date <= end; date = date.AddDays(1))
        {
            foreach (var cc in cycles)
            {
                ct.ThrowIfCancellationRequested();
                try
                {
                    var rows = await _gfs.FetchCycleAsync(
                        _cfg.Location, date, cc, leadHours, scratchDir, ct);
                    if (rows.Count > 0)
                    {
                        await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                        totalRows += rows.Count;
                        _log.LogInformation("  gfs_ncep {Date:yyyy-MM-dd} t{CC:00}z: {Rows} rows",
                            date, cc, rows.Count);
                    }
                    else
                    {
                        _log.LogWarning("  gfs_ncep {Date:yyyy-MM-dd} t{CC:00}z: no rows", date, cc);
                    }
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  gfs_ncep {Date:yyyy-MM-dd} t{CC:00}z FAILED", date, cc);
                }
            }
        }

        _log.LogInformation("GFS backfill done: {Total} rows, {Errors} errors", totalRows, errors);
        return errors == 0 ? 0 : 1;
    }
}
