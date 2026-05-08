using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Historical backfill from NOAA GEFS's S3 archive (<c>s3://noaa-gefs-pds/</c>).
/// Mirrors <see cref="GfsBackfillCommand"/> but reads the 31-member ensemble's
/// mean (<c>geavg</c>) from the 0.5° <c>pgrb2ap5</c> product. Output rows are
/// stamped <c>Model = "gefs_ncep_mean"</c> and tagged
/// <c>RunTimeSource = "exact"</c> so they sit alongside the deterministic
/// <c>gfs_ncep</c> column without overlapping.
///
/// Archive depth verified back to 2017-01-01 — comfortable margin on the
/// 1-2y window the blender trains over. Lead set capped at f120 so the
/// backfill stays inside the dense 3-hourly segment of GEFS's grid (3h to
/// f240, 6h to f384) and matches the lead set we use for the other
/// exact-runtime sources.
/// </summary>
public sealed class GefsBackfillCommand
{
    private readonly AppConfig _cfg;
    private readonly GefsClient _gefs;
    private readonly ILogger<GefsBackfillCommand> _log;

    public GefsBackfillCommand(AppConfig cfg, GefsClient gefs, ILogger<GefsBackfillCommand> log)
    {
        _cfg = cfg;
        _gefs = gefs;
        _log = log;
    }

    public async Task<int> RunAsync(
        DateOnly start,
        DateOnly end,
        IReadOnlyList<int>? cycles,
        CancellationToken ct)
    {
        cycles ??= GefsClient.CycleHours;
        // GEFS pgrb2a publishes f000, f003, ..., f240, then f246..f384 at 6h
        // step. Backfill stays at the leads we'd use in 2d/3d to keep wire
        // traffic bounded. Same set as GfsBackfillCommand's f120 cap so a
        // joined (gfs_ncep, gefs_ncep_mean) row pair always exists at
        // matching valid_times.
        var leadHours = new[]
        {
            3, 6, 12, 24, 36, 48, 72, 96, 120,
        };

        var scratchDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "tmp", "gefs");
        scratchDir = Path.GetFullPath(scratchDir);

        _log.LogInformation(
            "GEFS backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} cycles=[{Cycles}] leads=[{Leads}] scratch={Scratch}",
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
                    var rows = await _gefs.FetchCycleAsync(
                        _cfg.Location, date, cc, leadHours, scratchDir, ct);
                    if (rows.Count > 0)
                    {
                        await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                        totalRows += rows.Count;
                        _log.LogInformation("  gefs_ncep_mean {Date:yyyy-MM-dd} t{CC:00}z: {Rows} rows",
                            date, cc, rows.Count);
                    }
                    else
                    {
                        _log.LogWarning("  gefs_ncep_mean {Date:yyyy-MM-dd} t{CC:00}z: no rows", date, cc);
                    }
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  gefs_ncep_mean {Date:yyyy-MM-dd} t{CC:00}z FAILED", date, cc);
                }
            }
        }

        _log.LogInformation("GEFS backfill done: {Total} rows, {Errors} errors", totalRows, errors);
        return errors == 0 ? 0 : 1;
    }
}
