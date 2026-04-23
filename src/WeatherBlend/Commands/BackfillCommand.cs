using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Backfill historical data from one or more sources:
///   previous-runs - Open-Meteo Previous Runs API (per model, monthly chunks):
///                   returns separate columns per 24h-lead bucket (1..7 days), so we
///                   get real per-lead training signal at 24-167h. Rows tagged
///                   RunTimeSource=offset_day, LeadHours in {24,48,...,168}.
///   era5          - Open-Meteo ERA5 reanalysis archive (gapless training truth)
///   metar         - OGIMET historical METAR archive (verification truth, gappy)
///   rainfall      - EA Hydrology 15-min rainfall totals (precip verification truth)
///   all           - everything above
///
/// Previous-runs/ERA5 hit Open-Meteo (generous limits, 2s polite delay).
/// METAR hits OGIMET, which is rate-limited to ~5s between requests; one
/// request per (station, day). For 3 years × 2 stations that's ~2.5 hours.
/// </summary>
public sealed class BackfillCommand
{
    private readonly AppConfig _cfg;
    private readonly OpenMeteoClient _forecasts;
    private readonly Era5Client _era5;
    private readonly OgimetClient _ogimet;
    private readonly EaHydrologyClient _rainfall;
    private readonly ILogger<BackfillCommand> _log;

    public BackfillCommand(
        AppConfig cfg,
        OpenMeteoClient forecasts,
        Era5Client era5,
        OgimetClient ogimet,
        EaHydrologyClient rainfall,
        ILogger<BackfillCommand> log)
    {
        _cfg = cfg;
        _forecasts = forecasts;
        _era5 = era5;
        _ogimet = ogimet;
        _rainfall = rainfall;
        _log = log;
    }

    public async Task<int> RunAsync(string source, DateOnly start, DateOnly end, CancellationToken ct)
    {
        _log.LogInformation("Backfill source={Source} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} for {Location}",
            source, start, end, _cfg.Location.DisplayName);

        var src = source.ToLowerInvariant();
        var errors = 0;

        if (src is "previous-runs" or "all") errors += await BackfillPreviousRunsAsync(start, end, ct);
        if (src is "era5" or "all")          errors += await BackfillEra5Async(start, end, ct);
        if (src is "metar" or "all")         errors += await BackfillMetarAsync(start, end, ct);
        if (src is "rainfall" or "all")      errors += await BackfillRainfallAsync(start, end, ct);

        if (src is not ("previous-runs" or "era5" or "metar" or "rainfall" or "all"))
        {
            _log.LogError("Unknown source '{Source}'. Use: previous-runs | era5 | metar | rainfall | all", source);
            return 2;
        }

        return errors == 0 ? 0 : 1;
    }

    // ---- Previous Runs (per model, monthly chunks) -------------------------------
    //
    // Rows are tagged RunTimeSource=offset_day with LeadHours in {24,48,...,168}.
    // UKMO's archive on this endpoint starts later than the other five; for dates
    // before it went live the API returns all-null columns and ParsePreviousRuns
    // drops them, so early-date UKMO chunks will log 0 rows — that's expected.

    private async Task<int> BackfillPreviousRunsAsync(DateOnly start, DateOnly end, CancellationToken ct)
    {
        var errors = 0;
        foreach (var model in _cfg.Models)
        {
            var cursor = start;
            while (cursor <= end)
            {
                var chunkEnd = cursor.AddMonths(1).AddDays(-1);
                if (chunkEnd > end) chunkEnd = end;
                try
                {
                    var rows = await _forecasts.FetchPreviousRunsAsync(
                        _cfg.Location, model.Id, _cfg.Variables, cursor, chunkEnd, ct);
                    await ParquetWriter.WritePreviousRunsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                    _log.LogInformation("  previous-runs/{Model} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                        model.Id, cursor, chunkEnd, rows.Count);
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  previous-runs/{Model} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                        model.Id, cursor, chunkEnd);
                }
                await Task.Delay(TimeSpan.FromSeconds(2), ct);
                cursor = chunkEnd.AddDays(1);
            }
        }
        return errors;
    }

    // ---- ERA5 (monthly chunks, single source) ------------------------------------

    private async Task<int> BackfillEra5Async(DateOnly start, DateOnly end, CancellationToken ct)
    {
        var errors = 0;
        var cursor = start;
        while (cursor <= end)
        {
            var chunkEnd = cursor.AddMonths(1).AddDays(-1);
            if (chunkEnd > end) chunkEnd = end;
            try
            {
                var rows = await _era5.FetchAsync(_cfg.Location, cursor, chunkEnd, ct);
                await ParquetWriter.WriteEra5Async(_cfg.Storage.Era5Path, rows, ct);
                _log.LogInformation("  era5 {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                    cursor, chunkEnd, rows.Count);
            }
            catch (Exception ex)
            {
                errors++;
                _log.LogError(ex, "  era5 {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                    cursor, chunkEnd);
            }
            await Task.Delay(TimeSpan.FromSeconds(2), ct);
            cursor = chunkEnd.AddDays(1);
        }
        return errors;
    }

    // ---- OGIMET METAR (per station, 7-day chunks, 5s polite delay) ---------------

    private async Task<int> BackfillMetarAsync(DateOnly start, DateOnly end, CancellationToken ct)
    {
        var errors = 0;
        var stations = new[] { _cfg.Location.Metar.Primary, _cfg.Location.Metar.Fallback }
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Distinct()
            .ToArray();

        foreach (var station in stations)
        {
            var cursor = start;
            while (cursor <= end)
            {
                var chunkEnd = cursor.AddDays(6);
                if (chunkEnd > end) chunkEnd = end;
                var begin = cursor.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
                var endTime = chunkEnd.ToDateTime(new TimeOnly(23, 59), DateTimeKind.Utc);
                try
                {
                    var rows = await _ogimet.FetchAsync(_cfg.Location.Name, station, begin, endTime, ct);
                    await ParquetWriter.WriteObservationsAsync(_cfg.Storage.ObservationsPath, rows, ct);
                    _log.LogInformation("  metar/{Station} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                        station, cursor, chunkEnd, rows.Count);
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  metar/{Station} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                        station, cursor, chunkEnd);
                }
                // OGIMET recommendation is 20 requests / 10 min ≈ one every 30s.
                // 5s caused ~50% failure rate on the first full run — honour 30s now.
                await Task.Delay(TimeSpan.FromSeconds(30), ct);
                cursor = chunkEnd.AddDays(1);
            }
        }
        return errors;
    }

    // ---- EA rainfall (per station, monthly chunks) -------------------------------
    //
    // EA Hydrology has no rate-limit worth worrying about and a 2M-row hard cap
    // per call. A month of 15-min data is ~2900 rows so monthly chunks give
    // comfortable headroom and tidy per-chunk logs. A 1-second polite delay is
    // more than enough.

    private async Task<int> BackfillRainfallAsync(DateOnly start, DateOnly end, CancellationToken ct)
    {
        var errors = 0;
        foreach (var station in _cfg.Location.Rainfall.Stations)
        {
            var cursor = start;
            while (cursor <= end)
            {
                var chunkEnd = cursor.AddMonths(1).AddDays(-1);
                if (chunkEnd > end) chunkEnd = end;
                try
                {
                    var rows = await _rainfall.FetchAsync(
                        _cfg.Location.Name, station, cursor, chunkEnd, ct);
                    await ParquetWriter.WriteRainfallAsync(_cfg.Storage.RainfallPath, rows, ct);
                    _log.LogInformation("  rainfall/{Station} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                        station.Name, cursor, chunkEnd, rows.Count);
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  rainfall/{Station} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                        station.Name, cursor, chunkEnd);
                }
                await Task.Delay(TimeSpan.FromSeconds(1), ct);
                cursor = chunkEnd.AddDays(1);
            }
        }
        return errors;
    }
}
