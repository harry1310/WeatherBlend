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
/// Open-Meteo Previous Runs is throttled by HttpConfig.PreviousRunsBackfillDelaySeconds
/// (default 15s); that endpoint uses an hourly token bucket and a 30 min burst at
/// ~5 calls/min tipped the limit on 2026-04-25. ERA5 has its own (much smaller) load
/// profile and uses a hardcoded 2s delay. METAR hits OGIMET, rate-limited to ~30s
/// between requests (one per station per 7-day chunk; 3y × 2 stations ≈ 2.5h).
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

    public async Task<int> RunAsync(string source, DateOnly start, DateOnly end, string? model, string? location, CancellationToken ct)
    {
        // Resolve the locations to iterate. Default (location==null) = all
        // configured. With --location, restrict to that single named entry —
        // useful for adding a new location without re-pulling existing
        // locations' historical data (which is otherwise wastefully
        // idempotent against the OM previous_runs rate limiter).
        var src = source.ToLowerInvariant();
        var locations = ResolveLocations(location);
        if (locations is null) return 2;   // unknown location name; ResolveLocations already logged

        _log.LogInformation("Backfill source={Source} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}{ModelFilter} — {N} location(s): [{Names}]",
            source, start, end, model is null ? "" : $" model={model}",
            locations.Count, string.Join(", ", locations.Select(l => l.Name)));

        var errors = 0;

        if (src is "previous-runs" or "all") errors += await BackfillPreviousRunsAsync(start, end, model, locations, ct);
        if (src is "era5" or "all")          errors += await BackfillEra5Async(start, end, locations, ct);
        if (src is "metar" or "all")         errors += await BackfillMetarAsync(start, end, locations, ct);
        if (src is "rainfall" or "all")      errors += await BackfillRainfallAsync(start, end, locations, ct);
        // hist-forecast is deliberately NOT part of "all": it's a one-off gap-fill
        // for the upper-air (…hPa) history the Previous Runs API can't serve.
        // Recent pressure accrues forward via the live collector (reported rows).
        if (src is "hist-forecast")          errors += await BackfillHistoricalForecastAsync(start, end, model, locations, ct);

        if (src is not ("previous-runs" or "era5" or "metar" or "rainfall" or "hist-forecast" or "all"))
        {
            _log.LogError("Unknown source '{Source}'. Use: previous-runs | era5 | metar | rainfall | hist-forecast | all", source);
            return 2;
        }

        return errors == 0 ? 0 : 1;
    }

    /// <summary>Locations to iterate for previous-runs / rainfall. Returns
    /// null when <paramref name="locationFilter"/> doesn't match any
    /// configured Locations[].Name — caller treats null as exit-2.</summary>
    private List<LocationConfig>? ResolveLocations(string? locationFilter)
    {
        var all = _cfg.Locations.Count > 0 ? _cfg.Locations : new List<LocationConfig> { _cfg.Location };
        if (string.IsNullOrWhiteSpace(locationFilter)) return all;
        var match = all.FirstOrDefault(l => string.Equals(l.Name, locationFilter, StringComparison.OrdinalIgnoreCase));
        if (match is null)
        {
            _log.LogError("--location '{L}' not in configured locations. Available: [{Avail}]",
                locationFilter, string.Join(", ", all.Select(l => l.Name)));
            return null;
        }
        return new List<LocationConfig> { match };
    }

    // ---- Previous Runs (per model, monthly chunks) -------------------------------
    //
    // Rows are tagged RunTimeSource=offset_day with LeadHours in {24,48,...,168}.
    // UKMO's archive on this endpoint starts later than the other five; for dates
    // before it went live the API returns all-null columns and ParsePreviousRuns
    // drops them, so early-date UKMO chunks will log 0 rows — that's expected.

    private async Task<int> BackfillPreviousRunsAsync(DateOnly start, DateOnly end, string? modelFilter, IReadOnlyList<LocationConfig> locations, CancellationToken ct)
    {
        var errors = 0;
        var models = modelFilter is null
            ? (IEnumerable<ModelConfig>)_cfg.Models
            : _cfg.Models.Where(m => string.Equals(m.Id, modelFilter, StringComparison.OrdinalIgnoreCase));
        if (modelFilter is not null && !models.Any())
        {
            _log.LogError("--model '{M}' not in configured models list. Available: [{Avail}]",
                modelFilter, string.Join(", ", _cfg.Models.Select(m => m.Id)));
            return 1;
        }
        // Multi-location 2026-05-11: iterate the locations the caller
        // resolved (all configured OR a single one when --location is
        // set). Open-Meteo previous_runs is keyed by (lat, lon) so
        // different locations get independently keyed parquet partitions
        // (location=<name>/model=<id>/date=*/run=*.parquet). The token-
        // bucket rate limit is per-API-key not per-location, so the
        // existing PreviousRunsBackfillDelaySeconds knob still applies
        // across the combined fetch volume.
        foreach (var location in locations)
        {
            _log.LogInformation("previous-runs backfill: location={Name} ({Lat:0.0000}, {Lon:0.0000})",
                location.Name, location.Latitude, location.Longitude);
            foreach (var model in models)
            {
                var cursor = start;
                while (cursor <= end)
                {
                    var chunkEnd = cursor.AddMonths(1).AddDays(-1);
                    if (chunkEnd > end) chunkEnd = end;
                    try
                    {
                        var rows = await _forecasts.FetchPreviousRunsAsync(
                            location, model.Id, _cfg.Variables.Forecast, cursor, chunkEnd, ct);
                        await ParquetWriter.WritePreviousRunsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                        _log.LogInformation("  previous-runs/{Name}/{Model} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                            location.Name, model.Id, cursor, chunkEnd, rows.Count);
                    }
                    catch (Exception ex)
                    {
                        errors++;
                        _log.LogError(ex, "  previous-runs/{Name}/{Model} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                            location.Name, model.Id, cursor, chunkEnd);
                    }
                    await Task.Delay(TimeSpan.FromSeconds(_cfg.Http.PreviousRunsBackfillDelaySeconds), ct);
                    cursor = chunkEnd.AddDays(1);
                }
            }
        }
        return errors;
    }

    // ---- Historical forecast pressure (monthly chunks, per model) ----------------
    // Backfills ONLY the pressure-level (…hPa) fields the Previous Runs API rejects
    // (400). Source: Open-Meteo Historical Forecast API (~2022→present). The data
    // is lead-unlabelled (RunTimeSource=hist_forecast) and lands in a SEPARATE
    // file — date=*/hist_forecast.parquet — alongside previous_runs.parquet, never
    // overwriting it. Models that don't archive upper-air levels (e.g. UKMO, AIFS)
    // return all-null columns → 0 rows, logged and skipped (not an error).

    private async Task<int> BackfillHistoricalForecastAsync(DateOnly start, DateOnly end, string? modelFilter, IReadOnlyList<LocationConfig> locations, CancellationToken ct)
    {
        var errors = 0;
        var models = modelFilter is null
            ? (IEnumerable<ModelConfig>)_cfg.Models
            : _cfg.Models.Where(m => string.Equals(m.Id, modelFilter, StringComparison.OrdinalIgnoreCase));
        if (modelFilter is not null && !models.Any())
        {
            _log.LogError("--model '{M}' not in configured models list. Available: [{Avail}]",
                modelFilter, string.Join(", ", _cfg.Models.Select(m => m.Id)));
            return 1;
        }
        foreach (var location in locations)
        {
            _log.LogInformation("hist-forecast backfill: location={Name} ({Lat:0.0000}, {Lon:0.0000})",
                location.Name, location.Latitude, location.Longitude);
            foreach (var model in models)
            {
                var cursor = start;
                while (cursor <= end)
                {
                    var chunkEnd = cursor.AddMonths(1).AddDays(-1);
                    if (chunkEnd > end) chunkEnd = end;
                    try
                    {
                        var rows = await _forecasts.FetchHistoricalForecastAsync(
                            location, model.Id, cursor, chunkEnd, ct);
                        await ParquetWriter.WriteHistoricalForecastAsync(_cfg.Storage.ForecastsPath, rows, ct);
                        _log.LogInformation("  hist-forecast/{Name}/{Model} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                            location.Name, model.Id, cursor, chunkEnd, rows.Count);
                    }
                    catch (Exception ex)
                    {
                        errors++;
                        _log.LogError(ex, "  hist-forecast/{Name}/{Model} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                            location.Name, model.Id, cursor, chunkEnd);
                    }
                    await Task.Delay(TimeSpan.FromSeconds(_cfg.Http.PreviousRunsBackfillDelaySeconds), ct);
                    cursor = chunkEnd.AddDays(1);
                }
            }
        }
        return errors;
    }

    // ---- ERA5 (monthly chunks, single source) ------------------------------------

    private async Task<int> BackfillEra5Async(DateOnly start, DateOnly end, IReadOnlyList<LocationConfig> locations, CancellationToken ct)
    {
        var errors = 0;
        // Multi-location 2026-05-14: iterate every configured location so
        // Membury (or any future addition) accumulates ERA5 partitions under
        // data/truth/era5/location=<name>/ without a separate command. The
        // monthly chunk loop is unchanged per-location — Open-Meteo's ERA5
        // archive is keyed by (lat, lon) so each location is an independent
        // fetch.
        foreach (var location in locations)
        {
            _log.LogInformation("era5 backfill: location={Name} ({Lat:0.0000}, {Lon:0.0000})",
                location.Name, location.Latitude, location.Longitude);
            var cursor = start;
            while (cursor <= end)
            {
                var chunkEnd = cursor.AddMonths(1).AddDays(-1);
                if (chunkEnd > end) chunkEnd = end;
                try
                {
                    var rows = await _era5.FetchAsync(location, cursor, chunkEnd, ct);
                    await ParquetWriter.WriteEra5Async(_cfg.Storage.Era5Path, rows, ct);
                    _log.LogInformation("  era5/{Name} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                        location.Name, cursor, chunkEnd, rows.Count);
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  era5/{Name} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                        location.Name, cursor, chunkEnd);
                }
                await Task.Delay(TimeSpan.FromSeconds(2), ct);
                cursor = chunkEnd.AddDays(1);
            }
        }
        return errors;
    }

    // ---- OGIMET METAR (per station, 7-day chunks, 5s polite delay) ---------------

    private async Task<int> BackfillMetarAsync(DateOnly start, DateOnly end, IReadOnlyList<LocationConfig> locations, CancellationToken ct)
    {
        var errors = 0;
        // METAR text is per-ICAO. Two locations can map to the same airport
        // (Bonehill + Membury both use EGTE/EGDY). Dedupe by ICAO across all
        // locations so we only hit OGIMET once per unique station per chunk;
        // pick the FIRST location that lists each ICAO as its data-tagger
        // (LocationName column in the parquet). The read path
        // (TruthRepository.GetMetarTemperature) is keyed by Station only,
        // so subsequent locations sharing the same ICAO can read the same
        // rows without their own backfill.
        var seen = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var location in locations)
        {
            foreach (var icao in new[] { location.Metar.Primary, location.Metar.Fallback })
            {
                if (string.IsNullOrWhiteSpace(icao)) continue;
                if (!seen.ContainsKey(icao)) seen[icao] = location.Name;
            }
        }
        if (seen.Count == 0)
        {
            _log.LogInformation("metar backfill: no locations have METAR ICAOs configured — skipping.");
            return 0;
        }

        foreach (var (station, locationName) in seen)
        {
            _log.LogInformation("metar backfill: station={Station} tagging location={Loc} (shared across {N} locations)",
                station, locationName,
                locations.Count(l =>
                    string.Equals(l.Metar.Primary, station, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(l.Metar.Fallback, station, StringComparison.OrdinalIgnoreCase)));
            var cursor = start;
            while (cursor <= end)
            {
                var chunkEnd = cursor.AddDays(6);
                if (chunkEnd > end) chunkEnd = end;
                var begin = cursor.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
                var endTime = chunkEnd.ToDateTime(new TimeOnly(23, 59), DateTimeKind.Utc);
                try
                {
                    var rows = await _ogimet.FetchAsync(locationName, station, begin, endTime, ct);
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

    private async Task<int> BackfillRainfallAsync(DateOnly start, DateOnly end, IReadOnlyList<LocationConfig> locations, CancellationToken ct)
    {
        var errors = 0;
        // Multi-location 2026-05-11: each location has its own stations
        // list (e.g. Bonehill = 3 Dartmoor gauges, Membury = 3 East Devon
        // gauges). Parquet partitions on (location, station) so the
        // rainfall tree fans out as
        // data/truth/rainfall/location=<name>/station=<n>/date=*/observations.parquet
        // — no collision risk across locations even if a station name
        // happens to coincide. Locations missing the rainfall block
        // (.Stations empty) are silently skipped.
        foreach (var location in locations)
        {
            if (location.Rainfall.Stations.Count == 0)
            {
                _log.LogInformation("rainfall backfill: location={Name} has no rainfall stations configured — skipping",
                    location.Name);
                continue;
            }
            _log.LogInformation("rainfall backfill: location={Name} — {N} stations",
                location.Name, location.Rainfall.Stations.Count);
            foreach (var station in location.Rainfall.Stations)
            {
                var cursor = start;
                while (cursor <= end)
                {
                    var chunkEnd = cursor.AddMonths(1).AddDays(-1);
                    if (chunkEnd > end) chunkEnd = end;
                    try
                    {
                        var rows = await _rainfall.FetchAsync(
                            location.Name, station, cursor, chunkEnd, ct);
                        await ParquetWriter.WriteRainfallAsync(_cfg.Storage.RainfallPath, rows, ct);
                        _log.LogInformation("  rainfall/{Name}/{Station} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd}: {Rows} rows",
                            location.Name, station.Name, cursor, chunkEnd, rows.Count);
                    }
                    catch (Exception ex)
                    {
                        errors++;
                        _log.LogError(ex, "  rainfall/{Name}/{Station} {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} FAILED",
                            location.Name, station.Name, cursor, chunkEnd);
                    }
                    await Task.Delay(TimeSpan.FromSeconds(1), ct);
                    cursor = chunkEnd.AddDays(1);
                }
            }
        }
        return errors;
    }
}
