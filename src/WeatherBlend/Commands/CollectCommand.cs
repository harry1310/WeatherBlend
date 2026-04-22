using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Runs a single collection cycle: one forecast pull per configured model,
/// plus one METAR pull from the primary station (falling back to secondary).
/// Idempotent - safe to run multiple times per hour; data will be overwritten
/// (forecasts) or deduped (observations).
/// </summary>
public sealed class CollectCommand
{
    private readonly AppConfig _cfg;
    private readonly OpenMeteoClient _forecasts;
    private readonly MetarClient _metar;
    private readonly EaHydrologyClient _rainfall;
    private readonly ILogger<CollectCommand> _log;

    public CollectCommand(
        AppConfig cfg,
        OpenMeteoClient forecasts,
        MetarClient metar,
        EaHydrologyClient rainfall,
        ILogger<CollectCommand> log)
    {
        _cfg = cfg;
        _forecasts = forecasts;
        _metar = metar;
        _rainfall = rainfall;
        _log = log;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        _log.LogInformation("Collecting for {Location} ({Lat:F4}, {Lon:F4}, {Elev}m)",
            _cfg.Location.DisplayName,
            _cfg.Location.Latitude,
            _cfg.Location.Longitude,
            _cfg.Location.ElevationMeters);

        var forecastErrors = 0;
        foreach (var model in _cfg.Models)
        {
            try
            {
                var rows = await _forecasts.FetchLiveAsync(
                    _cfg.Location, model.Id, _cfg.Variables, _cfg.ForecastDays, ct);
                await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                _log.LogInformation("  {Model}: wrote {Rows} rows", model.Id, rows.Count);
            }
            catch (Exception ex)
            {
                forecastErrors++;
                _log.LogError(ex, "  {Model}: FAILED", model.Id);
            }
        }

        try
        {
            var station = _cfg.Location.Metar.Primary;
            var obs = await _metar.FetchAsync(_cfg.Location.Name, station, hoursBack: 6, ct);

            if (obs.Count == 0 && !string.IsNullOrEmpty(_cfg.Location.Metar.Fallback))
            {
                _log.LogWarning("Primary METAR {Station} returned nothing, trying fallback {Fallback}",
                    station, _cfg.Location.Metar.Fallback);
                station = _cfg.Location.Metar.Fallback;
                obs = await _metar.FetchAsync(_cfg.Location.Name, station, hoursBack: 6, ct);
            }

            await ParquetWriter.WriteObservationsAsync(_cfg.Storage.ObservationsPath, obs, ct);
            _log.LogInformation("  METAR {Station}: wrote {Rows} observations", station, obs.Count);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "METAR collection FAILED");
            return 2;
        }

        // EA rainfall: pull the last 2 days per station. Interval-end timestamps
        // plus overnight latency mean a 1-day window can miss readings still
        // being backfilled by EA; 2 days gives a safe overlap and the dedup
        // writer makes repeated runs cheap.
        var endDate = DateOnly.FromDateTime(DateTime.UtcNow);
        var startDate = endDate.AddDays(-1);
        var rainfallErrors = 0;
        foreach (var st in _cfg.Location.Rainfall.Stations)
        {
            try
            {
                var rain = await _rainfall.FetchAsync(_cfg.Location.Name, st, startDate, endDate, ct);
                await ParquetWriter.WriteRainfallAsync(_cfg.Storage.RainfallPath, rain, ct);
                _log.LogInformation("  EA {Station}: wrote {Rows} readings", st.Name, rain.Count);
            }
            catch (Exception ex)
            {
                rainfallErrors++;
                _log.LogError(ex, "  EA {Station}: FAILED", st.Name);
            }
        }

        if (forecastErrors > 0 || rainfallErrors > 0) return 1;
        return 0;
    }
}
