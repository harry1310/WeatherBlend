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
    private readonly MetOfficeSpotClient _metOfficeSpot;
    private readonly MetOfficeObservationsClient _metOfficeObs;
    private readonly ILogger<CollectCommand> _log;

    public CollectCommand(
        AppConfig cfg,
        OpenMeteoClient forecasts,
        MetarClient metar,
        EaHydrologyClient rainfall,
        MetOfficeSpotClient metOfficeSpot,
        MetOfficeObservationsClient metOfficeObs,
        ILogger<CollectCommand> log)
    {
        _cfg = cfg;
        _forecasts = forecasts;
        _metar = metar;
        _rainfall = rainfall;
        _metOfficeSpot = metOfficeSpot;
        _metOfficeObs = metOfficeObs;
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

        var metOfficeErrors = await CollectMetOfficeAsync(ct);

        if (forecastErrors > 0 || rainfallErrors > 0 || metOfficeErrors > 0) return 1;
        return 0;
    }

    private async Task<int> CollectMetOfficeAsync(CancellationToken ct)
    {
        var mo = _cfg.MetOffice;
        if (!mo.Enabled)
        {
            _log.LogInformation("  Met Office: disabled via config");
            return 0;
        }

        var errors = 0;

        try
        {
            var spotKey = MetOfficeSecrets.TryLoad(mo.SpotKeyEnvVar, mo.SpotKeyFile);
            if (string.IsNullOrWhiteSpace(spotKey))
            {
                _log.LogWarning("  Met Office Spot: no API key ({Env} / {File}); skipping",
                    mo.SpotKeyEnvVar, mo.SpotKeyFile);
            }
            else
            {
                var rows = await _metOfficeSpot.FetchAsync(_cfg.Location, mo.SpotModelTag, spotKey, ct);
                await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                _log.LogInformation("  Met Office Spot: wrote {Rows} rows", rows.Count);
            }
        }
        catch (Exception ex)
        {
            errors++;
            _log.LogError(ex, "  Met Office Spot: FAILED");
        }

        try
        {
            var obsKey = MetOfficeSecrets.TryLoad(mo.ObsKeyEnvVar, mo.ObsKeyFile);
            if (string.IsNullOrWhiteSpace(obsKey))
            {
                _log.LogInformation("  Met Office Obs: no API key ({Env} / {File}); skipping",
                    mo.ObsKeyEnvVar, mo.ObsKeyFile);
            }
            else if (string.IsNullOrWhiteSpace(mo.ObsGeohash))
            {
                _log.LogInformation("  Met Office Obs: geohash not configured (run met-office-bootstrap); skipping");
            }
            else
            {
                var rows = await _metOfficeObs.FetchAsync(
                    _cfg.Location.Name, mo.ObsGeohash, mo.ObsArea ?? "", obsKey, ct);
                await ParquetWriter.WriteMetOfficeObservationsAsync(_cfg.Storage.MetOfficeObsPath, rows, ct);
                _log.LogInformation("  Met Office Obs: wrote {Rows} rows", rows.Count);
            }
        }
        catch (Exception ex)
        {
            errors++;
            _log.LogError(ex, "  Met Office Obs: FAILED");
        }

        return errors;
    }
}
