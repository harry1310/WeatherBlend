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
        // Iterate every configured location for forecasts + METAR + EA
        // rainfall. Each is gated on its own config so a location with
        // no METAR config (e.g. Membury, precip-only) silently skips
        // the METAR pull rather than logging a spurious "FAILED". This
        // closes the live-collect gap where Membury's 6h forecast tree
        // + EA rainfall stopped at the last previous-runs-refresh
        // window — pre-2026-05-11 collect ran for `_cfg.Location`
        // (singular = primary = Bonehill) only.
        //
        // Met Office Spot now loops over _cfg.Locations (Spot is lat/lon-
        // driven so it works for any configured location — used as a
        // skill-page comparison line only, not a blender input). Met
        // Office Obs stays primary-only because the geohash is pinned at
        // the AppConfig.MetOffice level — extending it would need a
        // per-location obsGeohash field.
        var forecastErrors = 0;
        var metarErrors    = 0;
        var rainfallErrors = 0;

        // EA rainfall window: last 2 days per station. Interval-end
        // timestamps + overnight latency mean a 1-day window can miss
        // readings still being backfilled by EA; 2 days gives safe
        // overlap and the dedup writer makes repeated runs cheap.
        var endDate   = DateOnly.FromDateTime(DateTime.UtcNow);
        var startDate = endDate.AddDays(-1);

        foreach (var location in _cfg.Locations)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("=== Collecting for {Location} ({Lat:F4}, {Lon:F4}, {Elev}m) ===",
                location.DisplayName, location.Latitude, location.Longitude, location.ElevationMeters);

            foreach (var model in _cfg.Models)
            {
                try
                {
                    var rows = await _forecasts.FetchLiveAsync(
                        location, model.Id, _cfg.Variables.Forecast, _cfg.ForecastDays, ct);
                    await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                    _log.LogInformation("  {Loc}/{Model}: wrote {Rows} rows", location.Name, model.Id, rows.Count);
                }
                catch (Exception ex)
                {
                    forecastErrors++;
                    _log.LogError(ex, "  {Loc}/{Model}: FAILED", location.Name, model.Id);
                }
            }

            // METAR — gated on a configured Primary station. Locations
            // without one (e.g. Membury, precip-only) are silently skipped.
            if (!string.IsNullOrWhiteSpace(location.Metar.Primary))
            {
                try
                {
                    var station = location.Metar.Primary;
                    var obs = await _metar.FetchAsync(location.Name, station, hoursBack: 6, ct);

                    if (obs.Count == 0 && !string.IsNullOrEmpty(location.Metar.Fallback))
                    {
                        _log.LogWarning("  Primary METAR {Station} returned nothing, trying fallback {Fallback}",
                            station, location.Metar.Fallback);
                        station = location.Metar.Fallback;
                        obs = await _metar.FetchAsync(location.Name, station, hoursBack: 6, ct);
                    }

                    await ParquetWriter.WriteObservationsAsync(_cfg.Storage.ObservationsPath, obs, ct);
                    _log.LogInformation("  {Loc}/METAR {Station}: wrote {Rows} observations",
                        location.Name, station, obs.Count);
                }
                catch (Exception ex)
                {
                    // Log + count, do not abort. Pre-2026-04-26 a METAR
                    // exit 2 skipped rainfall + MO + the post-collect R2
                    // push, breaking everything else when aviationweather
                    // .gov hiccupped. METAR is one truth source among
                    // many now.
                    metarErrors++;
                    _log.LogError(ex, "  {Loc}/METAR collection FAILED (non-fatal)", location.Name);
                }
            }

            // EA rainfall — silently skipped for locations with no
            // rainfall.stations (e.g. a future temp-only secondary).
            foreach (var st in location.Rainfall.Stations)
            {
                try
                {
                    var rain = await _rainfall.FetchAsync(location.Name, st, startDate, endDate, ct);
                    await ParquetWriter.WriteRainfallAsync(_cfg.Storage.RainfallPath, rain, ct);
                    _log.LogInformation("  {Loc}/EA {Station}: wrote {Rows} readings",
                        location.Name, st.Name, rain.Count);
                }
                catch (Exception ex)
                {
                    rainfallErrors++;
                    _log.LogError(ex, "  {Loc}/EA {Station}: FAILED", location.Name, st.Name);
                }
            }
        }

        var metOfficeErrors = await CollectMetOfficeAsync(ct);

        if (forecastErrors > 0 || metarErrors > 0 || rainfallErrors > 0 || metOfficeErrors > 0) return 1;
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

        // Spot: lat/lon-driven, one shot per configured location. Skill-pages
        // only (not a blender input) so this just lands in the forecasts tree
        // under model=met_office_spot, location=<slug> for the render side to
        // pick up. Obs stays primary-only — see below.
        var spotKey = MetOfficeSecrets.TryLoad(mo.SpotKeyEnvVar, mo.SpotKeyFile);
        if (string.IsNullOrWhiteSpace(spotKey))
        {
            _log.LogWarning("  Met Office Spot: no API key ({Env} / {File}); skipping all locations",
                mo.SpotKeyEnvVar, mo.SpotKeyFile);
        }
        else
        {
            foreach (var location in _cfg.Locations)
            {
                try
                {
                    var rows = await _metOfficeSpot.FetchAsync(location, mo.SpotModelTag, spotKey, ct);
                    await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                    _log.LogInformation("  Met Office Spot ({Loc}): wrote {Rows} rows", location.Name, rows.Count);
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  Met Office Spot ({Loc}): FAILED", location.Name);
                }
            }
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

        // Met Office Global Det + UKV 2km AWS-archive collectors removed
        // 2026-04-29 — bake-off rejected as blender inputs and the Python
        // writer was poisoning the forecast-tree schema (see Program.cs
        // DI-block comment for the full why).

        return errors;
    }
}
