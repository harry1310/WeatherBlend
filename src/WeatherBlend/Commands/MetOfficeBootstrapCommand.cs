using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// One-off pre-populate of Met Office data at the start of the capture window.
/// Fetches the current Spot forecast (first forward-only data point) and the
/// rolling 48h Land Observations window. Intended to be run once, on the day
/// capture begins — the normal <see cref="CollectCommand"/> handles every cycle
/// after that.
/// </summary>
public sealed class MetOfficeBootstrapCommand
{
    private readonly AppConfig _cfg;
    private readonly MetOfficeSpotClient _spot;
    private readonly MetOfficeObservationsClient _obs;
    private readonly ILogger<MetOfficeBootstrapCommand> _log;

    public MetOfficeBootstrapCommand(
        AppConfig cfg,
        MetOfficeSpotClient spot,
        MetOfficeObservationsClient obs,
        ILogger<MetOfficeBootstrapCommand> log)
    {
        _cfg = cfg;
        _spot = spot;
        _obs = obs;
        _log = log;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var mo = _cfg.MetOffice;
        if (!mo.Enabled)
        {
            _log.LogWarning("Met Office is disabled in config; nothing to bootstrap");
            return 0;
        }

        _log.LogInformation("Bootstrapping Met Office capture for {Location}", _cfg.Location.DisplayName);

        var errors = 0;

        var spotKey = MetOfficeSecrets.TryLoad(mo.SpotKeyEnvVar, mo.SpotKeyFile);
        if (string.IsNullOrWhiteSpace(spotKey))
        {
            _log.LogWarning("Spot API key missing ({Env} / {File}); skipping Spot bootstrap",
                mo.SpotKeyEnvVar, mo.SpotKeyFile);
        }
        else
        {
            try
            {
                var rows = await _spot.FetchAsync(_cfg.Location, mo.SpotModelTag, spotKey, ct);
                await ParquetWriter.WriteForecastsAsync(_cfg.Storage.ForecastsPath, rows, ct);
                _log.LogInformation("Spot: wrote {Rows} rows (first forward-only capture)", rows.Count);
            }
            catch (Exception ex)
            {
                errors++;
                _log.LogError(ex, "Spot bootstrap FAILED");
            }
        }

        var obsKey = MetOfficeSecrets.TryLoad(mo.ObsKeyEnvVar, mo.ObsKeyFile);
        if (string.IsNullOrWhiteSpace(obsKey))
        {
            _log.LogWarning("Obs API key missing ({Env} / {File}); skipping Obs bootstrap",
                mo.ObsKeyEnvVar, mo.ObsKeyFile);
            return errors > 0 ? 1 : 0;
        }

        // Resolve geohash: either reuse the pinned value from config, or call
        // /nearest once to discover it. The API spec asks callers to cache this.
        var geohash = mo.ObsGeohash;
        var area = mo.ObsArea ?? "";
        if (string.IsNullOrWhiteSpace(geohash))
        {
            try
            {
                var nearest = await _obs.FindNearestAsync(
                    _cfg.Location.Latitude, _cfg.Location.Longitude, obsKey, ct);
                if (nearest is null)
                {
                    _log.LogWarning("Obs bootstrap: /nearest returned no location; skipping obs fetch");
                    return errors > 0 ? 1 : 0;
                }
                geohash = nearest.Geohash;
                area = nearest.Area;
                _log.LogInformation(
                    "Obs: /nearest resolved geohash={Geohash} area={Area} — pin these in config.yaml "
                    + "(metOffice.obsGeohash / metOffice.obsArea) to avoid re-querying on every bootstrap.",
                    geohash, area);
            }
            catch (Exception ex)
            {
                errors++;
                _log.LogError(ex, "Obs /nearest lookup FAILED");
                return 1;
            }
        }

        try
        {
            var rows = await _obs.FetchAsync(_cfg.Location.Name, geohash, area, obsKey, ct);
            await ParquetWriter.WriteMetOfficeObservationsAsync(_cfg.Storage.MetOfficeObsPath, rows, ct);
            _log.LogInformation("Obs: wrote {Rows} rows (last 48h rolling window)", rows.Count);
        }
        catch (Exception ex)
        {
            errors++;
            _log.LogError(ex, "Obs bootstrap FAILED");
        }

        return errors > 0 ? 1 : 0;
    }
}
