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
    private readonly MarineClient _marine;
    private readonly BuoyClient _buoys;
    private readonly ILogger<CollectCommand> _log;

    public CollectCommand(
        AppConfig cfg,
        OpenMeteoClient forecasts,
        MetarClient metar,
        EaHydrologyClient rainfall,
        MetOfficeSpotClient metOfficeSpot,
        MetOfficeObservationsClient metOfficeObs,
        MarineClient marine,
        BuoyClient buoys,
        ILogger<CollectCommand> log)
    {
        _cfg = cfg;
        _forecasts = forecasts;
        _metar = metar;
        _rainfall = rainfall;
        _metOfficeSpot = metOfficeSpot;
        _metOfficeObs = metOfficeObs;
        _marine = marine;
        _buoys = buoys;
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
        var marineErrors   = 0;

        // Temporary EA-rainfall bypass. The EA Hydrology readings API
        // (/id/measures/.../readings.json) can hang server-side for minutes
        // with no response (confirmed down 2026-06-09); our per-request
        // timeout + Polly retries then run the whole cycle past the
        // collect workflow's 15-min ceiling, surfacing as a `cancelled`
        // run and starving the predict-4a chain that fires on collect
        // success. EA rainfall is precip/dry-window TRUTH (training +
        // verify) — it is NOT a predict input — so skipping it leaves
        // predict fully intact. Set WEATHERBLEND_SKIP_RAINFALL=1 to skip;
        // unset before the Sunday retrain so training sees fresh truth.
        var skipRainfall = (Environment.GetEnvironmentVariable("WEATHERBLEND_SKIP_RAINFALL") ?? "")
            .Trim() is "1" or "true" or "TRUE";
        if (skipRainfall)
            _log.LogWarning("WEATHERBLEND_SKIP_RAINFALL set — SKIPPING all EA rainfall collection "
                + "this cycle (truth-only source; predict unaffected). Restore before the next retrain.");

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
            // rainfall.stations (e.g. a future temp-only secondary), or
            // wholesale when WEATHERBLEND_SKIP_RAINFALL is set (see above).
            // The skip does NOT touch rainfallErrors, so the cycle still
            // exits 0 and the predict chain proceeds.
            foreach (var st in skipRainfall
                         ? Enumerable.Empty<RainfallStationConfig>()
                         : location.Rainfall.Stations)
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
            // Marine (sea-state) — locations with a marine: block only
            // (Sennen). NON-CRITICAL like Met Office below: nothing
            // downstream consumes waves yet (Phase 0 = accumulate only), so
            // a marine-API hiccup must not fail the cycle or starve the
            // predict chain. Logged loudly; promote to fatal when the wave
            // blender ships. The offset-day pull is the one that matters
            // long-term — per-lead rows can NEVER be backfilled (the marine
            // hindcast archive has no previous_day columns), so every missed
            // cycle is per-lead training data lost for good.
            if (location.Marine is not null)
                marineErrors += await CollectMarineAsync(location, ct);
        }

        var metOfficeErrors = await CollectMetOfficeAsync(ct);

        if (marineErrors > 0)
            _log.LogWarning(
                "  Marine: {N} pull(s) FAILED — non-critical, NOT failing the cycle (see ERR lines above)",
                marineErrors);

        // Met Office Spot + Land Obs are NON-CRITICAL and deliberately excluded
        // from the exit code. Spot is a skill-page comparison line (not a blender
        // input) and Obs is a supplemental truth signal — neither feeds predict or
        // retrain. DataHub is also markedly less reliable than Open-Meteo / EA, so
        // a transient outage there (2026-06-02: every DataHub call timed out while
        // all 9 NWP models + EA rainfall + METAR succeeded and pushed to R2) must
        // not fail the whole cycle red. We log loudly so the gap is still visible,
        // but the cycle stands. OM / METAR failures stay fatal.
        if (metOfficeErrors > 0)
            _log.LogWarning(
                "  Met Office: {N} sub-collector(s) FAILED — non-critical, NOT failing the cycle (see ERR lines above)",
                metOfficeErrors);

        // EA rainfall is NON-FATAL in collect (2026-06-16). The flaky EA Hydrology
        // endpoint (frequent 403 / multi-minute hangs) was failing the 3-hourly
        // collect RED, which starved the predict-4a chain that fires only on collect
        // SUCCESS. Rainfall is best-effort here — collect keeps the antecedent-rain
        // persistence feature (3c/3o/4a) fresh when EA is up, and degrades to
        // last-good when it isn't. The DAILY truth-refresh pulls EA as a FATAL step,
        // so a persistent EA outage still surfaces as a red run there (the right
        // cadence to alert on) without flapping the predict chain every few hours.
        if (rainfallErrors > 0)
            _log.LogWarning(
                "  EA rainfall: {N} gauge(s) FAILED — non-fatal here, NOT failing the cycle; "
                + "the daily truth-refresh is the EA canary (see ERR lines above).",
                rainfallErrors);

        if (forecastErrors > 0 || metarErrors > 0) return 1;
        return 0;
    }

    /// <summary>
    /// One marine cycle for one location: per wave model a live pull
    /// (current forecast, synthesised run time) + an offset-day pull
    /// (previous_day1..7 over past_days=2 — the per-lead archive that only
    /// exists if we capture it at collect time). best_match is pulled
    /// implicitly on top of the configured models; its live pull carries
    /// the site extras (tide / SST / secondary swell) and its offset pull
    /// sticks to the standard wave variables like everyone else's.
    /// </summary>
    private async Task<int> CollectMarineAsync(LocationConfig location, CancellationToken ct)
    {
        var errors = 0;
        var marine = location.Marine!;
        var waveVars = _cfg.Variables.Marine;
        var modelIds = marine.Models.Select(m => m.Id)
            .Append(MarineClient.BestMatchModel)
            .ToList();

        foreach (var modelId in modelIds)
        {
            ct.ThrowIfCancellationRequested();
            var liveVars = modelId == MarineClient.BestMatchModel
                ? waveVars.Concat(_cfg.Variables.MarineSite).ToList()
                : waveVars;
            try
            {
                var rows = await _marine.FetchLiveAsync(location, modelId, liveVars, _cfg.ForecastDays, ct);
                await ParquetWriter.WriteMarineForecastsAsync(_cfg.Storage.MarinePath, rows, ct);
                _log.LogInformation("  {Loc}/marine {Model} live: wrote {Rows} rows", location.Name, modelId, rows.Count);
            }
            catch (Exception ex)
            {
                errors++;
                _log.LogError(ex, "  {Loc}/marine {Model} live: FAILED", location.Name, modelId);
            }

            try
            {
                var rows = await _marine.FetchOffsetDaysAsync(location, modelId, waveVars, pastDays: 2, ct);
                await ParquetWriter.WriteMarinePreviousRunsAsync(_cfg.Storage.MarinePath, rows, ct);
                _log.LogInformation("  {Loc}/marine {Model} offset-days: wrote {Rows} rows", location.Name, modelId, rows.Count);
            }
            catch (Exception ex)
            {
                errors++;
                _log.LogError(ex, "  {Loc}/marine {Model} offset-days: FAILED", location.Name, modelId);
            }
        }

        // Buoy realtime (WaveNet, last ~24h per station, non-QC'd). The
        // writer merges + dedups on ValidTimeUtc, and collect.yml pre-pulls
        // the recent waves window from R2, so overlapping cycles and later
        // QC'd archive re-pulls are both safe.
        foreach (var buoy in marine.Buoys)
        {
            try
            {
                var rows = await _buoys.FetchRealtimeAsync(location.Name, buoy, ct);
                await ParquetWriter.WriteWaveTruthAsync(_cfg.Storage.WavesPath, rows, ct);
                _log.LogInformation("  {Loc}/buoy {Slug}: wrote {Rows} rows", location.Name, buoy.Slug, rows.Count);
            }
            catch (Exception ex)
            {
                errors++;
                _log.LogError(ex, "  {Loc}/buoy {Slug}: FAILED", location.Name, buoy.Slug);
            }
        }
        return errors;
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

        // Supplemental obs geohashes — physically distinct stations
        // collected as additional truth signals for bake-offs / cross-validation.
        // Each pulled in its own try/catch so a single bad geohash doesn't kill
        // the rest of the cycle, mirroring the primary obs error policy above.
        // Skipped silently when the API key is absent.
        if (mo.SupplementalObsGeohashes is { Count: > 0 })
        {
            var obsKey = MetOfficeSecrets.TryLoad(mo.ObsKeyEnvVar, mo.ObsKeyFile);
            if (string.IsNullOrWhiteSpace(obsKey))
            {
                _log.LogInformation("  Met Office Obs (supplemental): no API key; skipping {N} entries",
                    mo.SupplementalObsGeohashes.Count);
            }
            else
            {
                foreach (var sup in mo.SupplementalObsGeohashes)
                {
                    if (string.IsNullOrWhiteSpace(sup.Geohash) || string.IsNullOrWhiteSpace(sup.Label))
                    {
                        _log.LogWarning("  Met Office Obs (supplemental): entry missing geohash or label; skipping");
                        continue;
                    }
                    try
                    {
                        // LocationName on stored rows becomes the friendly label
                        // (e.g. "dunkeswell_aerodrome") so SQL queries can scope
                        // by source without having to know geohashes by heart.
                        var rows = await _metOfficeObs.FetchAsync(
                            sup.Label, sup.Geohash, sup.Area, obsKey, ct);
                        await ParquetWriter.WriteMetOfficeObservationsAsync(_cfg.Storage.MetOfficeObsPath, rows, ct);
                        _log.LogInformation("  Met Office Obs ({Label} @ {Gh}): wrote {Rows} rows",
                            sup.Label, sup.Geohash, rows.Count);
                    }
                    catch (Exception ex)
                    {
                        errors++;
                        _log.LogError(ex, "  Met Office Obs ({Label} @ {Gh}): FAILED",
                            sup.Label, sup.Geohash);
                    }
                }
            }
        }

        // Met Office Global Det + UKV 2km AWS-archive collectors removed
        // 2026-04-29 — bake-off rejected as blender inputs and the Python
        // writer was poisoning the forecast-tree schema (see Program.cs
        // DI-block comment for the full why).

        return errors;
    }
}
