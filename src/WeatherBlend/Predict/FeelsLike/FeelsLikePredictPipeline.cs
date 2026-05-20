using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Element;

namespace WeatherBlend.Predict.FeelsLike;

/// <summary>
/// Joins per-anchor predictions from the temperature blender (lean 2b) and
/// the four element blenders (humidity / wind / shortwave-radiation /
/// cloud-cover) by (ValidTimeUtc, LeadHours), then computes BOTH UTCI
/// (Bröde 2012 biothermal index) and Steadman 1994 apparent temperature
/// per row via <see cref="FeelsLikeCalculator"/>. One parquet, two
/// "feels like" indices on every row.
///
/// Reads each input from its own
/// <c>data/predictions/{slug}/model_version={v}/date={anchor}/predictions.parquet</c>.
/// When a slug has multiple Active versions, the *first* listed version is used
/// (mirrors the convention "first Active = champion"). Drops any (lead, valid)
/// row that is missing in any one input — UTCI requires all five.
///
/// **All inputs come from the blenders** — temperature, humidity, wind, cloud
/// cover, and shortwave radiation are read from their respective
/// <c>BlendValue</c> columns. Earlier ("v1", before 2026-05-04) the pipeline
/// pulled cloud + radiation from per-model ECMWF raw on the theory that the
/// two would be physically self-consistent within one model; in practice
/// ECMWF was the optimistic outlier on cloud (e.g. 71% vs blend 82%) AND
/// reported clear-sky-like SW (596 W/m² under "71% cloud"), which produced
/// Tmrt ≈ Ta + 25 K under cloudy May days and a UTCI 6–8 K above air
/// temperature. The blender values are noisier per-row but better calibrated
/// against ERA5; the cloud-aware SW cap below handles the residual
/// "NWP-SW-too-bright" case.
///
/// **Shortwave cap**: before computing Tmrt, the SW input is capped via
/// <see cref="FeelsLikeCalculator.CapShortwaveByCloud"/> using the location's
/// solar geometry and the row's blended cloud cover. Acts as a physical
/// upper bound on `clear-sky × Kasten-Czeplak`; passes NWP SW through
/// unchanged when it already sits below that bound.
/// </summary>
public static class FeelsLikePredictPipeline
{
    /// <summary>
    /// Bumped from v1 to v2 on 2026-05-04: pipeline now uses blended values
    /// for every input and applies a cloud-aware SW cap before Tmrt. Numbers
    /// will differ from v1 — write under a fresh model_version so v1 history
    /// stays untouched on R2 and the change is auditable.
    /// </summary>
    public const string OutputModelVersion = "v2";

    public static async Task<List<FeelsLikePredictionRow>> ComposeForAnchorAsync(
        ILogger log,
        string locationName,
        double latitudeDeg,
        double longitudeDeg,
        string predictionsRoot,
        string modelsRoot,
        DateTime anchor,
        DateTime predictionMadeAtUtc,
        CancellationToken ct)
    {
        // Temperature is per-station (per-location) on disk since 2026-05-14.
        // locationName here is the FeelsLike target location, which doubles as
        // the temperature manifest station key.
        var temp = await LoadTemperatureLeanAsync(log, predictionsRoot, modelsRoot, anchor, locationName, ct);
        var hum  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.Humidity,           anchor, locationName, ct);
        var wnd  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.Wind,               anchor, locationName, ct);
        var rad  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.ShortwaveRadiation, anchor, locationName, ct);
        var cld  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.CloudCover,         anchor, locationName, ct);

        if (temp.Rows.Count == 0 || hum.Rows.Count == 0 || wnd.Rows.Count == 0 ||
            rad.Rows.Count == 0 || cld.Rows.Count == 0)
        {
            log.LogError("Feels-like: at least one input is empty for anchor={Anchor:yyyy-MM-dd} (temp={T}, hum={H}, wind={W}, rad={R}, cloud={C}). Run `predict` for those targets first.",
                anchor, temp.Rows.Count, hum.Rows.Count, wnd.Rows.Count, rad.Rows.Count, cld.Rows.Count);
            return new List<FeelsLikePredictionRow>();
        }

        var rows = new List<FeelsLikePredictionRow>();
        foreach (var (key, t) in temp.Rows)
        {
            if (!hum.Rows.TryGetValue(key, out var h)) continue;
            if (!wnd.Rows.TryGetValue(key, out var w)) continue;
            if (!rad.Rows.TryGetValue(key, out var r)) continue;
            if (!cld.Rows.TryGetValue(key, out var c)) continue;

            double ta = t.Value;
            double rh = ClampPercent(h.Value);
            double ws = Math.Max(w.Value, 0.0);
            double swInput = Math.Max(r.Value, 0.0);
            double cc = ClampPercent(c.Value);

            // Cap NWP SW by `clear-sky × Kasten-Czeplak(cloud)`. Passes
            // through unchanged when NWP SW is already below the bound.
            double sw = FeelsLikeCalculator.CapShortwaveByCloud(
                swInput, cc / 100.0, latitudeDeg, longitudeDeg, key.Valid);

            double pHpa = FeelsLikeCalculator.VapourPressureHpa(ta, rh);
            double lDown = FeelsLikeCalculator.LongwaveDownWm2(ta, cc / 100.0, pHpa);
            double lUp = FeelsLikeCalculator.LongwaveUpWm2(ta);
            double tmrt = FeelsLikeCalculator.Tmrt(ta, sw, cc / 100.0, rh);
            double utci = FeelsLikeCalculator.Utci(ta, pHpa, ws, tmrt);
            // Companion BBC/BoM-style shade apparent temperature, persisted alongside
            // UTCI so the home-card chip reads both from the row instead of
            // recomputing one of them at render time.
            double apparent = FeelsLikeCalculator.Steadman1994(ta, pHpa, ws);

            rows.Add(new FeelsLikePredictionRow
            {
                LocationName = locationName,
                ModelVersion = OutputModelVersion,
                PredictionMadeAtUtc = predictionMadeAtUtc,
                ValidTimeUtc = key.Valid,
                LeadHours = key.Lead,
                TemperatureC = ta,
                RelativeHumidityPct = rh,
                WindSpeed10mMs = ws,
                WindSpeed1m1Ms = FeelsLikeCalculator.WindAt1m1(ws),
                ShortwaveDownWm2 = sw,
                CloudCoverPct = cc,
                VapourPressureHpa = pHpa,
                LongwaveDownWm2 = lDown,
                LongwaveUpWm2 = lUp,
                MeanRadiantTemperatureC = tmrt,
                UtciC = utci,
                Band = FeelsLikeCalculator.Band(utci).ToString(),
                ApparentTemperatureC = apparent,
                TempModelVersion = temp.Version,
                HumidityModelVersion = hum.Version,
                WindModelVersion = wnd.Version,
                RadiationModelVersion = rad.Version,
                CloudModelVersion = cld.Version,
            });
        }

        return rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours).ToList();
    }

    private static double ClampPercent(double v) => Math.Clamp(v, 0.0, 100.0);

    private record InputStream(string Version, Dictionary<(int Lead, DateTime Valid), Sample> Rows);
    private record Sample(double Value, DateTime PredictionMadeAt);

    /// <summary>
    /// Per-station Active version pick. Every target manifest is
    /// location-keyed; temperature and the element_* blenders alike resolve
    /// their Active version through the location/station key. Returns an
    /// empty string when there's no Active version for this (target,
    /// station) — the caller treats that as "no input data" (same path as
    /// "predictions parquet not found") so the pipeline returns an empty
    /// rows list and the command returns soft-skip code 3. Phase C commit 3
    /// fix-up (2026-05-20): previously threw an InvalidOperationException
    /// which the predict-and-render workflow propagated as exit 1, failing
    /// the whole job whenever a non-primary location had no element
    /// bundles trained yet (Membury, post-3c rollout).
    /// </summary>
    private static string PickActiveStationVersion(string modelsRoot, string target, string stationKey, ILogger log)
    {
        var active = ModelArtifact.ResolveStationActive(modelsRoot, target, stationKey);
        if (active.Count == 0)
        {
            log.LogWarning("  {Target}/{Station}: no Active version in manifest — feels-like will soft-skip this input.",
                target, stationKey);
            return "";
        }
        if (active.Count > 1)
            log.LogInformation("  {Target}/{Station}: {N} Active versions, using first ({V}).",
                target, stationKey, active.Count, active[0]);
        return active[0];
    }

    private static async Task<InputStream> LoadTemperatureLeanAsync(
        ILogger log, string predictionsRoot, string modelsRoot, DateTime anchor,
        string stationKey, CancellationToken ct)
    {
        var version = PickActiveStationVersion(modelsRoot, "temperature", stationKey, log);
        if (string.IsNullOrEmpty(version))
            return new InputStream(version, new Dictionary<(int, DateTime), Sample>());
        var path = Path.Combine(predictionsRoot, "temperature", stationKey,
            $"model_version={version}", $"date={anchor:yyyy-MM-dd}", "predictions.parquet");
        if (!File.Exists(path))
        {
            log.LogWarning("  temperature: predictions parquet not found at {Path}", path);
            return new InputStream(version, new Dictionary<(int, DateTime), Sample>());
        }
        var raw = (await ParquetSerializer.DeserializeAsync<TempPredictionRow>(path, cancellationToken: ct)).ToList();
        var grouped = raw
            .GroupBy(r => (Lead: r.LeadHours, Valid: r.ValidTimeUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .ToDictionary(
                r => (r.LeadHours, r.ValidTimeUtc),
                r => new Sample(r.BlendTemperature, r.PredictionMadeAtUtc));
        return new InputStream(version, grouped);
    }

    private static async Task<InputStream> LoadElementAsync(
        ILogger log, string predictionsRoot, string modelsRoot, ElementTarget target,
        DateTime anchor, string locationName, CancellationToken ct)
    {
        var version = PickActiveStationVersion(modelsRoot, target.ModelDirName, locationName, log);
        if (string.IsNullOrEmpty(version))
            return new InputStream(version, new Dictionary<(int, DateTime), Sample>());
        var path = Path.Combine(predictionsRoot, target.ModelDirName,
            $"model_version={version}", $"date={anchor:yyyy-MM-dd}", "predictions.parquet");
        if (!File.Exists(path))
        {
            log.LogWarning("  {Slug}: predictions parquet not found at {Path}", target.CliName, path);
            return new InputStream(version, new Dictionary<(int, DateTime), Sample>());
        }
        var raw = (await ParquetSerializer.DeserializeAsync<ElementPredictionRow>(path, cancellationToken: ct)).ToList();
        var grouped = raw
            .GroupBy(r => (Lead: r.LeadHours, Valid: r.ValidTimeUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .ToDictionary(
                r => (r.LeadHours, r.ValidTimeUtc),
                r => new Sample(r.BlendValue, r.PredictionMadeAtUtc));
        return new InputStream(version, grouped);
    }
}
