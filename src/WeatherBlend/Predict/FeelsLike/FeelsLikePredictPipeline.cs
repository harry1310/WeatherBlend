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
    /// Per-station version pick for a feels-like input. Prefers the
    /// phases.yaml champion (via <see cref="ModelArtifact.ResolveStationChampionVersion"/>),
    /// falls back to the first Active entry when the champion resolver
    /// returns empty. Returns "" only when there's no Active version at all
    /// (treated as "no input data", soft-skip exit 3).
    ///
    /// History: the original implementation returned <c>active[0]</c>
    /// outright ("first Active = champion"). That convention silently broke
    /// for Bonehill temperature once the 2026-05-17 auto-retrain promoted
    /// three new bundles without displacing the older 2d entry sitting at
    /// index 0 — feels-like read the stale 2d's empty / missing date
    /// partition and the home-page UTCI chart stopped advancing. Today's
    /// straight champion-resolver swap (commit d6c5b12) fixed temperature
    /// but exposed a SECOND latent mismatch: the element trainers stamp
    /// <c>training_metadata.Phase</c> with a tag like <c>"lean-humidity"</c>
    /// while phases.yaml declares each element's champion phase id as plain
    /// <c>"humidity"</c>/<c>"wind"</c>/<c>"shortwave_radiation"</c>/<c>"cloud_cover"</c>.
    /// Strict-champion lookup matches on Phase and returns empty for every
    /// element, soft-skipping them and dropping feels-like to 0 rows.
    ///
    /// The fallback restores element behaviour (one Active version per
    /// element station — <c>active[0]</c> is the right pick) while keeping
    /// the temperature fix (where multiple Active versions live, champion
    /// is correctly resolved and the fallback never fires). The proper
    /// long-term cleanup is a phases.yaml / element-trainer pass to align
    /// the phase tags; the fallback log makes the mismatch visible until
    /// then.
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

        var champion = ModelArtifact.ResolveStationChampionVersion(modelsRoot, target, stationKey);
        if (!string.IsNullOrEmpty(champion))
        {
            if (active.Count > 1)
                log.LogInformation("  {Target}/{Station}: {N} Active versions, using champion ({V}).",
                    target, stationKey, active.Count, champion);
            return champion;
        }

        // Champion resolver returned empty — typically a phases.yaml champion-
        // phase id vs trainer metadata.Phase mismatch (the element trainers
        // stamp "lean-{element}" while phases.yaml declares "{element}").
        // Fall back to the first Active entry; for elements that IS the only
        // version anyway, so the fallback is behaviour-equivalent to the
        // pre-d6c5b12 heuristic. Warn so the mismatch is visible.
        log.LogWarning(
            "  {Target}/{Station}: champion resolver returned no version — phases.yaml champion-phase / metadata.Phase tag mismatch?. Falling back to first Active ({V}).",
            target, stationKey, active[0]);
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
