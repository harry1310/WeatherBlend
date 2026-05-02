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
        var temp = await LoadTemperatureLeanAsync(log, predictionsRoot, modelsRoot, anchor, ct);
        var hum  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.Humidity,           anchor, ct);
        var wnd  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.Wind,               anchor, ct);
        var rad  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.ShortwaveRadiation, anchor, ct);
        var cld  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.CloudCover,         anchor, ct);

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

    private static string PickActiveVersion(string modelsRoot, string slug, ILogger log)
    {
        var active = ModelArtifact.ResolveActive(modelsRoot, slug);
        if (active.Count == 0)
            throw new InvalidOperationException($"No Active version for {slug}; train it first.");
        if (active.Count > 1)
            log.LogInformation("  {Slug}: {N} Active versions, using first ({V}).", slug, active.Count, active[0]);
        return active[0];
    }

    private static async Task<InputStream> LoadTemperatureLeanAsync(
        ILogger log, string predictionsRoot, string modelsRoot, DateTime anchor, CancellationToken ct)
    {
        var version = PickActiveVersion(modelsRoot, "temperature", log);
        var path = Path.Combine(predictionsRoot, "temperature",
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
        DateTime anchor, CancellationToken ct)
    {
        var version = PickActiveVersion(modelsRoot, target.ModelDirName, log);
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
