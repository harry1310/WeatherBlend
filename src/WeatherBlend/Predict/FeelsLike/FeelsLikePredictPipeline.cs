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
/// **Cloud sourcing**: <see cref="UseEcmwfRawForCloud"/> defaults to true.
/// The cloud blender currently loses to best-single ECMWF at 24h (~−10.8% MAE)
/// and is roughly tied at longer leads. While that's true we read the
/// per-model <c>ModelEcmwf</c> field from the cloud blender's prediction
/// parquet (it's persisted alongside <c>BlendValue</c> for exactly this kind
/// of fallback). Flip the flag back once the cloud blender beats ECMWF.
/// Provenance row carries the suffix <c>+ecmwf_raw</c> so the source is
/// auditable per-row.
/// </summary>
public static class FeelsLikePredictPipeline
{
    public const string OutputModelVersion = "v1";

    /// <summary>Pull the ECMWF per-model value from the cloud prediction parquet
    /// instead of the blended value. Temporary workaround until the cloud blender
    /// beats best-single ECMWF — see class docstring.</summary>
    public static bool UseEcmwfRawForCloud { get; set; } = true;

    public static async Task<List<FeelsLikePredictionRow>> ComposeForAnchorAsync(
        ILogger log,
        string locationName,
        string predictionsRoot,
        string modelsRoot,
        DateTime anchor,
        DateTime predictionMadeAtUtc,
        CancellationToken ct)
    {
        var temp = await LoadTemperatureLeanAsync(log, predictionsRoot, modelsRoot, anchor, ct);
        var hum  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.Humidity,         anchor, ct);
        var wnd  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.Wind,             anchor, ct);
        var rad  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.ShortwaveRadiation, anchor, ct);
        var cld  = await LoadElementAsync(log, predictionsRoot, modelsRoot, ElementTargets.CloudCover,       anchor, ct,
            useEcmwfRaw: UseEcmwfRawForCloud);

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
            double sw = Math.Max(r.Value, 0.0);
            double cc = ClampPercent(c.Value);

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
        var raw = (await ParquetSerializer.DeserializeAsync<PredictionRow>(path, cancellationToken: ct)).ToList();
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
        DateTime anchor, CancellationToken ct, bool useEcmwfRaw = false)
    {
        var version = PickActiveVersion(modelsRoot, target.ModelDirName, log);
        if (useEcmwfRaw) version += "+ecmwf_raw";
        var path = Path.Combine(predictionsRoot, target.ModelDirName,
            $"model_version={(useEcmwfRaw ? version[..^"+ecmwf_raw".Length] : version)}",
            $"date={anchor:yyyy-MM-dd}", "predictions.parquet");
        if (!File.Exists(path))
        {
            log.LogWarning("  {Slug}: predictions parquet not found at {Path}", target.CliName, path);
            return new InputStream(version, new Dictionary<(int, DateTime), Sample>());
        }
        var raw = (await ParquetSerializer.DeserializeAsync<ElementPredictionRow>(path, cancellationToken: ct)).ToList();
        if (useEcmwfRaw)
        {
            log.LogInformation("  {Slug}: using ModelEcmwf per-row instead of BlendValue (UseEcmwfRawForCloud=true)", target.CliName);
        }
        var grouped = raw
            .Select(r => (Row: r, Sample: useEcmwfRaw ? r.ModelEcmwf : (double?)r.BlendValue))
            .Where(x => x.Sample.HasValue)
            .GroupBy(x => (Lead: x.Row.LeadHours, Valid: x.Row.ValidTimeUtc))
            .Select(g => g.MaxBy(x => x.Row.PredictionMadeAtUtc))
            .ToDictionary(
                x => (x.Row.LeadHours, x.Row.ValidTimeUtc),
                x => new Sample(x.Sample!.Value, x.Row.PredictionMadeAtUtc));
        return new InputStream(version, grouped);
    }
}
