using Microsoft.Extensions.Logging;
using WeatherBlend.Models;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// Predict-time pipeline for the wind blender. Spec-driven: per-lead
/// <see cref="WeatherBlend.Train.Common.BlenderSpec"/> loaded from
/// <c>feature_schema.json</c>; the SQL pivot pulls only spec.Models columns;
/// the feature row is composed via <see cref="WindFeatureBuilder.ComposeRow"/>
/// so train + predict shapes are byte-for-byte identical.
///
/// Output <see cref="ElementPredictionRow"/> still has six per-model slots
/// (Gfs/Ecmwf/Icon/Mf/Ukmo/Gem). We populate only the slots that map back to a
/// model in <c>spec.Models</c>; the rest stay <c>null</c>. The fossil 5-to-6
/// <c>OutputModelFields</c> mapping helper is gone — the spec is the single
/// source of truth for which output fields get populated.
///
/// Orchestration lives in <see cref="ElementPredictHarness"/> (shared with the
/// other four element pipelines); this file only declares the wind specifics.
/// NB: the .NET wind blender emits no band columns — the wind speed
/// prediction-interval band (BandLoMs/BandHiMs) is owned by the Python
/// wind_speed_lgb predict path (predict_wind_speed_pi.py), not this pipeline.
/// </summary>
public static class WindPredictPipeline
{
    private static readonly ElementPredictHarness.Descriptor D = new(
        Element: "wind",
        // Channel 0 = wind speed (primary/output variable), channel 1 = direction.
        PivotColumns: new[] { "WindSpeed10m", "WindDirection10m" },
        // Required models must supply speed; direction NaN propagates into the
        // sin/cos feature slots, which LightGBM tolerates natively.
        RequiredChannels: new[] { 0 },
        MissingNoun: "wind",
        ComposeRow: ctx => WindFeatureBuilder.ComposeRow(
            ctx.Spec, ctx.Valid, ctx.Vals[0], ctx.Vals[1], era5WindSpeed: float.NaN),
        // Spread features at offset 3*N (after spd + sin block + cos block).
        SummaryStats: (ctx, row) => ElementPredictHarness.SpreadFromFeatures(row, 3 * ctx.Spec.Models.Count),
        LogResult: (log, lead, yhat, valid, mean) =>
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.000} m/s (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.000} m/s)",
                lead, yhat, valid, mean));

    public static List<ElementPredictionRow> PredictForCycle(
        ILogger log,
        string locationName,
        string forecastsPath,
        string versionDir,
        string modelVersion,
        DateTime anchor,
        DateTime predictionMadeAt,
        int[] leads,
        CancellationToken ct)
        => ElementPredictHarness.PredictForCycle(
            log, D, locationName, forecastsPath, versionDir, modelVersion,
            anchor, predictionMadeAt, leads, ct);
}
