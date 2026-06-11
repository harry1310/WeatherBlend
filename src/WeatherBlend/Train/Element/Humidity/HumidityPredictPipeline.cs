using Microsoft.Extensions.Logging;
using WeatherBlend.Models;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Humidity;

/// <summary>
/// Predict-time pipeline for the humidity blender. Spec-driven via BlenderSpec.
/// Pulls the canonical 6-slot pivot once for RH + DP (+ the aux-mean source
/// columns); per lead filters down to spec.Models in spec order.
///
/// Orchestration lives in <see cref="ElementPredictHarness"/>; this file keeps
/// the humidity specifics — most importantly the schema-gated aux-means block
/// (2026-06-10, humidity-aifs-bakeoff) that serves BOTH the legacy 17-feature
/// bundles and the new 23-feature ones.
/// </summary>
public static class HumidityPredictPipeline
{
    // Pivot channel indices (must match PivotColumns order below).
    private const int ChRh = 0, ChDp = 1, ChTemp = 2,
        ChCloudLow = 3, ChCloudMid = 4, ChCloudHigh = 5, ChWind = 6;

    private static readonly ElementPredictHarness.Descriptor D = new(
        Element: "humidity",
        PivotColumns: new[]
        {
            "RelativeHumidity2m", "DewPoint2m", "Temperature2m",
            "CloudCoverLow", "CloudCoverMid", "CloudCoverHigh", "WindSpeed10m",
        },
        // A required model must supply BOTH rh and dp — each feeds an
        // N-block of the feature vector.
        RequiredChannels: new[] { ChRh, ChDp },
        MissingNoun: "rh/dp",
        ComposeRow: ctx =>
        {
            // Aux ensemble means (schema-gated: only post-2026-06-10 bundles
            // carry temp_mean in their saved spec). NaN-safe means over THIS
            // spec's model set — mirrors the trainer's SQL AVG over the
            // spec-filtered `latest` rows. Computed over the raw FLOAT pivot
            // channels (not the widened doubles) so the accumulation is
            // bit-identical to the pre-harness inline block.
            double[]? aux = null;
            if (ctx.Spec.FeatureNames.Contains("temp_mean"))
            {
                int N = ctx.Spec.Models.Count;
                var ch = ctx.Pivot.Channels;
                double MeanOf(Func<int, float> get)
                {
                    double s = 0; int k = 0;
                    for (int i = 0; i < N; i++)
                    {
                        var x = get(ctx.CanonIdx[i]);
                        if (float.IsNaN(x)) continue;
                        s += x; k++;
                    }
                    return k == 0 ? double.NaN : s / k;
                }
                aux = new[]
                {
                    MeanOf(ci => ch[ChTemp][ci]),
                    MeanOf(ci => float.IsNaN(ch[ChTemp][ci]) || float.IsNaN(ch[ChDp][ci]) ? float.NaN : ch[ChTemp][ci] - ch[ChDp][ci]),
                    MeanOf(ci => ch[ChCloudLow][ci]),
                    MeanOf(ci => ch[ChCloudMid][ci]),
                    MeanOf(ci => ch[ChCloudHigh][ci]),
                    MeanOf(ci => ch[ChWind][ci]),
                };
            }
            return HumidityFeatureBuilder.ComposeRow(
                ctx.Spec, ctx.Valid, ctx.Vals[ChRh], ctx.Vals[ChDp], era5Rh: float.NaN, aux);
        },
        // Spread features at offset 2N (rh + dp blocks first).
        SummaryStats: (ctx, row) => ElementPredictHarness.SpreadFromFeatures(row, 2 * ctx.Spec.Models.Count),
        LogResult: (log, lead, yhat, valid, mean) =>
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0}% (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0}%)",
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
