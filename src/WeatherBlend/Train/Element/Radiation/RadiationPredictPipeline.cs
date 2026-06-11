using Microsoft.Extensions.Logging;
using WeatherBlend.Models;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Radiation;

/// <summary>
/// Predict-time pipeline for the shortwave-radiation blender. Spec-driven via
/// BlenderSpec. Pulls the canonical 6-slot pivot once for SW + direct +
/// diffuse (+ the rich-mean source columns); per lead filters down to
/// spec.Models in spec order.
///
/// Orchestration lives in <see cref="ElementPredictHarness"/>; this file keeps
/// the radiation specifics (the schema-gated cloud_mean/rh_mean rich features).
/// </summary>
public static class RadiationPredictPipeline
{
    private const int ChSw = 0, ChDr = 1, ChDf = 2, ChCc = 3, ChRh = 4;

    private static readonly ElementPredictHarness.Descriptor D = new(
        Element: "shortwave-radiation",
        PivotColumns: new[]
        {
            "ShortwaveRadiation", "DirectRadiation", "DiffuseRadiation",
            "CloudCover", "RelativeHumidity2m",
        },
        RequiredChannels: new[] { ChSw },
        MissingNoun: "SW",
        ComposeRow: ctx =>
        {
            // Rich bundles carry cloud_mean/rh_mean (FeatureCount 3N+9); compute the
            // NaN-safe ensemble means over the spec's models, matching BuildForLead's
            // AVG. Harmless for lean bundles — ComposeRow only writes them when the
            // loaded spec includes them. Accumulates over the raw FLOAT pivot
            // channels (bit-identical to the pre-harness inline block).
            int N = ctx.Spec.Models.Count;
            double cSum = 0, rSum = 0; int cCnt = 0, rCnt = 0;
            for (int i = 0; i < N; i++)
            {
                var ci = ctx.CanonIdx[i];
                var cv = ctx.Pivot.Channels[ChCc][ci]; if (!float.IsNaN(cv)) { cSum += cv; cCnt++; }
                var rv = ctx.Pivot.Channels[ChRh][ci]; if (!float.IsNaN(rv)) { rSum += rv; rCnt++; }
            }
            var cloudMean = cCnt > 0 ? cSum / cCnt : double.NaN;
            var rhMean    = rCnt > 0 ? rSum / rCnt : double.NaN;

            return RadiationFeatureBuilder.ComposeRow(
                ctx.Spec, ctx.Valid, ctx.Vals[ChSw], ctx.Vals[ChDr], ctx.Vals[ChDf],
                era5Sw: float.NaN, cloudMean: cloudMean, rhMean: rhMean);
        },
        // Spread features at offset 3N (sw + direct + diffuse blocks first).
        SummaryStats: (ctx, row) => ElementPredictHarness.SpreadFromFeatures(row, 3 * ctx.Spec.Models.Count),
        LogResult: (log, lead, yhat, valid, mean) =>
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0} W/m² (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0} W/m²)",
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
