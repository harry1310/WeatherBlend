using Microsoft.Extensions.Logging;
using WeatherBlend.Models;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Cloud;

/// <summary>
/// Predict-time pipeline for the cloud-cover blender. Spec-driven via
/// BlenderSpec loaded from feature_schema.json. Pulls the 6-slot canonical
/// pivot once (cloud cover + CAPE), then per lead filters down to spec.Models
/// in spec order.
///
/// Orchestration lives in <see cref="ElementPredictHarness"/>; this file keeps
/// the cloud specifics (the lead-gated cape_mean aux feature).
/// </summary>
public static class CloudPredictPipeline
{
    private const int ChCc = 0, ChCape = 1;

    private static readonly ElementPredictHarness.Descriptor D = new(
        Element: "cloud-cover",
        PivotColumns: new[] { "CloudCover", "Cape" },
        RequiredChannels: new[] { ChCc },
        MissingNoun: "cloud cover",
        ComposeRow: ctx =>
        {
            // 24h-lead bundles carry cape_mean (FeatureCount n+8); compute the NaN-safe
            // ensemble-mean CAPE over the spec's models, matching BuildForLead's AVG.
            // Harmless for lean 48/72h bundles — ComposeRow only writes it when present.
            // Accumulates over the raw FLOAT pivot channel (bit-identical to the
            // pre-harness inline block).
            int N = ctx.Spec.Models.Count;
            double cSum = 0; int cCnt = 0;
            for (int i = 0; i < N; i++)
            {
                var cv = ctx.Pivot.Channels[ChCape][ctx.CanonIdx[i]];
                if (!float.IsNaN(cv)) { cSum += cv; cCnt++; }
            }
            var capeMean = cCnt > 0 ? cSum / cCnt : double.NaN;

            return CloudFeatureBuilder.ComposeRow(
                ctx.Spec, ctx.Valid, ctx.Vals[ChCc], era5Cc: float.NaN, capeMean: capeMean);
        },
        // Spread features at offset N (per-model cc block first).
        SummaryStats: (ctx, row) => ElementPredictHarness.SpreadFromFeatures(row, ctx.Spec.Models.Count),
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
