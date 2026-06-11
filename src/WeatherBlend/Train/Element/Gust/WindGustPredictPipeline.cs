using Microsoft.Extensions.Logging;
using WeatherBlend.Models;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Gust;

/// <summary>
/// Predict-time pipeline for the wind-gust blender. Spec-driven: per-lead
/// BlenderSpec loaded from <c>feature_schema.json</c>; the SQL pivot pulls
/// only spec.Models columns; the feature row is composed via
/// <see cref="WindGustFeatureBuilder.ComposeRow"/> (which owns the per-NWP
/// gust/speed ratio + clip logic) so train + predict shapes are byte-for-byte
/// identical.
///
/// Output <see cref="ElementPredictionRow"/> uses semantic overload of the
/// per-model slots: <c>ModelGfs/Icon/Gem/Ukmo</c> carry per-NWP gust m/s
/// (not wind speed — the row class is shared across elements; the
/// <c>Element</c> field discriminates). ECMWF / MF / AIFS slots stay null
/// because those NWPs publish no gust on Open-Meteo Previous Runs.
///
/// <c>Mean / Std / Range</c> hold cross-NWP gust spread, computed
/// independently from the feature vector — the gust feature vector ends in
/// <c>gust_ratio_mean / gust_ratio_std</c>, which are useful to the LightGBM
/// but not what the site card needs.
///
/// Orchestration lives in <see cref="ElementPredictHarness"/>; this file
/// keeps the gust specifics.
/// </summary>
public static class WindGustPredictPipeline
{
    // Channel 0 = gust (primary/output). wsp pulled alongside because
    // ComposeRow needs both to build the ratio.
    private const int ChGust = 0, ChWsp = 1;

    private static readonly ElementPredictHarness.Descriptor D = new(
        Element: "wind-gust",
        PivotColumns: new[] { "WindGusts10m", "WindSpeed10m" },
        RequiredChannels: new[] { ChGust },
        MissingNoun: "gust",
        ComposeRow: ctx => WindGustFeatureBuilder.ComposeRow(
            ctx.Spec, ctx.Valid, ctx.Vals[ChGust], ctx.Vals[ChWsp], era5GustMs: float.NaN),
        SummaryStats: (ctx, row) =>
        {
            // Cross-NWP gust spread for the site card. Independent of the feature
            // vector (which ends in ratio_mean/ratio_std — useful to LightGBM, not
            // to the card). NaN-safe over the N gust values; null when no NWP
            // supplied a gust at all.
            var s = InterModelSpread.From(ctx.Vals[ChGust]);
            return double.IsNaN(s.Mean)
                ? (null, null, null)
                : (s.Mean, s.Std, s.Range);
        },
        LogResult: (log, lead, yhat, valid, mean) =>
            log.LogInformation("  Lead {Lead}h → blend gust {Blend:0.000} m/s (valid {Valid:yyyy-MM-dd HH:mm}Z, NWP-mean {Mean:0.000} m/s)",
                lead, yhat, valid, mean ?? double.NaN));

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
