using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Humidity;

/// <summary>
/// Predict-time pipeline for the humidity blender. Spec-driven via BlenderSpec.
/// Pulls the canonical 6-slot pivot once for RH + DP; per lead filters down to
/// spec.Models in spec order.
/// </summary>
public static class HumidityPredictPipeline
{
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
    {
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();
        // 24 hourly targets per lead bucket — see WindPredictPipeline for
        // the rationale (mirrors TempPredictCommand so feels-like joins
        // populate every hour, not just at anchor+lead).
        var anchorDayUtc = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc);
        var targets = leads.SelectMany(L =>
        {
            var dayStart = anchorDayUtc.AddDays(L / 24);
            return Enumerable.Range(0, 24).Select(h => (Lead: L, Valid: dayStart.AddHours(h)));
        }).ToArray();
        var pivot = QueryPivot(forecastsPath, locationName, anchor,
            targets.Min(t => t.Valid), targets.Max(t => t.Valid), ct);

        var ml = new MLContext(seed: 42);
        var output = new List<ElementPredictionRow>();

        foreach (var (lead, valid) in targets)
        {
            ct.ThrowIfCancellationRequested();
            if (!pivot.TryGetValue(valid, out var p))
            {
                log.LogWarning("  Lead {Lead}h: no live forecast for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.", lead, valid);
                continue;
            }
            if (!specs.TryGetValue(lead, out var spec))
            {
                log.LogWarning("  Lead {Lead}h: no BlenderSpec in feature_schema.json; skipping.", lead);
                continue;
            }

            int N = spec.Models.Count;
            var rh = new double[N];
            var dp = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                rh[i] = float.IsNaN(p.Rh[ci]) ? double.NaN : p.Rh[ci];
                dp[i] = float.IsNaN(p.Dp[ci]) ? double.NaN : p.Dp[ci];
                if ((double.IsNaN(rh[i]) || double.IsNaN(dp[i])) && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model rh/dp for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }

            // Aux ensemble means (schema-gated: only post-2026-06-10 bundles
            // carry temp_mean in their saved spec). NaN-safe means over THIS
            // spec's model set — mirrors the trainer's SQL AVG over the
            // spec-filtered `latest` rows.
            double[]? aux = null;
            if (spec.FeatureNames.Contains("temp_mean"))
            {
                double MeanOf(Func<int, float> get)
                {
                    double s = 0; int k = 0;
                    for (int i = 0; i < N; i++)
                    {
                        var ci = canonOrder.IndexOf(spec.Models[i]);
                        var x = get(ci);
                        if (float.IsNaN(x)) continue;
                        s += x; k++;
                    }
                    return k == 0 ? double.NaN : s / k;
                }
                aux = new[]
                {
                    MeanOf(ci => p.Temp[ci]),
                    MeanOf(ci => float.IsNaN(p.Temp[ci]) || float.IsNaN(p.Dp[ci]) ? float.NaN : p.Temp[ci] - p.Dp[ci]),
                    MeanOf(ci => p.CloudLow[ci]),
                    MeanOf(ci => p.CloudMid[ci]),
                    MeanOf(ci => p.CloudHigh[ci]),
                    MeanOf(ci => p.Wind[ci]),
                };
            }

            var row = HumidityFeatureBuilder.ComposeRow(spec, valid, rh, dp, era5Rh: float.NaN, aux);
            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TempTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            // ElementPredictionRow has 7 named slots (Gfs..Gem + Aifs).
            var modelRh  = new double?[ElementPredictionRow.PerModelFieldCount];
            var modelRun = new DateTime?[ElementPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= ElementPredictionRow.PerModelFieldCount) continue;     // JMA — not in humidity output schema
                modelRh[ci]  = double.IsNaN(rh[i]) ? null : rh[i];
                modelRun[ci] = p.RunTimes[ci];
            }

            // Spread features at offset 2N (rh + dp blocks first).
            var spreadStart = 2 * N;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "humidity",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = modelRh[0], ModelEcmwf = modelRh[1], ModelIcon  = modelRh[2],
                ModelMf    = modelRh[3], ModelUkmo  = modelRh[4], ModelGem   = modelRh[5],
                ModelAifs  = modelRh[6],
                RunTimeGfs = modelRun[0], RunTimeEcmwf = modelRun[1], RunTimeIcon = modelRun[2],
                RunTimeMf  = modelRun[3], RunTimeUkmo  = modelRun[4], RunTimeGem  = modelRun[5],
                RunTimeAifs = modelRun[6],
                Mean = row.Features[spreadStart + 0],
                Std  = row.Features[spreadStart + 1],
                Range = row.Features[spreadStart + 2],
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0}% (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0}%)",
                lead, yhat, valid, row.Features[spreadStart + 0]);
        }
        return output;
    }

    private sealed record Pivoted(
        float[] Rh, float[] Dp, DateTime?[] RunTimes,
        float[] Temp, float[] CloudLow, float[] CloudMid, float[] CloudHigh, float[] Wind);

    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, RelativeHumidity2m, DewPoint2m,
           Temperature2m, CloudCoverLow, CloudCoverMid, CloudCoverHigh, WindSpeed10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, RelativeHumidity2m, DewPoint2m,
       Temperature2m, CloudCoverLow, CloudCoverMid, CloudCoverHigh, WindSpeed10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = TempFeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);

        static float[] NaNs() => Enumerable.Repeat(float.NaN, 8).ToArray();
        var working = new Dictionary<DateTime, Pivoted>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var model = r.GetString(1);
            if (!slotByModel.TryGetValue(model, out var idx)) continue;
            if (!working.TryGetValue(valid, out var slot))
            {
                slot = new Pivoted(NaNs(), NaNs(), new DateTime?[8], NaNs(), NaNs(), NaNs(), NaNs(), NaNs());
                working[valid] = slot;
            }
            float Col(int i) => r.IsDBNull(i) ? float.NaN : (float)r.GetDouble(i);
            slot.RunTimes[idx]  = r.GetDateTime(2);
            slot.Rh[idx]        = Col(3);
            slot.Dp[idx]        = Col(4);
            slot.Temp[idx]      = Col(5);
            slot.CloudLow[idx]  = Col(6);
            slot.CloudMid[idx]  = Col(7);
            slot.CloudHigh[idx] = Col(8);
            slot.Wind[idx]      = Col(9);
        }
        return working;
    }
}
