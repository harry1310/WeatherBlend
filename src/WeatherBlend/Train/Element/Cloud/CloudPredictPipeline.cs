using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Cloud;

/// <summary>
/// Predict-time pipeline for the cloud-cover blender. Spec-driven via
/// <see cref="BlenderSpec"/> loaded from feature_schema.json. Pulls the 6-slot
/// canonical pivot once, then per lead filters down to spec.Models in spec order.
/// </summary>
public static class CloudPredictPipeline
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
        var canonOrder = FeatureBuilder.CanonicalModelOrder.ToList();
        var targets = leads.Select(L => (Lead: L, Valid: anchor.AddHours(L))).ToArray();
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
            var cc = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                cc[i] = float.IsNaN(p.Cc[ci]) ? double.NaN : p.Cc[ci];
                if (double.IsNaN(cc[i]) && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model cloud cover for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }

            var row = CloudFeatureBuilder.ComposeRow(spec, valid, cc, era5Cc: float.NaN);
            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            var modelCc  = new double?[6];
            var modelRun = new DateTime?[6];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                modelCc[ci]  = double.IsNaN(cc[i]) ? null : cc[i];
                modelRun[ci] = p.RunTimes[ci];
            }

            var spreadStart = N;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "cloud-cover",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = modelCc[0], ModelEcmwf = modelCc[1], ModelIcon  = modelCc[2],
                ModelMf    = modelCc[3], ModelUkmo  = modelCc[4], ModelGem   = modelCc[5],
                RunTimeGfs = modelRun[0], RunTimeEcmwf = modelRun[1], RunTimeIcon = modelRun[2],
                RunTimeMf  = modelRun[3], RunTimeUkmo  = modelRun[4], RunTimeGem  = modelRun[5],
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

    private sealed record Pivoted(float[] Cc, DateTime?[] RunTimes);

    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, CloudCover,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, CloudCover
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = FeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);

        var working = new Dictionary<DateTime, (float[] Cc, DateTime?[] Rt)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var model = r.GetString(1);
            if (!slotByModel.TryGetValue(model, out var idx)) continue;
            if (!working.TryGetValue(valid, out var slot))
            {
                slot = (Enumerable.Repeat(float.NaN, 6).ToArray(), new DateTime?[6]);
                working[valid] = slot;
            }
            slot.Rt[idx] = r.GetDateTime(2);
            slot.Cc[idx] = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
        }
        return working.ToDictionary(kv => kv.Key, kv => new Pivoted(kv.Value.Cc, kv.Value.Rt));
    }
}
