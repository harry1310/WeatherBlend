using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Radiation;

/// <summary>
/// Predict-time pipeline for the shortwave-radiation blender. Spec-driven via BlenderSpec.
/// Pulls the canonical 6-slot pivot once for SW + direct + diffuse; per lead filters
/// down to spec.Models in spec order.
/// </summary>
public static class RadiationPredictPipeline
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
            var sw = new double[N];
            var dr = new double[N];
            var df = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                sw[i] = float.IsNaN(p.Sw[ci]) ? double.NaN : p.Sw[ci];
                dr[i] = float.IsNaN(p.Dr[ci]) ? double.NaN : p.Dr[ci];
                df[i] = float.IsNaN(p.Df[ci]) ? double.NaN : p.Df[ci];
                if (double.IsNaN(sw[i]) && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model SW for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }

            var row = RadiationFeatureBuilder.ComposeRow(spec, valid, sw, dr, df, era5Sw: float.NaN);
            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            // ElementPredictionRow has 7 named slots (Gfs..Gem + Aifs).
            var modelSw  = new double?[ElementPredictionRow.PerModelFieldCount];
            var modelRun = new DateTime?[ElementPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= ElementPredictionRow.PerModelFieldCount) continue;     // JMA — not in radiation output schema (no JMA SW data via Open-Meteo anyway)
                modelSw[ci]  = double.IsNaN(sw[i]) ? null : sw[i];
                modelRun[ci] = p.RunTimes[ci];
            }

            // Spread features at offset 3N (sw + direct + diffuse blocks first).
            var spreadStart = 3 * N;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "shortwave-radiation",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = modelSw[0], ModelEcmwf = modelSw[1], ModelIcon  = modelSw[2],
                ModelMf    = modelSw[3], ModelUkmo  = modelSw[4], ModelGem   = modelSw[5],
                ModelAifs  = modelSw[6],
                RunTimeGfs = modelRun[0], RunTimeEcmwf = modelRun[1], RunTimeIcon = modelRun[2],
                RunTimeMf  = modelRun[3], RunTimeUkmo  = modelRun[4], RunTimeGem  = modelRun[5],
                RunTimeAifs = modelRun[6],
                Mean = row.Features[spreadStart + 0],
                Std  = row.Features[spreadStart + 1],
                Range = row.Features[spreadStart + 2],
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0} W/m² (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0} W/m²)",
                lead, yhat, valid, row.Features[spreadStart + 0]);
        }
        return output;
    }

    private sealed record Pivoted(float[] Sw, float[] Dr, float[] Df, DateTime?[] RunTimes);

    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, ShortwaveRadiation, DirectRadiation, DiffuseRadiation,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, ShortwaveRadiation, DirectRadiation, DiffuseRadiation
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = FeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);

        var working = new Dictionary<DateTime, (float[] Sw, float[] Dr, float[] Df, DateTime?[] Rt)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var model = r.GetString(1);
            if (!slotByModel.TryGetValue(model, out var idx)) continue;
            if (!working.TryGetValue(valid, out var slot))
            {
                slot = (
                    Enumerable.Repeat(float.NaN, 8).ToArray(),
                    Enumerable.Repeat(float.NaN, 8).ToArray(),
                    Enumerable.Repeat(float.NaN, 8).ToArray(),
                    new DateTime?[8]);
                working[valid] = slot;
            }
            slot.Rt[idx] = r.GetDateTime(2);
            slot.Sw[idx] = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Dr[idx] = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
            slot.Df[idx] = r.IsDBNull(5) ? float.NaN : (float)r.GetDouble(5);
        }
        return working.ToDictionary(kv => kv.Key,
            kv => new Pivoted(kv.Value.Sw, kv.Value.Dr, kv.Value.Df, kv.Value.Rt));
    }
}
