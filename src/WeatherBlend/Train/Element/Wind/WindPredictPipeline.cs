using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// Predict-time pipeline for the wind blender. Spec-driven: per-lead
/// <see cref="BlenderSpec"/> loaded from <c>feature_schema.json</c>; the SQL pivot
/// pulls only spec.Models columns; the feature row is composed via
/// <see cref="WindFeatureBuilder.ComposeRow"/> so train + predict shapes are
/// byte-for-byte identical.
///
/// Output <see cref="ElementPredictionRow"/> still has six per-model slots
/// (Gfs/Ecmwf/Icon/Mf/Ukmo/Gem). We populate only the slots that map back to a
/// model in <c>spec.Models</c>; the rest stay <c>null</c>. The fossil 5-to-6
/// <c>OutputModelFields</c> mapping helper is gone — the spec is the single
/// source of truth for which output fields get populated.
/// </summary>
public static class WindPredictPipeline
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
            var spd = new double[N];
            var dir = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                spd[i] = float.IsNaN(p.Speeds[ci]) ? double.NaN : p.Speeds[ci];
                dir[i] = float.IsNaN(p.Directions[ci]) ? double.NaN : p.Directions[ci];
                if (double.IsNaN(spd[i]) && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model wind for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }

            var row = WindFeatureBuilder.ComposeRow(spec, valid, spd, dir, era5WindSpeed: float.NaN);
            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            // Per-model output fields: populate only spec.Models, null elsewhere.
            // ElementPredictionRow has 6 named slots; AIFS contributes implicitly via BlendValue.
            var modelSpd = new double?[6];
            var modelRun = new DateTime?[6];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= 6) continue;     // AIFS — not in 6-wide output yet
                modelSpd[ci] = double.IsNaN(spd[i]) ? null : spd[i];
                modelRun[ci] = p.RunTimes[ci];
            }

            // Spread features at offset 3*N (after spd + sin block + cos block).
            var spreadStart = 3 * N;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "wind",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = modelSpd[0], ModelEcmwf = modelSpd[1], ModelIcon  = modelSpd[2],
                ModelMf    = modelSpd[3], ModelUkmo  = modelSpd[4], ModelGem   = modelSpd[5],
                RunTimeGfs   = modelRun[0], RunTimeEcmwf = modelRun[1],
                RunTimeIcon  = modelRun[2], RunTimeMf    = modelRun[3],
                RunTimeUkmo  = modelRun[4], RunTimeGem   = modelRun[5],
                Mean = row.Features[spreadStart + 0],
                Std  = row.Features[spreadStart + 1],
                Range = row.Features[spreadStart + 2],
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.000} m/s (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.000} m/s)",
                lead, yhat, valid, row.Features[spreadStart + 0]);
        }

        return output;
    }

    private sealed record Pivoted(float[] Speeds, float[] Directions, DateTime?[] RunTimes);

    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        // Pull all 6 models — the spec filter happens later in the loop. Cheap enough
        // and lets the per-lead spec drive what survives without re-querying per lead.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, WindSpeed10m, WindDirection10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, WindSpeed10m, WindDirection10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        // Slot ordering matches FeatureBuilder.CanonicalModelOrder so the predict
        // command can map spec.Models[i] → canonical-slot index.
        var slotByModel = FeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);

        var working = new Dictionary<DateTime, (float[] Speeds, float[] Directions, DateTime?[] RunTimes)>();
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
                    Enumerable.Repeat(float.NaN, 7).ToArray(),
                    Enumerable.Repeat(float.NaN, 7).ToArray(),
                    new DateTime?[7]);
                working[valid] = slot;
            }
            slot.RunTimes[idx]  = r.GetDateTime(2);
            slot.Speeds[idx]    = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Directions[idx] = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
        }

        return working.ToDictionary(
            kv => kv.Key,
            kv => new Pivoted(kv.Value.Speeds, kv.Value.Directions, kv.Value.RunTimes));
    }
}
