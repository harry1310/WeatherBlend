using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Gust;

/// <summary>
/// Predict-time pipeline for the wind-gust blender. Spec-driven: per-lead
/// <see cref="BlenderSpec"/> loaded from <c>feature_schema.json</c>; the SQL
/// pivot pulls only spec.Models columns; the feature row is composed via
/// <see cref="WindGustFeatureBuilder.ComposeRow"/> so train + predict shapes
/// are byte-for-byte identical.
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
/// </summary>
public static class WindGustPredictPipeline
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
        // 24 hourly targets per lead bucket — mirrors WindPredictPipeline so
        // gust rows emit a full hourly trajectory matching the wind card.
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
            var gust = new double[N];
            var wsp = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                gust[i] = float.IsNaN(p.Gusts[ci]) ? double.NaN : p.Gusts[ci];
                wsp[i] = float.IsNaN(p.Speeds[ci]) ? double.NaN : p.Speeds[ci];
                if (double.IsNaN(gust[i]) && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model gust for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }

            var row = WindGustFeatureBuilder.ComposeRow(spec, valid, gust, wsp, era5GustMs: float.NaN);
            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TempTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            // Per-model output: populate only spec.Models with the per-NWP GUST values.
            // ElementPredictionRow has 7 named slots (Gfs..Gem + Aifs) and a JMA tail
            // beyond that — wind-gust will populate at most 4 of those (the production
            // gust NWPs); the rest stay null.
            var modelGust = new double?[ElementPredictionRow.PerModelFieldCount];
            var modelRun = new DateTime?[ElementPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= ElementPredictionRow.PerModelFieldCount) continue;     // beyond named slots
                modelGust[ci] = double.IsNaN(gust[i]) ? null : gust[i];
                modelRun[ci] = p.RunTimes[ci];
            }

            // Cross-NWP gust spread for the site card. Independent of the feature
            // vector (which ends in ratio_mean/ratio_std — useful to LightGBM, not
            // to the card). NaN-safe over the N gust values.
            double sum = 0, sumSq = 0, gMin = double.MaxValue, gMax = double.MinValue;
            int present = 0;
            for (int i = 0; i < N; i++)
            {
                var x = gust[i];
                if (double.IsNaN(x)) continue;
                sum += x;
                sumSq += x * x;
                if (x < gMin) gMin = x;
                if (x > gMax) gMax = x;
                present++;
            }
            double? gustMean = present == 0 ? null : sum / present;
            double? gustStd = present == 0 ? null
                : Math.Sqrt(Math.Max(0.0, (sumSq / present) - (gustMean!.Value * gustMean.Value)));
            double? gustRange = present == 0 ? null : gMax - gMin;

            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "wind-gust",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = modelGust[0], ModelEcmwf = modelGust[1], ModelIcon  = modelGust[2],
                ModelMf    = modelGust[3], ModelUkmo  = modelGust[4], ModelGem   = modelGust[5],
                ModelAifs  = modelGust[6],
                RunTimeGfs   = modelRun[0], RunTimeEcmwf = modelRun[1],
                RunTimeIcon  = modelRun[2], RunTimeMf    = modelRun[3],
                RunTimeUkmo  = modelRun[4], RunTimeGem   = modelRun[5],
                RunTimeAifs  = modelRun[6],
                Mean = gustMean,
                Std = gustStd,
                Range = gustRange,
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });
            log.LogInformation("  Lead {Lead}h → blend gust {Blend:0.000} m/s (valid {Valid:yyyy-MM-dd HH:mm}Z, NWP-mean {Mean:0.000} m/s)",
                lead, yhat, valid, gustMean ?? double.NaN);
        }

        return output;
    }

    private sealed record Pivoted(float[] Gusts, float[] Speeds, DateTime?[] RunTimes);

    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        // Pull all canonical NWPs — the spec filter happens later in the loop. Cheap
        // enough and lets the per-lead spec drive what survives without re-querying.
        // wsp pulled alongside gust because ComposeRow needs both to build the ratio.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, WindGusts10m, WindSpeed10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, WindGusts10m, WindSpeed10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = TempFeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);

        var working = new Dictionary<DateTime, (float[] Gusts, float[] Speeds, DateTime?[] RunTimes)>();
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
                    new DateTime?[8]);
                working[valid] = slot;
            }
            slot.RunTimes[idx]  = r.GetDateTime(2);
            slot.Gusts[idx]    = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Speeds[idx]   = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
        }

        return working.ToDictionary(
            kv => kv.Key,
            kv => new Pivoted(kv.Value.Gusts, kv.Value.Speeds, kv.Value.RunTimes));
    }
}
