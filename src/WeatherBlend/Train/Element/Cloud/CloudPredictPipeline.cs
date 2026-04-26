using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;

namespace WeatherBlend.Train.Element.Cloud;

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
            // Required-models check is per-lead — at 48/72h MF is structurally absent.
            var requiredModels = CloudFeatureBuilder.ModelsForLead(lead).ToHashSet();
            var missing = CloudFeatureBuilder.ModelAccessors
                .Select((acc, i) => (acc.ModelId, i))
                .Where(t => requiredModels.Contains(t.ModelId))
                .Where(t => float.IsNaN(p.Cc[t.i]))
                .Select(t => t.ModelId)
                .ToArray();
            if (missing.Length > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing per-model cloud cover for [{Models}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missing), valid);
                continue;
            }

            // Force NaN for models excluded from this lead's training schema.
            for (int i = 0; i < CloudFeatureBuilder.ModelAccessors.Count; i++)
            {
                if (requiredModels.Contains(CloudFeatureBuilder.ModelAccessors[i].ModelId)) continue;
                p.Cc[i] = float.NaN;
            }

            var row = CloudFeatureBuilder.ComposeRow(valid, p.Cc, era5Cc: float.NaN);
            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.Predict(ml, model, new[] { row })[0];

            static double? Nz(float v) => float.IsNaN(v) ? null : v;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "cloud-cover",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = Nz(p.Cc[0]), ModelEcmwf = Nz(p.Cc[1]), ModelIcon  = Nz(p.Cc[2]),
                ModelMf    = Nz(p.Cc[3]), ModelUkmo  = Nz(p.Cc[4]), ModelGem   = Nz(p.Cc[5]),
                RunTimeGfs = p.RunTimes[0], RunTimeEcmwf = p.RunTimes[1], RunTimeIcon = p.RunTimes[2],
                RunTimeMf  = requiredModels.Contains("meteofrance_seamless") ? p.RunTimes[3] : null,
                RunTimeUkmo  = p.RunTimes[4], RunTimeGem  = p.RunTimes[5],
                Mean = row.CcMean, Std = row.CcStd, Range = row.CcRange,
                FeatureVectorHash = FeatureHash(row),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0}% (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0}%)",
                lead, yhat, valid, row.CcMean);
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

        var modelSlot = new Dictionary<string, int>
        {
            ["gfs_seamless"] = 0, ["ecmwf_ifs025"] = 1, ["icon_seamless"] = 2,
            ["meteofrance_seamless"] = 3, ["ukmo_seamless"] = 4, ["gem_seamless"] = 5,
        };

        var working = new Dictionary<DateTime, (float[] Cc, DateTime?[] Rt)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var model = r.GetString(1);
            if (!modelSlot.TryGetValue(model, out var idx)) continue;
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

    private static string FeatureHash(CloudRow row)
    {
        Span<float> v = stackalloc float[]
        {
            row.CcGfs, row.CcEcmwf, row.CcIcon, row.CcMf, row.CcUkmo, row.CcGem,
            row.CcMean, row.CcStd, row.CcRange,
            row.HourSin, row.HourCos, row.DoySin, row.DoyCos,
        };
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(v);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes));
    }
}
