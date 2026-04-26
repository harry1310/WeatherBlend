using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;

namespace WeatherBlend.Train.Element.Humidity;

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
            // Required-models check is per-lead — at 48/72h MF is structurally absent
            // (live API horizon ~36h) and the model was trained without it, so we don't
            // flag it as missing here.
            var requiredModels = HumidityFeatureBuilder.ModelsForLead(lead).ToHashSet();
            var missing = HumidityFeatureBuilder.ModelAccessors
                .Select((acc, i) => (acc.ModelId, i))
                .Where(t => requiredModels.Contains(t.ModelId))
                .Where(t => float.IsNaN(p.Rhs[t.i]) || float.IsNaN(p.DewPoints[t.i]))
                .Select(t => t.ModelId)
                .ToArray();
            if (missing.Length > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing per-model RH/dew for [{Models}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missing), valid);
                continue;
            }

            // Force NaN for models the per-lead training schema excluded — keeps predict
            // input identical to what the model saw in training. Without this, a lucky
            // live MF row at lead 48h would feed a value the model never trained on.
            for (int i = 0; i < HumidityFeatureBuilder.ModelAccessors.Count; i++)
            {
                if (requiredModels.Contains(HumidityFeatureBuilder.ModelAccessors[i].ModelId)) continue;
                p.Rhs[i] = float.NaN;
                p.DewPoints[i] = float.NaN;
            }

            var row = HumidityFeatureBuilder.ComposeRow(valid, p.Rhs, p.DewPoints, era5Rh: float.NaN);
            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.Predict(ml, model, new[] { row })[0];

            // Model values: NaN → null for parquet readability (NaN is what the model
            // sees; null is what the prediction record reports for "not used at this lead").
            static double? Nz(float v) => float.IsNaN(v) ? null : v;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "humidity",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = Nz(p.Rhs[0]), ModelEcmwf = Nz(p.Rhs[1]), ModelIcon  = Nz(p.Rhs[2]),
                ModelMf    = Nz(p.Rhs[3]), ModelUkmo  = Nz(p.Rhs[4]), ModelGem   = Nz(p.Rhs[5]),
                RunTimeGfs = p.RunTimes[0], RunTimeEcmwf = p.RunTimes[1], RunTimeIcon = p.RunTimes[2],
                RunTimeMf  = requiredModels.Contains("meteofrance_seamless") ? p.RunTimes[3] : null,
                RunTimeUkmo  = p.RunTimes[4], RunTimeGem  = p.RunTimes[5],
                Mean = row.RhMean, Std = row.RhStd, Range = row.RhRange,
                FeatureVectorHash = FeatureHash(row),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0}% (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0}%)",
                lead, yhat, valid, row.RhMean);
        }

        return output;
    }

    private sealed record Pivoted(float[] Rhs, float[] DewPoints, DateTime?[] RunTimes);

    /// <summary>
    /// Pull live-cycle pivot scoped to the union of per-lead model memberships:
    /// at 24h we want all 6 models, at 48/72h we want 5 (no MF). Querying the
    /// union (all 6) once and letting the per-lead missing check filter is fine
    /// because MF rows at 48/72h are absent in the live tree anyway — the slot
    /// stays NaN, and the lead-aware check skips it.
    /// </summary>
    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, RelativeHumidity2m, DewPoint2m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, RelativeHumidity2m, DewPoint2m
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

        var working = new Dictionary<DateTime, (float[] Rhs, float[] Dp, DateTime?[] Rt)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var model = r.GetString(1);
            if (!modelSlot.TryGetValue(model, out var idx)) continue;
            if (!working.TryGetValue(valid, out var slot))
            {
                slot = (
                    Enumerable.Repeat(float.NaN, 6).ToArray(),
                    Enumerable.Repeat(float.NaN, 6).ToArray(),
                    new DateTime?[6]);
                working[valid] = slot;
            }
            slot.Rt[idx]  = r.GetDateTime(2);
            slot.Rhs[idx] = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Dp[idx]  = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
        }
        return working.ToDictionary(kv => kv.Key, kv => new Pivoted(kv.Value.Rhs, kv.Value.Dp, kv.Value.Rt));
    }

    private static string FeatureHash(HumidityRow row)
    {
        Span<float> v = stackalloc float[]
        {
            row.RhGfs, row.RhEcmwf, row.RhIcon, row.RhMf, row.RhUkmo, row.RhGem,
            row.DpGfs, row.DpEcmwf, row.DpIcon, row.DpMf, row.DpUkmo, row.DpGem,
            row.RhMean, row.RhStd, row.RhRange,
            row.HourSin, row.HourCos, row.DoySin, row.DoyCos,
        };
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(v);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes));
    }
}
