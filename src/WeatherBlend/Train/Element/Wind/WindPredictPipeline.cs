using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// Predict-time pipeline for the wind blender. Pulls one live-cycle pivot for the
/// {valid_24h, valid_48h, valid_72h} window, composes a <see cref="WindRow"/> per lead
/// using <see cref="WindFeatureBuilder.ComposeRow"/> (so the predict-time and
/// train-time row shapes are byte-for-byte identical), then runs the per-lead
/// model.zip from <paramref name="versionDir"/>.
///
/// MF is excluded — the trained model has no MF feature columns. The output
/// <see cref="ElementPredictionRow.ModelMf"/> is therefore always null for wind.
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

            // Required-models check is per-lead — UKMO (and MF) are excluded from
            // training and live data isn't expected, so don't flag them as "missing".
            var requiredModels = WindFeatureBuilder.ModelsForLead(lead).ToHashSet();
            var missing = WindFeatureBuilder.ModelAccessors
                .Select((acc, i) => (acc.ModelId, i))
                .Where(t => requiredModels.Contains(t.ModelId))
                .Where(t => float.IsNaN(p.Speeds[t.i]) || float.IsNaN(p.Directions[t.i]))
                .Select(t => t.ModelId)
                .ToArray();
            if (missing.Length > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing per-model wind speed/dir for [{Models}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missing), valid);
                continue;
            }

            // Force NaN for models the per-lead training schema excluded — keeps predict
            // input identical to what the model saw in training.
            for (int i = 0; i < WindFeatureBuilder.ModelAccessors.Count; i++)
            {
                if (requiredModels.Contains(WindFeatureBuilder.ModelAccessors[i].ModelId)) continue;
                p.Speeds[i] = float.NaN;
                p.Directions[i] = float.NaN;
            }

            var row = WindFeatureBuilder.ComposeRow(valid, p.Speeds, p.Directions, era5WindSpeed: float.NaN);

            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.Predict(ml, model, new[] { row })[0];

            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "wind",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                // Per-model values — UKMO and MF excluded from the wind blender at every
                // lead, so always null in the prediction row. Order in p.Speeds matches
                // the WindFeatureBuilder accessor list: gfs, ecmwf, icon, ukmo, gem.
                ModelGfs   = p.Speeds[0],
                ModelEcmwf = p.Speeds[1],
                ModelIcon  = p.Speeds[2],
                ModelMf    = null,
                ModelUkmo  = null,
                ModelGem   = p.Speeds[4],
                RunTimeGfs   = p.RunTimes[0],
                RunTimeEcmwf = p.RunTimes[1],
                RunTimeIcon  = p.RunTimes[2],
                RunTimeMf    = null,
                RunTimeUkmo  = null,
                RunTimeGem   = p.RunTimes[4],
                Mean = row.SpdMean,
                Std  = row.SpdStd,
                Range = row.SpdRange,
                FeatureVectorHash = FeatureHash(row),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.000} m/s (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.000} m/s)",
                lead, yhat, valid, row.SpdMean);
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
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, WindSpeed10m, WindDirection10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
      AND Model IN ('gfs_seamless','ecmwf_ifs025','icon_seamless','ukmo_seamless','gem_seamless')
)
SELECT ValidTimeUtc, Model, RunTimeUtc, WindSpeed10m, WindDirection10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        // Order matches WindFeatureBuilder.ModelAccessors: gfs, ecmwf, icon, ukmo, gem.
        var modelSlot = new Dictionary<string, int>
        {
            ["gfs_seamless"]  = 0,
            ["ecmwf_ifs025"]  = 1,
            ["icon_seamless"] = 2,
            ["ukmo_seamless"] = 3,
            ["gem_seamless"]  = 4,
        };

        var working = new Dictionary<DateTime, (float[] Speeds, float[] Directions, DateTime?[] RunTimes)>();
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
                    Enumerable.Repeat(float.NaN, 5).ToArray(),
                    Enumerable.Repeat(float.NaN, 5).ToArray(),
                    new DateTime?[5]);
                working[valid] = slot;
            }
            slot.RunTimes[idx] = r.GetDateTime(2);
            slot.Speeds[idx]    = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Directions[idx] = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
        }

        return working.ToDictionary(
            kv => kv.Key,
            kv => new Pivoted(kv.Value.Speeds, kv.Value.Directions, kv.Value.RunTimes));
    }

    /// <summary>Stable SHA-256 hex of all feature floats in WindRow internal-column order.</summary>
    private static string FeatureHash(WindRow row)
    {
        Span<float> v = stackalloc float[]
        {
            row.SpdGfs, row.SpdEcmwf, row.SpdIcon, row.SpdUkmo, row.SpdGem,
            row.DirSinGfs, row.DirCosGfs, row.DirSinEcmwf, row.DirCosEcmwf,
            row.DirSinIcon, row.DirCosIcon, row.DirSinUkmo, row.DirCosUkmo,
            row.DirSinGem, row.DirCosGem,
            row.SpdMean, row.SpdStd, row.SpdRange,
            row.HourSin, row.HourCos, row.DoySin, row.DoyCos,
        };
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(v);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes));
    }
}
