using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;

namespace WeatherBlend.Train.Element.Radiation;

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
            // Required-models check is per-lead — at 48/72h UKMO is structurally absent
            // (Open-Meteo serves UKMO radiation only at lead < 48h). At 24h UKMO is
            // optional (kept with NaN tolerance because it's only ~26% null).
            var requiredModels = RadiationFeatureBuilder.ModelsForLead(lead).ToHashSet();
            // At 24h UKMO is in ModelsForLead but is optional (NaN allowed). At 48/72h
            // UKMO is excluded entirely. So the required-non-null set is "required ∖ {ukmo}"
            // at every lead — UKMO never blocks a prediction.
            var requiredNonNull = requiredModels.Where(m => m != "ukmo_seamless").ToHashSet();
            var missing = RadiationFeatureBuilder.ModelAccessors
                .Select((acc, i) => (acc.ModelId, i))
                .Where(t => requiredNonNull.Contains(t.ModelId))
                .Where(t => float.IsNaN(p.Sw[t.i]))
                .Select(t => t.ModelId)
                .ToArray();
            if (missing.Length > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required SW for [{Models}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missing), valid);
                continue;
            }

            // Force NaN for models excluded from this lead's training schema (UKMO at 48/72h).
            for (int i = 0; i < RadiationFeatureBuilder.ModelAccessors.Count; i++)
            {
                if (requiredModels.Contains(RadiationFeatureBuilder.ModelAccessors[i].ModelId)) continue;
                p.Sw[i] = float.NaN;
                p.Direct[i] = float.NaN;
                p.Diffuse[i] = float.NaN;
            }

            var row = RadiationFeatureBuilder.ComposeRow(valid, p.Sw, p.Direct, p.Diffuse, era5Sw: float.NaN);
            var model = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TemperatureTrainer.Predict(ml, model, new[] { row })[0];

            static double? Nz(float v) => float.IsNaN(v) ? null : v;
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = "shortwave-radiation",
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = Nz(p.Sw[0]), ModelEcmwf = Nz(p.Sw[1]), ModelIcon  = Nz(p.Sw[2]),
                ModelMf    = Nz(p.Sw[3]), ModelUkmo  = Nz(p.Sw[4]), ModelGem   = Nz(p.Sw[5]),
                RunTimeGfs = p.RunTimes[0], RunTimeEcmwf = p.RunTimes[1], RunTimeIcon = p.RunTimes[2],
                RunTimeMf  = p.RunTimes[3],
                RunTimeUkmo = requiredModels.Contains("ukmo_seamless") ? p.RunTimes[4] : null,
                RunTimeGem  = p.RunTimes[5],
                Mean = row.SwMean, Std = row.SwStd, Range = row.SwRange,
                FeatureVectorHash = FeatureHash(row),
            });
            log.LogInformation("  Lead {Lead}h → blend {Blend:0.0} W/m² (valid {Valid:yyyy-MM-dd HH:mm}Z, mean {Mean:0.0} W/m²)",
                lead, yhat, valid, row.SwMean);
        }
        return output;
    }

    private sealed record Pivoted(float[] Sw, float[] Direct, float[] Diffuse, DateTime?[] RunTimes);

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

        var modelSlot = new Dictionary<string, int>
        {
            ["gfs_seamless"] = 0, ["ecmwf_ifs025"] = 1, ["icon_seamless"] = 2,
            ["meteofrance_seamless"] = 3, ["ukmo_seamless"] = 4, ["gem_seamless"] = 5,
        };

        var working = new Dictionary<DateTime, (float[] Sw, float[] Dr, float[] Df, DateTime?[] Rt)>();
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
                    Enumerable.Repeat(float.NaN, 6).ToArray(),
                    new DateTime?[6]);
                working[valid] = slot;
            }
            slot.Rt[idx] = r.GetDateTime(2);
            slot.Sw[idx] = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Dr[idx] = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
            slot.Df[idx] = r.IsDBNull(5) ? float.NaN : (float)r.GetDouble(5);
        }
        return working.ToDictionary(kv => kv.Key, kv => new Pivoted(kv.Value.Sw, kv.Value.Dr, kv.Value.Df, kv.Value.Rt));
    }

    private static string FeatureHash(RadiationRow row)
    {
        Span<float> v = stackalloc float[]
        {
            row.SwGfs, row.SwEcmwf, row.SwIcon, row.SwMf, row.SwUkmo, row.SwGem,
            row.DirectGfs, row.DirectEcmwf, row.DirectIcon, row.DirectMf, row.DirectUkmo, row.DirectGem,
            row.DiffuseGfs, row.DiffuseEcmwf, row.DiffuseIcon, row.DiffuseMf, row.DiffuseUkmo, row.DiffuseGem,
            row.SwMean, row.SwStd, row.SwRange,
            row.HourSin, row.HourCos, row.DoySin, row.DoyCos,
        };
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(v);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes));
    }
}
