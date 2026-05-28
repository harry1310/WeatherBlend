using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Oro;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// Predict-time pipeline for the Phase 3 <c>wind_speed_lgb</c> blender.
/// Parallel to <see cref="WindPredictPipeline"/>, but the wider feature
/// set (6 wsp + 6 wdir + 5 ORO_LEAN + 12 SPREAD) means the SQL pivot has
/// to pull seven per-NWP variables instead of two, and ORO_LEAN is
/// derived per row from the NWP-mean wind components + the location's
/// static orographic JSON.
///
/// Spec-driven, same as the train side: per-lead
/// <see cref="BlenderSpec"/> loaded from <c>feature_schema.json</c>;
/// the SQL pulls only spec.Models columns; the feature row is composed
/// via <see cref="WindSpeedLgbFeatureBuilder.ComposeRow"/> so train +
/// predict shapes are byte-for-byte identical.
///
/// Output <see cref="ElementPredictionRow"/> stamps <c>Element="wind"</c>
/// (same physical target as the existing wind blender) — the wind_blend
/// mint command (Slice 3.C) disambiguates by ModelVersion, which carries
/// the <c>_wind_speed_lgb</c> suffix the harness writes at train time.
/// </summary>
public static class WindSpeedLgbPredictPipeline
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

        // Resolve orographic JSON from the same place the trainer reads —
        // ForecastsPath's parent dir, NOT CWD-relative (so parallel
        // smoke runs don't collide). Single load per cycle is fine: the
        // record is small and the pipeline iterates many (lead, valid)
        // pairs against it.
        var oroRoot = Path.Combine(
            Path.GetDirectoryName(forecastsPath)!, "static", "orographic");
        var oroBySlug = OroStaticFeatures.LoadAll(oroRoot);
        if (!oroBySlug.TryGetValue(locationName, out var oro))
        {
            log.LogError(
                "Missing orographic JSON for location '{Loc}' under {Root} — cannot predict wind_speed_lgb.",
                locationName, oroRoot);
            return new List<ElementPredictionRow>();
        }

        // 24 hourly targets per lead bucket — mirrors WindPredictPipeline
        // / TempPredictCommand / PrecipPredictCommand so feels-like and
        // wind_blend (Slice 3.C) join cleanly on (valid_time).
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
                log.LogWarning("  Lead {Lead}h: no live forecast for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, valid);
                continue;
            }
            if (!specs.TryGetValue(lead, out var spec))
            {
                log.LogWarning("  Lead {Lead}h: no BlenderSpec in feature_schema.json; skipping.", lead);
                continue;
            }

            int N = spec.Models.Count;
            var wsp  = new double[N];
            var wdir = new double[N];
            var gust = new double[N];
            var t    = new double[N];
            var td   = new double[N];
            var pres = new double[N];
            var cc   = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                wsp[i]  = float.IsNaN(p.Wsp[ci])  ? double.NaN : p.Wsp[ci];
                wdir[i] = float.IsNaN(p.Wdir[ci]) ? double.NaN : p.Wdir[ci];
                gust[i] = float.IsNaN(p.Gust[ci]) ? double.NaN : p.Gust[ci];
                t[i]    = float.IsNaN(p.Temp[ci]) ? double.NaN : p.Temp[ci];
                td[i]   = float.IsNaN(p.Dew[ci])  ? double.NaN : p.Dew[ci];
                pres[i] = float.IsNaN(p.Pres[ci]) ? double.NaN : p.Pres[ci];
                cc[i]   = float.IsNaN(p.Cloud[ci]) ? double.NaN : p.Cloud[ci];
                if ((double.IsNaN(wsp[i]) || double.IsNaN(wdir[i])) && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model wind for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, string.Join(",", missingRequired), valid);
                continue;
            }

            // ComposeRow handles ORO_LEAN + SPREAD derivation; label is unused at predict.
            var row = WindSpeedLgbFeatureBuilder.ComposeRow(
                spec, valid, wsp, wdir, gust, t, td, pres, cc, oro,
                dunkeswellWspMs: double.NaN);
            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var yhat = TempTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            // Per-NWP wsp slot population — mirrors WindPredictPipeline.
            // ElementPredictionRow has 7 named slots (Gfs..Aifs); JMA
            // (canonical index 7) falls off the named-slot end and is
            // consumed by the LGB but not surfaced in the per-NWP fields.
            var modelSpd = new double?[ElementPredictionRow.PerModelFieldCount];
            var modelRun = new DateTime?[ElementPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= ElementPredictionRow.PerModelFieldCount) continue;
                modelSpd[ci] = double.IsNaN(wsp[i]) ? null : wsp[i];
                modelRun[ci] = p.RunTimes[ci];
            }

            // Spread (mean/std/range) here is over wsp only — matches the
            // existing wind row's Mean/Std/Range semantics. The 12-feature
            // SPREAD block (wsp + gust + t + td + p + cc means and stds)
            // is INSIDE row.Features at offsets >= 2N+5; not surfaced
            // here, just used by the LGB.
            double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
            int present = 0;
            for (int i = 0; i < N; i++)
            {
                if (double.IsNaN(wsp[i])) continue;
                sum += wsp[i]; sumSq += wsp[i] * wsp[i];
                if (wsp[i] < min) min = wsp[i];
                if (wsp[i] > max) max = wsp[i];
                present++;
            }
            var wspMean = present == 0 ? double.NaN : sum / present;
            var wspStd  = present == 0 ? double.NaN
                : Math.Sqrt(Math.Max(0.0, (sumSq / present) - (wspMean * wspMean)));
            var wspRange = present == 0 ? double.NaN : max - min;

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
                ModelAifs  = modelSpd[6],
                RunTimeGfs   = modelRun[0], RunTimeEcmwf = modelRun[1],
                RunTimeIcon  = modelRun[2], RunTimeMf    = modelRun[3],
                RunTimeUkmo  = modelRun[4], RunTimeGem   = modelRun[5],
                RunTimeAifs  = modelRun[6],
                Mean = double.IsNaN(wspMean) ? null : wspMean,
                Std  = double.IsNaN(wspStd)  ? null : wspStd,
                Range = double.IsNaN(wspRange) ? null : wspRange,
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });
            log.LogInformation(
                "  Lead {Lead}h → wind_speed_lgb blend {Blend:0.000} m/s (valid {Valid:yyyy-MM-dd HH:mm}Z, NWP-mean {Mean:0.000} m/s)",
                lead, yhat, valid, wspMean);
        }

        return output;
    }

    private sealed record Pivoted(
        float[] Wsp, float[] Wdir, float[] Gust,
        float[] Temp, float[] Dew, float[] Pres, float[] Cloud,
        DateTime?[] RunTimes);

    private static Dictionary<DateTime, Pivoted> QueryPivot(
        string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        // Pull all 8 canonical NWPs in one pass; spec.Models filtering
        // happens in the per-(lead, valid) loop above. Cheap enough at
        // this row volume to read everything once rather than per-lead.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc,
           WindSpeed10m, WindDirection10m, WindGusts10m,
           Temperature2m, DewPoint2m, SurfacePressure, CloudCover,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc,
       WindSpeed10m, WindDirection10m, WindGusts10m,
       Temperature2m, DewPoint2m, SurfacePressure, CloudCover
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = TempFeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);
        const int Slots = 8;

        var working = new Dictionary<DateTime, (
            float[] Wsp, float[] Wdir, float[] Gust,
            float[] Temp, float[] Dew, float[] Pres, float[] Cloud,
            DateTime?[] RunTimes)>();

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
                    NewNan(Slots), NewNan(Slots), NewNan(Slots),
                    NewNan(Slots), NewNan(Slots), NewNan(Slots), NewNan(Slots),
                    new DateTime?[Slots]);
                working[valid] = slot;
            }
            slot.RunTimes[idx] = r.GetDateTime(2);
            slot.Wsp[idx]   = r.IsDBNull(3) ? float.NaN : (float)r.GetDouble(3);
            slot.Wdir[idx]  = r.IsDBNull(4) ? float.NaN : (float)r.GetDouble(4);
            slot.Gust[idx]  = r.IsDBNull(5) ? float.NaN : (float)r.GetDouble(5);
            slot.Temp[idx]  = r.IsDBNull(6) ? float.NaN : (float)r.GetDouble(6);
            slot.Dew[idx]   = r.IsDBNull(7) ? float.NaN : (float)r.GetDouble(7);
            slot.Pres[idx]  = r.IsDBNull(8) ? float.NaN : (float)r.GetDouble(8);
            slot.Cloud[idx] = r.IsDBNull(9) ? float.NaN : (float)r.GetDouble(9);
        }

        return working.ToDictionary(
            kv => kv.Key,
            kv => new Pivoted(kv.Value.Wsp, kv.Value.Wdir, kv.Value.Gust,
                              kv.Value.Temp, kv.Value.Dew, kv.Value.Pres, kv.Value.Cloud,
                              kv.Value.RunTimes));
    }

    private static float[] NewNan(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = float.NaN;
        return a;
    }
}
