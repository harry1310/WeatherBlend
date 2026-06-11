using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Generic predict-time pipeline used by every Element blender — the predict
/// mirror of <see cref="ElementTrainerHarness"/>. Owns the ~80% that the five
/// per-element pipelines (wind / gust / humidity / radiation / cloud) used to
/// repeat verbatim:
///   - per-lead <see cref="BlenderSpec"/> load from <c>feature_schema.json</c>
///   - hourly target-grid expansion (24 valid-times per lead bucket)
///   - the canonical-slot SQL pivot (one query, all NWPs, freshest run per
///     (ValidTime, Model) — only the projected columns differ per element)
///   - spec-lookup + missing-required-model guards (warn + skip per target)
///   - per-lead LightGBM model cache (re-deserialising the same zip up to
///     24x per lead was pure waste — mirrors PrecipPredictCommand)
///   - canonical-slot → <see cref="ElementPredictionRow"/> per-model field
///     mapping (populate only spec.Models; null elsewhere)
///
/// Per-element specifics are injected via <see cref="Descriptor"/>: which
/// forecast columns to pivot, which channels a REQUIRED model must supply,
/// how to compose the feature row (incl. any schema-gated aux means), and
/// what the site card's Mean/Std/Range should hold. Each element's pipeline
/// file keeps only that genuinely-different code + its rationale comments.
///
/// Behaviour is bit-identical to the pre-harness pipelines — the element
/// smoke tests (ElementBlenderSmokeTests / ElementWindSmokeTests) are the
/// gate, and the per-element ComposeRow delegates operate on exactly the
/// same float/double values the inlined loops saw.
/// </summary>
public static class ElementPredictHarness
{
    /// <summary>
    /// One valid-time's pivoted forecast data in CANONICAL slot order.
    /// <see cref="Channels"/>[c][slot] is pivot column c for canonical model
    /// slot <c>slot</c> (float.NaN when that model has no row / NULL value);
    /// slot indices follow <see cref="TempFeatureBuilder.CanonicalModelOrder"/>.
    /// </summary>
    public sealed record PivotRow(float[][] Channels, DateTime?[] RunTimes);

    /// <summary>
    /// Everything a per-element delegate needs for ONE (lead, valid) target.
    /// </summary>
    /// <param name="Spec">The lead's <see cref="BlenderSpec"/> (loaded from the bundle — schema-gated features key off THIS, not the code's current BuildSpec).</param>
    /// <param name="Valid">Target valid time (UTC).</param>
    /// <param name="Vals">Per pivot-channel, an N-length array aligned with <c>Spec.Models</c> (double.NaN when missing) — the shape ComposeRow wants.</param>
    /// <param name="Pivot">The raw canonical-slot pivot row — aux computations that must stay bit-identical to the old inline code (float accumulation) read this.</param>
    /// <param name="CanonIdx">Canonical slot index per spec model i (<c>CanonIdx[i]</c> = slot of <c>Spec.Models[i]</c>).</param>
    public sealed record RowContext(
        BlenderSpec Spec,
        DateTime Valid,
        double[][] Vals,
        PivotRow Pivot,
        int[] CanonIdx);

    /// <summary>
    /// Per-element wiring. Channel 0 is ALWAYS the element's primary/output
    /// variable: it populates the per-model output slots and feeds the
    /// missing-required guard by default.
    /// </summary>
    /// <param name="Element">Value stamped on <see cref="ElementPredictionRow.Element"/> (e.g. "wind", "wind-gust").</param>
    /// <param name="PivotColumns">Forecast-tree columns to project, channel order. Pull all canonical NWPs — the per-lead spec filter happens in the loop; cheap enough and lets the spec drive what survives without re-querying per lead.</param>
    /// <param name="RequiredChannels">Channels a REQUIRED model must have non-NaN for the target to score (wind/gust/cloud/radiation: just {0}; humidity: {0,1} — rh AND dp feed the feature row).</param>
    /// <param name="MissingNoun">Human noun for the missing-required skip log (e.g. "wind", "rh/dp", "gust", "cloud cover", "SW").</param>
    /// <param name="ComposeRow">Build the feature row via the element's FeatureBuilder.ComposeRow so train + predict shapes stay byte-for-byte identical. Owns any per-element aux block (schema-gated means etc.).</param>
    /// <param name="SummaryStats">Mean/Std/Range for the site card. Most elements echo the feature vector's spread block (<see cref="SpreadFromFeatures"/>); gust computes an independent cross-NWP spread.</param>
    /// <param name="LogResult">Per-target success log line — formats/units differ per element.</param>
    public sealed record Descriptor(
        string Element,
        string[] PivotColumns,
        int[] RequiredChannels,
        string MissingNoun,
        Func<RowContext, RegressionTrainingRow> ComposeRow,
        Func<RowContext, RegressionTrainingRow, (double? Mean, double? Std, double? Range)> SummaryStats,
        Action<ILogger, int, double, DateTime, double?> LogResult);

    /// <summary>
    /// Standard SummaryStats: echo the feature vector's ensemble-spread block
    /// (mean/std/range packed at <paramref name="spreadStart"/> by the
    /// element's ComposeRow). Floats widen to double; NaN passes through —
    /// matching what the pre-harness pipelines wrote to the card columns.
    /// </summary>
    public static (double? Mean, double? Std, double? Range) SpreadFromFeatures(
        RegressionTrainingRow row, int spreadStart)
        => (row.Features[spreadStart + 0], row.Features[spreadStart + 1], row.Features[spreadStart + 2]);

    public static List<ElementPredictionRow> PredictForCycle(
        ILogger log,
        Descriptor d,
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
        // 24 hourly targets per lead bucket — mirrors TempPredictCommand /
        // PrecipPredictCommand so each predict cycle emits a full hourly
        // trajectory `[anchor-day + L/24 days, +24h)` per lead, not just
        // a single point at anchor + L hours. Feels-like joins temp +
        // element rows on (valid_time); without this, elements were
        // sparse to 4 valid_times/day at HH ∈ {03,09,15,21} and 9/17
        // home tiles per day showed no Feels-like / UTCI chip.
        var anchorDayUtc = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc);
        var targets = leads.SelectMany(L =>
        {
            var dayStart = anchorDayUtc.AddDays(L / 24);
            return Enumerable.Range(0, 24).Select(h => (Lead: L, Valid: dayStart.AddHours(h)));
        }).ToArray();
        var pivot = QueryPivot(d, forecastsPath, locationName, anchor,
            targets.Min(t => t.Valid), targets.Max(t => t.Valid), ct);

        var ml = new MLContext(seed: 42);

        // Per-lead model cache — the per-target loop below revisits each
        // lead up to 24x (one hourly target per hour of the lead's day),
        // and re-deserialising the same LightGBM zip every iteration was
        // pure waste. Mirrors PrecipPredictCommand's modelCache pattern.
        var modelCache = new Dictionary<int, ITransformer>();
        ITransformer ModelFor(int modelLead)
        {
            if (!modelCache.TryGetValue(modelLead, out var m))
                modelCache[modelLead] = m = ModelArtifact.LoadLeadModel(ml, versionDir, modelLead, out _);
            return m;
        }
        var output = new List<ElementPredictionRow>();

        int channels = d.PivotColumns.Length;
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

            // Re-shape the canonical 8-slot pivot into N-length per-spec-model
            // arrays (float widens to double exactly; NaN maps to NaN).
            int N = spec.Models.Count;
            var canonIdx = new int[N];
            for (int i = 0; i < N; i++) canonIdx[i] = canonOrder.IndexOf(spec.Models[i]);
            var vals = new double[channels][];
            for (int c = 0; c < channels; c++)
            {
                vals[c] = new double[N];
                for (int i = 0; i < N; i++)
                {
                    var x = p.Channels[c][canonIdx[i]];
                    vals[c][i] = float.IsNaN(x) ? double.NaN : x;
                }
            }

            // Missing-required guard: every REQUIRED model must supply every
            // required channel (e.g. humidity needs rh AND dp per model).
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                if (!requiredSet.Contains(spec.Models[i])) continue;
                foreach (var c in d.RequiredChannels)
                {
                    if (!double.IsNaN(vals[c][i])) continue;
                    missingRequired.Add(spec.Models[i]);
                    break;
                }
            }
            if (missingRequired.Count > 0)
            {
                log.LogWarning("  Lead {Lead}h: missing required per-model {Noun} for [{M}] at valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    lead, d.MissingNoun, string.Join(",", missingRequired), valid);
                continue;
            }

            var ctx = new RowContext(spec, valid, vals, p, canonIdx);
            var row = d.ComposeRow(ctx);
            var loadedModel = ModelFor(lead);
            var yhat = TempTrainer.PredictVector(ml, loadedModel, spec, new[] { row })[0];

            // Per-model output fields: populate only spec.Models with channel 0
            // (the element's primary variable), null elsewhere.
            // ElementPredictionRow has 7 named slots (Gfs..Gem + Aifs); any
            // canonical slot beyond that (JMA) has no output field — skip it.
            var modelVals = new double?[ElementPredictionRow.PerModelFieldCount];
            var modelRun = new DateTime?[ElementPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonIdx[i];
                if (ci >= ElementPredictionRow.PerModelFieldCount) continue;     // JMA — not in the element output schema
                modelVals[ci] = double.IsNaN(vals[0][i]) ? null : vals[0][i];
                modelRun[ci] = p.RunTimes[ci];
            }

            var (mean, std, range) = d.SummaryStats(ctx, row);
            output.Add(new ElementPredictionRow
            {
                LocationName = locationName,
                Element = d.Element,
                ModelVersion = modelVersion,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                BlendValue = yhat,
                ModelGfs   = modelVals[0], ModelEcmwf = modelVals[1], ModelIcon  = modelVals[2],
                ModelMf    = modelVals[3], ModelUkmo  = modelVals[4], ModelGem   = modelVals[5],
                ModelAifs  = modelVals[6],
                RunTimeGfs   = modelRun[0], RunTimeEcmwf = modelRun[1],
                RunTimeIcon  = modelRun[2], RunTimeMf    = modelRun[3],
                RunTimeUkmo  = modelRun[4], RunTimeGem   = modelRun[5],
                RunTimeAifs  = modelRun[6],
                Mean = mean,
                Std = std,
                Range = range,
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
            });
            d.LogResult(log, lead, yhat, valid, mean);
        }

        return output;
    }

    /// <summary>
    /// One query for the whole cycle: freshest run per (ValidTime, Model)
    /// inside the live window, all canonical NWPs, the descriptor's columns.
    /// Slot ordering matches <see cref="TempFeatureBuilder.CanonicalModelOrder"/>
    /// so the per-target loop can map spec.Models[i] → canonical-slot index.
    /// </summary>
    private static Dictionary<DateTime, PivotRow> QueryPivot(
        Descriptor d, string forecastsPath, string locationName, DateTime asOf,
        DateTime earliestValid, DateTime latestValid, CancellationToken ct)
    {
        var fcGlob = SqlGlob.Escape(Path.Combine(forecastsPath, "**", "*.parquet"));
        var live = PredictForecastFilters.LiveCycleAsOf(locationName, asOf, earliestValid, latestValid);
        var cols = string.Join(", ", d.PivotColumns);
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, {cols},
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)
    WHERE {live}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, {cols}
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var slotByModel = TempFeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);

        // 8 canonical slots (incl. JMA) — slots beyond ElementPredictionRow's
        // 7 named output fields still participate in feature composition.
        const int slots = 8;
        int channels = d.PivotColumns.Length;
        var working = new Dictionary<DateTime, PivotRow>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            var model = r.GetString(1);
            if (!slotByModel.TryGetValue(model, out var idx)) continue;
            if (!working.TryGetValue(valid, out var slot))
            {
                var ch = new float[channels][];
                for (int c = 0; c < channels; c++)
                    ch[c] = Enumerable.Repeat(float.NaN, slots).ToArray();
                slot = new PivotRow(ch, new DateTime?[slots]);
                working[valid] = slot;
            }
            slot.RunTimes[idx] = r.GetDateTime(2);
            for (int c = 0; c < channels; c++)
                slot.Channels[c][idx] = r.IsDBNull(3 + c) ? float.NaN : (float)r.GetDouble(3 + c);
        }
        return working;
    }
}
