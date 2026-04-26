using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Evaluate.Element;

/// <summary>
/// Pure verification logic for the per-variable element blenders. Mirrors the temperature
/// <c>Verifier</c>: joins predictions to ERA5 truth, stratifies metrics by (version, lead),
/// flags drift when rolling blend MAE exceeds threshold × training test MAE.
///
/// One generic verifier serves all four elements (wind / humidity / radiation / cloud)
/// because their <see cref="ElementPredictionRow"/> shape is uniform — only the truth
/// column read upstream and the displayed units differ. METAR secondary metric is not
/// included here; it lives in a separate stage and is per-element.
/// </summary>
public static class ElementVerifier
{
    /// <summary>Per-model accessor on a prediction row. Order matches the slot ordering
    /// in <see cref="ElementPredictionRow"/>.</summary>
    internal static readonly (string Name, Func<ElementPredictionRow, double?> Get)[] ModelAccessors =
    {
        ("gfs",   p => p.ModelGfs),
        ("ecmwf", p => p.ModelEcmwf),
        ("icon",  p => p.ModelIcon),
        ("mf",    p => p.ModelMf),
        ("ukmo",  p => p.ModelUkmo),
        ("gem",   p => p.ModelGem),
    };

    public sealed class Inputs
    {
        public required IReadOnlyList<ElementPredictionRow> Predictions { get; init; }

        /// <summary>ERA5 truth (units = element's natural units) keyed by valid time.</summary>
        public required IReadOnlyDictionary<DateTime, double> TruthByTime { get; init; }

        public required IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata> MetadataByVersion { get; init; }

        public required DateTime AsOfUtc { get; init; }
        public required int WindowDays { get; init; }
        public required int Era5LatencyDays { get; init; }
        public required double DriftThreshold { get; init; }
    }

    public sealed record VerifyRow(
        string ModelVersion,
        int LeadHours,
        int N,
        double BlendMae,
        double BlendRmse,
        double BlendBias,
        double MeanMae,
        double MeanBias,
        string BestSingleName,
        double BestSingleMae,
        double? ReferenceTestMae,
        bool DriftFlag,
        IReadOnlyDictionary<int, (double Mae, int N)> MaeByMonth,
        IReadOnlyDictionary<int, (double Mae, int N)> MaeByTruthQuintile);

    public static IReadOnlyList<VerifyRow> Compute(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.Era5LatencyDays);

        var kept = inputs.Predictions
            .Where(p => p.ValidTimeUtc >= windowStart && p.ValidTimeUtc <= windowEnd)
            .Where(p => inputs.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();

        var groups = kept
            .GroupBy(p => (p.ModelVersion, p.LeadHours))
            .OrderBy(g => g.Key.ModelVersion, StringComparer.Ordinal)
            .ThenBy(g => g.Key.LeadHours);

        var result = new List<VerifyRow>();
        foreach (var g in groups)
            result.Add(BuildRow(g.Key.ModelVersion, g.Key.LeadHours, g.ToList(), inputs));
        return result;
    }

    private static VerifyRow BuildRow(
        string version,
        int leadHours,
        IReadOnlyList<ElementPredictionRow> preds,
        Inputs inputs)
    {
        var actual = preds.Select(p => inputs.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend  = preds.Select(p => p.BlendValue).ToArray();
        var blendStats = Metrics.Compute(blend, actual);

        var meanPred = preds.Select(RowMean).ToArray();
        var meanStats = Metrics.Compute(meanPred, actual);

        var bestName = "";
        var bestMae  = double.PositiveInfinity;
        foreach (var (name, get) in ModelAccessors)
        {
            var p = preds.Select(row => get(row) ?? double.NaN).ToArray();
            var s = Metrics.Compute(p, actual);
            if (s.N > 0 && s.Mae < bestMae)
            {
                bestMae = s.Mae;
                bestName = name;
            }
        }

        // Stratifications — month (calendar bucket) and truth-value quintile.
        var months = preds.Select(p => p.ValidTimeUtc.Month).ToArray();
        var maeByMonth = Metrics.StratifiedMae(blend, actual, months)
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        var quintiles = Metrics.Quintiles(actual.Select(v => (double)v).ToArray());
        var maeByQuintile = Metrics.StratifiedMae(blend, actual, quintiles)
            .Where(kv => kv.Key >= 0)
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        double? refMae = null;
        var drift = false;
        if (inputs.MetadataByVersion.TryGetValue(version, out var md)
            && md.PerLead.TryGetValue(leadHours.ToString(), out var ls)
            && ls.BlendTestMae > 0)
        {
            refMae = ls.BlendTestMae;
            drift = blendStats.N > 0 && blendStats.Mae > inputs.DriftThreshold * ls.BlendTestMae;
        }

        return new VerifyRow(
            ModelVersion:        version,
            LeadHours:           leadHours,
            N:                   blendStats.N,
            BlendMae:            blendStats.Mae,
            BlendRmse:           blendStats.Rmse,
            BlendBias:           blendStats.Bias,
            MeanMae:             meanStats.Mae,
            MeanBias:            meanStats.Bias,
            BestSingleName:      bestName,
            BestSingleMae:       double.IsPositiveInfinity(bestMae) ? double.NaN : bestMae,
            ReferenceTestMae:    refMae,
            DriftFlag:           drift,
            MaeByMonth:          maeByMonth,
            MaeByTruthQuintile:  maeByQuintile);
    }

    private static double RowMean(ElementPredictionRow p)
    {
        if (p.Mean.HasValue) return p.Mean.Value;
        double sum = 0; int n = 0;
        foreach (var (_, get) in ModelAccessors)
        {
            var v = get(p);
            if (v.HasValue && !double.IsNaN(v.Value)) { sum += v.Value; n++; }
        }
        return n == 0 ? double.NaN : sum / n;
    }
}
