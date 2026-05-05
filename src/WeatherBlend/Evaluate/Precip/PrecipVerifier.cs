using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Evaluate.Precip;

/// <summary>
/// Pure precipitation verification. Stratifies rolling Brier/BSS/frequency-bias
/// by (truth station, model version, lead) and flags drift when rolling blend
/// Brier exceeds a threshold × the training-time test Brier for that lead.
///
/// Reuses the ClimatologyPWet column on each prediction row so the BSS reference
/// needs no access to <c>climatology.json</c> at verification time — predictions
/// carry their own reference forward. Per-model wet-day indicators (1 if Precip≥0.1mm,
/// 0 otherwise, NaN if missing) are recomputed from the prediction row so the best-
/// single and mean-of-models baselines are observed-in-production rankings.
///
/// Inputs are pre-joined in-memory; no DuckDB here so every metric path is unit-testable.
/// </summary>
public static class PrecipVerifier
{
    /// <summary>Per-phase per-model column lists. 3a/3c (offset_day) read
    /// from the canonical 8-NWP slots; 3d (exact-runtime) reads from the
    /// 5 *Exact slots since its model identities don't fully overlap
    /// (raw IFS oper vs Open-Meteo ecmwf_ifs025, raw MO Global vs
    /// ukmo_seamless, plus UKV which has no offset_day twin).</summary>
    internal static readonly string[] ModelNamesOffsetDay =
        { "gfs", "ecmwf", "icon", "mf", "ukmo", "gem", "aifs", "jma" };

    /// <summary>Phase 3d per-model column list — 5 exact-runtime sources.
    /// Order matches PrecipExactFeatureBuilder canonical column order plus
    /// UKV at the end.</summary>
    internal static readonly string[] ModelNamesExact =
        { "gfs_exact", "ifs_oper_exact", "aifs_oper_exact", "moglobal_exact", "ukv_exact" };

    /// <summary>Resolve the per-NWP column list for a given training phase.
    /// Defaults to the offset_day list — preserves behaviour for any version
    /// whose metadata is missing (legacy artefacts) or has an unknown phase.</summary>
    internal static string[] ModelNamesForPhase(string? phase) =>
        string.Equals(phase, "3d", StringComparison.Ordinal) ? ModelNamesExact : ModelNamesOffsetDay;

    [Obsolete("Use ModelNamesForPhase(phase). Kept for tests that pre-date 3d.")]
    internal static readonly string[] ModelNames = ModelNamesOffsetDay;

    public const double WetThresholdMm = PrecipFeatureBuilder.WetThresholdMm;

    public sealed class Inputs
    {
        public required IReadOnlyList<PrecipPredictionRow> Predictions { get; init; }

        /// <summary>Hourly truth mm/hour keyed by (TruthStation slug, ValidTimeUtc). Must cover
        /// the window and the persistence-lookback range (ValidTime − LeadHours).</summary>
        public required IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> TruthByStationTime { get; init; }

        /// <summary>One entry per distinct (TruthStation, ModelVersion) that appears in Predictions.</summary>
        public required IReadOnlyDictionary<(string Station, string Version), ModelArtifact.TrainingMetadata> MetadataByKey { get; init; }

        public required DateTime AsOfUtc { get; init; }
        public required int WindowDays { get; init; }

        /// <summary>Skip predictions whose ValidTimeUtc is within this many days of AsOf —
        /// EA rainfall API readings settle as they're validated, so a small cutoff avoids
        /// scoring against provisional data.</summary>
        public required int LatencyDays { get; init; }

        public required double DriftThreshold { get; init; }

        /// <summary>Reliability-diagram bucket count. 10 equal-width bins over [0,1].</summary>
        public int ReliabilityBins { get; init; } = 10;
    }

    public sealed record VerifyRow(
        string TruthStation,
        string ModelVersion,
        int LeadHours,
        int N,
        int WetN,
        double WetRate,
        double BlendBrier,
        double ClimBrier,
        double Bss,
        double MeanOfModelsBrier,
        string BestSingleName,
        double BestSingleBrier,
        double? PersistenceBrier,
        int PersistenceDropped,
        double FrequencyBiasAt05,
        double? ReferenceTestBrier,
        bool DriftFlag,
        IReadOnlyList<PrecipMetrics.ReliabilityBin> Reliability);

    public static IReadOnlyList<VerifyRow> Compute(Inputs inputs)
    {
        var windowStart = inputs.AsOfUtc.AddDays(-inputs.WindowDays);
        var windowEnd   = inputs.AsOfUtc.AddDays(-inputs.LatencyDays);

        // Keep rows inside the window and with matching truth for this row's station.
        var kept = inputs.Predictions
            .Where(p => p.ValidTimeUtc >= windowStart && p.ValidTimeUtc <= windowEnd)
            .Where(p => inputs.TruthByStationTime.TryGetValue(p.TruthStation, out var byTime)
                        && byTime.ContainsKey(p.ValidTimeUtc))
            .ToList();

        var groups = kept
            .GroupBy(p => (p.TruthStation, p.ModelVersion, p.LeadHours))
            .OrderBy(g => g.Key.TruthStation, StringComparer.Ordinal)
            .ThenBy(g => g.Key.ModelVersion, StringComparer.Ordinal)
            .ThenBy(g => g.Key.LeadHours);

        var result = new List<VerifyRow>();
        foreach (var g in groups)
            result.Add(BuildRow(g.Key.TruthStation, g.Key.ModelVersion, g.Key.LeadHours, g.ToList(), inputs));
        return result;
    }

    private static VerifyRow BuildRow(
        string station,
        string version,
        int leadHours,
        IReadOnlyList<PrecipPredictionRow> preds,
        Inputs inputs)
    {
        var byTime = inputs.TruthByStationTime[station];
        var truthMm    = preds.Select(p => byTime[p.ValidTimeUtc]).ToArray();
        var truthBin   = truthMm.Select(mm => mm >= WetThresholdMm ? 1.0 : 0.0).ToArray();
        var blendProb  = preds.Select(p => p.ProbWet).ToArray();
        var climProb   = preds.Select(p => p.ClimatologyPWet).ToArray();

        var blendBrier = PrecipMetrics.Brier(blendProb, truthBin);
        var climBrier  = PrecipMetrics.Brier(climProb,  truthBin);
        var bss        = PrecipMetrics.BrierSkillScore(blendBrier, climBrier);

        // Mean-of-models: PrecipAgreementWet01 is fraction-of-6-predicting-wet, already a
        // probability in [0,1]. Rows with all models null → 0 (matches trainer convention).
        var meanProb = preds.Select(p => p.PrecipAgreementWet01 ?? 0.0).ToArray();
        var meanBrier = PrecipMetrics.Brier(meanProb, truthBin);

        // Best single: binarise each per-model precip at 0.1mm, pick lowest Brier over the
        // window. Missing models yield NaN — PrecipMetrics.Brier silently skips them.
        // Phase-aware ModelNames: 3a/3c reads offset_day slots, 3d reads exact slots.
        var modelNames = ModelNamesForPhase(PhaseFor(station, version, inputs));
        var bestName = "";
        var bestBrier = double.PositiveInfinity;
        foreach (var name in modelNames)
        {
            var p = preds.Select(row => PrecipIndicator(row, name)).ToArray();
            var b = PrecipMetrics.Brier(p, truthBin);
            if (!double.IsNaN(b) && b < bestBrier)
            {
                bestBrier = b;
                bestName = name;
            }
        }

        // Persistence: truth at (ValidTime − LeadHours), binarised at 0.1mm. Dropped rows
        // (missing lagged truth) don't count toward N but are reported.
        var persProb = new double[preds.Count];
        var persDropped = 0;
        for (int i = 0; i < preds.Count; i++)
        {
            var lagT = preds[i].ValidTimeUtc.AddHours(-leadHours);
            if (byTime.TryGetValue(lagT, out var mm))
                persProb[i] = mm >= WetThresholdMm ? 1.0 : 0.0;
            else
            {
                persProb[i] = double.NaN;
                persDropped++;
            }
        }
        var persBrier = PrecipMetrics.Brier(persProb, truthBin);

        // Frequency bias at p=0.5: forecast-wet count / observed-wet count.
        var blendBinaryAt05 = blendProb.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
        var freqBias = PrecipMetrics.FrequencyBias(blendBinaryAt05, truthBin);

        var reliability = PrecipMetrics.Reliability(blendProb, truthBin, inputs.ReliabilityBins);

        // Reference Brier from training metadata. Phase 3a stores the held-out test Brier
        // in the BlendTestMae slot (same field, repurposed — see ModelArtifact comments).
        double? refBrier = null;
        var drift = false;
        if (inputs.MetadataByKey.TryGetValue((station, version), out var md)
            && md.PerLead.TryGetValue(leadHours.ToString(), out var ls)
            && ls.BlendTestMae > 0)
        {
            refBrier = ls.BlendTestMae;
            drift = !double.IsNaN(blendBrier) && blendBrier > inputs.DriftThreshold * ls.BlendTestMae;
        }

        var wetN = (int)truthBin.Sum();

        return new VerifyRow(
            TruthStation:       station,
            ModelVersion:       version,
            LeadHours:          leadHours,
            N:                  preds.Count,
            WetN:               wetN,
            WetRate:            preds.Count == 0 ? double.NaN : (double)wetN / preds.Count,
            BlendBrier:         blendBrier,
            ClimBrier:          climBrier,
            Bss:                bss,
            MeanOfModelsBrier:  meanBrier,
            BestSingleName:     bestName,
            BestSingleBrier:    double.IsPositiveInfinity(bestBrier) ? double.NaN : bestBrier,
            PersistenceBrier:   double.IsNaN(persBrier) ? (double?)null : persBrier,
            PersistenceDropped: persDropped,
            FrequencyBiasAt05:  freqBias,
            ReferenceTestBrier: refBrier,
            DriftFlag:          drift,
            Reliability:        reliability);
    }

    private static double PrecipIndicator(PrecipPredictionRow p, string name)
    {
        double? mm = name switch
        {
            "gfs"   => p.PrecipGfs,
            "ecmwf" => p.PrecipEcmwf,
            "icon"  => p.PrecipIcon,
            "mf"    => p.PrecipMf,
            "ukmo"  => p.PrecipUkmo,
            "gem"   => p.PrecipGem,
            "aifs"  => p.PrecipAifs,
            "jma"   => p.PrecipJma,
            // Phase 3d (exact-runtime) per-model slots.
            "gfs_exact"      => p.PrecipGfsExact,
            "ifs_oper_exact" => p.PrecipIfsOperExact,
            "aifs_oper_exact"=> p.PrecipAifsOperExact,
            "moglobal_exact" => p.PrecipMoGlobalExact,
            "ukv_exact"      => p.PrecipUkvExact,
            _ => null,
        };
        if (!mm.HasValue) return double.NaN;
        return mm.Value >= WetThresholdMm ? 1.0 : 0.0;
    }

    /// <summary>Resolve the training phase for a (station, version) pair by
    /// looking it up in <see cref="Inputs.MetadataByKey"/>. Versions whose
    /// metadata isn't loaded get null → falls back to the offset_day column
    /// list, which matches every shipped phase pre-3d.</summary>
    private static string? PhaseFor(string station, string version, Inputs inputs)
    {
        if (inputs.MetadataByKey.TryGetValue((station, version), out var md))
            return md.Phase;
        return null;
    }
}
