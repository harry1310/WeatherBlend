using WeatherBlend.Models;

namespace WeatherBlend.Predict.StartHour;

/// <summary>
/// Pure derivation of the dry-window start-hour curve. Given hourly P(wet)
/// across the daytime UTC range and the dry-window blender's daily P(∃ N-hour
/// dry block), produce one <see cref="StartHourPredictionRow"/> per candidate
/// start hour.
///
/// <para>The curve assumes hourly independence — <c>p_s = ∏_{h=s..s+N-1}(1−q_h)</c>
/// — which is wrong in detail (NWP errors are correlated across consecutive
/// hours) but produces a useful <em>shape</em>. The shape is normalised to a
/// proper conditional distribution <c>π_s = p_s / Σ p_s</c>, then re-scaled
/// against the dry-window blender's calibrated daily marginal so the magnitude
/// comes from a model that's actually scored against truth.</para>
///
/// <para>Backtest result on 14 months of held-out historical days (Bellever,
/// 6h window, lead 24h, 57 informative days): top-1 accuracy 78.9% vs uniform
/// baseline 59.4%; log-loss skill +0.108. See
/// <c>scripts/DryWindowStartHour/backtest_historical.py</c>.</para>
/// </summary>
public static class StartHourCurveDerivation
{
    /// <summary>
    /// Compose the curve for one (station, window, lead, target date). Returns
    /// an empty list when the inputs can't support a curve — caller treats
    /// that as "skip this composite for this anchor cycle".
    /// </summary>
    /// <param name="hourlyPWet">Hour-of-day → q_h (P(wet) at that UTC hour) on
    /// <paramref name="targetDate"/>. Must contain every hour in
    /// <c>[daytimeStartUtc, daytimeEndUtc)</c>; one missing hour drops the
    /// whole day rather than silently fudging a uniform fill.</param>
    public static List<StartHourPredictionRow> Derive(
        string locationName,
        string truthStation,
        int windowHours,
        string startHourVersion,
        DateTime predictionMadeAtUtc,
        int leadHours,
        DateTime targetDateUtc,
        int daytimeStartUtc,
        int daytimeEndUtc,
        IReadOnlyDictionary<int, double> hourlyPWet,
        double dailyProbAnyBlock,
        string precipVersion,
        string dryWindowVersion)
    {
        if (windowHours <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowHours), "Must be positive.");
        if (daytimeStartUtc < 0 || daytimeStartUtc >= 24)
            throw new ArgumentOutOfRangeException(nameof(daytimeStartUtc));
        if (daytimeEndUtc <= daytimeStartUtc || daytimeEndUtc > 24)
            throw new ArgumentOutOfRangeException(nameof(daytimeEndUtc));

        var span = daytimeEndUtc - daytimeStartUtc;
        if (windowHours > span) return new List<StartHourPredictionRow>();

        // Every daytime hour must be populated. A gap means we'd be silently
        // assuming dryness for a missing forecast hour, which would bias the
        // shape; better to drop the whole composite for the day.
        for (int h = daytimeStartUtc; h < daytimeEndUtc; h++)
            if (!hourlyPWet.ContainsKey(h))
                return new List<StartHourPredictionRow>();

        var nStarts = span - windowHours + 1;
        var raw = new double[nStarts];
        for (int i = 0; i < nStarts; i++)
        {
            int s = daytimeStartUtc + i;
            double p = 1.0;
            for (int h = s; h < s + windowHours; h++)
            {
                var q = Math.Clamp(hourlyPWet[h], 0.0, 1.0);
                p *= 1.0 - q;
            }
            raw[i] = p;
        }

        // Normalise. Σ ≈ 0 means every candidate window contains a near-certain
        // wet hour; the curve has no shape, fall back to uniform so the row
        // count + ordering is stable for downstream consumers.
        var sum = raw.Sum();
        var conditional = new double[nStarts];
        if (sum > 0)
        {
            for (int i = 0; i < nStarts; i++) conditional[i] = raw[i] / sum;
        }
        else
        {
            for (int i = 0; i < nStarts; i++) conditional[i] = 1.0 / nStarts;
        }

        var rows = new List<StartHourPredictionRow>(nStarts);
        for (int i = 0; i < nStarts; i++)
        {
            rows.Add(new StartHourPredictionRow
            {
                LocationName = locationName,
                TruthStation = truthStation,
                WindowHours = windowHours,
                ModelVersion = startHourVersion,
                PredictionMadeAtUtc = predictionMadeAtUtc,
                TargetDateUtc = targetDateUtc,
                LeadHours = leadHours,
                StartHourUtc = daytimeStartUtc + i,
                RawProduct = raw[i],
                ConditionalProb = conditional[i],
                CalibratedProb = conditional[i] * dailyProbAnyBlock,
                DailyProbAnyBlock = dailyProbAnyBlock,
                PrecipVersion = precipVersion,
                DryWindowVersion = dryWindowVersion,
            });
        }
        return rows;
    }
}
