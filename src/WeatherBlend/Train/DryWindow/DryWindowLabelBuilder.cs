using WeatherBlend.Train;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3b target construction. Given hourly rainfall totals at a gauge,
/// emit one binary label per (UTC calendar day, window length) answering
/// "did there exist a run of N consecutive hours within that UTC day where
/// every hour was &lt; 0.1 mm".
///
/// Rules:
///   - UTC calendar day boundaries only. A dry period that spans midnight
///     (22:00 Monday → 02:00 Tuesday) is split by this convention and
///     contributes to neither day's label. Documented as a limitation.
///   - A day is usable only if every hour in [00:00, 23:00] has a truth
///     reading that passed hourly-aggregation quality filtering (4 of 4
///     15-minute readings). Any missing or filtered hour → day dropped
///     for all window lengths; unknowable whether the gap hid a dry run.
///   - Wet threshold matches the 3a occurrence classifier (0.1 mm/h) so
///     the two phases speak the same language about "wet".
/// </summary>
public static class DryWindowLabelBuilder
{
    public const double WetThresholdMm = PrecipFeatureBuilder.WetThresholdMm;

    /// <summary>One quality-passed hour of rainfall. <paramref name="HourUtc"/>
    /// is expected to be midnight-aligned (<c>date_trunc('hour', ObservedTimeUtc)</c>).</summary>
    public sealed record HourlyTruth(DateTime HourUtc, double PrecipMmHour);

    /// <summary>One (day, window) training label.</summary>
    public sealed record DailyLabel(DateOnly Date, int WindowHours, bool HasDryWindow);

    public sealed record BuildResult(
        IReadOnlyList<DailyLabel> Labels,
        int CandidateDays,
        int UsableDays,
        int DroppedDays)
    {
        public double DroppedFraction => CandidateDays == 0 ? 0.0 : (double)DroppedDays / CandidateDays;
    }

    /// <summary>
    /// Build labels for the full span implied by <paramref name="hourly"/>. The span
    /// runs from the first full UTC day in the data to the last full UTC day. Days
    /// between those bounds with fewer than 24 readings are marked unusable and
    /// excluded from <see cref="BuildResult.Labels"/>.
    /// </summary>
    public static BuildResult Build(
        IReadOnlyList<HourlyTruth> hourly,
        IReadOnlyList<int> windowLengths)
    {
        if (windowLengths.Count == 0)
            throw new ArgumentException("At least one window length required.", nameof(windowLengths));
        foreach (var n in windowLengths)
        {
            if (n < 1 || n > 24)
                throw new ArgumentException($"Window length {n} out of range [1,24].", nameof(windowLengths));
        }

        if (hourly.Count == 0)
            return new BuildResult(Array.Empty<DailyLabel>(), 0, 0, 0);

        // Bucket by UTC date. Require Kind=Utc so equality keying doesn't depend
        // on local-time ambiguity.
        var byDate = new Dictionary<DateOnly, double?[]>();
        foreach (var h in hourly)
        {
            if (h.HourUtc.Kind != DateTimeKind.Utc && h.HourUtc.Kind != DateTimeKind.Unspecified)
                throw new ArgumentException($"Hour {h.HourUtc:o} must be UTC.", nameof(hourly));
            var date = DateOnly.FromDateTime(h.HourUtc);
            if (!byDate.TryGetValue(date, out var arr))
            {
                arr = new double?[24];
                byDate[date] = arr;
            }
            arr[h.HourUtc.Hour] = h.PrecipMmHour;
        }

        // Iterate the contiguous date range so missing days are counted as dropped
        // rather than silently skipped. A missing whole day means no label at all —
        // that's the same as a day with any missing hour.
        var firstDate = byDate.Keys.Min();
        var lastDate  = byDate.Keys.Max();

        var labels = new List<DailyLabel>();
        int candidate = 0, usable = 0, dropped = 0;

        for (var d = firstDate; d <= lastDate; d = d.AddDays(1))
        {
            candidate++;
            if (!byDate.TryGetValue(d, out var hours))
            {
                dropped++;
                continue;
            }

            // Need all 24 hours populated. Missing hour → drop the day for every window length.
            bool complete = true;
            for (int h = 0; h < 24; h++)
            {
                if (!hours[h].HasValue) { complete = false; break; }
            }
            if (!complete)
            {
                dropped++;
                continue;
            }

            usable++;
            foreach (var n in windowLengths)
                labels.Add(new DailyLabel(d, n, HasDryWindow(hours, n)));
        }

        return new BuildResult(labels, candidate, usable, dropped);
    }

    /// <summary>
    /// Does any length-<paramref name="n"/> sliding window inside the 24 hours
    /// have every hour &lt; 0.1 mm? Dry = strictly below the wet threshold.
    /// </summary>
    /// <remarks>
    /// Assumes <paramref name="hours"/> is fully populated (caller-checked).
    /// Returns false for <paramref name="n"/> greater than 24 — not reachable
    /// via the public API because <see cref="Build"/> rejects out-of-range
    /// window lengths up front.
    /// </remarks>
    public static bool HasDryWindow(double?[] hours, int n)
    {
        if (n < 1 || n > 24) return false;

        // Mark each hour dry/wet once.
        Span<bool> dry = stackalloc bool[24];
        for (int i = 0; i < 24; i++)
            dry[i] = hours[i].HasValue && hours[i]!.Value < WetThresholdMm;

        // Count consecutive dry hours; any run of length ≥ n qualifies.
        int run = 0;
        for (int i = 0; i < 24; i++)
        {
            run = dry[i] ? run + 1 : 0;
            if (run >= n) return true;
        }
        return false;
    }
}
