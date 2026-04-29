using WeatherBlend.Train;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Dry-daytime-window target construction. Given hourly rainfall totals at a
/// gauge, emit one binary label per (UTC calendar day, window length)
/// answering "did there exist a run of N consecutive dry hours within the
/// configured local-time daytime window for that day". A dry hour is one
/// strictly below the 0.1 mm/h threshold (4 of 4 15-minute EA readings).
///
/// Rules:
///   - The daytime window is given as an IANA-tz local-hour range (e.g.
///     09–18 Europe/London) and resolved to per-target-day UTC bounds via
///     <see cref="DaytimeWindow.UtcHourRangeFor"/>. DST is handled at scan
///     time, so the meaning of "daytime" stays stable through BST↔GMT
///     transitions.
///   - A day is usable only if every UTC hour falling inside the resolved
///     daytime range has a truth reading. A gap inside the daytime window
///     drops the day for every window length; gaps outside (e.g. an
///     overnight hour) don't matter because we never look at them.
///   - Cross-midnight dry stretches that bridge the local daytime window
///     of two adjacent UTC days are not credited — each day is scored
///     independently. For 09–18 London this never matters.
///   - Wet threshold matches the 3a occurrence classifier (0.1 mm/h) so
///     the two phases speak the same language about "wet".
///
/// The legacy "anywhere in 24h UTC" behaviour is still reachable by passing
/// a 0–24 daytime spec, used by tests pinning the original semantics.
/// </summary>
public static class DryWindowLabelBuilder
{
    public const double WetThresholdMm = PrecipFeatureBuilder.WetThresholdMm;

    /// <summary>
    /// Singleton "scan the whole UTC day" spec — preserves the pre-daytime
    /// label semantics for tests and any caller that hasn't been migrated.
    /// </summary>
    public static readonly DaytimeWindow FullUtcDay = new(0, 24, TimeZoneInfo.Utc);

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
    /// Build labels for the full span implied by <paramref name="hourly"/>.
    /// Each day is scored against the per-day UTC hour range derived from
    /// <paramref name="daytime"/>. Days that lack a truth reading at any hour
    /// inside that range are dropped. Window lengths greater than the
    /// daytime duration are rejected — there is no possible positive label.
    /// </summary>
    public static BuildResult Build(
        IReadOnlyList<HourlyTruth> hourly,
        IReadOnlyList<int> windowLengths,
        DaytimeWindow daytime)
    {
        if (windowLengths.Count == 0)
            throw new ArgumentException("At least one window length required.", nameof(windowLengths));
        foreach (var n in windowLengths)
        {
            if (n < 1 || n > daytime.DurationHours)
                throw new ArgumentException(
                    $"Window length {n} out of range [1, {daytime.DurationHours}] for the configured daytime window.",
                    nameof(windowLengths));
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
        // rather than silently skipped.
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

            var (startHour, endHour) = daytime.UtcHourRangeFor(d);
            if (startHour >= endHour)
            {
                // Daytime window doesn't overlap this UTC day at all (only
                // possible for far-east tzs that push the window outside the
                // calendar day). Skip without counting as either usable or
                // dropped — the day is genuinely irrelevant to the model.
                candidate--;
                continue;
            }

            // Need every hour inside the daytime range populated. A gap inside
            // the window hides a possible dry/wet flip; drop the day.
            bool complete = true;
            for (int h = startHour; h < endHour; h++)
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
                labels.Add(new DailyLabel(d, n, HasDryWindow(hours, n, startHour, endHour)));
        }

        return new BuildResult(labels, candidate, usable, dropped);
    }

    /// <summary>
    /// Does any length-<paramref name="n"/> sliding window inside the
    /// half-open UTC hour range <c>[startHour, endHour)</c> have every hour
    /// &lt; 0.1 mm? Dry = strictly below the wet threshold.
    /// </summary>
    /// <remarks>
    /// Assumes <paramref name="hours"/> is populated for every index in
    /// <c>[startHour, endHour)</c> (caller-checked). Returns false when
    /// <paramref name="n"/> exceeds the window duration.
    /// </remarks>
    public static bool HasDryWindow(double?[] hours, int n, int startHour, int endHour)
    {
        if (n < 1 || n > endHour - startHour) return false;

        // Walk the daytime hours, tracking the current consecutive-dry run.
        int run = 0;
        for (int i = startHour; i < endHour; i++)
        {
            var dry = hours[i].HasValue && hours[i]!.Value < WetThresholdMm;
            run = dry ? run + 1 : 0;
            if (run >= n) return true;
        }
        return false;
    }
}
