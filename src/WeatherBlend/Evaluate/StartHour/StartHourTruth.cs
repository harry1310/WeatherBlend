using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Evaluate.StartHour;

/// <summary>
/// Derive truth start hours from EA rainfall. For a UTC daytime range
/// <c>[startUtc, endUtc)</c> on one target date, return the set of start
/// hours <c>s</c> such that every hour in <c>[s, s+windowHours)</c> was dry
/// per the gauge (each hour < 0.1 mm via the same 4-of-4 gate
/// <see cref="DryWindowLabelBuilder"/> uses for training labels).
///
/// Returns <c>null</c> when any hour inside the daytime range is missing
/// from <paramref name="hourlyMm"/> — partial coverage means we can't
/// distinguish "dry" from "unobserved", and a truthful verify must drop
/// the day rather than fudge.
/// </summary>
public static class StartHourTruth
{
    public const double WetThresholdMm = DryWindowLabelBuilder.WetThresholdMm;

    public static HashSet<int>? ValidStartsFor(
        IReadOnlyDictionary<int, double> hourlyMm,
        int daytimeStartUtc,
        int daytimeEndUtc,
        int windowHours)
    {
        if (windowHours < 1) return new HashSet<int>();
        var span = daytimeEndUtc - daytimeStartUtc;
        if (windowHours > span) return new HashSet<int>();

        // Every daytime hour must be present. A missing hour drops the day
        // so partial-coverage scores never enter the aggregate.
        var dryByHour = new bool[span];
        for (int i = 0; i < span; i++)
        {
            var h = daytimeStartUtc + i;
            if (!hourlyMm.TryGetValue(h, out var mm)) return null;
            dryByHour[i] = mm < WetThresholdMm;
        }

        var valid = new HashSet<int>();
        var nStarts = span - windowHours + 1;
        for (int i = 0; i < nStarts; i++)
        {
            bool allDry = true;
            for (int j = 0; j < windowHours; j++)
            {
                if (!dryByHour[i + j]) { allDry = false; break; }
            }
            if (allDry) valid.Add(daytimeStartUtc + i);
        }
        return valid;
    }
}
