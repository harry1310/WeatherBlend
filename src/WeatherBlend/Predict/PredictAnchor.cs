namespace WeatherBlend.Predict;

/// <summary>
/// Computes the "anchor" timestamp that every predict command hangs its lead offsets off.
/// Live runs anchor on the current UTC hour (truncated to top-of-hour). --for-date backfills
/// anchor on the given date at <see cref="BackfillAnchor"/> (08:00 UTC — late enough that
/// the major NWP cycles have typically landed).
/// </summary>
public static class PredictAnchor
{
    public static readonly TimeSpan BackfillAnchor = new(8, 0, 0);

    public static DateTime Compute(DateTime utcNow, DateOnly? forDate)
    {
        if (forDate.HasValue)
            return forDate.Value.ToDateTime(TimeOnly.FromTimeSpan(BackfillAnchor), DateTimeKind.Utc);

        return new DateTime(utcNow.Year, utcNow.Month, utcNow.Day, utcNow.Hour, 0, 0, DateTimeKind.Utc);
    }
}
