namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Maps a local-time daytime window (e.g. "09:00–18:00 Europe/London") to the
/// per-target-day UTC hour range that should be scanned for a dry block.
///
/// DST is the whole reason this exists: Europe/London is UTC+1 in summer, +0
/// in winter. A naive "9–18 UTC" config would slide an hour against the
/// user's wall-clock walking window twice a year. Storing local hours +
/// IANA tz id and converting per target date keeps the meaning of "daytime"
/// stable across the whole training span.
/// </summary>
public sealed class DaytimeWindow
{
    public int StartLocalHour { get; }
    public int EndLocalHour { get; }
    public TimeZoneInfo Tz { get; }

    /// <summary>Number of hours in the local-time window (e.g. 9 for 09–18).</summary>
    public int DurationHours => EndLocalHour - StartLocalHour;

    public DaytimeWindow(int startLocalHour, int endLocalHour, TimeZoneInfo tz)
    {
        if (startLocalHour < 0 || startLocalHour > 23)
            throw new ArgumentOutOfRangeException(nameof(startLocalHour), "Must be in [0, 23].");
        if (endLocalHour <= startLocalHour || endLocalHour > 24)
            throw new ArgumentOutOfRangeException(nameof(endLocalHour), "Must be in (start, 24].");
        StartLocalHour = startLocalHour;
        EndLocalHour = endLocalHour;
        Tz = tz ?? throw new ArgumentNullException(nameof(tz));
    }

    /// <summary>
    /// Convenience ctor — accepts the IANA tz id directly. <c>"Europe/London"</c> works
    /// on .NET 8+ on every host platform (Windows uses ICU since the BCL ICU shim).
    /// </summary>
    public DaytimeWindow(int startLocalHour, int endLocalHour, string ianaTzId)
        : this(startLocalHour, endLocalHour, TimeZoneInfo.FindSystemTimeZoneById(ianaTzId))
    {
    }

    /// <summary>
    /// For a target UTC date, return the inclusive UTC hour range
    /// <c>[startUtcHour, endUtcHourExclusive)</c> that the daytime window
    /// covers within that single UTC day.
    ///
    /// Edge case: if the local window crosses the UTC day boundary (e.g.
    /// 23:00–04:00 local in a far-east tz), the part outside the target UTC
    /// day is dropped — we only scan within one calendar day. For the
    /// Europe/London / 9–18 use case this never happens (window fully fits
    /// in the same UTC day in both BST and GMT).
    /// </summary>
    public (int StartUtcHour, int EndUtcHourExclusive) UtcHourRangeFor(DateOnly targetDateUtc)
    {
        // Anchor at the wall-clock start hour for the day in local time, then
        // convert to UTC. Using Unspecified Kind so ConvertTimeToUtc treats it
        // as local-to-Tz rather than as already-UTC.
        var startLocal = new DateTime(targetDateUtc.Year, targetDateUtc.Month, targetDateUtc.Day,
            StartLocalHour, 0, 0, DateTimeKind.Unspecified);
        var endLocal = new DateTime(targetDateUtc.Year, targetDateUtc.Month, targetDateUtc.Day,
            EndLocalHour == 24 ? 23 : EndLocalHour, 0, 0, DateTimeKind.Unspecified);

        var startUtc = TimeZoneInfo.ConvertTimeToUtc(startLocal, Tz);
        var endUtc = TimeZoneInfo.ConvertTimeToUtc(endLocal, Tz);

        // Bound to the target UTC day. If the converted start lies in the
        // previous UTC day (only possible for far-east tzs), clip up to 0;
        // if the end overflows into next UTC day, clip to 24.
        var dayStart = targetDateUtc.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
        var dayEnd = targetDateUtc.AddDays(1).ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);

        var startHour = startUtc < dayStart ? 0
            : startUtc >= dayEnd ? 24
            : startUtc.Hour;
        var endHour = endUtc < dayStart ? 0
            : endUtc >= dayEnd ? 24
            : (EndLocalHour == 24 ? endUtc.Hour + 1 : endUtc.Hour);

        return (startHour, endHour);
    }
}
