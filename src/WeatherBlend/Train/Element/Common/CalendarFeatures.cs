namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Cyclical hour-of-day + day-of-year encodings, identical to the temperature
/// 2b lean blender so feature semantics stay consistent across blenders.
/// </summary>
public readonly record struct CalendarFeatures(
    float HourSin,
    float HourCos,
    float DoySin,
    float DoyCos)
{
    public static CalendarFeatures From(DateTime validTimeUtc)
    {
        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);

        var hourAngle = 2.0 * Math.PI * v.Hour / 24.0;
        var doyAngle  = 2.0 * Math.PI * (v.DayOfYear - 1) / 365.0;
        return new CalendarFeatures(
            HourSin: (float)Math.Sin(hourAngle),
            HourCos: (float)Math.Cos(hourAngle),
            DoySin:  (float)Math.Sin(doyAngle),
            DoyCos:  (float)Math.Cos(doyAngle));
    }
}
