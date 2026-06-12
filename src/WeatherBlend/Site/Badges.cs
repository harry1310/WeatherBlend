using System.Globalization;

namespace WeatherBlend.Site;

/// <summary>
/// Severity of an overview-tile warning badge. Unified rule across every
/// badge family (Harry's design 2026-06-12): ONE fired trigger = amber,
/// TWO or more = red. <see cref="None"/> = no badge rendered.
/// </summary>
public enum BadgeSeverity
{
    None,
    Amber,
    Red,
}

/// <summary>
/// Shared severity arithmetic for the overview-tile badges. Pure static —
/// the render layer maps the result onto the <c>badge-amber</c> /
/// <c>badge-red</c> CSS classes.
/// </summary>
public static class BadgeSeverityRule
{
    /// <summary>One fired trigger = amber, two-or-more = red, zero = none.</summary>
    public static BadgeSeverity FromFiredCount(int firedCount) => firedCount switch
    {
        <= 0 => BadgeSeverity.None,
        1 => BadgeSeverity.Amber,
        _ => BadgeSeverity.Red,
    };
}

/// <summary>
/// Low-cloud / mist badge severity (overview tiles). The two triggers are
/// the existing independent signals: "mist" (≥3/6 vis-publishing NWPs
/// forecasting sub-1km visibility) and "cloud base" (≥6/~11 NWPs
/// forecasting T−Td &lt; 1.5°C). The caller computes the booleans from
/// <c>LowCloudSignal</c> against the firing thresholds; this just applies
/// the unified one-amber / two-red rule.
/// </summary>
public static class LowCloudBadge
{
    public static BadgeSeverity Evaluate(bool mistFired, bool cloudBaseFired) =>
        BadgeSeverityRule.FromFiredCount((mistFired ? 1 : 0) + (cloudBaseFired ? 1 : 0));
}

/// <summary>
/// Render-layer view of the per-location <c>marine.seaStateBadge</c> config
/// block (see <c>SeaStateBadgeConfig</c> in AppConfig.cs and the sennen
/// entry in config.yaml). Carried on <see cref="SitePages.LocationDescriptor"/>
/// so the Site layer doesn't import AppConfig. Null descriptor field =
/// location has no sea-state badge.
/// </summary>
/// <param name="TideHighMsl">Tide trigger threshold — TideHeightMsl (m vs MSL) at/above which "high tide" fires.</param>
/// <param name="RunUpProxy">Wave trigger threshold — √Hs(m) × swell period(s) at/above which "big waves" fires.</param>
/// <param name="WindMinMph">Onshore-wind trigger minimum speed (mph).</param>
/// <param name="WindSectorFromDeg">Onshore sector start, "coming from" degrees (sector runs clockwise from→to, may wrap 360).</param>
/// <param name="WindSectorToDeg">Onshore sector end, "coming from" degrees.</param>
public sealed record SeaStateBadgeSpec(
    double TideHighMsl,
    double RunUpProxy,
    double WindMinMph,
    double WindSectorFromDeg,
    double WindSectorToDeg);

/// <summary>
/// Result of one hour's sea-state badge evaluation: the severity plus a
/// plain-text description per FIRED trigger with its live values (rendered
/// into the badge's pop-out &lt;li&gt; lines, HTML-escaped by the caller).
/// </summary>
public sealed record SeaStateBadgeResult(
    BadgeSeverity Severity,
    IReadOnlyList<string> FiredTriggers)
{
    public static readonly SeaStateBadgeResult None =
        new(BadgeSeverity.None, Array.Empty<string>());
}

/// <summary>
/// Sea-state badge evaluator (Sennen overview tiles — any marine location
/// with a <c>seaStateBadge</c> config block). Three independent triggers
/// per hour:
///
///   1. HIGH TIDE — TideHeightMsl ≥ tideHighMsl.
///   2. BIG WAVES — run-up proxy √Hs × swell period ≥ runUpProxy
///      (Hunt-style scaling: run-up grows with wave energy flux, and
///      period matters as much as height for how far water runs up a
///      cliff apron).
///   3. ONSHORE WIND — speed ≥ windMinMph AND "from" direction inside
///      [sectorFromDeg, sectorToDeg] (clockwise, may wrap 360).
///
/// A trigger whose inputs are missing (null) is NOT EVALUABLE and is
/// simply skipped — missing data never counts as fired. The badge then
/// works on whatever triggers remain (e.g. tide + waves alone while the
/// Sennen wind_mvn direction tree hasn't landed). All threshold
/// comparisons are inclusive (exactly-at fires).
///
/// Pure static for testability — no I/O, no SiteInputs.
/// </summary>
public static class SeaStateBadge
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

    /// <param name="tideHeightMsl">Tide height, m vs MSL (wave-predictions pass-through). Null = tide trigger skipped.</param>
    /// <param name="waveHeightM">Blended significant wave height Hs, m (wave-predictions BlendValue). Null = wave trigger skipped.</param>
    /// <param name="swellPeriodS">Swell period, s (pass-through on the same wave row). Null = wave trigger skipped.</param>
    /// <param name="windMph">10 m wind speed, mph (caller converts the element blender's m/s ×2.23694). Null = wind trigger skipped.</param>
    /// <param name="windDirDeg">Wind "coming from" direction, degrees (wind_mvn). Null = wind trigger skipped — direction rows can lag the speed tree.</param>
    /// <param name="spec">Per-location thresholds from config.</param>
    public static SeaStateBadgeResult Evaluate(
        double? tideHeightMsl,
        double? waveHeightM,
        double? swellPeriodS,
        double? windMph,
        double? windDirDeg,
        SeaStateBadgeSpec spec)
    {
        var fired = new List<string>();

        // 1. High tide.
        if (tideHeightMsl is double tide && tide >= spec.TideHighMsl)
        {
            fired.Add(string.Create(Ci,
                $"tide {tide:+0.0;-0.0;+0.0} m (≥ {spec.TideHighMsl:+0.00;-0.00;0.00})"));
        }

        // 2. Big waves (run-up proxy). Needs BOTH Hs and swell period —
        // either missing (or Hs negative → NaN proxy) skips the trigger.
        if (waveHeightM is double hs && swellPeriodS is double tp)
        {
            var proxy = Math.Sqrt(hs) * tp;
            if (proxy >= spec.RunUpProxy)
            {
                fired.Add(string.Create(Ci,
                    $"run-up proxy {proxy:0.0} = √{hs:0.0} m × {tp:0.0} s (≥ {spec.RunUpProxy:0.##})"));
            }
        }

        // 3. Onshore wind. Needs BOTH speed and direction — when the
        // direction tree hasn't landed for this hour the trigger is not
        // evaluable and the badge runs on tide + waves alone.
        if (windMph is double mph && windDirDeg is double dir
            && mph >= spec.WindMinMph
            && DirectionInSector(dir, spec.WindSectorFromDeg, spec.WindSectorToDeg))
        {
            fired.Add(string.Create(Ci,
                $"wind {mph:0} mph from {Normalize(dir):0}° (≥ {spec.WindMinMph:0.##} from {Normalize(spec.WindSectorFromDeg):0}–{Normalize(spec.WindSectorToDeg):0}°)"));
        }

        return fired.Count == 0
            ? SeaStateBadgeResult.None
            : new SeaStateBadgeResult(BadgeSeverityRule.FromFiredCount(fired.Count), fired);
    }

    /// <summary>
    /// True iff <paramref name="directionDeg"/> lies inside the clockwise
    /// sector from <paramref name="fromDeg"/> to <paramref name="toDeg"/>,
    /// inclusive at both ends. Handles sectors that wrap through 360
    /// (e.g. from 300° to 60° contains 350° and 20° but not 180°). All
    /// inputs are normalised into [0, 360) first, so out-of-range and
    /// negative degrees behave.
    /// </summary>
    internal static bool DirectionInSector(double directionDeg, double fromDeg, double toDeg)
    {
        var d = Normalize(directionDeg);
        var from = Normalize(fromDeg);
        var to = Normalize(toDeg);
        return from <= to
            ? d >= from && d <= to
            : d >= from || d <= to;   // sector wraps through 360
    }

    private static double Normalize(double degrees) => ((degrees % 360.0) + 360.0) % 360.0;
}
