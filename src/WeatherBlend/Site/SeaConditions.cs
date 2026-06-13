using System.Globalization;

namespace WeatherBlend.Site;

/// <summary>
/// Sea-state QUALITY factor for the climbing-conditions index at a sea cliff
/// (Sennen, 2026-06-13). Distinct from <see cref="SeaStateBadge"/> — that emits
/// discrete tide/run-up/onshore-wind TRIGGERS for the tile badge; this folds the
/// same physics into ONE graded 0–1 quality score the conditions index can blend
/// (lower = worse), naming the dominant cause.
///
/// Three ways the sea spoils a session, worst-one-wins (the climber cares about
/// the limiting factor):
///   * HIGH TIDE — the base gets covered / cut off.
///   * RUN-UP — big groundswell washes the base (√Hs × swell period).
///   * ONSHORE SPRAY — breaking-wave spray driven onto the rock by an onshore
///     wind soaks the holds even in zero rain. This is the distinctive Atlantic
///     sea-cliff factor (Harry 2026-06-13): it needs the wind blowing FROM the
///     sea (the onshore sector), blowing HARD, and WAVES to break — so it scales
///     as onshore × wind-strength × wave-energy.
///
/// Tide/run-up ramp around the location's draft <see cref="SeaStateBadgeSpec"/>
/// thresholds (≈0 a little below, 0.5 at the threshold, 1.0 well above). The
/// spray scales below are DRAFT (like the badge thresholds) — tune against real
/// sessions. Pure + static, unit-testable.
/// </summary>
public static class SeaConditions
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

    // ---- Spray scales (DRAFT — Harry to tune) ----
    /// <summary>Onshore wind below this (mph) lofts negligible spray.</summary>
    public const double SprayWindOnsetMph = 10.0;
    /// <summary>Onshore wind at/above this (mph) = full spray drive.</summary>
    public const double SprayWindHeavyMph = 28.0;
    /// <summary>Below this Hs (m) there's too little breaking wave to spray.</summary>
    public const double SprayHsMinM = 0.3;
    /// <summary>Hs (m) at/above which wave energy fully enables spray.</summary>
    public const double SprayHsFullM = 1.8;

    private static double Clamp01(double x) => Math.Clamp(x, 0.0, 1.0);

    /// <summary>Threshold ramp: 0 at 0.7·thr, ~0.5 at thr, 1 at 1.3·thr.
    /// Matches "amber at the badge threshold, red well above".</summary>
    private static double Ramp(double x, double thr)
        => thr <= 0 ? 0.0 : Clamp01((x - 0.7 * thr) / (0.6 * thr));

    /// <summary>
    /// Sea-state quality factor for one hour, or null when NO sea inputs are
    /// available (so the conditions index simply omits the factor rather than
    /// guessing). Each sub-factor is skipped individually when its own inputs
    /// are missing (e.g. no wind_mvn direction row yet → no spray term, tide +
    /// run-up still score).
    /// </summary>
    public static ClimbingConditions.Factor? Evaluate(
        double? tideHeightMsl, double? waveHeightM, double? swellPeriodS,
        double? windMph, double? windDirDeg, SeaStateBadgeSpec spec)
    {
        var any = false;
        double tideBad = 0, runUpBad = 0, sprayBad = 0;

        if (tideHeightMsl is double tide)
        {
            any = true;
            tideBad = Ramp(tide, spec.TideHighMsl);
        }

        double hsForSpray = 0;
        if (waveHeightM is double hs && swellPeriodS is double tp)
        {
            any = true;
            hsForSpray = hs;
            runUpBad = Ramp(Math.Sqrt(Math.Max(hs, 0)) * tp, spec.RunUpProxy);
        }

        if (windMph is double mph && windDirDeg is double dir && waveHeightM is double hs2)
        {
            any = true;
            var onshore = SeaStateBadge.DirectionInSector(dir, spec.WindSectorFromDeg, spec.WindSectorToDeg) ? 1.0 : 0.0;
            var windStrength = Clamp01((mph - SprayWindOnsetMph) / (SprayWindHeavyMph - SprayWindOnsetMph));
            var waveAmt = Clamp01((hs2 - SprayHsMinM) / (SprayHsFullM - SprayHsMinM));
            sprayBad = onshore * windStrength * waveAmt;
        }

        if (!any) return null;

        // Worst sub-factor wins; name it so the verdict's reason is honest.
        var badness = Math.Max(tideBad, Math.Max(runUpBad, sprayBad));
        string detail;
        if (badness < 0.05)
            detail = "calm";
        else if (sprayBad >= tideBad && sprayBad >= runUpBad)
            detail = string.Create(Ci, $"onshore spray ({windMph:0} mph onshore, {hsForSpray:0.0} m)");
        else if (runUpBad >= tideBad)
            detail = string.Create(Ci, $"big swell run-up ({waveHeightM:0.0} m)");
        else
            detail = string.Create(Ci, $"high tide ({tideHeightMsl:+0.0;-0.0;0.0} m)");

        return new ClimbingConditions.Factor("Sea", Clamp01(1.0 - badness), detail);
    }
}
