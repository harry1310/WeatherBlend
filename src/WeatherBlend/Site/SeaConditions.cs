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

    /// <summary>
    /// Onshore-spray QUALITY factor for one hour, or null when the spray inputs
    /// (onshore wind + waves) aren't available — the conditions index then omits
    /// the factor rather than guessing. Spray scales as onshore × wind-strength ×
    /// wave-energy.
    ///
    /// De-dup 2026-06-16 (Harry): tide + run-up are ACCESS hazards, not climbing
    /// QUALITY, so they live SOLELY in the Alerts badge (<see cref="SeaStateBadge"/>)
    /// now — this factor carries only the onshore spray that actually wets the
    /// holds. The index scores quality; the badge warns of access.
    /// </summary>
    public static ClimbingConditions.Factor? Evaluate(
        double? waveHeightM, double? windMph, double? windDirDeg, SeaStateBadgeSpec spec)
    {
        // Needs onshore wind (speed + direction) AND waves to break — any
        // missing and there's no spray term to score, so omit the factor.
        if (windMph is not double mph || windDirDeg is not double dir || waveHeightM is not double hs)
            return null;

        var onshore = SeaStateBadge.DirectionInSector(dir, spec.WindSectorFromDeg, spec.WindSectorToDeg) ? 1.0 : 0.0;
        var windStrength = Clamp01((mph - SprayWindOnsetMph) / (SprayWindHeavyMph - SprayWindOnsetMph));
        var waveAmt = Clamp01((hs - SprayHsMinM) / (SprayHsFullM - SprayHsMinM));
        var sprayBad = onshore * windStrength * waveAmt;

        var detail = sprayBad < 0.05
            ? "no onshore spray"
            : string.Create(Ci, $"onshore spray ({mph:0} mph onshore, {hs:0.0} m)");

        return new ClimbingConditions.Factor("Spray", Clamp01(1.0 - sprayBad), detail);
    }
}
