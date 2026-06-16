using WeatherBlend.Predict.Surface;

namespace WeatherBlend.Site;

/// <summary>
/// Climbing-conditions index (idea #1, 2026-06-13). Fuses the per-hour outputs
/// the site already has — rain chance, rock friction (from the rock-surface
/// model), wind, air temperature, daylight — into ONE per-hour verdict for the
/// overview tiles, with the limiting factor always named so the reader trusts
/// it. Sea-state is a Sennen-only factor added later (Bonehill-first per Harry).
///
/// This is a HEURISTIC, not a trained model: there's no ground truth for "good
/// conditions" (it's a judgement call), so the curves below are hand-tuned to
/// Harry's domain knowledge (2026-06-13) and live as named constants for easy
/// tweaking — the same philosophy as the sea-state badge + greasy threshold.
/// Pure + static so the curve shapes are unit-testable without any I/O.
///
/// Two layers (climbers don't average — one dealbreaker kills the session):
///   * GATES → <see cref="ConditionsTier.Off"/>: no daylight; rain likely now;
///     rock wet from condensation.
///   * QUALITY (for surviving hours): a weakest-link blend of dryness, friction,
///     wind comfort, air-temp comfort. The blend is sqrt(geomean × min) so one
///     poor factor drags the verdict down hard (matching "12 mph on an exposed
///     tor is off-putting no matter how nice everything else is") without a
///     single mediocre factor zeroing it outright.
/// </summary>
public static class ClimbingConditions
{
    // ---- Calibration constants (Harry 2026-06-13) -----------------------

    /// <summary>Rock friction peaks at this temperature: "as cold as possible
    /// down to 5°C", below which it degrades sharply (numb hands / verglas).</summary>
    public const double RockFrictionPeakC = 5.0;
    /// <summary>°C above the peak at which rock friction hits zero (sun-warm rock = greasy/poor grip).</summary>
    public const double RockFrictionWarmSpanC = 15.0;
    /// <summary>Per-°C friction loss below the peak — steep ("degrades pretty sharply").</summary>
    public const double RockFrictionColdSlope = 0.17;

    /// <summary>Air-temperature comfort sweet spot (inclusive band).</summary>
    public const double AirIdealLoC = 8.0, AirIdealHiC = 10.0;
    /// <summary>Below this, air is "bad" (sharp penalty); above <see cref="AirTooHotC"/> it's too hot.</summary>
    public const double AirColdC = 5.0, AirTooHotC = 20.0;

    /// <summary>Wind comfort band (mph): a breeze beats dead calm, but an
    /// exposed tor turns "pretty shite" past ~10 mph.</summary>
    public const double WindIdealLoMph = 5.0, WindIdealHiMph = 7.0, WindHarshMph = 10.0;
    /// <summary>Wind-comfort floor at dead calm (no drying breeze, but fine).</summary>
    public const double WindCalmScore = 0.6;

    /// <summary>At or above this hourly P(wet), the rain gate fires (Off).</summary>
    public const double RainGateProb = 0.5;

    /// <summary>Solar elevation (deg) below which it's too dark to climb (gate).</summary>
    public const double DaylightMinElevationDeg = 0.0;

    /// <summary>Friction multiplier when the rock is flagged "potentially greasy"
    /// (within the greasy margin of dew point but not yet condensing).</summary>
    public const double GreasyFrictionPenalty = 0.4;

    // Tier thresholds on the final 0–1 score.
    public const double PrimeAt = 0.70, GoodAt = 0.50, MarginalAt = 0.30;

    private static double Clamp01(double x) => Math.Clamp(x, 0.0, 1.0);

    /// <summary>Rock-friction sub-score: peak at <see cref="RockFrictionPeakC"/>,
    /// gradual decline as it warms, sharp drop as it cools below the peak.</summary>
    public static double RockFriction(double rockTempC) => rockTempC >= RockFrictionPeakC
        ? Clamp01(1.0 - (rockTempC - RockFrictionPeakC) / RockFrictionWarmSpanC)
        : Clamp01(1.0 - (RockFrictionPeakC - rockTempC) * RockFrictionColdSlope);

    /// <summary>Air-temperature comfort: hump on <see cref="AirIdealLoC"/>..
    /// <see cref="AirIdealHiC"/>, sharp below <see cref="AirColdC"/>, falling
    /// off above to "too hot" past <see cref="AirTooHotC"/>.</summary>
    public static double AirComfort(double airTempC)
    {
        if (airTempC < AirColdC) return Clamp01(0.4 - (AirColdC - airTempC) * 0.12);
        if (airTempC < AirIdealLoC) return 0.4 + (airTempC - AirColdC) / (AirIdealLoC - AirColdC) * 0.6;
        if (airTempC <= AirIdealHiC) return 1.0;
        if (airTempC <= AirTooHotC) return Clamp01(1.0 - (airTempC - AirIdealHiC) / (AirTooHotC - AirIdealHiC) * 0.85);
        return Clamp01(0.15 - (airTempC - AirTooHotC) * 0.03);
    }

    /// <summary>Wind comfort (mph): rises from a calm-day floor to a 5–7 mph
    /// peak, then falls off — "pretty shite" past ~10 mph on exposed ground.</summary>
    public static double WindComfort(double windMph)
    {
        if (windMph <= WindIdealLoMph) return WindCalmScore + windMph / WindIdealLoMph * (1.0 - WindCalmScore);
        if (windMph <= WindIdealHiMph) return 1.0;
        if (windMph <= WindHarshMph) return Clamp01(1.0 - (windMph - WindIdealHiMph) / (WindHarshMph - WindIdealHiMph) * 0.6);
        return Clamp01(0.4 - (windMph - WindHarshMph) * 0.06);
    }

    /// <summary>One contributing factor's score + a human detail string.</summary>
    public readonly record struct Factor(string Name, double Score, string Detail);

    public readonly record struct Result(
        ConditionsTier Tier,
        double Score,
        string Reason,
        IReadOnlyList<Factor> Factors)
    {
        public string TierLabel => ClimbingConditions.TierLabel(Tier);
        public string TierColor => ClimbingConditions.TierColor(Tier);
    }

    /// <summary>
    /// Score one hour. <paramref name="rock"/> is the rock-surface row for this
    /// hour when available (friction + greasy gate come from it); when absent
    /// (e.g. lead &lt; 24h, before the rock model's window) friction falls back
    /// to the air temperature as a rough proxy and the greasy gate is skipped.
    /// <paramref name="pWet"/> / <paramref name="windMph"/> null = that factor
    /// is omitted rather than guessed.
    /// </summary>
    public static Result Evaluate(
        DateTime validUtc, double latitude, double longitude,
        double airTempC, double? pWet, double? windMph,
        SitePages.RockSurfaceForecastPoint? rock,
        Factor? sea = null)
    {
        // ---- Gates (any one → Off) ----
        var (elevDeg, _) = SolarGeometry.SolarPosition(validUtc, latitude, longitude);
        if (elevDeg <= DaylightMinElevationDeg)
            return Gate(ConditionsTier.Off, "Dark — sun below the horizon");
        if (pWet is double pw && pw >= RainGateProb)
            return Gate(ConditionsTier.Off, $"Rain likely ({pw * 100:0}%)");
        if (rock is { GreasinessStatus: RockSurfacePhysics.StatusCondensation } rkc)
            // Spell out the why (surface ≤ dew point) — the gate returns no
            // factor table, so without this the verdict was a bare "Rock wet".
            return Gate(ConditionsTier.Off,
                $"Rock wet — condensation (surface {rkc.RockSurfaceTempC:0.0}°C ≤ dew point {rkc.DewPointC:0.0}°C, margin {rkc.CondensationMarginC:+0.0;-0.0;0.0}°C)");

        // ---- Quality factors ----
        var factors = new List<Factor>(4);

        // Friction: from rock temp when present (the rock-surface model's
        // payoff), else an air-temp proxy. A "potentially greasy" flag knocks
        // friction down without gating outright.
        double frictionScore;
        string frictionDetail;
        if (rock is { } rk)
        {
            frictionScore = RockFriction(rk.RockSurfaceTempC);
            frictionDetail = $"rock {rk.RockSurfaceTempC:0.0}°C";
            if (rk.GreasinessStatus == RockSurfacePhysics.StatusPotentiallyGreasy)
            {
                frictionScore *= GreasyFrictionPenalty;
                frictionDetail += ", greasy";
            }
        }
        else
        {
            frictionScore = RockFriction(airTempC);
            frictionDetail = $"~{airTempC:0.0}°C (from air)";
        }
        factors.Add(new Factor("Friction", frictionScore, frictionDetail));

        factors.Add(new Factor("Air temp", AirComfort(airTempC), $"{airTempC:0.0}°C"));

        if (windMph is double mph)
            factors.Add(new Factor("Wind", WindComfort(mph), $"{mph:0} mph"));

        if (pWet is double pwet)
            factors.Add(new Factor("Dry", Clamp01(1.0 - pwet), $"P(wet) {pwet * 100:0}%"));

        // Sea state (sea-cliff locations) — tide / run-up / onshore spray folded
        // into one factor by SeaConditions; null for inland crags.
        if (sea is { } seaFactor)
            factors.Add(seaFactor);

        // Weakest-link blend: sqrt(geomean × min). geomean alone is too
        // forgiving of a single off-putting factor; pure min ignores that
        // several mediocre factors compound. The blend does both.
        double logSum = 0;
        double min = 1.0;
        Factor worst = factors[0];
        foreach (var f in factors)
        {
            var s = Math.Max(f.Score, 1e-6);
            logSum += Math.Log(s);
            if (f.Score < min) { min = f.Score; worst = f; }
        }
        var geomean = Math.Exp(logSum / factors.Count);
        var score = Math.Sqrt(geomean * min);

        var tier = score >= PrimeAt ? ConditionsTier.Prime
                 : score >= GoodAt ? ConditionsTier.Good
                 : score >= MarginalAt ? ConditionsTier.Marginal
                 : ConditionsTier.Poor;

        // Reason = the limiting factor for anything short of Prime; for Prime
        // we celebrate the best factor instead.
        var reason = tier == ConditionsTier.Prime
            ? $"{worst.Name.ToLowerInvariant()} fine ({worst.Detail})"
            : $"limited by {worst.Name.ToLowerInvariant()} ({worst.Detail})";

        return new Result(tier, score, reason, factors);
    }

    private static Result Gate(ConditionsTier tier, string reason)
        => new(tier, 0.0, reason, Array.Empty<Factor>());

    public static string TierLabel(ConditionsTier t) => t switch
    {
        ConditionsTier.Prime => "Prime",
        ConditionsTier.Good => "Good",
        ConditionsTier.Marginal => "Marginal",
        ConditionsTier.Poor => "Poor",
        _ => "Off",
    };

    /// <summary>Tier → hex colour (greens for go, amber/red for caution,
    /// slate for Off). Aligned with the badge palette used elsewhere.</summary>
    public static string TierColor(ConditionsTier t) => t switch
    {
        ConditionsTier.Prime => "#2e7d32",     // green
        ConditionsTier.Good => "#7cb342",      // light green
        ConditionsTier.Marginal => "#f9a825",  // amber
        ConditionsTier.Poor => "#c62828",      // red
        _ => "#607d8b",                        // slate (Off)
    };
}

/// <summary>Climbing-conditions verdict, best → worst, with Off for gated hours
/// (dark / raining / rock wet).</summary>
public enum ConditionsTier
{
    Off,
    Poor,
    Marginal,
    Good,
    Prime,
}
