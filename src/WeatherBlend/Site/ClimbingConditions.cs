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

    /// <summary>Rock friction peaks across this band: 5–10°C is the sweet spot
    /// ("as cold as possible, but a numb-hands floor"), below which it degrades
    /// sharply (numb hands / verglas) and above which it eases off (greasy).
    /// Widened from a single 5°C point to a 5–10°C plateau (Harry 2026-06-17).</summary>
    public const double RockFrictionPeakLoC = 5.0;
    /// <summary>Top of the peak-friction band; warm decline starts above here.</summary>
    public const double RockFrictionPeakHiC = 10.0;
    /// <summary>°C above the peak band over which warm rock decays to
    /// <see cref="RockFrictionWarmFloorScore"/> (rather than to zero). 15°C →
    /// the floor is reached at 25°C rock temp (Harry 2026-06-17).</summary>
    public const double RockFrictionWarmSpanC = 15.0;
    /// <summary>Warm-rock friction floor: greasy sun-warm rock is poor grip but
    /// still climbable, so it bottoms out HERE rather than at zero. A zero factor
    /// would annihilate the weakest-link blend (the <c>min</c> term in
    /// <see cref="Evaluate"/>) — same fix applied to strong wind
    /// (<see cref="WindStrongFloorScore"/>). Harry 2026-06-17.</summary>
    public const double RockFrictionWarmFloorScore = 0.25;
    /// <summary>Per-°C friction loss below the peak band — steep ("degrades pretty sharply").</summary>
    public const double RockFrictionColdSlope = 0.17;

    /// <summary>Air-temperature comfort sweet spot (inclusive band).</summary>
    public const double AirIdealLoC = 8.0, AirIdealHiC = 10.0;
    /// <summary>Below this, air is "bad" (sharp penalty); above <see cref="AirTooHotC"/> it's too hot.</summary>
    public const double AirColdC = 5.0, AirTooHotC = 20.0;

    /// <summary>Wind comfort band (mph): a breeze beats dead calm, but an
    /// exposed tor turns "pretty shite" past ~10 mph, decaying toward a non-zero
    /// floor past that (see <see cref="WindStrongFloorScore"/>).</summary>
    public const double WindIdealLoMph = 5.0, WindIdealHiMph = 7.0, WindHarshMph = 10.0;
    /// <summary>Wind-comfort floor at dead calm (no drying breeze, but fine).</summary>
    public const double WindCalmScore = 0.6;
    /// <summary>Wind-comfort value at the harsh point (10 mph) — the anchor the
    /// strong-wind decay starts from.</summary>
    public const double WindHarshScore = 0.4;
    /// <summary>Per-mph wind-comfort loss above the harsh point. 0.02/mph carries
    /// 0.4 at 10 mph down to <see cref="WindStrongFloorScore"/> at ~25 mph.</summary>
    public const double WindHarshSlope = 0.02;
    /// <summary>Strong-wind floor: past ~25 mph the curve bottoms out HERE rather
    /// than hitting zero (Harry 2026-06-17). A single 0 factor would zero the
    /// whole weakest-link blend (the <c>min</c> term in <see cref="Evaluate"/>),
    /// so a gale would otherwise annihilate the verdict no matter how good
    /// everything else was — and 18 mph scored identically to a 40 mph storm.
    /// A non-zero floor lets strong wind drag the verdict down hard while still
    /// distinguishing "very windy" from "impossible".</summary>
    public const double WindStrongFloorScore = 0.1;

    /// <summary>At or above this hourly P(wet), the rain gate fires (Off).</summary>
    public const double RainGateProb = 0.5;

    /// <summary>Rain-derived surface-water film (mm) at/above which the rock
    /// counts as wet-from-rain and the verdict hard-gates Off (with a dry-by
    /// ETA). The index's own copy of the rock model's <c>WetThresholdMm</c> knob
    /// — they share a default; this is the gate-decision authority. Only consulted
    /// when the location has the drying model enabled (Harry 2026-06-16).</summary>
    public const double RainWetThresholdMm = 0.05;

    /// <summary>Solar elevation (deg) below which it's too dark to climb (gate).</summary>
    public const double DaylightMinElevationDeg = 0.0;

    /// <summary>Friction floor when the rock is condensing (margin ≤ 0): poor grip,
    /// but deliberately NOT a hard Off-gate (Harry 2026-06-16) — the rock-temp calc
    /// is still being field-validated, so an uncertain "wet" call drags the verdict
    /// to Poor rather than nuking it to Off. Also the LOW anchor of the continuous
    /// greasiness ramp (<see cref="GreasyFrictionMultiplier"/>); the old flat
    /// "greasy ×0.4" multiplier is gone — greasiness now scales smoothly with the
    /// rock-vs-dew margin (Harry 2026-06-20).</summary>
    public const double WetFrictionPenalty = 0.15;

    /// <summary>Fallback greasy margin (°C) when a caller doesn't supply the
    /// location's own value — matches the global RockSurface default (Bonehill 3.0).
    /// Sennen passes a tighter 1.5 (maritime dew point sits close to the rock). Sets
    /// the WIDTH of the greasiness ramp: friction recovers from the wet floor to no
    /// penalty as the rock-vs-dew margin climbs from 0 to this.</summary>
    public const double DefaultGreasyMarginC = 3.0;

    // Tier thresholds on the final 0–1 score.
    public const double PrimeAt = 0.70, GoodAt = 0.50, MarginalAt = 0.30;

    private static double Clamp01(double x) => Math.Clamp(x, 0.0, 1.0);

    /// <summary>Rock-friction sub-score: a flat peak across the
    /// <see cref="RockFrictionPeakLoC"/>–<see cref="RockFrictionPeakHiC"/> band,
    /// a gentle warm decline above it down to a non-zero floor
    /// (<see cref="RockFrictionWarmFloorScore"/>), and a sharp cold drop below it
    /// (numb hands / verglas).</summary>
    public static double RockFriction(double rockTempC)
    {
        if (rockTempC < RockFrictionPeakLoC)
            return Clamp01(1.0 - (RockFrictionPeakLoC - rockTempC) * RockFrictionColdSlope);
        if (rockTempC <= RockFrictionPeakHiC)
            return 1.0;
        // Warm side: 1.0 at the band top decaying linearly to the floor at
        // (PeakHi + WarmSpan)°C, then holding the floor — warm rock is poor
        // grip but never a hard zero.
        var warm = Clamp01(1.0 - (rockTempC - RockFrictionPeakHiC) / RockFrictionWarmSpanC);
        return RockFrictionWarmFloorScore + (1.0 - RockFrictionWarmFloorScore) * warm;
    }

    /// <summary>Continuous greasiness friction multiplier from the rock-vs-dew
    /// margin <paramref name="marginC"/> (= rock temp − dew point). Replaces the
    /// old discrete tier multiplier — a hard ×0.4 greasy / ×0.15 wet / ×1.0 dry
    /// step that snapped the verdict a whole tier when the margin crossed the
    /// greasy threshold by a fraction of a degree (Sennen 08→09Z 2026-06-20: rock
    /// warming pushed the margin 1.44→1.83 across the 1.5°C cut, vaulting
    /// Marginal→Prime). Smoothstep from the wet floor
    /// (<see cref="WetFrictionPenalty"/>) at margin ≤ 0 up to no penalty (1.0) at
    /// margin ≥ <paramref name="greasyMarginC"/>, so friction recovers smoothly as
    /// the rock warms clear of the dew point.</summary>
    public static double GreasyFrictionMultiplier(double marginC, double greasyMarginC)
    {
        if (marginC <= 0.0) return WetFrictionPenalty;          // condensing — wet floor
        if (marginC >= greasyMarginC) return 1.0;               // clear of dew — dry
        var t = marginC / greasyMarginC;                        // 0..1 across the greasy band
        var s = t * t * (3.0 - 2.0 * t);                        // smoothstep (flat ends, no kink)
        return WetFrictionPenalty + (1.0 - WetFrictionPenalty) * s;
    }

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
    /// peak, then falls off — "pretty shite" past ~10 mph on exposed ground —
    /// and decays to a non-zero floor (<see cref="WindStrongFloorScore"/>) past
    /// ~25 mph rather than hitting zero, so a gale doesn't annihilate the whole
    /// verdict via the blend's <c>min</c> term.</summary>
    public static double WindComfort(double windMph)
    {
        if (windMph <= WindIdealLoMph) return WindCalmScore + windMph / WindIdealLoMph * (1.0 - WindCalmScore);
        if (windMph <= WindIdealHiMph) return 1.0;
        if (windMph <= WindHarshMph) return Clamp01(1.0 - (windMph - WindIdealHiMph) / (WindHarshMph - WindIdealHiMph) * (1.0 - WindHarshScore));
        return Math.Max(WindStrongFloorScore, WindHarshScore - (windMph - WindHarshMph) * WindHarshSlope);
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
        Factor? sea = null,
        bool surfaceWaterGate = false,
        double wetThresholdMm = RainWetThresholdMm,
        DateTime? rainDryByUtc = null,
        double greasyMarginC = DefaultGreasyMarginC,
        double windExposure = 1.0)
    {
        // ---- Gates (any one → Off) ----
        var (elevDeg, _) = SolarGeometry.SolarPosition(validUtc, latitude, longitude);
        if (elevDeg <= DaylightMinElevationDeg)
            return Gate(ConditionsTier.Off, "Dark — sun below the horizon");
        if (pWet is double pw && pw >= RainGateProb)
            return Gate(ConditionsTier.Off, $"Rain likely ({pw * 100:0}%)");
        // Rock still wet from RAIN — a hard Off gate (Harry 2026-06-16): no point
        // calling an hour climbable when a downpour an hour or two ago left a film
        // of standing water on the slab. Fires ONLY where the drying model is
        // enabled for the location AND the wetness is rain-derived; the dry-by ETA
        // (when the rain film clears) tells the reader when it comes good. Dew
        // wetness is deliberately NOT gated here — it's the friction penalty below,
        // because the rock-temp/dew margin is still being field-validated.
        if (surfaceWaterGate && rock is { } wetRock && wetRock.RainWaterMm >= wetThresholdMm)
        {
            var eta = rainDryByUtc is { } dry
                ? $" — drying, climbable from ~{dry:HH'Z'}"
                : "";
            return Gate(ConditionsTier.Off, $"Rock wet from rain{eta}");
        }
        // NB condensation (wet rock) is NOT a gate — it's a heavy friction
        // penalty in the factor block below (Harry 2026-06-16), because the
        // rock-temp calc is still being validated and an uncertain "wet" call
        // shouldn't force the whole verdict Off.

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
            // Greasiness scales friction CONTINUOUSLY with the rock-vs-dew margin
            // (smooth ramp: wet floor at margin ≤ 0 → no penalty at margin ≥
            // greasyMarginC), so the verdict glides as the rock warms past the dew
            // point instead of snapping a whole tier at the old hard greasy/dry cut
            // (Harry 2026-06-20). The label still names the tier for the reader.
            frictionScore *= GreasyFrictionMultiplier(rk.CondensationMarginC, greasyMarginC);
            if (rk.GreasinessStatus == RockSurfacePhysics.StatusPotentiallyGreasy)
                frictionDetail += ", greasy";
            else if (rk.GreasinessStatus == RockSurfacePhysics.StatusCondensation)
                frictionDetail += $", wet (≤ dew {rk.DewPointC:0.0}°C)";
        }
        else
        {
            frictionScore = RockFriction(airTempC);
            frictionDetail = $"~{airTempC:0.0}°C (from air)";
        }
        factors.Add(new Factor("Friction", frictionScore, frictionDetail));

        factors.Add(new Factor("Air temp", AirComfort(airTempC), $"{airTempC:0.0}°C"));

        if (windMph is double mph)
        {
            // Per-location exposure scales the forecast wind before the comfort
            // curve (the curve is tuned to the exposed tor; sheltered crags scale
            // down). Detail shows the forecast wind, plus the effective sheltered
            // wind the score actually used when exposure ≠ 1.
            var effMph = mph * windExposure;
            var windDetail = windExposure == 1.0 ? $"{mph:0} mph" : $"{mph:0} mph (≈{effMph:0} mph here)";
            factors.Add(new Factor("Wind", WindComfort(effMph), windDetail));
        }

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

    /// <summary>
    /// For each rain-wet hour in a rock-surface series (RainWaterMm ≥
    /// <paramref name="wetThresholdMm"/>), the first LATER hour whose rain film
    /// has dropped below the threshold — the "climbable from ~HHZ" ETA the Off
    /// gate shows. A null value means the rain film never clears within the
    /// series window (still wet to the end of the forecast). Hours that are
    /// already dry are absent from the map. Keyed by the rock row's ValidTimeUtc.
    /// </summary>
    public static Dictionary<DateTime, DateTime?> RainDryByMap(
        IEnumerable<SitePages.RockSurfaceForecastPoint> rockSeries,
        double wetThresholdMm = RainWetThresholdMm)
    {
        var ordered = rockSeries.OrderBy(r => r.ValidTimeUtc).ToList();
        var map = new Dictionary<DateTime, DateTime?>();
        for (var i = 0; i < ordered.Count; i++)
        {
            if (ordered[i].RainWaterMm < wetThresholdMm) continue; // dry now
            DateTime? dry = null;
            for (var j = i + 1; j < ordered.Count; j++)
            {
                if (ordered[j].RainWaterMm < wetThresholdMm) { dry = ordered[j].ValidTimeUtc; break; }
            }
            map[ordered[i].ValidTimeUtc] = dry; // null = wet to end of window
        }
        return map;
    }

    public static string TierLabel(ConditionsTier t) => t switch
    {
        ConditionsTier.Prime => "Prime",
        ConditionsTier.Good => "Good",
        ConditionsTier.Marginal => "Marginal",
        ConditionsTier.Poor => "Poor",
        _ => "Off",
    };

    /// <summary>Tier → hex colour. A grey→green ramp (slate Off → light-grey Poor
    /// → pale-green Marginal → light-green Good → green Prime) that deliberately
    /// avoids amber/red so the climbing verdict never clashes with the weather
    /// alert palette (which owns amber/red). Harry 2026-06-18.</summary>
    public static string TierColor(ConditionsTier t) => t switch
    {
        ConditionsTier.Prime => "#2e7d32",     // green
        ConditionsTier.Good => "#7cb342",      // light green
        ConditionsTier.Marginal => "#aed581",  // paler green
        ConditionsTier.Poor => "#b0bec5",      // light grey
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
