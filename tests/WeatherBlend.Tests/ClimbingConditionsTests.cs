using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Climbing-conditions index (idea #1). No ground truth exists for "good
/// conditions", so these pin the QUALITATIVE shape of the hand-tuned curves +
/// the gate behaviour to Harry's 2026-06-13 calibration — not absolute scores.
/// </summary>
public class ClimbingConditionsTests
{
    // Bonehill, a winter midday so daylight gate is open for the quality tests.
    private const double Lat = 50.5831, Lon = -3.7931;
    private static readonly DateTime Midday = new(2026, 1, 15, 12, 0, 0, DateTimeKind.Utc);

    private static SitePages.RockSurfaceForecastPoint Rock(
        double rockC, string greasy = "dry", double rainWaterMm = 0.0, double dewWaterMm = 0.0,
        DateTime? validUtc = null)
    {
        // Keep the condensation margin consistent with the greasiness status —
        // the climbing index's friction penalty is now a continuous function of
        // the rock-vs-dew margin (not the status string), so a representative
        // margin per tier is what exercises it: condensation ⇒ margin ≤ 0;
        // potentially_greasy ⇒ mid-band (< the 3.0 default greasy margin); dry ⇒ clear.
        double margin = greasy switch
        {
            "condensation" => -0.5,
            "potentially_greasy" => 1.0,
            _ => 4.0,
        };
        return new(
            Version: "v1", PredictedAtUtc: Midday.AddDays(-1), ValidTimeUtc: validUtc ?? Midday, LeadHours: 24,
            RockSurfaceTempC: rockC, AirTempC: rockC + 1, DewPointC: rockC - margin,
            CondensationMarginC: margin, GreasinessStatus: greasy,
            SurfaceWaterMm: rainWaterMm + dewWaterMm, RainWaterMm: rainWaterMm,
            LocationName: "bonehill_rocks", Face: "");
    }

    // ---- Curve shapes ----

    [Fact]
    public void RockFriction_plateaus_5_to_10C_then_falls_off_both_sides()
    {
        // Flat peak band 5–10°C — both edges score a perfect 1.0.
        ClimbingConditions.RockFriction(5).Should().BeApproximately(1.0, 1e-9);
        ClimbingConditions.RockFriction(10).Should().BeApproximately(1.0, 1e-9);
        ClimbingConditions.RockFriction(7).Should().BeApproximately(1.0, 1e-9);
        // Warmer than the band = gradually worse grip, but it bottoms out at the
        // non-zero floor (warm rock is poor, not impossible).
        ClimbingConditions.RockFriction(20).Should().BeLessThan(ClimbingConditions.RockFriction(15));
        ClimbingConditions.RockFriction(25).Should().BeApproximately(
            ClimbingConditions.RockFrictionWarmFloorScore, 1e-9);
        // Past the floor point it holds the floor, never zero.
        ClimbingConditions.RockFriction(35).Should().BeApproximately(
            ClimbingConditions.RockFrictionWarmFloorScore, 1e-9);
        // Below the band degrades SHARPLY — 0°C is well under the warm floor
        // (verglas / numb hands genuinely near-unclimbable).
        ClimbingConditions.RockFriction(0).Should().BeLessThan(
            ClimbingConditions.RockFrictionWarmFloorScore);
    }

    [Fact]
    public void AirComfort_peaks_in_the_9_to_10_band_and_rejects_hot_and_cold()
    {
        var ideal = ClimbingConditions.AirComfort(9.5);
        ideal.Should().BeApproximately(1.0, 1e-9);
        ClimbingConditions.AirComfort(25).Should().BeLessThan(0.2, "over 20°C is too hot");
        ClimbingConditions.AirComfort(2).Should().BeLessThan(0.3, "under 5°C is bad");
        ClimbingConditions.AirComfort(15).Should().BeInRange(0.3, 0.9);
    }

    [Fact]
    public void WindComfort_peaks_at_a_light_breeze_and_tanks_past_10mph()
    {
        ClimbingConditions.WindComfort(6).Should().BeApproximately(1.0, 1e-9);
        ClimbingConditions.WindComfort(0).Should().BeInRange(0.5, 0.75, "calm is fine but a breeze is better");
        ClimbingConditions.WindComfort(0).Should().BeLessThan(ClimbingConditions.WindComfort(6));
        ClimbingConditions.WindComfort(15).Should().BeLessThan(ClimbingConditions.WindComfort(10), "strong wind drags comfort down");
        ClimbingConditions.WindComfort(12).Should().BeLessThan(ClimbingConditions.WindComfort(10));
    }

    [Fact]
    public void WindComfort_decays_to_a_nonzero_floor_not_to_zero()
    {
        // 2026-06-17: a gale must drag the verdict down hard but NOT zero the
        // whole weakest-link blend. The curve bottoms out at the strong-wind
        // floor (~25 mph+) and never goes below it.
        ClimbingConditions.WindComfort(18).Should().BeGreaterThan(0.0, "18 mph is grim but not annihilating");
        ClimbingConditions.WindComfort(18).Should().BeLessThan(ClimbingConditions.WindComfort(12), "but worse than 12 mph");
        ClimbingConditions.WindComfort(40).Should().BeApproximately(
            ClimbingConditions.WindStrongFloorScore, 1e-9, "a storm sits on the floor");
        ClimbingConditions.WindComfort(40).Should().BeGreaterThan(0.0);
    }

    // ---- Gates ----

    [Fact]
    public void Night_is_gated_Off()
    {
        var night = new DateTime(2026, 1, 15, 23, 0, 0, DateTimeKind.Utc);
        var r = ClimbingConditions.Evaluate(night, Lat, Lon, 9, 0.0, 6, Rock(5));
        r.Tier.Should().Be(ConditionsTier.Off);
        r.Reason.Should().Contain("Dark");
    }

    [Fact]
    public void Rain_is_gated_Off()
    {
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.8, 6, Rock(5));
        r.Tier.Should().Be(ConditionsTier.Off);
        r.Reason.Should().Contain("Rain");
    }

    [Fact]
    public void Condensation_heavily_penalises_friction_without_gating_Off()
    {
        // 2026-06-16: condensation is NOT a hard Off-gate any more (the rock-temp
        // calc is still being validated) — it's a heavy friction penalty that
        // drags the verdict down, heavier than greasy, with the why in the detail.
        var wet    = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.0, 6, Rock(5, greasy: "condensation"));
        var greasy = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.0, 6, Rock(5, greasy: "potentially_greasy"));

        wet.Tier.Should().NotBe(ConditionsTier.Off);
        wet.Tier.Should().BeOneOf(ConditionsTier.Marginal, ConditionsTier.Poor);
        wet.Score.Should().BeLessThan(greasy.Score, "wet rock is worse than merely greasy");
        wet.Factors.Single(f => f.Name == "Friction").Detail.Should().Contain("wet");
    }

    // ---- Continuous greasiness ramp (no hard snap) ----

    [Fact]
    public void GreasyFrictionMultiplier_is_continuous_across_the_dry_threshold()
    {
        // The old hard cut jumped ×0.4 → ×1.0 at the greasy margin. The ramp must
        // be continuous: a hair below the threshold ≈ a hair above (≈ 1.0).
        const double g = 1.5;
        var justUnder = ClimbingConditions.GreasyFrictionMultiplier(1.49, g);
        var justOver  = ClimbingConditions.GreasyFrictionMultiplier(1.51, g);
        justOver.Should().Be(1.0);
        justUnder.Should().BeApproximately(1.0, 0.02);
        (justOver - justUnder).Should().BeLessThan(0.05);
    }

    [Fact]
    public void GreasyFrictionMultiplier_floors_at_wet_and_climbs_monotonically()
    {
        const double g = 1.5;
        ClimbingConditions.GreasyFrictionMultiplier(0.0, g).Should().Be(ClimbingConditions.WetFrictionPenalty);
        ClimbingConditions.GreasyFrictionMultiplier(-2.0, g).Should().Be(ClimbingConditions.WetFrictionPenalty);
        ClimbingConditions.GreasyFrictionMultiplier(3.0, g).Should().Be(1.0);
        // Strictly increasing across the band.
        var prev = -1.0;
        foreach (var m in new[] { 0.1, 0.4, 0.7, 1.0, 1.3 })
        {
            var v = ClimbingConditions.GreasyFrictionMultiplier(m, g);
            v.Should().BeGreaterThan(prev);
            prev = v;
        }
    }

    [Fact]
    public void Greasy_to_dry_no_longer_snaps_a_whole_tier_as_the_rock_warms()
    {
        // The Sennen 08→09Z case: rock warming nudged the margin 1.44 → 1.83 across
        // the 1.5°C maritime greasy cut and the verdict vaulted Marginal→Prime. With
        // the continuous ramp the two adjacent hours must sit close, not snap.
        const double g = 1.5;
        var eightZ = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6, RockMargin(15.0, 1.44), greasyMarginC: g);
        var nineZ  = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6, RockMargin(15.4, 1.83), greasyMarginC: g);
        (nineZ.Score - eightZ.Score).Should().BeLessThan(0.10, "the verdict should glide, not snap a tier");
    }

    // Rock point at an explicit condensation margin (dew = rock − margin), greasy
    // status derived from the margin so the detail label stays sensible.
    private static SitePages.RockSurfaceForecastPoint RockMargin(double rockC, double marginC) => new(
        Version: "v1", PredictedAtUtc: Midday.AddDays(-1), ValidTimeUtc: Midday, LeadHours: 24,
        RockSurfaceTempC: rockC, AirTempC: rockC + 1, DewPointC: rockC - marginC,
        CondensationMarginC: marginC,
        GreasinessStatus: marginC <= 0 ? "condensation" : marginC <= 1.5 ? "potentially_greasy" : "dry",
        LocationName: "sennen_cove", Face: "south");

    // ---- End-to-end verdicts ----

    [Fact]
    public void A_cold_dry_light_breeze_day_is_Prime()
    {
        // Rock 5°C, air 9°C, 6 mph, dry, daylight.
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6, Rock(5));
        r.Tier.Should().Be(ConditionsTier.Prime);
    }

    [Fact]
    public void Strong_wind_alone_drags_an_otherwise_perfect_day_down()
    {
        var calm = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6, Rock(5));
        var windy = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 18, Rock(5));
        windy.Score.Should().BeLessThan(calm.Score);
        windy.Tier.Should().BeOneOf(ConditionsTier.Marginal, ConditionsTier.Poor);
        windy.Reason.Should().Contain("wind");
    }

    [Fact]
    public void Greasy_rock_knocks_friction_down_without_gating()
    {
        var dry = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6, Rock(5));
        var greasy = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6, Rock(5, greasy: "potentially_greasy"));
        greasy.Tier.Should().NotBe(ConditionsTier.Off);
        greasy.Score.Should().BeLessThan(dry.Score);
    }

    [Fact]
    public void Missing_rock_falls_back_to_air_temp_friction_and_omits_factors_cleanly()
    {
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, null, null, rock: null);
        r.Tier.Should().NotBe(ConditionsTier.Off);
        // Only Friction (air proxy) + Air temp present — no wind, no dry factor.
        r.Factors.Select(f => f.Name).Should().BeEquivalentTo("Friction", "Air temp");
        r.Factors.Single(f => f.Name == "Friction").Detail.Should().Contain("air");
    }

    // ---- Surface-water drying gate (Phase A) ----

    [Fact]
    public void Rain_wet_rock_is_not_gated_when_the_drying_model_is_off()
    {
        // Default surfaceWaterGate=false: a wet rain film is ignored entirely,
        // so an otherwise-perfect hour stays Prime — the pre-drying behaviour.
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6,
            Rock(5, rainWaterMm: 0.3));
        r.Tier.Should().Be(ConditionsTier.Prime);
    }

    [Fact]
    public void Rain_wet_rock_hard_gates_Off_when_the_drying_model_is_on()
    {
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6,
            Rock(5, rainWaterMm: 0.3), surfaceWaterGate: true);
        r.Tier.Should().Be(ConditionsTier.Off);
        r.Reason.Should().Contain("wet from rain");
    }

    [Fact]
    public void Rain_gate_names_the_dry_by_eta_when_known()
    {
        var dryBy = new DateTime(2026, 1, 15, 14, 0, 0, DateTimeKind.Utc);
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6,
            Rock(5, rainWaterMm: 0.3), surfaceWaterGate: true, rainDryByUtc: dryBy);
        r.Tier.Should().Be(ConditionsTier.Off);
        r.Reason.Should().Contain("14Z");
    }

    [Fact]
    public void Dew_only_film_is_not_rain_gated_even_with_the_model_on()
    {
        // A dew film (RainWaterMm 0) must NOT trip the rain gate — dew is the
        // friction penalty, not a hard Off (the rock-temp margin is uncertain).
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6,
            Rock(5, greasy: "condensation", dewWaterMm: 0.3), surfaceWaterGate: true);
        r.Tier.Should().NotBe(ConditionsTier.Off);
    }

    [Fact]
    public void Sub_threshold_rain_film_does_not_gate()
    {
        var r = ClimbingConditions.Evaluate(Midday, Lat, Lon, 9, 0.02, 6,
            Rock(5, rainWaterMm: 0.01), surfaceWaterGate: true);
        r.Tier.Should().NotBe(ConditionsTier.Off, "a trace film below the wet threshold isn't 'wet'");
    }

    [Fact]
    public void RainDryByMap_maps_each_wet_hour_to_its_first_dry_hour()
    {
        var t0 = new DateTime(2026, 1, 15, 8, 0, 0, DateTimeKind.Utc);
        var series = new[]
        {
            Rock(5, rainWaterMm: 0.3, validUtc: t0),                 // wet
            Rock(5, rainWaterMm: 0.2, validUtc: t0.AddHours(1)),     // wet
            Rock(5, rainWaterMm: 0.01, validUtc: t0.AddHours(2)),    // dry ← ETA
            Rock(5, rainWaterMm: 0.0, validUtc: t0.AddHours(3)),     // dry
        };
        var map = ClimbingConditions.RainDryByMap(series);

        map.Should().ContainKey(t0);
        map[t0].Should().Be(t0.AddHours(2));
        map[t0.AddHours(1)].Should().Be(t0.AddHours(2));
        map.Should().NotContainKey(t0.AddHours(2), "already-dry hours aren't in the map");
    }

    [Fact]
    public void RainDryByMap_leaves_a_never_drying_hour_with_a_null_eta()
    {
        var t0 = new DateTime(2026, 1, 15, 8, 0, 0, DateTimeKind.Utc);
        var series = new[]
        {
            Rock(5, rainWaterMm: 0.3, validUtc: t0),
            Rock(5, rainWaterMm: 0.3, validUtc: t0.AddHours(1)),
        };
        var map = ClimbingConditions.RainDryByMap(series);
        map[t0].Should().BeNull("the film never clears within the window");
    }
}
