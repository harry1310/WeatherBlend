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
        DateTime? validUtc = null) => new(
        Version: "v1", PredictedAtUtc: Midday.AddDays(-1), ValidTimeUtc: validUtc ?? Midday, LeadHours: 24,
        RockSurfaceTempC: rockC, AirTempC: rockC + 1, DewPointC: rockC - 3,
        CondensationMarginC: 3, GreasinessStatus: greasy,
        SurfaceWaterMm: rainWaterMm + dewWaterMm, RainWaterMm: rainWaterMm,
        LocationName: "bonehill_rocks", Face: "");

    // ---- Curve shapes ----

    [Fact]
    public void RockFriction_peaks_at_5C_and_falls_off_both_sides()
    {
        var peak = ClimbingConditions.RockFriction(5);
        peak.Should().BeApproximately(1.0, 1e-9);
        // Warmer rock = gradually worse grip.
        ClimbingConditions.RockFriction(10).Should().BeLessThan(peak);
        ClimbingConditions.RockFriction(20).Should().BeLessThan(ClimbingConditions.RockFriction(12));
        // Below the peak degrades SHARPLY — 0°C worse than 10°C even though
        // both are 5° off the peak.
        ClimbingConditions.RockFriction(0).Should().BeLessThan(ClimbingConditions.RockFriction(10));
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
        ClimbingConditions.WindComfort(15).Should().BeLessThan(0.2, "pretty shite on an exposed tor");
        ClimbingConditions.WindComfort(12).Should().BeLessThan(ClimbingConditions.WindComfort(10));
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
