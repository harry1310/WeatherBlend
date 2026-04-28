using FluentAssertions;
using WeatherBlend.Predict.Utci;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins each helper in <see cref="FeelsLikeCalculator"/> against published
/// reference behaviour so the polynomial transcription, vapour-pressure
/// formula, and Tmrt radiation-balance can't silently drift.
///
/// Where exact reference numerics aren't readily quotable, tests assert
/// physical direction-of-effect (e.g. clear sky &gt; cloudy Tmrt) plus a
/// loose magnitude band — those are the regressions we care about catching.
/// </summary>
public class FeelsLikeCalculatorTests
{
    // ---------- vapour pressure ----------

    [Theory]
    [InlineData(0.0,   6.11,  0.10)]   // ~6.11 hPa at 0°C
    [InlineData(20.0, 23.39,  0.20)]
    [InlineData(25.0, 31.69,  0.20)]
    [InlineData(30.0, 42.46,  0.30)]
    public void SaturationVapourPressure_matches_Wexler(double tempC, double expectedHpa, double tol)
    {
        FeelsLikeCalculator.SaturationVapourPressureHpa(tempC)
            .Should().BeApproximately(expectedHpa, tol);
    }

    [Fact]
    public void VapourPressure_scales_linearly_with_RH()
    {
        var es20 = FeelsLikeCalculator.SaturationVapourPressureHpa(20.0);
        FeelsLikeCalculator.VapourPressureHpa(20.0, 50.0).Should().BeApproximately(es20 * 0.5, 1e-6);
        FeelsLikeCalculator.VapourPressureHpa(20.0, 100.0).Should().BeApproximately(es20, 1e-6);
        FeelsLikeCalculator.VapourPressureHpa(20.0, 0.0).Should().Be(0.0);
    }

    // ---------- longwave ----------

    [Fact]
    public void LongwaveUp_matches_StefanBoltzmann_with_ground_emissivity()
    {
        // 0.95 * σ * 293.15^4 ≈ 397.6 W/m²
        FeelsLikeCalculator.LongwaveUpWm2(20.0).Should().BeApproximately(397.6, 1.0);
    }

    [Fact]
    public void LongwaveDown_overcast_approaches_blackbody_at_Ta()
    {
        // cloud=1 → εeff = 1 → Ldown = σ·Ta⁴ ≈ 418.6 W/m² at 20°C
        var pv = FeelsLikeCalculator.VapourPressureHpa(20.0, 50.0);
        FeelsLikeCalculator.LongwaveDownWm2(20.0, 1.0, pv).Should().BeApproximately(418.6, 0.5);
    }

    [Fact]
    public void LongwaveDown_clear_sky_below_blackbody_at_Ta()
    {
        // εclear < 1 → clear-sky Ldown is materially below blackbody. At
        // Ta=20°C, e≈11.7 hPa, εclear ≈ 0.78 → Ldown ≈ 326 W/m².
        var pv = FeelsLikeCalculator.VapourPressureHpa(20.0, 50.0);
        var lDownClear = FeelsLikeCalculator.LongwaveDownWm2(20.0, 0.0, pv);
        lDownClear.Should().BeLessThan(380.0);
        lDownClear.Should().BeGreaterThan(280.0);
    }

    [Fact]
    public void LongwaveDown_increases_with_cloud_cover()
    {
        var pv = FeelsLikeCalculator.VapourPressureHpa(20.0, 50.0);
        var l0 = FeelsLikeCalculator.LongwaveDownWm2(20.0, 0.0, pv);
        var l5 = FeelsLikeCalculator.LongwaveDownWm2(20.0, 0.5, pv);
        var l1 = FeelsLikeCalculator.LongwaveDownWm2(20.0, 1.0, pv);
        l5.Should().BeGreaterThan(l0);
        l1.Should().BeGreaterThan(l5);
    }

    // ---------- Tmrt ----------

    [Fact]
    public void Tmrt_overcast_no_sun_close_to_air_temperature()
    {
        // Heavy cloud, no shortwave → Tmrt slightly below Ta because ground
        // emissivity (0.95) &lt; assumed body emissivity (0.97), so the body
        // emits more than it absorbs at the same temperature. Real effect;
        // the 1–2 K gap matches published outdoor radiation-balance studies.
        var t = FeelsLikeCalculator.Tmrt(15.0, shortwaveDownWm2: 0.0, cloudFrac: 1.0, rhPercent: 80.0);
        t.Should().BeApproximately(15.0, 2.5);
        t.Should().BeLessThan(15.0);
    }

    [Fact]
    public void Tmrt_clear_summer_midday_lifts_well_above_Ta()
    {
        // Ta=25°C, RH=50%, full sun (~800 W/m²), no cloud — published Höppe /
        // Lindberg curves put Tmrt ≈ 50–60°C for an upright unobstructed
        // person under these conditions.
        var t = FeelsLikeCalculator.Tmrt(25.0, shortwaveDownWm2: 800.0, cloudFrac: 0.0, rhPercent: 50.0);
        t.Should().BeGreaterThan(45.0);
        t.Should().BeLessThan(60.0);
    }

    [Fact]
    public void Tmrt_clear_night_cools_below_Ta()
    {
        // Clear sky, no sun → radiative cooling pulls Tmrt below Ta.
        var t = FeelsLikeCalculator.Tmrt(10.0, shortwaveDownWm2: 0.0, cloudFrac: 0.0, rhPercent: 60.0);
        t.Should().BeLessThan(10.0);
        t.Should().BeGreaterThan(-15.0);
    }

    [Fact]
    public void Tmrt_responds_to_shortwave_monotonically()
    {
        var lo = FeelsLikeCalculator.Tmrt(20.0, shortwaveDownWm2:   0.0, cloudFrac: 0.5, rhPercent: 50.0);
        var md = FeelsLikeCalculator.Tmrt(20.0, shortwaveDownWm2: 200.0, cloudFrac: 0.5, rhPercent: 50.0);
        var hi = FeelsLikeCalculator.Tmrt(20.0, shortwaveDownWm2: 600.0, cloudFrac: 0.5, rhPercent: 50.0);
        md.Should().BeGreaterThan(lo);
        hi.Should().BeGreaterThan(md);
    }

    // ---------- wind ----------

    [Fact]
    public void WindAt1m1_uses_log_law_factor()
    {
        // log(1.1/0.01) / log(10/0.01) ≈ 0.6809
        FeelsLikeCalculator.WindAt1m1(5.0).Should().BeApproximately(3.4045, 0.001);
        FeelsLikeCalculator.WindAt1m1(0.0).Should().Be(0.0);
    }

    // ---------- UTCI polynomial ----------

    [Fact]
    public void Utci_thermoneutral_case_close_to_Ta()
    {
        // Ta=25°C, vapour pressure ≈ 16 hPa (RH 50), va=1, Tmrt=Ta.
        // The neutral case lands near the "no thermal stress" band.
        var pv = FeelsLikeCalculator.VapourPressureHpa(25.0, 50.0);
        var u = FeelsLikeCalculator.Utci(25.0, pv, 1.0, 25.0);
        u.Should().BeInRange(20.0, 30.0);
    }

    [Fact]
    public void Utci_strong_heat_stress_when_radiation_dominant()
    {
        // Hot day with high Tmrt and weak wind — should land in heat-stress.
        var pv = FeelsLikeCalculator.VapourPressureHpa(35.0, 50.0);
        var u = FeelsLikeCalculator.Utci(35.0, pv, 1.0, 50.0);
        u.Should().BeGreaterThan(38.0); // very strong heat stress band
    }

    [Fact]
    public void Utci_strong_cold_stress_when_windchill_dominant()
    {
        // Cold + wind → cold stress
        var pv = FeelsLikeCalculator.VapourPressureHpa(-5.0, 70.0);
        var u = FeelsLikeCalculator.Utci(-5.0, pv, 10.0, -5.0);
        u.Should().BeLessThan(-13.0); // strong cold stress
    }

    [Fact]
    public void Utci_increases_with_Tmrt_holding_others_fixed()
    {
        var pv = FeelsLikeCalculator.VapourPressureHpa(20.0, 50.0);
        var lo = FeelsLikeCalculator.Utci(20.0, pv, 2.0, 10.0);
        var md = FeelsLikeCalculator.Utci(20.0, pv, 2.0, 20.0);
        var hi = FeelsLikeCalculator.Utci(20.0, pv, 2.0, 40.0);
        md.Should().BeGreaterThan(lo);
        hi.Should().BeGreaterThan(md);
    }

    [Fact]
    public void Utci_decreases_with_wind_in_cool_weather()
    {
        // Wind chill effect at 5°C with mild radiation
        var pv = FeelsLikeCalculator.VapourPressureHpa(5.0, 70.0);
        var calm  = FeelsLikeCalculator.Utci(5.0, pv, 0.5, 5.0);
        var windy = FeelsLikeCalculator.Utci(5.0, pv, 8.0, 5.0);
        windy.Should().BeLessThan(calm);
    }

    // ---------- bands ----------

    [Theory]
    [InlineData(-50.0, UtciStress.ExtremeCold)]
    [InlineData(-30.0, UtciStress.VeryStrongCold)]
    [InlineData(-20.0, UtciStress.StrongCold)]
    [InlineData(-5.0,  UtciStress.ModerateCold)]
    [InlineData( 5.0,  UtciStress.SlightCold)]
    [InlineData(15.0,  UtciStress.NoStress)]
    [InlineData(28.0,  UtciStress.ModerateHeat)]
    [InlineData(35.0,  UtciStress.StrongHeat)]
    [InlineData(40.0,  UtciStress.VeryStrongHeat)]
    [InlineData(48.0,  UtciStress.ExtremeHeat)]
    public void Band_picks_published_thresholds(double utci, UtciStress expected)
    {
        FeelsLikeCalculator.Band(utci).Should().Be(expected);
    }
}
