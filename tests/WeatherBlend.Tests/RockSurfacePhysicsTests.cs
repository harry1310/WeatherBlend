using FluentAssertions;
using WeatherBlend.Config;
using WeatherBlend.Predict.Surface;
using Xunit;
using static WeatherBlend.Predict.Surface.RockSurfacePhysics;

namespace WeatherBlend.Tests;

/// <summary>
/// Rock surface temperature physics — the same physical-behaviour checks the P0
/// spike (scripts/rock_temp_spike.py) validates, as deterministic unit tests on
/// synthetic diurnal forcing. There is no observed rock-temp truth, so these
/// prove the model is physically CREDIBLE (correct signs + diurnal shape), not
/// accurate — absolute level is a P2 on-site-calibration target.
/// </summary>
public class RockSurfacePhysicsTests
{
    private static readonly RockSurfaceConfig Cfg = new(); // defaults = spike PARAMS
    private const int Spinup = 48;

    /// <summary>Synthetic clear/cloudy diurnal forcing: air temp 10±5°C (peak
    /// ~15:00, min ~03:00), shortwave a daytime half-sine (0 at night, ~600 peak
    /// at noon), constant dew point + wind + cloud per scenario.</summary>
    private static List<ForcingHour> MakeForcing(int days, double cloudFrac, double windMs, double dewC = 6.0, double? seaC = null)
    {
        var start = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var list = new List<ForcingHour>(days * 24);
        for (var h = 0; h < days * 24; h++)
        {
            var hod = h % 24;
            var ta = 10.0 + 5.0 * Math.Sin((hod - 9) / 24.0 * 2.0 * Math.PI);
            var sw = Math.Max(0.0, 600.0 * Math.Sin((hod - 6) / 12.0 * Math.PI));
            list.Add(new ForcingHour(start.AddHours(h), ta, dewC, cloudFrac, windMs, sw, seaC));
        }
        return list;
    }

    private static IEnumerable<RockHour> AfterSpinup(IReadOnlyList<RockHour> r) => r.Skip(Spinup);

    [Fact]
    public void Clear_calm_night_cools_below_air()
    {
        var r = Integrate(MakeForcing(10, cloudFrac: 0.0, windMs: 1.0), Cfg);
        var nightDelta = AfterSpinup(r)
            .Where(h => SwAt(h) < 5.0)
            .Select(h => h.RockTempC - h.AirTempC)
            .ToList();
        nightDelta.Should().NotBeEmpty();
        nightDelta.Average().Should().BeLessThan(-0.2, "radiative cooling drives the rock below air on a clear, calm night");
    }

    [Fact]
    public void Sunny_day_runs_warmer_than_air()
    {
        var f = MakeForcing(10, cloudFrac: 0.0, windMs: 1.0);
        var r = Integrate(f, Cfg);
        // pair each post-spinup hour with its forcing SW
        var dayDelta = r.Skip(Spinup)
            .Select((h, i) => (h, sw: f[i + Spinup].ShortwaveWm2))
            .Where(x => x.sw > 300.0)
            .Select(x => x.h.RockTempC - x.h.AirTempC)
            .ToList();
        dayDelta.Should().NotBeEmpty();
        dayDelta.Average().Should().BeGreaterThan(1.0, "absorbed shortwave drives the rock above air by day");
    }

    [Fact]
    public void Overcast_night_is_better_coupled_than_clear_night()
    {
        var clear = Integrate(MakeForcing(10, 0.0, 1.0), Cfg);
        var overcast = Integrate(MakeForcing(10, 1.0, 1.0), Cfg);
        var clearNight = Math.Abs(AfterSpinup(clear).Where(h => SwAt(h) < 5.0).Average(h => h.RockTempC - h.AirTempC));
        var overcastNight = Math.Abs(AfterSpinup(overcast).Where(h => SwAt(h) < 5.0).Average(h => h.RockTempC - h.AirTempC));
        overcastNight.Should().BeLessThan(clearNight, "cloud's extra LW↓ keeps the rock closer to air at night");
    }

    [Fact]
    public void Windy_clear_night_cools_less_than_calm_clear_night()
    {
        var calm = Integrate(MakeForcing(10, 0.0, 1.0), Cfg);
        var windy = Integrate(MakeForcing(10, 0.0, 8.0), Cfg);
        var calmNight = AfterSpinup(calm).Where(h => SwAt(h) < 5.0).Average(h => h.RockTempC - h.AirTempC);
        var windyNight = AfterSpinup(windy).Where(h => SwAt(h) < 5.0).Average(h => h.RockTempC - h.AirTempC);
        windyNight.Should().BeGreaterThan(calmNight, "convective coupling pulls the rock back toward air");
    }

    [Fact]
    public void Diurnal_swing_of_rock_is_wider_than_air()
    {
        var f = MakeForcing(10, 0.0, 1.0);
        var r = Integrate(f, Cfg);
        // amplitude of the last full day
        var lastDay = r.Skip(r.Count - 24).ToList();
        var tsAmp = lastDay.Max(h => h.RockTempC) - lastDay.Min(h => h.RockTempC);
        var taAmp = lastDay.Max(h => h.AirTempC) - lastDay.Min(h => h.AirTempC);
        tsAmp.Should().BeGreaterThan(taAmp, "the radiating skin over/under-shoots air across the day");
    }

    [Fact]
    public void Longwave_rises_with_cloud_and_matches_brutsaert()
    {
        // Brutsaert clear-sky εclear at a known point.
        var taK = 283.15; // 10°C
        var eClear = ClearSkyEmissivity(dewPointC: 6.0, airTempK: taK);
        eClear.Should().BeInRange(0.7, 0.95); // typical mild-moist clear sky

        var clear = LongwaveDownWm2(6.0, 10.0, cloudFrac: 0.0, Cfg.LwClearK, Cfg.LwCloudK);
        var cloudy = LongwaveDownWm2(6.0, 10.0, cloudFrac: 1.0, Cfg.LwClearK, Cfg.LwCloudK);
        cloudy.Should().BeGreaterThan(clear, "cloud enhances downwelling longwave");
        // sanity: both within a plausible LW↓ envelope for ~10°C
        clear.Should().BeInRange(200.0, 360.0);
        cloudy.Should().BeInRange(clear, 400.0);
    }

    [Fact]
    public void MuFromProps_is_positive_and_scales_with_mu_scale()
    {
        var baseMu = MuFromProps(Cfg);
        baseMu.Should().BeGreaterThan(0);
        var doubled = MuFromProps(new RockSurfaceConfig { MuScale = Cfg.MuScale * 2 });
        doubled.Should().BeApproximately(baseMu * 2, baseMu * 1e-9);
    }

    [Theory]
    [InlineData(-0.5, "condensation")]
    [InlineData(0.0, "condensation")]
    [InlineData(1.5, "potentially_greasy")]
    [InlineData(3.0, "potentially_greasy")]
    [InlineData(3.01, "dry")]
    [InlineData(10.0, "dry")]
    public void Greasiness_tiers_honour_the_threshold(double margin, string expected)
        => Greasiness(margin, greasyMarginC: 3.0).Should().Be(expected);

    // Reconstruct the forcing SW for a result hour from its hour-of-day (the
    // synthetic forcing is deterministic), so night masking doesn't need the
    // original forcing list threaded through.
    private static double SwAt(RockHour h)
        => Math.Max(0.0, 600.0 * Math.Sin((h.ValidTimeUtc.Hour - 6) / 12.0 * Math.PI));

    // ------------------------------------------------------------------
    // S4 — sea longwave (SENNEN_ROCK_TEMP_PLAN.md): a wall above the
    // Atlantic sees the sea where a tor sees cold sky; the warm sea damps
    // night cooling, a cold sea trims daytime heating, and a full sky view
    // makes the term inert.
    // ------------------------------------------------------------------

    [Fact]
    public void Warm_sea_in_the_view_damps_clear_night_cooling()
    {
        var cliff = new RockSurfaceConfig { FSky = 0.5 };
        var neutral = Integrate(MakeForcing(10, 0.0, 1.0), cliff);
        var sea = Integrate(MakeForcing(10, 0.0, 1.0, seaC: 14.0), cliff);

        var neutralNight = AfterSpinup(neutral).Where(h => SwAt(h) < 5.0).Average(h => h.RockTempC - h.AirTempC);
        var seaNight = AfterSpinup(sea).Where(h => SwAt(h) < 5.0).Average(h => h.RockTempC - h.AirTempC);

        seaNight.Should().BeGreaterThan(neutralNight + 0.1,
            "a 14°C sea filling half the view radiates warmth the open sky does not");
    }

    [Fact]
    public void Cold_sea_in_the_view_trims_daytime_heating()
    {
        var cliff = new RockSurfaceConfig { FSky = 0.5 };
        var fNeutral = MakeForcing(10, 0.0, 1.0);
        var neutral = Integrate(fNeutral, cliff);
        var sea = Integrate(MakeForcing(10, 0.0, 1.0, seaC: 5.0), cliff);

        double DayMean(IReadOnlyList<RockHour> r) => r.Skip(Spinup)
            .Select((h, i) => (h, sw: fNeutral[i + Spinup].ShortwaveWm2))
            .Where(x => x.sw > 300.0)
            .Average(x => x.h.RockTempC);

        DayMean(sea).Should().BeLessThan(DayMean(neutral) - 0.1,
            "sun-warmed rock radiates to a 5°C sea where neutral surroundings return nothing");
    }

    [Fact]
    public void Full_sky_view_makes_the_sea_term_inert()
    {
        // FSky = 1 (default): the sea weight (1−FSky) is zero, so the sea
        // forcing must not change a single hour — Bonehill stays untouched
        // even if a sea temp ever reached its forcing.
        var withSea = Integrate(MakeForcing(6, 0.3, 2.0, seaC: 14.0), Cfg);
        var without = Integrate(MakeForcing(6, 0.3, 2.0), Cfg);

        withSea.Should().HaveSameCount(without);
        for (var i = 0; i < withSea.Count; i++)
            withSea[i].RockTempC.Should().Be(without[i].RockTempC);
    }
}
