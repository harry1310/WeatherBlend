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
    public void DampedHorizontalSw_scales_only_the_direct_beam()
    {
        // 600 W/m² horizontal, 70% direct / 30% diffuse.
        DampedHorizontalSw(600, 0.7, 1.0).Should().BeApproximately(600, 1e-9); // unchanged
        DampedHorizontalSw(600, 0.7, 0.0).Should().BeApproximately(180, 1e-9); // diffuse only = 600×0.3
        DampedHorizontalSw(600, 0.7, 0.5).Should().BeApproximately(390, 1e-9); // 180 + 0.5×420
        DampedHorizontalSw(300, 0.0, 0.2).Should().BeApproximately(300, 1e-9); // all-diffuse sky → factor irrelevant
        DampedHorizontalSw(500, 1.5, 0.0).Should().BeApproximately(0, 1e-9);   // directFrac clamps to [0,1]
    }

    [Fact]
    public void Damped_direct_beam_lowers_the_midday_rock_peak()
    {
        // Same forcing, but pre-damp the direct beam exactly as the pipeline's
        // horizontal branch does before Integrate. The damped run must sit cooler
        // above air at midday; night (SW=0) is unaffected by construction.
        var full = MakeForcing(10, cloudFrac: 0.2, windMs: 2.0);
        var damped = full
            .Select(h => h with { ShortwaveWm2 = DampedHorizontalSw(h.ShortwaveWm2, 0.7, 0.4) })
            .ToList();
        double Peak(IReadOnlyList<RockHour> r) => AfterSpinup(r).Max(h => h.RockTempC - h.AirTempC);
        Peak(Integrate(damped, Cfg)).Should().BeLessThan(Peak(Integrate(full, Cfg)),
            "damping the direct beam cuts daytime absorbed SW, so the rock runs less far above air");
    }

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

    // ------------------------------------------------------------------
    // Surface-water drying model (Phase A) + latent-heat feedback (Phase B).
    // The film accrues from rain (and dew), dries by VPD + radiation, runs
    // off above the cap, and — only when SurfaceWaterEnabled — pulls latent
    // heat out of the slab as it evaporates.
    // ------------------------------------------------------------------

    /// <summary>Rain at one hour of an otherwise-dry sunny day, holding enough
    /// film to survive runoff, then drying out over the following hours.</summary>
    private static List<ForcingHour> MakeForcingWithMorningRain(int days, double rainMm, int rainHod = 6)
        => MakeForcing(days, cloudFrac: 0.1, windMs: 2.0)
            .Select(h => h.ValidTimeUtc.Hour == rainHod ? h with { PrecipMm = rainMm } : h)
            .ToList();

    [Fact]
    public void Rain_wets_the_film_then_it_dries_out()
    {
        // Generous cap so the rain isn't all shed as runoff; watch the rain
        // film build at the rain hour then fall back toward zero over the day.
        var cfg = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0 };
        var r = Integrate(MakeForcingWithMorningRain(6, rainMm: 3.0), cfg);

        var lastDay = r.Skip(r.Count - 24).ToList();
        var atRain = lastDay.First(h => h.ValidTimeUtc.Hour == 6);
        var lateAfternoon = lastDay.First(h => h.ValidTimeUtc.Hour == 16);

        atRain.RainWaterMm.Should().BeGreaterThan(0.5, "rain deposits a film on the slab");
        atRain.SurfaceWaterMm.Should().BeGreaterThanOrEqualTo(atRain.RainWaterMm);
        lateAfternoon.RainWaterMm.Should().BeLessThan(atRain.RainWaterMm,
            "sun + wind dry the rain film back down through the day");

        // The wet-event timestamp is stamped at the rain hour and carried forward
        // (so a later, still-drying hour still knows when it last rained).
        atRain.LastRainAtUtc.Should().Be(atRain.ValidTimeUtc);
        lateAfternoon.LastRainAtUtc.Should().Be(atRain.ValidTimeUtc,
            "the wet-event time persists after the rain stops, while the film dries");
    }

    [Fact]
    public void Vertical_face_drains_the_rain_film_far_faster_than_a_horizontal_slab()
    {
        // The vertical-face fix (2026-06-20): gravity drainage sheds the bulk of the
        // rain film within ~an hour, leaving only the retained residual to evaporate
        // — so a vertical wall (sinθ=1) is much drier an hour or two after rain than
        // a horizontal slab (sinθ=0, evaporation only — the legacy behaviour).
        var cfg = new RockSurfaceConfig { MaxSurfaceWaterMm = 0.4, RetainedFilmMm = 0.1 };
        var forcing = MakeForcingWithMorningRain(6, rainMm: 3.0);

        var horizontal = Integrate(forcing, cfg);                          // sinθ default 0 → ponds
        var vertical   = Integrate(forcing, cfg, drainageSlopeSine: 1.0);  // vertical wall → drains

        // Two hours after the 06Z rain, on the last modelled day.
        var hHor = horizontal.Last(h => h.ValidTimeUtc.Hour == 8);
        var hVer = vertical.Last(h => h.ValidTimeUtc.Hour == 8);

        hVer.RainWaterMm.Should().BeLessThan(hHor.RainWaterMm,
            "the vertical face sheds water by gravity; the horizontal slab can only evaporate it");
        hVer.RainWaterMm.Should().BeLessThan(cfg.RetainedFilmMm + 0.05,
            "a vertical face drains to near the retained residual within an hour or two");
    }

    [Fact]
    public void Horizontal_slab_is_unaffected_by_the_drainage_term()
    {
        // sinθ = 0 (the default) must reproduce the pre-drainage behaviour exactly —
        // a horizontal slab ponds; nothing drains.
        var cfg = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0, RetainedFilmMm = 0.1 };
        var forcing = MakeForcingWithMorningRain(6, rainMm: 3.0);
        var noArg = Integrate(forcing, cfg);
        var explicitZero = Integrate(forcing, cfg, drainageSlopeSine: 0.0);
        for (var i = 0; i < noArg.Count; i++)
            explicitZero[i].SurfaceWaterMm.Should().Be(noArg[i].SurfaceWaterMm);
    }

    [Fact]
    public void Surface_water_never_exceeds_the_runoff_cap()
    {
        var cfg = new RockSurfaceConfig { MaxSurfaceWaterMm = 0.4 };
        // Hammer it with heavy rain every hour — the film must still cap out.
        var soaked = MakeForcing(4, 0.8, 2.0).Select(h => h with { PrecipMm = 10.0 }).ToList();
        var r = Integrate(soaked, cfg);
        r.Should().OnlyContain(h => h.SurfaceWaterMm <= cfg.MaxSurfaceWaterMm + 1e-9,
            "granite holds only a thin film; the rest runs off");
    }

    [Fact]
    public void Dew_film_is_attributed_to_dew_not_rain()
    {
        // Clear calm night with a high dew point forces condensation (Ts < Td);
        // the deposited film must land in the dew bucket, leaving RainWaterMm 0.
        var r = Integrate(MakeForcing(6, cloudFrac: 0.0, windMs: 1.0, dewC: 11.0), Cfg);
        var condensingNight = AfterSpinup(r)
            .Where(h => SwAt(h) < 5.0 && h.MarginC <= 0.0)
            .ToList();
        condensingNight.Should().NotBeEmpty("a warm-dew clear night should condense");
        condensingNight.Should().OnlyContain(h => h.RainWaterMm <= 1e-9,
            "no rain fell — any film is dew");
        condensingNight.Max(h => h.SurfaceWaterMm).Should().BeGreaterThan(0.0,
            "dew deposits a measurable film");
    }

    [Fact]
    public void Latent_feedback_is_inert_when_the_drying_model_is_off()
    {
        // SurfaceWaterEnabled = false (default): rain may wet the film, but Ts
        // must be bit-for-bit identical to a run with no rain at all — the
        // latent term is gated off, so the film cannot touch the energy budget.
        var cfg = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0 };
        var wet = Integrate(MakeForcingWithMorningRain(6, rainMm: 3.0), cfg);
        var dry = Integrate(MakeForcing(6, cloudFrac: 0.1, windMs: 2.0), cfg);

        wet.Should().HaveSameCount(dry);
        for (var i = 0; i < wet.Count; i++)
            wet[i].RockTempC.Should().Be(dry[i].RockTempC,
                "with the gate off the surface film must not perturb Ts");
    }

    [Fact]
    public void Latent_cooling_pulls_a_wet_slab_below_its_dry_self()
    {
        // Same wet forcing, gate ON vs OFF: evaporating the film must cost the
        // slab latent heat, so the gate-on run sits cooler while it dries.
        var on = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0, SurfaceWaterEnabled = true };
        var off = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0, SurfaceWaterEnabled = false };
        var forcing = MakeForcingWithMorningRain(8, rainMm: 4.0);

        var rOn = Integrate(forcing, on);
        var rOff = Integrate(forcing, off);

        // Mean Ts over the drying daytime hours (rain at 06, dries through ~17).
        double DayMean(IReadOnlyList<RockHour> r) => r.Skip(Spinup)
            .Where(h => h.ValidTimeUtc.Hour is >= 7 and <= 15)
            .Average(h => h.RockTempC);

        DayMean(rOn).Should().BeLessThan(DayMean(rOff),
            "latent heat of evaporation cools the wet slab relative to the dry-physics run");
    }

    [Fact]
    public void Latent_coeff_zero_reproduces_the_gate_off_temperatures()
    {
        // Belt-and-braces: even with the gate ON, a zero latent coefficient must
        // leave Ts identical to the gate-off run — the feedback is the only path
        // by which the film reaches the budget.
        var zeroCoeff = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0, SurfaceWaterEnabled = true, LatentHeatWm2PerMmHr = 0.0 };
        var off = new RockSurfaceConfig { MaxSurfaceWaterMm = 5.0, SurfaceWaterEnabled = false };
        var forcing = MakeForcingWithMorningRain(6, rainMm: 3.0);

        var rZero = Integrate(forcing, zeroCoeff);
        var rOff = Integrate(forcing, off);
        for (var i = 0; i < rZero.Count; i++)
            rZero[i].RockTempC.Should().BeApproximately(rOff[i].RockTempC, 1e-9);
    }
}
