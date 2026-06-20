using WeatherBlend.Config;

namespace WeatherBlend.Predict.Surface;

/// <summary>
/// Granite surface-temperature physics — a faithful C# port of the validated P0
/// Force-Restore spike (scripts/rock_temp_spike.py). Pure + static so the
/// physical-behaviour checks (clear-calm night cools below air, sunny day runs
/// above, wind/cloud couple toward air, diurnal swing wider than air) are
/// unit-testable on synthetic forcing without any I/O.
///
/// Model (docs/ROCK_SURFACE_TEMP_PLAN.md §2):
///   dTs/dt    =  G_net / μ  − (2π/τ)·(Ts − Td_deep)
///   dTd_deep/dt = (Ts − Td_deep) / τ_long
///   G_net = (1−α)·SW↓ + ε·Fsky·(LW↓ − σ·Ts⁴) − h(V)·(Ts − Ta)  [+ sea term]
/// with h(V) = 5.7 + 3.8·V (McAdams) and LW↓ a Brutsaert clear-sky + linear
/// cloud-enhancement parameterisation (GFS-calibrated k_cloud=0.54).
///
/// SEA LONGWAVE (docs/SENNEN_ROCK_TEMP_PLAN.md S4): with Fsky &lt; 1 the
/// non-sky view fraction is implicitly exchange-neutral — right for a
/// boulder field (neighbours sit near Ts), wrong for a wall above the
/// Atlantic, where ~half the view is sea: a warm radiator that returns
/// longwave the open sky does not, strongly damping night cooling. When a
/// forcing hour carries a sea temperature, the budget gains
///   ε·(1−Fsky)·σ·(T_sea⁴ − Ts⁴)
/// (sea emissivity ~0.98 folded into ε to one knob). Null sea temp =
/// neutral surroundings, the exact pre-S4 behaviour.
/// </summary>
public static class RockSurfacePhysics
{
    /// <summary>Stefan–Boltzmann constant (W/m²/K⁴).</summary>
    public const double Sigma = 5.670374419e-8;

    // --- Gravity-drainage (Jeffreys 1930 / Nusselt thin-film) constants ---
    /// <summary>Water density (kg/m³) for the gravity-drainage flux.</summary>
    private const double WaterDensityKgM3 = 1000.0;
    /// <summary>Gravitational acceleration (m/s²).</summary>
    private const double GravityMS2 = 9.81;
    /// <summary>Dynamic viscosity of water (Pa·s) at a cool ~10 °C wet-rock
    /// temperature. The drainage term is fast across any plausible value, so this
    /// is not sensitive.</summary>
    private const double WaterViscosityPaS = 1.3e-3;

    /// <summary>Magnus vapour pressure (hPa) from dew point (°C).</summary>
    public static double VapourPressureHpa(double dewPointC)
        => 6.112 * Math.Exp(17.62 * dewPointC / (243.12 + dewPointC));

    /// <summary>Brutsaert (1975) clear-sky emissivity. <paramref name="airTempK"/> in kelvin.</summary>
    public static double ClearSkyEmissivity(double dewPointC, double airTempK)
        => 1.24 * Math.Pow(VapourPressureHpa(dewPointC) / airTempK, 1.0 / 7.0);

    /// <summary>Effective sky emissivity = clip(k_clear·εclear + k_cloud·(1−εclear)·C, 0, 1).</summary>
    public static double SkyEmissivity(double dewPointC, double airTempC, double cloudFrac, double kClear, double kCloud)
    {
        var airK = airTempC + 273.15;
        var eClear = ClearSkyEmissivity(dewPointC, airK);
        var c = Math.Clamp(cloudFrac, 0.0, 1.0);
        return Math.Clamp(kClear * eClear + kCloud * (1.0 - eClear) * c, 0.0, 1.0);
    }

    /// <summary>Downwelling longwave (W/m²) = εsky·σ·Ta⁴.</summary>
    public static double LongwaveDownWm2(double dewPointC, double airTempC, double cloudFrac, double kClear, double kCloud)
    {
        var airK = airTempC + 273.15;
        return SkyEmissivity(dewPointC, airTempC, cloudFrac, kClear, kCloud) * Sigma * airK * airK * airK * airK;
    }

    /// <summary>
    /// Canonical force-restore areal heat capacity Cg = √(λ·ρ·c / (2ω)),
    /// ω = 2π/τ_day (Deardorff 1978 / Hu &amp; Islam 1995), scaled by
    /// <see cref="RockSurfaceConfig.MuScale"/> — the effective radiating skin is
    /// thinner than the full diurnal-damping depth, so a scale &lt;1 lets the
    /// surface cool below air at night.
    /// </summary>
    public static double MuFromProps(RockSurfaceConfig p)
    {
        var omega = 2.0 * Math.PI / p.TauDaySeconds;
        return Math.Sqrt(p.Lambda * p.Rho * p.CpRock / (2.0 * omega)) * p.MuScale;
    }

    /// <summary>
    /// Whole-crag horizontal shortwave with the DIRECT beam damped by a
    /// representative factor (boulder self-shading / steep climbing faces rarely
    /// catch the full noon beam). Diffuse is untouched and NO aspect is assumed:
    /// SW_eff = diffuse + factor·direct = SW·(1 − directFrac·(1 − factor)).
    /// <paramref name="directBeamFactor"/> = 1 → unchanged; 0 → diffuse only.
    /// <paramref name="directFrac"/> (the direct share of horizontal SW) comes
    /// from the NWP radiation split. Cliff-face mode does NOT use this — its
    /// per-face projection handles the beam geometry directly.
    /// </summary>
    public static double DampedHorizontalSw(double swHorizontalWm2, double directFrac, double directBeamFactor)
    {
        var df = Math.Clamp(directFrac, 0.0, 1.0);
        var direct = swHorizontalWm2 * df;
        return swHorizontalWm2 - direct * (1.0 - directBeamFactor);
    }

    /// <summary>One hour of forcing for the integrator. <paramref name="SeaTempC"/>
    /// is the sea surface temperature filling the rock's non-sky view (sea-cliff
    /// mode); null = exchange-neutral surroundings (boulder field / pre-S4).</summary>
    public sealed record ForcingHour(
        DateTime ValidTimeUtc, double AirTempC, double DewPointC,
        double CloudFrac, double WindMs, double ShortwaveWm2, double? SeaTempC = null,
        double PrecipMm = 0.0);

    /// <summary>Hourly precipitation (mm) at/above which an hour counts as the
    /// "wet event" that <see cref="RockHour.LastRainAtUtc"/> records — a noticeable
    /// shower, not a trace. The film itself accrues from ANY precip; this only
    /// gates the human-facing "wet from rain since HHZ" timestamp.</summary>
    public const double RainEventThresholdMm = 0.1;

    /// <summary>One integrated hour: rock surface + deep reservoir temps, the
    /// effective LW↓ used, the air/dew it was driven by, and the surface-water
    /// film (Phase A drying model). <paramref name="SurfaceWaterMm"/> is the
    /// total film (rain + dew); <paramref name="RainWaterMm"/> is the rain-derived
    /// part, so a consumer can tell "wet from rain" from "wet from dew".
    /// <paramref name="LastRainAtUtc"/> is the most recent hour (including the
    /// hidden spin-up) with rain ≥ <see cref="RainEventThresholdMm"/> — so the site
    /// can say WHEN the rock got wet even when the wetting happened before the
    /// reported window; null = no rain seen yet this trajectory.</summary>
    public sealed record RockHour(
        DateTime ValidTimeUtc, double RockTempC, double DeepTempC,
        double LongwaveDownWm2, double AirTempC, double DewPointC,
        double SurfaceWaterMm = 0.0, double RainWaterMm = 0.0,
        DateTime? LastRainAtUtc = null)
    {
        /// <summary>Condensation margin m = Ts − Td. ≤ 0 ⇒ condensation.</summary>
        public double MarginC => RockTempC - DewPointC;
    }

    /// <summary>
    /// March the Force-Restore ODE forward over a CONTIGUOUS hourly forcing
    /// series (Euler, <see cref="RockSurfaceConfig.Substeps"/> sub-steps/hour).
    /// Seeds Ts = first hour's air temp and the deep reservoir = mean of the
    /// first 24 h of air temp (matches the spike). Caller is responsible for
    /// prepending spin-up hours and slicing reported hours off the result.
    /// <para><paramref name="drainageSlopeSine"/> = sin(face steepness): 1 for a
    /// vertical wall (the rain film sheds the bulk under gravity within ~an hour,
    /// Jeffreys/Nusselt), 0 for a horizontal slab (no drainage — it ponds, which
    /// is the right behaviour and the legacy default). Whole-crag/horizontal
    /// callers leave it 0.</para>
    /// </summary>
    public static IReadOnlyList<RockHour> Integrate(
        IReadOnlyList<ForcingHour> forcing, RockSurfaceConfig p, double drainageSlopeSine = 0.0)
    {
        var n = forcing.Count;
        if (n == 0) return Array.Empty<RockHour>();

        var mu = MuFromProps(p);
        var sub = Math.Max(1, p.Substeps);
        var dt = 3600.0 / sub;
        var twoPiTau = 2.0 * Math.PI / p.TauDaySeconds;
        var invTauLong = 1.0 / p.TauLongSeconds;

        var tsC = forcing[0].AirTempC;
        var seedCount = Math.Min(24, n);
        var tdeepC = forcing.Take(seedCount).Average(h => h.AirTempC);

        // Surface-water film (mm), tracked by source so the climbing index can
        // tell rain-wet (hard gate) from dew-wet (friction). The film always
        // accrues for inspection; its latent-heat feedback on Ts (Phase B) is
        // applied ONLY when SurfaceWaterEnabled — so a gate-off location's Ts is
        // bit-for-bit the pre-drying value.
        double wRain = 0.0, wDew = 0.0;
        // Most recent hour with a meaningful shower — carried forward (incl. spin-up)
        // so each emitted hour knows when the rock last got rained on.
        DateTime? lastRainAt = null;
        var eAirCache = VapourPressureHpa(forcing[0].DewPointC);

        var outp = new List<RockHour>(n);
        foreach (var h in forcing)
        {
            var sw = Math.Max(h.ShortwaveWm2, 0.0);
            var v = Math.Max(h.WindMs, 0.0);
            var lwDown = LongwaveDownWm2(h.DewPointC, h.AirTempC, h.CloudFrac, p.LwClearK, p.LwCloudK);
            var hConv = 5.7 + 3.8 * v;
            var windFn = p.EvapWindBase + p.EvapWindSlope * v;
            // Actual vapour pressure = saturation vp at the dew point (air is
            // saturated there). Surface saturation vp uses Ts each substep.
            eAirCache = VapourPressureHpa(h.DewPointC);

            // Sea longwave: σ·T_sea⁴ once per hour (SST moves slowly).
            var seaEmitWm2 = h.SeaTempC is double seaC
                ? Sigma * Math.Pow(seaC + 273.15, 4)
                : (double?)null;

            // Rain wets the surface at the top of the hour (mm this hour).
            wRain += Math.Max(h.PrecipMm, 0.0);
            if (h.PrecipMm >= RainEventThresholdMm) lastRainAt = h.ValidTimeUtc;

            for (var s = 0; s < sub; s++)
            {
                var tsK = tsC + 273.15;
                var tsK4 = tsK * tsK * tsK * tsK;
                var swAbs = (1.0 - p.Albedo) * sw;
                var lwNet = p.EpsRock * p.FSky * (lwDown - Sigma * tsK4);
                if (seaEmitWm2 is double seaEmit)
                    lwNet += p.EpsRock * (1.0 - p.FSky) * (seaEmit - Sigma * tsK4);
                var hLoss = hConv * (tsC - h.AirTempC);

                // --- Surface-water moisture flux for THIS substep, off the
                //     CURRENT Ts: VPD > 0 dries the film, VPD < 0 deposits dew,
                //     plus a radiation-driven drying term while drying. Evaporation
                //     is capped at the water actually present (can't dry more film
                //     than there is). ---
                var vpd = VapourPressureHpa(tsC) - eAirCache;   // hPa, >0 dry / <0 condensing
                var fluxMmHr = p.EvapVpdCoeff * windFn * vpd
                             + (vpd > 0.0 ? p.EvapRadCoeff * swAbs : 0.0);
                var deltaMm = fluxMmHr * (dt / 3600.0);          // signed mm this substep
                var wTot = wRain + wDew;
                var actualDeltaMm = deltaMm > 0.0 ? Math.Min(deltaMm, wTot) : deltaMm;

                // Latent-heat feedback (Phase B): evaporating water cools the slab,
                // condensing dew warms it. Gated on SurfaceWaterEnabled so a
                // gate-off location's Ts is bit-for-bit the pre-drying value.
                // actualDeltaMm is per substep → ÷(dt/3600) back to mm/hr.
                var latentWm2 = p.SurfaceWaterEnabled
                    ? p.LatentHeatWm2PerMmHr * (actualDeltaMm / (dt / 3600.0))
                    : 0.0;

                var gNet = swAbs + lwNet - hLoss - latentWm2;
                var dts = gNet / mu - twoPiTau * (tsC - tdeepC);
                var dtd = (tsC - tdeepC) * invTauLong;
                tsC += dts * dt;
                tdeepC += dtd * dt;

                // Apply the moisture exchange to the bucket (evaporation removes
                // proportionally by source so the rain/dew split is preserved).
                if (actualDeltaMm > 0.0)
                {
                    if (wTot > 1e-9)
                    {
                        var frac = actualDeltaMm / wTot;
                        wRain -= wRain * frac;
                        wDew  -= wDew * frac;
                    }
                }
                else
                {
                    wDew += -actualDeltaMm;   // dew deposits
                }
                // Runoff cap — most water a face holds even instantaneously, before
                // gravity drainage acts (a backstop; drainage below does the real
                // shedding on a sloped face).
                var total = wRain + wDew;
                if (total > p.MaxSurfaceWaterMm && total > 1e-9)
                {
                    var k = p.MaxSurfaceWaterMm / total;
                    wRain *= k; wDew *= k;
                }

                // --- Gravity drainage (Jeffreys 1930 / Nusselt thin-film): on a
                //     SLOPED face the film above the retained residual sheds under
                //     gravity. The film flux is the Nusselt result q = ρ·g·sinθ·h³/3μ;
                //     for a lumped streak of length L that gives dh/dt = -k·h³ with
                //     k = ρ·g·sinθ/(3·μ·L). Integrated ANALYTICALLY over the substep
                //     (h(t) = h₀/√(1 + 2k·h₀²·t)) so it's stable at this dt — explicit
                //     Euler on the stiff h³ term would overshoot wildly. Acts on the
                //     drainable EXCESS above RetainedFilmMm; a vertical wall drains the
                //     bulk in minutes, a horizontal slab (sinθ=0) doesn't drain at all
                //     (it ponds — evaporation only). Removes proportionally by source.
                //     Evaporation (above) still clears the retained residual to zero. ---
                if (drainageSlopeSine > 0.0)
                {
                    var excessMm = wRain + wDew - p.RetainedFilmMm;
                    if (excessMm > 0.0)
                    {
                        var excessM = excessMm * 1e-3;
                        var kDrain = WaterDensityKgM3 * GravityMS2 * drainageSlopeSine
                                     / (3.0 * WaterViscosityPaS * Math.Max(p.DrainagePathLengthM, 1e-3));
                        var drainedM = excessM / Math.Sqrt(1.0 + 2.0 * kDrain * excessM * excessM * dt);
                        var removedMm = (excessM - drainedM) * 1e3;
                        var tot = wRain + wDew;
                        if (removedMm > 0.0 && tot > 1e-9)
                        {
                            var frac = removedMm / tot;
                            wRain -= wRain * frac;
                            wDew  -= wDew * frac;
                        }
                    }
                }
            }

            outp.Add(new RockHour(h.ValidTimeUtc, tsC, tdeepC, lwDown, h.AirTempC, h.DewPointC,
                SurfaceWaterMm: wRain + wDew, RainWaterMm: wRain, LastRainAtUtc: lastRainAt));
        }
        return outp;
    }

    /// <summary>Greasiness tier for a condensation margin (°C):
    /// <c>"condensation"</c> (m ≤ 0), <c>"potentially_greasy"</c>
    /// (0 &lt; m ≤ <paramref name="greasyMarginC"/>), else <c>"dry"</c>.</summary>
    public static string Greasiness(double marginC, double greasyMarginC)
        => marginC <= 0.0 ? "condensation"
         : marginC <= greasyMarginC ? "potentially_greasy"
         : "dry";

    /// <summary>The three greasiness status strings (stable, for portable
    /// parquet readers + the renderer).</summary>
    public const string StatusCondensation = "condensation";
    public const string StatusPotentiallyGreasy = "potentially_greasy";
    public const string StatusDry = "dry";
}
