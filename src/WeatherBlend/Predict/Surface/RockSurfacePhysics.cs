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
///   G_net = (1−α)·SW↓ + ε·Fsky·(LW↓ − σ·Ts⁴) − h(V)·(Ts − Ta)
/// with h(V) = 5.7 + 3.8·V (McAdams) and LW↓ a Brutsaert clear-sky + linear
/// cloud-enhancement parameterisation (GFS-calibrated k_cloud=0.54).
/// </summary>
public static class RockSurfacePhysics
{
    /// <summary>Stefan–Boltzmann constant (W/m²/K⁴).</summary>
    public const double Sigma = 5.670374419e-8;

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

    /// <summary>One hour of forcing for the integrator.</summary>
    public sealed record ForcingHour(
        DateTime ValidTimeUtc, double AirTempC, double DewPointC,
        double CloudFrac, double WindMs, double ShortwaveWm2);

    /// <summary>One integrated hour: rock surface + deep reservoir temps, the
    /// effective LW↓ used, and the air/dew it was driven by.</summary>
    public sealed record RockHour(
        DateTime ValidTimeUtc, double RockTempC, double DeepTempC,
        double LongwaveDownWm2, double AirTempC, double DewPointC)
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
    /// </summary>
    public static IReadOnlyList<RockHour> Integrate(IReadOnlyList<ForcingHour> forcing, RockSurfaceConfig p)
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

        var outp = new List<RockHour>(n);
        foreach (var h in forcing)
        {
            var sw = Math.Max(h.ShortwaveWm2, 0.0);
            var v = Math.Max(h.WindMs, 0.0);
            var lwDown = LongwaveDownWm2(h.DewPointC, h.AirTempC, h.CloudFrac, p.LwClearK, p.LwCloudK);
            var hConv = 5.7 + 3.8 * v;

            for (var s = 0; s < sub; s++)
            {
                var tsK = tsC + 273.15;
                var swAbs = (1.0 - p.Albedo) * sw;
                var lwNet = p.EpsRock * p.FSky * (lwDown - Sigma * tsK * tsK * tsK * tsK);
                var hLoss = hConv * (tsC - h.AirTempC);
                var gNet = swAbs + lwNet - hLoss;
                var dts = gNet / mu - twoPiTau * (tsC - tdeepC);
                var dtd = (tsC - tdeepC) * invTauLong;
                tsC += dts * dt;
                tdeepC += dtd * dt;
            }

            outp.Add(new RockHour(h.ValidTimeUtc, tsC, tdeepC, lwDown, h.AirTempC, h.DewPointC));
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
