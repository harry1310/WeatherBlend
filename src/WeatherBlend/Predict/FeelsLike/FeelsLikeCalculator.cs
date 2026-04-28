namespace WeatherBlend.Predict.FeelsLike;

/// <summary>
/// "Feels-like" temperature calculators plus the supporting helpers they need:
/// vapour pressure (RH + Ta), longwave-down (cloud + Ta), mean radiant
/// temperature (shortwave-down + cloud + Ta).
///
/// Two indices live here:
///   <see cref="Utci"/> — the rigorous Bröde 2012 6th-order regression. Uses
///     Tmrt so it responds to radiation; the gold-standard biothermal index
///     used in academic comfort/heat-stress work.
///   <see cref="Steadman1994"/> — the simpler shade-form apparent temperature
///     behind BoM's published "AT" and the BBC's public "feels like" chip.
///     One-line formula, no radiation term, gentler on wind chill.
///
/// Tmrt is the radiation-balance approximation popularised by Thorsson et al.
/// 2007 / Höppe 1999: a person in radiative equilibrium with sky + ground.
/// Inputs come from our four element blenders (radiation, cloud, humidity,
/// wind) plus the temperature blender. The MRT step is what turns UTCI from
/// "Ta + wind chill + humidity" into a real outdoor-comfort number that
/// responds to clear-sky midday sun (Tmrt &gt;&gt; Ta) versus overcast
/// conditions (Tmrt ≈ Ta).
///
/// All UTCI polynomial coefficients are taken verbatim from the official
/// UTCI_approx 6th-order regression (Bröde et al. 2012; reference Visual
/// Basic implementation at utci.org). Validity range:
///   Ta ∈ [-50, +50] °C, va10 ∈ [0.5, 17] m/s, Tmrt-Ta ∈ [-30, +70] K.
/// Outside this domain the polynomial extrapolates and should not be trusted.
/// </summary>
public static class FeelsLikeCalculator
{
    private const double SigmaSb = 5.670374419e-8;     // Stefan-Boltzmann, W/m²/K⁴
    private const double EmissivityHuman = 0.97;        // αl = εp
    private const double AbsorptivityShortwave = 0.7;   // αk for human body
    private const double AlbedoSurface = 0.15;          // grass / short vegetation
    private const double EmissivityGround = 0.95;
    private const double WindReductionFactor10mTo1m1 = 0.6809; // log(1.1/0.01)/log(10/0.01)

    /// <summary>Lumped "human shortwave coefficient" applied to total
    /// horizontal Kdown so the calculator can run without a direct/diffuse
    /// split or solar-zenith input. Calibrated against published Tmrt curves
    /// for an upright unobstructed person (Höppe 1999 / Lindberg et al. 2008
    /// SOLWEIG): clear-sky midday in summer should give Tmrt ≈ Ta + 25–30 K.
    /// At αk=0.7 with hemisphere-weighted longwave (0.5 each to sky/ground),
    /// 0.30 reproduces that response while staying physically defensible
    /// (≈ αk · (0.25·fdir + 0.5·fdiff + 0.5·αs) for fdir=fdiff=0.5, αs=0.15).</summary>
    private const double HumanShortwaveCoefficient = 0.30;

    /// <summary>Saturation vapour pressure (hPa) over liquid water from Wexler/Hardy
    /// expansion — the same used by the official UTCI reference code.</summary>
    public static double SaturationVapourPressureHpa(double tempC)
    {
        double tk = tempC + 273.15;
        double[] g = {
            -2836.5744, -6028.076559, 19.54263612, -0.02737830188,
            0.000016261698, 7.0229056e-10, -1.8680009e-13,
        };
        double e = 2.7150305 * Math.Log(tk);
        for (int i = 0; i < 7; i++)
            e += g[i] * Math.Pow(tk, i - 2);
        return Math.Exp(e) * 0.01;  // Pa → hPa
    }

    /// <summary>Actual vapour pressure (hPa) from temperature and relative
    /// humidity. UTCI's polynomial expects vapour pressure in hPa.</summary>
    public static double VapourPressureHpa(double tempC, double rhPercent)
        => SaturationVapourPressureHpa(tempC) * (rhPercent / 100.0);

    /// <summary>Atmospheric (downwelling) longwave radiation [W/m²] using
    /// Brutsaert clear-sky emissivity boosted by cloud cover (Crawford &amp;
    /// Duchon 1999). cloudFrac is 0..1.</summary>
    public static double LongwaveDownWm2(double tempC, double cloudFrac, double vapourPressureHpa)
    {
        double tk = tempC + 273.15;
        // Brutsaert 1975: εclear = 1.24 * (e/T)^(1/7), e in hPa, T in K.
        double eClear = 1.24 * Math.Pow(Math.Max(vapourPressureHpa, 0.01) / tk, 1.0 / 7.0);
        // Crawford-Duchon: εeff = (1 - cc)*εclear + cc.
        double cc = Math.Clamp(cloudFrac, 0.0, 1.0);
        double eEff = (1.0 - cc) * eClear + cc;
        return eEff * SigmaSb * tk * tk * tk * tk;
    }

    /// <summary>Surface (upwelling) longwave radiation [W/m²] assuming
    /// ground temperature equals air temperature (reasonable assumption
    /// during day; weak at night under clear sky — but the SW=0 + cloud-
    /// modulated Ldown handles the night case in aggregate).</summary>
    public static double LongwaveUpWm2(double tempC)
    {
        double tk = tempC + 273.15;
        return EmissivityGround * SigmaSb * tk * tk * tk * tk;
    }

    /// <summary>Mean radiant temperature (°C) from a hemisphere-weighted
    /// radiation balance for an upright person (Thorsson et al. 2007 /
    /// Lindberg et al. 2008 simplified form):
    ///   Sstr = K_human·Kdown + 0.5·εp·(Ldown + Lup)
    /// The longwave term uses 0.5 view factors to sky and ground (an upright
    /// person sees half-sky, half-ground); the shortwave term lumps direct +
    /// diffuse + ground-reflected into <see cref="HumanShortwaveCoefficient"/>
    /// so the calculator can run on just total Kdown, no solar geometry.
    /// Validated against typical observed Tmrt:
    ///   overcast Ta=25 sw≈100 → Tmrt ≈ Ta + 3 K
    ///   clear-sky Ta=25 sw≈800 → Tmrt ≈ Ta + 30 K (matches outdoor globe-
    ///                                              thermometer studies).</summary>
    public static double Tmrt(double tempC, double shortwaveDownWm2, double cloudFrac, double rhPercent)
    {
        double pVap = VapourPressureHpa(tempC, rhPercent);
        double sw = Math.Max(shortwaveDownWm2, 0.0);
        double lDown = LongwaveDownWm2(tempC, cloudFrac, pVap);
        double lUp = LongwaveUpWm2(tempC);
        double sStr = HumanShortwaveCoefficient * sw
                    + 0.5 * EmissivityHuman * (lDown + lUp);
        double tmrtK = Math.Pow(sStr / (EmissivityHuman * SigmaSb), 0.25);
        return tmrtK - 273.15;
    }

    /// <summary>Reduce 10 m wind speed to 1.1 m using a log law over short
    /// grass (z0 = 0.01 m). Multiplier ≈ 0.68. Provided for callers who
    /// want the reduced wind for display — UTCI's polynomial takes the
    /// 10 m value directly (the regression was fit on it).</summary>
    public static double WindAt1m1(double wind10mMs) => wind10mMs * WindReductionFactor10mTo1m1;

    /// <summary>
    /// Steadman 1994 outdoor "apparent temperature" (°C) — the shade-comfort form
    /// behind BoM's published AT and the BBC's "feels like":
    ///   AT = Ta + 0.33·e − 0.70·ws − 4.00
    /// where <paramref name="vapourPressureHpa"/> is the actual vapour pressure
    /// (e, hPa) and <paramref name="wind10mMs"/> is the 10 m wind speed (m/s).
    /// Companion to <see cref="Utci"/>: gentler on wind chill than UTCI's 6th-order
    /// regression, no radiation term (1994 is the shade form). Used as the
    /// publicly-recognisable feels-like number alongside UTCI on the home cards.
    /// </summary>
    public static double Steadman1994(double tempC, double vapourPressureHpa, double wind10mMs)
        => tempC + 0.33 * vapourPressureHpa - 0.70 * wind10mMs - 4.0;

    /// <summary>Universal Thermal Climate Index (°C) — the Bröde 2012 6th-
    /// order regression. Inputs:
    ///   ta   — air temperature, °C
    ///   pHpa — vapour pressure, hPa
    ///   va10 — 10 m wind speed, m/s
    ///   tmrt — mean radiant temperature, °C.</summary>
    public static double Utci(double ta, double pHpa, double va10, double tmrt)
    {
        double dTmrt = tmrt - ta;
        double pa = pHpa / 10.0; // polynomial uses kPa
        double va = va10;

        // Common power monomials used by the polynomial.
        double ta2 = ta * ta, ta3 = ta2 * ta, ta4 = ta3 * ta, ta5 = ta4 * ta, ta6 = ta5 * ta;
        double va2 = va * va, va3 = va2 * va, va4 = va3 * va, va5 = va4 * va, va6 = va5 * va;
        double dt2 = dTmrt * dTmrt, dt3 = dt2 * dTmrt, dt4 = dt3 * dTmrt, dt5 = dt4 * dTmrt, dt6 = dt5 * dTmrt;
        double pa2 = pa * pa, pa3 = pa2 * pa, pa4 = pa3 * pa, pa5 = pa4 * pa, pa6 = pa5 * pa;

        double u = ta
            + 6.07562052e-01
            + -2.27712343e-02 * ta
            + 8.06470249e-04 * ta2
            + -1.54271372e-04 * ta3
            + -3.24651735e-06 * ta4
            + 7.32602852e-08 * ta5
            + 1.35959073e-09 * ta6
            + -2.25836520e+00 * va
            + 8.80326035e-02 * ta * va
            + 2.16844454e-03 * ta2 * va
            + -1.53347087e-05 * ta3 * va
            + -5.72983704e-07 * ta4 * va
            + -2.55090145e-09 * ta5 * va
            + -7.51269505e-01 * va2
            + -4.08350271e-03 * ta * va2
            + -5.21670675e-05 * ta2 * va2
            + 1.94544667e-06 * ta3 * va2
            + 1.14099531e-08 * ta4 * va2
            + 1.58137256e-01 * va3
            + -6.57263143e-05 * ta * va3
            + 2.22697524e-07 * ta2 * va3
            + -4.16117031e-08 * ta3 * va3
            + -1.27762753e-02 * va4
            + 9.66891875e-06 * ta * va4
            + 2.52785852e-09 * ta2 * va4
            + 4.56306672e-04 * va5
            + -1.74202546e-07 * ta * va5
            + -5.91491269e-06 * va6
            + 3.98374029e-01 * dTmrt
            + 1.83945314e-04 * ta * dTmrt
            + -1.73754510e-04 * ta2 * dTmrt
            + -7.60781159e-07 * ta3 * dTmrt
            + 3.77830287e-08 * ta4 * dTmrt
            + 5.43079673e-10 * ta5 * dTmrt
            + -2.00518269e-02 * va * dTmrt
            + 8.92859837e-04 * ta * va * dTmrt
            + 3.45433048e-06 * ta2 * va * dTmrt
            + -3.77925774e-07 * ta3 * va * dTmrt
            + -1.69699377e-09 * ta4 * va * dTmrt
            + 1.69992415e-04 * va2 * dTmrt
            + -4.99204314e-05 * ta * va2 * dTmrt
            + 2.47417178e-07 * ta2 * va2 * dTmrt
            + 1.07596466e-08 * ta3 * va2 * dTmrt
            + 8.49242932e-05 * va3 * dTmrt
            + 1.35191328e-06 * ta * va3 * dTmrt
            + -6.21531254e-09 * ta2 * va3 * dTmrt
            + -4.99410301e-06 * va4 * dTmrt
            + -1.89489258e-08 * ta * va4 * dTmrt
            + 8.15300114e-08 * va5 * dTmrt
            + 7.55043090e-04 * dt2
            + -5.65095215e-05 * ta * dt2
            + -4.52166564e-07 * ta2 * dt2
            + 2.46688878e-08 * ta3 * dt2
            + 2.42674348e-10 * ta4 * dt2
            + 1.54547250e-04 * va * dt2
            + 5.24110970e-06 * ta * va * dt2
            + -8.75874982e-08 * ta2 * va * dt2
            + -1.50743064e-09 * ta3 * va * dt2
            + -1.56236307e-05 * va2 * dt2
            + -1.33895614e-07 * ta * va2 * dt2
            + 2.49709824e-09 * ta2 * va2 * dt2
            + 6.51711721e-07 * va3 * dt2
            + 1.94960053e-09 * ta * va3 * dt2
            + -1.00361113e-08 * va4 * dt2
            + -1.21206673e-05 * dt3
            + -2.18203660e-07 * ta * dt3
            + 7.51269482e-09 * ta2 * dt3
            + 9.79063848e-11 * ta3 * dt3
            + 1.25006734e-06 * va * dt3
            + -1.81584736e-09 * ta * va * dt3
            + -3.52197671e-10 * ta2 * va * dt3
            + -3.36514630e-08 * va2 * dt3
            + 1.35908359e-10 * ta * va2 * dt3
            + 4.17032620e-10 * va3 * dt3
            + -1.30369025e-09 * dt4
            + 4.13908461e-10 * ta * dt4
            + 9.22652254e-12 * ta2 * dt4
            + -5.08220384e-09 * va * dt4
            + -2.24730961e-11 * ta * va * dt4
            + 1.17139133e-10 * va2 * dt4
            + 6.62154879e-10 * dt5
            + 4.03863260e-13 * ta * dt5
            + 1.95087203e-12 * va * dt5
            + -4.73602469e-12 * dt6
            + 5.12733497e+00 * pa
            + -3.12788561e-01 * ta * pa
            + -1.96701861e-02 * ta2 * pa
            + 9.99690870e-04 * ta3 * pa
            + 9.51738512e-06 * ta4 * pa
            + -4.66426341e-07 * ta5 * pa
            + 5.48050612e-01 * va * pa
            + -3.30552823e-03 * ta * va * pa
            + -1.64119440e-03 * ta2 * va * pa
            + -5.16670694e-06 * ta3 * va * pa
            + 9.52692432e-07 * ta4 * va * pa
            + -4.29223622e-02 * va2 * pa
            + 5.00845667e-03 * ta * va2 * pa
            + 1.00601257e-06 * ta2 * va2 * pa
            + -1.81748644e-06 * ta3 * va2 * pa
            + -1.25813502e-03 * va3 * pa
            + -1.79330391e-04 * ta * va3 * pa
            + 2.34994441e-06 * ta2 * va3 * pa
            + 1.29735808e-04 * va4 * pa
            + 1.29064870e-06 * ta * va4 * pa
            + -2.28558686e-06 * va5 * pa
            + -3.69476348e-02 * dTmrt * pa
            + 1.62325322e-03 * ta * dTmrt * pa
            + -3.14279680e-05 * ta2 * dTmrt * pa
            + 2.59835559e-06 * ta3 * dTmrt * pa
            + -4.77136523e-08 * ta4 * dTmrt * pa
            + 8.64203390e-03 * va * dTmrt * pa
            + -6.87405181e-04 * ta * va * dTmrt * pa
            + -9.13863872e-06 * ta2 * va * dTmrt * pa
            + 5.15916806e-07 * ta3 * va * dTmrt * pa
            + -3.59217476e-05 * va2 * dTmrt * pa
            + 3.28696511e-05 * ta * va2 * dTmrt * pa
            + -7.10542454e-07 * ta2 * va2 * dTmrt * pa
            + -1.24382300e-05 * va3 * dTmrt * pa
            + -7.38584400e-09 * ta * va3 * dTmrt * pa
            + 2.20609296e-07 * va4 * dTmrt * pa
            + -7.32469180e-04 * dt2 * pa
            + -1.87381964e-05 * ta * dt2 * pa
            + 4.80925239e-06 * ta2 * dt2 * pa
            + -8.75492040e-08 * ta3 * dt2 * pa
            + 2.77862930e-05 * va * dt2 * pa
            + -5.06004592e-06 * ta * va * dt2 * pa
            + 1.14325367e-07 * ta2 * va * dt2 * pa
            + 2.53016723e-06 * va2 * dt2 * pa
            + -1.72857035e-08 * ta * va2 * dt2 * pa
            + -3.95079398e-08 * va3 * dt2 * pa
            + -3.59413173e-07 * dt3 * pa
            + 7.04388046e-07 * ta * dt3 * pa
            + -1.89309167e-08 * ta2 * dt3 * pa
            + -4.79768731e-07 * va * dt3 * pa
            + 7.96079978e-09 * ta * va * dt3 * pa
            + 1.62897058e-09 * va2 * dt3 * pa
            + 3.94367674e-08 * dt4 * pa
            + -1.18566247e-09 * ta * dt4 * pa
            + 3.34678041e-10 * va * dt4 * pa
            + -1.15606447e-10 * dt5 * pa
            + -2.80626406e+00 * pa2
            + 5.48712484e-01 * ta * pa2
            + -3.99428410e-03 * ta2 * pa2
            + -9.54009191e-04 * ta3 * pa2
            + 1.93090978e-05 * ta4 * pa2
            + -3.08806365e-01 * va * pa2
            + 1.16952364e-02 * ta * va * pa2
            + 4.95271903e-04 * ta2 * va * pa2
            + -1.90710882e-05 * ta3 * va * pa2
            + 2.10787756e-03 * va2 * pa2
            + -6.98445738e-04 * ta * va2 * pa2
            + 2.30109073e-05 * ta2 * va2 * pa2
            + 4.17856590e-04 * va3 * pa2
            + -1.27043871e-05 * ta * va3 * pa2
            + -3.04620472e-06 * va4 * pa2
            + 5.14507424e-02 * dTmrt * pa2
            + -4.32510997e-03 * ta * dTmrt * pa2
            + 8.99281156e-05 * ta2 * dTmrt * pa2
            + -7.14663943e-07 * ta3 * dTmrt * pa2
            + -2.66016305e-04 * va * dTmrt * pa2
            + 2.63789586e-04 * ta * va * dTmrt * pa2
            + -7.01199003e-06 * ta2 * va * dTmrt * pa2
            + -1.06823306e-04 * va2 * dTmrt * pa2
            + 3.61341136e-06 * ta * va2 * dTmrt * pa2
            + 2.29748967e-07 * va3 * dTmrt * pa2
            + 3.04788893e-04 * dt2 * pa2
            + -6.42070836e-05 * ta * dt2 * pa2
            + 1.16257971e-06 * ta2 * dt2 * pa2
            + 7.68023384e-06 * va * dt2 * pa2
            + -5.47446896e-07 * ta * va * dt2 * pa2
            + -3.59937910e-08 * va2 * dt2 * pa2
            + -4.36497725e-06 * dt3 * pa2
            + 1.68737969e-07 * ta * dt3 * pa2
            + 2.67489271e-08 * va * dt3 * pa2
            + 3.23926897e-09 * dt4 * pa2
            + -3.53874123e-02 * pa3
            + -2.21201190e-01 * ta * pa3
            + 1.55126038e-02 * ta2 * pa3
            + -2.63917279e-04 * ta3 * pa3
            + 4.53433455e-02 * va * pa3
            + -4.32943862e-03 * ta * va * pa3
            + 1.45389826e-04 * ta2 * va * pa3
            + 2.17508610e-04 * va2 * pa3
            + -6.66724702e-05 * ta * va2 * pa3
            + 3.33217140e-05 * va3 * pa3
            + -2.26921615e-03 * dTmrt * pa3
            + 3.80261982e-04 * ta * dTmrt * pa3
            + -5.45314314e-09 * ta2 * dTmrt * pa3
            + -7.96355448e-04 * va * dTmrt * pa3
            + 2.53458034e-05 * ta * va * dTmrt * pa3
            + -6.31223658e-06 * va2 * dTmrt * pa3
            + 3.02122035e-04 * dt2 * pa3
            + -4.77403547e-06 * ta * dt2 * pa3
            + 1.73825715e-06 * va * dt2 * pa3
            + -4.09087898e-07 * dt3 * pa3
            + 6.14155345e-01 * pa4
            + -6.16755931e-02 * ta * pa4
            + 1.33374846e-03 * ta2 * pa4
            + 3.55375387e-03 * va * pa4
            + -5.13027851e-04 * ta * va * pa4
            + 1.02449757e-04 * va2 * pa4
            + -1.48526421e-03 * dTmrt * pa4
            + -4.11469183e-05 * ta * dTmrt * pa4
            + -6.80434415e-06 * va * dTmrt * pa4
            + -9.77675906e-06 * dt2 * pa4
            + 8.82773108e-02 * pa5
            + -3.01859306e-03 * ta * pa5
            + 1.04452989e-03 * va * pa5
            + 2.47090539e-04 * dTmrt * pa5
            + 1.48348065e-03 * pa6;

        return u;
    }

    /// <summary>UTCI thermal-stress band (Bröde 2012 Table 1).</summary>
    public static UtciStress Band(double utciC) => utciC switch
    {
        < -40 => UtciStress.ExtremeCold,
        < -27 => UtciStress.VeryStrongCold,
        < -13 => UtciStress.StrongCold,
        < 0   => UtciStress.ModerateCold,
        < 9   => UtciStress.SlightCold,
        < 26  => UtciStress.NoStress,
        < 32  => UtciStress.ModerateHeat,
        < 38  => UtciStress.StrongHeat,
        < 46  => UtciStress.VeryStrongHeat,
        _     => UtciStress.ExtremeHeat,
    };
}

public enum UtciStress
{
    ExtremeCold,
    VeryStrongCold,
    StrongCold,
    ModerateCold,
    SlightCold,
    NoStress,
    ModerateHeat,
    StrongHeat,
    VeryStrongHeat,
    ExtremeHeat,
}
