namespace WeatherBlend.Predict.Surface;

/// <summary>
/// Solar position + tilted-surface shortwave projection for the cliff-face
/// rock-temp extension (docs/SENNEN_ROCK_TEMP_PLAN.md S3). The NWP/blend
/// <c>shortwave_radiation</c> is flux on a HORIZONTAL surface; a vertical
/// W/NW sea-cliff face has an entirely different heating day (no direct sun
/// until afternoon, near-perpendicular sun in the evening), so the direct
/// beam must be re-projected onto the face plane.
///
/// Solar position is the standard NOAA solar-calculator algorithm (the
/// general-purpose spreadsheet port, accurate to well under 1° for
/// 1900–2100), without atmospheric refraction — irrelevant at the &lt;1°
/// level for an energy-budget driver. Azimuth is degrees clockwise from
/// true north; elevation is degrees above the horizon.
/// </summary>
public static class SolarGeometry
{
    /// <summary>Below this solar elevation the direct beam is dropped
    /// entirely (the horizontal→normal conversion divides by sin(elevation)
    /// and blows up at grazing sun; the real direct flux there is tiny and
    /// heavily attenuated anyway).</summary>
    public const double MinElevationForDirectDeg = 3.0;

    /// <summary>Physical ceiling on direct-normal irradiance (W/m²) — guards
    /// the sin(elevation) division against a model SW/direct-fraction glitch
    /// at low sun. Clear-sky DNI at the surface tops out ~1000–1050.</summary>
    public const double MaxDniWm2 = 1100.0;

    private static double Rad(double deg) => deg * Math.PI / 180.0;
    private static double Deg(double rad) => rad * 180.0 / Math.PI;

    /// <summary>
    /// Sun elevation + azimuth (degrees; azimuth clockwise from north) at a
    /// UTC instant and location. NOAA solar-calculator algorithm.
    /// </summary>
    public static (double ElevationDeg, double AzimuthDeg) SolarPosition(DateTime utc, double latDeg, double lonDeg)
    {
        // Julian century from J2000. OADate epoch 1899-12-30 → JD offset 2415018.5.
        var jd = utc.ToOADate() + 2415018.5;
        var jc = (jd - 2451545.0) / 36525.0;

        var meanLon = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0;
        var meanAnom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
        var ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);

        var eqCentre = Math.Sin(Rad(meanAnom)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
                     + Math.Sin(Rad(2 * meanAnom)) * (0.019993 - 0.000101 * jc)
                     + Math.Sin(Rad(3 * meanAnom)) * 0.000289;
        var trueLon = meanLon + eqCentre;
        var apparentLon = trueLon - 0.00569 - 0.00478 * Math.Sin(Rad(125.04 - 1934.136 * jc));

        var meanObliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
        var obliq = meanObliq + 0.00256 * Math.Cos(Rad(125.04 - 1934.136 * jc));

        var declDeg = Deg(Math.Asin(Math.Sin(Rad(obliq)) * Math.Sin(Rad(apparentLon))));

        var varY = Math.Tan(Rad(obliq / 2.0)) * Math.Tan(Rad(obliq / 2.0));
        var eqTimeMin = 4.0 * Deg(
            varY * Math.Sin(2.0 * Rad(meanLon))
            - 2.0 * ecc * Math.Sin(Rad(meanAnom))
            + 4.0 * ecc * varY * Math.Sin(Rad(meanAnom)) * Math.Cos(2.0 * Rad(meanLon))
            - 0.5 * varY * varY * Math.Sin(4.0 * Rad(meanLon))
            - 1.25 * ecc * ecc * Math.Sin(2.0 * Rad(meanAnom)));

        var trueSolarMin = (utc.TimeOfDay.TotalMinutes + eqTimeMin + 4.0 * lonDeg + 1440.0) % 1440.0;
        var hourAngle = trueSolarMin / 4.0 < 0 ? trueSolarMin / 4.0 + 180.0 : trueSolarMin / 4.0 - 180.0;

        var cosZen = Math.Sin(Rad(latDeg)) * Math.Sin(Rad(declDeg))
                   + Math.Cos(Rad(latDeg)) * Math.Cos(Rad(declDeg)) * Math.Cos(Rad(hourAngle));
        var zenDeg = Deg(Math.Acos(Math.Clamp(cosZen, -1.0, 1.0)));
        var elevDeg = 90.0 - zenDeg;

        double azDeg;
        var denom = Math.Cos(Rad(latDeg)) * Math.Sin(Rad(zenDeg));
        if (Math.Abs(denom) < 1e-9)
        {
            azDeg = 180.0; // sun at zenith/nadir — azimuth undefined; any value works
        }
        else
        {
            var cosAz = (Math.Sin(Rad(latDeg)) * Math.Cos(Rad(zenDeg)) - Math.Sin(Rad(declDeg))) / denom;
            var acos = Deg(Math.Acos(Math.Clamp(cosAz, -1.0, 1.0)));
            azDeg = hourAngle > 0.0 ? (acos + 180.0) % 360.0 : (540.0 - acos) % 360.0;
        }
        return (elevDeg, azDeg);
    }

    /// <summary>
    /// Shortwave incident on a tilted face (W/m²) from horizontal GHI and a
    /// direct fraction: the direct beam is converted to direct-normal
    /// (÷ sin elevation, capped) and projected onto the face plane; diffuse
    /// is isotropic-scaled by the face's sky-view factor. Sea/ground-reflected
    /// shortwave is neglected (ocean albedo ~0.06). A horizontal face
    /// (slope 0) with fSky 1 returns GHI exactly, so the whole-crag mode is
    /// the degenerate case of this projection.
    /// </summary>
    /// <param name="ghiWm2">Horizontal shortwave (the blend / NWP value).</param>
    /// <param name="directFrac">Direct share of GHI (0..1, NWP-mean direct/(direct+diffuse)).</param>
    /// <param name="elevationDeg">Sun elevation (deg).</param>
    /// <param name="azimuthDeg">Sun azimuth (deg clockwise from north).</param>
    /// <param name="faceAspectDeg">Compass bearing the face looks toward (deg).</param>
    /// <param name="faceSlopeDeg">Face steepness: 0 = horizontal, 90 = vertical.</param>
    /// <param name="fSky">Sky-view factor of the face (vertical wall ≈ 0.5).</param>
    public static double FaceIncidentSw(
        double ghiWm2, double directFrac, double elevationDeg, double azimuthDeg,
        double faceAspectDeg, double faceSlopeDeg, double fSky)
    {
        var ghi = Math.Max(ghiWm2, 0.0);
        var frac = Math.Clamp(directFrac, 0.0, 1.0);
        var skyDiffuse = ghi * (1.0 - frac) * fSky;
        if (elevationDeg <= MinElevationForDirectDeg)
            return skyDiffuse;

        var sinElev = Math.Sin(Rad(elevationDeg));
        var dni = Math.Min(ghi * frac / sinElev, MaxDniWm2);
        var cosIncidence = Math.Cos(Rad(elevationDeg)) * Math.Sin(Rad(faceSlopeDeg)) * Math.Cos(Rad(azimuthDeg - faceAspectDeg))
                         + sinElev * Math.Cos(Rad(faceSlopeDeg));
        return dni * Math.Max(0.0, cosIncidence) + skyDiffuse;
    }
}
