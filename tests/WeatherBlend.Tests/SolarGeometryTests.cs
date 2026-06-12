using FluentAssertions;
using WeatherBlend.Predict.Surface;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Solar position + cliff-face shortwave projection (SENNEN_ROCK_TEMP_PLAN.md
/// S3). Position checks pin the NOAA algorithm against well-known sky facts
/// for Sennen (50.0786°N, 5.7044°W) — solar noon there is ~12:23 UTC (+23 min
/// for longitude, ± the equation of time). Projection checks pin the physical
/// behaviours that motivated face mode: a south wall peaks around midday, a
/// west wall in the evening, and low evening sun hits a vertical west face
/// near-perpendicular (face-incident flux far above horizontal).
/// </summary>
public class SolarGeometryTests
{
    private const double SennenLat = 50.0786;
    private const double SennenLon = -5.7044;

    [Fact]
    public void Solar_noon_midsummer_at_sennen_is_high_and_due_south()
    {
        // 2026-06-21 12:24 UTC ≈ local solar noon. Max elevation = 90 − (lat − 23.44).
        var (elev, az) = SolarGeometry.SolarPosition(
            new DateTime(2026, 6, 21, 12, 24, 0, DateTimeKind.Utc), SennenLat, SennenLon);
        elev.Should().BeApproximately(63.4, 1.0);
        az.Should().BeApproximately(180.0, 3.0);
    }

    [Fact]
    public void Solar_noon_midwinter_at_sennen_is_low()
    {
        var (elev, az) = SolarGeometry.SolarPosition(
            new DateTime(2026, 12, 21, 12, 24, 0, DateTimeKind.Utc), SennenLat, SennenLon);
        elev.Should().BeApproximately(16.5, 1.0);
        az.Should().BeApproximately(180.0, 4.0);
    }

    [Fact]
    public void Midsummer_evening_sun_is_low_in_the_west_to_northwest()
    {
        var (elev, az) = SolarGeometry.SolarPosition(
            new DateTime(2026, 6, 21, 20, 0, 0, DateTimeKind.Utc), SennenLat, SennenLon);
        elev.Should().BeInRange(2.0, 15.0);
        az.Should().BeInRange(285.0, 310.0);
    }

    [Fact]
    public void Midnight_sun_is_below_the_horizon()
    {
        var (elev, _) = SolarGeometry.SolarPosition(
            new DateTime(2026, 6, 21, 0, 30, 0, DateTimeKind.Utc), SennenLat, SennenLon);
        elev.Should().BeLessThan(-5.0);
    }

    // ------------------------------------------------------------------
    // Face-incident shortwave projection
    // ------------------------------------------------------------------

    [Fact]
    public void Horizontal_face_with_full_sky_view_recovers_ghi()
    {
        // slope 0 + fSky 1: cos(incidence) folds back to sin(elev), so
        // direct_h/sin(elev)·sin(elev) + diffuse·1 = GHI. Whole-crag mode is
        // the degenerate case of the projection.
        var sw = SolarGeometry.FaceIncidentSw(
            ghiWm2: 500, directFrac: 0.7, elevationDeg: 40, azimuthDeg: 200,
            faceAspectDeg: 0, faceSlopeDeg: 0, fSky: 1.0);
        sw.Should().BeApproximately(500.0, 1e-6);
    }

    [Fact]
    public void Vertical_west_face_gets_no_direct_beam_from_midday_southerly_sun()
    {
        // Sun due south, face looks west: 90° between beam and the face
        // normal's horizontal component → direct projection 0; only the
        // fSky-scaled diffuse remains.
        var sw = SolarGeometry.FaceIncidentSw(
            ghiWm2: 600, directFrac: 0.7, elevationDeg: 60, azimuthDeg: 180,
            faceAspectDeg: 270, faceSlopeDeg: 90, fSky: 0.5);
        sw.Should().BeApproximately(600 * 0.3 * 0.5, 1e-6);
    }

    [Fact]
    public void Vertical_west_face_in_low_evening_sun_far_exceeds_horizontal_flux()
    {
        // Low western sun strikes a vertical west wall near-perpendicular:
        // the face-incident flux is several times the horizontal value. This
        // is the effect the horizontal assumption gets most wrong.
        var sw = SolarGeometry.FaceIncidentSw(
            ghiWm2: 200, directFrac: 0.7, elevationDeg: 10, azimuthDeg: 280,
            faceAspectDeg: 270, faceSlopeDeg: 90, fSky: 0.5);
        sw.Should().BeGreaterThan(600.0);
    }

    [Fact]
    public void Dni_is_capped_at_the_physical_ceiling()
    {
        // Grazing sun + absurd direct share: the ÷sin(elev) conversion must
        // not manufacture a multi-kW beam from a model glitch.
        var sw = SolarGeometry.FaceIncidentSw(
            ghiWm2: 800, directFrac: 1.0, elevationDeg: 3.5, azimuthDeg: 270,
            faceAspectDeg: 270, faceSlopeDeg: 90, fSky: 0.5);
        sw.Should().BeLessThan(SolarGeometry.MaxDniWm2 + 1.0);
    }

    [Fact]
    public void Below_minimum_elevation_only_diffuse_survives()
    {
        var sw = SolarGeometry.FaceIncidentSw(
            ghiWm2: 100, directFrac: 0.9, elevationDeg: 1.0, azimuthDeg: 270,
            faceAspectDeg: 270, faceSlopeDeg: 90, fSky: 0.5);
        sw.Should().BeApproximately(100 * 0.1 * 0.5, 1e-6);
    }

    [Fact]
    public void West_face_heating_peaks_later_in_the_day_than_south_face()
    {
        // Constant GHI/direct share isolates pure geometry: over a June day
        // at Sennen the south wall's incident flux must peak around midday
        // and the west wall's in the evening.
        static int PeakHourUtc(double aspectDeg)
        {
            double best = -1; var bestHour = -1;
            for (var hour = 4; hour <= 21; hour++)
            {
                var (elev, az) = SolarGeometry.SolarPosition(
                    new DateTime(2026, 6, 21, hour, 30, 0, DateTimeKind.Utc), SennenLat, SennenLon);
                var sw = SolarGeometry.FaceIncidentSw(500, 0.7, elev, az, aspectDeg, 90, 0.5);
                if (sw > best) { best = sw; bestHour = hour; }
            }
            return bestHour;
        }

        var southPeak = PeakHourUtc(180);
        var westPeak = PeakHourUtc(270);
        southPeak.Should().BeInRange(11, 14);
        westPeak.Should().BeInRange(16, 20);
        westPeak.Should().BeGreaterThan(southPeak);
    }
}
