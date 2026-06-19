using FluentAssertions;
using WeatherBlend.Predict.Surface;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Beam-projection check behind the 2026-06-20 Bonehill "sunlit vertical face"
/// move (docs/ROCK_IR_CALIBRATION.md). The whole-crag horizontal surface caught
/// the full noon beam and ran ~4°C warm vs the vertical rock climbers are on;
/// a south-facing vertical face must catch materially less shortwave at summer
/// noon — the mechanism that pulls the modelled temperature down toward the
/// field reading (horizontal-sun 25°C vs vertical-sun 20°C).
/// </summary>
public class SolarGeometryFaceTests
{
    // Bonehill, 18 Jun ~noon (the pipeline samples the half-hour mean).
    private const double Lat = 50.5831, Lon = -3.7931;
    private static readonly DateTime Noon = new(2026, 6, 18, 11, 30, 0, DateTimeKind.Utc);

    [Fact]
    public void South_vertical_face_catches_far_less_noon_beam_than_horizontal()
    {
        var (elev, az) = SolarGeometry.SolarPosition(Noon, Lat, Lon);
        elev.Should().BeGreaterThan(55, "midsummer noon sun is high at 50.6°N");

        const double ghi = 600.0, directFrac = 0.6;
        // Whole-crag horizontal mode: slope 0, fSky 1 returns GHI exactly.
        var horizontal = SolarGeometry.FaceIncidentSw(ghi, directFrac, elev, az, 0, 0, fSky: 1.0);
        // Sunlit vertical face: south (180°), vertical (90°), half-sky view.
        var southVertical = SolarGeometry.FaceIncidentSw(ghi, directFrac, elev, az, 180, 90, fSky: 0.5);

        horizontal.Should().BeApproximately(ghi, 1.0);
        southVertical.Should().BeLessThan(horizontal);
        // High sun ⇒ a vertical wall grazes the beam: well under three-quarters of
        // the horizontal load. This is the reduction that cools the modelled rock.
        southVertical.Should().BeLessThan(horizontal * 0.75);
    }

    [Fact]
    public void Vertical_face_beam_falls_off_away_from_the_sun_azimuth()
    {
        var (elev, az) = SolarGeometry.SolarPosition(Noon, Lat, Lon);
        const double ghi = 600.0, directFrac = 0.6;
        // A vertical face pointing at the sun catches more beam than one turned
        // 90° away from it (which sees no direct beam at all).
        var facingSun = SolarGeometry.FaceIncidentSw(ghi, directFrac, elev, az, az, 90, fSky: 0.5);
        var sideOn    = SolarGeometry.FaceIncidentSw(ghi, directFrac, elev, az, az + 90, 90, fSky: 0.5);
        facingSun.Should().BeGreaterThan(sideOn);
    }
}
