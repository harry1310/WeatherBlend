using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

// Tile sky glyph from cloud cover + radiation (2026-06-20). Daytime folds the
// reported cloud fraction with the radiation clearness index so a bright hour
// reads sunnier than the cloud % alone.
public class SitePagesSkyGlyphTests
{
    private const double Lat = 50.5831, Lon = -3.7931;   // Bonehill

    [Fact]
    public void No_cloud_value_yields_no_glyph()
    {
        SitePages.SkyGlyph(null, 400, new DateTime(2026, 6, 21, 12, 0, 0, DateTimeKind.Utc), Lat, Lon)
            .Should().BeEmpty();
    }

    [Fact]
    public void Night_shows_the_moon_when_clear()
    {
        // Midwinter midnight — sun well below the horizon.
        var s = SitePages.SkyGlyph(10, 0, new DateTime(2026, 1, 15, 0, 0, 0, DateTimeKind.Utc), Lat, Lon);
        s.Should().Contain("\U0001F319").And.Contain("night");
    }

    [Fact]
    public void Clear_summer_noon_is_sunny()
    {
        var s = SitePages.SkyGlyph(5, 850, new DateTime(2026, 6, 21, 12, 0, 0, DateTimeKind.Utc), Lat, Lon);
        s.Should().Contain("☀").And.Contain("Sunny");
    }

    [Fact]
    public void Overcast_dim_noon_is_cloudy()
    {
        var s = SitePages.SkyGlyph(95, 80, new DateTime(2026, 6, 21, 12, 0, 0, DateTimeKind.Utc), Lat, Lon);
        s.Should().Contain("☁");
    }

    [Fact]
    public void Bright_but_high_cloud_reads_sunnier_than_the_cloud_pct_alone()
    {
        // 99% reported cloud but strong shortwave (the globals-leak case): the
        // clearness index pulls it off "Cloudy" toward partly cloudy.
        var s = SitePages.SkyGlyph(99, 700, new DateTime(2026, 6, 21, 12, 0, 0, DateTimeKind.Utc), Lat, Lon);
        s.Should().NotContain("☁");
        s.Should().Contain("⛅");
    }
}
