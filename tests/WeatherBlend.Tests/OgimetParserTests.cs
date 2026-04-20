using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

public class OgimetParserTests
{
    private const string Station = "EGTE";
    private const string Loc = "bonehill";

    [Fact]
    public void Parses_standard_metar_body_with_Q_pressure()
    {
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 19010KT 9999 SCT009 11/05 Q1013=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.LocationName.Should().Be(Loc);
        row.Station.Should().Be(Station);
        row.ObservedTimeUtc.Should().Be(new DateTime(2026, 4, 15, 6, 20, 0, DateTimeKind.Utc));
        row.Temperature2m.Should().Be(11);
        row.DewPoint2m.Should().Be(5);
        row.WindDirection10m.Should().Be(190);
        row.WindSpeed10m.Should().BeApproximately(10 * 0.514444, 1e-6);
        row.SurfacePressure.Should().Be(1013);
        row.Visibility.Should().Be(9999);
        row.HasPrecipitation.Should().BeFalse();
    }

    [Fact]
    public void Cavok_sets_visibility_to_10000()
    {
        var line = "EGTE,2026,04,15,12,00,METAR EGTE 151200Z 27008KT CAVOK 15/10 Q1020=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.Visibility.Should().Be(10000);
    }

    [Fact]
    public void Nil_report_returns_null()
    {
        var line = "EGTE,2026,04,15,00,20,METAR EGTE 150020Z NIL=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);
        row.Should().BeNull();
    }

    [Fact]
    public void Negative_temperature_and_dewpoint_parsed_as_minus()
    {
        var line = "EGTE,2026,01,05,03,20,METAR EGTE 050320Z 00000KT 9999 FEW020 M02/M05 Q1030=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.Temperature2m.Should().Be(-2);
        row.DewPoint2m.Should().Be(-5);
    }

    [Fact]
    public void Variable_wind_leaves_direction_null_but_keeps_speed()
    {
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z VRB03KT 9999 11/10 Q1013=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.WindDirection10m.Should().BeNull();
        row.WindSpeed10m.Should().BeApproximately(3 * 0.514444, 1e-6);
    }

    [Fact]
    public void InHg_pressure_is_converted_to_hPa()
    {
        // A2992 = 29.92 inHg → 29.92 * 33.8639 ≈ 1013.2 hPa
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 19010KT 9999 11/05 A2992=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.SurfacePressure.Should().BeApproximately(1013.2, 0.1);
    }

    [Fact]
    public void Gust_is_parsed_and_converted_to_ms()
    {
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 19015G25KT 9999 11/05 Q1013=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.WindGusts10m.Should().BeApproximately(25 * 0.514444, 1e-6);
    }

    [Fact]
    public void Precipitation_code_sets_has_precipitation_true()
    {
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 19010KT 3000 -RA BKN005 10/10 Q1010=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.HasPrecipitation.Should().BeTrue();
        // The regex uses \b boundaries, so the leading "-" intensity marker is not captured —
        //   only the precip code "RA" ends up in PresentWeather. Presence is what HasPrecipitation flags.
        row.PresentWeather.Should().Contain("RA");
    }

    [Fact]
    public void Fog_without_precipitation_leaves_has_precipitation_false()
    {
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 00000KT 0200 FG VV001 05/05 Q1015=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.HasPrecipitation.Should().BeFalse();
        row.PresentWeather.Should().Contain("FG");
    }

    [Fact]
    public void Wrong_station_returns_null()
    {
        var line = "EGDY,2026,04,15,06,20,METAR EGDY 150620Z 19010KT 9999 11/05 Q1013=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);
        row.Should().BeNull();
    }

    [Fact]
    public void Too_few_csv_fields_returns_null()
    {
        var line = "EGTE,2026,04,15,06,20";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);
        row.Should().BeNull();
    }

    [Fact]
    public void Raw_metar_preserved_without_trailing_equals()
    {
        var line = "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 19010KT 9999 11/05 Q1013=";
        var row = OgimetClient.TryParseCsvLine(line, Loc, Station);

        row.Should().NotBeNull();
        row!.RawMetar.Should().Be("METAR EGTE 150620Z 19010KT 9999 11/05 Q1013");
    }
}
