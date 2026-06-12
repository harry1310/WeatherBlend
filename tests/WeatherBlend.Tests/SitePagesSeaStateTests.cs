using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Sea state page tests (Sennen sea-state Phase 2, 2026-06-11):
///   * Wave / tide / swell sections render from <see cref="SitePages.WaveForecastPoint"/> rows
///   * The standard version-dedup rule (newest ModelVersion per ValidTimeUtc)
///     holds at page level — the second-ModelVersion-straddling-a-retrain
///     case has broken charts twice before (see SeriesDedup)
///   * Wind section renders its "no data yet" empty state until the Sennen
///     wind model produces rows, and a speed line once it does
///   * The per-loc nav gains a Sea state button only when the tab is configured
/// </summary>
public class SitePagesSeaStateTests
{
    private static readonly DateTime GeneratedAt = new(2026, 6, 11, 12, 0, 0, DateTimeKind.Utc);

    private static SitePages.LocationDescriptor Sennen(IReadOnlyList<string>? tabs = null) =>
        new(
            Name: "sennen_cove",
            DisplayName: "Sennen, Cornwall",
            RainStationSlugs: new[] { "ea_trengwainton" },
            IsPrimary: false,
            Tabs: tabs ?? new[] { "overview", "temperature", "rain", "sea_state", "skill", "models" },
            OverviewFirstVisibleHourUtc: 6,
            OverviewLastVisibleHourUtcExclusive: 21);

    private static SitePages.SiteInputs MakeInput(
        SitePages.LocationDescriptor loc,
        IReadOnlyList<SitePages.WaveForecastPoint>? waves = null,
        IReadOnlyList<SitePages.WindForecastPoint>? wind = null) =>
        new()
        {
            ActiveLocationName = loc.Name,
            RenderingFor = loc,
            AssetPrefix = "../",
            LocationDisplay = loc.DisplayName,
            Latitude = 50.0786, Longitude = -5.7044, ElevationMeters = 35,
            MetarStation = "EGDR",
            GeneratedAtUtc = GeneratedAt,
            WindowStartUtc = GeneratedAt.AddDays(-30),
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
            Locations = new[] { loc },
            WavePredictions = waves ?? Array.Empty<SitePages.WaveForecastPoint>(),
            WindPredictions = wind ?? Array.Empty<SitePages.WindForecastPoint>(),
        };

    /// <summary>One wave row with every pass-through populated. Valid times
    /// default into the future of <see cref="GeneratedAt"/> so the swell
    /// detail table (future rows only) sees them.</summary>
    private static SitePages.WaveForecastPoint Wave(
        DateTime validTime,
        double heightM,
        string version = "v2026-06-11_220452_wave_height_lgb",
        DateTime? predictedAt = null,
        double? tideM = 1.2) =>
        new(
            Version: version,
            PredictedAtUtc: predictedAt ?? GeneratedAt.AddHours(-1),
            ValidTimeUtc: validTime,
            LeadHours: (int)(validTime - GeneratedAt).TotalHours,
            WaveHeightM: heightM,
            BandLoM: heightM - 0.2,
            BandHiM: heightM + 0.3,
            WavePeriodS: 6.5,
            WaveDirectionDeg: 269.0,
            SwellHeightM: 1.4,
            SwellPeriodS: 6.4,
            SwellDirectionDeg: 288.0,
            SecondarySwellHeightM: 0.84,
            SecondarySwellPeriodS: 4.75,
            TideHeightMsl: tideM,
            SeaSurfaceTempC: 14.2,
            LocationName: "sennen_cove");

    [Fact]
    public void RenderSeaState_renders_wave_band_tide_and_swell_sections()
    {
        var waves = new[]
        {
            Wave(GeneratedAt,             1.8),
            Wave(GeneratedAt.AddHours(3), 1.7, tideM: -0.5),
            Wave(GeneratedAt.AddHours(6), 1.6, tideM: 2.1),
        };
        var html = SitePages.RenderSeaState(MakeInput(Sennen(), waves));

        // Wave height chart + its CQR band ribbon series.
        html.Should().Contain("Significant wave height");
        html.Should().Contain("90% lo").And.Contain("90% hi");

        // Tide chart with the dashed MSL reference series.
        html.Should().Contain("Tide height vs mean sea level");
        html.Should().Contain("MSL (0 m)");

        // Swell chart + 3-hourly detail table. 288° "from" reads as WNW.
        html.Should().Contain("Swell height");
        html.Should().Contain("3-hourly swell detail");
        html.Should().Contain("WNW");
    }

    [Fact]
    public void RenderSeaState_collapses_to_newest_model_version_per_valid_time()
    {
        // Two versions straddling a retrain write the SAME valid hour. The
        // chart must plot the newer version's value only — the rule that
        // broke the wind speed/direction charts twice (SeriesDedup header).
        var valid = GeneratedAt.AddHours(2);
        var waves = new[]
        {
            // Older version, FRESHER made-at — version wins over made-at.
            Wave(valid, 9.75, version: "v2026-06-04_120000_wave_height_lgb",
                 predictedAt: GeneratedAt),
            Wave(valid, 4.25, version: "v2026-06-11_220452_wave_height_lgb",
                 predictedAt: GeneratedAt.AddHours(-3)),
        };
        var html = SitePages.RenderSeaState(MakeInput(Sennen(), waves));

        html.Should().Contain("4.25");
        html.Should().NotContain("9.75");
    }

    [Fact]
    public void RenderSeaState_renders_page_level_empty_state_when_no_wave_rows()
    {
        // Fresh deploy / unsynced tree: the page must render (not throw),
        // with the wave empty state and the wind section still present.
        var html = SitePages.RenderSeaState(MakeInput(Sennen()));

        html.Should().Contain("No wave-height predictions available yet");
        html.Should().Contain("<h3>Wind</h3>");
    }

    [Fact]
    public void RenderSeaState_renders_wind_empty_state_when_no_wind_rows()
    {
        // The Sennen wind model trains after this page ships — until its
        // rows land the section must degrade, not fail the page.
        var html = SitePages.RenderSeaState(MakeInput(Sennen(), new[] { Wave(GeneratedAt, 1.8) }));

        html.Should().Contain("No wind predictions for this location yet");
        html.Should().NotContain("Blend wind speed (mph)");
    }

    [Fact]
    public void RenderSeaState_renders_wind_speed_line_when_rows_exist()
    {
        var wind = new[]
        {
            new SitePages.WindForecastPoint(
                Version: "v2026-06-12_temp",
                PredictedAtUtc: GeneratedAt.AddHours(-2),
                ValidTimeUtc: GeneratedAt.AddHours(4),
                LeadHours: 24,
                SpeedMs: 10.0,
                Gfs: null, Ecmwf: null, Icon: null, Mf: null,
                Ukmo: null, Gem: null, Aifs: null,
                LocationName: "sennen_cove"),
        };
        var html = SitePages.RenderSeaState(MakeInput(Sennen(), new[] { Wave(GeneratedAt, 1.8) }, wind));

        html.Should().Contain("Blend wind speed (mph)");
        html.Should().NotContain("No wind predictions for this location yet");
    }

    [Fact]
    public void Per_location_nav_includes_sea_state_button_when_configured()
    {
        var html = SitePages.RenderSeaState(MakeInput(Sennen(), new[] { Wave(GeneratedAt, 1.8) }));

        // The nav button links the page and the active-tab highlight lands
        // on it (pageId "sea-state").
        html.Should().Contain("<li><a href=\"forecasts-sea-state.html\" class=\"active\">Sea state</a>");
    }

    [Fact]
    public void Per_location_nav_omits_sea_state_button_when_tab_absent()
    {
        // A Membury-shaped tab list must not grow a Sea state button.
        var loc = Sennen(tabs: new[] { "overview", "temperature", "rain", "skill", "models" });
        var html = SitePages.RenderTempSkill(MakeInput(loc));

        html.Should().NotContain("forecasts-sea-state.html");
    }

    [Theory]
    [InlineData(0.0,   "N")]
    [InlineData(359.9, "N")]
    [InlineData(45.0,  "NE")]
    [InlineData(90.0,  "E")]
    [InlineData(180.0, "S")]
    [InlineData(270.0, "W")]
    [InlineData(288.0, "WNW")]
    [InlineData(292.5, "WNW")]
    [InlineData(315.0, "NW")]
    [InlineData(-90.0, "W")]    // out-of-range bearings wrap
    public void CompassPoint_maps_bearings_to_sixteen_point_labels(double degrees, string expected)
    {
        SitePages.CompassPoint(degrees).Should().Be(expected);
    }
}
