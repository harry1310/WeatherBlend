using FluentAssertions;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Phase D (2026-05-20) site-restructure regression tests:
///   * <see cref="LocationConfig.HasTab"/> / <see cref="LocationConfig.Tabs"/>
///   * Per-loc nav + cross-loc top-level nav in <see cref="SitePages"/>
///   * Tab filtering on <see cref="SitePages"/> Skill / Models sub-navs
///   * Picker page renders all configured locations
///   * <c>AssetPrefix</c> threads to top-level asset hrefs
/// </summary>
public class PhaseDSiteTests
{
    private static SitePages.LocationDescriptor MakeDescriptor(
        string name = "bonehill_rocks",
        string displayName = "Bonehill Rocks",
        IReadOnlyList<string>? tabs = null,
        bool isPrimary = true) =>
        new(
            Name: name,
            DisplayName: displayName,
            RainStationSlugs: new[] { "ea_bellever_dartmoor" },
            IsPrimary: isPrimary,
            Tabs: tabs ?? new[] { "overview", "temperature", "rain", "dry_window", "skill", "models" },
            OverviewFirstVisibleHourUtc: 4,
            OverviewLastVisibleHourUtcExclusive: 22);

    private static SitePages.SiteInputs MakeInput(
        SitePages.LocationDescriptor? renderingFor = null,
        IReadOnlyList<SitePages.LocationDescriptor>? locations = null,
        string assetPrefix = "../")
    {
        var generatedAt = new DateTime(2026, 5, 20, 0, 0, 0, DateTimeKind.Utc);
        var locs = locations ?? (renderingFor is null ? Array.Empty<SitePages.LocationDescriptor>() : new[] { renderingFor });
        return new SitePages.SiteInputs
        {
            ActiveLocationName = renderingFor?.Name ?? "",
            RenderingFor = renderingFor,
            AssetPrefix = assetPrefix,
            LocationDisplay = renderingFor?.DisplayName ?? "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = generatedAt.AddDays(-30),
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
            Locations = locs,
        };
    }

    [Fact]
    public void LocationConfig_HasTab_is_case_insensitive()
    {
        var cfg = new LocationConfig { Tabs = new List<string> { "overview", "Temperature", "RAIN" } };
        cfg.HasTab("overview").Should().BeTrue();
        cfg.HasTab("temperature").Should().BeTrue();
        cfg.HasTab("rain").Should().BeTrue();
        cfg.HasTab("dry_window").Should().BeFalse();
    }

    [Fact]
    public void LocationConfig_Overview_defaults_to_full_day()
    {
        var cfg = new LocationConfig();
        cfg.Overview.FirstVisibleHourUtc.Should().Be(0);
        cfg.Overview.LastVisibleHourUtcExclusive.Should().Be(24);
    }

    [Fact]
    public void LocationDescriptor_HasTab_is_case_insensitive()
    {
        var loc = MakeDescriptor(tabs: new[] { "overview", "rain" });
        loc.HasTab("OVERVIEW").Should().BeTrue();
        loc.HasTab("rain").Should().BeTrue();
        loc.HasTab("temperature").Should().BeFalse();
    }

    [Fact]
    public void WrapPage_prepends_AssetPrefix_to_stylesheet_and_chart_js_links()
    {
        var loc = MakeDescriptor();
        var input = MakeInput(renderingFor: loc, assetPrefix: "../");
        // Render via any per-loc page to exercise WrapPage. RenderHomePicker
        // is the cleanest minimal case — top-level, no charts. For per-loc
        // verification use a page rendered with a non-empty AssetPrefix.
        var html = SitePages.RenderTempSkill(input);

        html.Should().Contain("href=\"../styles.css\"");
        html.Should().Contain("src=\"../chart.js\"");
        // Cross-cutting nav links the global Specs page via the prefix.
        html.Should().Contain("href=\"../specs.html\"");
    }

    [Fact]
    public void WrapPage_top_level_emits_no_prefix()
    {
        var bonehill = MakeDescriptor("bonehill_rocks", "Bonehill Rocks");
        var membury = MakeDescriptor("membury_devon", "Membury", tabs: new[] { "overview", "rain" }, isPrimary: false);
        var input = MakeInput(renderingFor: null, locations: new[] { bonehill, membury }, assetPrefix: "");
        var html = SitePages.RenderHomePicker(input);

        html.Should().Contain("href=\"styles.css\"");
        html.Should().Contain("src=\"chart.js\"");
        // Per-loc cards link into the location subdir.
        html.Should().Contain("href=\"bonehill_rocks/index.html\"");
        html.Should().Contain("href=\"membury_devon/index.html\"");
        // Top-level chrome's Specs link is same-level.
        html.Should().Contain("href=\"specs.html\"");
    }

    [Fact]
    public void RenderHomePicker_lists_each_location_with_its_tabs()
    {
        var bonehill = MakeDescriptor("bonehill_rocks", "Bonehill Rocks");
        var membury = MakeDescriptor("membury_devon", "Membury", tabs: new[] { "overview", "rain" }, isPrimary: false);
        var input = MakeInput(renderingFor: null, locations: new[] { bonehill, membury }, assetPrefix: "");
        var html = SitePages.RenderHomePicker(input);

        html.Should().Contain("Bonehill Rocks");
        html.Should().Contain("Membury");
        // Pretty tab labels surface in the card subtitle line.
        html.Should().Contain("Overview");
        html.Should().Contain("Rain");
    }

    [Fact]
    public void Per_location_nav_omits_tabs_not_in_LocationDescriptor()
    {
        // Membury config: overview + rain only — no temperature, no dry-window,
        // no skill, no models nav buttons.
        var membury = MakeDescriptor("membury_devon", "Membury",
            tabs: new[] { "overview", "rain" }, isPrimary: false);
        var input = MakeInput(renderingFor: membury);
        var html = SitePages.RenderRainSkill(input, stationSlug: null);

        // Per-loc site-nav builds from descriptor.Tabs — only the two buttons.
        html.Should().Contain("forecasts-rain-24h.html");
        // No top-nav link for Temperature / Dry-window / Models.
        html.Should().NotContain("forecasts-temp-24h.html");
        html.Should().NotContain("forecasts-dry-window.html");
        html.Should().NotContain("models-temp.html");
    }

    [Fact]
    public void RenderSkillSubNav_filters_entries_by_loc_Tabs()
    {
        // Membury — temp + rain only, no dry-window.
        var membury = MakeDescriptor("membury_devon", "Membury",
            tabs: new[] { "overview", "temperature", "rain", "skill" }, isPrimary: false);
        var input = MakeInput(renderingFor: membury);
        var html = SitePages.RenderTempSkill(input);

        // The skill sub-nav (temp/rain/dry-window pills) drops dry-window.
        html.Should().Contain("skill-temperature.html");
        html.Should().Contain("skill-rainfall.html");
        html.Should().NotContain("skill-dry-window.html");
    }

    [Fact]
    public void RenderForecastsTemp_NWP_overlay_scopes_to_rendered_location_not_primary()
    {
        // Regression: pre-fix, the temperature forecast page filtered the raw
        // NWP overlay rows to PrimaryLocation (Bonehill), so the Membury temp
        // chart drew Bonehill's 393 m grid-cell NWP lines under Membury's
        // 120 m blend — leaving the blend ~4-5°C above every NWP line.
        var generatedAt = new DateTime(2026, 5, 21, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var membury = MakeDescriptor("membury_devon", "Membury",
            tabs: new[] { "overview", "temperature", "rain" }, isPrimary: false);
        var bonehill = MakeDescriptor("bonehill_rocks", "Bonehill Rocks");

        // Distinctive temps: Bonehill cold (333), Membury warm (444).
        var nwp = new[]
        {
            new SitePages.NwpTemperatureForecastPoint("gfs_seamless", validTime, 333.0, "bonehill_rocks"),
            new SitePages.NwpTemperatureForecastPoint("gfs_seamless", validTime, 444.0, "membury_devon"),
        };
        var pred = new TempPredictionRow
        {
            LocationName = "membury_devon", ModelVersion = "v2c",
            PredictionMadeAtUtc = generatedAt,
            ValidTimeUtc = validTime, LeadHours = 24,
            BlendTemperature = 440.0, FeatureVectorHash = "",
        };

        var input = MakeInput(renderingFor: membury, locations: new[] { bonehill, membury })
            with
            {
                GeneratedAtUtc = generatedAt,
                Predictions = new[] { pred },
                NwpTemperatures = nwp,
                PhaseByVersion = new Dictionary<string, string> { ["v2c"] = "2b" },
            };

        var html = SitePages.RenderForecastsTemp(input, 24);

        // Membury's NWP point (444) is on the chart; Bonehill's (333) is not.
        html.Should().Contain("444");
        html.Should().NotContain("333");
    }

    [Fact]
    public void Top_nav_Models_link_points_at_rain_when_temperature_absent()
    {
        // Rain-only loc — Models button should go to models-rain.html (the
        // first available target), not models-temp.html (which won't exist).
        var rainOnly = MakeDescriptor("rain_only", "Rain Only",
            tabs: new[] { "overview", "rain", "models" }, isPrimary: false);
        var input = MakeInput(renderingFor: rainOnly);
        var html = SitePages.RenderRainSkill(input);

        // Match the live link from the per-loc site-nav (not the sub-nav
        // inside the page, which is the Skill sub-nav here).
        html.Should().MatchRegex("<li><a href=\"models-rain.html\"[^>]*>Models</a>");
        html.Should().NotContain("<li><a href=\"models-temp.html\"");
    }
}
