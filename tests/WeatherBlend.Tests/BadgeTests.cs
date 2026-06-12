using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Unified amber/red badge system (Harry's design 2026-06-12): one fired
/// trigger = amber, two-or-more = red, pop-out lists the fired triggers
/// with live values. Pure-evaluator tests for the low-cloud and sea-state
/// badges plus render-level coverage of the Sennen sea-state badge wiring
/// (degraded-direction case included — direction rows lag the speed tree).
/// </summary>
public class BadgeTests
{
    // Sennen's draft config.yaml thresholds — keep in sync conceptually,
    // but the tests pin their own values so a yaml recalibration doesn't
    // break them.
    private static readonly SeaStateBadgeSpec Spec = new(
        TideHighMsl: 1.71,
        RunUpProxy: 12.0,
        WindMinMph: 15.0,
        WindSectorFromDeg: 225,
        WindSectorToDeg: 335);

    // ------------------------------------------------------------------
    // Severity rule + low-cloud evaluator
    // ------------------------------------------------------------------

    [Theory]
    [InlineData(0, BadgeSeverity.None)]
    [InlineData(1, BadgeSeverity.Amber)]
    [InlineData(2, BadgeSeverity.Red)]
    [InlineData(3, BadgeSeverity.Red)]
    public void BadgeSeverityRule_one_amber_two_plus_red(int fired, BadgeSeverity expected) =>
        BadgeSeverityRule.FromFiredCount(fired).Should().Be(expected);

    [Theory]
    [InlineData(false, false, BadgeSeverity.None)]
    [InlineData(true, false, BadgeSeverity.Amber)]   // mist only
    [InlineData(false, true, BadgeSeverity.Amber)]   // cloud-base only
    [InlineData(true, true, BadgeSeverity.Red)]      // both signals
    public void LowCloudBadge_amber_for_one_signal_red_for_both(
        bool mist, bool cloudBase, BadgeSeverity expected) =>
        LowCloudBadge.Evaluate(mist, cloudBase).Should().Be(expected);

    // ------------------------------------------------------------------
    // Sea-state evaluator — single triggers
    // ------------------------------------------------------------------

    [Fact]
    public void SeaState_high_tide_alone_is_amber_with_live_values()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 1.8, waveHeightM: 1.0, swellPeriodS: 8.0,
            windMph: 5.0, windDirDeg: 270.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Amber);
        r.FiredTriggers.Should().ContainSingle()
            .Which.Should().Be("tide +1.8 m (≥ +1.71)");
    }

    [Fact]
    public void SeaState_big_waves_alone_is_amber_with_proxy_breakdown()
    {
        // √2.1 × 9.2 = 13.33… ≥ 12 — wave trigger only.
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 0.0, waveHeightM: 2.1, swellPeriodS: 9.2,
            windMph: 5.0, windDirDeg: 270.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Amber);
        r.FiredTriggers.Should().ContainSingle()
            .Which.Should().Be("run-up proxy 13.3 = √2.1 m × 9.2 s (≥ 12)");
    }

    [Fact]
    public void SeaState_onshore_wind_alone_is_amber_with_sector_detail()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 0.0, waveHeightM: 1.0, swellPeriodS: 8.0,
            windMph: 19.0, windDirDeg: 290.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Amber);
        r.FiredTriggers.Should().ContainSingle()
            .Which.Should().Be("wind 19 mph from 290° (≥ 15 from 225–335°)");
    }

    [Fact]
    public void SeaState_strong_wind_outside_sector_does_not_fire()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 0.0, waveHeightM: 1.0, swellPeriodS: 8.0,
            windMph: 30.0, windDirDeg: 90.0, Spec);   // offshore easterly

        r.Severity.Should().Be(BadgeSeverity.None);
        r.FiredTriggers.Should().BeEmpty();
    }

    [Fact]
    public void SeaState_in_sector_wind_below_min_speed_does_not_fire()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 0.0, waveHeightM: 1.0, swellPeriodS: 8.0,
            windMph: 14.9, windDirDeg: 290.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.None);
    }

    // ------------------------------------------------------------------
    // Pairs / all three → red
    // ------------------------------------------------------------------

    [Fact]
    public void SeaState_tide_plus_waves_is_red()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 2.4, waveHeightM: 2.5, swellPeriodS: 10.0,
            windMph: 5.0, windDirDeg: 270.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Red);
        r.FiredTriggers.Should().HaveCount(2);
        r.FiredTriggers[0].Should().StartWith("tide ");
        r.FiredTriggers[1].Should().StartWith("run-up proxy ");
    }

    [Fact]
    public void SeaState_tide_plus_wind_is_red()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 1.71, waveHeightM: 0.5, swellPeriodS: 6.0,
            windMph: 20.0, windDirDeg: 250.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Red);
        r.FiredTriggers.Should().HaveCount(2);
    }

    [Fact]
    public void SeaState_waves_plus_wind_is_red()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: -1.0, waveHeightM: 3.0, swellPeriodS: 12.0,
            windMph: 25.0, windDirDeg: 300.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Red);
        r.FiredTriggers.Should().HaveCount(2);
    }

    [Fact]
    public void SeaState_all_three_triggers_is_red_listing_all()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 2.0, waveHeightM: 3.0, swellPeriodS: 12.0,
            windMph: 25.0, windDirDeg: 300.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Red);
        r.FiredTriggers.Should().HaveCount(3);
    }

    // ------------------------------------------------------------------
    // Missing inputs → trigger skipped, never fired
    // ------------------------------------------------------------------

    [Fact]
    public void SeaState_missing_direction_skips_wind_and_badge_runs_on_tide_and_waves()
    {
        // Direction tree absent (Sennen wind_mvn rows start flowing later
        // than speed): the gale-force speed alone must NOT fire the wind
        // trigger; tide + waves still evaluate normally.
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 2.0, waveHeightM: 3.0, swellPeriodS: 12.0,
            windMph: 40.0, windDirDeg: null, Spec);

        r.Severity.Should().Be(BadgeSeverity.Red);   // tide + waves
        r.FiredTriggers.Should().HaveCount(2);
        r.FiredTriggers.Should().NotContain(t => t.StartsWith("wind"));
    }

    [Fact]
    public void SeaState_missing_wind_speed_skips_wind_trigger()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 2.0, waveHeightM: 0.5, swellPeriodS: 6.0,
            windMph: null, windDirDeg: 290.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Amber);   // tide only
        r.FiredTriggers.Should().ContainSingle().Which.Should().StartWith("tide");
    }

    [Fact]
    public void SeaState_missing_tide_skips_tide_trigger()
    {
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: null, waveHeightM: 3.0, swellPeriodS: 12.0,
            windMph: 5.0, windDirDeg: 270.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.Amber);   // waves only
        r.FiredTriggers.Should().ContainSingle().Which.Should().StartWith("run-up proxy");
    }

    [Fact]
    public void SeaState_missing_swell_period_skips_wave_trigger()
    {
        // Hs alone can't form the run-up proxy — pass-through period can gap
        // independently of the blended height.
        var r = SeaStateBadge.Evaluate(
            tideHeightMsl: 0.0, waveHeightM: 5.0, swellPeriodS: null,
            windMph: 5.0, windDirDeg: 270.0, Spec);

        r.Severity.Should().Be(BadgeSeverity.None);
    }

    [Fact]
    public void SeaState_everything_missing_or_calm_is_none()
    {
        SeaStateBadge.Evaluate(null, null, null, null, null, Spec)
            .Should().Be(SeaStateBadgeResult.None);
    }

    // ------------------------------------------------------------------
    // Boundary semantics — exactly-at threshold fires (inclusive)
    // ------------------------------------------------------------------

    [Fact]
    public void SeaState_tide_exactly_at_threshold_fires()
    {
        SeaStateBadge.Evaluate(1.71, null, null, null, null, Spec)
            .Severity.Should().Be(BadgeSeverity.Amber);
        SeaStateBadge.Evaluate(1.7099, null, null, null, null, Spec)
            .Severity.Should().Be(BadgeSeverity.None);
    }

    [Fact]
    public void SeaState_runup_proxy_exactly_at_threshold_fires()
    {
        // √4 × 6 = 12.0 exactly.
        SeaStateBadge.Evaluate(null, 4.0, 6.0, null, null, Spec)
            .Severity.Should().Be(BadgeSeverity.Amber);
        SeaStateBadge.Evaluate(null, 4.0, 5.99, null, null, Spec)
            .Severity.Should().Be(BadgeSeverity.None);
    }

    [Fact]
    public void SeaState_wind_exactly_at_speed_and_sector_edges_fires()
    {
        SeaStateBadge.Evaluate(null, null, null, 15.0, 225.0, Spec)
            .Severity.Should().Be(BadgeSeverity.Amber);   // both boundaries inclusive
        SeaStateBadge.Evaluate(null, null, null, 15.0, 335.0, Spec)
            .Severity.Should().Be(BadgeSeverity.Amber);
        SeaStateBadge.Evaluate(null, null, null, 15.0, 224.0, Spec)
            .Severity.Should().Be(BadgeSeverity.None);    // just outside sector
        SeaStateBadge.Evaluate(null, null, null, 15.0, 336.0, Spec)
            .Severity.Should().Be(BadgeSeverity.None);
    }

    // ------------------------------------------------------------------
    // Sector wrap through 360
    // ------------------------------------------------------------------

    [Theory]
    [InlineData(350.0, true)]
    [InlineData(0.0, true)]
    [InlineData(20.0, true)]
    [InlineData(60.0, true)]    // inclusive end
    [InlineData(300.0, true)]   // inclusive start
    [InlineData(180.0, false)]
    [InlineData(61.0, false)]
    [InlineData(299.0, false)]
    public void DirectionInSector_handles_wrap_through_360(double dir, bool expected) =>
        SeaStateBadge.DirectionInSector(dir, 300.0, 60.0).Should().Be(expected);

    [Theory]
    [InlineData(-70.0, true)]    // -70 ≡ 290, inside 225–335
    [InlineData(650.0, true)]    // 650 ≡ 290
    [InlineData(360.0, false)]   // ≡ 0, outside 225–335
    public void DirectionInSector_normalises_out_of_range_degrees(double dir, bool expected) =>
        SeaStateBadge.DirectionInSector(dir, 225.0, 335.0).Should().Be(expected);

    [Fact]
    public void SeaState_wind_fires_through_wrapping_sector()
    {
        var wrap = Spec with { WindSectorFromDeg = 300, WindSectorToDeg = 60 };
        SeaStateBadge.Evaluate(null, null, null, 20.0, 10.0, wrap)
            .Severity.Should().Be(BadgeSeverity.Amber);
        SeaStateBadge.Evaluate(null, null, null, 20.0, 180.0, wrap)
            .Severity.Should().Be(BadgeSeverity.None);
    }

    // ------------------------------------------------------------------
    // Render-level wiring — RenderIndex tiles
    // ------------------------------------------------------------------

    private static readonly DateTime GeneratedAt = new(2026, 6, 11, 0, 0, 0, DateTimeKind.Utc);
    private static readonly DateTime Valid = GeneratedAt.Date.AddHours(12);

    private static SitePages.LocationDescriptor SennenDescriptor(SeaStateBadgeSpec? spec) => new(
        Name: "sennen_cove",
        DisplayName: "Sennen, Cornwall",
        RainStationSlugs: Array.Empty<string>(),
        IsPrimary: false,
        Tabs: new[] { "overview", "sea_state" },
        OverviewFirstVisibleHourUtc: 6,
        OverviewLastVisibleHourUtcExclusive: 21,
        SeaStateBadge: spec);

    private static SitePages.SiteInputs MakeTileInput(SeaStateBadgeSpec? spec) => new()
    {
        ActiveLocationName = "sennen_cove",
        RenderingFor = SennenDescriptor(spec),
        LocationDisplay = "Sennen, Cornwall",
        Latitude = 50.0786, Longitude = -5.7044, ElevationMeters = 35,
        MetarStation = "",
        GeneratedAtUtc = GeneratedAt,
        WindowStartUtc = GeneratedAt.AddDays(-30),
        Predictions = new[]
        {
            new TempPredictionRow
            {
                LocationName = "sennen_cove", ModelVersion = "v", PredictionMadeAtUtc = GeneratedAt,
                ValidTimeUtc = Valid, LeadHours = 12, BlendTemperature = 14.0, FeatureVectorHash = "",
            },
        },
        CurrentVersion = "v",
        TruthByTime = new Dictionary<DateTime, double>(),
        MetarByTime = Array.Empty<(DateTime, double)>(),
        RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
        PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
        DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
        RainfallTruth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
    };

    private static SitePages.WaveForecastPoint WaveRow(
        double hs, double? swellPeriod, double? tide) => new(
        Version: "wv1", PredictedAtUtc: GeneratedAt, ValidTimeUtc: Valid, LeadHours: 12,
        WaveHeightM: hs, BandLoM: null, BandHiM: null,
        WavePeriodS: null, WaveDirectionDeg: null,
        SwellHeightM: null, SwellPeriodS: swellPeriod, SwellDirectionDeg: null,
        SecondarySwellHeightM: null, SecondarySwellPeriodS: null,
        TideHeightMsl: tide, SeaSurfaceTempC: null,
        LocationName: "sennen_cove");

    private static SitePages.WindForecastPoint WindRow(double speedMs) => new(
        Version: "w1", PredictedAtUtc: GeneratedAt, ValidTimeUtc: Valid, LeadHours: 12,
        SpeedMs: speedMs, Gfs: null, Ecmwf: null, Icon: null, Mf: null, Ukmo: null,
        Gem: null, Aifs: null, LocationName: "sennen_cove");

    private static SitePages.WindDirectionForecastPoint WindDirRow(double dirDeg) => new(
        Version: "wd1", PredictedAtUtc: GeneratedAt, ValidTimeUtc: Valid, LeadHours: 12,
        DirectionDeg: dirDeg, DirectionCi95LoDeg: dirDeg - 10, DirectionCi95HiDeg: dirDeg + 10,
        SpeedMs: 0, SpeedCi95LoMs: 0, SpeedCi95HiMs: 0, LocationName: "sennen_cove");

    [Fact]
    public void RenderIndex_sea_state_badge_red_with_all_triggers_listed()
    {
        // Tide + waves + onshore wind (10 m/s = 22.4 mph from 290°) all fire.
        var input = MakeTileInput(Spec) with
        {
            WavePredictions = new[] { WaveRow(hs: 3.0, swellPeriod: 12.0, tide: 2.0) },
            WindPredictions = new[] { WindRow(speedMs: 10.0) },
            WindDirectionPredictions = new[] { WindDirRow(290.0) },
        };

        var html = SitePages.RenderIndex(input, dayOffset: 0);

        html.Should().Contain("sea-badge").And.Contain("🌊 sea state");
        html.Should().Contain("badge-red");
        html.Should().Contain("tide +2.0 m (≥ +1.71)");
        html.Should().Contain("run-up proxy 20.8 = √3.0 m × 12.0 s (≥ 12)");
        html.Should().Contain("wind 22 mph from 290° (≥ 15 from 225–335°)");
    }

    [Fact]
    public void RenderIndex_sea_state_badge_degrades_when_direction_rows_absent()
    {
        // No wind_mvn rows (Sennen direction starts flowing later than
        // speed): the wind trigger is skipped, so strong wind + high tide
        // render an amber tide-only badge — never a fired wind line.
        var input = MakeTileInput(Spec) with
        {
            WavePredictions = new[] { WaveRow(hs: 0.5, swellPeriod: 6.0, tide: 2.0) },
            WindPredictions = new[] { WindRow(speedMs: 20.0) },   // 44.7 mph
            WindDirectionPredictions = Array.Empty<SitePages.WindDirectionForecastPoint>(),
        };

        var html = SitePages.RenderIndex(input, dayOffset: 0);

        html.Should().Contain("sea-badge").And.Contain("badge-amber");
        html.Should().NotContain("badge-red");
        html.Should().Contain("tide +2.0 m");
        html.Should().NotContain("wind 45 mph");
    }

    [Fact]
    public void RenderIndex_no_sea_state_badge_without_config_block_or_when_calm()
    {
        // Location without a seaStateBadge spec → no badge even with wild sea.
        var noSpec = MakeTileInput(spec: null) with
        {
            WavePredictions = new[] { WaveRow(hs: 5.0, swellPeriod: 15.0, tide: 3.0) },
        };
        SitePages.RenderIndex(noSpec, dayOffset: 0).Should().NotContain("sea-badge");

        // Spec present but nothing fired → no badge either.
        var calm = MakeTileInput(Spec) with
        {
            WavePredictions = new[] { WaveRow(hs: 0.4, swellPeriod: 5.0, tide: -1.2) },
        };
        SitePages.RenderIndex(calm, dayOffset: 0).Should().NotContain("sea-badge");
    }

    [Fact]
    public void RenderIndex_low_cloud_badge_is_amber_for_one_signal_red_for_both()
    {
        // One signal (cloud-base) over threshold → amber pill, one pop-out line.
        var oneSignal = MakeTileInput(spec: null) with
        {
            LowCloudByValid = new Dictionary<DateTime, SitePages.LowCloudSignal>
            {
                [Valid] = new(VisFiredCount: 1, VisTotalCount: 6,
                              CloudBaseFiredCount: 7, CloudBaseTotalCount: 11),
            },
        };
        var amberHtml = SitePages.RenderIndex(oneSignal, dayOffset: 0);
        amberHtml.Should().Contain("low-cloud-badge badge-amber").And.Contain("low cloud");
        amberHtml.Should().Contain("cloud base below tor").And.NotContain("mist (vis");

        // Both signals → red pill, both pop-out lines retained.
        var bothSignals = MakeTileInput(spec: null) with
        {
            LowCloudByValid = new Dictionary<DateTime, SitePages.LowCloudSignal>
            {
                [Valid] = new(VisFiredCount: 4, VisTotalCount: 6,
                              CloudBaseFiredCount: 8, CloudBaseTotalCount: 11),
            },
        };
        var redHtml = SitePages.RenderIndex(bothSignals, dayOffset: 0);
        redHtml.Should().Contain("low-cloud-badge badge-red");
        redHtml.Should().Contain("mist (vis").And.Contain("cloud base below tor");
    }

    [Fact]
    public void RenderIndex_rock_badge_styles_onto_shared_severity_classes()
    {
        // Trigger logic unchanged (greasy = amber tier, condensation = red
        // tier) — only the styling moved onto badge-amber / badge-red.
        var rock = new SitePages.RockSurfaceForecastPoint(
            "v1", GeneratedAt, Valid, 12,
            RockSurfaceTempC: 9.0, AirTempC: 11.0, DewPointC: 8.5,
            CondensationMarginC: 0.5, GreasinessStatus: "potentially_greasy",
            LocationName: "sennen_cove");

        var greasyHtml = SitePages.RenderIndex(
            MakeTileInput(spec: null) with { RockSurfacePredictions = new[] { rock } }, dayOffset: 0);
        greasyHtml.Should().Contain("rock-greasy badge-amber").And.Contain("rock greasy?");

        var wet = rock with { CondensationMarginC = -0.3, GreasinessStatus = "condensation" };
        var wetHtml = SitePages.RenderIndex(
            MakeTileInput(spec: null) with { RockSurfacePredictions = new[] { wet } }, dayOffset: 0);
        wetHtml.Should().Contain("rock-wet badge-red").And.Contain("rock wet");
    }
}
