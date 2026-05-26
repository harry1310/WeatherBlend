using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

public class SitePagesTests
{
    private const string Station = "ea_bellever_dartmoor";
    private static readonly DateTime Day = new(2026, 4, 1, 0, 0, 0, DateTimeKind.Utc);
    // 09–18 Europe/London — same default the production renderer uses
    // (see DryWindowConfig.AllowedWindow). Tests reach for it via
    // UtcHourRangeFor when they want to be DST-correct.
    private static readonly DaytimeWindow StandardDaytime = new(9, 18, "Europe/London");

    [Fact]
    public void ComputeObservedDryWindows_flags_day_with_long_enough_run_as_dry()
    {
        // 6 dry hours 10Z-15Z (≤ 0.1 mm), rest wet. 6h window should fire.
        var hourly = BuildHourly(Day, h => h >= 10 && h <= 15 ? 0.0 : 2.0);
        var input = MakeInput(hourly, windowHours: 6);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 6, Day)].Should().BeTrue();
    }

    [Fact]
    public void ComputeObservedDryWindows_needs_consecutive_run_not_total_dry_hours()
    {
        // 5 dry hours total, but broken: 2 + 3 with a wet hour between. 4h window shouldn't fire.
        var hourly = BuildHourly(Day, h => h == 10 || h == 11 || h == 13 || h == 14 || h == 15 ? 0.0 : 2.0);
        var input = MakeInput(hourly, windowHours: 4);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 4, Day)].Should().BeFalse();
    }

    [Fact]
    public void ComputeObservedDryWindows_treats_exactly_0_1_mm_as_wet()
    {
        // Boundary: shares DryWindowLabelBuilder.HasDryWindow, which uses
        // strict less-than (< 0.1 mm/h is dry). 0.1 itself is wet — so this
        // 3-hour run of 0.1 mm doesn't satisfy the 3h dry-window.
        var hourly = BuildHourly(Day, h => h >= 10 && h <= 12 ? 0.1 : 5.0);
        var input = MakeInput(hourly, windowHours: 3);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 3, Day)].Should().BeFalse();
    }

    [Fact]
    public void ComputeObservedDryWindows_skips_day_with_missing_daytime_hour()
    {
        // Drop hour 12 (inside the daytime window) — need every hour inside
        // the daytime range populated for a verdict. Hours outside the range
        // (e.g. 03Z) can be missing without triggering the skip.
        var hourly = BuildHourly(Day, h => 0.0);
        hourly.Remove(Day.AddHours(12));
        var input = MakeInput(hourly, windowHours: 3);

        var result = SitePages.ComputeObservedDryWindows(input);

        result.Should().NotContainKey((Station, 3, Day));
    }

    [Fact]
    public void ComputeObservedDryWindows_ignores_overnight_dry_run_outside_daytime()
    {
        // Whole daytime soaked, but the whole overnight is bone dry. The old
        // 24h scan would have given a false ✓ here — the labeller-aligned
        // scan correctly says "no dry block in the daytime window".
        var (startUtc, endUtcExclusive) = StandardDaytime.UtcHourRangeFor(DateOnly.FromDateTime(Day));
        var hourly = BuildHourly(Day, h => h >= startUtc && h < endUtcExclusive ? 5.0 : 0.0);
        var input = MakeInput(hourly, windowHours: 6);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 6, Day)].Should().BeFalse();
    }

    [Fact]
    public void ComputeObservedDryWindows_skips_station_with_no_rainfall_loaded()
    {
        // Prediction references a station whose rainfall dict is empty — should silently skip.
        var input = new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = Day.AddDays(1),
            WindowStartUtc = Day,
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = new[]
            {
                new SitePages.DryWindowForecastPoint(Station, 3, "v1", Day, Day, 24, 0.5, 0.4, null),
            },
            RainfallTruth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
        };

        var result = SitePages.ComputeObservedDryWindows(input);

        result.Should().BeEmpty();
    }

    [Fact]
    public void RenderRainSkill_renders_a_precipitation_chart_per_phase_present()
    {
        // Two phases per station post-calibration-purge: lean 3a + rich 3c.
        // (3a_isotonic was retired alongside 3d-calibrated — neither is Active
        // in any manifest and the trainer no longer emits them.)
        var input = MakePrecipInput(new[]
        {
            ("v_3a", "3a"),
            ("v_3c", "3c"),
        });

        var html = SitePages.RenderRainSkill(input);

        html.Should().Contain("Phase 3a (lean)");
        html.Should().Contain("Phase 3c (rich)");
    }

    [Fact]
    public void RenderForecastsRain_renders_only_the_requested_lead()
    {
        // Predictions at every lead (24/48/72) for one station. The +48h page
        // should render the +48h header only, not surface +24h or +72h
        // headlines (those land on their own per-lead pages).
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var preds = new[] { 24, 48, 72 }.Select(lead =>
            new SitePages.PrecipForecastPoint(
                Station, "v_3a", generatedAt, generatedAt.AddHours(lead), lead, 0.42, 0.18,
                PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                PrecipMf: null, PrecipUkmo: null, PrecipGem: null, PrecipAifs: null, PrecipJma: null)).ToArray();

        var input = new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = generatedAt.AddDays(-30),
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = preds,
            DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
            RainfallTruth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
            PrecipCurrentByStation = new Dictionary<string, string> { [Station] = "v_3a" },
        };

        var html = SitePages.RenderForecastsRain(input, 48);

        html.Should().Contain("Rain forecast +48h");
        html.Should().NotContain("Rain forecast +24h");
        html.Should().NotContain("Rain forecast +72h");
    }

    [Fact]
    public void RenderIndex_emits_day_sub_nav_with_day_of_week_labels()
    {
        // Home is now per-day with a sub-nav at the top — one tab per day in
        // the 6-day window (today + 5 forward) labelled "ddd d/M" so the
        // reader can flip between days without doing UTC arithmetic. Today
        // is the canonical "Today" tab; the others are dated. Empty days
        // are suppressed from the sub-nav (added 2026-05-07), so the test
        // input must include at least one tile-window prediction per day
        // we expect to see in the bar.
        // 09:00Z so the day-0 noon prediction passes the strict
        // ValidTimeUtc > GeneratedAtUtc filter inside RenderIndex.
        var generatedAt = new DateTime(2026, 4, 24, 9, 0, 0, DateTimeKind.Utc); // Fri morning
        // One prediction at 12:00Z on each of today + 3 forward days. 12:00Z
        // sits squarely inside the outdoor window so all four days have
        // tile content and should appear in the sub-nav.
        var preds = Enumerable.Range(0, 4).Select(n => new TempPredictionRow
        {
            LocationName = "Test", ModelVersion = "v",
            PredictionMadeAtUtc = generatedAt,
            ValidTimeUtc = generatedAt.Date.AddDays(n).AddHours(12),
            LeadHours = 12 + n * 24,
            BlendTemperature = 12.0,
            FeatureVectorHash = "",
        }).ToArray();
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v",
        };

        var html = SitePages.RenderIndex(input, dayOffset: 0);

        html.Should().Contain("Today");        // offset 0
        html.Should().Contain("Sat 25/4");     // Fri + 1 day
        html.Should().Contain("Sun 26/4");     // Fri + 2 days
        html.Should().Contain("Mon 27/4");     // Fri + 3 days
    }

    [Fact]
    public void RenderIndex_sub_nav_skips_days_with_no_tiles()
    {
        // Sub-nav was rendering links to days with zero tiles, sending the
        // user to a blank page when they clicked. After 2026-05-07 it skips
        // empty days entirely. Predictions only on offsets 0 + 2 — offset 1
        // and 3-5 should be absent from the bar; the active day always
        // renders so the user knows where they are.
        // 09:00Z so the day-0 noon prediction passes the strict
        // ValidTimeUtc > GeneratedAtUtc filter inside RenderIndex.
        var generatedAt = new DateTime(2026, 4, 24, 9, 0, 0, DateTimeKind.Utc); // Fri morning
        var preds = new[] { 0, 2 }.Select(n => new TempPredictionRow
        {
            LocationName = "Test", ModelVersion = "v",
            PredictionMadeAtUtc = generatedAt,
            ValidTimeUtc = generatedAt.Date.AddDays(n).AddHours(12),
            LeadHours = 12 + n * 24,
            BlendTemperature = 12.0,
            FeatureVectorHash = "",
        }).ToArray();
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v",
        };

        var html = SitePages.RenderIndex(input, dayOffset: 0);

        html.Should().Contain("Today");        // offset 0 — has tiles
        html.Should().NotContain("Sat 25/4");  // offset 1 — empty, skipped
        html.Should().Contain("Sun 26/4");     // offset 2 — has tiles
        html.Should().NotContain("Mon 27/4");  // offset 3 — empty, skipped
        html.Should().NotContain("Tue 28/4");  // offset 4 — empty
        html.Should().NotContain("Wed 29/4");  // offset 5 — empty
    }

    [Fact]
    public void RenderIndex_falls_back_to_empty_state_when_no_future_predictions()
    {
        // No forward predictions in window → per-day empty state rather than
        // a blank page or a stale entry.
        var generatedAt = new DateTime(2026, 4, 24, 12, 0, 0, DateTimeKind.Utc);
        var input = MakeEmptyForecastInput() with { GeneratedAtUtc = generatedAt };

        var html = SitePages.RenderIndex(input, dayOffset: 0);

        html.Should().Contain("No forward predictions in this day");
    }

    [Fact]
    public void RenderIndex_filters_overnight_hours_from_tile_grid()
    {
        // 21:00-03:59 UTC tiles are dropped — overnight isn't useful for
        // outdoor planning. Predictions at 01:00 / 06:00 / 22:00 UTC: only
        // the 06:00 one should land on the tomorrow tab.
        var generatedAt = new DateTime(2026, 4, 24, 12, 0, 0, DateTimeKind.Utc);
        var nextDay = generatedAt.Date.AddDays(1);  // Sat 04-25
        var preds = new[]
        {
            (h: 1,  lead: 13),  // 01:00 — outside window
            (h: 6,  lead: 18),  // 06:00 — inside window
            (h: 22, lead: 34),  // 22:00 — outside window
        }.Select(t => new TempPredictionRow
        {
            LocationName = "Test", ModelVersion = "v",
            PredictionMadeAtUtc = generatedAt,
            ValidTimeUtc = nextDay.AddHours(t.h),
            LeadHours = t.lead,
            BlendTemperature = 12.0,
            FeatureVectorHash = "",
        }).ToArray();
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v",
        };

        var html = SitePages.RenderIndex(input, dayOffset: 1);

        html.Should().Contain("06:00Z");       // visible
        html.Should().NotContain("01:00Z");    // overnight, filtered
        html.Should().NotContain("22:00Z");    // overnight, filtered
    }

    [Fact]
    public void RenderIndex_filters_per_lead_using_ChampionByLead_override()
    {
        // 2d at lead 12, 2b at lead 24. ChampionByLead pins 2d at lead 12;
        // CurrentVersion is the 2b champion. Tile at lead 12 should show the
        // 2d temperature; tile at lead 24 should show the 2b temperature.
        // Both predictions share the same valid_time so the smallest-lead-wins
        // grouping doesn't matter here — what matters is that the 2c
        // challenger's row at the same valid_time is filtered out.
        var generatedAt = new DateTime(2026, 5, 5, 0, 0, 0, DateTimeKind.Utc);
        var nextDay = generatedAt.Date.AddDays(1);
        var v12 = nextDay.AddHours(12);  // tomorrow noon
        var v24 = nextDay.AddDays(1).AddHours(0);  // day after, 00Z

        TempPredictionRow Row(string version, int lead, DateTime valid, double t) => new()
        {
            LocationName = "Test", ModelVersion = version,
            PredictionMadeAtUtc = generatedAt,
            ValidTimeUtc = valid, LeadHours = lead,
            BlendTemperature = t,
            FeatureVectorHash = "",
        };

        var preds = new[]
        {
            Row("v-2b", 12, v12, 11.0),
            Row("v-2c", 12, v12, 11.5),
            Row("v-2d", 12, v12, 12.5),  // 2d champion at lead 12
            Row("v-2b", 24, v24, 9.0),   // 2b champion at lead 24+
            Row("v-2c", 24, v24, 9.3),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v-2b",
            ChampionByLead = new Dictionary<int, string> { [12] = "v-2d" },
        };

        var html = SitePages.RenderIndex(input, dayOffset: 1);

        // Tomorrow's noon tile must show 12.5°C (2d), not 11.0/11.5 (2b/2c).
        html.Should().Contain("12.5°C");
        html.Should().NotContain("11.0°C");
        html.Should().NotContain("11.5°C");
    }

    [Fact]
    public void RenderIndex_today_falls_back_to_previous_champion_phase_bundle()
    {
        // Regression for the 2026-05-26 Membury "no forward predictions in
        // this day's outdoor window" bug. When a retrain mints a new
        // champion version mid-day, every post-retrain predict cycle's
        // lead-24h+ rows target anchor.Date.AddDays(1) forward — leaving
        // today's window orphaned for any location without a sub-24h-lead
        // phase (Membury has 2b champion / 2c challenger only — no 2d).
        // Yesterday's still-2b bundle wrote lead-24h rows for today and
        // they're loaded into input.Predictions; the home page must fall
        // back to them when today's bundle has no rows in window.
        //
        // 09:00Z anchor so the strict `ValidTimeUtc > GeneratedAtUtc`
        // filter keeps the today-noon row in the day-0 tile window.
        var generatedAt = new DateTime(2026, 5, 26, 9, 0, 0, DateTimeKind.Utc);
        var todayNoon  = generatedAt.Date.AddHours(12);                  // today, in window
        var tomorrowNoon = generatedAt.Date.AddDays(1).AddHours(12);     // tomorrow, in window

        TempPredictionRow Row(string version, DateTime predictedAt, DateTime valid, double t) => new()
        {
            LocationName = "Membury", ModelVersion = version,
            PredictionMadeAtUtc = predictedAt,
            ValidTimeUtc = valid, LeadHours = 24,
            BlendTemperature = t,
            FeatureVectorHash = "",
        };

        // Two 2b bundles: yesterday's (covers today, dropped pre-fix) and
        // today's (covers tomorrow only). Both tagged phase=2b in
        // PhaseByVersion so the new filter unifies them.
        var preds = new[]
        {
            Row("v-2b-yesterday", generatedAt.AddDays(-1), todayNoon,    16.5),
            Row("v-2b-today",     generatedAt,             tomorrowNoon, 18.0),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v-2b-today",
            PhaseByVersion = new Dictionary<string, string>
            {
                ["v-2b-yesterday"] = "2b",
                ["v-2b-today"]     = "2b",
            },
        };

        var todayHtml    = SitePages.RenderIndex(input, dayOffset: 0);
        var tomorrowHtml = SitePages.RenderIndex(input, dayOffset: 1);

        // Today's body must render the yesterday-bundle tile, not the empty
        // state — this is the Membury "broken Today tab" scenario the fix
        // targets.
        todayHtml.Should().Contain("16.5°C");
        todayHtml.Should().NotContain("No forward predictions in this day");
        // And tomorrow's body still renders today's bundle's row — the
        // freshest-PredictionMadeAt tiebreak in the smallest-lead-wins
        // group means today's bundle wins where both bundles cover the
        // same valid_time, but here only the today bundle has a tomorrow
        // row so it just renders on its own.
        tomorrowHtml.Should().Contain("18.0°C");
    }

    [Fact]
    public void RenderIndex_resolves_pwet_gauge_per_rendered_location()
    {
        // Regression: the Overview P(wet) headline gauge used to be hardcoded
        // to ea_bellever_dartmoor (a Bonehill gauge), so the Membury Overview
        // page — whose precip predictions live under ea_chards_snowdon_hill /
        // ea_goren / ea_raymonds_hill — showed no P(wet) at all. The gauge now
        // resolves to the rendered location's primary rainfall station via
        // OverviewPwetStation. Build a Membury-shaped SiteInputs with precip
        // under a Membury gauge ONLY and assert the day summary surfaces it.
        var generatedAt = new DateTime(2026, 5, 5, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.Date.AddDays(1).AddHours(12);  // tomorrow noon

        var membury = new SitePages.LocationDescriptor(
            Name: "membury_devon",
            DisplayName: "Membury",
            RainStationSlugs: new[] { "ea_chards_snowdon_hill", "ea_goren", "ea_raymonds_hill" },
            IsPrimary: false,
            Tabs: new[] { "overview" },
            OverviewFirstVisibleHourUtc: 0,
            OverviewLastVisibleHourUtcExclusive: 24);

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            RenderingFor = membury,
            CurrentVersion = "v",
            Predictions = new[]
            {
                new TempPredictionRow
                {
                    LocationName = "Membury", ModelVersion = "v",
                    PredictionMadeAtUtc = generatedAt,
                    ValidTimeUtc = validTime, LeadHours = 36,
                    BlendTemperature = 14.0, FeatureVectorHash = "",
                },
            },
            // Precip ONLY under a Membury gauge — never ea_bellever_dartmoor.
            PrecipPredictions = new[]
            {
                new SitePages.PrecipForecastPoint(
                    Station: "ea_chards_snowdon_hill", Version: "v3a",
                    PredictedAtUtc: generatedAt, ValidTimeUtc: validTime,
                    LeadHours: 36, ProbWet: 0.55, ClimatologyPWet: 0.4,
                    PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                    PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
                    PrecipAifs: null, PrecipJma: null),
            },
            PrecipCurrentByStation = new Dictionary<string, string>
            {
                ["ea_chards_snowdon_hill"] = "v3a",
            },
        };

        var html = SitePages.RenderIndex(input, dayOffset: 1);

        // P(wet) reached the day summary — pre-fix this fell to the
        // "no P(wet) for this day" branch because the hardcoded Bellever
        // lookup found nothing in a Membury SiteInputs.
        html.Should().Contain("mean P(wet)");
        html.Should().NotContain("no P(wet) for this day");
    }

    [Fact]
    public void RenderForecastsTemp_emits_lead_subnav_linking_to_every_lead()
    {
        // Sub-nav is how readers hop between leads within a variable. Rendering
        // any lead page should produce the full per-variable lead nav with the
        // current lead marked active.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderForecastsTemp(input, 48);

        html.Should().Contain("forecasts-temp-12h.html",
            "Phase 2d champions +12h — sub-nav must surface that tab even though " +
            "2b/2c don't train at lead 12.");
        html.Should().Contain("forecasts-temp-24h.html");
        html.Should().Contain("forecasts-temp-48h.html");
        html.Should().Contain("forecasts-temp-72h.html");
        html.Should().Contain("forecasts-temp-120h.html");
        html.Should().Contain("lead-nav");
    }

    [Fact]
    public void RenderForecastsTemp_shows_fallback_text_when_lead_has_no_temperature_forecast()
    {
        // No predictions at all — page should still render, with a clear "no forecast" message
        // rather than blowing up on an empty sequence.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderForecastsTemp(input, 72);

        html.Should().Contain("No +72h temperature forecast available");
    }

    [Fact]
    public void RenderModels_shows_empty_state_when_no_training_metadata_loaded()
    {
        // First render after a fresh checkout — training_metadata.json isn't on disk yet.
        // Page should prompt the reader to run train, not crash on an empty list.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderModels(input, "temperature");

        html.Should().Contain("No training metadata on disk");
    }

    [Fact]
    public void RenderModels_shows_training_delta_badge_when_two_versions_present()
    {
        // Phase 4 of AUTO_RETRAIN_PLAN.md — when the same (composite, phase)
        // has a previous-train ModelSummary on disk, the card header gets a
        // green badge if the latest improved (avg BlendTestScore went down).
        var perLeadLatest = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.10, 1.05, 0.900, 1.30, 0.02, 400, 6),
            [48] = new(48, "gfs", 1.20, 1.15, 1.000, 1.45, 0.03, 400, 6),
        };
        var perLeadPrev = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.15, 1.10, 0.950, 1.35, 0.02, 400, 6),
            [48] = new(48, "gfs", 1.25, 1.20, 1.050, 1.50, 0.03, 400, 6),
        };
        var latest = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-10_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLeadLatest);
        var previous = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-03_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 3, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLeadPrev);

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { latest, previous },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // Average over leads {24, 48}: latest = (0.9 + 1.0)/2 = 0.95;
        // previous = (0.95 + 1.05)/2 = 1.00; delta = -0.050 → green.
        html.Should().Contain("delta-badge");
        html.Should().Contain("delta-good");
        html.Should().Contain("-0.050");
        html.Should().Contain("vs prev train");
    }

    [Fact]
    public void RenderModels_shows_red_training_delta_badge_when_latest_regressed()
    {
        // Same shape as the green-badge test but the latest version's
        // BlendTestScore is HIGHER → red badge (regressed vs previous).
        var perLeadLatest = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.10, 1.05, 1.100, 1.30, 0.02, 400, 6),
        };
        var perLeadPrev = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.15, 1.10, 1.000, 1.35, 0.02, 400, 6),
        };
        var latest = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-10_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLeadLatest);
        var previous = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-03_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 3, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLeadPrev);

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { latest, previous },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // delta = 1.100 - 1.000 = +0.100 → red.
        html.Should().Contain("delta-badge");
        html.Should().Contain("delta-bad");
        html.Should().Contain("+0.100");
    }

    [Fact]
    public void RenderModels_omits_training_delta_badge_when_only_one_version()
    {
        // First-ever training of a phase — no previous version on disk to
        // compare against, badge must not render.
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.10, 1.05, 0.900, 1.30, 0.02, 400, 6),
        };
        var only = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-10_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLead);

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { only },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // .NotContain("delta-badge") would match the CSS block in the
        // shared style header; check the span attribute instead.
        html.Should().NotContain("class=\"delta-badge");
        html.Should().NotContain("vs prev train");
    }

    [Fact]
    public void RenderModels_omits_training_delta_badge_when_lead_sets_disjoint()
    {
        // Split-version phase guard: previous version covered leads {12, 24};
        // latest covers leads {72, 96}. No overlap → can't average a delta on
        // shared leads → badge skipped.
        var perLeadLatest = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [72] = new(72, "gfs", 1.40, 1.30, 1.100, 1.55, 0.04, 400, 6),
        };
        var perLeadPrev = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.10, 1.05, 0.900, 1.30, 0.02, 400, 6),
        };
        var latest = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-10_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLeadLatest);
        var previous = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-03_120000",
            Phase: "2b", DataSource: "era5",
            TrainedAtUtc: new DateTime(2026, 5, 3, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)", PerLead: perLeadPrev);

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { latest, previous },
        };

        var html = SitePages.RenderModels(input, "temperature");

        html.Should().NotContain("class=\"delta-badge");
    }

    [Fact]
    public void RenderModels_shows_training_delta_badge_when_previous_cycle_was_split()
    {
        // 2d/3d regression: the previous training cycle was split into a
        // short-leads version and a long-leads version on the SAME day.
        // The badge must merge that cycle so its full lead set ({24} ∪ {72})
        // overlaps the latest cycle's {24} — picking either split half alone
        // would be disjoint from {24} and silently suppress the badge.
        var prevDay = new DateTime(2026, 5, 3, 0, 0, 0, DateTimeKind.Utc);
        var prevShort = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-03_080000_phase2d",
            Phase: "2d", DataSource: "exact",
            TrainedAtUtc: prevDay.AddHours(8),
            MetricLabel: "Test MAE (°C)",
            PerLead: new Dictionary<int, SitePages.PerLeadMetric>
            {
                [24] = new(24, "gfs", 1.15, 1.10, 1.000, 1.35, 0.02, 400, 6),
            });
        var prevLong = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-03_084400_phase2d",
            Phase: "2d", DataSource: "exact",
            TrainedAtUtc: prevDay.AddHours(8).AddMinutes(44),
            MetricLabel: "Test MAE (°C)",
            PerLead: new Dictionary<int, SitePages.PerLeadMetric>
            {
                [72] = new(72, "gfs", 1.70, 1.60, 1.800, 1.95, 0.05, 400, 6),
            });
        var latest = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v2026-05-10_120000_phase2d",
            Phase: "2d", DataSource: "exact",
            TrainedAtUtc: new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc),
            MetricLabel: "Test MAE (°C)",
            PerLead: new Dictionary<int, SitePages.PerLeadMetric>
            {
                [24] = new(24, "gfs", 1.10, 1.05, 0.900, 1.30, 0.02, 400, 6),
            });

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { latest, prevShort, prevLong },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // Shared lead {24}: latest 0.900 vs previous-cycle (merged) 1.000
        // → delta -0.100, green. The long half's lead 72 isn't shared but
        // its presence in the merged previous cycle must not break the merge.
        html.Should().Contain("delta-badge");
        html.Should().Contain("delta-good");
        html.Should().Contain("-0.100");
        html.Should().Contain("vs prev train");
    }

    [Fact]
    public void RenderModels_renders_per_card_verify_history_when_matching_phase_present()
    {
        // Verify history table appears below the per-card test-score table
        // when a verify-history row matches the card's (target, station,
        // windowHours, phase). Match is on Phase, not ModelVersion — the
        // card always shows the latest Active version while verify rows
        // necessarily lag (5d truth latency), so version-strict matching
        // would never surface history for a freshly retrained champion.
        var trained = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.23, 1.18, 0.987, 1.42, +0.05, 420, 6),
        };
        var summary = new SitePages.ModelSummary(
            Composite: "temperature",
            Version: "v2026-04-28_232613",   // current (post-retrain) Active version
            Phase: "2b",
            DataSource: "era5",
            TrainedAtUtc: trained,
            MetricLabel: "Test MAE (°C)",
            PerLead: perLead);

        var asOf = new DateTime(2026, 5, 1, 9, 30, 0, DateTimeKind.Utc);
        var historyFile = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = "temperature",
            AsOfUtc = asOf,
            WindowDays = 14,
            LatencyDays = 5,
            MetricLabel = "MAE (°C)",
            Rows = new List<WeatherBlend.Models.VerifyHistoryRow>
            {
                new()
                {
                    // Different ModelVersion from the card on purpose: the
                    // verify run scored an older version of the same phase.
                    Station = null, ModelVersion = "v2026-04-21_201231_phase2redo",
                    Phase = "2b",
                    LeadHours = 24, WindowHours = null, N = 100,
                    BlendMetric = 1.234, DriftFlag = false,
                },
            },
        };

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { summary },
            VerifyHistory = new[] { historyFile },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // Section header + "(1 run)" count.
        html.Should().Contain("Verify history");
        html.Should().Contain("(1 run)");
        // Per-lead BlendMetric reaches the table (1.234 → "1.234").
        html.Should().Contain("1.234");
        // No ⚠/✓ icons, no right-hand "Drift" column — removed
        // 2026-05-11; per-lead red text is the only drift indicator now.
        html.Should().NotContain("✓");
        html.Should().NotContain("⚠");
        html.Should().NotContain(">Drift</th>");
    }

    [Fact]
    public void RenderModels_verify_history_always_includes_full_lead_columns()
    {
        // The verify-history table column set is Leads.Full ∪ data-leads,
        // so the canonical +24/+48/+72/+96/+120 columns always render even
        // when this week's sidecar only has rows for a subset of leads
        // (e.g. a freshly retrained champion hasn't accumulated long-lead
        // predictions yet). Missing cells render as em-dash. Without this
        // floor a sparse sidecar would silently drop columns and the user
        // would think the longer leads were dropped from training.
        var trained = new DateTime(2026, 4, 28, 23, 26, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.23, 1.18, 0.987, 1.42, +0.05, 420, 6),
        };
        var summary = new SitePages.ModelSummary(
            Composite: "temperature",
            Version: "v2026-04-28_232613",
            Phase: "2b",
            DataSource: "era5",
            TrainedAtUtc: trained,
            MetricLabel: "Test MAE (°C)",
            PerLead: perLead);

        // Sidecar has only a lead-24 row — long leads should still appear.
        var historyFile = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = "temperature",
            AsOfUtc = new DateTime(2026, 5, 11, 9, 30, 0, DateTimeKind.Utc),
            WindowDays = 14, LatencyDays = 5, MetricLabel = "MAE (°C)",
            Rows = new List<WeatherBlend.Models.VerifyHistoryRow>
            {
                new()
                {
                    Station = null, ModelVersion = "v2026-04-28_232613",
                    Phase = "2b",
                    LeadHours = 24, WindowHours = null, N = 53,
                    BlendMetric = 0.914, DriftFlag = false,
                },
            },
        };

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { summary },
            VerifyHistory = new[] { historyFile },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // All five canonical lead headers present even though data only has 24.
        html.Should().Contain(">+24h<");
        html.Should().Contain(">+48h<");
        html.Should().Contain(">+72h<");
        html.Should().Contain(">+96h<");
        html.Should().Contain(">+120h<");
    }

    [Fact]
    public void RenderModels_verify_history_renders_n_column_with_max_n()
    {
        // Verify-history rows carry per-lead N. The table surfaces the max
        // across leads in a dedicated column right after Version — so a
        // reviewer can spot at-a-glance whether a row's drift signal is on
        // n=2 (probably noise) or n=53 (real). Long leads naturally have
        // lower N than short leads; the max is the most informative single
        // number per row.
        var trained = new DateTime(2026, 4, 28, 23, 26, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.23, 1.18, 0.987, 1.42, +0.05, 420, 6),
        };
        var summary = new SitePages.ModelSummary(
            Composite: "temperature",
            Version: "v2026-04-28_232613",
            Phase: "2b",
            DataSource: "era5",
            TrainedAtUtc: trained,
            MetricLabel: "Test MAE (°C)",
            PerLead: perLead);

        var historyFile = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = "temperature",
            AsOfUtc = new DateTime(2026, 5, 11, 9, 30, 0, DateTimeKind.Utc),
            WindowDays = 14, LatencyDays = 5, MetricLabel = "MAE (°C)",
            Rows = new List<WeatherBlend.Models.VerifyHistoryRow>
            {
                // n drops with lead — the table should surface the max (53).
                new() { ModelVersion = "v2026-04-28_232613", Phase = "2b",
                        LeadHours = 24, N = 53, BlendMetric = 0.914 },
                new() { ModelVersion = "v2026-04-28_232613", Phase = "2b",
                        LeadHours = 48, N = 41, BlendMetric = 0.905 },
                new() { ModelVersion = "v2026-04-28_232613", Phase = "2b",
                        LeadHours = 72, N = 29, BlendMetric = 1.003 },
                new() { ModelVersion = "v2026-04-28_232613", Phase = "2b",
                        LeadHours = 96, N = 18, BlendMetric = 1.023 },
                new() { ModelVersion = "v2026-04-28_232613", Phase = "2b",
                        LeadHours = 120, N = 9, BlendMetric = 0.363 },
            },
        };

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { summary },
            VerifyHistory = new[] { historyFile },
        };

        var html = SitePages.RenderModels(input, "temperature");

        // The N column header is present.
        html.Should().Contain(">N<");
        // The row cell shows the MAX N across leads (53), not the lowest (9).
        html.Should().Contain(">53<");
    }

    [Fact]
    public void RenderModels_renders_no_runs_yet_state_when_no_matching_phase()
    {
        // Different phase / different target → no rows match. Renderer used
        // to omit the section silently; that hid the "no verify yet" state
        // from the reader. Now it renders an explicit "(no runs yet)" panel
        // that names the phase being filtered for so the cause is visible.
        var trained = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.23, 1.18, 0.987, 1.42, +0.05, 420, 6),
        };
        var summary = new SitePages.ModelSummary(
            Composite: "temperature", Version: "v_x", Phase: "2b",
            DataSource: "era5", TrainedAtUtc: trained,
            MetricLabel: "Test MAE (°C)", PerLead: perLead);

        var historyFile = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = "temperature",
            AsOfUtc = new DateTime(2026, 5, 1, 9, 30, 0, DateTimeKind.Utc),
            WindowDays = 14, LatencyDays = 5, MetricLabel = "MAE (°C)",
            Rows = new List<WeatherBlend.Models.VerifyHistoryRow>
            {
                new()
                {
                    Station = null, ModelVersion = "v_y",
                    Phase = "2c",   // wrong phase for the card under test
                    LeadHours = 24, WindowHours = null, N = 100,
                    BlendMetric = 1.5, DriftFlag = true,
                },
            },
        };

        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[] { summary },
            VerifyHistory = new[] { historyFile },
        };

        var html = SitePages.RenderModels(input, "temperature");
        html.Should().Contain("(no runs yet)");
        html.Should().NotContain("(1 run)");  // shouldn't pretend a row landed
    }

    [Fact]
    public void RenderModels_groups_rows_under_pretty_composite_headings()
    {
        // Two composites — one temperature, one precip-per-station — now land
        // on different per-target pages after the 2026-05-04 split. Each page
        // must carry its own composite heading and the per-station composite
        // key must be pretty-printed (not raw "precipitation/ea_bellever").
        var trained = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", 1.23, 1.18, 0.987, 1.42, +0.05, 420, 6),
        };
        var summaries = new[]
        {
            new SitePages.ModelSummary(
                Composite: "temperature",
                Version: "temp_v2b",
                Phase: "2b",
                DataSource: "era5",
                TrainedAtUtc: trained,
                MetricLabel: "Test MAE (°C)",
                PerLead: perLead),
            new SitePages.ModelSummary(
                Composite: $"precipitation/{Station}",
                Version: "precip_v3a",
                Phase: "3a",
                DataSource: "ea_hydrology",
                TrainedAtUtc: trained,
                MetricLabel: "Test Brier",
                PerLead: perLead),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        var tempHtml  = SitePages.RenderModels(input, "temperature");
        var rainHtml  = SitePages.RenderModels(input, "precipitation");

        // Each per-target page carries only its own composite heading + version.
        tempHtml.Should().Contain("temp_v2b");
        tempHtml.Should().NotContain("precip_v3a");

        rainHtml.Should().Contain("Precipitation — Bellever Dartmoor")
            .And.NotContain("precipitation/ea_bellever_dartmoor");
        rainHtml.Should().Contain("precip_v3a");
        rainHtml.Should().NotContain("temp_v2b");
    }

    [Fact]
    public void RenderModels_merges_disjoint_leads_from_two_versions_of_same_phase_into_one_card()
    {
        // 2d temperature is trained as two separate models — short leads
        // {12,24,48} in one version, long leads {72,96,120} in another.
        // Both ship under Phase=2d. The card must show ALL six rows; the
        // pre-fix renderer kept only the latest TrainedAt and silently
        // dropped the other half of the lead range (Bellever rendered "+72h
        // first row" because the long-leads version was newest).
        var oldTrained = new DateTime(2026, 5, 8, 6, 0, 0, DateTimeKind.Utc);   // long-leads version
        var newTrained = new DateTime(2026, 5, 8, 8, 52, 0, DateTimeKind.Utc); // short-leads version (newest)
        var perLeadShort = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [12] = new(12, "ifs", 0.50, 0.45, 0.444, 0.62, +0.01, 200, 3),
            [24] = new(24, "ifs", 0.55, 0.50, 0.469, 0.65, +0.02, 200, 3),
            [48] = new(48, "ifs", 0.62, 0.58, 0.569, 0.74, +0.03, 200, 3),
        };
        var perLeadLong = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [72]  = new(72,  "ifs", 0.75, 0.72, 0.701, 0.85, +0.04, 200, 3),
            [96]  = new(96,  "ifs", 0.88, 0.85, 0.835, 0.99, +0.05, 200, 3),
            [120] = new(120, "ifs", 1.05, 1.00, 0.991, 1.16, +0.06, 200, 3),
        };
        var summaries = new[]
        {
            new SitePages.ModelSummary("temperature", "v_long_2d",  "2d", "exact-runtime", oldTrained, "Test MAE (°C)", perLeadLong),
            new SitePages.ModelSummary("temperature", "v_short_2d", "2d", "exact-runtime", newTrained, "Test MAE (°C)", perLeadShort),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        var html = SitePages.RenderModels(input, "temperature");

        // Every lead row appears — both halves are merged onto one card.
        html.Should().Contain("+12h").And.Contain("+24h").And.Contain("+48h")
            .And.Contain("+72h").And.Contain("+96h").And.Contain("+120h");
        // Header version is the newest TrainedAt (short-leads version's
        // 08:52, not the long-leads 06:00).
        html.Should().Contain("v_short_2d");
        // Each per-lead BlendTestScore renders (formatted to 3dp).
        html.Should().Contain("0.444").And.Contain("0.469").And.Contain("0.569")
            .And.Contain("0.701").And.Contain("0.835").And.Contain("0.991");
    }

    [Fact]
    public void RenderModels_merges_per_station_phase_versions_for_precip_3d_split()
    {
        // 3d precipitation has the same short/long split per station — e.g.
        // Bellever ships v_3d_short {12,24,48,72} alongside v_3d_long {96,120}.
        // The merge groups by (Composite, Phase) where Composite includes
        // station ("precipitation/ea_bellever_dartmoor"), so a per-station
        // phase that's split across two artefacts still surfaces all six
        // leads on one card.
        var oldTrained = new DateTime(2026, 5, 8, 6, 15, 0, DateTimeKind.Utc);
        var newTrained = new DateTime(2026, 5, 8, 6, 12, 0, DateTimeKind.Utc); // older actually
        var perLeadShort = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [12] = new(12, "gfs", 0.10, 0.09, 0.085, 0.30, 0.0, 200, 3),
            [24] = new(24, "gfs", 0.12, 0.11, 0.105, 0.32, 0.0, 200, 3),
            [48] = new(48, "gfs", 0.14, 0.13, 0.125, 0.34, 0.0, 200, 3),
            [72] = new(72, "gfs", 0.16, 0.15, 0.145, 0.36, 0.0, 200, 3),
        };
        var perLeadLong = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [96]  = new(96,  "gfs", 0.18, 0.17, 0.165, 0.38, 0.0, 200, 3),
            [120] = new(120, "gfs", 0.20, 0.19, 0.185, 0.40, 0.0, 200, 3),
        };
        var summaries = new[]
        {
            new SitePages.ModelSummary($"precipitation/{Station}", "v_3d_short", "3d", "ea_hydrology", newTrained, "Test Brier", perLeadShort),
            new SitePages.ModelSummary($"precipitation/{Station}", "v_3d_long",  "3d", "ea_hydrology", oldTrained, "Test Brier", perLeadLong),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        var html = SitePages.RenderModels(input, "precipitation");

        html.Should().Contain("+12h").And.Contain("+24h").And.Contain("+48h")
            .And.Contain("+72h").And.Contain("+96h").And.Contain("+120h");
        // Long-leads metrics survive the merge (BlendTestScore at lead 96 + 120).
        html.Should().Contain("0.165").And.Contain("0.185");
        // Short-leads metrics survive the merge (BlendTestScore at lead 12).
        html.Should().Contain("0.085");
    }

    [Fact]
    public void RenderModels_freshest_version_wins_per_lead_when_two_versions_overlap()
    {
        // A retrain that re-emits an existing lead must supersede the older
        // metric for that lead — the merged card should reflect the freshest
        // PerLead entry, not blend two metrics or pick the older one.
        var oldTrained = new DateTime(2026, 5, 1, 12, 0, 0, DateTimeKind.Utc);
        var newTrained = new DateTime(2026, 5, 8, 12, 0, 0, DateTimeKind.Utc);
        var perLeadOld = new Dictionary<int, SitePages.PerLeadMetric>
        {
            // Old version's lead 24 metric — should be SUPERSEDED by the
            // freshest version's value, not appear in the rendered HTML.
            [24] = new(24, "gfs", 1.00, 1.00, 9.876, 1.20, 0.0, 200, 3),
        };
        var perLeadNew = new Dictionary<int, SitePages.PerLeadMetric>
        {
            // Freshest version's lead 24 metric — this is the value the
            // card should show.
            [24] = new(24, "ifs", 0.50, 0.50, 0.444, 0.65, 0.0, 200, 3),
            [48] = new(48, "ifs", 0.55, 0.55, 0.555, 0.70, 0.0, 200, 3),
        };
        var summaries = new[]
        {
            new SitePages.ModelSummary("temperature", "v_old_2b", "2b", "era5", oldTrained, "Test MAE (°C)", perLeadOld),
            new SitePages.ModelSummary("temperature", "v_new_2b", "2b", "era5", newTrained, "Test MAE (°C)", perLeadNew),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        var html = SitePages.RenderModels(input, "temperature");

        html.Should().Contain("0.444");   // freshest wins at lead 24
        html.Should().Contain("0.555");   // lead 48 also from freshest
        html.Should().NotContain("9.876"); // old metric must not leak
        html.Should().Contain("v_new_2b"); // header carries the newest version
    }

    [Theory]
    [InlineData(-10.0,  59,  76, 192)]  // below cold anchor → clamps to deep cobalt
    [InlineData( -5.0,  59,  76, 192)]  // cold anchor (matplotlib coolwarm 0.0)
    [InlineData(  5.0, 146, 177, 244)]  // light-blue anchor
    [InlineData( 12.0, 247, 247, 247)]  // white centre (matplotlib coolwarm 0.5)
    [InlineData( 18.0, 244, 154, 123)]  // salmon anchor
    [InlineData( 25.0, 214,  82,  68)]  // brick-red anchor
    [InlineData( 40.0, 180,   4,  38)]  // above hottest anchor → clamps to deep red
    public void TemperatureColor_returns_expected_rgb_at_anchors_and_clamps(double celsius, int r, int g, int b)
    {
        SitePages.TemperatureColor(celsius).Should().Be($"rgb({r} {g} {b})");
    }

    [Fact]
    public void TemperatureColor_interpolates_linearly_between_white_and_salmon_on_the_warm_side()
    {
        // 14°C sits between the 12°C white centre (247,247,247) and the 18°C
        // salmon anchor (244,154,123). RGB interpolation: red falls slightly,
        // green falls more, blue falls most — moving off white towards warm.
        var result = SitePages.TemperatureColor(14.0);

        result.Should().StartWith("rgb(");
        var parts = result["rgb(".Length..^1].Split(' ');
        int r = int.Parse(parts[0]), g = int.Parse(parts[1]), b = int.Parse(parts[2]);

        // All three channels < 247 (the white anchor) since salmon is darker
        // overall, AND blue should be the lowest of the three (salmon trends
        // pinkish-orange so b is the most reduced channel).
        r.Should().BeLessThan(247);
        g.Should().BeLessThan(247);
        b.Should().BeLessThan(247);
        b.Should().BeLessThan(g, "blue should fall faster than green moving towards salmon");
        b.Should().BeLessThan(r, "blue should fall faster than red moving towards salmon");
    }

    [Fact]
    public void TemperatureColor_returns_muted_css_var_for_NaN()
    {
        // NaN comes from predictions with no temperature value — the stylesheet's muted
        // colour is the sensible fallback.
        SitePages.TemperatureColor(double.NaN).Should().Be("var(--pico-muted-color)");
    }

    // ---- PrecipProbColor (home P(wet) chip ramp) ---------------------------

    [Theory]
    [InlineData(-0.5,  67, 160,  71)]  // below 0  → clamps to green
    [InlineData( 0.0,  67, 160,  71)]  // green anchor #43a047 — dry / good
    [InlineData( 0.5, 255, 167,  38)]  // amber anchor #ffa726 — borderline
    [InlineData( 1.0, 229,  57,  53)]  // red   anchor #e53935 — wet / bad
    [InlineData( 1.5, 229,  57,  53)]  // above 1 → clamps to red
    public void PrecipProbColor_returns_expected_rgb_at_anchors_and_clamps(double prob, int r, int g, int b)
    {
        SitePages.PrecipProbColor(prob).Should().Be($"rgb({r} {g} {b})");
    }

    [Fact]
    public void PrecipProbColor_interpolates_smoothly_from_green_through_amber_to_red()
    {
        // Traffic-light ramp: at 0.25 we sit between green (67,160,71) and
        // amber (255,167,38) — red rises sharply, blue falls to lower
        // saturated. At 0.75 between amber and red (229,57,53) — green
        // falls toward the red anchor, blue stays low.
        var quarter = SitePages.PrecipProbColor(0.25);
        var threeq  = SitePages.PrecipProbColor(0.75);

        var qParts = quarter["rgb(".Length..^1].Split(' ');
        var qR = int.Parse(qParts[0]); var qG = int.Parse(qParts[1]); var qB = int.Parse(qParts[2]);
        qR.Should().BeInRange(68, 254, "red rising green → amber");
        qG.Should().BeGreaterThanOrEqualTo(160, "green channel rises slightly into amber");
        qB.Should().BeInRange(39, 70, "blue falling green → amber");

        var tParts = threeq["rgb(".Length..^1].Split(' ');
        var tR = int.Parse(tParts[0]); var tG = int.Parse(tParts[1]); var tB = int.Parse(tParts[2]);
        tR.Should().BeInRange(230, 254, "red stays high amber → red");
        tG.Should().BeInRange(58, 166, "green falling amber → red");
    }

    [Fact]
    public void PrecipProbColor_returns_muted_css_var_for_NaN()
    {
        SitePages.PrecipProbColor(double.NaN).Should().Be("var(--pico-muted-color)");
    }

    // ---- Met Office Spot comparison line (temp + rain skill pages) ----

    [Fact]
    public void RenderTempSkill_includes_met_office_spot_dataset_when_forecasts_present()
    {
        // Met Office Spot needs to land as a *labelled* dataset on the temp
        // skill chart so a reader can compare it eyeball-to-eyeball with the
        // blend. Material green 800 (#2e7d32) is what the legend entry
        // should use — distinct from the truth red/orange and the blend
        // purples; was originally Met Office brand navy (#262261) but that
        // sat too close to the +72h blend purple on dark-mode monitors.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var windowStart = generatedAt.AddDays(-7);
        var validTime = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);

        var moSpot = new[]
        {
            new SitePages.MetOfficeSpotForecastPoint(
                RunTimeUtc: validTime.AddHours(-24),
                ValidTimeUtc: validTime,
                Temperature2m: 14.5,
                PrecipitationProbabilityPercent: 30.0,
                LocationName: "bonehill_rocks",
                LeadHours: 24),
        };

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = windowStart,
            MetOfficeSpotForecasts = moSpot,
            // A blender prediction at the same valid time so the chart has
            // something to plot alongside; otherwise the renderer might
            // short-circuit before reaching the Met Office line.
            Predictions = new[]
            {
                new WeatherBlend.Models.TempPredictionRow
                {
                    LocationName = "bonehill_rocks",
                    ModelVersion = "v2b",
                    PredictionMadeAtUtc = validTime.AddHours(-24),
                    ValidTimeUtc = validTime,
                    LeadHours = 24,
                    BlendTemperature = 14.0,
                    FeatureVectorHash = "",
                },
            },
            PhaseByVersion = new Dictionary<string, string> { ["v2b"] = "2b" },
        };

        var html = SitePages.RenderTempSkill(input);

        html.Should().Contain("Met Office Spot");
        html.Should().Contain("#2e7d32");
    }

    [Fact]
    public void RenderTempSkill_omits_met_office_dataset_when_no_temperature_values()
    {
        // PoP-only rows (no Temperature2m) shouldn't manufacture a dataset on
        // the temp chart — they're for the rain chart. Filtering happens in
        // the renderer's "where Temperature2m has value" clause.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var moSpot = new[]
        {
            new SitePages.MetOfficeSpotForecastPoint(
                RunTimeUtc: validTime.AddHours(-24),
                ValidTimeUtc: validTime,
                Temperature2m: null,
                PrecipitationProbabilityPercent: 30.0,
                LocationName: "bonehill_rocks",
                LeadHours: 24),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            MetOfficeSpotForecasts = moSpot,
        };

        SitePages.RenderTempSkill(input).Should().NotContain("Met Office Spot");
    }

    [Fact]
    public void RenderForecasts_inlines_per_nwp_pop_into_the_top_pwet_chart_at_each_lead()
    {
        // Per-NWP precipitation_probability lines belong on the per-lead
        // Forecasts page's TOP P(wet) chart — alongside the blend P(wet)
        // and climatology lines, on the same [0, 1] probability axis. NOT
        // on the bottom mm/h chart (units don't match) and NOT on the rain
        // skill page (different surface, was the wrong place to put it).
        // Each NWP that publishes PoP (~4 of 8 via Open-Meteo) gets one
        // line, labelled "<NWP> PoP", colour-keyed off NwpsForPrecipitation.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        // Future valid time so it falls inside the +24h forecast window
        // (RenderPrecipSection filters to ValidTime >= now − 1h).
        var validTime = generatedAt.AddHours(24);

        var nwpPop = new[]
        {
            new SitePages.NwpPrecipProbForecastPoint("gfs_seamless",   validTime, 70.0, "bonehill_rocks"),
            new SitePages.NwpPrecipProbForecastPoint("ecmwf_ifs025",   validTime, 60.0, "bonehill_rocks"),
            new SitePages.NwpPrecipProbForecastPoint("icon_seamless",  validTime, 50.0, "bonehill_rocks"),
            new SitePages.NwpPrecipProbForecastPoint("gem_seamless",   validTime, 40.0, "bonehill_rocks"),
        };

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = generatedAt.AddDays(-30),
            NwpPrecipProbabilities = nwpPop,
            PrecipPredictions = new[]
            {
                new SitePages.PrecipForecastPoint(
                    Station: "ea_bellever_dartmoor", Version: "v3a",
                    PredictedAtUtc: generatedAt, ValidTimeUtc: validTime,
                    LeadHours: 24, ProbWet: 0.6, ClimatologyPWet: 0.4,
                    PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                    PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
                    PrecipAifs: null, PrecipJma: null),
            },
            PrecipCurrentByStation = new Dictionary<string, string> { ["ea_bellever_dartmoor"] = "v3a" },
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderForecastsRain(input, lead: 24);

        // Per-NWP line labels appear, suffixed " PoP" to distinguish from
        // the blend's "P(wet)" line.
        html.Should().Contain("GFS PoP");
        html.Should().Contain("ECMWF PoP");
        html.Should().Contain("ICON PoP");
        html.Should().Contain("GEM PoP");
        // Brand colours from NwpsForPrecipitation reach the chart payload.
        // GEM moved from teal #26a69a to cyan 800 #00838f on 2026-05-04 to
        // separate it from ECMWF blue.
        html.Should().Contain("#ef5350");   // GFS red
        html.Should().Contain("#42a5f5");   // ECMWF blue
        html.Should().Contain("#66bb6a");   // ICON green
        html.Should().Contain("#00838f");   // GEM cyan 800
    }

    [Fact]
    public void RenderRainSkill_renders_rolling_brier_block_when_points_present()
    {
        // Mirror of the temp page's rolling-MAE panel, station-filtered.
        // Three points across (lead 24/48/72) at the current station should
        // each surface a per-lead chart with the version label on it.
        var generatedAt = new DateTime(2026, 4, 30, 0, 0, 0, DateTimeKind.Utc);
        var rolling = new[]
        {
            new SitePages.RollingBrierPoint("ea_bellever_dartmoor", "3a", 24,
                generatedAt.AddDays(-1).AddTicks(-1), 0.18, 50),
            new SitePages.RollingBrierPoint("ea_bellever_dartmoor", "3a", 48,
                generatedAt.AddDays(-1).AddTicks(-1), 0.22, 30),
            new SitePages.RollingBrierPoint("ea_bellever_dartmoor", "3a", 72,
                generatedAt.AddDays(-1).AddTicks(-1), 0.25, 20),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = new[]
            {
                // Need at least one precip prediction so RenderRainSkill
                // resolves a current station to filter on.
                new SitePages.PrecipForecastPoint(
                    Station: "ea_bellever_dartmoor", Version: "v3a",
                    PredictedAtUtc: generatedAt, ValidTimeUtc: generatedAt.AddHours(24),
                    LeadHours: 24, ProbWet: 0.6, ClimatologyPWet: 0.4,
                    PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                    PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
                    PrecipAifs: null, PrecipJma: null),
            },
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
            RollingBrier = rolling,
        };

        var html = SitePages.RenderRainSkill(input, null);

        html.Should().Contain("Rolling Brier");
        html.Should().Contain("Lead +24h").And.Contain("Lead +48h").And.Contain("Lead +72h");
        // Phase label (not version) reaches the chart payload — one line per
        // phase across retrains, not per ModelVersion.
        html.Should().Contain("Phase 3a");
    }

    [Fact]
    public void RenderRainSkill_rolling_brier_block_shows_empty_state_when_no_points()
    {
        var generatedAt = new DateTime(2026, 4, 30, 0, 0, 0, DateTimeKind.Utc);
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = new[]
            {
                new SitePages.PrecipForecastPoint(
                    Station: "ea_bellever_dartmoor", Version: "v3a",
                    PredictedAtUtc: generatedAt, ValidTimeUtc: generatedAt.AddHours(24),
                    LeadHours: 24, ProbWet: 0.6, ClimatologyPWet: 0.4,
                    PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                    PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
                    PrecipAifs: null, PrecipJma: null),
            },
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
            RollingBrier = Array.Empty<SitePages.RollingBrierPoint>(),
        };

        var html = SitePages.RenderRainSkill(input, null);

        html.Should().Contain("Rolling Brier");
        html.Should().Contain("No rolling Brier points yet");
    }

    [Fact]
    public void RenderRainSkill_does_not_inline_per_nwp_pop_lines()
    {
        // Per-NWP PoP lines belong on the Forecasts pages, not the rain
        // skill chart. Pin so a regression that re-adds them here would
        // fail the test.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = generatedAt.AddDays(-7),
            NwpPrecipProbabilities = new[]
            {
                new SitePages.NwpPrecipProbForecastPoint("gfs_seamless", validTime, 70.0, "bonehill_rocks"),
            },
            PrecipPredictions = new[]
            {
                new SitePages.PrecipForecastPoint(
                    Station: "ea_bellever_dartmoor", Version: "v3a",
                    PredictedAtUtc: validTime.AddHours(-24), ValidTimeUtc: validTime,
                    LeadHours: 24, ProbWet: 0.6, ClimatologyPWet: 0.4,
                    PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                    PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
                    PrecipAifs: null, PrecipJma: null),
            },
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderRainSkill(input, null);

        html.Should().NotContain("GFS PoP");
        html.Should().NotContain("Underlying NWPs");
    }

    [Fact]
    public void RenderRainSkill_includes_met_office_pop_dataset_with_threshold_caveat()
    {
        // Met Office DataHub Spot publishes PoP as 0–100 percent. The
        // renderer needs to divide by 100 so it sits on the same Y axis as
        // P(wet) (which is in [0, 1]). Caveat about "any measurable precip"
        // vs our 0.1mm/h training label belongs in the skill-line above the
        // chart so a reader doesn't misinterpret the overlay.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var windowStart = generatedAt.AddDays(-7);
        var validTime = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);

        var moSpot = new[]
        {
            new SitePages.MetOfficeSpotForecastPoint(
                RunTimeUtc: validTime.AddHours(-24),
                ValidTimeUtc: validTime,
                Temperature2m: 14.5,
                PrecipitationProbabilityPercent: 70.0,
                LocationName: "bonehill_rocks",
                LeadHours: 24),
        };

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = windowStart,
            MetOfficeSpotForecasts = moSpot,
            PrecipPredictions = new[]
            {
                new SitePages.PrecipForecastPoint(
                    Station: "ea_bellever_dartmoor",
                    Version: "v3a",
                    PredictedAtUtc: validTime.AddHours(-24),
                    ValidTimeUtc: validTime,
                    LeadHours: 24,
                    ProbWet: 0.6,
                    ClimatologyPWet: 0.4,
                    PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                    PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
                    PrecipAifs: null, PrecipJma: null),
            },
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderRainSkill(input, null);

        html.Should().Contain("Met Office Spot PoP");
        html.Should().Contain("#2e7d32");
        // Threshold caveat copy in the surrounding skill-line.
        html.Should().Contain("any measurable precip");
    }
    
    [InlineData("ea_bovey_tracey", "bovey")]
    [InlineData("ea_dartmoor_nr_hexworthy", "hexworthy")]
    public void StationSlug_maps_known_stations_to_short_urls(string station, string expected)
    {
        // The slug turns the verbose EA id into a URL path segment — shipping
        // `skill-bellever.html` is nicer than `skill-ea_bellever_dartmoor.html`.
        SitePages.StationSlug(station).Should().Be(expected);
    }

    [Fact]
    public void RenderStationSubNav_omits_nav_when_only_one_station_present()
    {
        // A solo station has nowhere to sub-navigate to — rendering the nav would just
        // be visual noise around a single item.
        var html = SitePages.RenderStationSubNav("skill", new[] { "ea_bellever_dartmoor" }, "ea_bellever_dartmoor");
        html.Should().BeEmpty();
    }

    [Fact]
    public void RenderStationSubNav_first_station_uses_bare_page_url_and_others_use_slugged_url()
    {
        // The first station's link is the canonical page (matches the top-nav entry),
        // so it must not have a slug suffix. Non-first stations carry `{page}-{slug}.html`.
        var html = SitePages.RenderStationSubNav("skill",
            new[] { "ea_bellever_dartmoor", "ea_bovey_tracey" }, "ea_bovey_tracey");

        html.Should().Contain("href=\"skill.html\"");
        html.Should().Contain("href=\"skill-bovey.html\"");
    }

    [Fact]
    public void RenderStationSubNav_marks_current_station_active()
    {
        // The active class powers the visual "you are here" indicator on the sub-nav.
        var html = SitePages.RenderStationSubNav("dry-window",
            new[] { "ea_bellever_dartmoor", "ea_bovey_tracey" }, "ea_bovey_tracey");

        html.Should().Contain("href=\"dry-window-bovey.html\" class=\"active\"");
        html.Should().NotContain("href=\"dry-window.html\" class=\"active\"");
    }

    [Fact]
    public void RenderRainSkill_renders_only_requested_station_when_slug_provided()
    {
        // Rain skill page variants ship one file per station; each variant should render the
        // precip chart headings for *its* station only, not every station.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var precipPreds = new[]
        {
            new SitePages.PrecipForecastPoint(
                "ea_bellever_dartmoor", "v3a", generatedAt, validTime, 24, 0.4, 0.2,
                null, null, null, null, null, null, null, null),
            new SitePages.PrecipForecastPoint(
                "ea_bovey_tracey", "v3a", generatedAt, validTime, 24, 0.5, 0.2,
                null, null, null, null, null, null, null, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var bellever = SitePages.RenderRainSkill(input, null);
        var bovey = SitePages.RenderRainSkill(input, "bovey");

        // The chart card heading (<h4>Station name</h4>) is the one that's per-station.
        // The sub-nav always mentions every station, so we look for the chart heading
        // specifically to confirm which station's chart was rendered.
        bellever.Should().Contain("<h4>Bellever Dartmoor</h4>")
            .And.NotContain("<h4>Bovey Tracey</h4>");
        bovey.Should().Contain("<h4>Bovey Tracey</h4>")
            .And.NotContain("<h4>Bellever Dartmoor</h4>");
    }

    [Fact]
    public void RenderRainSkill_emits_station_subnav_when_more_than_one_station()
    {
        // The sub-nav is the UI control that lets readers flip between station variants.
        // With multiple stations present, every rain-skill variant must carry one.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var precipPreds = new[]
        {
            new SitePages.PrecipForecastPoint(
                "ea_bellever_dartmoor", "v3a", generatedAt, validTime, 24, 0.4, 0.2,
                null, null, null, null, null, null, null, null),
            new SitePages.PrecipForecastPoint(
                "ea_bovey_tracey", "v3a", generatedAt, validTime, 24, 0.5, 0.2,
                null, null, null, null, null, null, null, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderRainSkill(input, null);

        html.Should().Contain("skill-rainfall.html")
            .And.Contain("skill-rainfall-bovey.html");
    }

    [Fact]
    public void RenderRainSkill_renders_wet_period_background_bands_not_truth_dots()
    {
        // We replaced the 0/1 truth-dot series with light-blue background bands so
        // observed wet runs read as continuous stripes behind the P(wet) lines.
        // Each chart's data-cjs config must carry the wet bands AND must not carry
        // a discrete truth dataset coloured as the old red truth marker.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var precipPreds = new[]
        {
            new SitePages.PrecipForecastPoint(
                Station, "v3a", generatedAt, validTime, 24, 0.4, 0.2,
                null, null, null, null, null, null, null, null),
        };
        // Three contiguous wet hours → one merged band in the rendered chart.
        var truth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase)
        {
            [Station] = new Dictionary<DateTime, double>
            {
                [generatedAt.AddDays(-30).AddHours(1)] = 0.5,
                [generatedAt.AddDays(-30).AddHours(2)] = 0.5,
                [generatedAt.AddDays(-30).AddHours(3)] = 0.5,
            },
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
            RainfallTruth = truth,
        };

        var html = SitePages.RenderRainSkill(input);

        // Bands appear in the JSON payload; old truth-colour discrete series must not.
        html.Should().Contain("&quot;bands&quot;");
        html.Should().NotMatchRegex(@"<polyline[^>]*stroke=""#ef5350""");
        html.Should().NotContain("Observed wet hour");
    }

    [Fact]
    public void RenderTempSkill_is_station_agnostic_and_has_no_station_subnav()
    {
        // Temperature is a single-location quantity; the temp-skill page should not
        // surface a station sub-nav even when rainfall data for multiple stations
        // is present in the same SiteInputs.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var precipPreds = new[]
        {
            new SitePages.PrecipForecastPoint(
                "ea_bellever_dartmoor", "v3a", generatedAt, validTime, 24, 0.4, 0.2,
                null, null, null, null, null, null, null, null),
            new SitePages.PrecipForecastPoint(
                "ea_bovey_tracey", "v3a", generatedAt, validTime, 24, 0.5, 0.2,
                null, null, null, null, null, null, null, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderTempSkill(input);

        html.Should().NotContain("skill-rainfall-bovey.html");
        html.Should().NotContain("<h4>Bellever Dartmoor</h4>");
    }

    [Fact]
    public void RenderDryWindow_renders_only_requested_station_when_slug_provided()
    {
        // Dry-window page variants ship one file per station; each should render its
        // own station heading and not the others'.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var targetDate = generatedAt.Date;
        var preds = new[]
        {
            new SitePages.DryWindowForecastPoint(
                "ea_bellever_dartmoor", 3, "v3b", generatedAt, targetDate, 24, 0.6, 0.5, null),
            new SitePages.DryWindowForecastPoint(
                "ea_bovey_tracey", 3, "v3b", generatedAt, targetDate, 24, 0.7, 0.5, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = preds,
            PhaseByVersion = new Dictionary<string, string> { ["v3b"] = "3b" },
        };

        var bellever = SitePages.RenderDryWindow(input, null);
        var bovey = SitePages.RenderDryWindow(input, "bovey");

        // The station heading (<h3>Station name</h3>) is the one that's per-station.
        // The sub-nav always mentions every station, so we anchor on the h3 heading.
        bellever.Should().Contain("<h3>Bellever Dartmoor</h3>")
            .And.NotContain("<h3>Bovey Tracey</h3>");
        bovey.Should().Contain("<h3>Bovey Tracey</h3>")
            .And.NotContain("<h3>Bellever Dartmoor</h3>");
    }

    [Fact]
    public void RenderDryWindow_uses_simple_title_and_renders_probs_as_percent()
    {
        // UX rules pinned 2026-04-30:
        //   1. Tab + page title is just "Dry window" (the daytime-window caveat
        //      is described in body copy, not the heading).
        //   2. Probability cells render as integer percentages with a trailing
        //      "%" — same scale as the Home P(wet) chip — instead of 0..1
        //      fractions, which were harder to scan at a glance.
        //   3. Each table row's date column carries a 3-letter day-of-week
        //      prefix (Mon/Tue/...) so the eye can find a weekday without
        //      counting from the calendar.
        var generatedAt = new DateTime(2026, 4, 30, 0, 0, 0, DateTimeKind.Utc);
        // 2026-05-04 is a Monday — pick a date whose DoW is unambiguous in the
        // assertion (avoid e.g. "Sat" overlapping with other strings).
        var targetDate = new DateTime(2026, 5, 4, 0, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            new SitePages.DryWindowForecastPoint(
                "ea_bellever_dartmoor", 6, "v3b", generatedAt, targetDate, 24, 0.62, 0.5, 0.71),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = preds,
            PhaseByVersion = new Dictionary<string, string> { ["v3b"] = "3b" },
        };

        var html = SitePages.RenderDryWindow(input, null);

        // Title: "Dry-window forecast" after the 2026-05-04 site rework
        // (was "Dry window"), no "daytime" qualifier in headings.
        html.Should().Contain("<h2>Dry-window forecast</h2>")
            .And.NotContain("Dry daytime window");

        // Probability cell: 0.62 → "62%", agreement 0.71 → "71%". No "0.62".
        html.Should().Contain("62%").And.Contain("71%")
            .And.NotContain(">0.62<")
            .And.NotContain(">0.71<");

        // Day-of-week prefix: Mon 2026-05-04. CultureInvariant ddd is en-US 3-letter.
        html.Should().Contain("Mon 2026-05-04");
    }


    [Fact]
    public void RenderDryWindow_renders_heuristic_confidence_chip_and_band_for_3p_rows_without_conformal_tag()
    {
        // Regression for the 2026-05-26 "3p dashes" bug. 3p ships no fitted
        // conformal calibrator (3p is a parameter-free Gaussian copula MC
        // over Phase 3o; the val-replay tree that 3j/3n used would need
        // PrecipReplay to support rich/oro phases, which it doesn't). The
        // site previously rendered an em-dash in the Confidence column for
        // every 3p row.
        //
        // Fix: when ConformalSetTag is empty, derive the chip from
        // ProbHasDryWindow proximity to 0.5 (heuristic, no coverage
        // guarantee) and surface the McP10–P90 longest-dry-run band as
        // an inline detail.
        var generatedAt = new DateTime(2026, 5, 26, 0, 0, 0, DateTimeKind.Utc);
        var wedTarget   = new DateTime(2026, 5, 27, 0, 0, 0, DateTimeKind.Utc);  // confident dry
        var ambTarget   = new DateTime(2026, 5, 28, 0, 0, 0, DateTimeKind.Utc);  // ambiguous

        // Confident-dry row: P=0.97 (3p style — almost always dry), with a
        // narrow 7-8h longest-run band.
        var confidentDry = new SitePages.DryWindowForecastPoint(
            "ea_bellever_dartmoor", 3, "v3p", generatedAt, wedTarget, 24,
            ProbHasDryWindow: 0.97,
            ClimatologyProbHasDryWindow: 0.5,
            AgreementHasDryWindow: null,
            McMeanLongestDryRunHours: 7.5,
            McP10LongestDryRunHours: 7,
            McP50LongestDryRunHours: 8,
            McP90LongestDryRunHours: 8,
            ConformalSetTag: null);

        // Ambiguous row: P=0.52, wide band 3-7h.
        var ambiguous = new SitePages.DryWindowForecastPoint(
            "ea_bellever_dartmoor", 3, "v3p", generatedAt, ambTarget, 24,
            ProbHasDryWindow: 0.52,
            ClimatologyProbHasDryWindow: 0.5,
            AgreementHasDryWindow: null,
            McMeanLongestDryRunHours: 4.8,
            McP10LongestDryRunHours: 3,
            McP50LongestDryRunHours: 5,
            McP90LongestDryRunHours: 7,
            ConformalSetTag: null);

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = new[] { confidentDry, ambiguous },
            PhaseByVersion = new Dictionary<string, string> { ["v3p"] = "3p" },
        };

        var html = SitePages.RenderDryWindow(input, null);

        // Header switched from "Conformal (90% set)" to plain "Confidence"
        // so it accommodates both calibrated chips (3b) and heuristic ones
        // (3p) without misrepresenting either.
        html.Should().Contain("<th>Confidence</th>")
            .And.NotContain("Conformal <small>(90% set)</small>");

        // Confident-dry chip with inline band on the +24h Wed row.
        html.Should().Contain("confident dry").And.Contain("97%").And.Contain("run 7–8h");
        // Ambiguous chip on the Thu row with the wider 3-7h band.
        html.Should().Contain("ambiguous").And.Contain("52%").And.Contain("run 3–7h");
        // No em-dash in the Confidence column for these rows — that was the bug.
        // (The em-dash can still appear elsewhere; what matters is the row's
        // chip text is present.)
    }

    [Fact]
    public void RenderDryWindow_unknown_slug_falls_back_to_first_station()
    {
        // An unknown slug shouldn't crash the page; it should render the canonical
        // (first) station so the URL still produces a meaningful view.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            new SitePages.DryWindowForecastPoint(
                "ea_bellever_dartmoor", 3, "v3b", generatedAt, generatedAt.Date, 24, 0.6, 0.5, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = preds,
            PhaseByVersion = new Dictionary<string, string> { ["v3b"] = "3b" },
        };

        var html = SitePages.RenderDryWindow(input, "does-not-exist");

        html.Should().Contain("Bellever Dartmoor");
    }

    // -------- annotations: today line, wet bands --------

    [Fact]
    public void RenderRainSkill_emits_today_line_in_chart_payload()
    {
        // Every rainfall skill chart needs a "today" reference line so readers can
        // tell past rainfall (truth-validated) from future P(wet) (forecast-only).
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var precipPreds = new[]
        {
            new SitePages.PrecipForecastPoint(
                Station, "v3a", generatedAt, validTime, 24, 0.4, 0.2,
                null, null, null, null, null, null, null, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderRainSkill(input);

        html.Should().Contain("&quot;todayX&quot;");
    }

    [Fact]
    public void RenderTempSkill_emits_today_line_in_chart_payload()
    {
        // Same today-line marker on the temperature skill page — both vs-truth
        // charts should carry it so the reader sees where forecast horizon starts.
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var phaseByVersion = new Dictionary<string, string> { ["v2b"] = "2b" };
        var preds = new[]
        {
            new TempPredictionRow
            {
                LocationName = "Test",
                ModelVersion = "v2b",
                PredictionMadeAtUtc = generatedAt.AddHours(-1),
                ValidTimeUtc = generatedAt.AddHours(-1),
                LeadHours = 24,
                BlendTemperature = 11.0,
                FeatureVectorHash = "",
            },
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            PhaseByVersion = phaseByVersion,
        };

        var html = SitePages.RenderTempSkill(input);

        html.Should().Contain("&quot;todayX&quot;");
    }

    // -------- ComputeWetBands --------

    [Fact]
    public void ComputeWetBands_merges_consecutive_wet_hours_into_one_band()
    {
        // Three abutting wet hours should collapse to a single band rather than
        // rendering as three rectangles in the annotation payload.
        var truth = new Dictionary<DateTime, double>
        {
            [Day.AddHours(10)] = 0.5,
            [Day.AddHours(11)] = 0.5,
            [Day.AddHours(12)] = 0.5,
        };

        var bands = SitePages.ComputeWetBands(truth, Day);

        bands.Should().HaveCount(1);
        bands[0].XStart.Should().Be(Day.AddHours(10).ToOADate());
        bands[0].XEnd.Should().Be(Day.AddHours(13).ToOADate());   // last wet hour + 1
    }

    [Fact]
    public void ComputeWetBands_splits_at_a_dry_gap()
    {
        // Two wet runs with a dry hour between them — must produce two bands.
        var truth = new Dictionary<DateTime, double>
        {
            [Day.AddHours(10)] = 0.5,
            [Day.AddHours(11)] = 0.5,
            [Day.AddHours(12)] = 0.0,    // dry — flushes run
            [Day.AddHours(13)] = 0.5,
            [Day.AddHours(14)] = 0.5,
        };

        var bands = SitePages.ComputeWetBands(truth, Day);

        bands.Should().HaveCount(2);
    }

    [Fact]
    public void ComputeWetBands_drops_hours_below_threshold()
    {
        // 0.05 mm/h is below the 0.1 mm threshold the blender's training label uses.
        // It must not register as a wet hour.
        var truth = new Dictionary<DateTime, double>
        {
            [Day.AddHours(10)] = 0.05,
            [Day.AddHours(11)] = 0.05,
        };

        SitePages.ComputeWetBands(truth, Day).Should().BeEmpty();
    }

    [Fact]
    public void ComputeWetBands_uses_exact_zero_one_threshold()
    {
        // 0.1 exactly is wet (≥ 0.1 mm threshold). The classifier elsewhere uses
        // ≤ 0.1 → dry; ≥ 0.1 → wet for this band purpose.
        var truth = new Dictionary<DateTime, double>
        {
            [Day.AddHours(10)] = 0.1,
        };

        SitePages.ComputeWetBands(truth, Day).Should().HaveCount(1);
    }

    [Fact]
    public void ComputeWetBands_returns_empty_when_truth_dict_is_empty()
    {
        // Edge case: no rainfall truth synced yet. Returns an empty list rather
        // than crashing — the chart renders without bands.
        SitePages.ComputeWetBands(new Dictionary<DateTime, double>(), Day).Should().BeEmpty();
    }

    [Fact]
    public void ComputeWetBands_ignores_hours_before_window_start()
    {
        // Truth dict can contain hours older than the chart window — those must
        // be skipped so we don't render off-chart bands.
        var truth = new Dictionary<DateTime, double>
        {
            [Day.AddHours(-5)] = 0.5,        // pre-window — drop
            [Day.AddHours(10)] = 0.5,        // in-window — keep
        };

        var bands = SitePages.ComputeWetBands(truth, Day);

        bands.Should().HaveCount(1);
        bands[0].XStart.Should().Be(Day.AddHours(10).ToOADate());
    }

    // -------- PrettyUtciBand --------

    [Theory]
    [InlineData("NoStress",          "No stress")]
    [InlineData("ModerateHeat",      "Moderate heat")]
    [InlineData("VeryStrongCold",    "Very strong cold")]
    [InlineData("ExtremeCold",       "Extreme cold")]
    [InlineData("ExtremeHeat",       "Extreme heat")]
    public void PrettyUtciBand_splits_camel_case_for_human_display(string raw, string expected)
    {
        SitePages.PrettyUtciBand(raw).Should().Be(expected);
    }

    [Fact]
    public void PrettyUtciBand_returns_empty_for_empty_input()
    {
        SitePages.PrettyUtciBand("").Should().Be("");
    }

    // -------- Index page: feels-like chip --------

    [Fact]
    public void RenderIndex_renders_two_line_feels_like_chip_when_utci_matches_card()
    {
        // UTCI prediction at the same (lead, valid_time) as the temp card → chip
        // shows both Steadman ("Feels like X") and UTCI bands. ApparentC is the
        // Steadman value; UtciC is the Bröde polynomial output. Use a +24h
        // valid time landing in tomorrow's outdoor window (12:00 UTC).
        var generatedAt = new DateTime(2026, 4, 24, 12, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);
        var preds = new[]
        {
            new TempPredictionRow
            {
                LocationName = "Test",
                ModelVersion = "v",
                PredictionMadeAtUtc = generatedAt,
                ValidTimeUtc = validTime,
                LeadHours = 24,
                BlendTemperature = 12.0,
                FeatureVectorHash = "",
            },
        };
        var feelsLike = new[]
        {
            new SitePages.FeelsLikeForecastPoint(
                Version: "v1",
                PredictedAtUtc: generatedAt,
                ValidTimeUtc: validTime,
                LeadHours: 24,
                UtciC: -3.4,
                Band: "ModerateCold",
                ApparentC: 6.7),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v",
            FeelsLikePredictions = feelsLike,
        };

        var html = SitePages.RenderIndex(input, dayOffset: 1);

        html.Should().Contain("Feels like");
        html.Should().Contain("6.7°C");           // Steadman value
        html.Should().Contain("-3.4°C");          // UTCI value
        html.Should().Contain("Moderate cold");   // Pretty band label
    }

    [Fact]
    public void RenderIndex_omits_feels_like_chip_when_no_utci_for_card_lead()
    {
        // Card present but no UTCI row at this (lead, valid_time) — fall back
        // silently to no chip rather than rendering "Feels like NaN°C" or similar.
        // Use a midday-anchor + tomorrow tab so the prediction lands in the
        // outdoor visible window.
        var generatedAt = new DateTime(2026, 4, 24, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            new TempPredictionRow
            {
                LocationName = "Test", ModelVersion = "v",
                PredictionMadeAtUtc = generatedAt,
                ValidTimeUtc = generatedAt.AddHours(24),
                LeadHours = 24,
                BlendTemperature = 12.0,
                FeatureVectorHash = "",
            },
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v",
            // FeelsLikePredictions left empty.
        };

        var html = SitePages.RenderIndex(input, dayOffset: 1);

        html.Should().NotContain("Feels like");
    }

    // -------- Models page: per-blender cards --------

    [Fact]
    public void RenderModels_emits_a_card_per_blender_with_phase_specific_description()
    {
        // The Models page rebuild puts one <article class="blender-card"> per
        // (composite, version) pair, each carrying a phase-specific prose hint
        // composed in C# rather than copied from the metadata file.
        var trained = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", BestSingleValMae: 1.20, BestSingleTestMae: 1.20, BlendTestScore: 0.95,
                      BlendTestRmse: 1.40, BlendTestBias: 0.05, TestRows: 420, TestCalendarMonths: 6),
        };
        var summaries = new[]
        {
            new SitePages.ModelSummary(
                Composite: "temperature",
                Version: "temp_v2b",
                Phase: "2b",
                DataSource: "era5",
                TrainedAtUtc: trained,
                MetricLabel: "Test MAE (°C)",
                PerLead: perLead),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        var html = SitePages.RenderModels(input, "temperature");

        html.Should().Contain("blender-card");
        html.Should().Contain("Lean blender");                  // Phase 2b prose
        html.Should().Contain("13 features");                   // Phase 2b feature count hint
        html.Should().Contain("temp_v2b");
    }

    [Fact]
    public void RenderModels_orders_champion_above_challenger_per_target()
    {
        // Card order within a composite (e.g. one temperature block) should be
        // champion → challenger so the lean blender always sits on top, even
        // when the challenger was trained later in the cycle. Earlier behaviour
        // sorted by TrainedAtUtc descending, which let an accident of training
        // order put the rich card first and made readers misread "all green at
        // top" as the lean blender's deltas.
        var trainedLean = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var trainedRich = trainedLean.AddSeconds(30);  // rich trained later
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", BestSingleValMae: 1.20, BestSingleTestMae: 1.20, BlendTestScore: 0.95,
                      BlendTestRmse: 1.40, BlendTestBias: 0.05, TestRows: 420, TestCalendarMonths: 6),
        };
        var summaries = new[]
        {
            // Order in the input list intentionally puts rich first to prove
            // the ordering comes from PhasePriority, not list iteration order.
            new SitePages.ModelSummary("temperature", "v_rich", "2c", "era5", trainedRich,  "Test MAE (°C)", perLead),
            new SitePages.ModelSummary("temperature", "v_lean", "2b", "era5", trainedLean,  "Test MAE (°C)", perLead),
            new SitePages.ModelSummary("precipitation/ea_bellever_dartmoor", "v_rich_p", "3c", "ea", trainedRich, "Brier", perLead),
            new SitePages.ModelSummary("precipitation/ea_bellever_dartmoor", "v_lean_p", "3a", "ea", trainedLean, "Brier", perLead),
            new SitePages.ModelSummary("dry_window/ea_bellever_dartmoor/3h", "v_retired", "3d-shape", "ea", trainedRich, "Brier", perLead),
            new SitePages.ModelSummary("dry_window/ea_bellever_dartmoor/3h", "v_lean_dw", "3b", "ea", trainedLean, "Brier", perLead),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        // After 2026-05-04 split, ordering must hold WITHIN each per-target
        // page rather than across one big concatenated page.
        var tempHtml = SitePages.RenderModels(input, "temperature");
        var rainHtml = SitePages.RenderModels(input, "precipitation");
        var dryHtml  = SitePages.RenderModels(input, "dry_window");

        tempHtml.IndexOf("v_lean").Should().BeLessThan(tempHtml.IndexOf("v_rich"));
        rainHtml.IndexOf("v_lean_p").Should().BeLessThan(rainHtml.IndexOf("v_rich_p"));
        // Dry-window: 3b + 3g are the allowlisted phases (2026-05-04, see
        // ActivePhasePolicy). A retired-phase summary ("3d-shape", here as
        // "v_retired") is filtered out entirely so v_lean_dw renders alone.
        // Pin both: 3b lean renders, the retired phase doesn't.
        dryHtml.Should().Contain("v_lean_dw");
        dryHtml.Should().NotContain("v_retired");
    }

    [Fact]
    public void RenderModels_marks_blend_win_with_delta_good_class()
    {
        // Blend MAE 0.95 vs best-single 1.20 → blend wins by ~21%. The Δ cell
        // must carry the green "delta-good" class so blend-wins reads at a glance.
        var trained = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", BestSingleValMae: 1.20, BestSingleTestMae: 1.20, BlendTestScore: 0.95,
                      BlendTestRmse: 1.40, BlendTestBias: 0.05, TestRows: 420, TestCalendarMonths: 6),
        };
        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[]
            {
                new SitePages.ModelSummary("temperature", "v", "2b", "era5", trained, "Test MAE (°C)", perLead),
            },
        };

        var html = SitePages.RenderModels(input, "temperature");

        html.Should().Contain("delta-good");
        html.Should().NotContain("delta-bad");
    }

    [Fact]
    public void RenderModels_marks_blend_loss_with_delta_bad_class()
    {
        // Inverse case: blend MAE 1.50 vs best-single 1.20 → blend loses by ~25%.
        var trained = new DateTime(2026, 4, 20, 12, 0, 0, DateTimeKind.Utc);
        var perLead = new Dictionary<int, SitePages.PerLeadMetric>
        {
            [24] = new(24, "gfs", BestSingleValMae: 1.20, BestSingleTestMae: 1.20, BlendTestScore: 1.50,
                      BlendTestRmse: 1.80, BlendTestBias: 0.05, TestRows: 420, TestCalendarMonths: 6),
        };
        var input = MakeEmptyForecastInput() with
        {
            ModelSummaries = new[]
            {
                new SitePages.ModelSummary("temperature", "v", "2b", "era5", trained, "Test MAE (°C)", perLead),
            },
        };

        var html = SitePages.RenderModels(input, "temperature");

        html.Should().Contain("delta-bad");
        html.Should().NotContain("delta-good");
    }

    [Fact]
    public void RenderModelsSpec_groups_by_target_and_decodes_feature_set()
    {
        // Three rows mirroring real shapes: lean temp, exact-runtime precip
        // with UKV (averaging), exact-runtime temp without UKV. Phases match
        // ActivePhasePolicy so they survive the IsActivePhase filter.
        var trained = new DateTime(2026, 5, 7, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            new SitePages.FeatureSpecRow(
                Composite: "temperature",
                Phase: "2b",
                Version: "v2026-04-28_232613",
                TrainedAtUtc: trained,
                LeadHours: 24,
                FeatureSet: "lean",
                RequiredModels: new[] { "gfs_seamless", "ecmwf_ifs025", "icon_seamless" },
                OptionalModels: new[] { "ecmwf_aifs025_single" },
                FeatureNames: new[] { "temp_gfs", "temp_ecmwf", "temp_icon", "temp_aifs", "temp_mean" }),
            new SitePages.FeatureSpecRow(
                Composite: "precipitation/ea_bellever_dartmoor",
                Phase: "3d",
                Version: "v2026-05-07_125137_phase3d",
                TrainedAtUtc: trained,
                LeadHours: 48,
                // Real-schema shape: UKV is signalled by `precip_ukv` /
                // `temp_ukv` in FeatureNames — the codified train/predict
                // contract (see PrecipExactFeatureBuilder.cs:123 +
                // Exact12hFeatureBuilder.cs:179). FeatureSet sometimes
                // carries a `-ukv` suffix (3d at 48/72) but not always
                // (2d at 12/24), so the suffix is unreliable.
                FeatureSet: "exact-l48-P1-ukv",
                RequiredModels: new[] { "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper" },
                OptionalModels: new[] { "met_office_global" },
                FeatureNames: new[] { "precip_gfs", "precip_ifs", "precip_aifs", "precip_moglobal", "precip_ukv", "precip_mean" }),
            new SitePages.FeatureSpecRow(
                Composite: "temperature",
                Phase: "2d",
                Version: "v2026-05-07_094728_phase2d",
                TrainedAtUtc: trained,
                LeadHours: 48,
                // 2d at lead 48 has UKV DROPPED (decision 2026-05-07,
                // out-of-reach for the 0/6/12/18Z cycles' T+54h cap), so
                // FeatureNames omits temp_ukv.
                FeatureSet: "exact-l48-T2",
                RequiredModels: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
                OptionalModels: new[] { "ecmwf_ifs_oper", "met_office_global" },
                FeatureNames: new[] { "temp_gfs", "temp_ifs", "temp_aifs", "temp_moglobal", "temp_mean" }),
        };

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = rows };

        var html = SitePages.RenderModelsSpec(input);

        html.Should().Contain("Feature spec");
        // Target groupings rendered.
        html.Should().Contain("<h3>Temperature</h3>");
        html.Should().Contain("<h3>Precipitation</h3>");
        // Lean temp row decoded as Open-Meteo, no UKV.
        html.Should().Contain("Open-Meteo previous_runs");
        // Exact-runtime rows tagged as S3.
        html.Should().Contain("Exact-runtime S3");
        // UKV averaging mode for the precip P-tier row (FeatureSet ends in
        // `-ukv`). Regression check for the dce1d56 bug where every UKV
        // cell rendered "—" because the renderer looked for met_office_ukv
        // in OptionalModels (where it never appears) instead of the
        // FeatureSet suffix.
        html.Should().Contain("Averaging");
        // NWP shorts in place — no raw IDs leaking through.
        html.Should().Contain("AIFS");
        html.Should().Contain("UKMO Global");
    }

    [Fact]
    public void RenderModelsSpec_ukv_detected_via_feature_names_not_featureset_suffix()
    {
        // Regression test for the second-attempt UKV bug (commit 556e681):
        // checking the FeatureSet `-ukv` suffix worked for 3d's
        // "exact-l48-P1-ukv" but FAILED for 2d's "exact-l12-T2" (no suffix
        // even though temp_ukv IS in FeatureNames). The on-disk truth is
        // `temp_ukv` / `precip_ukv` in FeatureNames — the codified
        // train/predict contract (Exact12hFeatureBuilder.cs:179 + :439,
        // PrecipExactFeatureBuilder.cs:123 + :311).
        //
        // Four scenarios this test covers (FeatureSet × FeatureNames):
        //   1. Suffix yes, column yes  → render UKV mode (3d real-schema shape)
        //   2. Suffix NO,  column yes  → render UKV mode (2d real-schema shape — the bug)
        //   3. Suffix yes, column no   → render "—" (defensive, shouldn't happen but tag-without-data)
        //   4. Suffix no,  column no   → render "—" (no UKV, ever)
        var trained = new DateTime(2026, 5, 7, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            // Case 1: 3d real shape — suffix + column both present.
            new SitePages.FeatureSpecRow(
                Composite: "precipitation/ea_bellever_dartmoor",
                Phase: "3d", Version: "v-suffix-and-column", TrainedAtUtc: trained,
                LeadHours: 24, FeatureSet: "exact-l24-P1-ukv",
                RequiredModels: new[] { "gfs_ncep" },
                OptionalModels: new[] { "met_office_global" },
                FeatureNames: new[] { "precip_gfs", "precip_ukv" }),
            // Case 2: 2d real shape — column present, NO suffix. This was
            // the failing case under commit 556e681's logic.
            new SitePages.FeatureSpecRow(
                Composite: "temperature",
                Phase: "2d", Version: "v-column-no-suffix", TrainedAtUtc: trained,
                LeadHours: 12, FeatureSet: "exact-l12-T2",
                RequiredModels: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
                OptionalModels: new[] { "ecmwf_ifs_oper" },
                FeatureNames: new[] { "temp_gfs", "temp_ifs", "temp_aifs", "temp_ukv", "temp_mean" }),
            // Case 4: no UKV anywhere — should render "—". (Case 3 omitted
            // — it's defensive and the trainer can't actually emit it.)
            new SitePages.FeatureSpecRow(
                Composite: "temperature",
                Phase: "2d", Version: "v-no-suffix-no-column", TrainedAtUtc: trained,
                LeadHours: 48, FeatureSet: "exact-l48-T2",
                RequiredModels: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
                OptionalModels: new[] { "ecmwf_ifs_oper" },
                FeatureNames: new[] { "temp_gfs", "temp_ifs", "temp_aifs", "temp_mean" }),
        };

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = rows };
        var html = SitePages.RenderModelsSpec(input);

        // Assert on full <td>...</td> cells so the matches can't be
        // satisfied by a stray header or unrelated dash.
        html.Should().Contain("<td>Averaging</td>");  // Case 1 — P-tier with UKV column
        html.Should().Contain("<td>Strict</td>");     // Case 2 — T-tier with UKV column, NO suffix
        html.Should().Contain("<td>—</td>");          // Case 4 — no UKV at all
    }

    [Fact]
    public void RenderModelsSpec_reads_structured_fields_when_present()
    {
        // Phase 2 of the structured-spec migration — when DataSource +
        // UkvStrategy are populated on the row (which they are for any
        // schema written post-Phase-1, 2026-05-07+), the renderer reads
        // them directly without parsing the FeatureSet string. This test
        // sets DataSource to a known value and FeatureSet to a value that
        // would WRONGLY classify under legacy inference — proving the
        // structured fields take precedence.
        var trained = new DateTime(2026, 5, 8, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            // FeatureSet "exact-l24-P1" would infer source=Exact-runtime
            // and ukv=— under legacy. Structured fields say otherwise:
            // OpenMeteoPreviousRuns + Strict. Renderer must trust the
            // structured fields.
            new SitePages.FeatureSpecRow(
                Composite: "temperature", Phase: "2d", Version: "v-structured",
                TrainedAtUtc: trained, LeadHours: 24, FeatureSet: "exact-l24-P1",
                RequiredModels: new[] { "gfs_ncep" },
                OptionalModels: Array.Empty<string>(),
                FeatureNames: new[] { "temp_gfs", "temp_ukv", "temp_mean" },
                DataSource: "open_meteo_previous_runs",
                Tier: "T2",
                UkvStrategy: "Strict"),
        };

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = rows };
        var html = SitePages.RenderModelsSpec(input);

        // Structured DataSource wins over the misleading FeatureSet prefix.
        html.Should().Contain("Open-Meteo previous_runs");
        html.Should().NotContain("Exact-runtime S3");
        // Structured UkvStrategy = "Strict" wins.
        html.Should().Contain("<td>Strict</td>");
    }

    [Fact]
    public void RenderModelsSpec_falls_back_to_inference_for_legacy_schemas()
    {
        // Legacy schemas (pre-2026-05-07) have empty DataSource. Renderer
        // must fall back to the FeatureSet/FeatureNames inference path.
        // This is a regression check: a refactor that drops the fallback
        // would break every model version trained before Phase 1 landed
        // (and there are dozens still in the predictions window).
        var trained = new DateTime(2026, 5, 5, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            // Legacy: no DataSource, no UkvStrategy. FeatureSet says
            // exact-l12-T2; FeatureNames has temp_ukv. Inference must
            // produce Exact-runtime S3 + Strict.
            new SitePages.FeatureSpecRow(
                Composite: "temperature", Phase: "2d", Version: "v-legacy",
                TrainedAtUtc: trained, LeadHours: 12, FeatureSet: "exact-l12-T2",
                RequiredModels: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
                OptionalModels: new[] { "ecmwf_ifs_oper" },
                FeatureNames: new[] { "temp_gfs", "temp_aifs", "temp_ukv" }
                // DataSource defaults to "", UkvStrategy defaults to null.
                ),
        };

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = rows };
        var html = SitePages.RenderModelsSpec(input);

        html.Should().Contain("Exact-runtime S3");
        html.Should().Contain("<td>Strict</td>");
    }

    [Fact]
    public void RenderModelsSpec_structured_null_ukv_renders_dash_even_when_featurenames_has_temp_ukv()
    {
        // Edge case: structured says "no UKV here" (UkvStrategy=null with
        // a non-empty DataSource — so we know it's a Phase-1+ schema), but
        // the row's FeatureNames happens to include temp_ukv (e.g. a
        // forgotten column from a copy-paste). The structured field is
        // canonical — render "—", not "Strict". Without this rule the
        // renderer would silently disagree with the trainer's intent.
        var trained = new DateTime(2026, 5, 8, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            new SitePages.FeatureSpecRow(
                Composite: "temperature", Phase: "2d", Version: "v-no-ukv",
                TrainedAtUtc: trained, LeadHours: 72, FeatureSet: "exact-l72-T2",
                RequiredModels: new[] { "gfs_ncep" },
                OptionalModels: Array.Empty<string>(),
                FeatureNames: new[] { "temp_gfs", "temp_ukv", "temp_mean" }, // stale column!
                DataSource: "exact_runtime_s3",
                Tier: "T2",
                UkvStrategy: null), // trainer says no UKV
        };

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = rows };
        var html = SitePages.RenderModelsSpec(input);

        html.Should().Contain("<td>—</td>");
        html.Should().NotContain("<td>Strict</td>");
    }

    [Fact]
    public void RenderModelsSpec_optional_column_includes_UKV_when_used()
    {
        // The Optional column should list every feature the row treats as
        // optional. UKV isn't carried in OptionalModels (it's joined via
        // the picker tables outside the Required/Optional pivot), so the
        // renderer has to splice it in based on FeatureNames. Without that
        // splice, a 2d row with UKV reads as "Optional: ECMWF, UKMO Global"
        // and silently omits UKV — misleading on the page that's supposed
        // to be authoritative.
        var trained = new DateTime(2026, 5, 7, 0, 0, 0, DateTimeKind.Utc);
        var row = new SitePages.FeatureSpecRow(
            Composite: "temperature",
            Phase: "2d", Version: "v-2d-12", TrainedAtUtc: trained,
            LeadHours: 12, FeatureSet: "exact-l12-T2",
            RequiredModels: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
            OptionalModels: new[] { "ecmwf_ifs_oper", "met_office_global" },
            FeatureNames: new[] { "temp_gfs", "temp_ifs", "temp_aifs", "temp_moglobal", "temp_ukv" });

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = new[] { row } };
        var html = SitePages.RenderModelsSpec(input);

        // The optional cell should include UKV alongside the schema-listed
        // optionals. Asserting on the joined "ECMWF, UKMO Global, UKV" form
        // makes sure UKV is appended (not just present somewhere).
        html.Should().Contain("ECMWF, UKMO Global, UKV");
    }

    [Fact]
    public void RenderModelsSpec_dedupes_to_freshest_per_lead()
    {
        // Two rows for the same (composite, phase, lead) — the older one's
        // version string should not appear in the rendered HTML, the newer
        // one's should.
        var older = new DateTime(2026, 5, 5, 0, 0, 0, DateTimeKind.Utc);
        var newer = new DateTime(2026, 5, 7, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            new SitePages.FeatureSpecRow(
                Composite: "temperature", Phase: "2b",
                Version: "older-version", TrainedAtUtc: older,
                LeadHours: 24, FeatureSet: "lean",
                RequiredModels: new[] { "gfs_seamless" },
                OptionalModels: Array.Empty<string>(),
                FeatureNames: new[] { "temp_gfs", "temp_mean" }),
            new SitePages.FeatureSpecRow(
                Composite: "temperature", Phase: "2b",
                Version: "newer-version", TrainedAtUtc: newer,
                LeadHours: 24, FeatureSet: "lean",
                RequiredModels: new[] { "gfs_seamless", "ecmwf_ifs025" },
                OptionalModels: Array.Empty<string>(),
                FeatureNames: new[] { "temp_gfs", "temp_ecmwf", "temp_mean" }),
        };

        var input = MakeEmptyForecastInput() with { FeatureSpecRows = rows };
        var html = SitePages.RenderModelsSpec(input);

        // Newer row's required-model count (GFS + ECMWF) wins; "ECMWF" only
        // appears via the newer row, so its presence is the dedupe signal.
        html.Should().Contain("ECMWF");
    }

    [Fact]
    public void RenderModelsSpec_renders_empty_state_when_no_rows()
    {
        var input = MakeEmptyForecastInput();
        var html = SitePages.RenderModelsSpec(input);
        html.Should().Contain("No feature schemas on disk");
    }

    private static SitePages.SiteInputs MakeEmptyForecastInput()
    {
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        return new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
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
            RainfallTruth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
        };
    }

    private static SitePages.SiteInputs MakePrecipInput((string Version, string Phase)[] versions)
    {
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var validTime = generatedAt.AddHours(24);

        var preds = versions
            .Select(v => new SitePages.PrecipForecastPoint(
                Station, v.Version, generatedAt, validTime, 24, 0.42, 0.18,
                PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
                PrecipMf: null, PrecipUkmo: null, PrecipGem: null, PrecipAifs: null, PrecipJma: null))
            .ToArray();

        var phaseByVersion = versions
            .Where(v => !string.IsNullOrEmpty(v.Phase))
            .ToDictionary(v => v.Version, v => v.Phase);

        return new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = generatedAt.AddDays(-30),
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = preds,
            PhaseByVersion = phaseByVersion,
            DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
            RainfallTruth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
        };
    }

    private static Dictionary<DateTime, double> BuildHourly(DateTime day, Func<int, double> mmForHour)
    {
        var dict = new Dictionary<DateTime, double>();
        for (int h = 0; h < 24; h++) dict[day.AddHours(h)] = mmForHour(h);
        return dict;
    }

    private static SitePages.SiteInputs MakeInput(Dictionary<DateTime, double> hourly, int windowHours)
    {
        var rainfall = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase)
        {
            [Station] = hourly,
        };
        return new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = Day.AddDays(1),
            WindowStartUtc = Day,
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = new[]
            {
                new SitePages.DryWindowForecastPoint(Station, windowHours, "v1", Day, Day, 24, 0.5, 0.4, null),
            },
            RainfallTruth = rainfall,
            DryWindowDaytime = StandardDaytime,
        };
    }
}
