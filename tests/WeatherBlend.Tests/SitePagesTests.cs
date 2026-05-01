using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

public class SitePagesTests
{
    private const string Station = "ea_bellever_dartmoor";
    private static readonly DateTime Day = new(2026, 4, 1, 0, 0, 0, DateTimeKind.Utc);

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
    public void ComputeObservedDryWindows_treats_exactly_0_1_mm_as_dry()
    {
        // Boundary: the classifier uses ≤ 0.1 mm/h as "dry".
        var hourly = BuildHourly(Day, h => h >= 10 && h <= 12 ? 0.1 : 5.0);
        var input = MakeInput(hourly, windowHours: 3);

        var result = SitePages.ComputeObservedDryWindows(input);

        result[(Station, 3, Day)].Should().BeTrue();
    }

    [Fact]
    public void ComputeObservedDryWindows_skips_day_with_missing_hour()
    {
        // Drop hour 5 entirely — need full 24-hour coverage for a verdict.
        var hourly = BuildHourly(Day, h => 0.0);
        hourly.Remove(Day.AddHours(5));
        var input = MakeInput(hourly, windowHours: 3);

        var result = SitePages.ComputeObservedDryWindows(input);

        result.Should().NotContainKey((Station, 3, Day));
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
    public void RenderForecasts_renders_only_the_requested_lead()
    {
        // Predictions at every lead (24/48/72) for one station. RenderForecasts(input, 48)
        // should surface +48h content only, not the other leads' headline.
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

        var html = SitePages.RenderForecasts(input, 48);

        html.Should().Contain("+48h forecast");
        // Sub-nav still links to other leads, but the page body doesn't render +24h's chart.
        html.Should().NotContain("+24h forecast");
        html.Should().NotContain("+72h forecast");
    }

    [Fact]
    public void RenderIndex_tags_each_card_header_with_day_of_week()
    {
        // Home cards now key off valid_time, not lead — each card's header is
        // "{day} {HH:mm}Z". Readers should see the day name for each forward
        // prediction without doing mental arithmetic on the UTC timestamp.
        var generatedAt = new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc); // Fri
        var preds = new[] { 24, 48, 72 }.Select(lead =>
            new TempPredictionRow
            {
                LocationName = "Test",
                ModelVersion = "v",
                PredictionMadeAtUtc = generatedAt,
                ValidTimeUtc = generatedAt.AddHours(lead),
                LeadHours = lead,
                BlendTemperature = 12.0,
                FeatureVectorHash = "",
            }).ToArray();
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = preds,
            CurrentVersion = "v",
        };

        var html = SitePages.RenderIndex(input);

        html.Should().Contain("Sat 00:00Z");   // Fri + 24h = Sat
        html.Should().Contain("Sun 00:00Z");   // Fri + 48h = Sun
        html.Should().Contain("Mon 00:00Z");   // Fri + 72h = Mon
    }

    [Fact]
    public void RenderIndex_falls_back_to_empty_state_when_no_future_predictions()
    {
        // No forward predictions in window → grid shows the empty-state card
        // rather than a blank page or a stale entry.
        var generatedAt = new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc);
        var input = MakeEmptyForecastInput() with { GeneratedAtUtc = generatedAt };

        var html = SitePages.RenderIndex(input);

        html.Should().Contain("No forward predictions available");
    }

    [Fact]
    public void RenderForecasts_emits_lead_subnav_linking_to_every_lead()
    {
        // Sub-nav is how readers hop between leads. Rendering any lead page should
        // produce the full three-link nav, with the current lead marked active.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderForecasts(input, 48);

        html.Should().Contain("forecasts-24h.html");
        html.Should().Contain("forecasts-48h.html");
        html.Should().Contain("forecasts-72h.html");
        html.Should().Contain("lead-nav");
    }

    [Fact]
    public void RenderForecasts_shows_fallback_text_when_lead_has_no_temperature_forecast()
    {
        // No predictions at all — page should still render, with a clear "no forecast" message
        // rather than blowing up on an empty sequence.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderForecasts(input, 72);

        html.Should().Contain("No +72h temperature forecast available");
    }

    [Fact]
    public void RenderModels_shows_empty_state_when_no_training_metadata_loaded()
    {
        // First render after a fresh checkout — training_metadata.json isn't on disk yet.
        // Page should prompt the reader to run train, not crash on an empty list.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderModels(input);

        html.Should().Contain("No training metadata on disk");
    }

    [Fact]
    public void RenderModels_groups_rows_under_pretty_composite_headings()
    {
        // Two composites, one temperature and one precip-per-station. Pretty-printed
        // headings must disambiguate them so a reader scanning the page can find the
        // table they want.
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

        var html = SitePages.RenderModels(input);

        html.Should().Contain("Temperature");
        html.Should().Contain("Precipitation — Bellever Dartmoor")
            .And.NotContain("precipitation/ea_bellever_dartmoor");
        html.Should().Contain("temp_v2b");
        html.Should().Contain("precip_v3a");
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
                RunTimeUtc: validTime.AddHours(-12),
                ValidTimeUtc: validTime,
                Temperature2m: 14.5,
                PrecipitationProbabilityPercent: 30.0),
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
                RunTimeUtc: validTime.AddHours(-12),
                ValidTimeUtc: validTime,
                Temperature2m: null,
                PrecipitationProbabilityPercent: 30.0),
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
            new SitePages.NwpPrecipProbForecastPoint("gfs_seamless",   validTime, 70.0),
            new SitePages.NwpPrecipProbForecastPoint("ecmwf_ifs025",   validTime, 60.0),
            new SitePages.NwpPrecipProbForecastPoint("icon_seamless",  validTime, 50.0),
            new SitePages.NwpPrecipProbForecastPoint("gem_seamless",   validTime, 40.0),
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

        var html = SitePages.RenderForecasts(input, lead: 24);

        // Per-NWP line labels appear, suffixed " PoP" to distinguish from
        // the blend's "P(wet)" line.
        html.Should().Contain("GFS PoP");
        html.Should().Contain("ECMWF PoP");
        html.Should().Contain("ICON PoP");
        html.Should().Contain("GEM PoP");
        // Brand colours from NwpsForPrecipitation reach the chart payload.
        html.Should().Contain("#ef5350");   // GFS red
        html.Should().Contain("#42a5f5");   // ECMWF blue
        html.Should().Contain("#66bb6a");   // ICON green
        html.Should().Contain("#26a69a");   // GEM teal
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
                new SitePages.NwpPrecipProbForecastPoint("gfs_seamless", validTime, 70.0),
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
                RunTimeUtc: validTime.AddHours(-12),
                ValidTimeUtc: validTime,
                Temperature2m: 14.5,
                PrecipitationProbabilityPercent: 70.0),
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
    
    [InlineData("ea_princetown", "princetown")]
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
            new[] { "ea_bellever_dartmoor", "ea_princetown" }, "ea_princetown");

        html.Should().Contain("href=\"skill.html\"");
        html.Should().Contain("href=\"skill-princetown.html\"");
    }

    [Fact]
    public void RenderStationSubNav_marks_current_station_active()
    {
        // The active class powers the visual "you are here" indicator on the sub-nav.
        var html = SitePages.RenderStationSubNav("dry-window",
            new[] { "ea_bellever_dartmoor", "ea_princetown" }, "ea_princetown");

        html.Should().Contain("href=\"dry-window-princetown.html\" class=\"active\"");
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
                "ea_princetown", "v3a", generatedAt, validTime, 24, 0.5, 0.2,
                null, null, null, null, null, null, null, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var bellever = SitePages.RenderRainSkill(input, null);
        var princetown = SitePages.RenderRainSkill(input, "princetown");

        // The chart card heading (<h4>Station name</h4>) is the one that's per-station.
        // The sub-nav always mentions every station, so we look for the chart heading
        // specifically to confirm which station's chart was rendered.
        bellever.Should().Contain("<h4>Bellever Dartmoor</h4>")
            .And.NotContain("<h4>Princetown</h4>");
        princetown.Should().Contain("<h4>Princetown</h4>")
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
                "ea_princetown", "v3a", generatedAt, validTime, 24, 0.5, 0.2,
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
            .And.Contain("skill-rainfall-princetown.html");
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
                "ea_princetown", "v3a", generatedAt, validTime, 24, 0.5, 0.2,
                null, null, null, null, null, null, null, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            PrecipPredictions = precipPreds,
            PhaseByVersion = new Dictionary<string, string> { ["v3a"] = "3a" },
        };

        var html = SitePages.RenderTempSkill(input);

        html.Should().NotContain("skill-rainfall-princetown.html");
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
                "ea_princetown", 3, "v3b", generatedAt, targetDate, 24, 0.7, 0.5, null),
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = preds,
            PhaseByVersion = new Dictionary<string, string> { ["v3b"] = "3b" },
        };

        var bellever = SitePages.RenderDryWindow(input, null);
        var princetown = SitePages.RenderDryWindow(input, "princetown");

        // The station heading (<h3>Station name</h3>) is the one that's per-station.
        // The sub-nav always mentions every station, so we anchor on the h3 heading.
        bellever.Should().Contain("<h3>Bellever Dartmoor</h3>")
            .And.NotContain("<h3>Princetown</h3>");
        princetown.Should().Contain("<h3>Princetown</h3>")
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

        // Title: simple "Dry window", no "daytime" qualifier in headings.
        html.Should().Contain("<h2>Dry window</h2>")
            .And.NotContain("Dry daytime window");

        // Probability cell: 0.62 → "62%", agreement 0.71 → "71%". No "0.62".
        html.Should().Contain("62%").And.Contain("71%")
            .And.NotContain(">0.62<")
            .And.NotContain(">0.71<");

        // Day-of-week prefix: Mon 2026-05-04. CultureInvariant ddd is en-US 3-letter.
        html.Should().Contain("Mon 2026-05-04");
    }

    // ---- start-hour curve: PickBestStart + dry-window column rendering ----

    private static SitePages.StartHourForecastPoint StartHour(
        string station, int window, int lead, DateTime target,
        int startHour, double pi, double dailyP, double cal)
        => new(station, window, "v1", new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc),
               target, lead, startHour, pi, cal, dailyP);

    [Fact]
    public void PickBestStart_returns_argmax_when_curve_has_meaningful_shape()
    {
        // Peak − trough = 0.45 − 0.05 = 0.40, well above the 0.10 suppression
        // threshold. Daily P = 0.7 > 0.10. The 11Z start should win on
        // ConditionalProb (0.45) and its CalibratedProb is what the renderer
        // surfaces.
        var t = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var curve = new[]
        {
            StartHour("ea_bellever_dartmoor", 6, 24, t, 8,  0.20, 0.7, 0.14),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 9,  0.30, 0.7, 0.21),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 10, 0.05, 0.7, 0.035),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 11, 0.45, 0.7, 0.315),
        };

        var best = SitePages.PickBestStart(curve);
        best.Should().NotBeNull();
        best!.StartHourUtc.Should().Be(11);
    }

    [Fact]
    public void PickBestStart_returns_null_when_daily_prob_is_too_low()
    {
        // Daily P = 0.05 < 0.10 threshold: a "best start" would mislead the
        // reader into chasing a block that's almost certainly not happening.
        // Renderer should suppress the column entry for this row.
        var t = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var curve = new[]
        {
            StartHour("ea_bellever_dartmoor", 6, 24, t, 8,  0.10, 0.05, 0.005),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 9,  0.40, 0.05, 0.020),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 10, 0.30, 0.05, 0.015),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 11, 0.20, 0.05, 0.010),
        };

        SitePages.PickBestStart(curve).Should().BeNull();
    }

    [Fact]
    public void PickBestStart_returns_argmax_even_for_near_uniform_curves()
    {
        // peak − trough = 0.27 − 0.23 = 0.04. We used to gate on a 10pp
        // peak/trough range and suppress here, but that hid lead-48 / 72h
        // rows where the model said "block almost certain (91% daily) but
        // I have no strong opinion on when". The reader was left looking
        // at "—" and couldn't tell missing-curve from uniform-shape, which
        // was the wrong trade-off. Now we surface the argmax and let the
        // calibrated (NN%) printed alongside it carry the sharpness
        // signal: a flat curve renders as "09:00Z (~26%)", a peaked one
        // as "09:00Z (45%)", and the reader judges from the number.
        var t = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var curve = new[]
        {
            StartHour("ea_bellever_dartmoor", 6, 24, t, 8,  0.25, 0.95, 0.2375),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 9,  0.27, 0.95, 0.2565),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 10, 0.25, 0.95, 0.2375),
            StartHour("ea_bellever_dartmoor", 6, 24, t, 11, 0.23, 0.95, 0.2185),
        };

        var best = SitePages.PickBestStart(curve);
        best.Should().NotBeNull();
        best!.StartHourUtc.Should().Be(9);
    }

    [Fact]
    public void PickBestStart_returns_null_for_empty_curve()
    {
        SitePages.PickBestStart(Array.Empty<SitePages.StartHourForecastPoint>())
            .Should().BeNull();
    }

    [Fact]
    public void RenderDryWindow_renders_best_start_column_when_curves_present()
    {
        // Couple a sharp curve (peak at 11Z, suppression rules pass) with a
        // dry-window prediction at the same (station, window, lead, date).
        // Renderer should add a "Best start" column with "11:00Z (32%)".
        var generatedAt = new DateTime(2026, 4, 30, 0, 0, 0, DateTimeKind.Utc);
        var target = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var dryRow = new SitePages.DryWindowForecastPoint(
            "ea_bellever_dartmoor", 6, "v3b", generatedAt, target, 24, 0.7, 0.5, null);
        var curve = new[]
        {
            StartHour("ea_bellever_dartmoor", 6, 24, target, 8,  0.20, 0.7, 0.14),
            StartHour("ea_bellever_dartmoor", 6, 24, target, 9,  0.30, 0.7, 0.21),
            StartHour("ea_bellever_dartmoor", 6, 24, target, 10, 0.05, 0.7, 0.035),
            StartHour("ea_bellever_dartmoor", 6, 24, target, 11, 0.45, 0.7, 0.315),
        };

        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = new[] { dryRow },
            StartHourPredictions = curve,
            PhaseByVersion = new Dictionary<string, string> { ["v3b"] = "3b" },
        };

        var html = SitePages.RenderDryWindow(input, null);

        html.Should().Contain("Best start")
            .And.Contain("11:00Z")
            .And.Contain("(32%)"); // 0.315 → 32% rounded
    }

    [Fact]
    public void RenderDryWindow_omits_best_start_column_entirely_when_no_curves()
    {
        // Pre-step-3 rendering path: empty StartHourPredictions → header
        // doesn't appear, table layout matches the legacy four-column shape.
        var generatedAt = new DateTime(2026, 4, 30, 0, 0, 0, DateTimeKind.Utc);
        var target = new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            DryWindowPredictions = new[]
            {
                new SitePages.DryWindowForecastPoint(
                    "ea_bellever_dartmoor", 6, "v3b", generatedAt, target, 24, 0.7, 0.5, null),
            },
            StartHourPredictions = Array.Empty<SitePages.StartHourForecastPoint>(),
            PhaseByVersion = new Dictionary<string, string> { ["v3b"] = "3b" },
        };

        var html = SitePages.RenderDryWindow(input, null);

        html.Should().NotContain("Best start");
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
        // Steadman value; UtciC is the Bröde polynomial output.
        var generatedAt = new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc);
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

        var html = SitePages.RenderIndex(input);

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
        var generatedAt = new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc);
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

        var html = SitePages.RenderIndex(input);

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

        var html = SitePages.RenderModels(input);

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
            new SitePages.ModelSummary("dry_window/ea_bellever_dartmoor/3h", "v_shape", "3d-shape", "ea", trainedRich, "Brier", perLead),
            new SitePages.ModelSummary("dry_window/ea_bellever_dartmoor/3h", "v_lean_dw", "3b", "ea", trainedLean, "Brier", perLead),
        };
        var input = MakeEmptyForecastInput() with { ModelSummaries = summaries };

        var html = SitePages.RenderModels(input);

        // Lean version code appears before rich version code for each target.
        html.IndexOf("v_lean").Should().BeLessThan(html.IndexOf("v_rich"));
        html.IndexOf("v_lean_p").Should().BeLessThan(html.IndexOf("v_rich_p"));
        // Dry-window: 3b is the only allowlisted phase (2026-04-29, see
        // ActivePhasesByTarget). 3d-shape ("v_shape") is filtered out
        // entirely, so v_lean_dw renders alone with no rich card to order
        // against. Pin both: lean renders, shape doesn't.
        html.Should().Contain("v_lean_dw");
        html.Should().NotContain("v_shape");
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

        var html = SitePages.RenderModels(input);

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

        var html = SitePages.RenderModels(input);

        html.Should().Contain("delta-bad");
        html.Should().NotContain("delta-good");
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
        };
    }
}
