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
            Predictions = Array.Empty<PredictionRow>(),
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
    public void RenderSkill_renders_a_precipitation_chart_per_phase_present()
    {
        // Three predictions for one station, one per phase. Skill page groups per phase.
        var input = MakePrecipInput(new[]
        {
            ("v_3a",  "3a"),
            ("v_iso", "3a_isotonic"),
            ("v_3c",  "3c"),
        });

        var html = SitePages.RenderSkill(input);

        html.Should().Contain("Phase 3a (lean)");
        html.Should().Contain("Phase 3a_isotonic (lean + PAV calibration)");
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
                PrecipMf: null, PrecipUkmo: null, PrecipGem: null)).ToArray();

        var input = new SitePages.SiteInputs
        {
            LocationDisplay = "Test",
            Latitude = 0, Longitude = 0, ElevationMeters = 0,
            MetarStation = "",
            GeneratedAtUtc = generatedAt,
            WindowStartUtc = generatedAt.AddDays(-30),
            Predictions = Array.Empty<PredictionRow>(),
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
        // Home cards should label each lead with its day name, so readers can map
        // "+48h" onto "Saturday" without doing mental arithmetic on the UTC timestamp.
        var generatedAt = new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc); // Fri
        var preds = new[] { 24, 48, 72 }.Select(lead =>
            new PredictionRow
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

        html.Should().Contain("+24h · Sat");   // Fri + 24h = Sat
        html.Should().Contain("+48h · Sun");
        html.Should().Contain("+72h · Mon");
    }

    [Fact]
    public void RenderIndex_shows_day_of_week_even_when_lead_has_no_prediction()
    {
        // Empty card for a missing lead still tells the reader which day is affected,
        // derived from GeneratedAtUtc + lead hours.
        var generatedAt = new DateTime(2026, 4, 24, 0, 0, 0, DateTimeKind.Utc);
        var input = MakeEmptyForecastInput() with { GeneratedAtUtc = generatedAt };

        var html = SitePages.RenderIndex(input);

        html.Should().Contain("+24h · Sat");
        html.Should().Contain("No prediction available");
    }

    [Fact]
    public void RenderForecasts_does_not_render_blend_champion_card_anymore()
    {
        // The champion blend card duplicated the home page, so forecasts pages now
        // go straight to the per-NWP breakdown table. Guard against the duplicate
        // card creeping back in — the <h3> says "per-model inputs", not "Blend".
        var generatedAt = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc);
        var pred = new PredictionRow
        {
            LocationName = "Test",
            ModelVersion = "v",
            PredictionMadeAtUtc = generatedAt,
            ValidTimeUtc = generatedAt.AddHours(24),
            LeadHours = 24,
            BlendTemperature = 9.5,
            TempGfs = 9.7,
            TempEcmwf = 9.3,
            TempMean = 9.5,
            TempStd = 0.2,
            FeatureVectorHash = "",
        };
        var input = MakeEmptyForecastInput() with
        {
            GeneratedAtUtc = generatedAt,
            Predictions = new[] { pred },
            CurrentVersion = "v",
        };

        var html = SitePages.RenderForecasts(input, 24);

        html.Should().Contain("Temperature — per-model inputs");
        html.Should().NotContain("<h4>Blend</h4>");   // the removed champion card
        html.Should().Contain("<th class=\"num\">Blend</th>");   // table column kept
    }

    [Fact]
    public void RenderSkill_does_not_reference_dead_pre_2b_or_other_versions_section()
    {
        // The "Other versions" bucket described "pre-2b experiments left in the manifest".
        // That code and data are gone — guard against either text leaking back into any
        // page the Skill renderer produces.
        var input = MakeEmptyForecastInput();

        var html = SitePages.RenderSkill(input);

        html.Should().NotContain("Other versions")
            .And.NotContain("pre-2b")
            .And.NotContain("pre-3a")
            .And.NotContain("pre-3b");
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
            [24] = new(24, "gfs", 1.23, 0.987, 1.42, +0.05, 420, 6),
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
    [InlineData(-10.0, 57,  73, 171)]  // below the coldest anchor — clamps to indigo
    [InlineData( -5.0, 57,  73, 171)]  // cold anchor
    [InlineData( 12.0, 124, 77, 255)]  // brand purple anchor
    [InlineData( 25.0, 229, 57,  53)]  // red anchor
    [InlineData( 40.0, 183, 28,  28)]  // above hottest anchor — clamps to deep red
    public void TemperatureColor_returns_expected_rgb_at_anchors_and_clamps(double celsius, int r, int g, int b)
    {
        SitePages.TemperatureColor(celsius).Should().Be($"rgb({r} {g} {b})");
    }

    [Fact]
    public void TemperatureColor_interpolates_linearly_between_anchors_for_warming_feel()
    {
        // 14°C sits between the 12°C brand-purple anchor and the 18°C orange anchor.
        // RGB interpolation (not HSL shortest-arc) must not pass through magenta: the
        // green channel rises as we move off the purple anchor towards orange.
        var result = SitePages.TemperatureColor(14.0);

        result.Should().StartWith("rgb(");
        var parts = result["rgb(".Length..^1].Split(' ');
        int r = int.Parse(parts[0]), g = int.Parse(parts[1]), b = int.Parse(parts[2]);

        r.Should().BeGreaterThan(124, "red channel should rise towards orange");
        g.Should().BeGreaterThan(77, "green channel should rise — otherwise we'd be passing through magenta");
        b.Should().BeLessThan(255, "blue channel should fall towards orange");
    }

    [Fact]
    public void TemperatureColor_returns_muted_css_var_for_NaN()
    {
        // NaN comes from predictions with no temperature value — the stylesheet's muted
        // colour is the sensible fallback.
        SitePages.TemperatureColor(double.NaN).Should().Be("var(--pico-muted-color)");
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
            Predictions = Array.Empty<PredictionRow>(),
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
                PrecipMf: null, PrecipUkmo: null, PrecipGem: null))
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
            Predictions = Array.Empty<PredictionRow>(),
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
            Predictions = Array.Empty<PredictionRow>(),
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
