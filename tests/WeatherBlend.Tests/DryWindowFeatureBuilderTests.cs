using FluentAssertions;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Fixture tests on <see cref="DryWindowFeatureBuilder.ComposeRow"/>. The SQL
/// loaders are integration-tested implicitly by the diagnostic + training
/// commands running end-to-end; these unit tests target the feature maths.
/// </summary>
public class DryWindowFeatureBuilderTests
{
    private static DryWindowFeatureBuilder.ForecastDay DayOf(Func<int, double> precipForHour,
                                                              Func<int, double>? probForHour = null)
    {
        var day = new DryWindowFeatureBuilder.ForecastDay();
        for (int h = 0; h < 24; h++)
        {
            day.SetHour(h, new DryWindowFeatureBuilder.ForecastRow(
                new DateTime(2026, 3, 1, h, 0, 0, DateTimeKind.Utc),
                "gfs_seamless",
                precipForHour(h),
                probForHour?.Invoke(h),
                Rh: 80, T: 10, Td: 8,
                CloudLow: 50, CloudMid: 30, CloudHigh: 20,
                Cape: 100, Wind: 5));
        }
        return day;
    }

    [Fact]
    public void FeatureNames_count_matches_expected_53()
    {
        DryWindowFeatureBuilder.FeatureNames.Should().HaveCount(53);
        DryWindowFeatureBuilder.FeatureNames.Should().StartWith(new[]
        {
            "precip_sum_gfs", "precip_sum_ecmwf", "precip_sum_icon",
        });
        DryWindowFeatureBuilder.FeatureNames.Should().EndWith(new[] { "doy_sin", "doy_cos" });
    }

    [Fact]
    public void ModelIds_order_matches_per_model_feature_slots()
    {
        // First 6 feature slots are precip_sum_{m}, in ModelIds order suffixes.
        var suffixes = DryWindowFeatureBuilder.ModelIds
            .Select(id => id.Replace("_seamless", "")
                            .Replace("_ifs025", "")
                            .Replace("meteofrance", "mf"))
            .ToArray();
        var firstSix = DryWindowFeatureBuilder.FeatureNames.Take(6).Select(n => n.Replace("precip_sum_", "")).ToArray();
        firstSix.Should().Equal(suffixes);
    }

    [Fact]
    public void All_dry_day_all_six_models_present_yields_all_features_dry()
    {
        // Every model forecasts 0.0 mm every hour.
        var modelDays = Enumerable.Range(0, 6)
            .Select(_ => (DryWindowFeatureBuilder.ForecastDay?)DayOf(_ => 0.0, _ => 0.0))
            .ToList();

        var row = DryWindowFeatureBuilder.ComposeRow(
            targetDate: new DateOnly(2026, 3, 1),
            windowHours: 6,
            modelDays: modelDays,
            label: true,
            truthMmDay: 0.0);

        row.PrecipSumGfs.Should().Be(0f);
        row.WetHourCountGfs.Should().Be(0f);
        row.LongestDryRunGfs.Should().Be(24f);
        row.HasDryWindowGfs.Should().Be(1f);
        row.AgreementHasDryWindow.Should().Be(1f);
        row.PrecipSumMean.Should().Be(0f);
        row.PrecipSumStd.Should().Be(0f);
        row.WindowHours.Should().Be(6);
        row.HasDryWindow.Should().BeTrue();
    }

    [Fact]
    public void All_wet_day_produces_zero_longest_dry_run_and_zero_agreement()
    {
        var modelDays = Enumerable.Range(0, 6)
            .Select(_ => (DryWindowFeatureBuilder.ForecastDay?)DayOf(_ => 0.5))
            .ToList();

        var row = DryWindowFeatureBuilder.ComposeRow(
            targetDate: new DateOnly(2026, 3, 1),
            windowHours: 3,
            modelDays: modelDays,
            label: false,
            truthMmDay: 12.0);

        row.LongestDryRunGfs.Should().Be(0f);
        row.HasDryWindowGfs.Should().Be(0f);
        row.AgreementHasDryWindow.Should().Be(0f);
        row.WetHourCountGfs.Should().Be(24f);
        row.PrecipSumMean.Should().BeApproximately(12f, 1e-4f); // 24 hours * 0.5
        row.PrecipSumStd.Should().Be(0f); // all models identical
        row.PrecipSumMax.Should().BeApproximately(12f, 1e-4f);
    }

    [Fact]
    public void Exactly_four_hour_dry_run_crosses_window_threshold_between_4h_and_6h()
    {
        // Model has 4 consecutive dry hours mid-day, rest wet.
        var one = DayOf(h => (h >= 10 && h <= 13) ? 0.0 : 0.5);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { one, null, null, null, null, null };

        var row3 = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 3, modelDays, label: true, truthMmDay: 8.0);
        var row4 = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 4, modelDays, label: true, truthMmDay: 8.0);
        var row6 = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 6, modelDays, label: false, truthMmDay: 8.0);

        row3.LongestDryRunGfs.Should().Be(4f);
        row3.HasDryWindowGfs.Should().Be(1f);
        row4.HasDryWindowGfs.Should().Be(1f);
        row6.HasDryWindowGfs.Should().Be(0f);
    }

    [Fact]
    public void Missing_model_day_gives_nan_per_model_features_but_does_not_poison_ensemble()
    {
        // Three models fully populated with dry, the other three absent.
        var dryDay = DayOf(_ => 0.0);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?>
        {
            dryDay, dryDay, dryDay, null, null, null
        };

        var row = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 6, modelDays, label: true, truthMmDay: 0.0);

        row.PrecipSumGfs.Should().Be(0f);
        row.PrecipSumEcmwf.Should().Be(0f);
        row.PrecipSumIcon.Should().Be(0f);
        float.IsNaN(row.PrecipSumMf).Should().BeTrue();
        float.IsNaN(row.PrecipSumUkmo).Should().BeTrue();
        float.IsNaN(row.PrecipSumGem).Should().BeTrue();

        // Ensemble aggregates skip NaN slots.
        row.PrecipSumMean.Should().Be(0f);
        row.AgreementHasDryWindow.Should().Be(1f); // 3-of-3 present agree
    }

    [Fact]
    public void Incomplete_day_single_hour_missing_marks_that_model_nan()
    {
        // Model supplies 23 hours but no hour-12. Should still go NaN for that model's
        // per-model features — we can't honestly aggregate an unknown hour.
        var day = new DryWindowFeatureBuilder.ForecastDay();
        for (int h = 0; h < 24; h++)
        {
            if (h == 12) continue;
            day.SetHour(h, new DryWindowFeatureBuilder.ForecastRow(
                new DateTime(2026, 3, 1, h, 0, 0, DateTimeKind.Utc),
                "gfs_seamless", Precip: 0.0, Prob: 0.0,
                Rh: 80, T: 10, Td: 8,
                CloudLow: 50, CloudMid: 30, CloudHigh: 20,
                Cape: 100, Wind: 5));
        }
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { day, null, null, null, null, null };

        var row = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 6, modelDays, label: true, truthMmDay: 0.0);

        float.IsNaN(row.PrecipSumGfs).Should().BeTrue();
        float.IsNaN(row.HasDryWindowGfs).Should().BeTrue();
        float.IsNaN(row.AgreementHasDryWindow).Should().BeTrue(); // no complete models
    }

    [Fact]
    public void Ensemble_std_across_models_with_different_day_totals()
    {
        // Six models with increasing day totals: 0, 1, 2, 3, 4, 5 mm (uniformly wet at
        // 0/24, 1/24, 2/24 ... — doesn't matter for the sum, only that each day totals differ).
        var days = new List<DryWindowFeatureBuilder.ForecastDay?>();
        for (int i = 0; i < 6; i++)
        {
            var total = (double)i;
            days.Add(DayOf(_ => total / 24.0)); // uniform so sum = total
        }

        var row = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 6, days, label: false, truthMmDay: 0.0);

        // Mean of 0..5 = 2.5, variance = 35/12 ≈ 2.9167 → std ≈ 1.7078.
        row.PrecipSumMean.Should().BeApproximately(2.5f, 1e-3f);
        row.PrecipSumStd.Should().BeApproximately(1.7078f, 1e-3f);
        row.PrecipSumMax.Should().BeApproximately(5f, 1e-3f);
    }

    [Theory]
    [InlineData(1,   0.0,   1.0)]      // Jan 1: doy=1, angle=0 → sin=0, cos=1
    [InlineData(183, 0.0,  -1.0)]      // ≈ Jul 2: doy 183 ≈ half year → angle≈π
    public void Doy_encoding_wraps_on_unit_circle(int doy, double expSin, double expCos)
    {
        var date = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddDays(doy - 1);
        var modelDays = Enumerable.Range(0, 6).Select(_ => (DryWindowFeatureBuilder.ForecastDay?)DayOf(_ => 0.0)).ToList();
        var row = DryWindowFeatureBuilder.ComposeRow(
            DateOnly.FromDateTime(date), windowHours: 3, modelDays, label: true, truthMmDay: 0.0);

        row.DoySin.Should().BeApproximately((float)expSin, 0.05f);
        row.DoyCos.Should().BeApproximately((float)expCos, 0.05f);
    }

    // ---- Phase 3d shape features --------------------------------------------

    [Fact]
    public void ShapeFeatureNames_count_matches_expected_seven_and_phase_lookup_returns_60()
    {
        DryWindowFeatureBuilder.ShapeFeatureNames.Should().HaveCount(7);
        DryWindowFeatureBuilder.FeatureNamesForPhase(DryWindowFeatureBuilder.Phase3b)
            .Should().HaveCount(53);
        DryWindowFeatureBuilder.FeatureNamesForPhase(DryWindowFeatureBuilder.Phase3dShape)
            .Should().HaveCount(60);
        // 3d-calibrated reuses the 3b feature set (only the post-hoc mapping changes).
        DryWindowFeatureBuilder.FeatureNamesForPhase(DryWindowFeatureBuilder.Phase3dCalibrated)
            .Should().HaveCount(53);
    }

    [Fact]
    public void ShapeFeatures_all_dry_returns_sentinel_first_and_last_wet()
    {
        var dry = DayOf(_ => 0.0);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { dry, dry, dry, dry, dry, dry };

        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        shape.FirstWetHour.Should().Be(24.0);   // sentinel: nothing wet
        shape.LastWetHour.Should().Be(-1.0);    // sentinel: nothing wet
        shape.LongestDryRun.Should().Be(24.0);
        shape.LongestWetRun.Should().Be(0.0);
        shape.NRainEvents.Should().Be(0.0);
        shape.MorningPrecipSum.Should().Be(0.0);
        shape.AfternoonPrecipSum.Should().Be(0.0);
    }

    [Fact]
    public void ShapeFeatures_all_wet_makes_one_event_spanning_the_day()
    {
        var wet = DayOf(_ => 0.5);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { wet, wet };

        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        shape.FirstWetHour.Should().Be(0.0);
        shape.LastWetHour.Should().Be(23.0);
        shape.LongestDryRun.Should().Be(0.0);
        shape.LongestWetRun.Should().Be(24.0);
        shape.NRainEvents.Should().Be(1.0);
        shape.MorningPrecipSum.Should().BeApproximately(3.0, 1e-9);    // 6 hours × 0.5
        shape.AfternoonPrecipSum.Should().BeApproximately(3.0, 1e-9);  // 6 hours × 0.5
    }

    [Fact]
    public void ShapeFeatures_two_distinct_events_counts_them_separately_and_keeps_longest_wet()
    {
        // Wet 03–05 (3h), dry 06–09 (4h), wet 10–14 (5h), dry rest.
        var d = DayOf(h => (h >= 3 && h <= 5) || (h >= 10 && h <= 14) ? 0.4 : 0.0);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { d };

        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        shape.NRainEvents.Should().Be(2.0);
        shape.LongestWetRun.Should().Be(5.0);  // the second event is longer
        shape.LongestDryRun.Should().BeGreaterThanOrEqualTo(9.0); // 9-hour tail run after hour 14
        shape.FirstWetHour.Should().Be(3.0);
        shape.LastWetHour.Should().Be(14.0);
    }

    [Fact]
    public void ShapeFeatures_morning_and_afternoon_buckets_only_count_their_hours()
    {
        // Wet only in hours 6, 12 (one in morning bucket, one in afternoon bucket).
        var d = DayOf(h => (h == 6 || h == 12) ? 1.5 : 0.0);
        var shape = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { d });

        shape.MorningPrecipSum.Should().BeApproximately(1.5, 1e-9);
        shape.AfternoonPrecipSum.Should().BeApproximately(1.5, 1e-9);
        // Hour-2 wet would not contribute to either bucket — verify exclusivity.
        var d2 = DayOf(h => (h == 2) ? 1.0 : 0.0);
        var shape2 = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { d2 });
        shape2.MorningPrecipSum.Should().Be(0.0);
        shape2.AfternoonPrecipSum.Should().Be(0.0);
    }

    [Fact]
    public void ShapeFeatures_threshold_uses_inclusive_zero_point_one_mm()
    {
        // Hour 10 sits exactly on threshold (0.1 mm) — should count as wet.
        var atThreshold = DayOf(h => h == 10 ? 0.1 : 0.0);
        var shapeAt = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { atThreshold });
        shapeAt.NRainEvents.Should().Be(1.0);
        shapeAt.FirstWetHour.Should().Be(10.0);

        // Hour 10 below threshold (0.099) — no wet hour at all.
        var below = DayOf(h => h == 10 ? 0.099 : 0.0);
        var shapeBelow = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { below });
        shapeBelow.NRainEvents.Should().Be(0.0);
        shapeBelow.FirstWetHour.Should().Be(24.0);
    }

    [Fact]
    public void ShapeFeatures_no_models_present_returns_all_nan()
    {
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?> { null, null, null, null, null, null };
        var shape = DryWindowFeatureBuilder.ShapeFeatures(modelDays);

        double.IsNaN(shape.FirstWetHour).Should().BeTrue();
        double.IsNaN(shape.LastWetHour).Should().BeTrue();
        double.IsNaN(shape.LongestDryRun).Should().BeTrue();
        double.IsNaN(shape.LongestWetRun).Should().BeTrue();
        double.IsNaN(shape.NRainEvents).Should().BeTrue();
        double.IsNaN(shape.MorningPrecipSum).Should().BeTrue();
        double.IsNaN(shape.AfternoonPrecipSum).Should().BeTrue();
    }

    [Fact]
    public void ShapeFeatures_partial_model_coverage_uses_present_models_only_per_hour()
    {
        // Model A has hours 0–11 only with 0.5 mm/h; model B has hours 12–23 only with 0.5 mm/h.
        // Ensemble mean per hour stays 0.5 (single contributor each), so the day looks fully wet.
        var dayA = new DryWindowFeatureBuilder.ForecastDay();
        var dayB = new DryWindowFeatureBuilder.ForecastDay();
        for (int h = 0; h < 24; h++)
        {
            var which = h < 12 ? dayA : dayB;
            which.SetHour(h, new DryWindowFeatureBuilder.ForecastRow(
                new DateTime(2026, 3, 1, h, 0, 0, DateTimeKind.Utc),
                "gfs_seamless", Precip: 0.5, Prob: 0.0,
                Rh: 80, T: 10, Td: 8,
                CloudLow: 50, CloudMid: 30, CloudHigh: 20,
                Cape: 100, Wind: 5));
        }
        var shape = DryWindowFeatureBuilder.ShapeFeatures(new List<DryWindowFeatureBuilder.ForecastDay?> { dayA, dayB });
        shape.LongestWetRun.Should().Be(24.0);
        shape.NRainEvents.Should().Be(1.0);
    }

    [Fact]
    public void ComposeRow_populates_seven_shape_features_on_training_row()
    {
        // Wet 09–11 only; everything else dry. Single model populated, the
        // remaining five slots are null (per-model features go NaN, but the
        // ensemble-mean shape vector still resolves over the present model).
        var d = DayOf(h => (h >= 9 && h <= 11) ? 0.5 : 0.0);
        var modelDays = new List<DryWindowFeatureBuilder.ForecastDay?>
        {
            d, null, null, null, null, null,
        };

        var row = DryWindowFeatureBuilder.ComposeRow(
            new DateOnly(2026, 3, 1), windowHours: 3, modelDays, label: false, truthMmDay: 1.5);

        row.FirstWetHour.Should().Be(9f);
        row.LastWetHour.Should().Be(11f);
        row.LongestForecastDryRunHours.Should().BeGreaterThanOrEqualTo(12f); // tail dry run
        row.LongestForecastWetRunHours.Should().Be(3f);
        row.NRainEvents.Should().Be(1f);
        row.MorningPrecipSum.Should().BeApproximately(1.5f, 1e-4f);  // 0.5 + 0.5 + 0.5 over hours 9–11
        row.AfternoonPrecipSum.Should().Be(0f);
    }
}
