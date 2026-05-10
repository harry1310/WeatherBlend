using FluentAssertions;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

public class RetrainGuardTests
{
    private static TrainingSummary MakeSummary(
        int trainRows = 10000, int valRows = 2000, int testRows = 2000,
        int features = 22,
        Dictionary<string, FeatureStats>? perFeature = null,
        Dictionary<string, double>? labelRates = null)
    {
        return new TrainingSummary
        {
            SchemaVersion = "1",
            Composite = "test",
            Phase = "test",
            Version = "v-test",
            ComputedAtUtc = DateTime.UtcNow,
            RowsTrain = trainRows,
            RowsVal = valRows,
            RowsTest = testRows,
            FeaturesEffective = features,
            PerFeature = perFeature ?? new Dictionary<string, FeatureStats>(),
            LabelRates = labelRates ?? new Dictionary<string, double>(),
        };
    }

    [Fact]
    public void Check_returns_passing_with_note_when_previous_is_null()
    {
        // First-ever-train path: no baseline on disk, the guard has no
        // basis for comparison and the trainer should proceed unaltered.
        var result = RetrainGuard.Check(MakeSummary(), previous: null, RetrainGuard.Defaults);

        result.Passed.Should().BeTrue();
        result.Breaches.Should().BeEmpty();
        result.Note.Should().Contain("First-ever train");
    }

    [Fact]
    public void Check_passes_when_two_summaries_are_within_all_tolerances()
    {
        // Tiny jitter on each metric — well within bands. Acceptance test
        // for the "no false-positive on a healthy retrain" case.
        var prev = MakeSummary(trainRows: 10_000, valRows: 2_000, testRows: 2_000, features: 22);
        var curr = MakeSummary(trainRows: 10_500, valRows: 1_950, testRows: 2_050, features: 22);

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeTrue();
        result.Breaches.Should().BeEmpty();
    }

    [Fact]
    public void Check_fails_when_row_count_drops_more_than_tolerance()
    {
        // Train rows halved — way past the 30% relative tolerance.
        // Common upstream signal: refresh workflow failed and the trainer
        // ran on a stale partial pull.
        var prev = MakeSummary(trainRows: 10_000);
        var curr = MakeSummary(trainRows: 4_000);

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeFalse();
        result.Breaches.Should().ContainSingle(b => b.Field == "rowsTrain");
    }

    [Fact]
    public void Check_fails_when_features_effective_changes_at_all()
    {
        // FeaturesEffectiveDelta default = 0 — any change is a breach.
        // Detects a column dying (all-NaN, dropped at fit time) or being
        // added (new spec deployed without retrain coordination).
        var prev = MakeSummary(features: 22);
        var curr = MakeSummary(features: 21);

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeFalse();
        result.Breaches.Should().ContainSingle(b => b.Field == "featuresEffective");
    }

    [Fact]
    public void Check_fails_when_per_feature_NaN_pct_jumps_more_than_tolerance()
    {
        // Single feature flipped from 0% NaN to 50% NaN — way past the
        // 0.20 absolute tolerance. Real signal: an upstream model started
        // returning NaN for half the rows.
        var prev = MakeSummary(perFeature: new Dictionary<string, FeatureStats>
        {
            ["precip_gfs"] = new() { NanPct = 0.02, Mean = 0.3, Std = 0.5 },
            ["precip_ecmwf"] = new() { NanPct = 0.01, Mean = 0.3, Std = 0.5 },
        });
        var curr = MakeSummary(perFeature: new Dictionary<string, FeatureStats>
        {
            ["precip_gfs"] = new() { NanPct = 0.02, Mean = 0.3, Std = 0.5 },
            ["precip_ecmwf"] = new() { NanPct = 0.55, Mean = 0.3, Std = 0.5 },
        });

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeFalse();
        result.Breaches.Should().ContainSingle(b => b.Field == "perFeature.precip_ecmwf.nanPct");
    }

    [Fact]
    public void Check_fails_when_label_rate_shifts_more_than_tolerance()
    {
        // Bellever wet rate dropped from 35% to 20% — past the 0.10
        // absolute tolerance. Likely truth-source issue (gauge clog or
        // EA station decommissioned) rather than a real climate shift.
        var prev = MakeSummary(labelRates: new Dictionary<string, double>
        {
            ["ea_bellever_dartmoor"] = 0.35,
        });
        var curr = MakeSummary(labelRates: new Dictionary<string, double>
        {
            ["ea_bellever_dartmoor"] = 0.20,
        });

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeFalse();
        result.Breaches.Should().ContainSingle(b => b.Field == "labelRates.ea_bellever_dartmoor");
    }

    [Fact]
    public void Check_aggregates_multiple_breaches_into_one_result()
    {
        // Catastrophic upstream issue — both rows AND features changed.
        // Caller gets the full breach list so the [ci-fail] issue body
        // can be specific about everything that's wrong.
        var prev = MakeSummary(trainRows: 10_000, features: 22);
        var curr = MakeSummary(trainRows: 4_000, features: 20);

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeFalse();
        result.Breaches.Should().HaveCount(2);
        result.Breaches.Select(b => b.Field).Should().BeEquivalentTo(
            new[] { "rowsTrain", "featuresEffective" });
    }

    [Fact]
    public void Check_skips_relative_row_check_when_previous_count_is_zero()
    {
        // Edge: legacy / partial-write previous summary with no rows.
        // Relative tolerance is undefined on a zero baseline; skip rather
        // than divide-by-zero or always-fire.
        var prev = MakeSummary(trainRows: 0);
        var curr = MakeSummary(trainRows: 10_000);

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        // Other checks still run; rowsTrain check no-ops.
        result.Breaches.Should().NotContain(b => b.Field == "rowsTrain");
    }

    [Fact]
    public void Check_only_compares_features_present_in_both_summaries()
    {
        // FeaturesEffective change is reported separately; per-feature
        // NaN-pct check should NOT double-flag features that exist only
        // in one summary (those are already caught by the count check).
        var prev = MakeSummary(features: 2, perFeature: new Dictionary<string, FeatureStats>
        {
            ["a"] = new() { NanPct = 0.1 },
            ["b"] = new() { NanPct = 0.1 },
        });
        var curr = MakeSummary(features: 2, perFeature: new Dictionary<string, FeatureStats>
        {
            ["a"] = new() { NanPct = 0.1 },
            ["c"] = new() { NanPct = 0.95 },  // new feature, would breach if compared
        });

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        // No breach despite the wild new-feature NaN% — featuresEffective
        // stayed at 2, and the per-feature check skipped 'c' (not in prev).
        result.Breaches.Should().BeEmpty();
    }
}
