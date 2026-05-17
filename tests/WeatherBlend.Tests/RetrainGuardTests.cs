using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using WeatherBlend.Train;
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

    // ---- BuildCheckAndSave end-to-end -------------------------------------
    //
    // The trainer-facing entry point. Each of the 8 .NET trainer sites
    // calls this helper post-SaveTrainingMetadata and uses the returned
    // GuardResult.Passed to gate Promote*. Tests below stub the on-disk
    // version-dir layout (parent + previous version with summary) and
    // assert the helper does the right thing on pass, fail, and the
    // first-ever-train path.

    private static string MakeTempVersionDir(string composite, string phase, string label)
    {
        var root = Path.Combine(Path.GetTempPath(), $"wb-guard-{Guid.NewGuid():N}");
        var parent = Path.Combine(root, composite.Replace('/', Path.DirectorySeparatorChar));
        var versionDir = Path.Combine(parent, $"v-{label}");
        Directory.CreateDirectory(versionDir);
        return versionDir;
    }

    private static void WritePreviousSummary(
        string parentDir, string previousVersionName, TrainingSummary summary,
        string phase)
    {
        var prevDir = Path.Combine(parentDir, previousVersionName);
        Directory.CreateDirectory(prevDir);
        // Both training_metadata.json (so TryLoadPreviousSummary finds it
        // by phase match) AND training_summary.json must exist for the
        // helper to pick it up.
        ModelArtifact.SaveTrainingMetadata(prevDir, new ModelArtifact.TrainingMetadata
        {
            Version = previousVersionName,
            Target = summary.Composite.Split('/')[0],
            Phase = phase,
            DataSource = "test",
            TrainedAtUtc = summary.ComputedAtUtc.AddDays(-7),
            Hyperparameters = new(),
            TestMae = new(),
            PerLead = new(),
            DeviationsFromBrief = new(),
        });
        ModelArtifact.SaveTrainingSummary(prevDir, summary);
    }

    [Fact]
    public void BuildCheckAndSave_writes_summary_and_passes_when_no_previous_baseline()
    {
        // First-ever-train path: TryLoadPreviousSummary returns null,
        // Check returns Passed=true with the no-baseline note, helper
        // writes the new summary to disk as the new baseline.
        var versionDir = MakeTempVersionDir("temperature", "2b", "current");
        try
        {
            var features = Enumerable.Range(0, 100)
                .Select(_ => new[] { 1f, 2f, 3f })
                .ToList();

            var result = RetrainGuard.BuildCheckAndSave(
                NullLogger.Instance,
                versionDir,
                composite: "temperature", phase: "2b", version: "v-current",
                computedAtUtc: DateTime.UtcNow,
                rowsTrain: 100, rowsVal: 20, rowsTest: 20,
                trainFeatures: features,
                featureNames: new[] { "a", "b", "c" });

            result.Passed.Should().BeTrue();
            result.Note.Should().Contain("First-ever train");
            File.Exists(Path.Combine(versionDir, ModelArtifact.TrainingSummaryFileName))
                .Should().BeTrue("guard PASS must write the new summary to disk");
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(Path.GetDirectoryName(versionDir)!)!, recursive: true);
        }
    }

    [Fact]
    public void BuildCheckAndSave_writes_summary_when_within_tolerance()
    {
        // Healthy retrain: previous + current both exist, all bands within
        // tolerance, helper writes the new summary AND returns Passed=true.
        var versionDir = MakeTempVersionDir("temperature", "2b", "new");
        var parent = Path.GetDirectoryName(versionDir)!;
        try
        {
            var prev = MakeSummary(trainRows: 10_000, valRows: 2_000, testRows: 2_000, features: 3);
            prev.Phase = "2b";
            prev.Composite = "temperature";
            WritePreviousSummary(parent, "v-old", prev, phase: "2b");

            var features = Enumerable.Range(0, 10_500)
                .Select(_ => new[] { 1f, 2f, 3f })
                .ToList();

            var result = RetrainGuard.BuildCheckAndSave(
                NullLogger.Instance,
                versionDir,
                composite: "temperature", phase: "2b", version: "v-new",
                computedAtUtc: DateTime.UtcNow,
                rowsTrain: 10_500, rowsVal: 1_950, rowsTest: 2_050,
                trainFeatures: features,
                featureNames: new[] { "a", "b", "c" });

            result.Passed.Should().BeTrue();
            result.Breaches.Should().BeEmpty();
            File.Exists(Path.Combine(versionDir, ModelArtifact.TrainingSummaryFileName))
                .Should().BeTrue();
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(parent)!, recursive: true);
        }
    }

    [Fact]
    public void BuildCheckAndSave_does_NOT_write_summary_on_guard_fail()
    {
        // Critical contract: a failed guard must NOT overwrite the previous
        // summary on disk. Otherwise the next retrain would chase a partial
        // baseline and the alert chain would lose its reference point.
        var versionDir = MakeTempVersionDir("temperature", "2b", "new");
        var parent = Path.GetDirectoryName(versionDir)!;
        try
        {
            // Previous = 10k rows; current = 4k rows → past the 30% relative
            // tolerance, guard fires.
            var prev = MakeSummary(trainRows: 10_000, features: 3);
            prev.Phase = "2b";
            prev.Composite = "temperature";
            WritePreviousSummary(parent, "v-old", prev, phase: "2b");

            var features = Enumerable.Range(0, 4_000)
                .Select(_ => new[] { 1f, 2f, 3f })
                .ToList();

            var result = RetrainGuard.BuildCheckAndSave(
                NullLogger.Instance,
                versionDir,
                composite: "temperature", phase: "2b", version: "v-new",
                computedAtUtc: DateTime.UtcNow,
                rowsTrain: 4_000, rowsVal: 800, rowsTest: 800,
                trainFeatures: features,
                featureNames: new[] { "a", "b", "c" });

            result.Passed.Should().BeFalse();
            result.Breaches.Should().Contain(b => b.Field == "rowsTrain");
            File.Exists(Path.Combine(versionDir, ModelArtifact.TrainingSummaryFileName))
                .Should().BeFalse("guard FAIL must NOT write the new summary — preserves previous baseline for the next retrain");
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(parent)!, recursive: true);
        }
    }

    [Fact]
    public void BuildCheckAndSave_skips_guard_and_summary_when_features_empty()
    {
        // Degraded path: trainer reached the helper but had no buffered
        // features (e.g. zero leads trained). Helper short-circuits with a
        // Passed=true result and writes nothing — same as the pre-guard
        // BuildAndSave's empty-features no-op.
        var versionDir = MakeTempVersionDir("temperature", "2b", "current");
        try
        {
            var result = RetrainGuard.BuildCheckAndSave(
                NullLogger.Instance,
                versionDir,
                composite: "temperature", phase: "2b", version: "v-current",
                computedAtUtc: DateTime.UtcNow,
                rowsTrain: 0, rowsVal: 0, rowsTest: 0,
                trainFeatures: new List<float[]>(),
                featureNames: Array.Empty<string>());

            result.Passed.Should().BeTrue();
            result.Note.Should().Contain("Empty train slice");
            File.Exists(Path.Combine(versionDir, ModelArtifact.TrainingSummaryFileName))
                .Should().BeFalse("empty-feature degraded path writes nothing");
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(Path.GetDirectoryName(versionDir)!)!, recursive: true);
        }
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

    [Fact]
    public void Check_fails_when_LocationName_changes_between_runs()
    {
        // The catastrophic Phase A scenario: a Membury bundle accidentally
        // routed through Bonehill's manifest slot would silently overwrite
        // the wrong production model. Guard MUST refuse the write — same
        // composite, different LocationName is the unrecoverable case.
        var prev = MakeSummary();
        prev.LocationName = "bonehill_rocks";
        var curr = MakeSummary();
        curr.LocationName = "membury_devon";

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeFalse();
        result.Breaches.Should().ContainSingle(b => b.Field == "locationName");
    }

    [Fact]
    public void Check_passes_when_both_LocationNames_match()
    {
        var prev = MakeSummary();
        prev.LocationName = "bonehill_rocks";
        var curr = MakeSummary();
        curr.LocationName = "bonehill_rocks";

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeTrue();
        result.Breaches.Should().BeEmpty();
    }

    [Fact]
    public void Check_passes_when_both_LocationNames_are_default_empty()
    {
        // After Task #21 LocationName is [JsonRequired] (load throws on
        // missing) but the in-memory POCO still defaults to "" for tests
        // and fixtures that don't set it. The guard must treat empty ==
        // empty as "no info to compare" rather than a phantom breach,
        // otherwise unit tests (and the test harness's MakeSummary) would
        // all fail with a noisy locationName breach.
        var prev = MakeSummary();   // LocationName = ""
        var curr = MakeSummary();   // LocationName = ""

        var result = RetrainGuard.Check(curr, prev, RetrainGuard.Defaults);

        result.Passed.Should().BeTrue();
        result.Breaches.Should().NotContain(b => b.Field == "locationName");
    }
}
