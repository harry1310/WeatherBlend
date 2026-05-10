using FluentAssertions;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

public class TrainingSummaryBuilderTests
{
    [Fact]
    public void ComputeFeatureStats_returns_per_feature_descriptive_stats_on_clean_columns()
    {
        // 100 rows × 2 features. Col 0 = 1..100 (ascending integers), col 1 = constant 5.0.
        // Mean of 1..100 = 50.5; std (sample) ≈ 29.01; p01 ≈ 1.99; p99 ≈ 99.01 with R-7 interpolation.
        var rows = new List<float[]>();
        for (int i = 1; i <= 100; i++) rows.Add(new[] { (float)i, 5.0f });

        var stats = TrainingSummaryBuilder.ComputeFeatureStats(rows, new[] { "asc", "const" });

        stats.Should().ContainKey("asc");
        stats["asc"].NanPct.Should().Be(0.0);
        stats["asc"].Mean.Should().BeApproximately(50.5, 1e-6);
        stats["asc"].Std.Should().BeApproximately(29.011, 0.01);
        // R-7 quantile: 0.01*(99) = 0.99 → between sorted[0]=1 and sorted[1]=2 at fraction 0.99 → 1.99
        stats["asc"].P01.Should().BeApproximately(1.99, 1e-6);
        stats["asc"].P99.Should().BeApproximately(99.01, 1e-6);

        // Constant column: std should be exactly 0, p01 == p99 == 5.0.
        stats["const"].Mean.Should().Be(5.0);
        stats["const"].Std.Should().Be(0.0);
        stats["const"].P01.Should().Be(5.0);
        stats["const"].P99.Should().Be(5.0);
    }

    [Fact]
    public void ComputeFeatureStats_handles_NaN_correctly_and_reports_NanPct()
    {
        // 10 rows × 1 feature, half NaN. NaN% should be 0.5 and stats should be
        // computed only over the non-NaN values {1, 2, 3, 4, 5}.
        var rows = new List<float[]>
        {
            new[] { 1.0f }, new[] { float.NaN }, new[] { 2.0f }, new[] { float.NaN }, new[] { 3.0f },
            new[] { float.NaN }, new[] { 4.0f }, new[] { float.NaN }, new[] { 5.0f }, new[] { float.NaN },
        };

        var stats = TrainingSummaryBuilder.ComputeFeatureStats(rows, new[] { "f" });

        stats["f"].NanPct.Should().BeApproximately(0.5, 1e-9);
        stats["f"].Mean.Should().BeApproximately(3.0, 1e-6);
        stats["f"].P01.Should().BeApproximately(1.04, 0.01);
        stats["f"].P99.Should().BeApproximately(4.96, 0.01);
    }

    [Fact]
    public void ComputeFeatureStats_handles_all_NaN_column_with_zeros_and_full_NaN_pct()
    {
        // Edge case: column is 100% NaN. Stats should be zero (sentinel) and
        // NaN% should be 1.0 — guard's tolerance bands will fire on such a
        // column when comparing against a previous summary that had values.
        var rows = new List<float[]>
        {
            new[] { float.NaN }, new[] { float.NaN }, new[] { float.NaN },
        };

        var stats = TrainingSummaryBuilder.ComputeFeatureStats(rows, new[] { "dead" });

        stats["dead"].NanPct.Should().Be(1.0);
        stats["dead"].Mean.Should().Be(0.0);
        stats["dead"].Std.Should().Be(0.0);
        stats["dead"].P01.Should().Be(0.0);
        stats["dead"].P99.Should().Be(0.0);
    }

    [Fact]
    public void ComputeFeatureStats_throws_on_inconsistent_row_widths()
    {
        // Defensive: trainer is supposed to feed consistent-width rows, but
        // surface a clear error if shape drifts between rows so the summary
        // doesn't silently mis-attribute values to wrong feature names.
        var rows = new List<float[]>
        {
            new[] { 1.0f, 2.0f },
            new[] { 3.0f },          // wrong width
        };

        var act = () => TrainingSummaryBuilder.ComputeFeatureStats(rows, new[] { "a", "b" });
        act.Should().Throw<InvalidOperationException>().WithMessage("*1 columns but expected 2*");
    }

    [Fact]
    public void ComputeFeatureStats_throws_on_empty_train_slice()
    {
        // Building stats on an empty slice would silently yield empty
        // PerFeature dict — which the guard would then read as "no features
        // changed" and pass the retrain. Better to fail loudly at fit time.
        var act = () => TrainingSummaryBuilder.ComputeFeatureStats(
            new List<float[]>(), new[] { "f" });
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Build_composes_full_summary_with_metadata_fields_and_label_rates()
    {
        var rows = new List<float[]>
        {
            new[] { 0.10f, 1.0f }, new[] { 0.20f, 1.0f }, new[] { 0.30f, 0.0f },
            new[] { 0.40f, 0.0f }, new[] { 0.50f, 0.0f },
        };
        var labelRates = new Dictionary<string, double>
        {
            ["ea_bellever_dartmoor"] = 0.31,
        };
        var t0 = new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc);

        var summary = TrainingSummaryBuilder.Build(
            composite: "precipitation/ea_bellever_dartmoor",
            phase: "3a",
            version: "v2026-05-10_120000",
            computedAtUtc: t0,
            rowsTrain: rows.Count, rowsVal: 100, rowsTest: 200,
            trainFeatures: rows,
            featureNames: new[] { "precip_gfs", "wet_flag" },
            labelRates: labelRates);

        summary.Composite.Should().Be("precipitation/ea_bellever_dartmoor");
        summary.Phase.Should().Be("3a");
        summary.Version.Should().Be("v2026-05-10_120000");
        summary.ComputedAtUtc.Should().Be(t0);
        summary.RowsTrain.Should().Be(5);
        summary.RowsVal.Should().Be(100);
        summary.RowsTest.Should().Be(200);
        summary.FeaturesEffective.Should().Be(2);
        summary.PerFeature.Should().HaveCount(2);
        summary.PerFeature["precip_gfs"].Mean.Should().BeApproximately(0.30, 1e-6);
        summary.LabelRates["ea_bellever_dartmoor"].Should().Be(0.31);
    }
}
