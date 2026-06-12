using FluentAssertions;
using WeatherBlend.Evaluate.Waves;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pure-logic tests for <see cref="WaveVerifier"/> — no DuckDB or parquet.
/// Covers the any-lead 24h bucketing, the truth join, the band-coverage
/// computation, and the COVERAGE-ONLY drift contract (cal_mae is context,
/// never a trigger) with its MinDriftN noise gate.
/// </summary>
public class WaveVerifierTests
{
    private static readonly DateTime T0 = new(2026, 6, 1, 0, 0, 0, DateTimeKind.Utc);

    private static WavePredictionRow Pred(
        string version, int lead, DateTime valid, double blend,
        double? lo = null, double? hi = null) =>
        new()
        {
            LocationName = "sennen_cove",
            ModelVersion = version,
            PredictionMadeAtUtc = valid.AddHours(-lead),
            ValidTimeUtc = valid,
            LeadHours = lead,
            BlendValue = blend,
            BandLoM = lo,
            BandHiM = hi,
        };

    private static WaveVerifier.Inputs MakeInputs(
        IReadOnlyList<WavePredictionRow> preds,
        IReadOnlyDictionary<DateTime, double> truth,
        IReadOnlyDictionary<string, WaveVerifier.CalibrationReference>? refs = null,
        int minDriftN = 1,
        double coverageFloor = 0.75) =>
        new()
        {
            Predictions = preds,
            TruthByTime = truth,
            ReferenceByVersion = refs ?? new Dictionary<string, WaveVerifier.CalibrationReference>(),
            AsOfUtc = T0.AddDays(7),
            WindowDays = 14,
            TruthLatencyDays = 0,
            CoverageDriftFloor = coverageFloor,
            MinDriftN = minDriftN,
        };

    [Theory]
    [InlineData(0, 24)]
    [InlineData(23, 24)]
    [InlineData(24, 48)]
    [InlineData(47, 48)]
    [InlineData(48, 72)]
    [InlineData(145, 168)]
    [InlineData(-3, 24)]   // live-collector negative-lead artefacts clamp into the first bucket
    public void BucketLead_maps_hourly_leads_to_24h_upper_edge_buckets(int lead, int expected)
        => WaveVerifier.BucketLead(lead).Should().Be(expected);

    [Fact]
    public void Compute_groups_by_version_and_bucket_and_drops_rows_without_truth()
    {
        var preds = new[]
        {
            Pred("v1", 2,  T0,             blend: 1.5),
            Pred("v1", 13, T0.AddHours(11), blend: 2.0),
            Pred("v1", 30, T0.AddHours(28), blend: 2.5),  // second bucket (≤48h)
            Pred("v1", 5,  T0.AddHours(3),  blend: 9.9),  // no truth — dropped
        };
        var truth = new Dictionary<DateTime, double>
        {
            [T0]              = 1.5,
            [T0.AddHours(11)] = 2.0,
            [T0.AddHours(28)] = 2.0,
        };

        var rows = WaveVerifier.Compute(MakeInputs(preds, truth));

        rows.Should().HaveCount(2);
        rows[0].LeadBucketHours.Should().Be(24);
        rows[0].N.Should().Be(2);
        rows[0].BlendMae.Should().BeApproximately(0, 1e-9);
        rows[1].LeadBucketHours.Should().Be(48);
        rows[1].N.Should().Be(1);
        rows[1].BlendMae.Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void BandCoverage_counts_only_rows_with_both_band_bounds()
    {
        var preds = new[]
        {
            Pred("v1", 1, T0,             blend: 1.0, lo: 0.8, hi: 1.4),  // truth 1.0 → covered
            Pred("v1", 2, T0.AddHours(1), blend: 1.0, lo: 0.8, hi: 1.4),  // truth 2.0 → missed
            Pred("v1", 3, T0.AddHours(2), blend: 1.0),                    // no band — excluded from denominator
        };
        var truth = new Dictionary<DateTime, double>
        {
            [T0]              = 1.0,
            [T0.AddHours(1)]  = 2.0,
            [T0.AddHours(2)]  = 1.0,
        };

        var rows = WaveVerifier.Compute(MakeInputs(preds, truth));

        rows.Should().HaveCount(1);
        rows[0].N.Should().Be(3);
        rows[0].NBand.Should().Be(2);
        rows[0].BandCoverage.Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void BandCoverage_is_NaN_when_no_rows_carry_bands()
    {
        var preds = new[] { Pred("v1", 1, T0, blend: 1.0) };
        var truth = new Dictionary<DateTime, double> { [T0] = 1.0 };

        var rows = WaveVerifier.Compute(MakeInputs(preds, truth));

        rows[0].NBand.Should().Be(0);
        double.IsNaN(rows[0].BandCoverage).Should().BeTrue();
        rows[0].DriftFlag.Should().BeFalse("no banded pairs → no coverage signal");
    }

    [Fact]
    public void High_mae_alone_never_drifts_but_the_reference_is_carried_as_context()
    {
        // Blend off by 0.5 m everywhere — 5× the cal-slice MAE. Under the
        // coverage-only contract this must NOT flag (the cal_mae reference
        // is era5_ocean-based; the buoy's ~0.3 m location-mismatch floor
        // means a multiplier threshold would alarm permanently). The
        // reference still rides on the row for the report/sidecar.
        var preds = new[]
        {
            Pred("vX", 1, T0,             blend: 2.0),
            Pred("vX", 2, T0.AddHours(1), blend: 2.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [T0]             = 1.5,
            [T0.AddHours(1)] = 1.5,
        };
        var refs = new Dictionary<string, WaveVerifier.CalibrationReference>
        {
            ["vX"] = new(CalMae: 0.10, CalBandCoverage: 0.90, NominalCoverage: 0.90, LeadMode: "any-lead-v1", Note: null),
        };

        var rows = WaveVerifier.Compute(MakeInputs(preds, truth, refs));

        rows.Should().HaveCount(1);
        rows[0].BlendMae.Should().BeApproximately(0.5, 1e-9);
        rows[0].DriftFlag.Should().BeFalse("wave drift is coverage-only");
        rows[0].ReferenceCalMae.Should().Be(0.10);
    }

    [Fact]
    public void Coverage_drift_fires_below_floor_and_respects_MinDriftN()
    {
        // 10 banded pairs, all missed → coverage 0.0, well below the 0.75 floor.
        var preds = Enumerable.Range(0, 10)
            .Select(i => Pred("vX", 1 + i, T0.AddHours(i), blend: 1.0, lo: 0.9, hi: 1.1))
            .ToArray();
        var truth = Enumerable.Range(0, 10)
            .ToDictionary(i => T0.AddHours(i), _ => 5.0);

        var flagged = WaveVerifier.Compute(MakeInputs(preds, truth, minDriftN: 10));
        flagged.Should().HaveCount(1);
        flagged[0].DriftFlag.Should().BeTrue();

        // Same data but the noise gate set above the pair count → no flag.
        var gated = WaveVerifier.Compute(MakeInputs(preds, truth, minDriftN: 11));
        gated[0].DriftFlag.Should().BeFalse("n below MinDriftN must not alarm");
    }

    [Fact]
    public void Coverage_at_or_above_floor_does_not_flag()
    {
        // 4 banded pairs, 3 covered → 0.75, exactly at the floor (flag is strictly-below).
        var preds = new[]
        {
            Pred("vX", 1, T0,             blend: 1.0, lo: 0.5, hi: 1.5),
            Pred("vX", 2, T0.AddHours(1), blend: 1.0, lo: 0.5, hi: 1.5),
            Pred("vX", 3, T0.AddHours(2), blend: 1.0, lo: 0.5, hi: 1.5),
            Pred("vX", 4, T0.AddHours(3), blend: 1.0, lo: 0.5, hi: 1.5),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [T0]             = 1.0,
            [T0.AddHours(1)] = 1.0,
            [T0.AddHours(2)] = 1.0,
            [T0.AddHours(3)] = 9.0,  // the one miss
        };

        var rows = WaveVerifier.Compute(MakeInputs(preds, truth));

        rows[0].BandCoverage.Should().BeApproximately(0.75, 1e-9);
        rows[0].DriftFlag.Should().BeFalse();
    }

    [Fact]
    public void Window_excludes_rows_outside_the_as_of_window()
    {
        var inWindow  = T0;
        var tooOld    = T0.AddDays(-30);
        var preds = new[]
        {
            Pred("v1", 1, inWindow, blend: 1.0),
            Pred("v1", 1, tooOld,   blend: 1.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [inWindow] = 1.0,
            [tooOld]   = 1.0,
        };

        var rows = WaveVerifier.Compute(MakeInputs(preds, truth));

        rows.Should().HaveCount(1);
        rows[0].N.Should().Be(1);
    }
}
