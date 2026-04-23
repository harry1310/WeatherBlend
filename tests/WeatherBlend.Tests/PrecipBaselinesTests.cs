using FluentAssertions;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipBaselinesTests
{
    private static PrecipTrainingRow Row(
        DateTime t, bool wet,
        double gfs = 0, double ecmwf = 0, double icon = 0,
        double mf = 0, double ukmo = 0, double gem = 0,
        double agreement = 0)
        => new()
        {
            ValidTimeUtc = t,
            PrecipGfs = (float)gfs, PrecipEcmwf = (float)ecmwf, PrecipIcon = (float)icon,
            PrecipMf = (float)mf, PrecipUkmo = (float)ukmo, PrecipGem = (float)gem,
            ProbGfs = 0, ProbEcmwf = 0, ProbIcon = 0, ProbMf = 0, ProbUkmo = 0, ProbGem = 0,
            PrecipMean = 0, PrecipStd = 0, PrecipMax = 0,
            PrecipAgreementWet01 = (float)agreement,
            RhMean = 0, DewDepressionMean = 0,
            CloudLowMean = 0, CloudMidMean = 0, CloudHighMean = 0,
            CapeMean = 0, WindSpeedMean = 0,
            HourSin = 0, HourCos = 1, DoySin = 0, DoyCos = 1,
            WetBinary = wet,
            PrecipMmHour = wet ? 0.5f : 0.0f,
        };

    [Fact]
    public void SingleModelWet_returns_indicator_against_0_1mm_threshold()
    {
        var t = new DateTime(2025, 6, 1, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(t,             wet: true,  gfs: 0.5),    // above threshold → 1
            Row(t.AddHours(1), wet: false, gfs: 0.099),  // just below → 0
            Row(t.AddHours(2), wet: false, gfs: 0.1),    // exactly at threshold → 1 (>=)
            Row(t.AddHours(3), wet: false, gfs: 0.0),    // dry → 0
        };

        var p = PrecipBaselines.SingleModelWet(rows, "precip_gfs");

        p.Should().Equal(1.0, 0.0, 1.0, 0.0);
    }

    [Fact]
    public void SingleModelWet_propagates_nan_for_missing_model_forecast()
    {
        var rows = new[] { Row(new DateTime(2025, 6, 1, 0, 0, 0, DateTimeKind.Utc), false, gfs: double.NaN) };

        var p = PrecipBaselines.SingleModelWet(rows, "precip_gfs");

        double.IsNaN(p[0]).Should().BeTrue();
    }

    [Fact]
    public void SingleModelWet_throws_on_unknown_column()
    {
        var rows = new[] { Row(new DateTime(2025, 6, 1, 0, 0, 0, DateTimeKind.Utc), false) };

        var act = () => PrecipBaselines.SingleModelWet(rows, "precip_does_not_exist");

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void MeanOfModels_reads_agreement_column_and_maps_nan_to_zero()
    {
        var t = new DateTime(2025, 6, 1, 0, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(t,             false, agreement: 0.5),
            Row(t.AddHours(1), false, agreement: double.NaN),
            Row(t.AddHours(2), true,  agreement: 1.0),
        };

        var p = PrecipBaselines.MeanOfModels(rows);

        p.Should().HaveCount(3);
        p[0].Should().BeApproximately(0.5, 1e-9);
        p[1].Should().Be(0.0);  // NaN → 0, not NaN
        p[2].Should().BeApproximately(1.0, 1e-9);
    }

    [Fact]
    public void Climatology_returns_per_month_per_hour_wet_fraction_from_training()
    {
        // Jan 12:00 training bucket: 2 wet + 1 dry → P(wet)=2/3.
        // Jul 06:00 training bucket: 0 wet + 2 dry → P(wet)=0.
        var trainRows = new[]
        {
            Row(new DateTime(2024, 1, 5,  12, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 1, 6,  12, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 1, 7,  12, 0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2024, 7, 5,   6, 0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2024, 7, 6,   6, 0, 0, DateTimeKind.Utc), wet: false),
        };
        var targetRows = new[]
        {
            Row(new DateTime(2025, 1, 10, 12, 0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2025, 7, 10,  6, 0, 0, DateTimeKind.Utc), wet: false),
        };

        var p = PrecipBaselines.Climatology(trainRows, targetRows);

        p[0].Should().BeApproximately(2.0 / 3.0, 1e-9);
        p[1].Should().Be(0.0);
    }

    [Fact]
    public void Climatology_falls_back_to_global_mean_for_unseen_month_hour_bucket()
    {
        // Training: 2 wet / 4 total → global mean 0.5. No (Dec, 23:00) training rows.
        var trainRows = new[]
        {
            Row(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 1, 2, 0, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 6, 1, 0, 0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2024, 6, 2, 0, 0, 0, DateTimeKind.Utc), wet: false),
        };
        var targetRows = new[]
        {
            Row(new DateTime(2025, 12, 25, 23, 0, 0, DateTimeKind.Utc), wet: false),
        };

        var p = PrecipBaselines.Climatology(trainRows, targetRows);

        p[0].Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void Persistence_returns_wet_indicator_at_t_minus_lag()
    {
        var t = new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc);
        var rows = new[] { Row(t, wet: false) };
        var truth = new Dictionary<DateTime, double>
        {
            [t.AddHours(-24)] = 0.5,   // above threshold → predict wet
        };

        var p = PrecipBaselines.Persistence(rows, truth, lagHours: 24);

        p[0].Should().Be(1.0);
    }

    [Fact]
    public void Persistence_returns_nan_when_lagged_truth_is_missing()
    {
        var rows = new[] { Row(new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc), wet: false) };
        var truth = new Dictionary<DateTime, double>();

        var p = PrecipBaselines.Persistence(rows, truth, lagHours: 24);

        double.IsNaN(p[0]).Should().BeTrue();
    }

    [Fact]
    public void BestSingle_picks_column_with_lowest_brier()
    {
        // Build rows where ecmwf predicts wet perfectly and every other model is always wrong.
        // BestSingle should pick precip_ecmwf.
        var rows = new List<PrecipTrainingRow>();
        var t0 = new DateTime(2025, 6, 1, 0, 0, 0, DateTimeKind.Utc);
        for (int i = 0; i < 10; i++)
        {
            var wet = i % 2 == 0;
            rows.Add(Row(
                t0.AddHours(i),
                wet: wet,
                gfs:   wet ? 0.0 : 0.5,    // inverted — always wrong
                ecmwf: wet ? 0.5 : 0.0,    // always right
                icon:  wet ? 0.0 : 0.5,
                mf:    wet ? 0.0 : 0.5,
                ukmo:  wet ? 0.0 : 0.5,
                gem:   wet ? 0.0 : 0.5));
        }

        PrecipBaselines.BestSingle(rows).Should().Be("precip_ecmwf");
    }
}
