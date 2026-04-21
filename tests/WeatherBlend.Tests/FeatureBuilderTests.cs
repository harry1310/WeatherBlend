using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class FeatureBuilderTests
{
    [Fact]
    public void FeatureNames_has_exactly_13_columns_in_expected_order()
    {
        FeatureBuilder.FeatureNames.Should().HaveCount(13);
        FeatureBuilder.FeatureNames.Should().ContainInOrder(
            "temp_gfs", "temp_ecmwf", "temp_icon", "temp_mf", "temp_ukmo", "temp_gem",
            "temp_mean", "temp_std", "temp_range",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos");
    }

    [Fact]
    public void ComposeRow_computes_mean_std_range_correctly()
    {
        // Six values with known statistics.
        var temps = new[] { 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };
        var row = FeatureBuilder.ComposeRow(
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            temps,
            windDirMeanDeg: double.NaN,
            era5Temp: 15.0);

        row.TempMean.Should().BeApproximately(15f, 1e-4f);
        row.TempRange.Should().BeApproximately(10f, 1e-4f);
        // Population std of [10,12,14,16,18,20]: sqrt(((−5)^2+(−3)^2+(−1)^2+1^2+3^2+5^2)/6) = sqrt(70/6) ≈ 3.4156
        row.TempStd.Should().BeApproximately(3.4156f, 1e-3f);
    }

    [Theory]
    [InlineData(0,   0.0,   1.0)]   // sin(0)=0, cos(0)=1
    [InlineData(6,   1.0,   0.0)]   // sin(π/2)=1, cos(π/2)=0
    [InlineData(12,  0.0,  -1.0)]   // sin(π)=0, cos(π)=-1
    [InlineData(18, -1.0,   0.0)]   // sin(3π/2)=-1, cos(3π/2)=0
    public void Cyclical_hour_encoding_matches_unit_circle(int hour, double expectedSin, double expectedCos)
    {
        var row = FeatureBuilder.ComposeRow(
            new DateTime(2025, 1, 1, hour, 0, 0, DateTimeKind.Utc),
            new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 },
            windDirMeanDeg: double.NaN,
            era5Temp: 0.0);

        row.HourSin.Should().BeApproximately((float)expectedSin, 1e-5f);
        row.HourCos.Should().BeApproximately((float)expectedCos, 1e-5f);
    }

    [Fact]
    public void Cyclical_doy_encoding_wraps_to_near_zero_at_year_boundary()
    {
        // DoY 1 should map to (sin ≈ 0, cos ≈ 1) since angle = 2π * 0 / 365 = 0.
        var row = FeatureBuilder.ComposeRow(
            new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
            windDirMeanDeg: double.NaN,
            era5Temp: 0.0);

        row.DoySin.Should().BeApproximately(0f, 1e-5f);
        row.DoyCos.Should().BeApproximately(1f, 1e-5f);
    }

    [Fact]
    public void ComposeRow_rejects_wrong_number_of_model_temps()
    {
        var act = () => FeatureBuilder.ComposeRow(
            DateTime.UtcNow, new[] { 1.0, 2.0 }, double.NaN, 0);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void TrainingDataset_split_is_chronological_and_non_overlapping()
    {
        // 20 rows across ~20 distinct valid times.
        var rows = Enumerable.Range(0, 20).Select(i => new TrainingRow
        {
            ValidTimeUtc = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddHours(i),
            TempGfs = 1, TempEcmwf = 1, TempIcon = 1, TempMf = 1, TempUkmo = 1, TempGem = 1,
            TempMean = 1, TempStd = 0, TempRange = 0,
            HourSin = 0, HourCos = 1, DoySin = 0, DoyCos = 1,
            WindDirMean = 0, Era5Temp = 1,
        }).ToList();

        var ds = TrainingDataset.Split(rows);

        ds.Train.Count.Should().Be(14);          // floor(20*0.70)
        ds.Val.Count.Should().Be(3);             // floor(20*0.15)
        ds.Test.Count.Should().Be(3);            // remaining
        ds.TrainEnd.Should().BeBefore(ds.ValStart);
        ds.ValEnd.Should().BeBefore(ds.TestStart);
    }

    [Fact]
    public void TrainingDataset_split_throws_on_out_of_order_input()
    {
        var now = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var rows = new List<TrainingRow>
        {
            MakeRow(now.AddHours(5)),
            MakeRow(now.AddHours(3)),   // out of order — should trip the assertion
            MakeRow(now.AddHours(6)),
            MakeRow(now.AddHours(7)),
            MakeRow(now.AddHours(8)),
            MakeRow(now.AddHours(9)),
            MakeRow(now.AddHours(10)),
            MakeRow(now.AddHours(11)),
            MakeRow(now.AddHours(12)),
            MakeRow(now.AddHours(13)),
        };

        var act = () => TrainingDataset.Split(rows);
        act.Should().Throw<InvalidOperationException>().WithMessage("*ascending*");
    }

    private static TrainingRow MakeRow(DateTime t) => new()
    {
        ValidTimeUtc = t,
        TempGfs = 1, TempEcmwf = 1, TempIcon = 1, TempMf = 1, TempUkmo = 1, TempGem = 1,
        TempMean = 1, TempStd = 0, TempRange = 0,
        HourSin = 0, HourCos = 1, DoySin = 0, DoyCos = 1,
        WindDirMean = 0, Era5Temp = 1,
    };

    [Fact]
    public void LeadModelFileName_formats_as_leadNh_zip()
    {
        WeatherBlend.Train.ModelArtifact.LeadModelFileName(24).Should().Be("lead_24h.zip");
        WeatherBlend.Train.ModelArtifact.LeadModelFileName(72).Should().Be("lead_72h.zip");
    }

    [Fact]
    public void TemperatureTrainer_end_to_end_on_synthetic_data_produces_non_nan_metrics()
    {
        // Tiny synthetic set: ERA5 truth is a known linear combination of per-model temps
        // so the trainer has something to fit. Enough rows to survive the 70/15/15 split
        // with val/test having >= 1 row each.
        var rng = new Random(42);
        var rows = new List<TrainingRow>();
        var start = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        for (int i = 0; i < 400; i++)
        {
            var temps = new double[6];
            for (int m = 0; m < 6; m++) temps[m] = 10.0 + rng.NextDouble() * 5.0;

            // Simulate ERA5 = mean + small noise, so the optimal blend is close to mean.
            var mean = temps.Average();
            var era5 = mean + (rng.NextDouble() - 0.5) * 0.2;

            rows.Add(FeatureBuilder.ComposeRow(
                start.AddHours(i),
                temps,
                windDirMeanDeg: 180.0,
                era5Temp: era5));
        }

        var ds = TrainingDataset.Split(rows);
        var hp = new TemperatureTrainer.Hyperparameters(
            NumberOfIterations: 50,        // keep the test fast
            EarlyStoppingRound: 10);
        var trained = TemperatureTrainer.Train(ds, hp);

        var predicted = TemperatureTrainer.Predict(trained.Ml, trained.Model, ds.Test);
        predicted.Should().HaveCount(ds.Test.Count);
        predicted.All(x => !double.IsNaN(x) && !double.IsInfinity(x)).Should().BeTrue();

        var actual = ds.Test.Select(x => (double)x.Era5Temp).ToArray();
        var stats = WeatherBlend.Evaluate.Metrics.Compute(predicted, actual);
        double.IsNaN(stats.Mae).Should().BeFalse();
        // Very loose bound — we're checking the pipeline runs, not learned quality.
        stats.Mae.Should().BeLessThan(5.0);
    }
}
