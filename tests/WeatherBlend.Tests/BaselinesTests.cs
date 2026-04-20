using FluentAssertions;
using WeatherBlend.Evaluate;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class BaselinesTests
{
    private static TrainingRow Row(DateTime t, double era5, double gfs = 0, double ecmwf = 0,
        double icon = 0, double mf = 0, double ukmo = 0, double gem = 0)
        => new()
        {
            ValidTimeUtc = t,
            TempGfs = (float)gfs, TempEcmwf = (float)ecmwf, TempIcon = (float)icon,
            TempMf = (float)mf, TempUkmo = (float)ukmo, TempGem = (float)gem,
            TempMean = (float)((gfs + ecmwf + icon + mf + ukmo + gem) / 6.0),
            TempStd = 0, TempRange = 0,
            HourSin = 0, HourCos = 1, DoySin = 0, DoyCos = 1,
            WindDirMean = 0, Era5Temp = (float)era5,
        };

    [Fact]
    public void Persistence_returns_nan_when_lookup_is_missing()
    {
        var rows = new[]
        {
            Row(new DateTime(2025, 1, 2, 0, 0, 0, DateTimeKind.Utc), 5),
        };
        var truth = new Dictionary<DateTime, double>();  // nothing 24h before the target

        var p = Baselines.Persistence(rows, truth, 24);

        p.Should().HaveCount(1);
        double.IsNaN(p[0]).Should().BeTrue();
    }

    [Fact]
    public void Persistence_looks_up_exact_time_minus_lag()
    {
        var t = new DateTime(2025, 1, 2, 12, 0, 0, DateTimeKind.Utc);
        var rows = new[] { Row(t, 10) };
        var truth = new Dictionary<DateTime, double>
        {
            [t.AddHours(-24)] = 7.5,
        };

        var p = Baselines.Persistence(rows, truth, 24);
        p[0].Should().Be(7.5);
    }

    [Fact]
    public void Climatology_returns_per_month_per_hour_mean_from_training()
    {
        // Two training rows in (Jan, 12:00) with temps 4 and 6 → mean 5.
        // One target row also at (Jan, 12:00) expects the predicted 5.
        var trainRows = new[]
        {
            Row(new DateTime(2024, 1, 5, 12, 0, 0, DateTimeKind.Utc), 4),
            Row(new DateTime(2024, 1, 6, 12, 0, 0, DateTimeKind.Utc), 6),
        };
        var target = new[]
        {
            Row(new DateTime(2025, 1, 7, 12, 0, 0, DateTimeKind.Utc), 99), // 99 ignored by climatology
        };

        var p = Baselines.Climatology(trainRows, target);

        p.Should().HaveCount(1);
        p[0].Should().BeApproximately(5.0, 1e-9);
    }

    [Fact]
    public void Climatology_falls_back_to_global_mean_on_unseen_bucket()
    {
        var trainRows = new[]
        {
            Row(new DateTime(2024, 6, 1, 0, 0, 0, DateTimeKind.Utc), 10),
            Row(new DateTime(2024, 6, 1, 1, 0, 0, DateTimeKind.Utc), 20),
        };
        // Target in (Dec, 23:00) — not in training — global mean is 15.
        var target = new[] { Row(new DateTime(2025, 12, 31, 23, 0, 0, DateTimeKind.Utc), 0) };

        var p = Baselines.Climatology(trainRows, target);

        p[0].Should().BeApproximately(15.0, 1e-9);
    }

    [Fact]
    public void MeanOfModels_equals_arithmetic_mean()
    {
        var rows = new[]
        {
            Row(DateTime.UtcNow, era5: 0, gfs: 1, ecmwf: 2, icon: 3, mf: 4, ukmo: 5, gem: 6),
        };
        var p = Baselines.MeanOfModels(rows);
        p[0].Should().BeApproximately((1 + 2 + 3 + 4 + 5 + 6) / 6.0, 1e-6);
    }

    [Fact]
    public void BestSingle_picks_lowest_mae_on_provided_rows()
    {
        // GFS nails the target; others are off. Best should be temp_gfs.
        var rows = new[]
        {
            Row(DateTime.UtcNow,              era5: 10, gfs: 10, ecmwf: 13, icon: 14, mf: 15, ukmo: 16, gem: 17),
            Row(DateTime.UtcNow.AddHours(1),  era5: 11, gfs: 11, ecmwf: 14, icon: 15, mf: 16, ukmo: 17, gem: 18),
        };

        Baselines.BestSingle(rows).Should().Be("temp_gfs");
    }
}
