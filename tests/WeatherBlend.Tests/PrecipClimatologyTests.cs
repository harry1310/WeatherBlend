using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipClimatologyTests
{
    private static PrecipTrainingRow Row(DateTime t, bool wet) => new()
    {
        ValidTimeUtc = t,
        PrecipGfs = 0, PrecipEcmwf = 0, PrecipIcon = 0, PrecipMf = 0, PrecipUkmo = 0, PrecipGem = 0,
        ProbGfs = 0, ProbEcmwf = 0, ProbIcon = 0, ProbMf = 0, ProbUkmo = 0, ProbGem = 0,
        PrecipMean = 0, PrecipStd = 0, PrecipMax = 0, PrecipAgreementWet01 = 0,
        RhMean = 0, DewDepressionMean = 0,
        CloudLowMean = 0, CloudMidMean = 0, CloudHighMean = 0,
        CapeMean = 0, WindSpeedMean = 0,
        HourSin = 0, HourCos = 1, DoySin = 0, DoyCos = 1,
        WetBinary = wet,
        PrecipMmHour = wet ? 0.5f : 0.0f,
    };

    [Fact]
    public void BuildFromTraining_computes_bucket_wet_rate_and_global_mean()
    {
        // Jan 12:00 — 2 wet / 3 total.
        // Jul 06:00 — 0 wet / 2 total.
        // Global — 2 wet / 5 total = 0.4.
        var train = new[]
        {
            Row(new DateTime(2024, 1, 5,  12, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 1, 6,  12, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 1, 7,  12, 0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2024, 7, 5,   6, 0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2024, 7, 6,   6, 0, 0, DateTimeKind.Utc), wet: false),
        };

        var clim = PrecipClimatology.BuildFromTraining(train);

        clim.SourceRowCount.Should().Be(5);
        clim.GlobalWetRate.Should().BeApproximately(0.4, 1e-9);
        clim.Predict(new DateTime(2026, 1, 20, 12, 0, 0, DateTimeKind.Utc))
            .Should().BeApproximately(2.0 / 3.0, 1e-9);
        clim.Predict(new DateTime(2026, 7, 11, 6, 0, 0, DateTimeKind.Utc))
            .Should().Be(0.0);
    }

    [Fact]
    public void Predict_falls_back_to_global_on_unseen_month_hour_bucket()
    {
        var train = new[]
        {
            Row(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 1, 2, 0, 0, 0, DateTimeKind.Utc), wet: false),
        };

        var clim = PrecipClimatology.BuildFromTraining(train);

        // (Dec, 23:00) is unseen — falls back to global 0.5.
        clim.Predict(new DateTime(2026, 12, 25, 23, 0, 0, DateTimeKind.Utc))
            .Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void BuildFromTraining_produces_NaN_global_on_empty_input()
    {
        var clim = PrecipClimatology.BuildFromTraining(Array.Empty<PrecipTrainingRow>());

        double.IsNaN(clim.GlobalWetRate).Should().BeTrue();
        clim.SourceRowCount.Should().Be(0);
        clim.PWetByBucket.Should().BeEmpty();
    }

    [Fact]
    public void SaveTo_LoadFrom_roundtrips_identically()
    {
        var train = new[]
        {
            Row(new DateTime(2024, 3, 1, 9,  0, 0, DateTimeKind.Utc), wet: true),
            Row(new DateTime(2024, 3, 2, 9,  0, 0, DateTimeKind.Utc), wet: false),
            Row(new DateTime(2024, 9, 5, 15, 0, 0, DateTimeKind.Utc), wet: true),
        };
        var clim = PrecipClimatology.BuildFromTraining(train);

        var tmp = Path.Combine(Path.GetTempPath(), $"wb_climatology_{Guid.NewGuid():N}.json");
        try
        {
            clim.SaveTo(tmp);
            var loaded = PrecipClimatology.LoadFrom(tmp);

            loaded.SourceRowCount.Should().Be(clim.SourceRowCount);
            loaded.GlobalWetRate.Should().Be(clim.GlobalWetRate);
            loaded.WetThresholdMm.Should().Be(clim.WetThresholdMm);
            loaded.PWetByBucket.Should().BeEquivalentTo(clim.PWetByBucket);
            loaded.Predict(new DateTime(2026, 3, 10, 9, 0, 0, DateTimeKind.Utc))
                .Should().Be(clim.Predict(new DateTime(2026, 3, 10, 9, 0, 0, DateTimeKind.Utc)));
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }
}
