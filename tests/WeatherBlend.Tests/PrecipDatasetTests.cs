using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipDatasetTests
{
    private static PrecipTrainingRow At(DateTime t, bool wet = false) => new()
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
        PrecipMmHour = 0f,
    };

    private static IReadOnlyList<PrecipTrainingRow> ContiguousHours(DateTime start, int n)
        => Enumerable.Range(0, n).Select(i => At(start.AddHours(i))).ToList();

    [Fact]
    public void Split_70_15_15_gives_expected_row_counts()
    {
        var rows = ContiguousHours(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), 100);

        var ds = PrecipDataset.Split(rows);

        ds.Train.Should().HaveCount(70);
        ds.Val.Should().HaveCount(15);
        ds.Test.Should().HaveCount(15);
    }

    [Fact]
    public void Split_boundaries_are_strictly_chronological()
    {
        var rows = ContiguousHours(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), 100);

        var ds = PrecipDataset.Split(rows);

        ds.TrainEnd.Should().BeBefore(ds.ValStart);
        ds.ValEnd.Should().BeBefore(ds.TestStart);
        // Test partition endpoint accessors line up with last row.
        ds.TrainStart.Should().Be(rows[0].ValidTimeUtc);
        ds.TestEnd.Should().Be(rows[^1].ValidTimeUtc);
    }

    [Fact]
    public void Split_wet_counts_match_row_labels_per_partition()
    {
        // First 70 (train) contain 5 wet, next 15 (val) contain 2 wet, final 15 (test) contain 3 wet.
        var start = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var wetHours = new HashSet<int> { 10, 20, 30, 40, 50,   72, 78,   88, 92, 99 };
        var rows = Enumerable.Range(0, 100)
            .Select(i => At(start.AddHours(i), wet: wetHours.Contains(i)))
            .ToList();

        var ds = PrecipDataset.Split(rows);

        ds.TrainWet.Should().Be(5);
        ds.ValWet.Should().Be(2);
        ds.TestWet.Should().Be(3);
    }

    [Fact]
    public void Split_throws_when_rows_are_out_of_order()
    {
        var rows = ContiguousHours(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), 100).ToList();
        // Swap two rows to break monotonic ordering.
        (rows[50], rows[51]) = (rows[51], rows[50]);

        var act = () => PrecipDataset.Split(rows);

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*ascending by ValidTimeUtc*");
    }

    [Fact]
    public void Split_throws_on_too_few_rows()
    {
        var rows = ContiguousHours(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), 5);

        var act = () => PrecipDataset.Split(rows);

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*at least 10*");
    }

    [Fact]
    public void Split_throws_when_any_partition_is_empty()
    {
        // With 12 rows and default 70/15/15 fractions, val partition ends up being 1 row, still fine.
        // Force an empty test partition by passing fractions that sum above 1.0.
        var rows = ContiguousHours(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), 20);

        var act = () => PrecipDataset.Split(rows, trainFrac: 0.9, valFrac: 0.1);

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*empty partition*");
    }

    [Fact]
    public void Split_preserves_original_row_identity_within_partitions()
    {
        var rows = ContiguousHours(new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), 100);

        var ds = PrecipDataset.Split(rows);

        // train is rows[0..70), val is rows[70..85), test is rows[85..100).
        ds.Train[0].ValidTimeUtc.Should().Be(rows[0].ValidTimeUtc);
        ds.Train[^1].ValidTimeUtc.Should().Be(rows[69].ValidTimeUtc);
        ds.Val[0].ValidTimeUtc.Should().Be(rows[70].ValidTimeUtc);
        ds.Val[^1].ValidTimeUtc.Should().Be(rows[84].ValidTimeUtc);
        ds.Test[0].ValidTimeUtc.Should().Be(rows[85].ValidTimeUtc);
        ds.Test[^1].ValidTimeUtc.Should().Be(rows[99].ValidTimeUtc);
    }
}
