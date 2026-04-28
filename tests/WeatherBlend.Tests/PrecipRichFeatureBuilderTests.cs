using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipRichFeatureBuilderTests
{
    [Fact]
    public void ComputePersistence_sums_full_24h_and_72h_windows()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 96; h++) hourly[runTime.AddHours(-h)] = 0.1;  // 0.1 mm/h everywhere

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.Prev24hMm.Should().BeApproximately(0.1 * 24, 1e-9);
        p.Prev72hMm.Should().BeApproximately(0.1 * 72, 1e-9);
    }

    [Fact]
    public void ComputePersistence_counts_wet_hours_strictly_at_or_above_threshold()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 24; h++)
            hourly[runTime.AddHours(-h)] = h < 5 ? 0.1 : 0.09;

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.WetHoursLast24h.Should().Be(5);
    }

    [Fact]
    public void ComputePersistence_trailing_dry_walks_back_through_dry_hours_only()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 20; h++)
        {
            double mm = h switch { < 4 => 0.05, 4 => 1.0, _ => 0.0 };
            hourly[runTime.AddHours(-h)] = mm;
        }

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.DryHoursTrailing.Should().Be(4);
    }

    [Fact]
    public void ComputePersistence_returns_NaN_for_24h_sum_when_coverage_incomplete()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 72; h++) hourly[runTime.AddHours(-h)] = 0.0;
        hourly.Remove(runTime.AddHours(-10));

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        double.IsNaN(p.Prev24hMm).Should().BeTrue();
        double.IsNaN(p.Prev72hMm).Should().BeTrue();
        double.IsNaN(p.WetHoursLast24h).Should().BeTrue();
    }

    [Fact]
    public void ComputePersistence_trailing_dry_stops_at_missing_reading()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 3; h++) hourly[runTime.AddHours(-h)] = 0.0;

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.DryHoursTrailing.Should().Be(3);
    }

    [Fact]
    public void ComputePersistence_trailing_dry_zero_when_runtime_hour_is_wet()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double> { [runTime] = 2.0 };
        for (int h = 1; h < 20; h++) hourly[runTime.AddHours(-h)] = 0.0;

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.DryHoursTrailing.Should().Be(0);
    }

    [Fact]
    public void LoadHourlyRain_throws_diagnosable_error_when_rainfall_tree_is_empty()
    {
        // Regression: predict.yml used to skip 'data/truth/rainfall' in its R2 pull,
        // so Phase 3c predict would crash deep inside DuckDB with a raw "No files found"
        // IOException that gave CI operators no clue what tree to populate.
        using var tmp = new TempDirectory();
        var missingPath = Path.Combine(tmp.Path, "does_not_exist");

        var act = () => PrecipRichFeatureBuilder.LoadHourlyRain(
            rainfallPath: missingPath,
            locationName: "Bonehill Rocks, Dartmoor",
            stationName: "Bellever, Dartmoor",
            ct: CancellationToken.None);

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Rainfall truth tree is empty*")
            .WithMessage("*does_not_exist*")
            .WithMessage("*data/truth/rainfall*");
    }

    private sealed class TempDirectory : IDisposable
    {
        public string Path { get; }
        public TempDirectory()
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "wb_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }
        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { }
        }
    }
}
