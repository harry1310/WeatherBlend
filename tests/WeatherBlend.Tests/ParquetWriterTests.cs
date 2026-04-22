using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using Xunit;

namespace WeatherBlend.Tests;

public class ParquetWriterTests : IDisposable
{
    private readonly string _root;

    public ParquetWriterTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-parquet-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static ObservationRow Obs(DateTime t, double? temp, string raw)
        => new()
        {
            LocationName = "Bonehill Rocks",
            Station = "EGTE",
            ObservedTimeUtc = DateTime.SpecifyKind(t, DateTimeKind.Utc),
            RawMetar = raw,
            Temperature2m = temp,
        };

    private string FileFor(DateTime day)
        => Path.Combine(
            _root,
            "location=Bonehill Rocks",
            "station=EGTE",
            $"date={day:yyyy-MM-dd}",
            "observations.parquet");

    [Fact]
    public async Task WriteObservationsAsync_creates_hive_partitioned_file()
    {
        var t = new DateTime(2026, 4, 21, 10, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteObservationsAsync(_root, new[] { Obs(t, 12.3, "METAR A") });

        var expected = FileFor(t);
        File.Exists(expected).Should().BeTrue($"expected file at {expected}");
    }

    [Fact]
    public async Task WriteObservationsAsync_appends_non_colliding_rows_to_existing_file()
    {
        var day = new DateTime(2026, 4, 21, 0, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs(day.AddHours(10), 12.0, "A"),
            Obs(day.AddHours(11), 12.5, "B"),
        });
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs(day.AddHours(12), 13.0, "C"),
        });

        var rows = await ReadBack(FileFor(day));
        rows.Should().HaveCount(3);
        rows.Select(r => r.ObservedTimeUtc.Hour).Should().Equal(10, 11, 12);
        rows.Select(r => r.RawMetar).Should().Equal("A", "B", "C");
    }

    [Fact]
    public async Task WriteObservationsAsync_dedupes_on_ObservedTimeUtc_last_write_wins()
    {
        var day = new DateTime(2026, 4, 21, 0, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs(day.AddHours(10), 12.0, "stale"),
        });

        // Second write targets the same ObservedTimeUtc — new payload must win.
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs(day.AddHours(10), 12.9, "fresh"),
        });

        var rows = await ReadBack(FileFor(day));
        rows.Should().ContainSingle();
        rows[0].Temperature2m.Should().BeApproximately(12.9, 1e-9);
        rows[0].RawMetar.Should().Be("fresh");
    }

    [Fact]
    public async Task WriteObservationsAsync_orders_merged_rows_by_time()
    {
        var day = new DateTime(2026, 4, 21, 0, 0, 0, DateTimeKind.Utc);

        // Write in non-chronological order to ensure the writer sorts on merge.
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs(day.AddHours(14), 14.0, "D"),
            Obs(day.AddHours(9),  11.0, "A"),
        });
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs(day.AddHours(12), 13.0, "C"),
            Obs(day.AddHours(10), 12.0, "B"),
        });

        var rows = await ReadBack(FileFor(day));
        rows.Select(r => r.ObservedTimeUtc.Hour).Should().Equal(9, 10, 12, 14);
    }

    [Fact]
    public async Task WriteObservationsAsync_is_noop_for_empty_input()
    {
        await ParquetWriter.WriteObservationsAsync(_root, Array.Empty<ObservationRow>());

        Directory.EnumerateFileSystemEntries(_root).Should().BeEmpty(
            "no rows → no partitions should be created");
    }

    [Fact]
    public async Task WriteObservationsAsync_partitions_by_observation_date()
    {
        // Two rows spanning a UTC date boundary go into separate partitions.
        var rows = new[]
        {
            Obs(new DateTime(2026, 4, 21, 23, 30, 0), 12.0, "late-21"),
            Obs(new DateTime(2026, 4, 22,  0, 30, 0), 11.5, "early-22"),
        };
        await ParquetWriter.WriteObservationsAsync(_root, rows);

        File.Exists(FileFor(new DateTime(2026, 4, 21))).Should().BeTrue();
        File.Exists(FileFor(new DateTime(2026, 4, 22))).Should().BeTrue();
    }

    private static async Task<List<ObservationRow>> ReadBack(string path)
        => (await ParquetSerializer.DeserializeAsync<ObservationRow>(path)).ToList();

    // ---- Rainfall ---------------------------------------------------------------

    private static RainfallRow Rain(DateTime t, double? mm, string quality = "Good", string completeness = "Complete")
        => new()
        {
            LocationName = "Bonehill Rocks",
            StationId = "bellever",
            StationName = "Bellever Dartmoor",
            ObservedTimeUtc = DateTime.SpecifyKind(t, DateTimeKind.Utc),
            Value15MinMm = mm,
            Quality = quality,
            Completeness = completeness,
        };

    private string RainFileFor(DateTime day)
        => Path.Combine(
            _root,
            "location=Bonehill Rocks",
            "station=Bellever Dartmoor",
            $"date={day:yyyy-MM-dd}",
            "rainfall.parquet");

    [Fact]
    public async Task WriteRainfallAsync_creates_hive_partitioned_file()
    {
        var t = new DateTime(2026, 4, 21, 10, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteRainfallAsync(_root, new[] { Rain(t, 0.2) });

        File.Exists(RainFileFor(t)).Should().BeTrue();
    }

    [Fact]
    public async Task WriteRainfallAsync_dedupes_on_ObservedTimeUtc_last_write_wins()
    {
        var t = new DateTime(2026, 4, 21, 10, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteRainfallAsync(_root, new[] { Rain(t, 0.1, quality: "Unchecked") });
        await ParquetWriter.WriteRainfallAsync(_root, new[] { Rain(t, 0.3, quality: "Good") });

        var rows = (await ParquetSerializer.DeserializeAsync<RainfallRow>(RainFileFor(t))).ToList();
        rows.Should().ContainSingle();
        rows[0].Value15MinMm.Should().BeApproximately(0.3, 1e-9);
        rows[0].Quality.Should().Be("Good");
    }

    [Fact]
    public async Task WriteRainfallAsync_is_noop_for_empty_input()
    {
        await ParquetWriter.WriteRainfallAsync(_root, Array.Empty<RainfallRow>());
        Directory.EnumerateFileSystemEntries(_root).Should().BeEmpty();
    }

    [Fact]
    public async Task WriteRainfallAsync_partitions_by_observation_date_and_station()
    {
        // Rows spanning a date boundary — different partitions.
        var crossDay = new[]
        {
            Rain(new DateTime(2026, 4, 21, 23, 45, 0), 0.1),
            Rain(new DateTime(2026, 4, 22,  0,  0, 0), 0.2),
        };
        await ParquetWriter.WriteRainfallAsync(_root, crossDay);

        File.Exists(RainFileFor(new DateTime(2026, 4, 21))).Should().BeTrue();
        File.Exists(RainFileFor(new DateTime(2026, 4, 22))).Should().BeTrue();

        // Different station → different partition directory.
        var princetown = new RainfallRow
        {
            LocationName = "Bonehill Rocks",
            StationId = "princetown",
            StationName = "Princetown",
            ObservedTimeUtc = new DateTime(2026, 4, 21, 10, 0, 0, DateTimeKind.Utc),
            Value15MinMm = 0.4,
            Quality = "Good",
            Completeness = "Complete",
        };
        await ParquetWriter.WriteRainfallAsync(_root, new[] { princetown });

        var princetownFile = Path.Combine(_root,
            "location=Bonehill Rocks", "station=Princetown", "date=2026-04-21", "rainfall.parquet");
        File.Exists(princetownFile).Should().BeTrue();
    }

    [Fact]
    public async Task WriteRainfallAsync_orders_merged_rows_by_time()
    {
        var day = new DateTime(2026, 4, 21, 0, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteRainfallAsync(_root, new[]
        {
            Rain(day.AddHours(14).AddMinutes(30), 0.4),
            Rain(day.AddHours( 9).AddMinutes(15), 0.1),
        });
        await ParquetWriter.WriteRainfallAsync(_root, new[]
        {
            Rain(day.AddHours(12).AddMinutes(0),  0.3),
            Rain(day.AddHours(10).AddMinutes(45), 0.2),
        });

        var rows = (await ParquetSerializer.DeserializeAsync<RainfallRow>(RainFileFor(day))).ToList();
        rows.Select(r => r.ObservedTimeUtc)
            .Should().BeInAscendingOrder();
        rows.Select(r => r.Value15MinMm).Should().Equal(0.1, 0.2, 0.3, 0.4);
    }
}
