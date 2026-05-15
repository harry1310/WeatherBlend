using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Locks in the 2026-05-14 change to <see cref="TruthRepository.GetMetarTemperature"/>:
/// METAR is per-ICAO, not per-(location, ICAO), so the read query must NOT filter
/// by <c>LocationName</c>. The earlier filter forced Membury to re-collect EGTE
/// for itself even though Bonehill already had ~3 years of EGTE under
/// <c>location=bonehill_rocks/station=EGTE/</c>. After the fix, Membury (or any
/// other location whose config points at EGTE) reads the existing partition for
/// free. Tests this by seeding under one location and reading via a TruthRepository
/// configured for a different location.
/// </summary>
public class TruthRepositoryMetarTests : IDisposable
{
    private readonly string _root;

    public TruthRepositoryMetarTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-truth-metar-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static ObservationRow Obs(string locationName, DateTime t, double temp)
        => new()
        {
            LocationName = locationName,
            Station = "EGTE",
            ObservedTimeUtc = DateTime.SpecifyKind(t, DateTimeKind.Utc),
            RawMetar = "METAR EGTE TEST",
            Temperature2m = temp,
        };

    private TruthRepository MakeRepo(string configuredLocation)
    {
        var cfg = new AppConfig
        {
            Locations = new List<LocationConfig>
            {
                new LocationConfig
                {
                    Name = configuredLocation,
                    DisplayName = configuredLocation,
                    Latitude = 0, Longitude = 0, ElevationMeters = 0,
                },
            },
            Storage = new StorageConfig
            {
                ForecastsPath = Path.Combine(_root, "forecasts"),
                ObservationsPath = _root,
                Era5Path = Path.Combine(_root, "era5"),
                RainfallPath = Path.Combine(_root, "rainfall"),
                MetOfficeObsPath = Path.Combine(_root, "mo_obs"),
                ModelsPath = Path.Combine(_root, "models"),
                PredictionsPath = Path.Combine(_root, "predictions"),
                ReportsPath = Path.Combine(_root, "reports"),
            },
        };
        return new TruthRepository(NullLogger<TruthRepository>.Instance, cfg);
    }

    [Fact]
    public async Task GetMetarTemperature_reads_EGTE_partition_stored_under_a_different_location()
    {
        // Seed EGTE observations under location=bonehill_rocks (the partition
        // the live collector wrote them to historically).
        var t = new DateTime(2026, 5, 10, 12, 0, 0, DateTimeKind.Utc);
        await ParquetWriter.WriteObservationsAsync(_root, new[]
        {
            Obs("bonehill_rocks", t,          14.5),
            Obs("bonehill_rocks", t.AddHours(1), 15.0),
        });

        // A repo configured for a DIFFERENT location (e.g. Membury) reads the
        // same EGTE rows — the read path is per-ICAO only.
        var repo = MakeRepo(configuredLocation: "membury_devon");
        var rows = repo.GetMetarTemperature(
            station: "EGTE",
            start: t.AddHours(-1),
            end:   t.AddHours(2),
            ct:    default);

        rows.Should().HaveCount(2);
        rows.Select(r => r.Temperature2m).Should().BeEquivalentTo(new[] { 14.5, 15.0 });
    }

    [Fact]
    public void GetMetarTemperature_returns_empty_for_blank_station()
    {
        var repo = MakeRepo(configuredLocation: "membury_devon");
        var rows = repo.GetMetarTemperature(
            station: "",
            start: DateTime.UtcNow.AddDays(-1),
            end:   DateTime.UtcNow,
            ct:    default);
        rows.Should().BeEmpty();
    }
}
