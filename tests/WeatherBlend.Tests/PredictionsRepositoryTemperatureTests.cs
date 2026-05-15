using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Locks in the 2026-05-15 location-filter parameter on
/// <see cref="PredictionsRepository.GetTemperaturePredictions"/>. The 2026-05-14
/// station-keyed manifest migration moved temperature predictions under
/// per-location subtrees (<c>data/predictions/temperature/{location}/...</c>);
/// the read side now accepts an explicit locations list so verify can scope to
/// one location and future multi-loc rendering can scan all of them.
///
/// Previously the read query hardcoded <c>WHERE LocationName = '_cfg.Location.Name'</c>
/// which silently dropped any prediction whose location wasn't the configured
/// primary — that's exactly what would happen to Membury verify if the filter
/// regressed.
/// </summary>
public class PredictionsRepositoryTemperatureTests : IDisposable
{
    private readonly string _root;

    public PredictionsRepositoryTemperatureTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-temp-preds-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private async Task SeedAsync(string location, string version, DateTime valid, double temp)
    {
        var dir = Path.Combine(_root, "predictions", "temperature", location, $"model_version={version}", $"date={valid:yyyy-MM-dd}");
        Directory.CreateDirectory(dir);
        var rows = new[]
        {
            new TempPredictionRow
            {
                LocationName        = location,
                ModelVersion        = version,
                PredictionMadeAtUtc = valid.AddHours(-24),
                ValidTimeUtc        = valid,
                LeadHours           = 24,
                BlendTemperature    = temp,
                FeatureVectorHash   = "",
            },
        };
        await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "predictions.parquet"));
    }

    private PredictionsRepository MakeRepo(string configuredPrimary = "bonehill_rocks")
    {
        var cfg = new AppConfig
        {
            Locations = new List<LocationConfig>
            {
                new LocationConfig { Name = configuredPrimary, DisplayName = configuredPrimary, Latitude = 0, Longitude = 0, ElevationMeters = 0 },
            },
            Storage = new StorageConfig
            {
                ForecastsPath = Path.Combine(_root, "forecasts"),
                ObservationsPath = Path.Combine(_root, "obs"),
                Era5Path = Path.Combine(_root, "era5"),
                RainfallPath = Path.Combine(_root, "rainfall"),
                MetOfficeObsPath = Path.Combine(_root, "mo_obs"),
                ModelsPath = Path.Combine(_root, "models"),
                PredictionsPath = Path.Combine(_root, "predictions"),
                ReportsPath = Path.Combine(_root, "reports"),
            },
        };
        return new PredictionsRepository(NullLogger<PredictionsRepository>.Instance, cfg);
    }

    [Fact]
    public async Task GetTemperaturePredictions_scopes_to_named_location_when_filter_passed()
    {
        var t = new DateTime(2026, 5, 14, 12, 0, 0, DateTimeKind.Utc);
        await SeedAsync("bonehill_rocks", "v2026-04-28_232613",  t, 14.0);
        await SeedAsync("membury_devon",  "v2026-05-14_220359", t, 15.0);

        var repo = MakeRepo();
        var membury = repo.GetTemperaturePredictions(
            locations: new[] { "membury_devon" },
            start: t.AddHours(-1), end: t.AddHours(1), ct: default);

        membury.Should().HaveCount(1);
        membury[0].LocationName.Should().Be("membury_devon");
        membury[0].BlendTemperature.Should().Be(15.0);
    }

    [Fact]
    public async Task GetTemperaturePredictions_returns_all_locations_when_filter_null()
    {
        var t = new DateTime(2026, 5, 14, 12, 0, 0, DateTimeKind.Utc);
        await SeedAsync("bonehill_rocks", "v2026-04-28_232613",  t, 14.0);
        await SeedAsync("membury_devon",  "v2026-05-14_220359", t, 15.0);

        var repo = MakeRepo();
        var all = repo.GetTemperaturePredictions(
            locations: null,
            start: t.AddHours(-1), end: t.AddHours(1), ct: default);

        all.Should().HaveCount(2);
        all.Select(r => r.LocationName).Should().BeEquivalentTo(new[] { "bonehill_rocks", "membury_devon" });
    }

    [Fact]
    public async Task GetTemperaturePredictions_returns_all_locations_when_filter_empty()
    {
        var t = new DateTime(2026, 5, 14, 12, 0, 0, DateTimeKind.Utc);
        await SeedAsync("bonehill_rocks", "v2026-04-28_232613",  t, 14.0);
        await SeedAsync("membury_devon",  "v2026-05-14_220359", t, 15.0);

        var repo = MakeRepo();
        var all = repo.GetTemperaturePredictions(
            locations: Array.Empty<string>(),
            start: t.AddHours(-1), end: t.AddHours(1), ct: default);

        all.Should().HaveCount(2);
    }
}
