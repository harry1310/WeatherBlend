using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Evaluate;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin <see cref="VerifyHistoryWriter"/>'s file-naming convention and the
/// JSON shape on disk. Each weekly verify command builds one of these and
/// the Models page reads them back via System.Text.Json — so the camelCase
/// property naming, null-suppression, and target-prefixed filename are all
/// load-bearing for the round trip.
/// </summary>
public class VerifyHistoryWriterTests : IDisposable
{
    private readonly string _root;

    public VerifyHistoryWriterTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-verify-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Theory]
    [InlineData("temperature",          "verify_temperature_2026-05-04.json")]
    [InlineData("precipitation",        "verify_precipitation_2026-05-04.json")]
    [InlineData("dry_window",           "verify_dry_window_2026-05-04.json")]
    [InlineData("element_humidity",     "verify_element_humidity_2026-05-04.json")]
    [InlineData("element_shortwave_radiation", "verify_element_shortwave_radiation_2026-05-04.json")]
    public void FileNameFor_uses_target_prefix_and_yyyy_mm_dd_asof(string target, string expected)
    {
        var asOf = new DateTime(2026, 5, 4, 9, 30, 0, DateTimeKind.Utc);
        VerifyHistoryWriter.FileNameFor(target, asOf).Should().Be(expected);
    }

    [Fact]
    public async Task WriteAsync_round_trips_through_System_Text_Json_camelCase()
    {
        // The Models-page renderer deserialises the same files the verify
        // commands write, so the JSON property names + null-handling have
        // to round-trip cleanly. Pin the camelCase + null-suppression rules
        // here because any drift would break the on-site history table.
        var asOf = new DateTime(2026, 5, 4, 9, 30, 0, DateTimeKind.Utc);
        var file = new VerifyHistoryFile
        {
            Target = "precipitation",
            AsOfUtc = asOf,
            WindowDays = 30,
            LatencyDays = 5,
            MetricLabel = "Brier",
            Rows = new List<VerifyHistoryRow>
            {
                new()
                {
                    Station = "ea_bellever_dartmoor",
                    ModelVersion = "v2026-04-28_232709",
                    LeadHours = 24,
                    WindowHours = null,
                    N = 426,
                    BlendMetric = 0.184,
                    ClimMetric = 0.221,
                    MeanOfModelsMetric = 0.205,
                    BestSingleName = "ecmwf_ifs025",
                    BestSingleMetric = 0.196,
                    ReferenceTrainingMetric = 0.179,
                    DriftFlag = false,
                },
            },
        };

        await VerifyHistoryWriter.WriteAsync(_root, file, default);

        var path = Path.Combine(_root, "verify_precipitation_2026-05-04.json");
        File.Exists(path).Should().BeTrue();

        var json = await File.ReadAllTextAsync(path);
        // camelCase property names — needed because the renderer deserialises
        // case-insensitively but the on-disk shape should still be the
        // standard JS convention.
        json.Should().Contain("\"target\": \"precipitation\"");
        json.Should().Contain("\"asOfUtc\":");
        json.Should().Contain("\"metricLabel\": \"Brier\"");
        // null fields suppressed (WhenWritingNull) so windowHours doesn't
        // pollute precipitation/temperature payloads.
        json.Should().NotContain("\"windowHours\"");

        // Round-trip via the same options the renderer uses.
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
        };
        var reread = JsonSerializer.Deserialize<VerifyHistoryFile>(json, options);
        reread.Should().NotBeNull();
        reread!.Target.Should().Be("precipitation");
        reread.AsOfUtc.Should().Be(asOf);
        reread.Rows.Should().HaveCount(1);
        reread.Rows[0].Station.Should().Be("ea_bellever_dartmoor");
        reread.Rows[0].WindowHours.Should().BeNull();
        reread.Rows[0].DriftFlag.Should().BeFalse();
        reread.Rows[0].BlendMetric.Should().BeApproximately(0.184, 1e-9);
    }

    [Fact]
    public async Task WriteAsync_creates_target_dir_when_missing()
    {
        var subdir = Path.Combine(_root, "nested", "reports");
        var asOf = new DateTime(2026, 5, 4, 9, 30, 0, DateTimeKind.Utc);
        var file = new VerifyHistoryFile
        {
            Target = "temperature", AsOfUtc = asOf,
            WindowDays = 14, LatencyDays = 5, MetricLabel = "MAE (°C)",
            Rows = new List<VerifyHistoryRow>(),
        };

        await VerifyHistoryWriter.WriteAsync(subdir, file, default);

        Directory.Exists(subdir).Should().BeTrue();
        File.Exists(Path.Combine(subdir, "verify_temperature_2026-05-04.json")).Should().BeTrue();
    }
}
