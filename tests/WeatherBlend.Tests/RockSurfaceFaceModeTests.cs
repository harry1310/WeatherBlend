using System.Diagnostics.CodeAnalysis;
using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Models;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Cliff-face mode plumbing (SENNEN_ROCK_TEMP_PLAN.md S3): the new
/// <c>Face</c> column must not break the merge-on-write path against
/// pre-face parquet files, and the merge dedup key must keep per-face (and
/// per-location — all locations share one per-date file) rows that share a
/// PredictionMadeAtUtc.
/// </summary>
public class RockSurfaceFaceModeTests
{
    /// <summary>The row schema EXACTLY as written before the Face column —
    /// serialising this and reading it back through the current class pins
    /// Parquet.NET's missing-column behaviour (constructor default survives).</summary>
    private sealed class LegacyRow
    {
        [SetsRequiredMembers]
        public LegacyRow() { }

        public string LocationName { get; init; } = "";
        public string ModelVersion { get; init; } = "";
        public DateTime PredictionMadeAtUtc { get; init; }
        public DateTime ValidTimeUtc { get; init; }
        public int LeadHours { get; init; }
        public double RockSurfaceTempC { get; init; }
        public double AirTempC { get; init; }
        public double DewPointC { get; init; }
        public double CondensationMarginC { get; init; }
        public string GreasinessStatus { get; init; } = "";
        public double ShortwaveDownWm2 { get; init; }
        public double CloudCoverPct { get; init; }
        public double WindSpeed10mMs { get; init; }
        public double LongwaveDownWm2 { get; init; }
        public double DeepTempC { get; init; }
        public string TempModelVersion { get; init; } = "";
        public string WindModelVersion { get; init; } = "";
        public string RadiationModelVersion { get; init; } = "";
        public string CloudModelVersion { get; init; } = "";
        public string DewPointSource { get; init; } = "";
    }

    [Fact]
    public async Task Pre_face_parquet_deserialises_with_empty_face()
    {
        var path = Path.Combine(Path.GetTempPath(), $"rock_legacy_{Guid.NewGuid():N}.parquet");
        try
        {
            var legacy = new[]
            {
                new LegacyRow
                {
                    LocationName = "bonehill_rocks", ModelVersion = "v1",
                    PredictionMadeAtUtc = new DateTime(2026, 6, 10, 6, 0, 0, DateTimeKind.Utc),
                    ValidTimeUtc = new DateTime(2026, 6, 11, 12, 0, 0, DateTimeKind.Utc),
                    LeadHours = 24, RockSurfaceTempC = 14.2, AirTempC = 12.0, DewPointC = 9.5,
                    CondensationMarginC = 4.7, GreasinessStatus = "dry",
                    ShortwaveDownWm2 = 500, CloudCoverPct = 40, WindSpeed10mMs = 3.2,
                    LongwaveDownWm2 = 320, DeepTempC = 11.0,
                    TempModelVersion = "vT", WindModelVersion = "vW",
                    RadiationModelVersion = "vR", CloudModelVersion = "vC",
                    DewPointSource = "nwp_mean",
                },
            };
            await ParquetSerializer.SerializeAsync(legacy, path);

            var rows = (await ParquetSerializer.DeserializeAsync<RockSurfacePredictionRow>(path)).ToList();

            rows.Should().HaveCount(1);
            rows[0].Face.Should().Be("");
            rows[0].SeaSurfaceTempC.Should().BeNull("pre-S4 rows have no SST column");
            rows[0].LocationName.Should().Be("bonehill_rocks");
            rows[0].RockSurfaceTempC.Should().Be(14.2);
            rows[0].GreasinessStatus.Should().Be("dry");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Merge_keeps_per_face_and_per_location_rows_from_one_instant()
    {
        var madeAt = new DateTime(2026, 6, 13, 6, 0, 0, DateTimeKind.Utc);
        var valid = new DateTime(2026, 6, 14, 12, 0, 0, DateTimeKind.Utc);

        RockSurfacePredictionRow Row(string loc, string face) => new()
        {
            LocationName = loc, ModelVersion = "v1", Face = face,
            PredictionMadeAtUtc = madeAt, ValidTimeUtc = valid, LeadHours = 24,
            RockSurfaceTempC = 10, AirTempC = 10, DewPointC = 8, CondensationMarginC = 2,
            GreasinessStatus = "potentially_greasy",
            ShortwaveDownWm2 = 0, CloudCoverPct = 0, WindSpeed10mMs = 0,
            LongwaveDownWm2 = 0, DeepTempC = 0,
            TempModelVersion = "", WindModelVersion = "", RadiationModelVersion = "",
            CloudModelVersion = "", DewPointSource = "",
        };

        var incoming = new[]
        {
            Row("sennen_cove", "west"),
            Row("sennen_cove", "southwest"),
            Row("sennen_cove", "south"),
            Row("bonehill_rocks", ""),
        };

        var merged = RockSurfacePredictCommand.MergeRows(
            existing: Array.Empty<RockSurfacePredictionRow>(), incoming: incoming);

        merged.Should().HaveCount(4, "rows differing only in Face/Location share an instant and must all survive");

        // Re-running the same instant is still idempotent per (loc, face).
        var again = RockSurfacePredictCommand.MergeRows(existing: merged, incoming: incoming);
        again.Should().HaveCount(4);
    }
}
