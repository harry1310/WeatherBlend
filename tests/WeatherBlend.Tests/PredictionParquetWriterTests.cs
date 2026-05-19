using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins <see cref="PredictionParquetWriter"/> — the shared merge-on-write the
/// predict commands route through. The per-command (Precip / FeelsLike /
/// StartHour) MergeRows tests cover the (PMT, Lead, ValidTime) and
/// 4-tuple keys; this file covers the day-granular (PMT, Lead) key (the
/// dry-window shape, otherwise untested) and the WriteAsync IO round-trip.
/// </summary>
public class PredictionParquetWriterTests
{
    private static PrecipPredictionRow Row(DateTime madeAt, int leadHours, DateTime validTime, double pWet)
        => new()
        {
            LocationName = "bonehill_rocks",
            TruthStation = "ea_bellever_dartmoor",
            ModelVersion = "v1",
            PredictionMadeAtUtc = madeAt,
            ValidTimeUtc = validTime,
            LeadHours = leadHours,
            ProbWet = pWet,
            ClimatologyPWet = 0.5,
            FeatureVectorHash = "abc",
        };

    private static List<PrecipPredictionRow> MergeByPmtLeadValid(
        IEnumerable<PrecipPredictionRow> existing, IEnumerable<PrecipPredictionRow> incoming)
        => PredictionParquetWriter.Merge(existing, incoming,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours));

    [Fact]
    public void Merge_keeps_distinct_cycles_as_separate_rows()
    {
        // Two cycles predicting the same (lead, valid) — distinct
        // PredictionMadeAtUtc ⇒ distinct key (it is part of the key) ⇒ both
        // survive. This is the history-preservation property of merge-on-write.
        var valid = new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc);
        var cycleA = Row(new DateTime(2026, 4, 30, 6, 0, 0, DateTimeKind.Utc), 24, valid, 0.20);
        var cycleB = Row(new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc), 24, valid, 0.80);

        MergeByPmtLeadValid(new[] { cycleA }, new[] { cycleB })
            .Should().HaveCount(2);
    }

    [Fact]
    public void Merge_collapses_exact_key_repeats_to_one_row()
    {
        // A retry re-emitting the identical (PMT, Lead, ValidTime) row must not
        // double-count — GroupBy collapses it; MaxBy(freshness) picks the one.
        var made = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var valid = new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc);
        var rows = new[] { Row(made, 24, valid, 0.3), Row(made, 24, valid, 0.3) };

        MergeByPmtLeadValid(Array.Empty<PrecipPredictionRow>(), rows)
            .Should().ContainSingle();
    }

    [Fact]
    public void Merge_with_day_granular_key_collapses_ValidTime_siblings()
    {
        // The dry-window key is (PredictionMadeAtUtc, LeadHours) — no ValidTime.
        // Two rows that differ ONLY in ValidTime share a key and collapse to
        // one. Proves the caller-supplied key is honoured verbatim (omitting
        // ValidTime is a deliberate per-target choice, not a bug here).
        var made = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, 24, new DateTime(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc), 0.10),
            Row(made, 24, new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc), 0.90),
        };

        var merged = PredictionParquetWriter.Merge(
            Array.Empty<PrecipPredictionRow>(), rows,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rs => rs.OrderBy(r => r.ValidTimeUtc));

        merged.Should().ContainSingle();
    }

    [Fact]
    public void Merge_orders_by_ValidTime_then_Lead()
    {
        var made = new DateTime(2026, 4, 30, 10, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row(made, 72, new DateTime(2026, 5, 3, 0, 0, 0, DateTimeKind.Utc), 0.1),
            Row(made, 24, new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc), 0.2),
            Row(made, 48, new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc), 0.3),
        };

        var merged = MergeByPmtLeadValid(Array.Empty<PrecipPredictionRow>(), rows);

        merged.Select(r => (r.ValidTimeUtc, r.LeadHours)).Should().Equal(
            (new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc), 24),
            (new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc), 48),
            (new DateTime(2026, 5, 3, 0, 0, 0, DateTimeKind.Utc), 72));
    }

    [Fact]
    public async Task WriteAsync_merges_a_second_run_into_the_existing_parquet()
    {
        var dir = Path.Combine(Path.GetTempPath(), "wb_ppw_" + Guid.NewGuid().ToString("N"));
        var outPath = Path.Combine(dir, "model_version=v1", "date=2026-05-01", "predictions.parquet");
        try
        {
            var made1 = new DateTime(2026, 4, 30, 6, 0, 0, DateTimeKind.Utc);
            var made2 = new DateTime(2026, 4, 30, 12, 0, 0, DateTimeKind.Utc);
            var valid = new DateTime(2026, 5, 1, 6, 0, 0, DateTimeKind.Utc);

            // Run 1 — creates the parent dirs + file.
            var n1 = await PredictionParquetWriter.WriteAsync(
                outPath, new[] { Row(made1, 24, valid, 0.2) }, MergeByPmtLeadValid, CancellationToken.None);
            n1.Should().Be(1);
            File.Exists(outPath).Should().BeTrue();

            // Run 2 — a different cycle for the same (lead, valid). Distinct
            // PredictionMadeAtUtc ⇒ distinct key ⇒ both rows survive.
            var n2 = await PredictionParquetWriter.WriteAsync(
                outPath, new[] { Row(made2, 24, valid, 0.7) }, MergeByPmtLeadValid, CancellationToken.None);
            n2.Should().Be(2);

            var onDisk = (await ParquetSerializer.DeserializeAsync<PrecipPredictionRow>(outPath)).ToList();
            onDisk.Should().HaveCount(2);
            onDisk.Select(r => r.PredictionMadeAtUtc).Should().BeEquivalentTo(new[] { made1, made2 });
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
