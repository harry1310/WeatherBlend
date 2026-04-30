using DuckDB.NET.Data;
using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Integration tests pinning the original <c>ApparentTemperatureC</c> regression
/// against the schema-probe path now exposed by
/// <see cref="WeatherBlend.Storage.ParquetReader.HasColumn"/>. When no parquet
/// in the tree carried the column, <c>union_by_name=true</c> excluded it from
/// the unified schema and DuckDB binder-errored on any SELECT referencing it.
/// Probing first prevents the crash; these tests pin the probe's behaviour
/// across the variants the renderer cares about.
/// </summary>
public class RenderSiteCommandTests : IDisposable
{
    private readonly string _root;

    public RenderSiteCommandTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-rendersite-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    // ---------- ParquetSchemaHasColumn ----------

    [Fact]
    public void ParquetSchemaHasColumn_returns_false_when_glob_matches_no_files()
    {
        // Empty directory → read_parquet throws "No files found" — the helper
        // must catch it and return false rather than propagate the exception.
        // Renderer relies on this to render the no-data fallback chip silently
        // when the prediction tree hasn't been synced yet.
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC")
            .Should().BeFalse();
    }

    [Fact]
    public async Task ParquetSchemaHasColumn_returns_false_when_no_file_carries_the_column()
    {
        // Old-schema parquets only — none carry ApparentTemperatureC. This is the
        // exact state that caused the binder error: union_by_name unifies columns
        // present in at least one file, and "zero files" means the column is
        // absent from the unified schema and SELECTing it explodes.
        await WriteParquetAsync("data1.parquet", new[]
        {
            new SimpleRow { Name = "a", UtciC = 1.0 },
            new SimpleRow { Name = "b", UtciC = 2.0 },
        });
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC")
            .Should().BeFalse();
        // Sanity: a column that DOES exist still resolves true on the same tree.
        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "UtciC")
            .Should().BeTrue();
    }

    [Fact]
    public async Task ParquetSchemaHasColumn_returns_true_when_at_least_one_file_carries_it()
    {
        // Mixed schema: one old file (no ApparentTemperatureC), one new file
        // (carrying it). The post-rename steady state where the writer has
        // emitted some new-schema rows but old rows survive on R2. The probe
        // must say "yes, the column is in the unified schema" so the SELECT
        // can reference it (NULL for old rows, populated for new).
        await WriteParquetAsync("old.parquet", new[]
        {
            new SimpleRow { Name = "old", UtciC = 1.0 },
        });
        await WriteParquetAsync("new.parquet", new[]
        {
            new ExtendedRow { Name = "new", UtciC = 2.0, ApparentTemperatureC = 5.5 },
        });
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC")
            .Should().BeTrue();
    }

    [Fact]
    public async Task ParquetSchemaHasColumn_handles_quoted_column_names_safely()
    {
        // Column name containing a quote would otherwise inject into the SQL
        // and either crash or match nothing. The helper escapes the literal.
        await WriteParquetAsync("d.parquet", new[] { new SimpleRow { Name = "x", UtciC = 1.0 } });
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        // No injection — the predicate just doesn't match anything.
        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "Bobby'); DROP TABLE--")
            .Should().BeFalse();
    }

    // ---------- ComputeRollingMae ----------

    private static WeatherBlend.Models.TempPredictionRow Pred(
        string version, int leadHours, DateTime validTime, double blendTemperature)
        => new()
        {
            LocationName = "bonehill_rocks",
            ModelVersion = version,
            PredictionMadeAtUtc = validTime.AddHours(-leadHours),
            ValidTimeUtc = validTime,
            LeadHours = leadHours,
            BlendTemperature = blendTemperature,
            FeatureVectorHash = "",
        };

    [Fact]
    public void ComputeRollingMae_returns_empty_when_no_pairs()
    {
        // No predictions at all → empty. Same with predictions whose valid
        // times are missing from the truth dict — pairing drops them.
        var truth = new Dictionary<DateTime, double>();
        RenderSiteCommand.ComputeRollingMae(
                Array.Empty<WeatherBlend.Models.TempPredictionRow>(),
                truth, windowDays: 14)
            .Should().BeEmpty();

        var pred = new[] { Pred("v1", 24,
            new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc), 10.0) };
        RenderSiteCommand.ComputeRollingMae(pred, truth, 14).Should().BeEmpty();
    }

    [Fact]
    public void ComputeRollingMae_emits_partial_window_points_when_data_shorter_than_window()
    {
        // Bug fix 2026-04-30: with 3 days of paired data and a 14-day rolling
        // window, the loop used to start at minDate + windowDays - 1 (= 14
        // days into the future) and never enter the body, leaving the chart
        // empty even though pairs existed. Now we emit a point per paired
        // day from minDate, with a partial window where needed.
        var t1 = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var t2 = new DateTime(2026, 4, 23, 12, 0, 0, DateTimeKind.Utc);
        var t3 = new DateTime(2026, 4, 24, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Pred("v1", 24, t1, 10.0),
            Pred("v1", 24, t2, 11.0),
            Pred("v1", 24, t3, 12.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [t1] = 9.0, [t2] = 10.0, [t3] = 13.0,
        };

        var points = RenderSiteCommand.ComputeRollingMae(preds, truth, windowDays: 14);

        // One point per paired day. Cumulative MAE because the window covers
        // all of the paired data behind each day.
        points.Should().HaveCount(3);
        points.Select(p => p.WindowEndUtc.Date).Should().Equal(t1.Date, t2.Date, t3.Date);
        points.Select(p => p.N).Should().Equal(1, 2, 3);
        points[0].BlendMae.Should().BeApproximately(1.0, 1e-9);   // |10-9|
        points[1].BlendMae.Should().BeApproximately(1.0, 1e-9);   // (1 + 1) / 2
        points[2].BlendMae.Should().BeApproximately(1.0, 1e-9);   // (1 + 1 + 1) / 3
    }

    [Fact]
    public void ComputeRollingMae_window_slides_so_old_pairs_drop_off_after_windowDays()
    {
        // Five paired days, 3-day rolling window: the oldest pair leaves the
        // window once we get to day 4, so day-4's MAE is computed over days
        // 2..4 (not days 1..4).
        var t1 = new DateTime(2026, 4, 1, 12, 0, 0, DateTimeKind.Utc);
        var preds = Enumerable.Range(0, 5)
            .Select(i => Pred("v1", 24, t1.AddDays(i), 10.0 + i))
            .ToArray();
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, p => p.BlendTemperature - 1.0);

        var points = RenderSiteCommand.ComputeRollingMae(preds, truth, windowDays: 3);

        // 5 points, one per day. N caps at 3 from day 3 onwards.
        points.Should().HaveCount(5);
        points.Select(p => p.N).Should().Equal(1, 2, 3, 3, 3);
        // Each pair has |pred - truth| = 1, so MAE = 1 across every window.
        points.Select(p => p.BlendMae).Should().AllBeEquivalentTo(1.0);
    }

    [Fact]
    public void ComputeRollingMae_separates_per_version_and_per_lead()
    {
        // Two versions × two leads → four series of points, no cross-talk.
        var t = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Pred("v1", 24, t, 10.0),
            Pred("v1", 48, t, 11.0),
            Pred("v2", 24, t, 12.0),
            Pred("v2", 48, t, 13.0),
        };
        var truth = new Dictionary<DateTime, double> { [t] = 10.0 };

        var points = RenderSiteCommand.ComputeRollingMae(preds, truth, 14);
        points.Should().HaveCount(4);
        points.Select(p => (p.ModelVersion, p.LeadHours)).Should().BeEquivalentTo(
            new (string, int)[] { ("v1", 24), ("v1", 48), ("v2", 24), ("v2", 48) });
    }

    // ---------- helpers ----------

    private static DuckDBConnection OpenConn()
    {
        var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        return conn;
    }

    private async Task WriteParquetAsync<T>(string filename, IEnumerable<T> rows) where T : class
    {
        var path = Path.Combine(_root, filename);
        await ParquetSerializer.SerializeAsync(rows.ToList(), path);
    }

    // Tiny POCOs purely for fixture-writing — keep them flat so the parquet
    // serializer's reflection-based mapping doesn't drag in unrelated types.
    public sealed class SimpleRow
    {
        public string Name { get; set; } = "";
        public double UtciC { get; set; }
    }

    public sealed class ExtendedRow
    {
        public string Name { get; set; } = "";
        public double UtciC { get; set; }
        public double ApparentTemperatureC { get; set; }
    }
}
