using DuckDB.NET.Data;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Parquet.Serialization;
using WeatherBlend.Storage;
using Xunit;

namespace WeatherBlend.Tests;

public class ParquetReaderTests : IDisposable
{
    private readonly string _root;
    private readonly ILogger _log = NullLogger.Instance;

    public ParquetReaderTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-parquet-reader-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    public sealed class Probe
    {
        public string Name { get; set; } = "";
        public int N { get; set; }
    }

    private async Task SeedAsync(string subdir, IReadOnlyList<Probe> rows)
    {
        var dir = Path.Combine(_root, subdir);
        Directory.CreateDirectory(dir);
        await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "data.parquet"));
    }

    private string GlobUnder(string subdir)
        => ParquetReader.Glob(Path.Combine(_root, subdir, "**", "*.parquet"));

    [Fact]
    public void Glob_forward_slashes_path_separators()
    {
        var raw = @"C:\Projects\Weather\data\forecasts\**\*.parquet";
        ParquetReader.Glob(raw).Should().Be("C:/Projects/Weather/data/forecasts/**/*.parquet");
    }

    [Fact]
    public void Glob_doubles_embedded_single_quotes()
    {
        // O'Hare-style location names need escaping for SQL embedding.
        ParquetReader.Glob(@"C:\data\location=O'Hare\**\*.parquet")
            .Should().Be("C:/data/location=O''Hare/**/*.parquet");
    }

    [Fact]
    public void Query_returns_empty_when_tree_does_not_exist()
    {
        var glob = GlobUnder("never-written");
        var sql = $"SELECT Name, N FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)";

        var result = ParquetReader.Query(
            sql,
            r => new Probe { Name = r.GetString(0), N = r.GetInt32(1) },
            _log,
            "test tree empty",
            CancellationToken.None);

        result.Should().BeEmpty();
    }

    [Fact]
    public async Task Query_maps_rows_round_trip()
    {
        await SeedAsync("alpha", new[]
        {
            new Probe { Name = "a", N = 1 },
            new Probe { Name = "b", N = 2 },
            new Probe { Name = "c", N = 3 },
        });

        var glob = GlobUnder("alpha");
        var sql = $"SELECT Name, N FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true) ORDER BY N";

        var result = ParquetReader.Query(
            sql,
            r => new Probe { Name = r.GetString(0), N = r.GetInt32(1) },
            _log,
            "should not warn",
            CancellationToken.None);

        result.Should().HaveCount(3);
        result.Select(r => r.Name).Should().Equal("a", "b", "c");
        result.Select(r => r.N).Should().Equal(1, 2, 3);
    }

    [Fact]
    public async Task Query_propagates_non_empty_tree_DuckDB_errors()
    {
        // Real bugs (typos in column names, bad syntax) must NOT be swallowed
        // by the missing-tree catch — that would silently render empty pages
        // in production. Pin the boundary.
        await SeedAsync("real-tree", new[] { new Probe { Name = "x", N = 1 } });

        var glob = GlobUnder("real-tree");
        var bogusSql = $"SELECT NoSuchColumn FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)";

        var act = () => ParquetReader.Query(
            bogusSql,
            r => r.GetString(0),
            _log,
            "should not be reached",
            CancellationToken.None);

        act.Should().Throw<DuckDBException>();
    }

    [Fact]
    public async Task Query_with_existing_conn_reuses_connection()
    {
        await SeedAsync("compose", new[]
        {
            new Probe { Name = "x", N = 7 },
            new Probe { Name = "y", N = 8 },
        });

        var glob = GlobUnder("compose");

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        // Probe first, then query — same conn — proves the overload composes.
        ParquetReader.HasColumn(conn, glob, "N").Should().BeTrue();

        var sql = $"SELECT Name, N FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true) ORDER BY N";
        var result = ParquetReader.Query(conn, sql,
            r => (r.GetString(0), r.GetInt32(1)),
            _log, "should not warn", CancellationToken.None);

        result.Should().Equal(("x", 7), ("y", 8));
    }

    [Fact]
    public async Task HasColumn_returns_true_for_present_column()
    {
        await SeedAsync("schema-present", new[] { new Probe { Name = "x", N = 1 } });

        var glob = GlobUnder("schema-present");
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        ParquetReader.HasColumn(conn, glob, "N").Should().BeTrue();
        ParquetReader.HasColumn(conn, glob, "Name").Should().BeTrue();
    }

    [Fact]
    public async Task HasColumn_returns_false_for_missing_column()
    {
        await SeedAsync("schema-missing", new[] { new Probe { Name = "x", N = 1 } });

        var glob = GlobUnder("schema-missing");
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        ParquetReader.HasColumn(conn, glob, "NotThere").Should().BeFalse();
    }

    [Fact]
    public void HasColumn_returns_false_for_missing_tree()
    {
        var glob = GlobUnder("never-existed");
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        ParquetReader.HasColumn(conn, glob, "Name").Should().BeFalse();
    }

    [Fact]
    public async Task Query_observes_cancellation_between_rows()
    {
        await SeedAsync("cancel", Enumerable.Range(0, 50).Select(i => new Probe { Name = $"r{i}", N = i }).ToList());

        var glob = GlobUnder("cancel");
        var sql = $"SELECT Name, N FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)";

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var act = () => ParquetReader.Query(
            sql,
            r => new Probe { Name = r.GetString(0), N = r.GetInt32(1) },
            _log,
            "should not warn",
            cts.Token);

        act.Should().Throw<OperationCanceledException>();
    }
}
