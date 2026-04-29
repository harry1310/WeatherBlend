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
