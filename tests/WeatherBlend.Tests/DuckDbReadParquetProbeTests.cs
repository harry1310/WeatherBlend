using DuckDB.NET.Data;
using FluentAssertions;
using Parquet.Serialization;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests;

/// <summary>
/// Investigates exactly what DuckDB's <c>read_parquet</c> throws under the
/// "missing tree" scenarios commands rely on. Pins the strings the
/// <see cref="WeatherBlend.Storage.ParquetReader"/> catch predicate matches
/// against, so a future DuckDB.NET upgrade can't silently change the
/// behaviour without a test failure pointing right at it.
///
/// Particularly interested in the <c>[list of globs]</c> form, where one
/// historical command catches both "No files found" AND "IO Error" — which
/// suggests the list-form throws differently. This test class proves whether
/// the extra catch is necessary.
/// </summary>
public class DuckDbReadParquetProbeTests : IDisposable
{
    private readonly string _root;
    private readonly ITestOutputHelper _out;

    public DuckDbReadParquetProbeTests(ITestOutputHelper output)
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-duckdb-probe-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        _out = output;
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    public sealed class Row { public string Name { get; set; } = ""; public int N { get; set; } }

    private string GlobUnder(string subdir)
        => Path.Combine(_root, subdir, "**", "*.parquet").Replace('\\', '/');

    private async Task SeedAsync(string subdir, IReadOnlyList<Row> rows)
    {
        var dir = Path.Combine(_root, subdir);
        Directory.CreateDirectory(dir);
        await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "data.parquet"));
    }

    [Fact]
    public void Single_glob_missing_tree_throws_NoFilesFound()
    {
        var glob = GlobUnder("does-not-exist");
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $"SELECT * FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)";

        var ex = Assert.Throws<DuckDBException>(() => { using var r = cmd.ExecuteReader(); r.Read(); });
        _out.WriteLine($"single-form missing tree → {ex.Message}");
        ex.Message.Should().Contain("No files found");
    }

    [Fact]
    public void List_form_all_missing_throws_NoFilesFound()
    {
        var g1 = GlobUnder("missing-1");
        var g2 = GlobUnder("missing-2");
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT * FROM read_parquet(['{g1}', '{g2}'], hive_partitioning = false, union_by_name = true)";

        var ex = Assert.Throws<DuckDBException>(() => { using var r = cmd.ExecuteReader(); r.Read(); });
        _out.WriteLine($"list-form all missing → {ex.Message}");
        // This is what we want to know: does the list-form genuinely throw
        // something *other* than "No files found"? If this assert holds, the
        // extra "IO Error" catch in DryWindowVerifyCommand is unnecessary.
        ex.Message.Should().Contain("No files found");
    }

    [Fact]
    public async Task List_form_one_present_one_missing_throws_NoFilesFound()
    {
        // Real-world scenario: verifying multiple stations where some have
        // predictions and some don't. DuckDB does NOT silently union what's
        // available — it fails fast with "IO Error: No files found that
        // match the pattern <missing>". Surprising but useful to pin: any
        // command using the list-form swallows missing-tree errors at its
        // peril, since one missing station blanks out everything.
        await SeedAsync("present", new[] { new Row { Name = "a", N = 1 } });

        var present = GlobUnder("present");
        var missing = GlobUnder("missing");

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT Name, N FROM read_parquet(['{present}', '{missing}'], hive_partitioning = false, union_by_name = true)";

        var ex = Assert.Throws<DuckDBException>(() => { using var r = cmd.ExecuteReader(); r.Read(); });
        _out.WriteLine($"list-form mixed → {ex.Message}");
        ex.Message.Should().Contain("No files found");
    }
}
