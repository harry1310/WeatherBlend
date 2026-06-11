using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;

namespace WeatherBlend.Storage;

/// <summary>
/// Thin wrapper around the DuckDB-over-parquet pattern that every command in
/// this codebase repeats. Encapsulates two things and nothing more:
///
///  1. The "No files found" → empty-result convention. Commands that read trees
///     that may not yet exist (a fresh checkout, a new station, a deleted
///     subtree) need to degrade gracefully and log a warning instead of
///     crashing. <em>Other</em> <see cref="DuckDBException"/>s (SQL syntax,
///     schema mismatch, etc.) are real bugs and bubble up unchanged.
///
///  2. The open / read / dispose / map boilerplate. Each query method used to
///     own ~25 lines of plumbing on top of its actual SELECT projection and
///     row-mapping lambda. This helper keeps the SQL inline at the call site
///     (per-query idiosyncrasies — TIMESTAMP filters, HAVING aggregations,
///     PIVOT, hive flags — stay where they belong) and just lifts the wrapper.
///
/// Does NOT attempt to be a SQL builder, parameteriser, or ORM. DuckDB.NET
/// parameter binding has been flaky historically, so callers continue to
/// build SQL via <c>$@""</c> + <c>.Replace("'", "''")</c> for any embedded
/// strings — same as before, just less duplicated.
/// </summary>
public static class ParquetReader
{
    /// <summary>
    /// Normalises a filesystem path to a string ready to embed inside
    /// <c>read_parquet('...')</c>. Forward-slashes the separators (DuckDB
    /// treats backslashes as escape characters inside quoted strings) and
    /// doubles up embedded single quotes. Delegates to the shared
    /// <see cref="Train.Common.SqlGlob"/> so the two transforms are defined
    /// exactly once.
    /// </summary>
    public static string Glob(string path)
        => Train.Common.SqlGlob.Escape(path);

    /// <summary>
    /// Opens a fresh in-memory connection, executes <paramref name="sql"/>,
    /// maps every row through <paramref name="map"/>, and returns the list.
    /// Cancellation is checked between rows.
    ///
    /// On <c>"No files found"</c> the warning is logged and an empty list
    /// returned — the established graceful-degrade signal for missing trees.
    /// All other <see cref="DuckDBException"/>s propagate.
    /// </summary>
    public static IReadOnlyList<T> Query<T>(
        string sql,
        Func<IDataReader, T> map,
        ILogger log,
        string emptyTreeWarning,
        CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        return Query(conn, sql, map, log, emptyTreeWarning, ct);
    }

    /// <summary>
    /// Compose-with-existing-conn variant. Use when a schema probe such as
    /// <see cref="HasColumn"/> has already opened a connection that should
    /// be reused for the main query (avoids tearing down + reopening
    /// DuckDB's in-memory state for back-to-back operations).
    /// </summary>
    public static IReadOnlyList<T> Query<T>(
        DuckDBConnection conn,
        string sql,
        Func<IDataReader, T> map,
        ILogger log,
        string emptyTreeWarning,
        CancellationToken ct)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var rows = new List<T>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add(map(r));
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            log.LogWarning("{Warning}", emptyTreeWarning);
            return Array.Empty<T>();
        }
        return rows;
    }

    /// <summary>
    /// Cheap schema probe over a parquet glob. Returns <c>false</c> when no
    /// files match (<c>read_parquet</c> throws "No files found") or when the
    /// column is absent from the <c>union_by_name</c>'d schema. Used to gate
    /// queries that reference newly-added columns during the migration
    /// window between schema bumps.
    /// </summary>
    public static bool HasColumn(DuckDBConnection conn, string glob, string column)
    {
        var sql = $@"
SELECT 1
FROM (DESCRIBE SELECT * FROM read_parquet('{glob}',
                                          hive_partitioning = false,
                                          union_by_name = true))
WHERE column_name = '{column.Replace("'", "''")}'
LIMIT 1";
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        try
        {
            using var r = cmd.ExecuteReader();
            return r.Read();
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            return false;
        }
    }
}
