namespace WeatherBlend.Train.Common;

/// <summary>
/// Escapes a filesystem path / glob for inlining into a DuckDB SQL string
/// literal. Two transforms, both required:
///   * <c>'\' → '/'</c> — DuckDB's <c>read_parquet</c> glob matching wants
///     forward slashes even on Windows;
///   * <c>"'" → "''"</c> — standard SQL string-literal quote doubling, so a
///     path containing an apostrophe can't truncate (or inject into) the
///     query it is embedded in.
///
/// Single definition (2026-06-10) replacing the private NormaliseGlob /
/// Norm copies that had spread across the feature builders — two of which
/// (PrecipExactFeatureBuilder, Exact12hFeatureBuilder) had silently
/// dropped the quote-escaping half. <see cref="Storage.ParquetReader.Glob"/>
/// delegates here too.
/// </summary>
public static class SqlGlob
{
    public static string Escape(string path)
        => path.Replace('\\', '/').Replace("'", "''");
}
