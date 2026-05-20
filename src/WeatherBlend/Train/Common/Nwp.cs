namespace WeatherBlend.Train.Common;

/// <summary>
/// Single source of truth for the canonical NWP-id set the blenders consume
/// and the site renders as overlay lines on the rain + temp skill / forecast
/// charts. The full set of Open-Meteo models that are <em>collected</em> is
/// declared in <c>config.yaml</c>'s top-level <c>models:</c> list (~10 ids
/// including a couple of bake-off candidates not in the blender); this set
/// is the narrower 8 ids the blenders actually score against and the chart
/// legend shows.
///
/// Used by:
///   * <c>RenderSiteCommand</c>'s three NWP-overlay SQL queries (PoP / rate /
///     temperature) — <see cref="SqlInList"/>.
///   * <c>SitePages.Forecasts</c>'s chart-legend label lookup —
///     <see cref="DisplayLabel"/>.
///
/// Adding a 9th model is a one-line edit here plus updating the
/// <c>config.yaml</c> models list (for collection) and the blender feature
/// schemas (for training); the SQL filter + render labels then pick it up
/// automatically across both render and skill pages. Pre-2026-05-20 each of
/// these three SQL queries hand-pasted the 8 ids in a 2-line string literal
/// — adding AIFS + JMA required edits in 3+1 places that drifted twice
/// during the relevant bake-offs.
/// </summary>
public static class Nwp
{
    public static readonly string[] BlenderModelIds = new[]
    {
        "gfs_seamless",
        "ecmwf_ifs025",
        "icon_seamless",
        "meteofrance_seamless",
        "ukmo_seamless",
        "gem_seamless",
        "ecmwf_aifs025_single",
        "jma_seamless",
    };

    /// <summary>
    /// Short display label for chart legends and table headers (e.g.
    /// "GFS", "ECMWF", "AIFS"). Returns <paramref name="modelId"/>
    /// unchanged for unknown ids so a stray collected-but-unmapped model
    /// surfaces visibly rather than silently dropping out of the legend.
    /// Only ~4 of the 8 publish PoP via Open-Meteo (GFS / ECMWF / ICON /
    /// GEM); the others render zero rows for that chart but a non-empty
    /// label here keeps the lookup consistent across charts.
    /// </summary>
    public static string DisplayLabel(string modelId) => modelId switch
    {
        "gfs_seamless"          => "GFS",
        "ecmwf_ifs025"          => "ECMWF",
        "icon_seamless"         => "ICON",
        "meteofrance_seamless"  => "MF",
        "ukmo_seamless"         => "UKMO",
        "gem_seamless"          => "GEM",
        "ecmwf_aifs025_single"  => "AIFS",
        "jma_seamless"          => "JMA",
        _ => modelId,
    };

    /// <summary>
    /// SQL <c>IN (...)</c> clause body — parenthesised, comma-separated,
    /// single-quoted model ids built from <see cref="BlenderModelIds"/>.
    /// Use directly after <c>Model IN </c> in a WHERE clause. The model
    /// ids are config-controlled so no SQL injection surface here.
    /// </summary>
    public static string SqlInList()
        => "(" + string.Join(",", BlenderModelIds.Select(id => $"'{id}'")) + ")";

    /// <summary>
    /// Lowercase short suffix used in feature column names
    /// (<c>temp_gfs</c>, <c>rh_ecmwf</c>, <c>cc_aifs</c>, ...). Distinct from
    /// <see cref="DisplayLabel"/> which returns the uppercase chart-legend
    /// form. Throws on unknown ids so a typo in a feature builder fails the
    /// build instead of silently producing a malformed column name.
    /// </summary>
    public static string ColumnSuffix(string modelId) => modelId switch
    {
        "gfs_seamless"          => "gfs",
        "ecmwf_ifs025"          => "ecmwf",
        "icon_seamless"         => "icon",
        "meteofrance_seamless"  => "mf",
        "ukmo_seamless"         => "ukmo",
        "gem_seamless"          => "gem",
        "ecmwf_aifs025_single"  => "aifs",
        "jma_seamless"          => "jma",
        _ => throw new ArgumentException($"Unknown modelId '{modelId}'", nameof(modelId)),
    };
}
