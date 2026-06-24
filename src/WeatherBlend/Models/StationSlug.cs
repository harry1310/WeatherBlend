using System.Text;

namespace WeatherBlend.Models;

/// <summary>
/// EA / METAR station name → URL/path-safe slug. Single source of truth
/// for the rule that <c>"Bellever Dartmoor"</c> becomes <c>"bellever_dartmoor"</c>
/// — used by:
///
///   * the predict + verify pipelines when composing per-station parquet
///     paths (<c>data/predictions/precipitation/ea_bellever_dartmoor/...</c>),
///   * the CLI argument resolver in the precip / dry-window / start-hour
///     verify commands (<c>--truth-station bellever</c> → look up
///     <c>ea_bellever_dartmoor</c>),
///   * <see cref="Storage.TruthRepository.GetHourlyRainfallByStation"/>
///     for the slug → <c>StationName</c> reverse map.
///
/// Pre-2026-05-02 this lived as nine private copies across as many command
/// classes — three of them with subtly different rules (one only collapsed
/// space/hyphen/comma, the others handled all non-alphanumeric chars). The
/// canonical rule below matches the StringBuilder-based implementations
/// (<c>StartHourVerifyCommand</c>, <c>PrecipReplayCommand</c>, etc.) and is
/// equivalent to the LINQ-based ones on the current station name set
/// (Bellever Dartmoor / Bovey Tracey / Dartmoor nr Hexworthy) — simpler chars,
/// same output.
///
/// The <c>"ea_"</c> prefix is NOT added by <see cref="Of"/> — callers add it
/// where they need to ("future Met Office Bellever can live alongside EA
/// without colliding"). See <see cref="WithEaPrefix"/> for the common case.
/// </summary>
public static class StationSlug
{
    /// <summary>
    /// Lowercase the input, replace every non-alphanumeric character with an
    /// underscore, collapse adjacent underscores, and trim leading/trailing
    /// underscores. Pure function — no I/O, no allocation beyond the
    /// returned string and one StringBuilder.
    /// </summary>
    public static string Of(string input)
    {
        if (string.IsNullOrEmpty(input)) return "";
        var sb = new StringBuilder(input.Length);
        var lastUnderscore = false;
        foreach (var c in input.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(c);
                lastUnderscore = false;
            }
            else if (!lastUnderscore)
            {
                sb.Append('_');
                lastUnderscore = true;
            }
        }
        return sb.ToString().Trim('_');
    }

    /// <summary>
    /// Convenience: <c>"ea_" + Of(name)</c>. The EA prefix lets a future
    /// Met Office or other-provider station with the same name coexist
    /// without colliding in the parquet path tree.
    /// </summary>
    public static string WithEaPrefix(string name) => "ea_" + Of(name);

    /// <summary>
    /// Convenience: <c>"wl_" + Of(weatherLinkLocation)</c> for a Davis
    /// WeatherLink-sourced precip-truth station (e.g. the Lands End cove gauge
    /// for Sennen → <c>wl_lands_end</c>). The <c>wl_</c> prefix keeps it distinct
    /// from any EA gauge of the same name in the per-station parquet tree.
    /// </summary>
    public static string WithWlPrefix(string weatherLinkLocation) => "wl_" + Of(weatherLinkLocation);
}
