using Microsoft.Extensions.Logging;
using WeatherBlend.Config;

namespace WeatherBlend.Predict;

/// <summary>
/// Resolves the optional <c>--location</c> predict override into a concrete
/// <see cref="LocationConfig"/>.
///
/// The same ~13-line block — null-check, case-insensitive
/// <c>Locations.FirstOrDefault</c>, "not found" error log + exit 2, and the
/// override info log — was pasted into every predict command that honours
/// <c>--location</c> (the Phase B retrofit landed it in 5 places). This lifts
/// it so the validation + error message stay identical across targets.
///
/// Not used by <c>TempPredictCommand</c>: its <c>--location</c> is a
/// station-key selector that may legitimately name a location without a
/// config entry yet, validated per-station with warn-and-skip rather than
/// error-and-exit — genuinely different semantics, deliberately left alone.
/// </summary>
public static class PredictLocationResolver
{
    /// <summary>
    /// Resolves <paramref name="locationOverride"/> against <paramref name="cfg"/>:
    /// <list type="bullet">
    ///   <item>null/blank → <c>(cfg.Location, 0)</c> — the primary location.</item>
    ///   <item>names a configured location → <c>(match, 0)</c>, after an info log.</item>
    ///   <item>names nothing → <c>(null, 2)</c>, after the standard error log.</item>
    /// </list>
    /// Callers assign <c>Location</c> to their active-location field on success
    /// and <c>return ExitCode</c> when <c>Location</c> is null.
    /// </summary>
    public static (LocationConfig? Location, int ExitCode) Resolve(
        AppConfig cfg, string? locationOverride, ILogger log)
    {
        if (string.IsNullOrWhiteSpace(locationOverride))
            return (cfg.Location, 0);

        var match = cfg.Locations.FirstOrDefault(l =>
            l.Name.Equals(locationOverride, StringComparison.OrdinalIgnoreCase));
        if (match is null)
        {
            log.LogError("Location '{Name}' not found in config.yaml's `locations:` list. Available: [{All}]",
                locationOverride, string.Join(", ", cfg.Locations.Select(l => l.Name)));
            return (null, 2);
        }

        log.LogInformation("Predict location override → '{Loc}'.", match.Name);
        return (match, 0);
    }
}
