using System.Text.Json;
using WeatherBlend.Models;

namespace WeatherBlend.Evaluate;

/// <summary>
/// Serialises a <see cref="VerifyHistoryFile"/> to JSON next to the
/// existing markdown report. Single helper so all four verify commands
/// (temp / precip / dry-window / element) emit the same shape — the
/// Models-page renderer reads back via the same deserialiser and one
/// schema change updates every producer in lock-step.
/// </summary>
public static class VerifyHistoryWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        // Non-finite metrics (e.g. BSS = (clim - brier) / clim where clim=0
        // for an all-dry station-window) come through as ±Infinity. Default
        // System.Text.Json throws on them — caught the dry-window verify
        // 2026-05-05. Emit as "Infinity"/"-Infinity"/"NaN" string literals
        // so the report still writes; downstream readers handle them as
        // tombstones.
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static async Task WriteAsync(string reportsDir, VerifyHistoryFile file, CancellationToken ct)
    {
        Directory.CreateDirectory(reportsDir);
        var name = FileNameFor(file.Target, file.AsOfUtc);
        var path = Path.Combine(reportsDir, name);
        var json = JsonSerializer.Serialize(file, JsonOptions);
        await File.WriteAllTextAsync(path, json, ct);
    }

    /// <summary>
    /// File naming convention: <c>verify_{target}_{yyyy-MM-dd}.json</c>.
    /// Element targets pass the full element-prefixed identifier
    /// (<c>element_wind</c>, <c>element_humidity</c>, …) so the filename
    /// matches the markdown sibling exactly with only the extension swapped.
    /// </summary>
    public static string FileNameFor(string target, DateTime asOfUtc)
        => $"verify_{target}_{asOfUtc:yyyy-MM-dd}.json";
}
