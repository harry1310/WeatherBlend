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
        => await WriteAsync(reportsDir, file, locationName: null, ct);

    /// <summary>
    /// Per-location overload. Phase C commit 4 (2026-05-20): non-primary
    /// verify runs pass <paramref name="locationName"/> so the JSON sidecar
    /// gets a <c>{location}</c> segment in the filename and doesn't
    /// overwrite the primary's. Pass <c>null</c> (or use the back-compat
    /// 3-arg overload) for primary-location runs to keep the legacy
    /// filename pattern unchanged on R2.
    /// </summary>
    public static async Task WriteAsync(string reportsDir, VerifyHistoryFile file, string? locationName, CancellationToken ct)
    {
        Directory.CreateDirectory(reportsDir);
        var name = FileNameFor(file.Target, locationName, file.AsOfUtc);
        var path = Path.Combine(reportsDir, name);
        var json = JsonSerializer.Serialize(file, JsonOptions);
        await File.WriteAllTextAsync(path, json, ct);
    }

    /// <summary>
    /// File naming convention. Primary location:
    /// <c>verify_{target}_{yyyy-MM-dd}.json</c> — unchanged for back-compat
    /// with the existing R2 corpus. Non-primary:
    /// <c>verify_{target}_{location}_{yyyy-MM-dd}.json</c> (matches the
    /// markdown sibling the verify commands already write with the same
    /// per-loc disambiguator). Element targets pass the full element-prefixed
    /// identifier (<c>element_wind</c>, <c>element_humidity</c>, …) as
    /// <paramref name="target"/> so the filename matches the markdown
    /// sibling exactly with only the extension swapped.
    /// </summary>
    public static string FileNameFor(string target, string? locationName, DateTime asOfUtc)
        => string.IsNullOrEmpty(locationName)
            ? $"verify_{target}_{asOfUtc:yyyy-MM-dd}.json"
            : $"verify_{target}_{locationName}_{asOfUtc:yyyy-MM-dd}.json";

    /// <summary>Back-compat 2-arg overload — primary-location filename.</summary>
    public static string FileNameFor(string target, DateTime asOfUtc)
        => FileNameFor(target, locationName: null, asOfUtc);
}
