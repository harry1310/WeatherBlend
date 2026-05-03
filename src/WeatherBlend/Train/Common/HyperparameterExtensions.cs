using System.Text.Json;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Extension methods for reading typed values out of
/// <c>ModelArtifact.TrainingMetadata.Hyperparameters</c> (a
/// <c>Dictionary&lt;string, object&gt;</c>). Round-tripping the dictionary
/// through System.Text.Json deserialises every value as
/// <see cref="JsonElement"/>, so a naive <c>(int)dict[key]</c> throws
/// <see cref="InvalidCastException"/> at predict time — these helpers
/// unwrap the JsonElement explicitly while still accepting freshly-set
/// raw <c>int</c>/<c>string</c> values from in-memory metadata.
/// Both return null on missing keys / wrong types so callers can
/// pattern-match against a default with the null-coalescing operator.
/// </summary>
public static class HyperparameterExtensions
{
    public static string? HpString(this IReadOnlyDictionary<string, object>? hp, string key)
    {
        if (hp is null || !hp.TryGetValue(key, out var v) || v is null) return null;
        return v switch
        {
            string s => s,
            JsonElement je when je.ValueKind == JsonValueKind.String => je.GetString(),
            _ => v.ToString(),
        };
    }

    public static int? HpInt(this IReadOnlyDictionary<string, object>? hp, string key)
    {
        if (hp is null || !hp.TryGetValue(key, out var v) || v is null) return null;
        return v switch
        {
            int i => i,
            long l => (int)l,
            JsonElement je when je.ValueKind == JsonValueKind.Number => je.GetInt32(),
            _ => int.TryParse(v.ToString(), out var x) ? x : null,
        };
    }
}
