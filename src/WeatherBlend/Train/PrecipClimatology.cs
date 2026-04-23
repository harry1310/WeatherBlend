using System.Text.Json;

namespace WeatherBlend.Train;

/// <summary>
/// Base-rate climatology for P(wet) indexed by (month, hour-of-day). Built from
/// training rows alongside the blender so predict/verify don't have to re-derive
/// it from a live dataset at inference time. Unseen buckets fall back to the
/// global training wet rate.
/// </summary>
public sealed class PrecipClimatology
{
    /// <summary>P(wet) per (1..12 month, 0..23 hour). Buckets absent from training are not present; caller must fall back.</summary>
    public Dictionary<string, double> PWetByBucket { get; set; } = new();

    /// <summary>Mean wet rate across all training rows — fallback for unseen (month, hour) buckets.</summary>
    public double GlobalWetRate { get; set; }

    /// <summary>Number of training rows the climatology was built from. Informational — lets a verifier sanity-check provenance.</summary>
    public int SourceRowCount { get; set; }

    /// <summary>Wet threshold (mm/hour) applied when counting wet rows. Stored so a verifier catches silent threshold changes.</summary>
    public double WetThresholdMm { get; set; } = PrecipFeatureBuilder.WetThresholdMm;

    public static PrecipClimatology BuildFromTraining(IReadOnlyList<PrecipTrainingRow> trainRows)
    {
        var sums = new Dictionary<(int Month, int Hour), (int Wet, int N)>();
        foreach (var r in trainRows)
        {
            var k = (r.ValidTimeUtc.Month, r.ValidTimeUtc.Hour);
            sums.TryGetValue(k, out var cur);
            sums[k] = (cur.Wet + (r.WetBinary ? 1 : 0), cur.N + 1);
        }

        var pWet = sums.ToDictionary(
            kv => BucketKey(kv.Key.Month, kv.Key.Hour),
            kv => (double)kv.Value.Wet / kv.Value.N);

        var globalMean = trainRows.Count == 0
            ? double.NaN
            : (double)trainRows.Count(r => r.WetBinary) / trainRows.Count;

        return new PrecipClimatology
        {
            PWetByBucket = pWet,
            GlobalWetRate = globalMean,
            SourceRowCount = trainRows.Count,
        };
    }

    /// <summary>Probability of wet at <paramref name="validTimeUtc"/>. Falls back to the global rate on unseen buckets.</summary>
    public double Predict(DateTime validTimeUtc)
    {
        var key = BucketKey(validTimeUtc.Month, validTimeUtc.Hour);
        return PWetByBucket.TryGetValue(key, out var p) ? p : GlobalWetRate;
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    public void SaveTo(string path)
        => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOpts));

    public static PrecipClimatology LoadFrom(string path)
        => JsonSerializer.Deserialize<PrecipClimatology>(File.ReadAllText(path))
           ?? throw new InvalidOperationException($"Could not parse climatology at {path}");

    /// <summary>Stable "MM-HH" key so JSON stays human-readable (tuples aren't valid JSON dict keys).</summary>
    private static string BucketKey(int month, int hour) => $"{month:D2}-{hour:D2}";
}
