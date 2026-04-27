using Microsoft.ML.Data;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Generic dry-window-blender row. Day-anchored (TargetDateUtc, not ValidTimeUtc)
/// because the label is "is there at least one N-hour dry block in the target UTC
/// day". <see cref="WindowHours"/> identifies which N (3/4/6) — different blenders
/// per N, but they share this row shape.
/// </summary>
public sealed class DryWindowTrainingRow
{
    public DateTime TargetDateUtc { get; init; }

    public int WindowHours { get; init; }

    [VectorType] public float[] Features { get; init; } = Array.Empty<float>();

    [ColumnName("Label")] public bool Label { get; init; }

    /// <summary>Daily total mm — diagnostic for reports.</summary>
    public float PrecipMmDay { get; init; }
}
