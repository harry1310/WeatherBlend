using Microsoft.ML.Data;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Generic binary-classification blender row. Used by precipitation occurrence.
/// <see cref="TruthMmHour"/> is a diagnostic carried alongside the boolean label
/// so reports can compute conditional intensity stats without re-joining truth.
/// </summary>
public sealed class BinaryTrainingRow
{
    public DateTime ValidTimeUtc { get; init; }

    [VectorType] public float[] Features { get; init; } = Array.Empty<float>();

    [ColumnName("Label")] public bool Label { get; init; }

    public float TruthMmHour { get; init; }
}
