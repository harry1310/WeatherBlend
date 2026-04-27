using Microsoft.ML.Data;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Generic regression-blender row. <see cref="Features"/> is the only column
/// LightGBM trains on; its length is determined per-blender by
/// <see cref="BlenderSpec.FeatureCount"/>. Diagnostic fields (e.g. WindDirMean
/// for temperature) ride along but aren't fed to the model.
/// </summary>
public sealed class RegressionTrainingRow
{
    public DateTime ValidTimeUtc { get; init; }

    [VectorType] public float[] Features { get; init; } = Array.Empty<float>();

    [ColumnName("Label")] public float Label { get; init; }

    /// <summary>Optional per-row diagnostic — temperature uses it for wind-direction stratification.</summary>
    public float WindDirMean { get; init; } = float.NaN;
}
