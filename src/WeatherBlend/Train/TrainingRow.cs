using Microsoft.ML.Data;

namespace WeatherBlend.Train;

/// <summary>
/// One training row: 13 features + label (ERA5 temperature at valid time).
/// Feature ordering is load-bearing — matches <see cref="FeatureBuilder.FeatureNames"/>
/// and is persisted in feature_schema.json so inference uses the same order.
/// </summary>
public sealed class TrainingRow
{
    public DateTime ValidTimeUtc { get; set; }

    // Per-model temperature forecasts (°C).
    public float TempGfs { get; set; }
    public float TempEcmwf { get; set; }
    public float TempIcon { get; set; }
    public float TempMf { get; set; }
    public float TempUkmo { get; set; }
    public float TempGem { get; set; }

    // Inter-model spread.
    public float TempMean { get; set; }
    public float TempStd { get; set; }
    public float TempRange { get; set; }

    // Cyclical calendar features.
    public float HourSin { get; set; }
    public float HourCos { get; set; }
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    // Mean ensemble wind direction (degrees) — not a training feature; kept for
    // stratified evaluation (wind-quadrant slicing in the report).
    public float WindDirMean { get; set; }

    [ColumnName("Label")]
    public float Era5Temp { get; set; }
}
