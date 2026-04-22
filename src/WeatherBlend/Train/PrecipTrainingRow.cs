using Microsoft.ML.Data;

namespace WeatherBlend.Train;

/// <summary>
/// One training row for the precipitation occurrence blender.
///
/// Feature set: per-model precipitation, rain, showers, snowfall, precipitation
/// probability, plus a handful of meteorological covariates averaged across the
/// available models (RH, dewpoint depression, cloud cover, CAPE) and cyclical
/// calendar features. Feature ordering is load-bearing — must match
/// <see cref="PrecipFeatureBuilder.OccurrenceFeatureNames"/>.
///
/// Label is <c>WetBinary</c> — 1 if summed hourly Bellever rainfall &gt;= 0.1 mm,
/// 0 otherwise. <c>PrecipMmHour</c> carries the raw hourly total so the same
/// dataset can back the intensity regressor later.
/// </summary>
public sealed class PrecipTrainingRow
{
    public DateTime ValidTimeUtc { get; set; }

    // Per-model precipitation forecasts (mm for the hour).
    public float PrecipGfs { get; set; }
    public float PrecipEcmwf { get; set; }
    public float PrecipIcon { get; set; }
    public float PrecipMf { get; set; }
    public float PrecipUkmo { get; set; }
    public float PrecipGem { get; set; }

    // Per-model declared probability (if the model exposes it; NaN-handling via LightGBM defaults).
    public float ProbGfs { get; set; }
    public float ProbEcmwf { get; set; }
    public float ProbIcon { get; set; }
    public float ProbMf { get; set; }
    public float ProbUkmo { get; set; }
    public float ProbGem { get; set; }

    // Ensemble-derived spread features.
    public float PrecipMean { get; set; }
    public float PrecipStd { get; set; }
    public float PrecipMax { get; set; }
    public float PrecipAgreementWet01 { get; set; }   // count of models predicting >=0.1mm / 6

    // Mean meteorological covariates (across models; LightGBM tolerates NaN).
    public float RhMean { get; set; }
    public float DewDepressionMean { get; set; }      // Temperature2m - DewPoint2m (smaller = closer to saturation)
    public float CloudLowMean { get; set; }
    public float CloudMidMean { get; set; }
    public float CloudHighMean { get; set; }
    public float CapeMean { get; set; }
    public float WindSpeedMean { get; set; }

    // Cyclical calendar features.
    public float HourSin { get; set; }
    public float HourCos { get; set; }
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    // Labels.
    [ColumnName("Label")]
    public bool WetBinary { get; set; }

    /// <summary>Hourly Bellever total in mm — kept for intensity regressor / diagnostics.</summary>
    public float PrecipMmHour { get; set; }
}
