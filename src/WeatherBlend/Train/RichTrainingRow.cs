namespace WeatherBlend.Train;

/// <summary>
/// Phase 2c training row: inherits the 13 lean features (and Label) from
/// <see cref="TrainingRow"/> and adds 75 secondary-variable features:
///
///   • 11 per-model secondaries × 6 models = 66
///       dew, relative humidity, total cloud, cloud (low/mid/high),
///       wind speed, wind direction (sin + cos), wind gusts, surface pressure
///   • 9 ensemble aggregates
///       dew (mean/std), rh (mean/std), cloud mean,
///       wind speed (mean/std), pressure (mean/std)
///
/// 13 + 66 + 9 = 88 features. Persistence features (ERA5 at t−24h / t−168h) were
/// considered and dropped: ERA5 has a ~5d release lag, so for any non-trivial lead
/// the lookback would not exist at predict time.
///
/// Missing per-model secondaries are stored as <c>float.NaN</c> — LightGBM handles
/// these natively. Aggregates are computed across the non-NaN subset; if every model
/// is missing a variable, the aggregate is also NaN.
/// </summary>
public sealed class RichTrainingRow : TrainingRow
{
    // Per-model dew point (°C).
    public float DewGfs { get; set; }
    public float DewEcmwf { get; set; }
    public float DewIcon { get; set; }
    public float DewMf { get; set; }
    public float DewUkmo { get; set; }
    public float DewGem { get; set; }

    // Per-model relative humidity (%).
    public float RhGfs { get; set; }
    public float RhEcmwf { get; set; }
    public float RhIcon { get; set; }
    public float RhMf { get; set; }
    public float RhUkmo { get; set; }
    public float RhGem { get; set; }

    // Per-model total cloud cover (%).
    public float CloudGfs { get; set; }
    public float CloudEcmwf { get; set; }
    public float CloudIcon { get; set; }
    public float CloudMf { get; set; }
    public float CloudUkmo { get; set; }
    public float CloudGem { get; set; }

    // Per-model low cloud cover (%).
    public float CloudLowGfs { get; set; }
    public float CloudLowEcmwf { get; set; }
    public float CloudLowIcon { get; set; }
    public float CloudLowMf { get; set; }
    public float CloudLowUkmo { get; set; }
    public float CloudLowGem { get; set; }

    // Per-model mid cloud cover (%).
    public float CloudMidGfs { get; set; }
    public float CloudMidEcmwf { get; set; }
    public float CloudMidIcon { get; set; }
    public float CloudMidMf { get; set; }
    public float CloudMidUkmo { get; set; }
    public float CloudMidGem { get; set; }

    // Per-model high cloud cover (%).
    public float CloudHighGfs { get; set; }
    public float CloudHighEcmwf { get; set; }
    public float CloudHighIcon { get; set; }
    public float CloudHighMf { get; set; }
    public float CloudHighUkmo { get; set; }
    public float CloudHighGem { get; set; }

    // Per-model 10m wind speed (m/s).
    public float WindSpeedGfs { get; set; }
    public float WindSpeedEcmwf { get; set; }
    public float WindSpeedIcon { get; set; }
    public float WindSpeedMf { get; set; }
    public float WindSpeedUkmo { get; set; }
    public float WindSpeedGem { get; set; }

    // Per-model 10m wind direction encoded as sin (cyclical).
    public float WindDirSinGfs { get; set; }
    public float WindDirSinEcmwf { get; set; }
    public float WindDirSinIcon { get; set; }
    public float WindDirSinMf { get; set; }
    public float WindDirSinUkmo { get; set; }
    public float WindDirSinGem { get; set; }

    // Per-model 10m wind direction encoded as cos (cyclical).
    public float WindDirCosGfs { get; set; }
    public float WindDirCosEcmwf { get; set; }
    public float WindDirCosIcon { get; set; }
    public float WindDirCosMf { get; set; }
    public float WindDirCosUkmo { get; set; }
    public float WindDirCosGem { get; set; }

    // Per-model 10m wind gusts (m/s).
    public float WindGustsGfs { get; set; }
    public float WindGustsEcmwf { get; set; }
    public float WindGustsIcon { get; set; }
    public float WindGustsMf { get; set; }
    public float WindGustsUkmo { get; set; }
    public float WindGustsGem { get; set; }

    // Per-model surface pressure (hPa).
    public float PressureGfs { get; set; }
    public float PressureEcmwf { get; set; }
    public float PressureIcon { get; set; }
    public float PressureMf { get; set; }
    public float PressureUkmo { get; set; }
    public float PressureGem { get; set; }

    // Cross-model aggregates over the non-NaN subset.
    public float DewMean { get; set; }
    public float DewStd { get; set; }
    public float RhMean { get; set; }
    public float RhStd { get; set; }
    public float CloudMean { get; set; }
    public float WindSpeedMean { get; set; }
    public float WindSpeedStd { get; set; }
    public float PressureMean { get; set; }
    public float PressureStd { get; set; }
}
