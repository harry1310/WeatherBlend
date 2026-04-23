namespace WeatherBlend.Train;

/// <summary>
/// Phase 3c training row: inherits the 27 lean occurrence-blender features (and Label)
/// from <see cref="PrecipTrainingRow"/> and adds 28 extra features:
///
///   • 18 per-model humidity — 6 models × {dew, rh, depression (T-Td)}
///   • 6 per-model surface pressure
///   • 4 EA-observation persistence — trailing-rainfall context known at run time
///     (prev 24h sum, prev 72h sum, wet-hours-last-24h, trailing-dry-hours)
///
/// 27 + 18 + 6 + 4 = 55 features. Ordering is load-bearing and must match
/// <see cref="PrecipRichFeatureBuilder.OccurrenceFeatureNames"/>.
///
/// Forecast-time trailing persistence (H-1, H-2, H-3 of the same run) was considered
/// and dropped: Phase 1 training parquet only persists leads {24, 48, 72} per
/// <c>RunTimeSource='offset_day'</c> run, so the intermediate-lead cells those
/// features need don't exist. Pressure tendency was dropped for the same reason.
///
/// Missing secondaries stay as <c>float.NaN</c>; LightGBM handles NaN natively.
/// Persistence features are NaN when trailing coverage is incomplete (i.e. we
/// wouldn't have had the reading at run time either).
/// </summary>
public sealed class RichPrecipTrainingRow : PrecipTrainingRow
{
    // Per-model humidity — dewpoint, relative humidity, and dew-depression (T-Td).
    // Dewpoint and RH are shipped independently because they carry different
    // information at low vs high temperatures; depression is the most precip-predictive
    // of the three so we ship it explicitly rather than forcing the tree to re-derive it.
    public float DewGfs { get; set; }
    public float DewEcmwf { get; set; }
    public float DewIcon { get; set; }
    public float DewMf { get; set; }
    public float DewUkmo { get; set; }
    public float DewGem { get; set; }

    public float RhGfs { get; set; }
    public float RhEcmwf { get; set; }
    public float RhIcon { get; set; }
    public float RhMf { get; set; }
    public float RhUkmo { get; set; }
    public float RhGem { get; set; }

    public float DewDepressionGfs { get; set; }
    public float DewDepressionEcmwf { get; set; }
    public float DewDepressionIcon { get; set; }
    public float DewDepressionMf { get; set; }
    public float DewDepressionUkmo { get; set; }
    public float DewDepressionGem { get; set; }

    // Per-model surface pressure at the valid hour.
    public float PressureGfs { get; set; }
    public float PressureEcmwf { get; set; }
    public float PressureIcon { get; set; }
    public float PressureMf { get; set; }
    public float PressureUkmo { get; set; }
    public float PressureGem { get; set; }

    // EA-observation persistence, anchored at run_time = valid_time - leadHours.
    // Rainfall is the EA hourly total for the primary truth station. NaN when
    // trailing coverage is incomplete (same rule the label column enforces).
    public float EaRainPrev24hMm { get; set; }
    public float EaRainPrev72hMm { get; set; }
    public float EaWetHoursLast24h { get; set; }
    public float EaDryHoursTrailing { get; set; }
}
