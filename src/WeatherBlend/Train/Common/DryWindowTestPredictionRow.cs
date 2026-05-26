namespace WeatherBlend.Train.Common;

/// <summary>
/// Per-(station, window, lead, target_date) held-out test prediction
/// for the dry-window family, written to <c>test_predictions.parquet</c>
/// alongside the model bundle. Sibling of <see cref="TestPredictionRow"/>
/// which carries the hourly P(wet) shape — same purpose, different
/// granularity (day cell × window).
///
/// Consumed by bake-off scripts that need to compare 3b (direct LightGBM
/// dry-window) against MC-based estimators on the same (station, window,
/// lead, target_date) cells. The MC scripts read 3a's (or 3o's) hourly
/// TestPredictionRow, group by target_date, MC-sample, and inner-join here.
///
/// Saved at <c>data/models/dry_window/{station}/window_{N}h/{version}/test_predictions.parquet</c>.
/// </summary>
public sealed class DryWindowTestPredictionRow
{
    /// <summary>Target UTC date (00:00Z of the day the dry-window question applies to).</summary>
    public DateTime target_date { get; set; }

    /// <summary>Station slug, e.g. <c>ea_bellever_dartmoor</c>.</summary>
    public string station { get; set; } = "";

    /// <summary>Dry-window length in hours — 3, 4, or 6.</summary>
    public int window { get; set; }

    /// <summary>Forecast lead in hours — 24, 48, or 72.</summary>
    public int lead { get; set; }

    /// <summary>Predicted P(∃ contiguous window-hour dry block in the daytime window
    /// of <c>target_date</c>) — the SHIPPING probability (= calibrated when this station
    /// opted in via <c>DryWindow.CalibrationEnabled</c>, raw otherwise).</summary>
    public double p_dry_window { get; set; }

    /// <summary>Observed truth label (0/1) — does the actual hourly rainfall
    /// trace for <c>target_date</c> contain a contiguous run of >= <c>window</c>
    /// dry hours within the daytime window?</summary>
    public byte observed_dry_window { get; set; }
}
