namespace WeatherBlend.Train.Common;

/// <summary>
/// Per-row held-out test prediction, written to <c>test_predictions.parquet</c>
/// alongside the model bundle. Consumed by downstream bake-off scripts
/// (e.g. linear-pool 3a + 4a) that need raw probabilities aligned per
/// (valid_time, station, lead) to fit pooling weights.
///
/// Schema deliberately matches WeatherProbabilistic's
/// <c>scripts/run_phase5_bayesian.py</c> + <c>train_4a.py</c> output:
/// <c>{valid_time, station, lead, p_wet, observed_wet}</c>. A single
/// bake-off script can inner-join across phases without per-phase
/// schema branches.
///
/// Saved at <c>data/models/precipitation/{station}/{version}/test_predictions.parquet</c>
/// for binary phases (3a, 3c, 3d, 3b). Regression phases (temperature,
/// element) don't write this today — add a sibling row type if needed.
/// </summary>
public sealed class TestPredictionRow
{
    /// <summary>Valid time of the prediction (UTC).</summary>
    public DateTime valid_time { get; set; }

    /// <summary>Station slug, e.g. <c>ea_bellever_dartmoor</c>.</summary>
    public string station { get; set; } = "";

    /// <summary>Forecast lead in hours.</summary>
    public int lead { get; set; }

    /// <summary>Predicted P(wet ≥ 0.1 mm/h) on the held-out test row.</summary>
    public double p_wet { get; set; }

    /// <summary>Observed truth label (0/1) for the same (station, valid_time, lead).</summary>
    public byte observed_wet { get; set; }
}
