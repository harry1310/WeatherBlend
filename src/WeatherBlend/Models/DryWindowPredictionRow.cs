using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One Phase 3b P(dry-window) prediction with full provenance. Stored under
/// <c>data/predictions/dry_window/{truth_station}/window_{N}h/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>
/// — one subtree per (truth-station, window-length) pair because each pair has
/// its own blender.
///
/// Per-model `has_dry_window` self-predictions + per-model day totals are carried
/// through so verification can compute mean-of-models / best-single baselines
/// without re-reading the forecast tree for that target date.
/// </summary>
public sealed class DryWindowPredictionRow
{
    /// <summary>
    /// Number of named per-model slots (HasDryWindowGfs..HasDryWindowJma).
    /// See <see cref="TempPredictionRow.PerModelFieldCount"/> for the rationale.
    /// </summary>
    public const int PerModelFieldCount = 8;

    [SetsRequiredMembers]
    public DryWindowPredictionRow()
    {
        LocationName = "";
        TruthStation = "";
        ModelVersion = "";
        FeatureVectorHash = "";
    }

    public required string LocationName { get; init; }
    public required string TruthStation { get; init; }
    public required int WindowHours { get; init; }
    public required string ModelVersion { get; init; }

    public required DateTime PredictionMadeAtUtc { get; init; }
    /// <summary>UTC midnight of the target day.</summary>
    public required DateTime TargetDateUtc { get; init; }
    /// <summary>Lead in hours from the anchor: 24, 48, or 72.</summary>
    public required int LeadHours { get; init; }

    /// <summary>Blender-calibrated P(∃ dry window ≥ WindowHours in target day).</summary>
    public required double ProbHasDryWindow { get; init; }

    /// <summary>Month-keyed climatology P(dry window). Skill-score reference.</summary>
    public required double ClimatologyProbHasDryWindow { get; init; }

    /// <summary>Fraction of models (with complete day coverage) whose forecast contains a length-N dry run.</summary>
    public double? AgreementHasDryWindow { get; init; }
    public double? PrecipSumMean { get; init; }
    public double? LongestDryRunMean { get; init; }
    public double? WetHourCountMean { get; init; }

    // Per-model self-predictions (nullable if model was missing any hour).
    public double? HasDryWindowGfs   { get; init; }
    public double? HasDryWindowEcmwf { get; init; }
    public double? HasDryWindowIcon  { get; init; }
    public double? HasDryWindowMf    { get; init; }
    public double? HasDryWindowUkmo  { get; init; }
    public double? HasDryWindowGem   { get; init; }
    public double? HasDryWindowAifs  { get; init; }
    public double? HasDryWindowJma   { get; init; }

    // Per-model day totals (nullable if model was missing any hour).
    public double? PrecipSumGfs   { get; init; }
    public double? PrecipSumEcmwf { get; init; }
    public double? PrecipSumIcon  { get; init; }
    public double? PrecipSumMf    { get; init; }
    public double? PrecipSumUkmo  { get; init; }
    public double? PrecipSumGem   { get; init; }
    public double? PrecipSumAifs  { get; init; }
    public double? PrecipSumJma   { get; init; }

    /// <summary>SHA-256 hex of the 53 feature floats in schema order.</summary>
    public required string FeatureVectorHash { get; init; }
}
