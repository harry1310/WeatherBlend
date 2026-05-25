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

    // ---- 3g aleatoric uncertainty (nullable; populated only by Phase 3g) ----
    //
    // Summary of the per-MC-sample longest-dry-run distribution under
    // independence with 3a's per-hour q. ProbHasDryWindow above is the
    // headline P(longest >= windowHours); these four fields characterise
    // the spread of the underlying continuous quantity (longest dry run
    // in hours), which is what 3g actually computes during the same MC
    // pass. Useful as a confidence signal: narrow P10–P90 band → headline
    // robust; wide band → headline fragile. Null on 3b/3e/3c rows since
    // those phases don't run MC and the field has no analogue.

    public double? McMeanLongestDryRunHours { get; init; }
    public double? McP10LongestDryRunHours { get; init; }
    public double? McP50LongestDryRunHours { get; init; }
    public double? McP90LongestDryRunHours { get; init; }

    // ---- Phase 3a-uncertainty: epistemic envelope on ProbHasDryWindow
    // (nullable; populated only on 3g rows where a Bayesian CI parquet
    // for this (station, target_date, lead) cell was found at predict time).
    //
    // 3g's MC pass treated 3a's per-hour q as exact and reported the spread
    // of the *longest dry run* under independent Bernoullis (the McP10/50/90
    // fields above — aleatoric: "given my q is right, how does the day
    // play out?"). The fields below captured the orthogonal "what if my q
    // is off?" — epistemic perturbation by Bayesian CI80 width. The 3g
    // predictor that emitted them was retired 2026-05-25 in model-cleanup
    // Phase 1; columns stay so historic R2 parquets still deserialise.
    //
    // EpistemicProbDryWindowMean ≈ ProbHasDryWindow modulo MC noise — same
    // headline. EpistemicProbDryWindowQ10/Q90 give the 80% band; site can
    // render as e.g. "75% (band 60-85%)" so wide bands flag low-confidence
    // days. EpistemicSigmaUsed is for reproducibility and debugging.
    // Null on non-3g phases and on 3g rows where no Bayesian CI was joined.

    public double? EpistemicProbDryWindowMean { get; init; }
    public double? EpistemicProbDryWindowQ10  { get; init; }
    public double? EpistemicProbDryWindowQ90  { get; init; }
    public double? EpistemicSigmaUsed         { get; init; }

    // ---- Conformal-prediction set tag (nullable; populated when the
    // version dir has a fitted conformal calibrator) ----
    //
    // One of "Dry", "Wet", "Ambiguous" — the prediction set under split
    // conformal at the calibrator's α (typically 0.10 = 90% coverage).
    // "Ambiguous" flags rows where the model can't commit to a single
    // class with the requested coverage guarantee — the user-facing
    // confidence signal. Null when no conformal calibrator is present
    // (legacy versions or when --action skip was used).

    public string? ConformalSetTag { get; init; }
}
