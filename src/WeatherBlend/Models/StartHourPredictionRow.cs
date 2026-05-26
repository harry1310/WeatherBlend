using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One row of the dry-window start-hour curve: for a given
/// <c>(TruthStation, WindowHours, LeadHours, TargetDateUtc)</c> we emit one
/// row per candidate start hour <c>StartHourUtc</c> in the daytime UTC range.
///
/// Produced today by Phase 3p as a byproduct of its copula-MC pass — the
/// same draws that compute <c>ProbHasDryWindow</c> also track whether each
/// candidate window <c>[s, s+N)</c> was entirely dry. The start-hour curve
/// and the daily probability therefore stay numerically consistent (no
/// drift between "P(any block today) = X" and "Σ start-hour probs = Y" with
/// Y ≠ X under independence; under the copula the relationship is the
/// dependency-aware analogue).
///
/// <see cref="RawProduct"/> is the per-start-hour marginal — P(an N-hour dry
/// block starting at this hour), the "chance of a dry walk if you set off
/// then" number the site plots. <see cref="ConditionalProb"/> normalises that
/// to a sum-to-1 shape (where in the day, given a block exists); it is NOT a
/// probability of dryness. <see cref="CalibratedProb"/> rescales the shape to
/// sum to the daily P(any block).
/// </summary>
public sealed class StartHourPredictionRow
{
    [SetsRequiredMembers]
    public StartHourPredictionRow()
    {
        LocationName = "";
        TruthStation = "";
        ModelVersion = "";
        PrecipVersion = "";
        DryWindowVersion = "";
    }

    public required string LocationName { get; init; }
    public required string TruthStation { get; init; }
    public required int WindowHours { get; init; }

    /// <summary>Curve derivation version. Bumped if the math changes; the
    /// inputs (precip / dry-window champions) are tracked separately.</summary>
    public required string ModelVersion { get; init; }

    public required DateTime PredictionMadeAtUtc { get; init; }

    /// <summary>UTC midnight of the day the curve covers — same convention as
    /// <see cref="DryWindowPredictionRow.TargetDateUtc"/>.</summary>
    public required DateTime TargetDateUtc { get; init; }

    public required int LeadHours { get; init; }

    /// <summary>Candidate start hour in UTC (0–23). Block runs
    /// <c>[StartHourUtc, StartHourUtc + WindowHours)</c>.</summary>
    public required int StartHourUtc { get; init; }

    /// <summary>Per-start-hour marginal: P(the N-hour block starting at this
    /// hour is entirely dry) — the MC fraction of samples in which hours
    /// <c>[s, s+N)</c> were all dry. This IS a probability (each value in
    /// [0,1]); only the SUM across overlapping starts can exceed 1. This is
    /// the "chance of a dry block if you start at this hour" the site's
    /// start-hour chart plots.</summary>
    public required double RawProduct { get; init; }

    /// <summary>Conditional probability the block starts at this hour, given
    /// a block exists somewhere today: <c>π_s = p_s / Σ p_s</c>. Falls back to
    /// uniform when Σ = 0 (every candidate window has at least one
    /// guaranteed-wet hour). Sums to 1 across <see cref="StartHourUtc"/> for
    /// fixed (Station, Window, Lead, TargetDate).</summary>
    public required double ConditionalProb { get; init; }

    /// <summary>Calibrated marginal <c>π_s × DailyProbAnyBlock</c>. Reads as
    /// "P(N-hour dry block starting at StartHourUtc)". Sum across starts ≈
    /// <see cref="DailyProbAnyBlock"/>.</summary>
    public required double CalibratedProb { get; init; }

    /// <summary>The dry-window blender's daily P(∃ N-hour dry block) used as
    /// the calibration anchor for this row's <see cref="CalibratedProb"/>.</summary>
    public required double DailyProbAnyBlock { get; init; }

    /// <summary>Source precip-phase champion version that supplied the
    /// hourly P(wet) inputs (3o for the 3p start-hour curve). Provenance
    /// for retrospective re-scoring.</summary>
    public required string PrecipVersion { get; init; }

    /// <summary>Dry-window blender version that supplied
    /// <see cref="DailyProbAnyBlock"/>.</summary>
    public required string DryWindowVersion { get; init; }
}
