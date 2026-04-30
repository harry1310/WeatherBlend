using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One row of the dry-window start-hour curve: for a given
/// <c>(TruthStation, WindowHours, LeadHours, TargetDateUtc)</c> we emit one
/// row per candidate start hour <c>StartHourUtc</c> in the daytime UTC range.
///
/// Pure derivation from
///   - hourly P(wet) at lead L from the Phase-3a precipitation blender
///     (<see cref="PrecipPredictionRow"/>), plus
///   - daily P(∃ N-hour dry block) at lead L from the Phase-3b/3d dry-window
///     blender (<see cref="DryWindowPredictionRow"/>).
///
/// The interesting derivative is <see cref="ConditionalProb"/> — the
/// distribution of <em>where in the day the dry block is, conditional on one
/// existing</em>. <see cref="CalibratedProb"/> multiplies that conditional by
/// the dry-window blender's marginal so the row reads as "P(dry block of N
/// hours starting at StartHourUtc, given today's forecast)".
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

    /// <summary>Raw hourly-independence product
    /// <c>p_s = ∏_{h=s..s+N-1} (1 − q_h)</c> before normalisation. Sums across
    /// starts can exceed 1 (it's an upper bound on P(any block)) so it's not
    /// directly interpretable as a probability — we keep it for diagnostics
    /// and as an audit trail of the derivation.</summary>
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

    /// <summary>Phase-3a champion version that supplied the hourly P(wet)
    /// inputs. Provenance for retrospective re-scoring.</summary>
    public required string PrecipVersion { get; init; }

    /// <summary>Phase-3b/3d champion version that supplied
    /// <see cref="DailyProbAnyBlock"/>.</summary>
    public required string DryWindowVersion { get; init; }
}
