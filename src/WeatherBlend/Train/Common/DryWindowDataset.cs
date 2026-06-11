using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Chronological train/val/test split for day-anchored dry-window blender rows.
/// Mirrors <see cref="WeatherBlend.Train.DryWindow.DryWindowDataset"/> over the
/// new generic <see cref="DryWindowTrainingRow"/> shape. Splits on
/// <see cref="DryWindowTrainingRow.TargetDateUtc"/> (day granularity) — strictly
/// no shuffling so that storms don't leak across splits. Slicing + invariants
/// live in <see cref="ChronologicalSplit{T}"/>; this record adds the typed
/// shape and the positive-label counters.
/// </summary>
public sealed record DryWindowDataset(
    IReadOnlyList<DryWindowTrainingRow> Train,
    IReadOnlyList<DryWindowTrainingRow> Val,
    IReadOnlyList<DryWindowTrainingRow> Test)
{
    public DateTime TrainStart => Train[0].TargetDateUtc;
    public DateTime TrainEnd   => Train[^1].TargetDateUtc;
    public DateTime ValStart   => Val[0].TargetDateUtc;
    public DateTime ValEnd     => Val[^1].TargetDateUtc;
    public DateTime TestStart  => Test[0].TargetDateUtc;
    public DateTime TestEnd    => Test[^1].TargetDateUtc;

    public int TrainPositives => Train.Count(r => r.Label);
    public int ValPositives   => Val.Count(r => r.Label);
    public int TestPositives  => Test.Count(r => r.Label);

    public static DryWindowDataset Split(
        IReadOnlyList<DryWindowTrainingRow> rows,
        double trainFrac = 0.70,
        double valFrac = 0.15)
    {
        var s = ChronologicalSplit<DryWindowTrainingRow>.Split(
            rows, r => r.TargetDateUtc, trainFrac, valFrac,
            timeFieldName: "TargetDateUtc");
        return new DryWindowDataset(s.Train, s.Val, s.Test);
    }
}
