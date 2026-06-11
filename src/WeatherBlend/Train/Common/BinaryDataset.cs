using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Chronological train/val/test split for binary-classification blender rows.
/// Mirrors <see cref="WeatherBlend.Train.PrecipDataset"/> over the new generic
/// <see cref="BinaryTrainingRow"/> shape. Strictly no shuffling — random splits
/// would leak storms across splits. Slicing + invariants live in
/// <see cref="ChronologicalSplit{T}"/>; this record adds the typed shape and
/// the wet-label counters.
/// </summary>
public sealed record BinaryDataset(
    IReadOnlyList<BinaryTrainingRow> Train,
    IReadOnlyList<BinaryTrainingRow> Val,
    IReadOnlyList<BinaryTrainingRow> Test)
{
    public DateTime TrainStart => Train[0].ValidTimeUtc;
    public DateTime TrainEnd   => Train[^1].ValidTimeUtc;
    public DateTime ValStart   => Val[0].ValidTimeUtc;
    public DateTime ValEnd     => Val[^1].ValidTimeUtc;
    public DateTime TestStart  => Test[0].ValidTimeUtc;
    public DateTime TestEnd    => Test[^1].ValidTimeUtc;

    public int TrainWet => Train.Count(r => r.Label);
    public int ValWet   => Val.Count(r => r.Label);
    public int TestWet  => Test.Count(r => r.Label);

    public static BinaryDataset Split(
        IReadOnlyList<BinaryTrainingRow> rows,
        double trainFrac = 0.70,
        double valFrac = 0.15)
    {
        var s = ChronologicalSplit<BinaryTrainingRow>.Split(
            rows, r => r.ValidTimeUtc, trainFrac, valFrac);
        return new BinaryDataset(s.Train, s.Val, s.Test);
    }
}
