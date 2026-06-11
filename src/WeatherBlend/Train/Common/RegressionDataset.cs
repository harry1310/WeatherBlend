using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Chronological train/val/test split for regression-blender rows. Mirrors
/// <see cref="WeatherBlend.Train.TrainingDataset"/> but over the new generic
/// <see cref="RegressionTrainingRow"/> shape. Strictly no shuffling — random
/// splits would leak future into past. Slicing + invariants live in
/// <see cref="ChronologicalSplit{T}"/> (shared with the binary and
/// dry-window datasets); this record only adds the typed shape.
/// </summary>
public sealed record RegressionDataset(
    IReadOnlyList<RegressionTrainingRow> Train,
    IReadOnlyList<RegressionTrainingRow> Val,
    IReadOnlyList<RegressionTrainingRow> Test)
{
    public DateTime TrainStart => Train[0].ValidTimeUtc;
    public DateTime TrainEnd   => Train[^1].ValidTimeUtc;
    public DateTime ValStart   => Val[0].ValidTimeUtc;
    public DateTime ValEnd     => Val[^1].ValidTimeUtc;
    public DateTime TestStart  => Test[0].ValidTimeUtc;
    public DateTime TestEnd    => Test[^1].ValidTimeUtc;

    public static RegressionDataset Split(
        IReadOnlyList<RegressionTrainingRow> rows,
        double trainFrac = 0.70,
        double valFrac = 0.15)
    {
        var s = ChronologicalSplit<RegressionTrainingRow>.Split(
            rows, r => r.ValidTimeUtc, trainFrac, valFrac);
        return new RegressionDataset(s.Train, s.Val, s.Test);
    }
}
