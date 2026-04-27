namespace WeatherBlend.Train.Common;

/// <summary>
/// Chronological train/val/test split for day-anchored dry-window blender rows.
/// Mirrors <see cref="WeatherBlend.Train.DryWindow.DryWindowDataset"/> over the
/// new generic <see cref="DryWindowTrainingRow"/> shape. Splits on
/// <see cref="DryWindowTrainingRow.TargetDateUtc"/> (day granularity) — strictly
/// no shuffling so that storms don't leak across splits.
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
        if (rows.Count < 10)
            throw new InvalidOperationException(
                $"Need at least 10 rows to split meaningfully; got {rows.Count}.");

        for (int i = 1; i < rows.Count; i++)
            if (rows[i].TargetDateUtc < rows[i - 1].TargetDateUtc)
                throw new InvalidOperationException(
                    $"Rows must be ascending by TargetDateUtc. " +
                    $"Row {i} = {rows[i].TargetDateUtc:o} < row {i - 1} = {rows[i - 1].TargetDateUtc:o}.");

        var n = rows.Count;
        var trainEnd = (int)Math.Floor(n * trainFrac);
        var valEnd = trainEnd + (int)Math.Floor(n * valFrac);

        var train = rows.Take(trainEnd).ToList();
        var val   = rows.Skip(trainEnd).Take(valEnd - trainEnd).ToList();
        var test  = rows.Skip(valEnd).ToList();

        if (train.Count == 0 || val.Count == 0 || test.Count == 0)
            throw new InvalidOperationException(
                $"Split produced an empty partition (train={train.Count}, val={val.Count}, test={test.Count}).");

        if (!(train[^1].TargetDateUtc < val[0].TargetDateUtc))
            throw new InvalidOperationException(
                $"Train/val boundary not strictly ordered: " +
                $"train_end={train[^1].TargetDateUtc:o} val_start={val[0].TargetDateUtc:o}");
        if (!(val[^1].TargetDateUtc < test[0].TargetDateUtc))
            throw new InvalidOperationException(
                $"Val/test boundary not strictly ordered: " +
                $"val_end={val[^1].TargetDateUtc:o} test_start={test[0].TargetDateUtc:o}");

        return new DryWindowDataset(train, val, test);
    }
}
