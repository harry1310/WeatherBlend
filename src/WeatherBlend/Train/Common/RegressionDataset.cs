namespace WeatherBlend.Train.Common;

/// <summary>
/// Chronological train/val/test split for regression-blender rows. Mirrors
/// <see cref="WeatherBlend.Train.TrainingDataset"/> but over the new generic
/// <see cref="RegressionTrainingRow"/> shape. Strictly no shuffling — random
/// splits would leak future into past.
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
        if (rows.Count < 10)
            throw new InvalidOperationException(
                $"Need at least 10 rows to split meaningfully; got {rows.Count}.");

        for (int i = 1; i < rows.Count; i++)
            if (rows[i].ValidTimeUtc < rows[i - 1].ValidTimeUtc)
                throw new InvalidOperationException(
                    $"Rows must be ascending by ValidTimeUtc. " +
                    $"Row {i} = {rows[i].ValidTimeUtc:o} < row {i - 1} = {rows[i - 1].ValidTimeUtc:o}.");

        var n = rows.Count;
        var trainEnd = (int)Math.Floor(n * trainFrac);
        var valEnd = trainEnd + (int)Math.Floor(n * valFrac);

        var train = rows.Take(trainEnd).ToList();
        var val   = rows.Skip(trainEnd).Take(valEnd - trainEnd).ToList();
        var test  = rows.Skip(valEnd).ToList();

        if (train.Count == 0 || val.Count == 0 || test.Count == 0)
            throw new InvalidOperationException(
                $"Split produced an empty partition (train={train.Count}, val={val.Count}, test={test.Count}).");

        if (!(train[^1].ValidTimeUtc < val[0].ValidTimeUtc))
            throw new InvalidOperationException(
                $"Train/val boundary not strictly ordered: " +
                $"train_end={train[^1].ValidTimeUtc:o} val_start={val[0].ValidTimeUtc:o}");
        if (!(val[^1].ValidTimeUtc < test[0].ValidTimeUtc))
            throw new InvalidOperationException(
                $"Val/test boundary not strictly ordered: " +
                $"val_end={val[^1].ValidTimeUtc:o} test_start={test[0].ValidTimeUtc:o}");

        return new RegressionDataset(train, val, test);
    }
}
