namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Generic 70/15/15 chronological split over any row type with a
/// <c>ValidTimeUtc</c> property accessor. Mirrors <see cref="TrainingDataset.Split"/>
/// in shape but unbound from the temperature row class.
///
/// Strict invariants (matched to the temperature blender):
///   - Input must already be ascending by valid time (asserted, not silently sorted).
///   - All three partitions must be non-empty.
///   - Time boundaries between partitions must be strictly ordered.
/// </summary>
public sealed record ChronologicalSplit<T>(
    IReadOnlyList<T> Train,
    IReadOnlyList<T> Val,
    IReadOnlyList<T> Test)
{
    public static ChronologicalSplit<T> Split(
        IReadOnlyList<T> rows,
        Func<T, DateTime> timeOf,
        double trainFrac = 0.70,
        double valFrac = 0.15)
    {
        if (rows.Count < 10)
            throw new InvalidOperationException(
                $"Need at least 10 rows to split meaningfully; got {rows.Count}.");

        for (int i = 1; i < rows.Count; i++)
        {
            if (timeOf(rows[i]) < timeOf(rows[i - 1]))
                throw new InvalidOperationException(
                    $"Rows must be ascending by ValidTimeUtc. " +
                    $"Row {i} = {timeOf(rows[i]):o} < row {i - 1} = {timeOf(rows[i - 1]):o}.");
        }

        var n = rows.Count;
        var trainEnd = (int)Math.Floor(n * trainFrac);
        var valEnd = trainEnd + (int)Math.Floor(n * valFrac);

        var train = rows.Take(trainEnd).ToList();
        var val   = rows.Skip(trainEnd).Take(valEnd - trainEnd).ToList();
        var test  = rows.Skip(valEnd).ToList();

        if (train.Count == 0 || val.Count == 0 || test.Count == 0)
            throw new InvalidOperationException(
                $"Split produced an empty partition (train={train.Count}, val={val.Count}, test={test.Count}).");

        if (!(timeOf(train[^1]) < timeOf(val[0])))
            throw new InvalidOperationException(
                $"Train/val boundary not strictly ordered: " +
                $"train_end={timeOf(train[^1]):o} val_start={timeOf(val[0]):o}");
        if (!(timeOf(val[^1]) < timeOf(test[0])))
            throw new InvalidOperationException(
                $"Val/test boundary not strictly ordered: " +
                $"val_end={timeOf(val[^1]):o} test_start={timeOf(test[0]):o}");

        return new ChronologicalSplit<T>(train, val, test);
    }

    public DateTime TrainStart(Func<T, DateTime> timeOf) => timeOf(Train[0]);
    public DateTime TrainEnd  (Func<T, DateTime> timeOf) => timeOf(Train[^1]);
    public DateTime ValStart  (Func<T, DateTime> timeOf) => timeOf(Val[0]);
    public DateTime ValEnd    (Func<T, DateTime> timeOf) => timeOf(Val[^1]);
    public DateTime TestStart (Func<T, DateTime> timeOf) => timeOf(Test[0]);
    public DateTime TestEnd   (Func<T, DateTime> timeOf) => timeOf(Test[^1]);
}
