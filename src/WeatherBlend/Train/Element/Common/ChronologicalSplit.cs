namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Generic 70/15/15 chronological split over any row type with a time
/// accessor. The single implementation behind
/// <see cref="WeatherBlend.Train.Common.RegressionDataset.Split"/>,
/// <see cref="WeatherBlend.Train.Common.BinaryDataset.Split"/> and
/// <see cref="WeatherBlend.Train.Common.DryWindowDataset.Split"/> — those
/// records keep their typed shapes (and label-count accessors) but delegate
/// the slicing + invariants here so the rules can never drift apart.
///
/// Strict invariants (matched to the original temperature blender):
///   - Input must already be ascending by time (asserted, not silently sorted).
///   - All three partitions must be non-empty.
///   - Time boundaries between partitions must be strictly ordered.
///
/// <paramref name="timeFieldName"/> only feeds the exception text (the
/// dry-window split is day-anchored on TargetDateUtc, everything else on
/// ValidTimeUtc) so error messages keep naming the real field.
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
        double valFrac = 0.15,
        string timeFieldName = "ValidTimeUtc")
    {
        if (rows.Count < 10)
            throw new InvalidOperationException(
                $"Need at least 10 rows to split meaningfully; got {rows.Count}.");

        for (int i = 1; i < rows.Count; i++)
        {
            if (timeOf(rows[i]) < timeOf(rows[i - 1]))
                throw new InvalidOperationException(
                    $"Rows must be ascending by {timeFieldName}. " +
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
}
