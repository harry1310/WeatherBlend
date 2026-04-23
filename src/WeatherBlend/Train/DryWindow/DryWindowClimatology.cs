using System.Text.Json;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Base-rate climatology for P(any length-N dry window exists on a UTC day),
/// keyed by month of year (1..12). A day-level target doesn't have an
/// hour-of-day axis, so the 3a <c>(month, hour)</c> keying collapses to
/// month only. One file per (station, window-length) version directory.
///
/// Unseen months fall back to <see cref="GlobalPositiveRate"/>. Persisted
/// as <c>dry_window_climatology.json</c> so predict/verify don't have to
/// re-derive it at inference time.
/// </summary>
public sealed class DryWindowClimatology
{
    /// <summary>P(dry-window) per 1..12 month. Absent months → caller falls back to <see cref="GlobalPositiveRate"/>.</summary>
    public Dictionary<string, double> PDryByMonth { get; set; } = new();

    public double GlobalPositiveRate { get; set; }
    public int SourceRowCount { get; set; }
    public int WindowHours { get; set; }

    public static DryWindowClimatology BuildFromTraining(IReadOnlyList<DryWindowTrainingRow> trainRows, int windowHours)
    {
        if (trainRows.Count == 0)
        {
            return new DryWindowClimatology { WindowHours = windowHours };
        }

        var sums = new Dictionary<int, (int Pos, int N)>();
        foreach (var r in trainRows)
        {
            var m = r.TargetDateUtc.Month;
            sums.TryGetValue(m, out var cur);
            sums[m] = (cur.Pos + (r.HasDryWindow ? 1 : 0), cur.N + 1);
        }

        var pDry = sums.ToDictionary(
            kv => kv.Key.ToString("D2"),
            kv => (double)kv.Value.Pos / kv.Value.N);

        return new DryWindowClimatology
        {
            PDryByMonth = pDry,
            GlobalPositiveRate = (double)trainRows.Count(r => r.HasDryWindow) / trainRows.Count,
            SourceRowCount = trainRows.Count,
            WindowHours = windowHours,
        };
    }

    public double Predict(DateTime targetDateUtc)
    {
        var key = targetDateUtc.Month.ToString("D2");
        return PDryByMonth.TryGetValue(key, out var p) ? p : GlobalPositiveRate;
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    public void SaveTo(string path)
        => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOpts));

    public static DryWindowClimatology LoadFrom(string path)
        => JsonSerializer.Deserialize<DryWindowClimatology>(File.ReadAllText(path))
           ?? throw new InvalidOperationException($"Could not parse climatology at {path}");
}
