using WeatherBlend.Train.Common;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Multi-label dataset builder for Phase 3e (B2 conditional decomposition,
/// 3h + 4h windows). Wraps <see cref="DryWindowFeatureBuilder.BuildForLead"/>
/// to produce paired (3h-labeled, 4h-labeled) rows joined by
/// <see cref="DryWindow3eMultiLabelRow.TargetDateUtc"/>.
///
/// Cost: two feature-builder passes per lead (one per output window). Each
/// pass re-loads rainfall truth and re-runs the forecast SQL — wasted work
/// at the kilo-row scale. Acceptable at PoC because both passes share a
/// process and DuckDB's parquet reader is fast; revisit if training time
/// becomes a bottleneck. Sidestepping the duplicate I/O by exposing
/// <see cref="DryWindowFeatureBuilder"/>'s internals would tightly couple
/// 3e to 3b's private surface.
///
/// Why this shape: the trainer needs Label3h on every row + Label4h on the
/// subset where Label3h=true (per the conditional factorisation). One
/// row-per-day with both labels lets the trainer subset cleanly with
/// <c>.Where(r =&gt; r.Label3h).Select(r =&gt; ToCommonRow(r, r.Label4h))</c>.
/// </summary>
public static class DryWindow3eFeatureBuilder
{
    /// <summary>Phase identifier persisted in <c>training_metadata.Phase</c>.</summary>
    public const string Phase3e = "3e";

    /// <summary>The two output windows 3e ships at — 6h is intentionally
    /// excluded (sparser positives, smaller conditional subset, higher
    /// variance). Add 6h here and a third stage in the trainer if it ever
    /// gets promoted.</summary>
    public static readonly IReadOnlyList<int> OutputWindows = new[] { 3, 4 };

    /// <summary>
    /// Build day-anchored rows containing both the 3h and 4h labels for one
    /// (station, lead). Returns rows ordered by <c>TargetDateUtc</c>, dropping
    /// dates that aren't usable for BOTH window labels (i.e. dates where
    /// either pass dropped the day for missing truth or missing forecasts).
    /// </summary>
    public static List<DryWindow3eMultiLabelRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        BlenderSpec spec,
        DaytimeWindow daytime,
        CancellationToken ct = default)
    {
        // Two passes — one per output window — over the same underlying truth
        // and forecast trees. Inner-join by TargetDateUtc so every returned
        // row carries both labels.
        var rows3h = DryWindowFeatureBuilder.BuildForLead(
            forecastsPath, rainfallPath, locationName, stationName,
            spec, windowHours: 3, daytime, ct);
        var rows4h = DryWindowFeatureBuilder.BuildForLead(
            forecastsPath, rainfallPath, locationName, stationName,
            spec, windowHours: 4, daytime, ct);

        var label4ByDate = rows4h.ToDictionary(r => r.TargetDateUtc, r => r.Label);

        var joined = new List<DryWindow3eMultiLabelRow>(rows3h.Count);
        foreach (var r in rows3h)
        {
            if (!label4ByDate.TryGetValue(r.TargetDateUtc, out var label4)) continue;
            joined.Add(new DryWindow3eMultiLabelRow
            {
                TargetDateUtc = r.TargetDateUtc,
                Features      = r.Features,        // same vector — the 3h pass is canonical
                Label3h       = r.Label,
                Label4h       = label4,
                PrecipMmDay   = r.PrecipMmDay,
            });
        }
        return joined;
    }

    /// <summary>
    /// Project a multi-label row into a vanilla <see cref="CommonRow"/> with
    /// the chosen single label. Used by the trainer to feed the 3h-or-4h
    /// view of each row to <see cref="DryWindowTrainer.TrainVector"/>
    /// without the trainer having to know about 3e's two-label shape.
    /// </summary>
    public static CommonRow ToCommonRow(DryWindow3eMultiLabelRow row, bool label, int outputWindowHours)
        => new()
        {
            TargetDateUtc = row.TargetDateUtc,
            WindowHours   = outputWindowHours,
            Features      = row.Features,
            Label         = label,
            PrecipMmDay   = row.PrecipMmDay,
        };
}
