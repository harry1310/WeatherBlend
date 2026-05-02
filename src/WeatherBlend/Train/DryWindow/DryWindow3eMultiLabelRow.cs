namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// One day-anchored training row carrying BOTH the 3h and 4h dry-window
/// labels for the same (station, day). Powers the Phase 3e conditional
/// decomposition trainer — M_base trains against <see cref="Label3h"/> on
/// the full set, M_extend4 trains against <see cref="Label4h"/> on the
/// subset where Label3h is true. Features are the same 53-column 3b vector
/// (or 60-column 3d-shape vector if a future variant calls for it); the
/// only multi-label-specific addition is the second label column.
/// </summary>
public sealed class DryWindow3eMultiLabelRow
{
    public DateTime TargetDateUtc { get; init; }

    /// <summary>Same vector as 3b's <c>DryWindowTrainingRow.Features</c>.</summary>
    public float[] Features { get; init; } = Array.Empty<float>();

    public bool Label3h { get; init; }
    public bool Label4h { get; init; }

    /// <summary>Daily total mm — diagnostic, not a feature.</summary>
    public float PrecipMmDay { get; init; }
}
